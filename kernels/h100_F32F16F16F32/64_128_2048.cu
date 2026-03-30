#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda;

#define MM 64
#define NN 128
#define KK 2048

#define NUM_SPLITS_256 256
#define K_CHUNK_8      8

#define NUM_SPLITS_128 128
#define K_CHUNK_16     16

static float* g_ws        = nullptr;
static float* g_atomic_ws = nullptr;
static bool   g_init      = false;
static bool   g_coop_ok   = false;
static int    g_numSMs    = 0;
static int    g_maxBPS_128 = 0;
static int    g_maxBPS_256 = 0;

__device__ __forceinline__ uint32_t cvt_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ __launch_bounds__(128, 4)
void hgemm_fused_256(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ WS,
    half*       __restrict__ C)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_CHUNK_8;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int warp_m   = warp_id * 16;

    __shared__ __align__(128) half  smA[MM][16];
    __shared__ __align__(128) half  smB[16][NN];
    __shared__ __align__(128) float smem_out[MM][NN];

    {
        int idx = tid;
        #pragma unroll
        for (int i = idx; i < MM * 8; i += 128) {
            int r = i >> 3;
            int c = (i & 7) + 8;
            smA[r][c] = __float2half(0.f);
        }
        #pragma unroll
        for (int i = idx; i < 8 * NN; i += 128) {
            int r = (i >> 7) + 8;
            int c = i & 127;
            smB[r][c] = __float2half(0.f);
        }
    }

    {
        int r = tid >> 1;
        int c = (tid & 1) << 2;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(cvt_smem(&smA[r][c])),
               "l"((const void*)(A + (size_t)r * KK + k_start + c))
        );
    }

    {
        int r = tid >> 4;
        int c = (tid & 15) << 3;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(cvt_smem(&smB[r][c])),
               "l"((const void*)(B + (size_t)(k_start + r) * NN + c))
        );
    }

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::fill_fragment(acc[n], 0.f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::load_matrix_sync(fa, &smA[warp_m][0], 16);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[8];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::load_matrix_sync(fb[n], &smB[0][n * 16], NN);

    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::mma_sync(acc[n], fa, fb[n], acc[n]);

    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::store_matrix_sync(&smem_out[warp_m][n * 16], acc[n], NN, wmma::mem_row_major);
    __syncthreads();

    float* ws_base = WS + (size_t)split_id * MM * NN;
    #pragma unroll 4
    for (int i = tid; i < MM * NN / 4; i += 128)
        reinterpret_cast<float4*>(ws_base)[i] =
            reinterpret_cast<const float4*>(smem_out)[i];

    grid.sync();

    if (split_id < MM) {
        const int row   = split_id;
        const int col   = tid;
        const int idx   = row * NN + col;
        const int slice = MM * NN;
        const float* base = WS + idx;

        float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        float s8=0,s9=0,sa=0,sb=0,sc=0,sd=0,se=0,sf=0;

        #pragma unroll
        for (int s = 0; s < NUM_SPLITS_256; s += 16) {
            s0 += __ldg(base + (s+ 0)*slice);
            s1 += __ldg(base + (s+ 1)*slice);
            s2 += __ldg(base + (s+ 2)*slice);
            s3 += __ldg(base + (s+ 3)*slice);
            s4 += __ldg(base + (s+ 4)*slice);
            s5 += __ldg(base + (s+ 5)*slice);
            s6 += __ldg(base + (s+ 6)*slice);
            s7 += __ldg(base + (s+ 7)*slice);
            s8 += __ldg(base + (s+ 8)*slice);
            s9 += __ldg(base + (s+ 9)*slice);
            sa += __ldg(base + (s+10)*slice);
            sb += __ldg(base + (s+11)*slice);
            sc += __ldg(base + (s+12)*slice);
            sd += __ldg(base + (s+13)*slice);
            se += __ldg(base + (s+14)*slice);
            sf += __ldg(base + (s+15)*slice);
        }

        float sum = ((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)) +
                    ((s8+s9)+(sa+sb)) + ((sc+sd)+(se+sf));
        C[idx] = __float2half_rn(sum);
    }
}

__global__ __launch_bounds__(128, 4)
void hgemm_fused_128(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ WS,
    half*       __restrict__ C)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_CHUNK_16;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int warp_m   = warp_id * 16;

    __shared__ __align__(128) half  smA[MM][K_CHUNK_16];
    __shared__ __align__(128) half  smB[K_CHUNK_16][NN];
    __shared__ __align__(128) float smem_out[MM][NN];

    {
        int r = tid >> 1;
        int c = (tid & 1) << 3;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(cvt_smem(&smA[r][c])),
               "l"((const void*)(A + (size_t)r * KK + k_start + c))
        );
    }
    {
        int r0 = tid >> 3, c0 = (tid & 7) << 3;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(cvt_smem(&smB[r0][c0])),
               "l"((const void*)(B + (size_t)(k_start + r0) * NN + c0))
        );
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(cvt_smem(&smB[r0][c0 + 64])),
               "l"((const void*)(B + (size_t)(k_start + r0) * NN + c0 + 64))
        );
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) wmma::fill_fragment(acc[n], 0.f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::load_matrix_sync(fa, &smA[warp_m][0], K_CHUNK_16);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[8];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::load_matrix_sync(fb[n], &smB[0][n * 16], NN);

    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::mma_sync(acc[n], fa, fb[n], acc[n]);

    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::store_matrix_sync(&smem_out[warp_m][n * 16], acc[n], NN, wmma::mem_row_major);
    __syncthreads();

    float* ws_base = WS + (size_t)split_id * MM * NN;
    #pragma unroll 4
    for (int i = tid; i < MM * NN / 4; i += 128)
        reinterpret_cast<float4*>(ws_base)[i] =
            reinterpret_cast<const float4*>(smem_out)[i];

    grid.sync();

    if (split_id < MM) {
        const int row   = split_id;
        const int col   = tid;
        const int idx   = row * NN + col;
        const int slice = MM * NN;
        const float* base = WS + idx;

        float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        float s8=0,s9=0,sa=0,sb=0,sc=0,sd=0,se=0,sf=0;

        #pragma unroll
        for (int s = 0; s < NUM_SPLITS_128; s += 16) {
            s0 += __ldg(base + (s+ 0)*slice);
            s1 += __ldg(base + (s+ 1)*slice);
            s2 += __ldg(base + (s+ 2)*slice);
            s3 += __ldg(base + (s+ 3)*slice);
            s4 += __ldg(base + (s+ 4)*slice);
            s5 += __ldg(base + (s+ 5)*slice);
            s6 += __ldg(base + (s+ 6)*slice);
            s7 += __ldg(base + (s+ 7)*slice);
            s8 += __ldg(base + (s+ 8)*slice);
            s9 += __ldg(base + (s+ 9)*slice);
            sa += __ldg(base + (s+10)*slice);
            sb += __ldg(base + (s+11)*slice);
            sc += __ldg(base + (s+12)*slice);
            sd += __ldg(base + (s+13)*slice);
            se += __ldg(base + (s+14)*slice);
            sf += __ldg(base + (s+15)*slice);
        }

        float sum = ((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)) +
                    ((s8+s9)+(sa+sb)) + ((sc+sd)+(se+sf));
        C[idx] = __float2half_rn(sum);
    }
}

__global__ __launch_bounds__(128, 4)
void hgemm_splitk_noncoop(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ WS)
{
    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_CHUNK_16;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int warp_m   = warp_id * 16;

    __shared__ __align__(128) half  smA[MM][K_CHUNK_16];
    __shared__ __align__(128) half  smB[K_CHUNK_16][NN];

    {
        int r = tid >> 1, c = (tid & 1) << 3;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(cvt_smem(&smA[r][c])),
               "l"((const void*)(A + (size_t)r * KK + k_start + c))
        );
        int r0 = tid >> 3, c0 = (tid & 7) << 3;
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(cvt_smem(&smB[r0][c0])),
               "l"((const void*)(B + (size_t)(k_start + r0) * NN + c0))
        );
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(cvt_smem(&smB[r0][c0 + 64])),
               "l"((const void*)(B + (size_t)(k_start + r0) * NN + c0 + 64))
        );
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) wmma::fill_fragment(acc[n], 0.f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::load_matrix_sync(fa, &smA[warp_m][0], K_CHUNK_16);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[8];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::load_matrix_sync(fb[n], &smB[0][n * 16], NN);

    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::mma_sync(acc[n], fa, fb[n], acc[n]);

    __shared__ __align__(128) float smem_out[MM][NN];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::store_matrix_sync(&smem_out[warp_m][n * 16], acc[n], NN, wmma::mem_row_major);
    __syncthreads();

    float* ws_base = WS + (size_t)split_id * MM * NN;
    #pragma unroll 4
    for (int i = tid; i < MM * NN / 4; i += 128)
        reinterpret_cast<float4*>(ws_base)[i] =
            reinterpret_cast<const float4*>(smem_out)[i];
}

__global__ __launch_bounds__(128)
void reduce_noncoop(
    const float* __restrict__ WS,
    half*        __restrict__ C,
    int num_splits)
{
    const int row   = blockIdx.x;
    const int col   = threadIdx.x;
    const int idx   = row * NN + col;
    const int slice = MM * NN;
    const float* base = WS + idx;

    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    float s8=0,s9=0,sa=0,sb=0,sc=0,sd=0,se=0,sf=0;

    for (int s = 0; s < num_splits; s += 16) {
        s0 += __ldg(base + (s+ 0)*slice);
        s1 += __ldg(base + (s+ 1)*slice);
        s2 += __ldg(base + (s+ 2)*slice);
        s3 += __ldg(base + (s+ 3)*slice);
        s4 += __ldg(base + (s+ 4)*slice);
        s5 += __ldg(base + (s+ 5)*slice);
        s6 += __ldg(base + (s+ 6)*slice);
        s7 += __ldg(base + (s+ 7)*slice);
        s8 += __ldg(base + (s+ 8)*slice);
        s9 += __ldg(base + (s+ 9)*slice);
        sa += __ldg(base + (s+10)*slice);
        sb += __ldg(base + (s+11)*slice);
        sc += __ldg(base + (s+12)*slice);
        sd += __ldg(base + (s+13)*slice);
        se += __ldg(base + (s+14)*slice);
        sf += __ldg(base + (s+15)*slice);
    }

    float sum = ((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)) +
                ((s8+s9)+(sa+sb)) + ((sc+sd)+(se+sf));
    C[idx] = __float2half_rn(sum);
}

static bool   g_state_init   = false;
static bool   g_coop_256_ok  = false;
static bool   g_coop_128_ok  = false;
static int    g_maxBPS_fused_256 = 0;
static int    g_maxBPS_fused_128 = 0;
static int    g_nSMs         = 0;

static void init_all() {
    if (g_state_init) return;
    g_state_init = true;

    size_t ws_size = (size_t)NUM_SPLITS_256 * MM * NN * sizeof(float);
    cudaMalloc(&g_ws, ws_size);

    int dev = 0;
    cudaGetDevice(&dev);

    int sc = 0;
    cudaDeviceGetAttribute(&sc, cudaDevAttrCooperativeLaunch, dev);
    cudaDeviceGetAttribute(&g_nSMs, cudaDevAttrMultiProcessorCount, dev);

    if (sc) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &g_maxBPS_fused_256, hgemm_fused_256, 128, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &g_maxBPS_fused_128, hgemm_fused_128, 128, 0);

        g_coop_256_ok = (g_nSMs * g_maxBPS_fused_256 >= NUM_SPLITS_256);
        g_coop_128_ok = (g_nSMs * g_maxBPS_fused_128 >= NUM_SPLITS_128);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    init_all();

    if (g_coop_128_ok) {
        void* args[] = {
            (void*)&ptr_A,
            (void*)&ptr_B,
            (void*)&g_ws,
            (void*)&ptr_C
        };
        cudaLaunchCooperativeKernel(
            (void*)hgemm_fused_128,
            dim3(NUM_SPLITS_128), dim3(128),
            args, 0, nullptr);
        return;
    }

    if (g_coop_256_ok) {
        void* args[] = {
            (void*)&ptr_A,
            (void*)&ptr_B,
            (void*)&g_ws,
            (void*)&ptr_C
        };
        cudaLaunchCooperativeKernel(
            (void*)hgemm_fused_256,
            dim3(NUM_SPLITS_256), dim3(128),
            args, 0, nullptr);
        return;
    }

    hgemm_splitk_noncoop<<<NUM_SPLITS_128, 128>>>(ptr_A, ptr_B, g_ws);
    reduce_noncoop<<<MM, NN>>>(g_ws, ptr_C, NUM_SPLITS_128);
}