#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

namespace cg = cooperative_groups;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

static constexpr int GEMM_M = 64;
static constexpr int GEMM_N = 128;
static constexpr int GEMM_K = 8192;

static constexpr int NUM_SPLITS    = 128;
static constexpr int K_TILE        = 64;
static constexpr int WARPS_M       = 4;
static constexpr int WARPS_N       = 2;
static constexpr int NUM_WARPS     = WARPS_M * WARPS_N;
static constexpr int BLOCK_THREADS = NUM_WARPS * 32;

static __device__ __forceinline__
void cp_async16(void* dst, const void* src) {
    uint32_t addr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(addr), "l"(src) : "memory");
}

static __device__ __forceinline__
void mma_m16n8k16(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

static __device__ __forceinline__
void ldmatrix_x4(uint32_t* r, const void* ptr) {
    uint32_t addr = __cvta_generic_to_shared(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr)
    );
}

static __device__ __forceinline__
void ldmatrix_x2_trans(uint32_t* r, const void* ptr) {
    uint32_t addr = __cvta_generic_to_shared(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr)
    );
}

__global__ void hgemm_fused_cooperative(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ partials,
    half* __restrict__ C_out
) {
    cg::grid_group grid = cg::this_grid();

    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_TILE;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;

    uint32_t acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0u;

    __shared__ __align__(128) half sA[GEMM_M][K_TILE + 8];
    __shared__ __align__(128) half sB[K_TILE][GEMM_N + 8];

    {
        const int a_r0 = tid >> 3;
        const int a_c0 = (tid & 7) << 3;
        cp_async16(&sA[a_r0][a_c0], &A[a_r0 * GEMM_K + k_start + a_c0]);

        const int ch1  = tid + BLOCK_THREADS;
        const int a_r1 = ch1 >> 3;
        const int a_c1 = (ch1 & 7) << 3;
        cp_async16(&sA[a_r1][a_c1], &A[a_r1 * GEMM_K + k_start + a_c1]);
    }

    {
        const int bn = GEMM_N >> 3;
        const int ch0 = tid;
        const int ch1 = tid + 256;
        const int ch2 = tid + 512;
        const int ch3 = tid + 768;
        cp_async16(&sB[ch0/bn][(ch0%bn)<<3], &B[(k_start+ch0/bn)*GEMM_N+((ch0%bn)<<3)]);
        cp_async16(&sB[ch1/bn][(ch1%bn)<<3], &B[(k_start+ch1/bn)*GEMM_N+((ch1%bn)<<3)]);
        cp_async16(&sB[ch2/bn][(ch2%bn)<<3], &B[(k_start+ch2/bn)*GEMM_N+((ch2%bn)<<3)]);
        cp_async16(&sB[ch3/bn][(ch3%bn)<<3], &B[(k_start+ch3/bn)*GEMM_N+((ch3%bn)<<3)]);
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int a_smem_row = wm * 16;
    const int c_col_base = wn * 64;
    const int lane_mod16 = lane & 15;
    const int lane_div16 = lane >> 4;

    uint32_t fA_cur[4], fA_next[4];
    uint32_t fB_cur[8][2], fB_next[8][2];

    ldmatrix_x4(fA_cur, &sA[a_smem_row + lane_mod16][lane_div16 * 8]);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        ldmatrix_x2_trans(fB_cur[ni], &sB[lane_mod16][c_col_base + ni * 8]);
    }

    #pragma unroll
    for (int ki = 0; ki < K_TILE; ki += 16) {
        const int ki_next = ki + 16;
        if (ki_next < K_TILE) {
            ldmatrix_x4(fA_next, &sA[a_smem_row + lane_mod16][ki_next + lane_div16 * 8]);
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                ldmatrix_x2_trans(fB_next[ni], &sB[ki_next + lane_mod16][c_col_base + ni * 8]);
            }
        }
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            mma_m16n8k16(acc[ni], fA_cur, fB_cur[ni], acc[ni]);
        }
        if (ki_next < K_TILE) {
            #pragma unroll
            for (int r = 0; r < 4; r++) fA_cur[r] = fA_next[r];
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fB_cur[ni][0] = fB_next[ni][0];
                fB_cur[ni][1] = fB_next[ni][1];
            }
        }
    }

    half* my_partial = partials + (size_t)split_id * GEMM_M * GEMM_N;
    const int c_row0    = a_smem_row + (lane >> 2);
    const int c_row1    = c_row0 + 8;
    const int lane_mod4 = lane & 3;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c_col = c_col_base + ni * 8 + lane_mod4 * 2;
        half2 h0 = __float22half2_rn(make_float2(
            __uint_as_float(acc[ni][0]), __uint_as_float(acc[ni][1])));
        half2 h1 = __float22half2_rn(make_float2(
            __uint_as_float(acc[ni][2]), __uint_as_float(acc[ni][3])));
        *reinterpret_cast<half2*>(&my_partial[c_row0 * GEMM_N + c_col]) = h0;
        *reinterpret_cast<half2*>(&my_partial[c_row1 * GEMM_N + c_col]) = h1;
    }

    grid.sync();

    const int total_out = GEMM_M * GEMM_N;

    if (split_id < 32) {
        const int out_base = split_id * 256;
        const int out_idx = out_base + tid;
        if (out_idx < total_out) {
            float sum = 0.f;
            const int stride = total_out;
            const half* ptr = partials + out_idx;

            #pragma unroll 32
            for (int s = 0; s < NUM_SPLITS; s++) {
                sum += __half2float(ptr[s * stride]);
            }

            C_out[out_idx] = __float2half(sum);
        }
    }
}

__global__ __launch_bounds__(256, 5)
void hgemm_splitk128_fp16partial(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ partials
) {
    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_TILE;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;

    uint32_t acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0u;

    __shared__ __align__(128) half sA[GEMM_M][K_TILE + 8];
    __shared__ __align__(128) half sB[K_TILE][GEMM_N + 8];

    {
        const int a_r0 = tid >> 3;
        const int a_c0 = (tid & 7) << 3;
        cp_async16(&sA[a_r0][a_c0], &A[a_r0 * GEMM_K + k_start + a_c0]);
        const int ch1  = tid + BLOCK_THREADS;
        const int a_r1 = ch1 >> 3;
        const int a_c1 = (ch1 & 7) << 3;
        cp_async16(&sA[a_r1][a_c1], &A[a_r1 * GEMM_K + k_start + a_c1]);
    }

    {
        const int bn = GEMM_N >> 3;
        const int ch0 = tid, ch1 = tid+256, ch2 = tid+512, ch3 = tid+768;
        cp_async16(&sB[ch0/bn][(ch0%bn)<<3], &B[(k_start+ch0/bn)*GEMM_N+((ch0%bn)<<3)]);
        cp_async16(&sB[ch1/bn][(ch1%bn)<<3], &B[(k_start+ch1/bn)*GEMM_N+((ch1%bn)<<3)]);
        cp_async16(&sB[ch2/bn][(ch2%bn)<<3], &B[(k_start+ch2/bn)*GEMM_N+((ch2%bn)<<3)]);
        cp_async16(&sB[ch3/bn][(ch3%bn)<<3], &B[(k_start+ch3/bn)*GEMM_N+((ch3%bn)<<3)]);
    }

    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int a_smem_row = wm * 16;
    const int c_col_base = wn * 64;
    const int lane_mod16 = lane & 15;
    const int lane_div16 = lane >> 4;

    uint32_t fA_cur[4], fA_next[4];
    uint32_t fB_cur[8][2], fB_next[8][2];

    ldmatrix_x4(fA_cur, &sA[a_smem_row + lane_mod16][lane_div16 * 8]);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        ldmatrix_x2_trans(fB_cur[ni], &sB[lane_mod16][c_col_base + ni * 8]);
    }

    #pragma unroll
    for (int ki = 0; ki < K_TILE; ki += 16) {
        const int ki_next = ki + 16;
        if (ki_next < K_TILE) {
            ldmatrix_x4(fA_next, &sA[a_smem_row + lane_mod16][ki_next + lane_div16 * 8]);
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                ldmatrix_x2_trans(fB_next[ni], &sB[ki_next + lane_mod16][c_col_base + ni * 8]);
            }
        }
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            mma_m16n8k16(acc[ni], fA_cur, fB_cur[ni], acc[ni]);
        }
        if (ki_next < K_TILE) {
            #pragma unroll
            for (int r = 0; r < 4; r++) fA_cur[r] = fA_next[r];
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fB_cur[ni][0] = fB_next[ni][0];
                fB_cur[ni][1] = fB_next[ni][1];
            }
        }
    }

    half* my_partial = partials + (size_t)split_id * GEMM_M * GEMM_N;
    const int c_row0    = a_smem_row + (lane >> 2);
    const int c_row1    = c_row0 + 8;
    const int lane_mod4 = lane & 3;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c_col = c_col_base + ni * 8 + lane_mod4 * 2;
        half2 h0 = __float22half2_rn(make_float2(
            __uint_as_float(acc[ni][0]), __uint_as_float(acc[ni][1])));
        half2 h1 = __float22half2_rn(make_float2(
            __uint_as_float(acc[ni][2]), __uint_as_float(acc[ni][3])));
        *reinterpret_cast<half2*>(&my_partial[c_row0 * GEMM_N + c_col]) = h0;
        *reinterpret_cast<half2*>(&my_partial[c_row1 * GEMM_N + c_col]) = h1;
    }
}

__global__ __launch_bounds__(256)
void reduce_fp16_to_fp16(
    const half* __restrict__ partials,
    half* __restrict__ C_out
) {
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = GEMM_M * GEMM_N;
    const half* ptr = partials + out_idx;

    float sum = 0.f;

    #pragma unroll 16
    for (int s = 0; s < 128; s++) {
        sum += __half2float(__ldg(ptr + s * stride));
    }

    C_out[out_idx] = __float2half(sum);
}

__global__ __launch_bounds__(256)
void reduce_fp16_vec8(
    const half* __restrict__ partials,
    half* __restrict__ C_out
) {
    const int idx8 = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = idx8 * 8;
    const int stride = GEMM_M * GEMM_N;

    float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
    float s4=0.f, s5=0.f, s6=0.f, s7=0.f;

    const half* ptr = partials + base;

    #pragma unroll 16
    for (int s = 0; s < 128; s++) {
        const half* row = ptr + s * stride;
        half2 v0 = *reinterpret_cast<const half2*>(row + 0);
        half2 v1 = *reinterpret_cast<const half2*>(row + 2);
        half2 v2 = *reinterpret_cast<const half2*>(row + 4);
        half2 v3 = *reinterpret_cast<const half2*>(row + 6);
        s0 += __half2float(v0.x); s1 += __half2float(v0.y);
        s2 += __half2float(v1.x); s3 += __half2float(v1.y);
        s4 += __half2float(v2.x); s5 += __half2float(v2.y);
        s6 += __half2float(v3.x); s7 += __half2float(v3.y);
    }

    half* out = C_out + base;
    *reinterpret_cast<half2*>(out + 0) = __float22half2_rn(make_float2(s0, s1));
    *reinterpret_cast<half2*>(out + 2) = __float22half2_rn(make_float2(s2, s3));
    *reinterpret_cast<half2*>(out + 4) = __float22half2_rn(make_float2(s4, s5));
    *reinterpret_cast<half2*>(out + 6) = __float22half2_rn(make_float2(s6, s7));
}

static half*  g_fp16_partial    = nullptr;
static size_t g_fp16_partial_sz = 0;

static half* get_fp16_partial() {
    const size_t needed = (size_t)NUM_SPLITS * GEMM_M * GEMM_N * sizeof(half);
    if (g_fp16_partial_sz < needed) {
        if (g_fp16_partial) cudaFree(g_fp16_partial);
        cudaMalloc(&g_fp16_partial, needed);
        g_fp16_partial_sz = needed;
    }
    return g_fp16_partial;
}

static int g_coop_supported = -1;

static bool check_coop_support() {
    if (g_coop_supported == -1) {
        int dev;
        cudaGetDevice(&dev);
        int supported;
        cudaDeviceGetAttribute(&supported,
            cudaDevAttrCooperativeLaunch, dev);
        g_coop_supported = supported;
    }
    return g_coop_supported != 0;
}

static bool g_coop_attr_set = false;
static bool g_splitk_attr_set = false;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const half* pA = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* pB = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       pC = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    half* partial_fp16 = get_fp16_partial();

    bool use_coop = check_coop_support();

    if (use_coop) {
        if (!g_coop_attr_set) {
            g_coop_attr_set = true;
        }

        void* args[] = {
            (void*)&pA,
            (void*)&pB,
            (void*)&partial_fp16,
            (void*)&pC
        };

        dim3 grid(NUM_SPLITS);
        dim3 block(BLOCK_THREADS);

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)hgemm_fused_cooperative,
            grid, block,
            args,
            0,
            nullptr
        );

        if (err != cudaSuccess) {
            use_coop = false;
        }
    }

    if (!use_coop) {
        hgemm_splitk128_fp16partial<<<NUM_SPLITS, BLOCK_THREADS>>>(
            pA, pB, partial_fp16
        );

        reduce_fp16_to_fp16<<<32, 256>>>(partial_fp16, pC);
    }
}