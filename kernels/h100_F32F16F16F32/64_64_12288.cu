#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cooperative_groups.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

#define BM 64
#define BN 64
#define BK 32
#define SPLIT_K 128
#define BK_PER_CTA 96
#define NUM_STAGES 3
#define A_PAD 8
#define A_STRIDE (BK + A_PAD)

__global__ __launch_bounds__(128, 4)
void hgemm_cooperative(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_partial,
    half* __restrict__ C,
    int K,
    int MN
) {
    cg::grid_group grid = cg::this_grid();
    
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_row = warp_id * 16;
    const int k_base  = (int)blockIdx.x * BK_PER_CTA;

    __shared__ __align__(128) half smA[NUM_STAGES][BM][A_STRIDE];
    __shared__ __align__(128) half smB[NUM_STAGES][BN][A_STRIDE];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int n = 0; n < 4; n++)
        wmma::fill_fragment(acc[n], 0.0f);

    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES; stage++) {
        const int ks = k_base + stage * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int lin = tid + i * 128;
            const int m   = lin >> 2;
            const int kk  = (lin & 3) << 3;
            __pipeline_memcpy_async(&smA[stage][m][kk], A + (size_t)m * K + ks + kk, 16);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int lin = tid + i * 128;
            const int n   = lin >> 2;
            const int kk  = (lin & 3) << 3;
            __pipeline_memcpy_async(&smB[stage][n][kk], B_col + (size_t)n * K + ks + kk, 16);
        }
        __pipeline_commit();
    }

    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES; stage++) {
        __pipeline_wait_prior(NUM_STAGES - 1 - stage);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[2][4];

        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++)
            wmma::load_matrix_sync(fa[k16], &smA[stage][warp_row][k16 * 16], A_STRIDE);
        #pragma unroll
        for (int n16 = 0; n16 < 4; n16++)
            #pragma unroll
            for (int k16 = 0; k16 < 2; k16++)
                wmma::load_matrix_sync(fb[k16][n16], &smB[stage][n16 * 16][k16 * 16], A_STRIDE);
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++)
            #pragma unroll
            for (int n16 = 0; n16 < 4; n16++)
                wmma::mma_sync(acc[n16], fa[k16], fb[k16][n16], acc[n16]);

        if (stage < NUM_STAGES - 1) __syncthreads();
    }

    float* out = C_partial + (int)blockIdx.x * BM * BN;
    #pragma unroll
    for (int n16 = 0; n16 < 4; n16++) {
        wmma::store_matrix_sync(out + warp_row * BN + n16 * 16, acc[n16], BN, wmma::mem_row_major);
    }

    grid.sync();

    {
        const int global_tid = blockIdx.x * 128 + tid;
        const int my_elem = global_tid >> 2;
        const int my_sk_group = global_tid & 3;
        const int sk_start = my_sk_group * 32;

        float partial_sum = 0.f;
        const float* col = C_partial + my_elem;
        #pragma unroll
        for (int s = 0; s < 32; s++) {
            partial_sum += col[(size_t)(sk_start + s) * MN];
        }

        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, 1);
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, 2);
        
        if ((global_tid & 3) == 0 && my_elem < MN) {
            C[my_elem] = __float2half(partial_sum);
        }
    }
}

__global__ __launch_bounds__(128, 5)
void hgemm_main_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_partial,
    int K
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_row = warp_id * 16;
    const int k_base  = (int)blockIdx.z * BK_PER_CTA;

    __shared__ __align__(128) half smA[NUM_STAGES][BM][A_STRIDE];
    __shared__ __align__(128) half smB[NUM_STAGES][BN][A_STRIDE];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int n = 0; n < 4; n++)
        wmma::fill_fragment(acc[n], 0.0f);

    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES; stage++) {
        const int ks = k_base + stage * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int lin = tid + i * 128;
            const int m   = lin >> 2;
            const int kk  = (lin & 3) << 3;
            __pipeline_memcpy_async(&smA[stage][m][kk], A + (size_t)m * K + ks + kk, 16);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int lin = tid + i * 128;
            const int n   = lin >> 2;
            const int kk  = (lin & 3) << 3;
            __pipeline_memcpy_async(&smB[stage][n][kk], B_col + (size_t)n * K + ks + kk, 16);
        }
        __pipeline_commit();
    }

    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES; stage++) {
        __pipeline_wait_prior(NUM_STAGES - 1 - stage);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[2][4];

        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++)
            wmma::load_matrix_sync(fa[k16], &smA[stage][warp_row][k16 * 16], A_STRIDE);
        #pragma unroll
        for (int n16 = 0; n16 < 4; n16++)
            #pragma unroll
            for (int k16 = 0; k16 < 2; k16++)
                wmma::load_matrix_sync(fb[k16][n16], &smB[stage][n16 * 16][k16 * 16], A_STRIDE);
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++)
            #pragma unroll
            for (int n16 = 0; n16 < 4; n16++)
                wmma::mma_sync(acc[n16], fa[k16], fb[k16][n16], acc[n16]);

        if (stage < NUM_STAGES - 1) __syncthreads();
    }

    float* out = C_partial + (int)blockIdx.z * BM * BN;
    #pragma unroll
    for (int n16 = 0; n16 < 4; n16++) {
        wmma::store_matrix_sync(out + warp_row * BN + n16 * 16, acc[n16], BN, wmma::mem_row_major);
    }
}

__global__ __launch_bounds__(256, 6)
void hgemm_reduce_best(
    const float* __restrict__ C_partial,
    half* __restrict__ C,
    int MN
) {
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int out_idx = blockIdx.x * 8 + warp_id;
    if (out_idx >= MN) return;

    const float* col = C_partial + out_idx;
    float sum = 0.f;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int s = lane + i * 32;
        sum += __ldg(col + (size_t)s * MN);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0)
        C[out_idx] = __float2half(sum);
}

__global__ __launch_bounds__(256, 6)
void hgemm_reduce_dual(
    const float* __restrict__ C_partial,
    half* __restrict__ C,
    int MN
) {
    const int warp_id  = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int base_out = blockIdx.x * 16 + warp_id * 2;
    if (base_out + 1 >= MN) return;

    const float* col0 = C_partial + base_out;
    const float* col1 = C_partial + base_out + 1;
    float sum0 = 0.f, sum1 = 0.f;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int s = lane + i * 32;
        const size_t offset = (size_t)s * MN;
        sum0 += __ldg(col0 + offset);
        sum1 += __ldg(col1 + offset);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, off);
        sum1 += __shfl_down_sync(0xffffffff, sum1, off);
    }

    if (lane == 0) {
        *reinterpret_cast<half2*>(C + base_out) = __floats2half2_rn(sum0, sum1);
    }
}

static float* g_partial = nullptr;
static size_t g_partial_bytes = 0;
static bool g_coop_supported = false;
static bool g_coop_checked = false;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M  = (int)a.size(0);
    const int K  = (int)a.size(1);
    const int N  = (int)b.size(1);
    const int MN = M * N;

    const half* A     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C           = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const size_t need = (size_t)SPLIT_K * MN * sizeof(float);
    if (g_partial_bytes < need) {
        if (g_partial) cudaFree(g_partial);
        cudaMalloc(&g_partial, need);
        g_partial_bytes = need;
    }

    if (!g_coop_checked) {
        int dev;
        cudaGetDevice(&dev);
        int supports_coop = 0;
        cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, dev);
        g_coop_supported = (supports_coop != 0);
        g_coop_checked = true;
    }

    if (g_coop_supported) {
        void* args[] = {
            (void*)&A, (void*)&B_col, (void*)&g_partial, (void*)&C, (void*)&K, (void*)&MN
        };
        cudaLaunchCooperativeKernel(
            (void*)hgemm_cooperative,
            dim3(SPLIT_K),
            dim3(128),
            args,
            0,
            nullptr
        );
    } else {
        hgemm_main_kernel<<<dim3(1, 1, SPLIT_K), 128>>>(A, B_col, g_partial, K);
        hgemm_reduce_best<<<(MN + 7) / 8, 256>>>(g_partial, C, MN);
    }
}