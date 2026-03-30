#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda::wmma;

#define M_TOTAL   64
#define N_TOTAL   64
#define K_TOTAL   2048
#define KB        64
#define SA_STRIDE 72
#define SB_STRIDE 72

__global__ __launch_bounds__(128, 4)
void hgemm_optimized_v2(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C)
{
    const int cta_m = blockIdx.y * 16;
    const int cta_n = blockIdx.x * 16;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int k_per_warp   = K_TOTAL / 4;
    const int k_warp_start = warp_id * k_per_warp;
    const int num_kb       = k_per_warp / KB;

    __shared__ __half sA[4][2][16][SA_STRIDE];
    __shared__ __half sB[4][2][16][SB_STRIDE];
    __shared__ float  red[4][16][16];

    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);

    const int row   = lane_id >> 1;
    const int k_off = (lane_id & 1) * 32;
    const int n_loc = lane_id >> 1;

    const __half* A_row = A + (size_t)(cta_m + row) * K_TOTAL;
    const __half* B_n   = B_col + (size_t)(cta_n + n_loc) * K_TOTAL;

    #define LOAD_A(buf, k_base) do {                                          \
        const __half* _src = A_row + (k_base) + k_off;                       \
        __half* _dst = &sA[warp_id][buf][row][k_off];                        \
        *reinterpret_cast<float4*>(_dst)      = *reinterpret_cast<const float4*>(_src);      \
        *reinterpret_cast<float4*>(_dst + 8)  = *reinterpret_cast<const float4*>(_src + 8);  \
        *reinterpret_cast<float4*>(_dst + 16) = *reinterpret_cast<const float4*>(_src + 16); \
        *reinterpret_cast<float4*>(_dst + 24) = *reinterpret_cast<const float4*>(_src + 24); \
    } while(0)

    #define LOAD_B(buf, k_base) do {                                          \
        const __half* _src = B_n + (k_base) + k_off;                         \
        __half* _dst = &sB[warp_id][buf][n_loc][k_off];                      \
        *reinterpret_cast<float4*>(_dst)      = *reinterpret_cast<const float4*>(_src);      \
        *reinterpret_cast<float4*>(_dst + 8)  = *reinterpret_cast<const float4*>(_src + 8);  \
        *reinterpret_cast<float4*>(_dst + 16) = *reinterpret_cast<const float4*>(_src + 16); \
        *reinterpret_cast<float4*>(_dst + 24) = *reinterpret_cast<const float4*>(_src + 24); \
    } while(0)

    #define DO_MMA(cur) do {                                                   \
        fragment<matrix_a, 16, 16, 16, __half, row_major> _af0, _af1, _af2, _af3; \
        fragment<matrix_b, 16, 16, 16, __half, col_major> _bf0, _bf1, _bf2, _bf3; \
        load_matrix_sync(_af0, &sA[warp_id][cur][0][ 0], SA_STRIDE);          \
        load_matrix_sync(_bf0, &sB[warp_id][cur][0][ 0], SB_STRIDE);          \
        load_matrix_sync(_af1, &sA[warp_id][cur][0][16], SA_STRIDE);          \
        load_matrix_sync(_bf1, &sB[warp_id][cur][0][16], SB_STRIDE);          \
        mma_sync(acc, _af0, _bf0, acc);                                        \
        load_matrix_sync(_af2, &sA[warp_id][cur][0][32], SA_STRIDE);          \
        load_matrix_sync(_bf2, &sB[warp_id][cur][0][32], SB_STRIDE);          \
        mma_sync(acc, _af1, _bf1, acc);                                        \
        load_matrix_sync(_af3, &sA[warp_id][cur][0][48], SA_STRIDE);          \
        load_matrix_sync(_bf3, &sB[warp_id][cur][0][48], SB_STRIDE);          \
        mma_sync(acc, _af2, _bf2, acc);                                        \
        mma_sync(acc, _af3, _bf3, acc);                                        \
    } while(0)

    LOAD_A(0, k_warp_start);
    LOAD_B(0, k_warp_start);
    __syncwarp();

    LOAD_A(1, k_warp_start + KB);
    LOAD_B(1, k_warp_start + KB);

    DO_MMA(0);
    __syncwarp();

    #pragma unroll
    for (int wb = 1; wb < num_kb - 1; wb++) {
        int cur = wb & 1;
        int nxt = cur ^ 1;
        LOAD_A(nxt, k_warp_start + (wb + 1) * KB);
        LOAD_B(nxt, k_warp_start + (wb + 1) * KB);
        DO_MMA(cur);
        __syncwarp();
    }

    {
        int cur = (num_kb - 1) & 1;
        DO_MMA(cur);
        __syncwarp();
    }

    #undef LOAD_A
    #undef LOAD_B
    #undef DO_MMA

    store_matrix_sync(&red[warp_id][0][0], acc, 16, mem_row_major);
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e  = lane_id * 8 + i;
            const int lm = e >> 4;
            const int ln = e & 15;
            const float sum = red[0][lm][ln] + red[1][lm][ln]
                            + red[2][lm][ln] + red[3][lm][ln];
            C[(cta_m + lm) * N_TOTAL + (cta_n + ln)] = __float2half(sum);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const __half* A_ptr = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(b_col_major.data_ptr<at::Half>());
    __half* C_ptr       = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    dim3 grid(4, 4);
    dim3 block(128);
    hgemm_optimized_v2<<<grid, block>>>(A_ptr, B_ptr, C_ptr);
}