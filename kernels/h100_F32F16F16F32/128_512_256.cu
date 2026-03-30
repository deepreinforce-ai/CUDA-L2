#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

__global__ void __launch_bounds__(32, 20)
hgemm_async_3stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int lane = threadIdx.x;

    const int row = bm * 16;
    const int col = bn * 32;

    __shared__ __align__(128) half sA[3][32][24];
    __shared__ __align__(128) half sB[3][32][40];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa0, fa1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb00, fb01, fb10, fb11;

    const int r  = lane >> 1;
    const int lc = (lane & 1) << 3;

    const half* A_base = A + (row + r) * K;
    const half* B_base = B;

#define LOAD_STAGE(s, kc) \
    { \
        const int nk = (kc) * 32; \
        __pipeline_memcpy_async(&sA[s][r   ][lc],    A_base + nk + lc,                           16); \
        __pipeline_memcpy_async(&sA[s][r+16][lc],    A_base + nk + 16 + lc,                      16); \
        __pipeline_memcpy_async(&sB[s][r   ][lc],    B_base + (nk+r   )*N + col + lc,            16); \
        __pipeline_memcpy_async(&sB[s][r   ][lc+16], B_base + (nk+r   )*N + col + lc + 16,       16); \
        __pipeline_memcpy_async(&sB[s][r+16][lc],    B_base + (nk+r+16)*N + col + lc,            16); \
        __pipeline_memcpy_async(&sB[s][r+16][lc+16], B_base + (nk+r+16)*N + col + lc + 16,       16); \
        __pipeline_commit(); \
    }

    LOAD_STAGE(0, 0)
    LOAD_STAGE(1, 1)

#define COMPUTE_STAGE(s) \
    { \
        wmma::load_matrix_sync(fa0,  &sA[s][0 ][0],  24); \
        wmma::load_matrix_sync(fb00, &sB[s][0 ][0],  40); \
        wmma::load_matrix_sync(fb01, &sB[s][0 ][16], 40); \
        wmma::mma_sync(acc0, fa0, fb00, acc0); \
        wmma::mma_sync(acc1, fa0, fb01, acc1); \
        wmma::load_matrix_sync(fa1,  &sA[s][16][0],  24); \
        wmma::load_matrix_sync(fb10, &sB[s][16][0],  40); \
        wmma::load_matrix_sync(fb11, &sB[s][16][16], 40); \
        wmma::mma_sync(acc0, fa1, fb10, acc0); \
        wmma::mma_sync(acc1, fa1, fb11, acc1); \
    }

    LOAD_STAGE(2, 2)
    __pipeline_wait_prior(2);
    __syncwarp();
    COMPUTE_STAGE(0)

    LOAD_STAGE(0, 3)
    __pipeline_wait_prior(2);
    __syncwarp();
    COMPUTE_STAGE(1)

    LOAD_STAGE(1, 4)
    __pipeline_wait_prior(2);
    __syncwarp();
    COMPUTE_STAGE(2)

    LOAD_STAGE(0, 5)
    __pipeline_wait_prior(2);
    __syncwarp();
    COMPUTE_STAGE(0)

    LOAD_STAGE(1, 6)
    __pipeline_wait_prior(2);
    __syncwarp();
    COMPUTE_STAGE(1)

    LOAD_STAGE(2, 7)
    __pipeline_wait_prior(2);
    __syncwarp();
    COMPUTE_STAGE(2)

    __pipeline_wait_prior(1);
    __syncwarp();
    COMPUTE_STAGE(0)

    __pipeline_wait_prior(0);
    __syncwarp();
    COMPUTE_STAGE(1)

#undef LOAD_STAGE
#undef COMPUTE_STAGE

    __shared__ __align__(128) float tmp0[16][16];
    __shared__ __align__(128) float tmp1[16][16];
    wmma::store_matrix_sync(&tmp0[0][0], acc0, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp1[0][0], acc1, 16, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int i = lane; i < 256; i += 32) {
        const int ri = i >> 4;
        const int ci = i & 15;
        C[(row + ri) * N + (col + ci)]      = __float2half(tmp0[ri][ci]);
        C[(row + ri) * N + (col + 16 + ci)] = __float2half(tmp1[ri][ci]);
    }
}

__global__ void __launch_bounds__(32, 24)
hgemm_regpf_doublebuf(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int lane = threadIdx.x;

    const int row = bm * 16;
    const int col = bn * 32;

    __shared__ __align__(128) half sA[2][32][24];
    __shared__ __align__(128) half sB[2][32][40];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa0, fa1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb00, fb01, fb10, fb11;

    const int r   = lane >> 1;
    const int lc  = (lane & 1) << 3;

    const half* A_row  = A + (row + r) * K;
    const half* B_base = B;

    float4 pa0  = *((const float4*)(A_row + lc));
    float4 pa1  = *((const float4*)(A_row + 16 + lc));
    float4 pb00 = *((const float4*)(B_base + r       * N + col + lc));
    float4 pb01 = *((const float4*)(B_base + r       * N + col + 16 + lc));
    float4 pb10 = *((const float4*)(B_base + (16 + r)* N + col + lc));
    float4 pb11 = *((const float4*)(B_base + (16 + r)* N + col + 16 + lc));

    *((float4*)&sA[0][r   ][lc])    = pa0;
    *((float4*)&sA[0][r+16][lc])    = pa1;
    *((float4*)&sB[0][r   ][lc])    = pb00;
    *((float4*)&sB[0][r   ][16+lc]) = pb01;
    *((float4*)&sB[0][r+16][lc])    = pb10;
    *((float4*)&sB[0][r+16][16+lc]) = pb11;
    __syncthreads();

#define KSTEP(kc) \
    { \
        const int cur = (kc) & 1; \
        const int nxt = 1 - cur; \
        if ((kc) + 1 < 8) { \
            const int nk = ((kc)+1)*32; \
            pa0  = *((const float4*)(A_row + nk + lc)); \
            pa1  = *((const float4*)(A_row + nk + 16 + lc)); \
            pb00 = *((const float4*)(B_base + (nk+r   )*N + col + lc)); \
            pb01 = *((const float4*)(B_base + (nk+r   )*N + col + 16 + lc)); \
            pb10 = *((const float4*)(B_base + (nk+r+16)*N + col + lc)); \
            pb11 = *((const float4*)(B_base + (nk+r+16)*N + col + 16 + lc)); \
        } \
        wmma::load_matrix_sync(fa0,  &sA[cur][0 ][0],  24); \
        wmma::load_matrix_sync(fb00, &sB[cur][0 ][0],  40); \
        wmma::load_matrix_sync(fb01, &sB[cur][0 ][16], 40); \
        wmma::mma_sync(acc0, fa0, fb00, acc0); \
        wmma::mma_sync(acc1, fa0, fb01, acc1); \
        wmma::load_matrix_sync(fa1,  &sA[cur][16][0],  24); \
        wmma::load_matrix_sync(fb10, &sB[cur][16][0],  40); \
        wmma::load_matrix_sync(fb11, &sB[cur][16][16], 40); \
        wmma::mma_sync(acc0, fa1, fb10, acc0); \
        wmma::mma_sync(acc1, fa1, fb11, acc1); \
        if ((kc) + 1 < 8) { \
            *((float4*)&sA[nxt][r   ][lc])    = pa0; \
            *((float4*)&sA[nxt][r+16][lc])    = pa1; \
            *((float4*)&sB[nxt][r   ][lc])    = pb00; \
            *((float4*)&sB[nxt][r   ][16+lc]) = pb01; \
            *((float4*)&sB[nxt][r+16][lc])    = pb10; \
            *((float4*)&sB[nxt][r+16][16+lc]) = pb11; \
            __syncthreads(); \
        } \
    }

    KSTEP(0) KSTEP(1) KSTEP(2) KSTEP(3)
    KSTEP(4) KSTEP(5) KSTEP(6) KSTEP(7)
#undef KSTEP

    __shared__ __align__(128) float tmp0[16][16];
    __shared__ __align__(128) float tmp1[16][16];
    wmma::store_matrix_sync(&tmp0[0][0], acc0, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(&tmp1[0][0], acc1, 16, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int i = lane; i < 256; i += 32) {
        const int ri = i >> 4;
        const int ci = i & 15;
        C[(row + ri) * N + (col + ci)]      = __float2half(tmp0[ri][ci]);
        C[(row + ri) * N + (col + 16 + ci)] = __float2half(tmp1[ri][ci]);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C       = reinterpret_cast<half*>(c.data_ptr());

    dim3 grid(N / 32, M / 16);
    dim3 block(32);

    hgemm_regpf_doublebuf<<<grid, block>>>(A, B, C, M, N, K);
}