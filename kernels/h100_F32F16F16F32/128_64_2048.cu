#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_WARPS 16
#define SMEM_STRIDE 264

__global__ void __launch_bounds__(512, 2)
hgemm_optimized_best(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C)
{
    const int n16_tile = blockIdx.x;
    const int m16_tile = blockIdx.y;
    const int m_start  = m16_tile * WMMA_M;
    const int n_start  = n16_tile * WMMA_N;
    const int warp_id  = threadIdx.x >> 5;
    const int k_start  = warp_id * 128;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a0,a1,a2,a3,a4,a5,a6,a7;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b0,b1,b2,b3,b4,b5,b6,b7;

    const half* Abase = A + m_start * 2048 + k_start;
    const half* Bbase = B + k_start * 64 + n_start;

    wmma::load_matrix_sync(a0, Abase +   0, 2048);
    wmma::load_matrix_sync(a1, Abase +  16, 2048);
    wmma::load_matrix_sync(a2, Abase +  32, 2048);
    wmma::load_matrix_sync(a3, Abase +  48, 2048);
    wmma::load_matrix_sync(a4, Abase +  64, 2048);
    wmma::load_matrix_sync(a5, Abase +  80, 2048);
    wmma::load_matrix_sync(a6, Abase +  96, 2048);
    wmma::load_matrix_sync(a7, Abase + 112, 2048);

    wmma::load_matrix_sync(b0, Bbase +   0*64, 64);
    wmma::load_matrix_sync(b1, Bbase +  16*64, 64);
    wmma::load_matrix_sync(b2, Bbase +  32*64, 64);
    wmma::load_matrix_sync(b3, Bbase +  48*64, 64);
    wmma::load_matrix_sync(b4, Bbase +  64*64, 64);
    wmma::load_matrix_sync(b5, Bbase +  80*64, 64);
    wmma::load_matrix_sync(b6, Bbase +  96*64, 64);
    wmma::load_matrix_sync(b7, Bbase + 112*64, 64);

    wmma::mma_sync(acc, a0, b0, acc);
    wmma::mma_sync(acc, a1, b1, acc);
    wmma::mma_sync(acc, a2, b2, acc);
    wmma::mma_sync(acc, a3, b3, acc);
    wmma::mma_sync(acc, a4, b4, acc);
    wmma::mma_sync(acc, a5, b5, acc);
    wmma::mma_sync(acc, a6, b6, acc);
    wmma::mma_sync(acc, a7, b7, acc);

    __shared__ float smem[NUM_WARPS][SMEM_STRIDE];
    wmma::store_matrix_sync(&smem[warp_id][0], acc, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    if (threadIdx.x < 256) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            v += smem[w][threadIdx.x];
        }
        const int row = threadIdx.x >> 4;
        const int col = threadIdx.x & 15;
        C[(m_start + row) * 64 + (n_start + col)] = __float2half(v);
    }
}

__global__ void __launch_bounds__(512, 2)
hgemm_dual_tile(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C)
{
    const int n32_tile = blockIdx.x;
    const int m16_tile = blockIdx.y;
    const int m_start  = m16_tile * WMMA_M;
    const int n_start0 = n32_tile * 32;
    const int n_start1 = n32_tile * 32 + 16;
    const int warp_id  = threadIdx.x >> 5;
    const int k_start  = warp_id * 128;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a0,a1,a2,a3,a4,a5,a6,a7;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        b0_0,b1_0,b2_0,b3_0,b4_0,b5_0,b6_0,b7_0,
        b0_1,b1_1,b2_1,b3_1,b4_1,b5_1,b6_1,b7_1;

    const half* Abase  = A + m_start * 2048 + k_start;
    const half* Bbase0 = B + k_start * 64 + n_start0;
    const half* Bbase1 = B + k_start * 64 + n_start1;

    wmma::load_matrix_sync(a0, Abase +   0, 2048);
    wmma::load_matrix_sync(a1, Abase +  16, 2048);
    wmma::load_matrix_sync(a2, Abase +  32, 2048);
    wmma::load_matrix_sync(a3, Abase +  48, 2048);
    wmma::load_matrix_sync(a4, Abase +  64, 2048);
    wmma::load_matrix_sync(a5, Abase +  80, 2048);
    wmma::load_matrix_sync(a6, Abase +  96, 2048);
    wmma::load_matrix_sync(a7, Abase + 112, 2048);

    wmma::load_matrix_sync(b0_0, Bbase0 +   0*64, 64);
    wmma::load_matrix_sync(b1_0, Bbase0 +  16*64, 64);
    wmma::load_matrix_sync(b2_0, Bbase0 +  32*64, 64);
    wmma::load_matrix_sync(b3_0, Bbase0 +  48*64, 64);
    wmma::load_matrix_sync(b4_0, Bbase0 +  64*64, 64);
    wmma::load_matrix_sync(b5_0, Bbase0 +  80*64, 64);
    wmma::load_matrix_sync(b6_0, Bbase0 +  96*64, 64);
    wmma::load_matrix_sync(b7_0, Bbase0 + 112*64, 64);

    wmma::load_matrix_sync(b0_1, Bbase1 +   0*64, 64);
    wmma::load_matrix_sync(b1_1, Bbase1 +  16*64, 64);
    wmma::load_matrix_sync(b2_1, Bbase1 +  32*64, 64);
    wmma::load_matrix_sync(b3_1, Bbase1 +  48*64, 64);
    wmma::load_matrix_sync(b4_1, Bbase1 +  64*64, 64);
    wmma::load_matrix_sync(b5_1, Bbase1 +  80*64, 64);
    wmma::load_matrix_sync(b6_1, Bbase1 +  96*64, 64);
    wmma::load_matrix_sync(b7_1, Bbase1 + 112*64, 64);

    wmma::mma_sync(acc0, a0, b0_0, acc0); wmma::mma_sync(acc1, a0, b0_1, acc1);
    wmma::mma_sync(acc0, a1, b1_0, acc0); wmma::mma_sync(acc1, a1, b1_1, acc1);
    wmma::mma_sync(acc0, a2, b2_0, acc0); wmma::mma_sync(acc1, a2, b2_1, acc1);
    wmma::mma_sync(acc0, a3, b3_0, acc0); wmma::mma_sync(acc1, a3, b3_1, acc1);
    wmma::mma_sync(acc0, a4, b4_0, acc0); wmma::mma_sync(acc1, a4, b4_1, acc1);
    wmma::mma_sync(acc0, a5, b5_0, acc0); wmma::mma_sync(acc1, a5, b5_1, acc1);
    wmma::mma_sync(acc0, a6, b6_0, acc0); wmma::mma_sync(acc1, a6, b6_1, acc1);
    wmma::mma_sync(acc0, a7, b7_0, acc0); wmma::mma_sync(acc1, a7, b7_1, acc1);

    __shared__ float smem0[NUM_WARPS][SMEM_STRIDE];
    __shared__ float smem1[NUM_WARPS][SMEM_STRIDE];

    wmma::store_matrix_sync(&smem0[warp_id][0], acc0, WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem1[warp_id][0], acc1, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    const int tid = threadIdx.x;
    if (tid < 256) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) v += smem0[w][tid];
        const int row = tid >> 4;
        const int col = tid & 15;
        C[(m_start + row) * 64 + (n_start0 + col)] = __float2half(v);
    } else {
        const int idx = tid - 256;
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) v += smem1[w][idx];
        const int row = idx >> 4;
        const int col = idx & 15;
        C[(m_start + row) * 64 + (n_start1 + col)] = __float2half(v);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        dim3 grid(4, 8, 1);
        dim3 block(512);
        hgemm_optimized_best<<<grid, block>>>(ptr_A, ptr_B, ptr_C);
    }
}