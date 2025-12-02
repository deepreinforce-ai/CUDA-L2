#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>

using namespace cute;
using namespace nvcuda;

__global__ void __launch_bounds__(32, 4) cuda_l2_a100_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K) {
    
    const int warp_id = blockIdx.x;
    const int total_warps = gridDim.x;
    
    constexpr int total_tiles = 16;
    
    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += total_warps) {
        const int tile_m = (tile_idx >> 2) << 4;
        const int tile_n = (tile_idx & 3) << 4;
        
        const half* a_base = A + tile_m * K;
        const half* b_base = B_col + tile_n * K;
        half* c_base = C + tile_m * N + tile_n;
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
        wmma::fill_fragment(acc_frag, __float2half(0.0f));
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        
        wmma::load_matrix_sync(a_frag, a_base + 0, K);
        wmma::load_matrix_sync(b_frag, b_base + 0, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::load_matrix_sync(a_frag, a_base + 16, K);
        wmma::load_matrix_sync(b_frag, b_base + 16, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::load_matrix_sync(a_frag, a_base + 32, K);
        wmma::load_matrix_sync(b_frag, b_base + 32, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::load_matrix_sync(a_frag, a_base + 48, K);
        wmma::load_matrix_sync(b_frag, b_base + 48, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::load_matrix_sync(a_frag, a_base + 64, K);
        wmma::load_matrix_sync(b_frag, b_base + 64, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::load_matrix_sync(a_frag, a_base + 80, K);
        wmma::load_matrix_sync(b_frag, b_base + 80, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::load_matrix_sync(a_frag, a_base + 96, K);
        wmma::load_matrix_sync(b_frag, b_base + 96, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::load_matrix_sync(a_frag, a_base + 112, K);
        wmma::load_matrix_sync(b_frag, b_base + 112, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        wmma::store_matrix_sync(c_base, acc_frag, N, wmma::mem_row_major);
    }
}

template <typename T>
void launch_hgemm_ultra_optimized(T *a, T *b_col, T *c, int M, int N, int K) {
    dim3 block(32);
    dim3 grid(216);
    
    cuda_l2_a100_fp16_kernel<<<grid, block>>>(a, b_col, c, M, N, K);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_a100_fp16(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  launch_hgemm_ultra_optimized<half>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}