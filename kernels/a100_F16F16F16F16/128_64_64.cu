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

using namespace nvcuda;

__global__ void __launch_bounds__(128, 4)
cuda_l2_a100_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K) {
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    
    const int block_base_m = block_row << 6;
    const int block_base_n = block_col << 4;
    
    const int warp_m = block_base_m + (warp_id << 4);
    const int warp_n = block_base_n;
    
    if (warp_m >= M || warp_n >= N) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[3];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[3];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
    
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    
    constexpr int K_TILES = 4;
    
    if (K_TILES >= 1) {
        wmma::load_matrix_sync(a_frag[0], A + warp_m * K + 0, K);
        wmma::load_matrix_sync(b_frag[0], B + warp_n * K + 0, K);
    }
    if (K_TILES >= 2) {
        wmma::load_matrix_sync(a_frag[1], A + warp_m * K + 16, K);
        wmma::load_matrix_sync(b_frag[1], B + warp_n * K + 16, K);
    }
    
    #pragma unroll
    for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
        const int curr_buf = k_iter % 3;
        const int next_buf = (k_iter + 2) % 3;
        const int k_offset = k_iter << 4;
        
        if (k_iter + 2 < K_TILES) {
            const int prefetch_k = (k_iter + 2) << 4;
            wmma::load_matrix_sync(a_frag[next_buf], A + warp_m * K + prefetch_k, K);
            wmma::load_matrix_sync(b_frag[next_buf], B + warp_n * K + prefetch_k, K);
        }
        
        wmma::mma_sync(acc_frag, a_frag[curr_buf], b_frag[curr_buf], acc_frag);
    }
    
    wmma::store_matrix_sync(C + warp_m * N + warp_n, acc_frag, N, wmma::mem_row_major);
}

void launch_hgemm_hybrid_wmma(half *a, half *b, half *c, int M, int N, int K) {
    dim3 block(128);
    dim3 grid(4, 2);
    
    cuda_l2_a100_fp16_kernel<<<grid, block>>>(a, b, c, M, N, K);
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

void cuda_l2_a100_fp16(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  
  launch_hgemm_hybrid_wmma(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}