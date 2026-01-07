#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <mma.h>

using namespace nvcuda;

template <typename T>
__global__ void cuda_l2_a100_fp32_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B, 
    T* __restrict__ C,
    int M, int N, int K) {
    
  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  constexpr int TILE_K = 16;
  
  int block_id = blockIdx.x;
  
  int tiles_x = (N + TILE_N - 1) / TILE_N;
  int tiles_y = (M + TILE_M - 1) / TILE_M;
  int total_tiles = tiles_x * tiles_y;
  
  int tile_id = block_id % total_tiles;
  int tile_y = tile_id / tiles_x;
  int tile_x = tile_id % tiles_x;
  
  int row_base = tile_y * TILE_M;
  int col_base = tile_x * TILE_N;
  
  if (row_base >= M || col_base >= N) return;
  
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
  
  wmma::fill_fragment(acc_frag, __float2half(0.0f));
  
  #pragma unroll
  for (int k = 0; k < K; k += TILE_K) {
    wmma::load_matrix_sync(a_frag, A + row_base * K + k, K);
    
    wmma::load_matrix_sync(b_frag, B + col_base * K + k, K);
    
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  
  if (block_id / total_tiles == 0) {
    wmma::store_matrix_sync(C + row_base * N + col_base, acc_frag, N, wmma::mem_row_major);
  }
}

template <typename T>
void launch_hgemm_optimized(T* a, T* b, T* c, int M, int N, int K) {
  constexpr int TILE_M = 16;
  constexpr int TILE_N = 16;
  
  constexpr int TARGET_BLOCKS = 108;
  
  dim3 block(128);
  dim3 grid(TARGET_BLOCKS);
  
  cuda_l2_a100_fp32_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
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

void cuda_l2_a100_fp32(torch::Tensor a, torch::Tensor b, 
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

  launch_hgemm_optimized<half>(
      reinterpret_cast<half*>(a.data_ptr()),
      reinterpret_cast<half*>(b_col_major.data_ptr()),
      reinterpret_cast<half*>(c.data_ptr()), M, N, K);
}