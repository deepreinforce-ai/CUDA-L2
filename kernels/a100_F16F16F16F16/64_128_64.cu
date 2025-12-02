#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(128, 4)
cuda_l2_a100_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K) {
  
  const int block_m = blockIdx.x / 8;
  const int block_n = blockIdx.x % 8;
  const int tile_row = block_m * 16;
  const int tile_col = block_n * 16;
  
  const half* A_ptr_base = A + tile_row * K;
  const half* B_ptr_base = B_col + tile_col * K;
  half* C_ptr = C + tile_row * N + tile_col;
  
  const half* A_ptrs[4];
  const half* B_ptrs[4];
  
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    A_ptrs[k] = A_ptr_base + k * 16;
    B_ptrs[k] = B_ptr_base + k * 16;
  }
  
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frags[4];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frags[4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
  
  wmma::fill_fragment(acc_frag, __float2half(0.0f));
  
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    wmma::load_matrix_sync(a_frags[k], A_ptrs[k], K);
    wmma::load_matrix_sync(b_frags[k], B_ptrs[k], K);
  }
  
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    wmma::mma_sync(acc_frag, a_frags[k], b_frags[k], acc_frag);
  }
  
  wmma::store_matrix_sync(C_ptr, acc_frag, N, wmma::mem_row_major);
}

void launch_hgemm_optimized(half *a, half *b_col, half *c, int M, int N, int K) {
  dim3 grid(32);
  dim3 block(128);
  
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

  launch_hgemm_optimized(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}