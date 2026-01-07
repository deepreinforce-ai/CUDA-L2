#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

template <typename T>
__global__ void cuda_l2_a100_fp32_kernel(const T* __restrict__ A,
                                                   const T* __restrict__ B,
                                                   T* __restrict__ C,
                                                   int M, int N, int K) {
  constexpr int BM = 8;
  constexpr int BN = 32;
  constexpr int BK = 16;
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  const int tile_m = by * BM;
  const int tile_n = bx * BN;
  
  if (tile_m >= M || tile_n >= N) return;
  
  wmma::fragment<wmma::accumulator, 8, 32, 16, half> acc_frag;
  wmma::fill_fragment(acc_frag, __float2half(0.0f));
  
  const int num_k_tiles = (K + BK - 1) / BK;
  
  #pragma unroll
  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    const int k_offset = k_tile * BK;
    
    if (k_offset >= K) break;
    
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
    if (tile_m < M && k_offset < K) {
      wmma::load_matrix_sync(a_frag, A + tile_m * K + k_offset, K);
    } else {
      wmma::fill_fragment(a_frag, __float2half(0.0f));
    }
    
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
    if (tile_n < N && k_offset < K) {
      wmma::load_matrix_sync(b_frag, B + tile_n * K + k_offset, K);
    } else {
      wmma::fill_fragment(b_frag, __float2half(0.0f));
    }
    
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  
  if (tile_m < M && tile_n < N) {
    wmma::store_matrix_sync(C + tile_m * N + tile_n, acc_frag, N, wmma::mem_row_major);
  }
}

template <typename T>
void launch_hgemm_optimized(T* a, T* b, T* c, int M, int N, int K) {
  constexpr int BM = 8;
  constexpr int BN = 32;
  
  const int grid_x = (N + BN - 1) / BN;
  const int grid_y = (M + BM - 1) / BM;
  
  dim3 block(32);
  dim3 grid(grid_x, grid_y);
  
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
      reinterpret_cast<half*>(c.data_ptr()),
      M, N, K);
}