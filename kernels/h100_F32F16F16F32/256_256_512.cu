#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int THREADS_PER_CTA = 32;

constexpr int K_ITERATIONS = 32;

__global__ void __launch_bounds__(THREADS_PER_CTA, 20)
hgemm_wmma_ultimate_col_major_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
  const int tile_m = blockIdx.y;
  const int tile_n = blockIdx.x;
  
  const int warp_row = tile_m * TILE_M;
  const int warp_col = tile_n * TILE_N;
  
  if (warp_row >= M || warp_col >= N) return;
  
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc;
  wmma::fill_fragment(acc, __float2half(0.0f));
  
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  
  const half* a_base = A + warp_row * K;
  const half* b_base = B_col + warp_col * K;
  
  #pragma unroll
  for (int k = 0; k < K_ITERATIONS; k += 8) {
    wmma::load_matrix_sync(a_frag, a_base + k * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + k * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    
    wmma::load_matrix_sync(a_frag, a_base + (k+1) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + (k+1) * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    
    wmma::load_matrix_sync(a_frag, a_base + (k+2) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + (k+2) * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    
    wmma::load_matrix_sync(a_frag, a_base + (k+3) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + (k+3) * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    
    wmma::load_matrix_sync(a_frag, a_base + (k+4) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + (k+4) * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    
    wmma::load_matrix_sync(a_frag, a_base + (k+5) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + (k+5) * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    
    wmma::load_matrix_sync(a_frag, a_base + (k+6) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + (k+6) * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    
    wmma::load_matrix_sync(a_frag, a_base + (k+7) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag, b_base + (k+7) * WMMA_K, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
  }
  
  wmma::store_matrix_sync(C + warp_row * N + warp_col, acc, N, wmma::mem_row_major);
}

__global__ void __launch_bounds__(THREADS_PER_CTA, 20)
hgemm_wmma_register_blocked_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
  const int tile_m = blockIdx.y;
  const int tile_n = blockIdx.x;
  
  const int warp_row = tile_m * TILE_M;
  const int warp_col = tile_n * TILE_N;
  
  if (warp_row >= M || warp_col >= N) return;
  
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc;
  wmma::fill_fragment(acc, __float2half(0.0f));
  
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2];
  
  const half* a_base = A + warp_row * K;
  const half* b_base = B_col + warp_col * K;
  
  #pragma unroll
  for (int k = 0; k < K_ITERATIONS; k += 2) {
    wmma::load_matrix_sync(a_frag[0], a_base + k * WMMA_K, K);
    wmma::load_matrix_sync(a_frag[1], a_base + (k+1) * WMMA_K, K);
    wmma::load_matrix_sync(b_frag[0], b_base + k * WMMA_K, K);
    wmma::load_matrix_sync(b_frag[1], b_base + (k+1) * WMMA_K, K);
    
    wmma::mma_sync(acc, a_frag[0], b_frag[0], acc);
    wmma::mma_sync(acc, a_frag[1], b_frag[1], acc);
  }
  
  wmma::store_matrix_sync(C + warp_row * N + warp_col, acc, N, wmma::mem_row_major);
}

void launch_hgemm_wmma_ultimate(
    const half* A, 
    const half* B_col,
    half* C,
    int M, int N, int K)
{
  const int tiles_m = (M + TILE_M - 1) / TILE_M;
  const int tiles_n = (N + TILE_N - 1) / TILE_N;
  
  dim3 grid(tiles_n, tiles_m);
  dim3 block(THREADS_PER_CTA);
  
  hgemm_wmma_ultimate_col_major_kernel<<<grid, block>>>(A, B_col, C, M, N, K);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!"); \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  launch_hgemm_wmma_ultimate(
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b_col_major.data_ptr()),
      reinterpret_cast<half*>(c.data_ptr()),
      M, N, K);
}