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

using namespace nvcuda;

template <int BM, int BN, int BK>
__global__ void __launch_bounds__(32, 32)
hgemm_hyper_fused_pingpong_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K) {
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  const int block_row = by * BM;
  const int block_col = bx * BN;
  
  if (block_row >= M || block_col >= N) return;
  
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_even, a_odd;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_even, b_odd;
  
  wmma::fill_fragment(acc, __float2half(0.0f));
  
  const half* __restrict__ A_base = A + block_row * K;
  const half* __restrict__ B_base = B_col + block_col * K;
  
  wmma::load_matrix_sync(a_even, A_base + 0, K);
  wmma::load_matrix_sync(b_even, B_base + 0, K);
  wmma::load_matrix_sync(a_odd, A_base + 16, K);
  wmma::load_matrix_sync(b_odd, B_base + 16, K);
  
  wmma::mma_sync(acc, a_even, b_even, acc);
  
  wmma::load_matrix_sync(a_even, A_base + 32, K);
  wmma::load_matrix_sync(b_even, B_base + 32, K);
  
  wmma::mma_sync(acc, a_odd, b_odd, acc);
  
  wmma::load_matrix_sync(a_odd, A_base + 48, K);
  wmma::load_matrix_sync(b_odd, B_base + 48, K);
  
  wmma::mma_sync(acc, a_even, b_even, acc);
  
  wmma::mma_sync(acc, a_odd, b_odd, acc);
  
  half* __restrict__ C_ptr = C + block_row * N + block_col;
  wmma::store_matrix_sync(C_ptr, acc, N, wmma::mem_row_major);
}

template <int BM, int BN, int BK>
__global__ void __launch_bounds__(32, 32)
hgemm_extreme_interleaved_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K) {
  
  const int r = blockIdx.y * BM;
  const int c = blockIdx.x * BN;
  
  if (r >= M || c >= N) return;
  
  const half* ap = A + r * K;
  const half* bp = B_col + c * K;
  
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0, a1;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b0, b1;
  
  wmma::fill_fragment(acc, __float2half(0.0f));
  
  wmma::load_matrix_sync(a0, ap + 0, K);
  wmma::load_matrix_sync(b0, bp + 0, K);
  wmma::load_matrix_sync(a1, ap + 16, K);
  wmma::load_matrix_sync(b1, bp + 16, K);
  
  wmma::mma_sync(acc, a0, b0, acc);
  wmma::load_matrix_sync(a0, ap + 32, K);
  wmma::load_matrix_sync(b0, bp + 32, K);
  
  wmma::mma_sync(acc, a1, b1, acc);
  wmma::load_matrix_sync(a1, ap + 48, K);
  wmma::load_matrix_sync(b1, bp + 48, K);
  
  wmma::mma_sync(acc, a0, b0, acc);
  wmma::mma_sync(acc, a1, b1, acc);
  
  wmma::store_matrix_sync(C + r * N + c, acc, N, wmma::mem_row_major);
}

template <int BM, int BN, int BK>
__global__ void __launch_bounds__(32, 32)
hgemm_streamlined_speed_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K) {
  
  const int r = blockIdx.y << 4;
  const int c = blockIdx.x << 4;
  
  if (r >= M || c >= N) return;
  
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
  
  wmma::fill_fragment(acc, __float2half(0.0f));
  
  const half* a_ptr = A + (r * K);
  const half* b_ptr = B_col + (c * K);
  
  #pragma unroll
  for (int k = 0; k < 64; k += 16) {
    wmma::load_matrix_sync(af, a_ptr + k, K);
    wmma::load_matrix_sync(bf, b_ptr + k, K);
    wmma::mma_sync(acc, af, bf, acc);
  }
  
  wmma::store_matrix_sync(C + (r * N) + c, acc, N, wmma::mem_row_major);
}

template <int BM, int BN, int BK>
__global__ void __launch_bounds__(32, 32)
hgemm_quad_parallel_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K) {
  
  const int r = blockIdx.y * BM;
  const int c = blockIdx.x * BN;
  
  if (r >= M || c >= N) return;
  
  const half* a_base = A + r * K;
  const half* b_base = B_col + c * K;
  
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0, a1, a2, a3;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b0, b1, b2, b3;
  
  wmma::fill_fragment(acc, __float2half(0.0f));
  
  wmma::load_matrix_sync(a0, a_base + 0, K);
  wmma::load_matrix_sync(b0, b_base + 0, K);
  wmma::load_matrix_sync(a1, a_base + 16, K);
  wmma::load_matrix_sync(b1, b_base + 16, K);
  wmma::load_matrix_sync(a2, a_base + 32, K);
  wmma::load_matrix_sync(b2, b_base + 32, K);
  wmma::load_matrix_sync(a3, a_base + 48, K);
  wmma::load_matrix_sync(b3, b_base + 48, K);
  
  wmma::mma_sync(acc, a0, b0, acc);
  wmma::mma_sync(acc, a1, b1, acc);
  wmma::mma_sync(acc, a2, b2, acc);
  wmma::mma_sync(acc, a3, b3, acc);
  
  wmma::store_matrix_sync(C + r * N + c, acc, N, wmma::mem_row_major);
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

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
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
  
  const half* a_ptr = reinterpret_cast<const half*>(a.data_ptr());
  const half* b_col_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* c_ptr = reinterpret_cast<half*>(c.data_ptr());
  
  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 16;
  
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(32);
  
  hgemm_hyper_fused_pingpong_kernel<BM, BN, BK>
      <<<grid, block>>>(a_ptr, b_col_ptr, c_ptr, M, N, K);
}