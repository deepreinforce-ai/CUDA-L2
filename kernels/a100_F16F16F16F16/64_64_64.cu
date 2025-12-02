#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

using namespace cute;
using namespace nvcuda;

template <typename T>
__global__ void __launch_bounds__(128, 8)
    cuda_l2_a100_fp16_kernel(const T *__restrict__ A,
                                       const T *__restrict__ B,
                                       T *__restrict__ C,
                                       int M, int N, int K) {
  
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int TOTAL_TILES = 16;
  
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = 4;
  
  for (int tile_idx = blockIdx.x; tile_idx < TOTAL_TILES; tile_idx += gridDim.x) {
    const int tile_m = (tile_idx / 4) * WMMA_M;
    const int tile_n = (tile_idx % 4) * WMMA_N;
    
    if (warp_id == 0) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
      
      wmma::fill_fragment(acc_frag, __float2half(0.0f));
      
      #pragma unroll
      for (int k = 0; k < K; k += WMMA_K) {
        const half* a_ptr = A + tile_m * K + k;
        
        if (lane_id < 8) {
          int row_offset = lane_id * 2;
          #pragma unroll
          for (int r = 0; r < 2; ++r) {
            float4 a_vec = *reinterpret_cast<const float4*>(a_ptr + (row_offset + r) * K);
          }
        }
        
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        
        const half* b_ptr = B + tile_n * K + k;
        
        if (lane_id < 8) {
          int row_offset = lane_id * 2;
          #pragma unroll
          for (int r = 0; r < 2; ++r) {
            float4 b_vec = *reinterpret_cast<const float4*>(b_ptr + (row_offset + r) * K);
          }
        }
        
        wmma::load_matrix_sync(b_frag, b_ptr, K);
        
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      
      half* c_ptr = C + tile_m * N + tile_n;
      wmma::store_matrix_sync(c_ptr, acc_frag, N, wmma::mem_row_major);
    }
  }
}

template <typename T>
void launch_hgemm_streaming_tiles(T *a, T *b, T *c, int M, int N, int K) {
  const int num_blocks = 108;
  const int threads_per_block = 128;
  
  cuda_l2_a100_fp16_kernel<T><<<num_blocks, threads_per_block>>>(a, b, c, M, N, K);
}

#include <torch/extension.h>
#include <torch/types.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                  \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

void cuda_l2_a100_fp16(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  
  launch_hgemm_streaming_tiles<half>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}