#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void cuda_l2_a100_fp32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K) {
    
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    
    const int m_base = tile_row * 16;
    const int n_base = tile_col * 16;
    
    if (m_base >= M || n_base >= N) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    #pragma unroll
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        const int k_base = k_tile * 16;
        
        wmma::load_matrix_sync(a_frag, A + m_base * K + k_base, K);
        
        wmma::load_matrix_sync(b_frag, B + n_base * K + k_base, K);
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(C + m_base * N + n_base, c_frag, N, wmma::mem_row_major);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("values must be " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!"); \
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
    
    dim3 grid(4, 4);
    dim3 block(32);
    
    cuda_l2_a100_fp32_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K);
}