#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(32) cuda_l2_a100_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K) {
    
    const int tile_m = blockIdx.y * 16;
    const int tile_n = blockIdx.x * 16;
    
    if (tile_m >= M || tile_n >= N) return;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;
    wmma::fill_fragment(acc, __half2float(0.0f));
    
    #pragma unroll
    for (int k = 0; k < K; k += 16) {
        if (k >= K) break;
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        
        const half* a_ptr = A + tile_m * K + k;
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        
        const half* b_ptr = B_col + tile_n * K + k;
        wmma::load_matrix_sync(b_frag, b_ptr, K);
        
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    
    half* c_ptr = C + tile_m * N + tile_n;
    wmma::store_matrix_sync(c_ptr, acc, N, wmma::mem_row_major);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor " #T " must be " #th_type);               \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor " #T " shape mismatch!");                 \
  }

void cuda_l2_a100_fp16(torch::Tensor a, torch::Tensor b, 
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
    CHECK_TORCH_TENSOR_SHAPE(b_col_major, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    
    const int grid_x = (N + 15) / 16;
    const int grid_y = (M + 15) / 16;
    dim3 grid(grid_x, grid_y);
    dim3 block(32);
    
    cuda_l2_a100_fp16_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + 
                               cudaGetErrorString(err));
    }
}