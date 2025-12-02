#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <algorithm>
#include <vector>

using namespace nvcuda;

__global__ void __launch_bounds__(64) cuda_l2_a100_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 32;
    
    int tile_n_base = blockIdx.x + blockIdx.z * gridDim.x;
    int tile_m = blockIdx.y;
    
    if (tile_m * TILE_M >= M || tile_n_base * TILE_N >= N) return;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    int tile_n = tile_n_base * 2 + warp_id;
    
    if (tile_n * WMMA_N >= N) return;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    
    const half* a_tile_ptr = A + tile_m * WMMA_M * K;
    const half* b_tile_ptr = B_col_major + tile_n * WMMA_N * K;
    
    int buf_idx = 0;
    
    wmma::load_matrix_sync(a_frag[0], a_tile_ptr, K);
    wmma::load_matrix_sync(b_frag[0], b_tile_ptr, K);
    
    #pragma unroll
    for (int k_idx = 0; k_idx < 4; k_idx++) {
        int k_start = k_idx * WMMA_K;
        int next_buf = 1 - buf_idx;
        
        if (k_idx < 3) {
            int next_k = k_start + WMMA_K;
            wmma::load_matrix_sync(a_frag[next_buf], a_tile_ptr + next_k, K);
            wmma::load_matrix_sync(b_frag[next_buf], b_tile_ptr + next_k, K);
        }
        
        wmma::mma_sync(acc_frag, a_frag[buf_idx], b_frag[buf_idx], acc_frag);
        
        buf_idx = next_buf;
    }
    
    half* c_ptr = C + tile_m * WMMA_M * N + tile_n * WMMA_N;
    wmma::store_matrix_sync(c_ptr, acc_frag, N, wmma::mem_row_major);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch: expected " #th_type);     \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_a100_fp16(
    torch::Tensor a, 
    torch::Tensor b, 
    torch::Tensor b_col_major, 
    torch::Tensor c) {
    
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
    
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 32;
    constexpr int THREADS = 64;
    
    int grid_x = (N + TILE_N - 1) / TILE_N;
    int grid_y = (M + TILE_M - 1) / TILE_M;
    int grid_z = 32;
    
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(THREADS, 1, 1);
    
    cuda_l2_a100_fp16_kernel<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr()),
        reinterpret_cast<half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
    }
}