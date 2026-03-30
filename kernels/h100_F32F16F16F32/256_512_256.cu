#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor shape mismatch"); \
  }

template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__(128, 8)
software_pipelined_triple_buffer_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int global_warp_id = blockIdx.x * 4 + threadIdx.x / 32;
    
    constexpr int num_tiles_n = 32;
    const int tile_m = global_warp_id / num_tiles_n;
    const int tile_n = global_warp_id % num_tiles_n;
    
    const int warp_row = tile_m * WMMA_M;
    const int warp_col = tile_n * WMMA_N;
    if (warp_row >= M || warp_col >= N) return;
    
    const half* A_base = A + warp_row * K;
    const half* B_base = B + warp_col * K;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[3];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[3];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    wmma::load_matrix_sync(a_frag[0], A_base + WMMA_K * 0, K);
    wmma::load_matrix_sync(b_frag[0], B_base + WMMA_K * 0, K);
    
    wmma::load_matrix_sync(a_frag[1], A_base + WMMA_K * 1, K);
    wmma::load_matrix_sync(b_frag[1], B_base + WMMA_K * 1, K);
    
    wmma::load_matrix_sync(a_frag[2], A_base + WMMA_K * 2, K);
    wmma::load_matrix_sync(b_frag[2], B_base + WMMA_K * 2, K);
    
    wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
    wmma::load_matrix_sync(a_frag[0], A_base + WMMA_K * 3, K);
    wmma::load_matrix_sync(b_frag[0], B_base + WMMA_K * 3, K);
    
    wmma::mma_sync(c_frag, a_frag[1], b_frag[1], c_frag);
    wmma::load_matrix_sync(a_frag[1], A_base + WMMA_K * 4, K);
    wmma::load_matrix_sync(b_frag[1], B_base + WMMA_K * 4, K);
    
    wmma::mma_sync(c_frag, a_frag[2], b_frag[2], c_frag);
    wmma::load_matrix_sync(a_frag[2], A_base + WMMA_K * 5, K);
    wmma::load_matrix_sync(b_frag[2], B_base + WMMA_K * 5, K);
    
    wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
    wmma::load_matrix_sync(a_frag[0], A_base + WMMA_K * 6, K);
    wmma::load_matrix_sync(b_frag[0], B_base + WMMA_K * 6, K);
    
    wmma::mma_sync(c_frag, a_frag[1], b_frag[1], c_frag);
    wmma::load_matrix_sync(a_frag[1], A_base + WMMA_K * 7, K);
    wmma::load_matrix_sync(b_frag[1], B_base + WMMA_K * 7, K);
    
    wmma::mma_sync(c_frag, a_frag[2], b_frag[2], c_frag);
    wmma::load_matrix_sync(a_frag[2], A_base + WMMA_K * 8, K);
    wmma::load_matrix_sync(b_frag[2], B_base + WMMA_K * 8, K);
    
    wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
    wmma::load_matrix_sync(a_frag[0], A_base + WMMA_K * 9, K);
    wmma::load_matrix_sync(b_frag[0], B_base + WMMA_K * 9, K);
    
    wmma::mma_sync(c_frag, a_frag[1], b_frag[1], c_frag);
    wmma::load_matrix_sync(a_frag[1], A_base + WMMA_K * 10, K);
    wmma::load_matrix_sync(b_frag[1], B_base + WMMA_K * 10, K);
    
    wmma::mma_sync(c_frag, a_frag[2], b_frag[2], c_frag);
    wmma::load_matrix_sync(a_frag[2], A_base + WMMA_K * 11, K);
    wmma::load_matrix_sync(b_frag[2], B_base + WMMA_K * 11, K);
    
    wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
    wmma::load_matrix_sync(a_frag[0], A_base + WMMA_K * 12, K);
    wmma::load_matrix_sync(b_frag[0], B_base + WMMA_K * 12, K);
    
    wmma::mma_sync(c_frag, a_frag[1], b_frag[1], c_frag);
    wmma::load_matrix_sync(a_frag[1], A_base + WMMA_K * 13, K);
    wmma::load_matrix_sync(b_frag[1], B_base + WMMA_K * 13, K);
    
    wmma::mma_sync(c_frag, a_frag[2], b_frag[2], c_frag);
    wmma::load_matrix_sync(a_frag[2], A_base + WMMA_K * 14, K);
    wmma::load_matrix_sync(b_frag[2], B_base + WMMA_K * 14, K);
    
    wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
    wmma::load_matrix_sync(a_frag[0], A_base + WMMA_K * 15, K);
    wmma::load_matrix_sync(b_frag[0], B_base + WMMA_K * 15, K);
    
    wmma::mma_sync(c_frag, a_frag[1], b_frag[1], c_frag);
    wmma::mma_sync(c_frag, a_frag[2], b_frag[2], c_frag);
    wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
    
    __syncwarp();
    
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b, 
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b_col_major, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARPS_PER_CTA = 4;
    
    const int num_tiles = (M / WMMA_M) * (N / WMMA_N);
    const int num_ctas = (num_tiles + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    
    dim3 grid(num_ctas);
    dim3 block(WARPS_PER_CTA * 32);
    
    const half* a_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* b_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* c_ptr = reinterpret_cast<half*>(c.data_ptr());
    
    software_pipelined_triple_buffer_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
        a_ptr, b_ptr, c_ptr, M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}