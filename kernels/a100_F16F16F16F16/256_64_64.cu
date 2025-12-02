#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(64, 16)
cuda_l2_a100_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K) {
    
    const int tile_m = blockIdx.y * 8;
    const int tile_n = blockIdx.x * 16;
    
    if (tile_m >= M || tile_n >= N) return;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    const int wmma_m = (tile_m / 16) * 16;
    const int wmma_n = (tile_n / 16) * 16;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    
    const half* A_base = A + wmma_m * K;
    const half* B_base = B_col + wmma_n * K;
    
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, A_base, K);
        wmma::load_matrix_sync(b_frag, B_base, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, A_base + 16, K);
        wmma::load_matrix_sync(b_frag, B_base + 16, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, A_base + 32, K);
        wmma::load_matrix_sync(b_frag, B_base + 32, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, A_base + 48, K);
        wmma::load_matrix_sync(b_frag, B_base + 48, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    __shared__ half smem[16 * 16];
    wmma::store_matrix_sync(smem, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    const int local_m = tile_m % 16;
    const int local_n = tile_n % 16;
    
    half* C_tile = C + tile_m * N + tile_n;
    
    const int total_elements = 8 * 16;
    const int elements_per_thread = 2;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        const int idx = threadIdx.x * elements_per_thread + i;
        if (idx < total_elements) {
            const int row = idx / 16;
            const int col = idx % 16;
            
            if (tile_m + row < M && tile_n + col < N) {
                const int smem_row = local_m + row;
                const int smem_col = local_n + col;
                C_tile[row * N + col] = smem[smem_row * 16 + smem_col];
            }
        }
    }
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
    
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr());
    
    const int TILE_M = 8;
    const int TILE_N = 16;
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(64);
    
    cuda_l2_a100_fp16_kernel<<<grid, block>>>(
        A_ptr, B_ptr, C_ptr, M, N, K);
}