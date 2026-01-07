#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(32)
cuda_l2_a100_fp32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;
    
    const int warp_m = block_m * 16;
    const int warp_n = block_n * 16;
    
    if (warp_m >= M || warp_n >= N) return;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc;
    wmma::fill_fragment(acc, __float2half(0.0f));
    
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int k_base = k * 32;
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag0, b_frag1;
        
        const half* a_ptr0 = A + warp_m * K + k_base;
        const half* a_ptr1 = A + warp_m * K + k_base + 16;
        const half* b_ptr0 = B_col + warp_n * K + k_base;
        const half* b_ptr1 = B_col + warp_n * K + k_base + 16;
        
        wmma::load_matrix_sync(a_frag0, a_ptr0, K);
        wmma::load_matrix_sync(a_frag1, a_ptr1, K);
        wmma::load_matrix_sync(b_frag0, b_ptr0, K);
        wmma::load_matrix_sync(b_frag1, b_ptr1, K);
        
        wmma::mma_sync(acc, a_frag0, b_frag0, acc);
        wmma::mma_sync(acc, a_frag1, b_frag1, acc);
    }
    
    half* c_ptr = C + warp_m * N + warp_n;
    wmma::store_matrix_sync(c_ptr, acc, N, wmma::mem_row_major);
}

void launch_hgemm_optimized(
    const half* a,
    const half* b_col_major,
    half* c,
    int M, int N, int K)
{
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    dim3 block(32);
    
    cuda_l2_a100_fp32_kernel<<<grid, block>>>(a, b_col_major, c, M, N, K);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if (((T).options().dtype() != (th_type))) { \
        throw std::runtime_error("Tensor dtype mismatch"); \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
        throw std::runtime_error("Tensor shape mismatch"); \
    }

void cuda_l2_a100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
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
    
    launch_hgemm_optimized(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K);
}