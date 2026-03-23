#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(32, 48)
cuda_l2_3090_fp32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K) {
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int tile_m = by * 16;
    const int tile_n = bx * 16;
    
    if (tile_m >= M || tile_n >= N) return;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    
    #pragma unroll
    for (int k_tile = 0; k_tile < 4; k_tile++) {
        const int k_offset = k_tile * 16;
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, A + tile_m * K + k_offset, K);
        
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, B + k_offset * N + tile_n, N);
        
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_half;
    #pragma unroll
    for (int i = 0; i < acc.num_elements; i++) {
        acc_half.x[i] = __float2half(acc.x[i]);
    }
    
    wmma::store_matrix_sync(C + tile_m * N + tile_n, acc_half, N, wmma::mem_row_major);
}

void cuda_l2_3090_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
    TORCH_CHECK(a.scalar_type() == torch::kHalf, "Expected FP16 for A");
    TORCH_CHECK(b.scalar_type() == torch::kHalf, "Expected FP16 for B");
    TORCH_CHECK(c.scalar_type() == torch::kHalf, "Expected FP16 for C");
    
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    constexpr int TILE_SIZE = 16;
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(32);
    
    cuda_l2_3090_fp32_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
}