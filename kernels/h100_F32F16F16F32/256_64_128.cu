#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(32, 48)
hgemm_fp32_accumulate_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int warp_m = blockIdx.x * 16;
    
    if (warp_m >= M) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[4];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }
    
    const half* A_warp = A + warp_m * K;
    const half* B_base = B;
    
    constexpr int BK = 16;
    constexpr int num_k_iters = 8;
    
    wmma::load_matrix_sync(a_frag[0], A_warp + 0, K);
    
    #pragma unroll
    for (int k_iter = 0; k_iter < num_k_iters; ++k_iter) {
        const int a_current = k_iter & 1;
        const int a_next = (k_iter + 1) & 1;
        const int k_offset = k_iter * BK;
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            wmma::load_matrix_sync(b_frag[i], B_base + i * 16 * K + k_offset, K);
        }
        
        if (k_iter < num_k_iters - 1) {
            wmma::load_matrix_sync(a_frag[a_next], A_warp + (k_iter + 1) * BK, K);
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            wmma::mma_sync(acc_frag[i], a_frag[a_current], b_frag[i], acc_frag[i]);
        }
    }
    
    half* C_warp = C + warp_m * N;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_fp16[4];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < acc_frag[i].num_elements; ++j) {
            acc_fp16[i].x[j] = __float2half_rn(acc_frag[i].x[j]);
        }
        
        wmma::store_matrix_sync(C_warp + i * 16, acc_fp16[i], N, wmma::mem_row_major);
    }
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c
) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    TORCH_CHECK(a.dtype() == torch::kHalf, "Input a must be FP16");
    TORCH_CHECK(b.dtype() == torch::kHalf, "Input b must be FP16");
    TORCH_CHECK(c.dtype() == torch::kHalf, "Output c must be FP16");
    TORCH_CHECK(a.size(0) == M && a.size(1) == K, "Tensor a shape mismatch");
    TORCH_CHECK(b.size(0) == K && b.size(1) == N, "Tensor b shape mismatch");
    TORCH_CHECK(c.size(0) == M && c.size(1) == N, "Tensor c shape mismatch");
    
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr());
    
    const int BM = 16;
    const int num_blocks = (M + BM - 1) / BM;
    
    dim3 grid(num_blocks, 1, 1);
    dim3 block(32, 1, 1);
    
    hgemm_fp32_accumulate_kernel<<<grid, block>>>(
        A_ptr, B_ptr, C_ptr, M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}