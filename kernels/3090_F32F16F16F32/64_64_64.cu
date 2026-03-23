#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(128, 8) cuda_l2_3090_fp32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C)
{
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int m_base = tile_m * 16;
    const int n_base = tile_n * 16;
    
    const int warp_id = threadIdx.x / 32;
    
    if (warp_id == 0) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
        
        wmma::fill_fragment(acc_frag, __float2half(0.0f));
        
        #pragma unroll
        for (int k_tile = 0; k_tile < 4; k_tile++) {
            const int k_base = k_tile * 16;
            
            wmma::load_matrix_sync(a_frag, A + m_base * 64 + k_base, 64);
            wmma::load_matrix_sync(b_frag, B + k_base * 64 + n_base, 64);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        wmma::store_matrix_sync(C + m_base * 64 + n_base, acc_frag, 64, wmma::mem_row_major);
    }
}

void cuda_l2_3090_fp32(torch::Tensor a, torch::Tensor b, 
                                torch::Tensor b_col_major, torch::Tensor c) {
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    
    dim3 block(128);
    dim3 grid(4, 4);
    
    cuda_l2_3090_fp32_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr);
}