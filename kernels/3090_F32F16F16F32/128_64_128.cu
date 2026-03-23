#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(128) cuda_l2_3090_fp32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    const int M, const int N, const int K
) {
    const int tile_idx = blockIdx.x;
    const int tiles_n = 4;
    
    const int tile_m = tile_idx / tiles_n;
    const int tile_n = tile_idx % tiles_n;
    
    const int m_base = tile_m * 16;
    const int n_base = tile_n * 16;
    
    if (m_base >= M || n_base >= N) return;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    __shared__ half smem_A[16 * 16];
    __shared__ half smem_B[16 * 16];
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    if (warp_id == 0) {
        wmma::fill_fragment(acc_frag, 0.0f);
    }
    
    const int k_tiles = (K + 15) / 16;
    
    for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
        const int k_base = k_tile * 16;
        const int k_size = min(16, K - k_base);
        
        const int tid = threadIdx.x;
        
        for (int i = 0; i < 2; i++) {
            int idx = tid * 2 + i;
            int row = idx / 16;
            int col = idx % 16;
            
            if (row < 16 && col < k_size && m_base + row < M && k_base + col < K) {
                smem_A[row * 16 + col] = A[(m_base + row) * K + (k_base + col)];
            } else if (row < 16) {
                smem_A[row * 16 + col] = __float2half(0.0f);
            }
        }
        
        for (int i = 0; i < 2; i++) {
            int idx = tid * 2 + i;
            int col = idx / 16;
            int row = idx % 16;
            
            if (col < 16 && row < k_size && n_base + col < N && k_base + row < K) {
                smem_B[col * 16 + row] = B_col[(n_base + col) * K + (k_base + row)];
            } else if (col < 16) {
                smem_B[col * 16 + row] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        if (warp_id == 0) {
            wmma::load_matrix_sync(a_frag, smem_A, 16);
            wmma::load_matrix_sync(b_frag, smem_B, 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        __syncthreads();
    }
    
    if (warp_id == 0) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
        
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; i++) {
            c_frag.x[i] = __float2half(acc_frag.x[i]);
        }
        
        if (m_base + 16 <= M && n_base + 16 <= N) {
            wmma::store_matrix_sync(
                C + m_base * N + n_base,
                c_frag,
                N,
                wmma::mem_row_major
            );
        } else {
            half temp[256];
            wmma::store_matrix_sync(temp, c_frag, 16, wmma::mem_row_major);
            
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    if (m_base + i < M && n_base + j < N) {
                        C[(m_base + i) * N + (n_base + j)] = temp[i * 16 + j];
                    }
                }
            }
        }
    }
}

void cuda_l2_3090_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    TORCH_CHECK(a.dtype() == torch::kHalf, "A must be FP16");
    TORCH_CHECK(b.dtype() == torch::kHalf, "B must be FP16");
    TORCH_CHECK(b_col_major.dtype() == torch::kHalf, "B_col must be FP16");
    TORCH_CHECK(c.dtype() == torch::kHalf, "C must be FP16");
    
    TORCH_CHECK(a.size(0) == M && a.size(1) == K, "A shape mismatch");
    TORCH_CHECK(b.size(0) == K && b.size(1) == N, "B shape mismatch");
    TORCH_CHECK(b_col_major.size(0) == K && b_col_major.size(1) == N, "B_col shape mismatch");
    TORCH_CHECK(c.size(0) == M && c.size(1) == N, "C shape mismatch");
    
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_col_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr());
    
    const int tiles_m = (M + 15) / 16;
    const int tiles_n = (N + 15) / 16;
    const int total_tiles = tiles_m * tiles_n;
    
    dim3 grid(total_tiles);
    dim3 block(128);
    
    cuda_l2_3090_fp32_kernel<<<grid, block>>>(
        A_ptr, B_col_ptr, C_ptr, M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
}