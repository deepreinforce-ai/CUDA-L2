#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>

using namespace nvcuda;

__global__ void cuda_l2_a100_fp32_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    int total_tiles)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
        const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
        const int tile_row = (tile_id / tiles_n) * WMMA_M;
        const int tile_col = (tile_id % tiles_n) * WMMA_N;
        
        if (tile_row >= M || tile_col >= N) continue;
        
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
        
        wmma::fill_fragment(acc_frag, __float2half(0.0f));
        
        #pragma unroll
        for (int k = 0; k < K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, A + tile_row * K + k, K);
            
            wmma::load_matrix_sync(b_frag, B + tile_col * K + k, K);
            
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        wmma::store_matrix_sync(C + tile_row * N + tile_col, acc_frag, N, wmma::mem_row_major);
    }
}

void cuda_l2_a100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    
    const int tiles_m = (M + WMMA_M - 1) / WMMA_M;
    const int tiles_n = (N + WMMA_N - 1) / WMMA_N;
    const int total_tiles = tiles_m * tiles_n;
    
    constexpr int num_sms = 108;
    dim3 gridDim(num_sms);
    dim3 blockDim(32);
    
    cuda_l2_a100_fp32_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const half*>(b_col_major.data_ptr()),
        reinterpret_cast<half*>(c.data_ptr()),
        M, N, K, total_tiles
    );
}