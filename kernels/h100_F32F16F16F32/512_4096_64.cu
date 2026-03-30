#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda::wmma;

__global__ void hgemm_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B, 
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 64;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id >> 1;
    int warp_col = warp_id & 1;
    
    __shared__ half smemA[BLOCK_M][BLOCK_K + 8];
    __shared__ half smemB[BLOCK_K][BLOCK_N + 8];
    
    int tid = threadIdx.x;
    
    {
        int row_a = tid;
        int global_row_a = by * BLOCK_M + row_a;
        if (global_row_a < M) {
            const half* src = &A[global_row_a * K];
            half* dst = &smemA[row_a][0];
            *reinterpret_cast<float4*>(dst + 0)  = *reinterpret_cast<const float4*>(src + 0);
            *reinterpret_cast<float4*>(dst + 8)  = *reinterpret_cast<const float4*>(src + 8);
            *reinterpret_cast<float4*>(dst + 16) = *reinterpret_cast<const float4*>(src + 16);
            *reinterpret_cast<float4*>(dst + 24) = *reinterpret_cast<const float4*>(src + 24);
            *reinterpret_cast<float4*>(dst + 32) = *reinterpret_cast<const float4*>(src + 32);
            *reinterpret_cast<float4*>(dst + 40) = *reinterpret_cast<const float4*>(src + 40);
            *reinterpret_cast<float4*>(dst + 48) = *reinterpret_cast<const float4*>(src + 48);
            *reinterpret_cast<float4*>(dst + 56) = *reinterpret_cast<const float4*>(src + 56);
        }
    }
    
    {
        int row_b = tid % BLOCK_K;
        int col_base_b = (tid / BLOCK_K) * 64;
        int global_col_b = bx * BLOCK_N + col_base_b;
        int global_row_b = row_b;
        
        if (global_row_b < K && global_col_b + 63 < N) {
            const half* src = &B[global_row_b * N + global_col_b];
            half* dst = &smemB[row_b][col_base_b];
            *reinterpret_cast<float4*>(dst + 0)  = *reinterpret_cast<const float4*>(src + 0);
            *reinterpret_cast<float4*>(dst + 8)  = *reinterpret_cast<const float4*>(src + 8);
            *reinterpret_cast<float4*>(dst + 16) = *reinterpret_cast<const float4*>(src + 16);
            *reinterpret_cast<float4*>(dst + 24) = *reinterpret_cast<const float4*>(src + 24);
            *reinterpret_cast<float4*>(dst + 32) = *reinterpret_cast<const float4*>(src + 32);
            *reinterpret_cast<float4*>(dst + 40) = *reinterpret_cast<const float4*>(src + 40);
            *reinterpret_cast<float4*>(dst + 48) = *reinterpret_cast<const float4*>(src + 48);
            *reinterpret_cast<float4*>(dst + 56) = *reinterpret_cast<const float4*>(src + 56);
        } else if (global_row_b < K) {
            for (int i = 0; i < 64; i++) {
                int gc = global_col_b + i;
                smemB[row_b][col_base_b + i] = (gc < N) ? B[global_row_b * N + gc] : __float2half(0.f);
            }
        }
    }
    
    __syncthreads();
    
    int warp_m_base = warp_row * 64;
    int warp_n_base = warp_col * 64;
    
    const int WT_M = 4;
    const int WT_N = 4;
    
    fragment<accumulator, 16, 16, 16, float> frag_c[WT_M][WT_N];
    #pragma unroll
    for (int m = 0; m < WT_M; m++)
        #pragma unroll
        for (int n = 0; n < WT_N; n++)
            fill_fragment(frag_c[m][n], 0.0f);
    
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        fragment<matrix_a, 16, 16, 16, half, row_major> frag_a[WT_M];
        fragment<matrix_b, 16, 16, 16, half, row_major> frag_b[WT_N];
        
        #pragma unroll
        for (int m = 0; m < WT_M; m++) {
            load_matrix_sync(frag_a[m], &smemA[warp_m_base + m*16][k*16], BLOCK_K + 8);
        }
        #pragma unroll
        for (int n = 0; n < WT_N; n++) {
            load_matrix_sync(frag_b[n], &smemB[k*16][warp_n_base + n*16], BLOCK_N + 8);
        }
        #pragma unroll
        for (int m = 0; m < WT_M; m++)
            #pragma unroll
            for (int n = 0; n < WT_N; n++)
                mma_sync(frag_c[m][n], frag_a[m], frag_b[n], frag_c[m][n]);
    }
    
    #pragma unroll
    for (int m = 0; m < WT_M; m++) {
        #pragma unroll
        for (int n = 0; n < WT_N; n++) {
            int c_row = by * BLOCK_M + warp_m_base + m * 16;
            int c_col = bx * BLOCK_N + warp_n_base + n * 16;
            if (c_row < M && c_col < N) {
                fragment<accumulator, 16, 16, 16, half> tmp;
                #pragma unroll
                for (int i = 0; i < (int)frag_c[m][n].num_elements; i++) {
                    tmp.x[i] = __float2half(frag_c[m][n].x[i]);
                }
                store_matrix_sync(C + c_row * N + c_col, tmp, N, mem_row_major);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(128);
    
    hgemm_optimized<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
}