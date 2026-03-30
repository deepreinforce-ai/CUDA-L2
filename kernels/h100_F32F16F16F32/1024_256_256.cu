#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 64

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (WARP_SIZE * NUM_WARPS)

#define SMEM_PAD 8

#define VECTOR_SIZE 8

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
hgemm_wmma_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;

    __shared__ half smem_A[2][BM][BK + SMEM_PAD];
    __shared__ half smem_B[2][BN][BK + SMEM_PAD];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        wmma::fill_fragment(acc[i], __float2half(0.0f));
    }

    const half* A_block = A + by * BM * K;
    const half* B_block = B + bx * BN * K;

    const int num_k_tiles = (K + BK - 1) / BK;
    
    int write_stage = 0;

    {
        const int k_offset = 0;
        
        for (int i = tid * VECTOR_SIZE; i < BM * BK; i += THREADS_PER_BLOCK * VECTOR_SIZE) {
            const int row = i / BK;
            const int col = i % BK;
            
            const int global_row = by * BM + row;
            const int global_col = k_offset + col;
            
            if (global_row < M && global_col + VECTOR_SIZE <= K && col + VECTOR_SIZE <= BK) {
                *((float4*)(&smem_A[0][row][col])) = 
                    *((const float4*)(&A_block[row * K + col]));
            } else {
                #pragma unroll
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    if (global_row < M && (global_col + v) < K && (col + v) < BK) {
                        smem_A[0][row][col + v] = A_block[row * K + col + v];
                    } else {
                        smem_A[0][row][col + v] = __float2half(0.0f);
                    }
                }
            }
        }
        
        for (int i = tid * VECTOR_SIZE; i < BN * BK; i += THREADS_PER_BLOCK * VECTOR_SIZE) {
            const int row = i / BK;
            const int col = i % BK;
            
            const int global_row = bx * BN + row;
            const int global_col = k_offset + col;
            
            if (global_row < N && global_col + VECTOR_SIZE <= K && col + VECTOR_SIZE <= BK) {
                *((float4*)(&smem_B[0][row][col])) = 
                    *((const float4*)(&B_block[row * K + col]));
            } else {
                #pragma unroll
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    if (global_row < N && (global_col + v) < K && (col + v) < BK) {
                        smem_B[0][row][col + v] = B_block[row * K + col + v];
                    } else {
                        smem_B[0][row][col + v] = __float2half(0.0f);
                    }
                }
            }
        }
    }
    
    __syncthreads();

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int read_stage = write_stage;
        write_stage ^= 1;

        if (k_tile + 1 < num_k_tiles) {
            const int k_offset = (k_tile + 1) * BK;
            
            for (int i = tid * VECTOR_SIZE; i < BM * BK; i += THREADS_PER_BLOCK * VECTOR_SIZE) {
                const int row = i / BK;
                const int col = i % BK;
                
                const int global_row = by * BM + row;
                const int global_col = k_offset + col;
                
                if (global_row < M && global_col + VECTOR_SIZE <= K && col + VECTOR_SIZE <= BK) {
                    *((float4*)(&smem_A[write_stage][row][col])) = 
                        *((const float4*)(&A_block[row * K + k_offset + col]));
                } else {
                    #pragma unroll
                    for (int v = 0; v < VECTOR_SIZE; v++) {
                        if (global_row < M && (global_col + v) < K && (col + v) < BK) {
                            smem_A[write_stage][row][col + v] = A_block[row * K + k_offset + col + v];
                        } else {
                            smem_A[write_stage][row][col + v] = __float2half(0.0f);
                        }
                    }
                }
            }
            
            for (int i = tid * VECTOR_SIZE; i < BN * BK; i += THREADS_PER_BLOCK * VECTOR_SIZE) {
                const int row = i / BK;
                const int col = i % BK;
                
                const int global_row = bx * BN + row;
                const int global_col = k_offset + col;
                
                if (global_row < N && global_col + VECTOR_SIZE <= K && col + VECTOR_SIZE <= BK) {
                    *((float4*)(&smem_B[write_stage][row][col])) = 
                        *((const float4*)(&B_block[row * K + k_offset + col]));
                } else {
                    #pragma unroll
                    for (int v = 0; v < VECTOR_SIZE; v++) {
                        if (global_row < N && (global_col + v) < K && (col + v) < BK) {
                            smem_B[write_stage][row][col + v] = B_block[row * K + k_offset + col + v];
                        } else {
                            smem_B[write_stage][row][col + v] = __float2half(0.0f);
                        }
                    }
                }
            }
        }

        const int warp_tile_base = warp_id * 2;
        
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                const int tile_idx = warp_tile_base + t;
                const int tile_m = tile_idx / 4;
                const int tile_n = tile_idx % 4;
                
                wmma::load_matrix_sync(
                    a_frag,
                    &smem_A[read_stage][tile_m * WMMA_M][kk * WMMA_K],
                    BK + SMEM_PAD
                );
                
                wmma::load_matrix_sync(
                    b_frag,
                    &smem_B[read_stage][tile_n * WMMA_N][kk * WMMA_K],
                    BK + SMEM_PAD
                );
                
                wmma::mma_sync(acc[t], a_frag, b_frag, acc[t]);
            }
        }

        __syncthreads();
    }

    const int warp_tile_base = warp_id * 2;
    
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        const int tile_idx = warp_tile_base + t;
        const int tile_m = tile_idx / 4;
        const int tile_n = tile_idx % 4;
        
        const int c_row = by * BM + tile_m * WMMA_M;
        const int c_col = bx * BN + tile_n * WMMA_N;
        
        if (c_row < M && c_col < N) {
            wmma::store_matrix_sync(
                C + c_row * N + c_col,
                acc[t],
                N,
                wmma::mem_row_major
            );
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c) 
{
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS_PER_BLOCK);
    
    hgemm_wmma_optimized<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}