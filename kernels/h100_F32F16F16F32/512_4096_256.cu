#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

#define WARPS_M 4
#define WARPS_N 4
#define NUM_WARPS 16
#define NUM_THREADS 512

#define WARP_TILE_M 32
#define WARP_TILE_N 32

#define NUM_BUFFERS 2

__global__ void __launch_bounds__(NUM_THREADS, 1)
hgemm_wmma_large_tile_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];
    
    const int smem_a_size = BM * (BK + 8);
    const int smem_b_size = BK * (BN + 8);
    
    half* smem_a = smem;
    half* smem_b = smem + NUM_BUFFERS * smem_a_size;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[2][2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], __float2half(0.0f));
        }
    }
    
    constexpr int VECTOR_SIZE = 8;
    
    int write_buf = 0;
    
    auto load_a_tile = [&](int k_offset, int buf_idx) {
        half* buf_a = smem_a + buf_idx * smem_a_size;
        const int total_elements = BM * BK;
        
        for (int idx = tid * VECTOR_SIZE; idx < total_elements; idx += NUM_THREADS * VECTOR_SIZE) {
            int row = idx / BK;
            int col = idx % BK;
            int global_row = by * BM + row;
            int global_col = k_offset + col;
            
            if (global_row < M && global_col + VECTOR_SIZE <= K) {
                *reinterpret_cast<float4*>(&buf_a[row * (BK + 8) + col]) = 
                    *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
            } else {
                #pragma unroll
                for (int v = 0; v < VECTOR_SIZE; ++v) {
                    if (global_row < M && global_col + v < K) {
                        buf_a[row * (BK + 8) + col + v] = A[global_row * K + global_col + v];
                    } else {
                        buf_a[row * (BK + 8) + col + v] = __float2half(0.0f);
                    }
                }
            }
        }
    };
    
    auto load_b_tile = [&](int k_offset, int buf_idx) {
        half* buf_b = smem_b + buf_idx * smem_b_size;
        const int total_elements = BK * BN;
        
        for (int idx = tid * VECTOR_SIZE; idx < total_elements; idx += NUM_THREADS * VECTOR_SIZE) {
            int row = idx / BN;
            int col = idx % BN;
            int global_row = k_offset + row;
            int global_col = bx * BN + col;
            
            if (global_row < K && global_col + VECTOR_SIZE <= N) {
                *reinterpret_cast<float4*>(&buf_b[row * (BN + 8) + col]) = 
                    *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
            } else {
                #pragma unroll
                for (int v = 0; v < VECTOR_SIZE; ++v) {
                    if (global_row < K && global_col + v < N) {
                        buf_b[row * (BN + 8) + col + v] = B[global_row * N + global_col + v];
                    } else {
                        buf_b[row * (BN + 8) + col + v] = __float2half(0.0f);
                    }
                }
            }
        }
    };
    
    load_a_tile(0, write_buf);
    load_b_tile(0, write_buf);
    __syncthreads();
    
    for (int k_block = 0; k_block < K; k_block += BK) {
        int read_buf = write_buf;
        write_buf = 1 - write_buf;
        
        if (k_block + BK < K) {
            load_a_tile(k_block + BK, write_buf);
            load_b_tile(k_block + BK, write_buf);
        }
        
        half* read_a = smem_a + read_buf * smem_a_size;
        half* read_b = smem_b + read_buf * smem_b_size;
        
        #pragma unroll
        for (int k = 0; k < BK; k += WMMA_K) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int row_base = warp_row * WARP_TILE_M + i * WMMA_M;
                wmma::load_matrix_sync(
                    a_frag[i], 
                    &read_a[row_base * (BK + 8) + k], 
                    BK + 8);
            }
            
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int col_base = warp_col * WARP_TILE_N + j * WMMA_N;
                wmma::load_matrix_sync(
                    b_frag[j], 
                    &read_b[k * (BN + 8) + col_base], 
                    BN + 8);
            }
            
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int global_row = by * BM + warp_row * WARP_TILE_M + i * WMMA_M;
            int global_col = bx * BN + warp_col * WARP_TILE_N + j * WMMA_N;
            
            if (global_row < M && global_col < N) {
                wmma::store_matrix_sync(
                    &C[global_row * N + global_col],
                    acc[i][j],
                    N,
                    wmma::mem_row_major);
            }
        }
    }
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);
    
    const int smem_a_size = BM * (BK + 8);
    const int smem_b_size = BK * (BN + 8);
    const int smem_size = NUM_BUFFERS * (smem_a_size + smem_b_size) * sizeof(half);
    
    cudaFuncSetAttribute(
        hgemm_wmma_large_tile_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    
    hgemm_wmma_large_tile_kernel<<<grid, block, smem_size>>>(A, B, C, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}