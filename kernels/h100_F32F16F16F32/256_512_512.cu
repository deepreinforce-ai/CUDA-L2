#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 64
#define BK 64

#define WARP_TILE_M 32
#define WARP_TILE_N 32
#define WARPS_M 2
#define WARPS_N 2
#define NUM_WARPS 4
#define THREADS_PER_BLOCK 128

#define PIPELINE_STAGES 2

#define SMEM_PAD 8
#define SMEM_STRIDE_A (BK + SMEM_PAD)
#define SMEM_STRIDE_B (BK + SMEM_PAD)

#define ASYNC_COPY_BYTES 16

__device__ __forceinline__ void load_tile_A_async_strategic(
    const half* __restrict__ A_global,
    half* __restrict__ A_shared,
    int M, int K,
    int block_m, int k_offset,
    int tid,
    bool use_cg = true)
{
    constexpr int total_elements = BM * BK;
    constexpr int elements_per_cp = ASYNC_COPY_BYTES / sizeof(half);
    constexpr int num_loads = (total_elements + THREADS_PER_BLOCK * elements_per_cp - 1) / (THREADS_PER_BLOCK * elements_per_cp);
    
    #pragma unroll
    for (int load_iter = 0; load_iter < num_loads; ++load_iter) {
        int linear_idx = (tid + load_iter * THREADS_PER_BLOCK) * elements_per_cp;
        if (linear_idx >= total_elements) break;
        
        int row = linear_idx / BK;
        int col = linear_idx % BK;
        int global_row = block_m * BM + row;
        int global_col = k_offset + col;
        
        uint32_t smem_addr = __cvta_generic_to_shared(&A_shared[row * SMEM_STRIDE_A + col]);
        
        if (global_row < M && global_col + elements_per_cp <= K) {
            const void* global_addr = &A_global[global_row * K + global_col];
            if (use_cg) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :: "r"(smem_addr), "l"(global_addr), "n"(ASYNC_COPY_BYTES)
                );
            } else {
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], %2;\n"
                    :: "r"(smem_addr), "l"(global_addr), "n"(ASYNC_COPY_BYTES)
                );
            }
        } else {
            #pragma unroll
            for (int v = 0; v < elements_per_cp; ++v) {
                if (global_row < M && global_col + v < K) {
                    A_shared[row * SMEM_STRIDE_A + col + v] = A_global[global_row * K + global_col + v];
                } else {
                    A_shared[row * SMEM_STRIDE_A + col + v] = __float2half(0.0f);
                }
            }
        }
    }
}

__device__ __forceinline__ void load_tile_B_async_strategic(
    const half* __restrict__ B_col_major,
    half* __restrict__ B_shared,
    int N, int K,
    int block_n, int k_offset,
    int tid,
    bool use_cg = true)
{
    constexpr int total_elements = BN * BK;
    constexpr int elements_per_cp = ASYNC_COPY_BYTES / sizeof(half);
    constexpr int num_loads = (total_elements + THREADS_PER_BLOCK * elements_per_cp - 1) / (THREADS_PER_BLOCK * elements_per_cp);
    
    #pragma unroll
    for (int load_iter = 0; load_iter < num_loads; ++load_iter) {
        int linear_idx = (tid + load_iter * THREADS_PER_BLOCK) * elements_per_cp;
        if (linear_idx >= total_elements) break;
        
        int row = linear_idx / BK;
        int col = linear_idx % BK;
        int global_n = block_n * BN + row;
        int global_k = k_offset + col;
        
        uint32_t smem_addr = __cvta_generic_to_shared(&B_shared[row * SMEM_STRIDE_B + col]);
        
        if (global_n < N && global_k + elements_per_cp <= K) {
            const void* global_addr = &B_col_major[global_n * K + global_k];
            if (use_cg) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :: "r"(smem_addr), "l"(global_addr), "n"(ASYNC_COPY_BYTES)
                );
            } else {
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], %2;\n"
                    :: "r"(smem_addr), "l"(global_addr), "n"(ASYNC_COPY_BYTES)
                );
            }
        } else {
            #pragma unroll
            for (int v = 0; v < elements_per_cp; ++v) {
                if (global_n < N && global_k + v < K) {
                    B_shared[row * SMEM_STRIDE_B + col + v] = B_col_major[global_n * K + global_k + v];
                } else {
                    B_shared[row * SMEM_STRIDE_B + col + v] = __float2half(0.0f);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4)
hgemm_perfect_software_pipeline_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    
    __shared__ half smem_A[PIPELINE_STAGES][BM][SMEM_STRIDE_A];
    __shared__ half smem_B[PIPELINE_STAGES][BN][SMEM_STRIDE_B];
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], __float2half(0.0f));
        }
    }
    
    const int num_k_tiles = (K + BK - 1) / BK;
    if (num_k_tiles == 0) return;
    
    const int m_offset = warp_row * WARP_TILE_M;
    const int n_offset = warp_col * WARP_TILE_N;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2][2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2][2];
    
    load_tile_A_async_strategic(A, &smem_A[0][0][0], M, K, block_row, 0, tid, true);
    load_tile_B_async_strategic(B_col_major, &smem_B[0][0][0], N, K, block_col, 0, tid, true);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        wmma::load_matrix_sync(a_frag[0][i],
                              &smem_A[0][m_offset + i * WMMA_M][0],
                              SMEM_STRIDE_A);
        wmma::load_matrix_sync(b_frag[0][i],
                              &smem_B[0][n_offset + i * WMMA_N][0],
                              SMEM_STRIDE_B);
    }
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int read_stage = k_tile % PIPELINE_STAGES;
        int write_stage = (k_tile + 1) % PIPELINE_STAGES;
        
        if (k_tile + 1 < num_k_tiles) {
            bool is_last = (k_tile + 2 >= num_k_tiles);
            load_tile_A_async_strategic(A, &smem_A[write_stage][0][0], M, K,
                                       block_row, (k_tile + 1) * BK, tid, !is_last);
            load_tile_B_async_strategic(B_col_major, &smem_B[write_stage][0][0], N, K,
                                       block_col, (k_tile + 1) * BK, tid, !is_last);
            asm volatile("cp.async.commit_group;\n");
        }
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(a_frag[1][i],
                                  &smem_A[read_stage][m_offset + i * WMMA_M][WMMA_K],
                                  SMEM_STRIDE_A);
            wmma::load_matrix_sync(b_frag[1][i],
                                  &smem_B[read_stage][n_offset + i * WMMA_N][WMMA_K],
                                  SMEM_STRIDE_B);
        }
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[0][i], b_frag[0][j], acc[i][j]);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(a_frag[0][i],
                                  &smem_A[read_stage][m_offset + i * WMMA_M][2 * WMMA_K],
                                  SMEM_STRIDE_A);
            wmma::load_matrix_sync(b_frag[0][i],
                                  &smem_B[read_stage][n_offset + i * WMMA_N][2 * WMMA_K],
                                  SMEM_STRIDE_B);
        }
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[1][i], b_frag[1][j], acc[i][j]);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(a_frag[1][i],
                                  &smem_A[read_stage][m_offset + i * WMMA_M][3 * WMMA_K],
                                  SMEM_STRIDE_A);
            wmma::load_matrix_sync(b_frag[1][i],
                                  &smem_B[read_stage][n_offset + i * WMMA_N][3 * WMMA_K],
                                  SMEM_STRIDE_B);
        }
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[0][i], b_frag[0][j], acc[i][j]);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[1][i], b_frag[1][j], acc[i][j]);
            }
        }
        
        if (k_tile + 1 < num_k_tiles) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(a_frag[0][i],
                                      &smem_A[write_stage][m_offset + i * WMMA_M][0],
                                      SMEM_STRIDE_A);
                wmma::load_matrix_sync(b_frag[0][i],
                                      &smem_B[write_stage][n_offset + i * WMMA_N][0],
                                      SMEM_STRIDE_B);
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int m_out = block_row * BM + m_offset + i * WMMA_M;
            int n_out = block_col * BN + n_offset + j * WMMA_N;
            
            if (m_out < M && n_out < N) {
                wmma::store_matrix_sync(&C[m_out * N + n_out],
                                       acc[i][j], N, wmma::mem_row_major);
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
    const half* B_col_major = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS_PER_BLOCK);
    
    hgemm_perfect_software_pipeline_kernel<<<grid, block>>>(A, B_col_major, C, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}