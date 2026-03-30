#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>

using namespace nvcuda;

#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16
#define BK      128
#define BN      128
#define B_STRIDE 136
#define K_TILES 8
#define N_TILES 8

__global__ __launch_bounds__(128, 5)
void hgemm_optimized_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half*       __restrict__ C,
    int M)
{
    const int N  = 128;
    const int K  = 128;
    const int BM = 64;
    const int NWARPS = 4;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int global_row = blockIdx.x * BM + warp_id * WMMA_M;

    __shared__ __align__(128) __half smem_B[BN][B_STRIDE];
    __shared__ __align__(16) float smem_acc[NWARPS][WMMA_M * WMMA_N];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int f4    = tid + i * 128;
        int n_idx = f4 >> 4;
        int k_idx = (f4 & 15) << 3;
        *reinterpret_cast<float4*>(&smem_B[n_idx][k_idx]) =
            *reinterpret_cast<const float4*>(&B_col[n_idx * K + k_idx]);
    }
    __syncthreads();

    if (global_row >= M) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_TILES];
    #pragma unroll
    for (int ni = 0; ni < N_TILES; ni++)
        wmma::fill_fragment(acc[ni], 0.0f);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[N_TILES];

    const __half* A_warp = A + global_row * K;

    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        wmma::load_matrix_sync(a_frag, A_warp + ki * WMMA_K, K);
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
            wmma::load_matrix_sync(b_frag[ni], &smem_B[ni * WMMA_N][ki * WMMA_K], B_STRIDE);
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
            wmma::mma_sync(acc[ni], a_frag, b_frag[ni], acc[ni]);
    }

    float* warp_stage = smem_acc[warp_id];
    __half* C_warp = C + global_row * N;

    #pragma unroll
    for (int ni = 0; ni < N_TILES; ni++) {
        wmma::store_matrix_sync(warp_stage, acc[ni], WMMA_N, wmma::mem_row_major);
        __syncwarp();
        int col_base = ni * WMMA_N;
        #pragma unroll
        for (int t = lane_id; t < WMMA_M * WMMA_N; t += 32) {
            int r = t >> 4;
            int c = t & 15;
            C_warp[r * N + col_base + c] = __float2half(warp_stage[t]);
        }
    }
}

__global__ __launch_bounds__(256, 3)
void hgemm_optimized_bm128_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half*       __restrict__ C,
    int M)
{
    const int N  = 128;
    const int K  = 128;
    const int BM = 128;
    const int NWARPS = 8;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int global_row = blockIdx.x * BM + warp_id * WMMA_M;

    __shared__ __align__(128) __half smem_B[BN][B_STRIDE];
    __shared__ __align__(16) float smem_acc[NWARPS][WMMA_M * WMMA_N];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int f4    = tid + i * 256;
        int n_idx = f4 >> 4;
        int k_idx = (f4 & 15) << 3;
        *reinterpret_cast<float4*>(&smem_B[n_idx][k_idx]) =
            *reinterpret_cast<const float4*>(&B_col[n_idx * K + k_idx]);
    }
    __syncthreads();

    if (global_row >= M) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_TILES];
    #pragma unroll
    for (int ni = 0; ni < N_TILES; ni++)
        wmma::fill_fragment(acc[ni], 0.0f);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[N_TILES];

    const __half* A_warp = A + global_row * K;

    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        wmma::load_matrix_sync(a_frag, A_warp + ki * WMMA_K, K);
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
            wmma::load_matrix_sync(b_frag[ni], &smem_B[ni * WMMA_N][ki * WMMA_K], B_STRIDE);
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
            wmma::mma_sync(acc[ni], a_frag, b_frag[ni], acc[ni]);
    }

    float* warp_stage = smem_acc[warp_id];
    __half* C_warp = C + global_row * N;

    #pragma unroll
    for (int ni = 0; ni < N_TILES; ni++) {
        wmma::store_matrix_sync(warp_stage, acc[ni], WMMA_N, wmma::mem_row_major);
        __syncwarp();
        int col_base = ni * WMMA_N;
        #pragma unroll
        for (int t = lane_id; t < WMMA_M * WMMA_N; t += 32) {
            int r = t >> 4;
            int c = t & 15;
            C_warp[r * N + col_base + c] = __float2half(warp_stage[t]);
        }
    }
}

__global__ __launch_bounds__(128, 5)
void hgemm_optimized_persistent_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half*       __restrict__ C,
    int M, int num_tiles)
{
    const int N  = 128;
    const int K  = 128;
    const int BM = 64;
    const int NWARPS = 4;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    __shared__ __align__(128) __half smem_B[BN][B_STRIDE];
    __shared__ __align__(16) float smem_acc[NWARPS][WMMA_M * WMMA_N];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int f4    = tid + i * 128;
        int n_idx = f4 >> 4;
        int k_idx = (f4 & 15) << 3;
        *reinterpret_cast<float4*>(&smem_B[n_idx][k_idx]) =
            *reinterpret_cast<const float4*>(&B_col[n_idx * K + k_idx]);
    }
    __syncthreads();

    float* warp_stage = smem_acc[warp_id];

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        int global_row = tile * BM + warp_id * WMMA_M;
        if (global_row >= M) continue;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[N_TILES];
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++)
            wmma::fill_fragment(acc[ni], 0.0f);

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[N_TILES];

        const __half* A_warp = A + global_row * K;

        #pragma unroll
        for (int ki = 0; ki < K_TILES; ki++) {
            wmma::load_matrix_sync(a_frag, A_warp + ki * WMMA_K, K);
            #pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[ni * WMMA_N][ki * WMMA_K], B_STRIDE);
            #pragma unroll
            for (int ni = 0; ni < N_TILES; ni++)
                wmma::mma_sync(acc[ni], a_frag, b_frag[ni], acc[ni]);
        }

        __half* C_warp = C + global_row * N;

        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++) {
            wmma::store_matrix_sync(warp_stage, acc[ni], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            int col_base = ni * WMMA_N;
            #pragma unroll
            for (int t = lane_id; t < WMMA_M * WMMA_N; t += 32) {
                int r = t >> 4;
                int c = t & 15;
                C_warp[r * N + col_base + c] = __float2half(warp_stage[t]);
            }
        }
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const __half* A     = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B_col = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half*       C_out = reinterpret_cast<__half*>(c.data_ptr());

    if (K == 128 && N == 128) {
        const int BM_MAIN = 64;
        int num_tiles = (M + BM_MAIN - 1) / BM_MAIN;

        if (M % BM_MAIN == 0) {
            dim3 grid(num_tiles);
            dim3 block(128);
            hgemm_optimized_kernel<<<grid, block>>>(A, B_col, C_out, M);
        } else {
            int grid_size = min(num_tiles, 132 * 5);
            dim3 grid(grid_size);
            dim3 block(128);
            hgemm_optimized_persistent_kernel<<<grid, block>>>(A, B_col, C_out, M, num_tiles);
        }
    } else {
        int num_tiles = (M + 127) / 128;
        dim3 grid(num_tiles);
        dim3 block(256);
        hgemm_optimized_bm128_kernel<<<grid, block>>>(A, B_col, C_out, M);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in cuda_l2_h100_fp32: %s\n", cudaGetErrorString(err));
    }
}