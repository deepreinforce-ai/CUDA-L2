#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <stdio.h>

using namespace nvcuda;

#define TILE_N 64
#define TILE_K 32
#define STAGES 3

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__global__ void __launch_bounds__(128, 8)
hgemm_regdb_v1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int K, int N)
{
    const int n_start = blockIdx.x * TILE_N;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warp_m  = warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        wmma::fill_fragment(acc[i], 0.0f);

    __shared__ __align__(128) half smA[STAGES][64][40];
    __shared__ __align__(128) half smB[STAGES][TILE_K][72];

    const int num_tiles = K / TILE_K;
    const int t = threadIdx.x;

    const int a_r0 = t >> 2,    a_c = (t & 3) << 3;
    const int b_r0 = t >> 3,    b_c = (t & 7) << 3;

    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smA[0][a_r0][a_c])),
        "l"((const void*)&A[a_r0 * K + a_c]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smA[0][a_r0+32][a_c])),
        "l"((const void*)&A[(a_r0+32) * K + a_c]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smB[0][b_r0][b_c])),
        "l"((const void*)&B[b_r0 * N + n_start + b_c]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smB[0][b_r0+16][b_c])),
        "l"((const void*)&B[(b_r0+16) * N + n_start + b_c]));
    asm volatile("cp.async.commit_group;\n");

    if (num_tiles > 1) {
        const int k1 = TILE_K;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smA[1][a_r0][a_c])),
            "l"((const void*)&A[a_r0 * K + k1 + a_c]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smA[1][a_r0+32][a_c])),
            "l"((const void*)&A[(a_r0+32) * K + k1 + a_c]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smB[1][b_r0][b_c])),
            "l"((const void*)&B[(k1+b_r0) * N + n_start + b_c]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smB[1][b_r0+16][b_c])),
            "l"((const void*)&B[(k1+b_r0+16) * N + n_start + b_c]));
        asm volatile("cp.async.commit_group;\n");
    }

    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();

    #pragma unroll 1
    for (int tile = 0; tile < num_tiles; tile++) {
        const int cur = tile % STAGES;
        const int nxt = (tile + 2) % STAGES;

        if (tile + 2 < num_tiles) {
            const int k_off = (tile + 2) * TILE_K;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smA[nxt][a_r0][a_c])),
                "l"((const void*)&A[a_r0 * K + k_off + a_c]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smA[nxt][a_r0+32][a_c])),
                "l"((const void*)&A[(a_r0+32) * K + k_off + a_c]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smB[nxt][b_r0][b_c])),
                "l"((const void*)&B[(k_off+b_r0) * N + n_start + b_c]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smB[nxt][b_r0+16][b_c])),
                "l"((const void*)&B[(k_off+b_r0+16) * N + n_start + b_c]));
            asm volatile("cp.async.commit_group;\n");
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag0, a_frag1;
        wmma::load_matrix_sync(a_frag0, &smA[cur][warp_m][0],  40);
        wmma::load_matrix_sync(a_frag1, &smA[cur][warp_m][16], 40);

        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, &smB[cur][0][ni * 16], 72);
            wmma::mma_sync(acc[ni], a_frag0, b_frag, acc[ni]);
        }

        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, &smB[cur][16][ni * 16], 72);
            wmma::mma_sync(acc[ni], a_frag1, b_frag, acc[ni]);
        }

        if (tile + 1 < num_tiles) {
            asm volatile("cp.async.wait_group 1;\n");
            __syncthreads();
        }
    }

    __syncthreads();

    float* warp_tmp = (warp_id < 2)
        ? (reinterpret_cast<float*>(smA) + warp_id * 1088)
        : (reinterpret_cast<float*>(smB) + (warp_id - 2) * 1088);

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        wmma::store_matrix_sync(warp_tmp + ni * 16, acc[ni], 68, wmma::mem_row_major);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int pair_idx = i * 32 + lane_id;
        int row = pair_idx >> 5;
        int col = (pair_idx & 31) << 1;
        half2 h2 = __floats2half2_rn(warp_tmp[row * 68 + col], warp_tmp[row * 68 + col + 1]);
        *reinterpret_cast<half2*>(&C[(warp_m + row) * N + n_start + col]) = h2;
    }
}

__global__ void __launch_bounds__(128, 6)
hgemm_wmma_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int K, int N)
{
    const int n_start = blockIdx.x * TILE_N;
    const int warp_id = threadIdx.x >> 5;
    const int warp_m  = warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        wmma::fill_fragment(acc[i], 0.0f);

    __shared__ __align__(128) half smA[STAGES][64][40];
    __shared__ __align__(128) half smB[STAGES][TILE_K][72];
    __shared__ __align__(128) float smC[64][68];

    const int num_tiles = K / TILE_K;
    const int t = threadIdx.x;
    const int a_r0 = t >> 2,  a_c = (t & 3) << 3;
    const int b_r0 = t >> 3,  b_c = (t & 7) << 3;

    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smA[0][a_r0][a_c])),
        "l"((const void*)&A[a_r0 * K + a_c]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smA[0][a_r0+32][a_c])),
        "l"((const void*)&A[(a_r0+32) * K + a_c]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smB[0][b_r0][b_c])),
        "l"((const void*)&B[b_r0 * N + n_start + b_c]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(smem_u32addr(&smB[0][b_r0+16][b_c])),
        "l"((const void*)&B[(b_r0+16) * N + n_start + b_c]));
    asm volatile("cp.async.commit_group;\n");

    if (num_tiles > 1) {
        int k1 = TILE_K;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smA[1][a_r0][a_c])),
            "l"((const void*)&A[a_r0 * K + k1 + a_c]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smA[1][a_r0+32][a_c])),
            "l"((const void*)&A[(a_r0+32) * K + k1 + a_c]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smB[1][b_r0][b_c])),
            "l"((const void*)&B[(k1+b_r0) * N + n_start + b_c]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32addr(&smB[1][b_r0+16][b_c])),
            "l"((const void*)&B[(k1+b_r0+16) * N + n_start + b_c]));
        asm volatile("cp.async.commit_group;\n");
    }

    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();

    #pragma unroll 1
    for (int tile = 0; tile < num_tiles; tile++) {
        int cur = tile % STAGES;
        int nxt = (tile + 2) % STAGES;

        if (tile + 2 < num_tiles) {
            int k_off = (tile + 2) * TILE_K;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smA[nxt][a_r0][a_c])),
                "l"((const void*)&A[a_r0 * K + k_off + a_c]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smA[nxt][a_r0+32][a_c])),
                "l"((const void*)&A[(a_r0+32) * K + k_off + a_c]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smB[nxt][b_r0][b_c])),
                "l"((const void*)&B[(k_off+b_r0) * N + n_start + b_c]));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32addr(&smB[nxt][b_r0+16][b_c])),
                "l"((const void*)&B[(k_off+b_r0+16) * N + n_start + b_c]));
            asm volatile("cp.async.commit_group;\n");
        }

        #pragma unroll
        for (int ki = 0; ki < 2; ki++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smA[cur][warp_m][ki * 16], 40);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smB[cur][ki * 16][ni * 16], 72);
                wmma::mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
            }
        }

        if (tile + 1 < num_tiles) {
            asm volatile("cp.async.wait_group 1;\n");
            __syncthreads();
        }
    }

    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(&smC[warp_m][ni * 16], acc[ni], 68, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int linear = (i * 128 + t) * 2;
        int row = linear >> 6;
        int col = linear & 63;
        half2 h2 = __floats2half2_rn(smC[row][col], smC[row][col + 1]);
        *reinterpret_cast<half2*>(&C[row * N + n_start + col]) = h2;
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        dim3 grid(N / TILE_N, 1, 1);
        dim3 block(128, 1, 1);
        hgemm_regdb_v1<<<grid, block>>>(ptr_A, ptr_B, ptr_C, K, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        dim3 grid2(N / TILE_N, 1, 1);
        dim3 block2(128, 1, 1);
        hgemm_wmma_fallback<<<grid2, block2>>>(ptr_A, ptr_B, ptr_C, K, N);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel launch error: ") + cudaGetErrorString(err));
        }
    }
}