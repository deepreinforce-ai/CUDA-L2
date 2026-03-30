#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

using namespace nvcuda;

#define CP_ASYNC_CG_16(dst, src)                                               \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"                \
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"((uint64_t)(src)))
#define CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT(n)  asm volatile("cp.async.wait_group %0;\n" :: "n"(n) : "memory")

__global__ __launch_bounds__(128, 3)
void hgemm_bm64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int K  = 256;
    constexpr int N  = 64;
    constexpr int NUM_ITERS = K / BK;
    constexpr int SA_STRIDE = 40;
    constexpr int SB_STRIDE = 72;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int block_row = blockIdx.x * BM;
    const int warp_row  = block_row + warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.0f);

    __shared__ half smA[2][BM][SA_STRIDE];
    __shared__ half smB[2][BK][SB_STRIDE];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

    #pragma unroll
    for (int i = tid; i < BM * BK / 8; i += 128) {
        int row = i / 4; int col = (i % 4) * 8;
        CP_ASYNC_CG_16(&smA[0][row][col], &A[(block_row + row) * K + col]);
    }
    #pragma unroll
    for (int i = tid; i < BK * BN / 8; i += 128) {
        int row = i / 8; int col = (i % 8) * 8;
        CP_ASYNC_CG_16(&smB[0][row][col], &B[row * N + col]);
    }
    CP_ASYNC_COMMIT();
    CP_ASYNC_WAIT(0);
    __syncthreads();

    #pragma unroll 8
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        const int cur = iter & 1;
        const int nxt = 1 - cur;
        const int next_k = (iter + 1) * BK;

        if (iter < NUM_ITERS - 1) {
            #pragma unroll
            for (int i = tid; i < BM * BK / 8; i += 128) {
                int row = i / 4; int col = (i % 4) * 8;
                CP_ASYNC_CG_16(&smA[nxt][row][col], &A[(block_row + row) * K + next_k + col]);
            }
            #pragma unroll
            for (int i = tid; i < BK * BN / 8; i += 128) {
                int row = i / 8; int col = (i % 8) * 8;
                CP_ASYNC_CG_16(&smB[nxt][row][col], &B[(next_k + row) * N + col]);
            }
            CP_ASYNC_COMMIT();
        }

        #pragma unroll 2
        for (int ki = 0; ki < BK; ki += 16) {
            wmma::load_matrix_sync(a_frag, &smA[cur][warp_id * 16][ki], SA_STRIDE);
            #pragma unroll 4
            for (int n = 0; n < 4; n++) {
                wmma::load_matrix_sync(b_frag[n], &smB[cur][ki][n * 16], SB_STRIDE);
                wmma::mma_sync(acc[n], a_frag, b_frag[n], acc[n]);
            }
        }

        if (iter < NUM_ITERS - 1) {
            CP_ASYNC_WAIT(0);
            __syncthreads();
        }
    }

    __shared__ float smOut[4][16][16];
    #pragma unroll 4
    for (int n = 0; n < 4; n++) {
        wmma::store_matrix_sync(&smOut[warp_id][0][0], acc[n], 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int i = lane_id; i < 128; i += 32) {
            int r = i >> 3; int c = (i & 7) << 1;
            half2 out = __floats2half2_rn(smOut[warp_id][r][c], smOut[warp_id][r][c + 1]);
            *reinterpret_cast<half2*>(&C[(warp_row + r) * N + n * 16 + c]) = out;
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(256, 2)
void hgemm_bm128(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    constexpr int BM = 128;
    constexpr int BK = 32;
    constexpr int BN = 64;
    constexpr int SA_STRIDE = 40;
    constexpr int SB_STRIDE = 72;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int block_row = blockIdx.x * BM;
    const int warp_row  = block_row + warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.0f);

    __shared__ half smA[2][BM][SA_STRIDE];
    __shared__ half smB[2][BK][SB_STRIDE];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];

    #pragma unroll
    for (int i = tid; i < BM * BK / 8; i += 256) {
        int row = i / 4; int col = (i % 4) * 8;
        int gr = block_row + row;
        if (gr < M)
            CP_ASYNC_CG_16(&smA[0][row][col], &A[gr * K + col]);
    }
    #pragma unroll
    for (int i = tid; i < BK * BN / 8; i += 256) {
        int row = i / 8; int col = (i % 8) * 8;
        CP_ASYNC_CG_16(&smB[0][row][col], &B[row * N + col]);
    }
    CP_ASYNC_COMMIT();
    CP_ASYNC_WAIT(0);
    __syncthreads();

    for (int iter = 0; iter < K / BK; iter++) {
        const int cur = iter & 1;
        const int nxt = 1 - cur;
        const int next_k = (iter + 1) * BK;

        if (iter < K / BK - 1) {
            #pragma unroll
            for (int i = tid; i < BM * BK / 8; i += 256) {
                int row = i / 4; int col = (i % 4) * 8;
                int gr = block_row + row;
                if (gr < M)
                    CP_ASYNC_CG_16(&smA[nxt][row][col], &A[gr * K + next_k + col]);
            }
            #pragma unroll
            for (int i = tid; i < BK * BN / 8; i += 256) {
                int row = i / 8; int col = (i % 8) * 8;
                CP_ASYNC_CG_16(&smB[nxt][row][col], &B[(next_k + row) * N + col]);
            }
            CP_ASYNC_COMMIT();
        }

        #pragma unroll 2
        for (int ki = 0; ki < BK; ki += 16) {
            wmma::load_matrix_sync(a_frag, &smA[cur][warp_id * 16][ki], SA_STRIDE);
            #pragma unroll 4
            for (int n = 0; n < 4; n++) {
                wmma::load_matrix_sync(b_frag[n], &smB[cur][ki][n * 16], SB_STRIDE);
                wmma::mma_sync(acc[n], a_frag, b_frag[n], acc[n]);
            }
        }

        if (iter < K / BK - 1) {
            CP_ASYNC_WAIT(0);
            __syncthreads();
        }
    }

    __shared__ float smOut[8][16][16];
    #pragma unroll 4
    for (int n = 0; n < 4; n++) {
        wmma::store_matrix_sync(&smOut[warp_id][0][0], acc[n], 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int i = lane_id; i < 128; i += 32) {
            int r = i >> 3; int c = (i & 7) << 1;
            int gr = warp_row + r; int gc = n * 16 + c;
            if (gr < M && gc + 1 < N) {
                half2 out = __floats2half2_rn(smOut[warp_id][r][c], smOut[warp_id][r][c + 1]);
                *reinterpret_cast<half2*>(&C[gr * N + gc]) = out;
            }
        }
        __syncwarp();
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M     = a.size(0);
    const int K_dim = a.size(1);
    const int N_dim = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr());

    if (M == 8192 && N_dim == 64 && K_dim == 256) {
        hgemm_bm64<<<128, 128>>>(ptr_A, ptr_B, ptr_C);
    } else {
        int grid = (M + 127) / 128;
        hgemm_bm128<<<grid, 256>>>(ptr_A, ptr_B, ptr_C, M, N_dim, K_dim);
    }
}