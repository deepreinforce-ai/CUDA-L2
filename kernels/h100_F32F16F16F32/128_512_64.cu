#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

#define BK 16
#define PAD 8
#define SS (BK + PAD)

__global__ __launch_bounds__(128, 8)
void hgemm_32x128_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half smemA[2][32][SS];
    __shared__ half smemB[2][128][SS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int block_m = blockIdx.y * 32;
    const int block_n = blockIdx.x * 128;
    const int warp_n_base = warp_id * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> wacc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(wacc[i][j], 0.0f);

    {
        if (tid < 64) {
            int row = tid >> 1;
            int col = (tid & 1) << 3;
            unsigned addr = __cvta_generic_to_shared(&smemA[0][row][col]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addr), "l"(A + (block_m + row) * K + col));
        }
        {
            int row = tid;
            unsigned addr0 = __cvta_generic_to_shared(&smemB[0][row][0]);
            unsigned addr8 = __cvta_generic_to_shared(&smemB[0][row][8]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addr0), "l"(B_col + (block_n + row) * K + 0));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addr8), "l"(B_col + (block_n + row) * K + 8));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);

    #pragma unroll 4
    for (int k_tile = 0; k_tile < 4; k_tile++) {
        const int s  = k_tile & 1;
        const int ns = s ^ 1;

        if (k_tile < 3) {
            int nk = (k_tile + 1) * BK;
            if (tid < 64) {
                int row = tid >> 1;
                int col = (tid & 1) << 3;
                unsigned addr = __cvta_generic_to_shared(&smemA[ns][row][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr), "l"(A + (block_m + row) * K + nk + col));
            }
            {
                int row = tid;
                unsigned addr0 = __cvta_generic_to_shared(&smemB[ns][row][0]);
                unsigned addr8 = __cvta_generic_to_shared(&smemB[ns][row][8]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr0), "l"(B_col + (block_n + row) * K + nk));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr8), "l"(B_col + (block_n + row) * K + nk + 8));
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA[2];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            wmma::load_matrix_sync(fA[i], &smemA[s][i * 16][0], SS);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB[2];
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::load_matrix_sync(fB[j], &smemB[s][warp_n_base + j * 16][0], SS);

        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::mma_sync(wacc[i][j], fA[i], fB[j], wacc[i][j]);

        if (k_tile < 3)
            __syncthreads();
    }

    float* warp_buf = reinterpret_cast<float*>(&smemB[0][0][0]) + warp_id * 256;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int m_off = block_m + i * 16;
            const int n_off = block_n + warp_n_base + j * 16;

            wmma::store_matrix_sync(warp_buf, wacc[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int t = lane; t < 128; t += 32) {
                const int r = t >> 3;
                const int c = (t & 7) << 1;
                const half2 v = __floats2half2_rn(warp_buf[r * 16 + c], warp_buf[r * 16 + c + 1]);
                *reinterpret_cast<half2*>(&C[(m_off + r) * N + n_off + c]) = v;
            }
            __syncwarp();
        }
    }
}

__global__ __launch_bounds__(256, 3)
void hgemm_128x128_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_n = blockIdx.x * 128;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_row = warp_id >> 2;
    const int warp_col = warp_id & 3;
    const int warp_m_base = warp_row * 64;
    const int warp_n_base = warp_col * 32;

    __shared__ half smemA[2][128][SS];
    __shared__ half smemB[2][128][SS];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> wacc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(wacc[i][j], 0.0f);

    {
        int row = tid >> 1;
        int col = (tid & 1) << 3;
        unsigned addrA = __cvta_generic_to_shared(&smemA[0][row][col]);
        unsigned addrB = __cvta_generic_to_shared(&smemB[0][row][col]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(addrA), "l"(A + row * K + col));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(addrB), "l"(B_col + (block_n + row) * K + col));
    }
    asm volatile("cp.async.commit_group;\n" ::);

    #pragma unroll 4
    for (int k_tile = 0; k_tile < 4; k_tile++) {
        const int s  = k_tile & 1;
        const int ns = s ^ 1;

        if (k_tile < 3) {
            int nk = (k_tile + 1) * BK;
            int row = tid >> 1;
            int col = (tid & 1) << 3;
            unsigned addrA = __cvta_generic_to_shared(&smemA[ns][row][col]);
            unsigned addrB = __cvta_generic_to_shared(&smemB[ns][row][col]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addrA), "l"(A + row * K + nk + col));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addrB), "l"(B_col + (block_n + row) * K + nk + col));
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            wmma::load_matrix_sync(fA[i], &smemA[s][warp_m_base + i * 16][0], SS);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB[2];
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::load_matrix_sync(fB[j], &smemB[s][warp_n_base + j * 16][0], SS);

        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::mma_sync(wacc[i][j], fA[i], fB[j], wacc[i][j]);

        if (k_tile < 3)
            __syncthreads();
    }

    float* warp_buf = reinterpret_cast<float*>(&smemA[0][0][0]) + warp_id * 256;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int m_off = warp_m_base + i * 16;
            int n_off = block_n + warp_n_base + j * 16;

            wmma::store_matrix_sync(warp_buf, wacc[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int t = lane; t < 128; t += 32) {
                int r = t >> 3;
                int c = (t & 7) << 1;
                half2 v = __floats2half2_rn(warp_buf[r * 16 + c], warp_buf[r * 16 + c + 1]);
                *reinterpret_cast<half2*>(&C[(m_off + r) * N + n_off + c]) = v;
            }
            __syncwarp();
        }
    }
}

__global__ __launch_bounds__(128, 8)
void hgemm_64x64_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half smemA[2][64][SS];
    __shared__ half smemB[2][64][SS];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int block_m = blockIdx.y * 64;
    const int block_n = blockIdx.x * 64;

    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m_base = warp_row * 32;
    const int warp_n_base = warp_col * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> wacc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(wacc[i][j], 0.0f);

    {
        int row = tid >> 1;
        int ks = (tid & 1) << 3;
        unsigned addrA = __cvta_generic_to_shared(&smemA[0][row][ks]);
        unsigned addrB = __cvta_generic_to_shared(&smemB[0][row][ks]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(addrA), "l"(A + (block_m + row) * K + ks));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(addrB), "l"(B_col + (block_n + row) * K + ks));
    }
    asm volatile("cp.async.commit_group;\n" ::);

    #pragma unroll 4
    for (int k_tile = 0; k_tile < 4; k_tile++) {
        const int s = k_tile & 1;
        const int ns = 1 - s;

        if (k_tile < 3) {
            int row = tid >> 1;
            int ks = (tid & 1) << 3;
            int nk = (k_tile + 1) * BK;
            unsigned addrA = __cvta_generic_to_shared(&smemA[ns][row][ks]);
            unsigned addrB = __cvta_generic_to_shared(&smemB[ns][row][ks]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addrA), "l"(A + (block_m + row) * K + nk + ks));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addrB), "l"(B_col + (block_n + row) * K + nk + ks));
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA[2];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            wmma::load_matrix_sync(fA[i], &smemA[s][warp_m_base + i * 16][0], SS);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fB[2];
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::load_matrix_sync(fB[j], &smemB[s][warp_n_base + j * 16][0], SS);

        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::mma_sync(wacc[i][j], fA[i], fB[j], wacc[i][j]);

        if (k_tile < 3)
            __syncthreads();
    }

    float* warp_buf = reinterpret_cast<float*>(&smemA[0][0][0]) + warp_id * 256;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int m_off = block_m + warp_m_base + i * 16;
            const int n_off = block_n + warp_n_base + j * 16;

            wmma::store_matrix_sync(warp_buf, wacc[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int t = lane; t < 128; t += 32) {
                const int r = t >> 3;
                const int c = (t & 7) << 1;
                const half2 v = __floats2half2_rn(warp_buf[r * 16 + c], warp_buf[r * 16 + c + 1]);
                *reinterpret_cast<half2*>(&C[(m_off + r) * N + n_off + c]) = v;
            }
            __syncwarp();
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A_ptr     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*        C_ptr    = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        dim3 grid(N / 128, M / 32);
        dim3 block(128);
        hgemm_32x128_wmma<<<grid, block>>>(A_ptr, B_col_ptr, C_ptr, M, N, K);
    }
}