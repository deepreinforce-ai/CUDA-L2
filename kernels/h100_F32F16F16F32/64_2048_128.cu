#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <cuda.h>

using namespace nvcuda;

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__global__ __launch_bounds__(128, 6)
void hgemm_v8_bn64_regpipe(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int N
) {
    __shared__ __align__(256) half sA[64][136];
    __shared__ __align__(256) half sB[64][128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int bn      = blockIdx.x;
    const int n_base  = bn * 64;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int linear = (tid + i * 128) * 8;
        int row    = linear >> 7;
        int col    = linear & 127;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sA[row][col])),
               "l"(A + row * 128 + col)
            : "memory");
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int linear = (tid + i * 128) * 8;
        int n_idx  = linear >> 7;
        int k_idx  = linear & 127;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sB[n_idx][k_idx])),
               "l"(B_col + (n_base + n_idx) * 128 + k_idx)
            : "memory");
    }
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    const int warp_m = warp_id * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];

    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::fill_fragment(acc[ni], 0.f);

    wmma::load_matrix_sync(a_frag[0], &sA[warp_m][0], 136);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::load_matrix_sync(b_frag[0][ni], &sB[ni * 16][0], 128);

    #pragma unroll
    for (int k = 0; k < 7; k++) {
        const int cur   = k & 1;
        const int nxt   = cur ^ 1;
        const int k_nxt = (k + 1) * 16;

        wmma::load_matrix_sync(a_frag[nxt], &sA[warp_m][k_nxt], 136);
        wmma::load_matrix_sync(b_frag[nxt][0], &sB[0][k_nxt], 128);
        wmma::mma_sync(acc[0], a_frag[cur], b_frag[cur][0], acc[0]);
        wmma::load_matrix_sync(b_frag[nxt][1], &sB[16][k_nxt], 128);
        wmma::mma_sync(acc[1], a_frag[cur], b_frag[cur][1], acc[1]);
        wmma::load_matrix_sync(b_frag[nxt][2], &sB[32][k_nxt], 128);
        wmma::mma_sync(acc[2], a_frag[cur], b_frag[cur][2], acc[2]);
        wmma::load_matrix_sync(b_frag[nxt][3], &sB[48][k_nxt], 128);
        wmma::mma_sync(acc[3], a_frag[cur], b_frag[cur][3], acc[3]);
    }
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::mma_sync(acc[ni], a_frag[1], b_frag[1][ni], acc[ni]);

    __syncthreads();
    float* warp_buf = reinterpret_cast<float*>(sB) + warp_id * 1024;

    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(warp_buf + ni * 256, acc[ni], 16, wmma::mem_row_major);

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        const float* buf  = warp_buf + ni * 256;
        const int out_col = n_base + ni * 16;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx = lane_id * 2 + e * 64;
            const int r   = idx >> 4;
            const int nc  = idx & 15;
            float2 fv = *reinterpret_cast<const float2*>(&buf[r * 16 + nc]);
            *reinterpret_cast<half2*>(&C[(warp_m + r) * N + out_col + nc]) = __float22half2_rn(fv);
        }
    }
}

__global__ __launch_bounds__(64, 12)
void hgemm_v8_bn32_highoccupancy(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int N
) {
    __shared__ __align__(256) half sA[64][136];
    __shared__ __align__(256) half sB[32][128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int bn      = blockIdx.x;
    const int n_base  = bn * 32;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int linear = (tid + i * 64) * 8;
        int row    = linear >> 7;
        int col    = linear & 127;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sA[row][col])),
               "l"(A + row * 128 + col)
            : "memory");
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int linear = (tid + i * 64) * 8;
        int n_idx  = linear >> 7;
        int k_idx  = linear & 127;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sB[n_idx][k_idx])),
               "l"(B_col + (n_base + n_idx) * 128 + k_idx)
            : "memory");
    }
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    const int warp_m = warp_id * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];

    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);

    wmma::load_matrix_sync(a_frag[0], &sA[warp_m][0], 136);
    wmma::load_matrix_sync(b_frag[0][0], &sB[0][0], 128);
    wmma::load_matrix_sync(b_frag[0][1], &sB[16][0], 128);

    #pragma unroll
    for (int k = 0; k < 7; k++) {
        const int cur   = k & 1;
        const int nxt   = cur ^ 1;
        const int k_nxt = (k + 1) * 16;

        wmma::load_matrix_sync(a_frag[nxt], &sA[warp_m][k_nxt], 136);
        wmma::load_matrix_sync(b_frag[nxt][0], &sB[0][k_nxt], 128);
        wmma::mma_sync(acc[0], a_frag[cur], b_frag[cur][0], acc[0]);
        wmma::load_matrix_sync(b_frag[nxt][1], &sB[16][k_nxt], 128);
        wmma::mma_sync(acc[1], a_frag[cur], b_frag[cur][1], acc[1]);
    }
    wmma::mma_sync(acc[0], a_frag[1], b_frag[1][0], acc[0]);
    wmma::mma_sync(acc[1], a_frag[1], b_frag[1][1], acc[1]);

    __syncthreads();
    float* warp_buf = reinterpret_cast<float*>(sA) + warp_id * 512;

    wmma::store_matrix_sync(warp_buf,       acc[0], 16, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_buf + 256, acc[1], 16, wmma::mem_row_major);

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        const float* buf  = warp_buf + ni * 256;
        const int out_col = n_base + ni * 16;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx = lane_id * 2 + e * 64;
            const int r   = idx >> 4;
            const int nc  = idx & 15;
            float2 fv = *reinterpret_cast<const float2*>(&buf[r * 16 + nc]);
            *reinterpret_cast<half2*>(&C[(warp_m + r) * N + out_col + nc]) = __float22half2_rn(fv);
        }
    }
}

__global__ __launch_bounds__(128, 1)
void hgemm_v8_bn128(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int N
) {
    __shared__ __align__(256) half sA[64][128];
    __shared__ __align__(256) half sB[128][128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int bn      = blockIdx.x;
    const int n_base  = bn * 128;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int linear = (tid + i * 128) * 8;
        int row    = linear >> 7;
        int col    = linear & 127;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sA[row][col])),
               "l"(A + row * 128 + col)
            : "memory");
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int linear = (tid + i * 128) * 8;
        int n_idx  = linear >> 7;
        int k_idx  = linear & 127;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sB[n_idx][k_idx])),
               "l"(B_col + (n_base + n_idx) * 128 + k_idx)
            : "memory");
    }
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    const int warp_m = warp_id * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2][8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];

    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        wmma::fill_fragment(acc[ni], 0.f);

    wmma::load_matrix_sync(a_frag[0], &sA[warp_m][0], 128);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        wmma::load_matrix_sync(b_frag[0][ni], &sB[ni * 16][0], 128);

    #pragma unroll
    for (int k = 0; k < 7; k++) {
        const int cur   = k & 1;
        const int nxt   = cur ^ 1;
        const int k_nxt = (k + 1) * 16;

        wmma::load_matrix_sync(a_frag[nxt], &sA[warp_m][k_nxt], 128);
        wmma::load_matrix_sync(b_frag[nxt][0], &sB[0][k_nxt], 128);
        wmma::mma_sync(acc[0], a_frag[cur], b_frag[cur][0], acc[0]);
        wmma::load_matrix_sync(b_frag[nxt][1], &sB[16][k_nxt], 128);
        wmma::mma_sync(acc[1], a_frag[cur], b_frag[cur][1], acc[1]);
        wmma::load_matrix_sync(b_frag[nxt][2], &sB[32][k_nxt], 128);
        wmma::mma_sync(acc[2], a_frag[cur], b_frag[cur][2], acc[2]);
        wmma::load_matrix_sync(b_frag[nxt][3], &sB[48][k_nxt], 128);
        wmma::mma_sync(acc[3], a_frag[cur], b_frag[cur][3], acc[3]);
        wmma::load_matrix_sync(b_frag[nxt][4], &sB[64][k_nxt], 128);
        wmma::mma_sync(acc[4], a_frag[cur], b_frag[cur][4], acc[4]);
        wmma::load_matrix_sync(b_frag[nxt][5], &sB[80][k_nxt], 128);
        wmma::mma_sync(acc[5], a_frag[cur], b_frag[cur][5], acc[5]);
        wmma::load_matrix_sync(b_frag[nxt][6], &sB[96][k_nxt], 128);
        wmma::mma_sync(acc[6], a_frag[cur], b_frag[cur][6], acc[6]);
        wmma::load_matrix_sync(b_frag[nxt][7], &sB[112][k_nxt], 128);
        wmma::mma_sync(acc[7], a_frag[cur], b_frag[cur][7], acc[7]);
    }
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        wmma::mma_sync(acc[ni], a_frag[1], b_frag[1][ni], acc[ni]);

    __syncthreads();
    float* warp_buf = reinterpret_cast<float*>(sA) + warp_id * 256;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        wmma::store_matrix_sync(warp_buf, acc[ni], 16, wmma::mem_row_major);
        __syncwarp();
        const int out_col = n_base + ni * 16;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx = lane_id * 2 + e * 64;
            const int r   = idx >> 4;
            const int nc  = idx & 15;
            float2 fv = *reinterpret_cast<const float2*>(&warp_buf[idx]);
            *reinterpret_cast<half2*>(&C[(warp_m + r) * N + out_col + nc]) = __float22half2_rn(fv);
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(128, 1)
void hgemm_v8_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int N
) {
    __shared__ __align__(256) half sA[64][128];
    __shared__ __align__(256) half sB[2][64][128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int linear = (tid + i * 128) * 8;
        int row    = linear >> 7;
        int col    = linear & 127;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sA[row][col])),
               "l"(A + row * 128 + col)
            : "memory");
    }
    asm volatile("cp.async.commit_group;\n" :::);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int linear = (tid + i * 128) * 8;
        int n_idx  = linear >> 7;
        int k_idx  = linear & 127;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32addr(&sB[0][n_idx][k_idx])),
               "l"(B_col + n_idx * 128 + k_idx)
            : "memory");
    }
    asm volatile("cp.async.commit_group;\n" :::);
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    const int warp_m  = warp_id * 16;
    const int n_tiles = N / 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];

    for (int tile = 0; tile < n_tiles; tile++) {
        const int cur_buf  = tile & 1;
        const int next_buf = cur_buf ^ 1;

        if (tile + 1 < n_tiles) {
            int next_n = (tile + 1) * 64;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int linear = (tid + i * 128) * 8;
                int n_idx  = linear >> 7;
                int k_idx  = linear & 127;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(smem_u32addr(&sB[next_buf][n_idx][k_idx])),
                       "l"(B_col + (next_n + n_idx) * 128 + k_idx)
                    : "memory");
            }
            asm volatile("cp.async.commit_group;\n" :::);
        }

        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(acc[ni], 0.f);

        wmma::load_matrix_sync(a_frag[0], &sA[warp_m][0], 128);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b_frag[0][ni], &sB[cur_buf][ni * 16][0], 128);

        #pragma unroll
        for (int k = 0; k < 7; k++) {
            const int cur   = k & 1;
            const int nxt   = cur ^ 1;
            const int k_nxt = (k + 1) * 16;

            wmma::load_matrix_sync(a_frag[nxt], &sA[warp_m][k_nxt], 128);
            wmma::load_matrix_sync(b_frag[nxt][0], &sB[cur_buf][0][k_nxt], 128);
            wmma::mma_sync(acc[0], a_frag[cur], b_frag[cur][0], acc[0]);
            wmma::load_matrix_sync(b_frag[nxt][1], &sB[cur_buf][16][k_nxt], 128);
            wmma::mma_sync(acc[1], a_frag[cur], b_frag[cur][1], acc[1]);
            wmma::load_matrix_sync(b_frag[nxt][2], &sB[cur_buf][32][k_nxt], 128);
            wmma::mma_sync(acc[2], a_frag[cur], b_frag[cur][2], acc[2]);
            wmma::load_matrix_sync(b_frag[nxt][3], &sB[cur_buf][48][k_nxt], 128);
            wmma::mma_sync(acc[3], a_frag[cur], b_frag[cur][3], acc[3]);
        }
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(acc[ni], a_frag[1], b_frag[1][ni], acc[ni]);

        __syncthreads();
        float* warp_buf = reinterpret_cast<float*>(sA) + warp_id * 1024;

        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::store_matrix_sync(warp_buf + ni * 256, acc[ni], 16, wmma::mem_row_major);

        const int n_base = tile * 64;
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const float* buf  = warp_buf + ni * 256;
            const int out_col = n_base + ni * 16;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int idx = lane_id * 2 + e * 64;
                const int r   = idx >> 4;
                const int nc  = idx & 15;
                float2 fv = *reinterpret_cast<const float2*>(&buf[r * 16 + nc]);
                *reinterpret_cast<half2*>(&C[(warp_m + r) * N + out_col + nc]) = __float22half2_rn(fv);
            }
        }

        if (tile + 1 < n_tiles) {
            asm volatile("cp.async.wait_all;\n" :::);
            __syncthreads();
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const int N = (int)b.size(1);

    const half* A     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       C     = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (N % 64 == 0) {
        hgemm_v8_bn64_regpipe<<<N / 64, 128>>>(A, B_col, C, N);
    } else if (N % 32 == 0) {
        hgemm_v8_bn32_highoccupancy<<<N / 32, 64>>>(A, B_col, C, N);
    } else if (N % 128 == 0) {
        hgemm_v8_bn128<<<N / 128, 128>>>(A, B_col, C, N);
    } else {
        hgemm_v8_bn64_regpipe<<<(N + 63) / 64, 128>>>(A, B_col, C, N);
    }
}