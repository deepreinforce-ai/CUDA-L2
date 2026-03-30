#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda::wmma;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

__global__ void __launch_bounds__(256, 3)
hgemm_64x64_8warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 64, BN = 64, BK = 64;
    const int APAD = 8, BPAD = 8;

    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    int warp_m = wid >> 1;
    int warp_n = wid & 1;

    __shared__ half smA[BM][BK + APAD];
    __shared__ half smB[BK][BN + BPAD];

    int gm_base = bm * BM;
    int gn_base = bn * BN;

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int rr = row + r * 32;
            int gr = gm_base + rr;
            if (rr < BM && gr < M) {
                uint32_t sp = __cvta_generic_to_shared(&smA[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(A + gr * K + col));
            }
        }
    }

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int rr = row + r * 32;
            int gc = gn_base + col;
            if (rr < K && gc < N) {
                uint32_t sp = __cvta_generic_to_shared(&smB[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(B + rr * N + gc));
            }
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    int wm_off = warp_m * 16;
    int wn_off = warp_n * 32;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fa[4];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> fb0[4], fb1[4];

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fa[ki],  &smA[wm_off][ki * 16],       BK + APAD);
        load_matrix_sync(fb0[ki], &smB[ki * 16][wn_off],       BN + BPAD);
        load_matrix_sync(fb1[ki], &smB[ki * 16][wn_off + 16],  BN + BPAD);
    }

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1;
    fill_fragment(acc0, 0.0f);
    fill_fragment(acc1, 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        mma_sync(acc0, fa[ki], fb0[ki], acc0);
        mma_sync(acc1, fa[ki], fb1[ki], acc1);
    }

    __syncthreads();
    float (*smCf)[16][16] = reinterpret_cast<float(*)[16][16]>(smA);

    int gm_out = gm_base + wm_off;
    int gn_out0 = gn_base + wn_off;
    int gn_out1 = gn_out0 + 16;

    store_matrix_sync(&smCf[wid][0][0], acc0, 16, mem_row_major);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane + i * 32;
        int r = idx >> 3;
        int c2 = idx & 7;
        int c = c2 << 1;
        int gr = gm_out + r;
        int gc = gn_out0 + c;
        if (gr < M && gc + 1 < N) {
            __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
            *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
        } else if (gr < M && gc < N) {
            C[gr * N + gc] = __float2half(smCf[wid][r][c]);
            if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
        }
    }

    store_matrix_sync(&smCf[wid][0][0], acc1, 16, mem_row_major);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane + i * 32;
        int r = idx >> 3;
        int c2 = idx & 7;
        int c = c2 << 1;
        int gr = gm_out + r;
        int gc = gn_out1 + c;
        if (gr < M && gc + 1 < N) {
            __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
            *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
        } else if (gr < M && gc < N) {
            C[gr * N + gc] = __float2half(smCf[wid][r][c]);
            if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
        }
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_b_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 32, BN = 64, BK = 64;
    const int APAD = 8, BPAD = 8;

    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int warp_m = wid >> 1;
    int warp_n = wid & 1;

    __shared__ half smA[BM][BK + APAD];
    __shared__ half smB[BK][BN + BPAD];
    __shared__ float smCf[4][16][32];

    int gn_base = bn * BN;
    int wn_off  = warp_n * 32;

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int rr = row + r * 16;
            int gc = gn_base + col;
            if (rr < K && gc < N) {
                uint32_t sp = __cvta_generic_to_shared(&smB[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(B + rr * N + gc));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> fb0[4], fb1[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fb0[ki], &smB[ki * 16][wn_off],       BN + BPAD);
        load_matrix_sync(fb1[ki], &smB[ki * 16][wn_off + 16],  BN + BPAD);
    }

    int tiles_m = (M + BM - 1) / BM;
    for (int tm = 0; tm < tiles_m; tm++) {
        int gm_base = tm * BM;
        int wm_off  = warp_m * 16;

        {
            int row = tid >> 3;
            int col = (tid & 7) << 3;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                int rr = row + r * 16;
                int gr = gm_base + rr;
                if (rr < BM && gr < M) {
                    uint32_t sp = __cvta_generic_to_shared(&smA[rr][col]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :: "r"(sp), "l"(A + gr * K + col));
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fa[4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            load_matrix_sync(fa[ki], &smA[wm_off][ki * 16], BK + APAD);
        }

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1;
        fill_fragment(acc0, 0.0f);
        fill_fragment(acc1, 0.0f);
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            mma_sync(acc0, fa[ki], fb0[ki], acc0);
            mma_sync(acc1, fa[ki], fb1[ki], acc1);
        }

        store_matrix_sync(&smCf[wid][0][0],  acc0, 32, mem_row_major);
        store_matrix_sync(&smCf[wid][0][16], acc1, 32, mem_row_major);

        int gm = gm_base + wm_off;
        int gn = gn_base + wn_off;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx2 = lane + i * 32;
            int r  = idx2 >> 4;
            int c2 = idx2 & 15;
            int c  = c2 << 1;
            int gr = gm + r;
            int gc = gn + c;
            if (gr < M && gc + 1 < N) {
                __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
                *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = __float2half(smCf[wid][r][c]);
                if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
            }
        }

        if (tm + 1 < tiles_m) {
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_db_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 32, BN = 64, BK = 64;
    const int APAD = 8, BPAD = 8;

    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int warp_m = wid >> 1;
    int warp_n = wid & 1;

    __shared__ half smA[2][BM][BK + APAD];
    __shared__ half smB[BK][BN + BPAD];
    __shared__ float smCf[4][16][32];

    int gn_base = bn * BN;
    int wn_off  = warp_n * 32;
    int tiles_m = (M + BM - 1) / BM;

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int rr = row + r * 16;
            int gc = gn_base + col;
            if (rr < K && gc < N) {
                uint32_t sp = __cvta_generic_to_shared(&smB[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(B + rr * N + gc));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> fb0[4], fb1[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fb0[ki], &smB[ki * 16][wn_off],       BN + BPAD);
        load_matrix_sync(fb1[ki], &smB[ki * 16][wn_off + 16],  BN + BPAD);
    }

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int rr = row + r * 16;
            int gr = rr;
            if (rr < BM && gr < M) {
                uint32_t sp = __cvta_generic_to_shared(&smA[0][rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(A + gr * K + col));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);

    int buf = 0;
    for (int tm = 0; tm < tiles_m; tm++) {
        int nbuf = buf ^ 1;
        int next_tm = tm + 1;

        if (next_tm < tiles_m) {
            int gm_next = next_tm * BM;
            int row = tid >> 3;
            int col = (tid & 7) << 3;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                int rr = row + r * 16;
                int gr = gm_next + rr;
                if (rr < BM && gr < M) {
                    uint32_t sp = __cvta_generic_to_shared(&smA[nbuf][rr][col]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :: "r"(sp), "l"(A + gr * K + col));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        int gm_base = tm * BM;
        int wm_off  = warp_m * 16;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fa[4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            load_matrix_sync(fa[ki], &smA[buf][wm_off][ki * 16], BK + APAD);
        }

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1;
        fill_fragment(acc0, 0.0f);
        fill_fragment(acc1, 0.0f);
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            mma_sync(acc0, fa[ki], fb0[ki], acc0);
            mma_sync(acc1, fa[ki], fb1[ki], acc1);
        }

        store_matrix_sync(&smCf[wid][0][0],  acc0, 32, mem_row_major);
        store_matrix_sync(&smCf[wid][0][16], acc1, 32, mem_row_major);

        int gm = gm_base + wm_off;
        int gn = gn_base + wn_off;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx2 = lane + i * 32;
            int r  = idx2 >> 4;
            int c2 = idx2 & 15;
            int c  = c2 << 1;
            int gr = gm + r;
            int gc = gn + c;
            if (gr < M && gc + 1 < N) {
                __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
                *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = __float2half(smCf[wid][r][c]);
                if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
            }
        }

        buf ^= 1;
    }

    asm volatile("cp.async.wait_all;\n" ::);
}

__global__ void __launch_bounds__(128, 4)
hgemm_b_persistent_split(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    int m_groups
) {
    const int BM = 32, BN = 64, BK = 64;
    const int APAD = 8, BPAD = 8;

    int bn = blockIdx.x;
    int bg = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int warp_m = wid >> 1;
    int warp_n = wid & 1;

    __shared__ half smA[BM][BK + APAD];
    __shared__ half smB[BK][BN + BPAD];
    __shared__ float smCf[4][16][32];

    int gn_base = bn * BN;
    int wn_off  = warp_n * 32;

    int tiles_m = (M + BM - 1) / BM;
    int tiles_per_group = (tiles_m + m_groups - 1) / m_groups;
    int tm_start = bg * tiles_per_group;
    int tm_end   = min(tm_start + tiles_per_group, tiles_m);

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int rr = row + r * 16;
            int gc = gn_base + col;
            if (rr < K && gc < N) {
                uint32_t sp = __cvta_generic_to_shared(&smB[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(B + rr * N + gc));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> fb0[4], fb1[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fb0[ki], &smB[ki * 16][wn_off],       BN + BPAD);
        load_matrix_sync(fb1[ki], &smB[ki * 16][wn_off + 16],  BN + BPAD);
    }

    for (int tm = tm_start; tm < tm_end; tm++) {
        int gm_base = tm * BM;
        int wm_off  = warp_m * 16;

        {
            int row = tid >> 3;
            int col = (tid & 7) << 3;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                int rr = row + r * 16;
                int gr = gm_base + rr;
                if (rr < BM && gr < M) {
                    uint32_t sp = __cvta_generic_to_shared(&smA[rr][col]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :: "r"(sp), "l"(A + gr * K + col));
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fa[4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            load_matrix_sync(fa[ki], &smA[wm_off][ki * 16], BK + APAD);
        }

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1;
        fill_fragment(acc0, 0.0f);
        fill_fragment(acc1, 0.0f);
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            mma_sync(acc0, fa[ki], fb0[ki], acc0);
            mma_sync(acc1, fa[ki], fb1[ki], acc1);
        }

        store_matrix_sync(&smCf[wid][0][0],  acc0, 32, mem_row_major);
        store_matrix_sync(&smCf[wid][0][16], acc1, 32, mem_row_major);

        int gm = gm_base + wm_off;
        int gn = gn_base + wn_off;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx2 = lane + i * 32;
            int r  = idx2 >> 4;
            int c2 = idx2 & 15;
            int c  = c2 << 1;
            int gr = gm + r;
            int gc = gn + c;
            if (gr < M && gc + 1 < N) {
                __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
                *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = __float2half(smCf[wid][r][c]);
                if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
            }
        }

        if (tm + 1 < tm_end) {
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_32x64_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 32, BN = 64, BK = 64;
    const int APAD = 8, BPAD = 8;

    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int warp_m = wid >> 1;
    int warp_n = wid & 1;

    __shared__ half smA[BM][BK + APAD];
    __shared__ half smB[BK][BN + BPAD];
    __shared__ float smCf[4][16][32];

    int gm_base = bm * BM;
    int gn_base = bn * BN;

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int rr = row + r * 16;
            int gr = gm_base + rr;
            if (rr < BM && gr < M) {
                uint32_t sp = __cvta_generic_to_shared(&smA[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(A + gr * K + col));
            }
        }
    }
    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int rr = row + r * 16;
            int gc = gn_base + col;
            if (rr < K && gc < N) {
                uint32_t sp = __cvta_generic_to_shared(&smB[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(B + rr * N + gc));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    int wm_off = warp_m * 16;
    int wn_off = warp_n * 32;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fa[4];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> fb0[4], fb1[4];

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fa[ki],  &smA[wm_off][ki * 16],      BK + APAD);
        load_matrix_sync(fb0[ki], &smB[ki * 16][wn_off],      BN + BPAD);
        load_matrix_sync(fb1[ki], &smB[ki * 16][wn_off + 16], BN + BPAD);
    }

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1;
    fill_fragment(acc0, 0.0f);
    fill_fragment(acc1, 0.0f);

    mma_sync(acc0, fa[0], fb0[0], acc0);
    mma_sync(acc1, fa[0], fb1[0], acc1);
    mma_sync(acc0, fa[1], fb0[1], acc0);
    mma_sync(acc1, fa[1], fb1[1], acc1);
    mma_sync(acc0, fa[2], fb0[2], acc0);
    mma_sync(acc1, fa[2], fb1[2], acc1);
    mma_sync(acc0, fa[3], fb0[3], acc0);
    mma_sync(acc1, fa[3], fb1[3], acc1);

    store_matrix_sync(&smCf[wid][0][0],  acc0, 32, mem_row_major);
    store_matrix_sync(&smCf[wid][0][16], acc1, 32, mem_row_major);

    int gm = gm_base + wm_off;
    int gn = gn_base + wn_off;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx2 = lane + i * 32;
        int r  = idx2 >> 4;
        int c2 = idx2 & 15;
        int c  = c2 << 1;
        int gr = gm + r;
        int gc = gn + c;
        if (gr < M && gc + 1 < N) {
            __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
            *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
        } else if (gr < M && gc < N) {
            C[gr * N + gc] = __float2half(smCf[wid][r][c]);
            if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
        }
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_optimized_64x64_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 64, BN = 64, BK = 64;
    const int APAD = 8, BPAD = 8;

    int bn = blockIdx.x;
    int bg = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int warp_m = wid >> 1;
    int warp_n = wid & 1;

    __shared__ half smA[BM][BK + APAD];
    __shared__ half smB[BK][BN + BPAD];

    int gn_base = bn * BN;
    int wn_off  = warp_n * 32;
    int tiles_m = (M + BM - 1) / BM;
    int m_groups = gridDim.y;
    int tiles_per_group = (tiles_m + m_groups - 1) / m_groups;
    int tm_start = bg * tiles_per_group;
    int tm_end   = min(tm_start + tiles_per_group, tiles_m);

    {
        int row = tid >> 3;
        int col = (tid & 7) << 3;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int rr = row + r * 32;
            int gc = gn_base + col;
            if (rr < K && gc < N) {
                uint32_t sp = __cvta_generic_to_shared(&smB[rr][col]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(sp), "l"(B + rr * N + gc));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> fb0[4], fb1[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fb0[ki], &smB[ki * 16][wn_off],       BN + BPAD);
        load_matrix_sync(fb1[ki], &smB[ki * 16][wn_off + 16],  BN + BPAD);
    }

    for (int tm = tm_start; tm < tm_end; tm++) {
        int gm_base = tm * BM;
        int wm_off  = warp_m * 16;

        {
            int row = tid >> 3;
            int col = (tid & 7) << 3;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                int rr = row + r * 32;
                int gr = gm_base + rr;
                if (rr < BM && gr < M) {
                    uint32_t sp = __cvta_generic_to_shared(&smA[rr][col]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                        :: "r"(sp), "l"(A + gr * K + col));
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fa[4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            load_matrix_sync(fa[ki], &smA[wm_off][ki * 16], BK + APAD);
        }

        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1;
        fill_fragment(acc0, 0.0f);
        fill_fragment(acc1, 0.0f);

        mma_sync(acc0, fa[0], fb0[0], acc0);
        mma_sync(acc1, fa[0], fb1[0], acc1);
        mma_sync(acc0, fa[1], fb0[1], acc0);
        mma_sync(acc1, fa[1], fb1[1], acc1);
        mma_sync(acc0, fa[2], fb0[2], acc0);
        mma_sync(acc1, fa[2], fb1[2], acc1);
        mma_sync(acc0, fa[3], fb0[3], acc0);
        mma_sync(acc1, fa[3], fb1[3], acc1);

        __syncthreads();
        float (*smCf)[16][16] = reinterpret_cast<float(*)[16][16]>(smA);

        int gm_out = gm_base + wm_off;
        int gn_out0 = gn_base + wn_off;
        int gn_out1 = gn_out0 + 16;

        store_matrix_sync(&smCf[wid][0][0], acc0, 16, mem_row_major);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = lane + i * 32;
            int r = idx >> 3;
            int c2 = idx & 7;
            int c = c2 << 1;
            int gr = gm_out + r;
            int gc = gn_out0 + c;
            if (gr < M && gc + 1 < N) {
                __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
                *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = __float2half(smCf[wid][r][c]);
                if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
            }
        }

        store_matrix_sync(&smCf[wid][0][0], acc1, 16, mem_row_major);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = lane + i * 32;
            int r = idx >> 3;
            int c2 = idx & 7;
            int c = c2 << 1;
            int gr = gm_out + r;
            int gc = gn_out1 + c;
            if (gr < M && gc + 1 < N) {
                __half2 h2 = __float22half2_rn(make_float2(smCf[wid][r][c], smCf[wid][r][c+1]));
                *reinterpret_cast<__half2*>(&C[gr * N + gc]) = h2;
            } else if (gr < M && gc < N) {
                C[gr * N + gc] = __float2half(smCf[wid][r][c]);
                if (gc + 1 < N) C[gr * N + gc + 1] = __float2half(smCf[wid][r][c+1]);
            }
        }

        if (tm + 1 < tm_end) {
            __syncthreads();
        }
    }
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

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

    {
        int m_groups = 4;
        dim3 grid((N + 63) / 64, m_groups);
        dim3 block(256);
        hgemm_optimized_64x64_persistent<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    }
}