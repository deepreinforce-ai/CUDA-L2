#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

using namespace nvcuda::wmma;

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm volatile("{.reg .u64 u64addr;\n"
                 " cvta.to.shared.u64 u64addr, %1;\n"
                 " cvt.u32.u64 %0, u64addr;}\n"
                 : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void store128_f2h(half* addr,
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7) {
    uint4 out;
    uint16_t* p = reinterpret_cast<uint16_t*>(&out);
    p[0] = __half_as_ushort(__float2half(f0));
    p[1] = __half_as_ushort(__float2half(f1));
    p[2] = __half_as_ushort(__float2half(f2));
    p[3] = __half_as_ushort(__float2half(f3));
    p[4] = __half_as_ushort(__float2half(f4));
    p[5] = __half_as_ushort(__float2half(f5));
    p[6] = __half_as_ushort(__float2half(f6));
    p[7] = __half_as_ushort(__float2half(f7));
    *reinterpret_cast<uint4*>(addr) = out;
}

__global__ void __launch_bounds__(64, 2)
hgemm_v5_dual64_padA(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    constexpr int K       = 128;
    constexpr int N       = 128;
    constexpr int BM      = 64;
    constexpr int BK      = 16;
    constexpr int BK_PAD  = 24;
    constexpr int NUM_K   = K / BK;

    const int bm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bm_base      = bm * BM;
    const int warp_m_base0 = warp_id * 16;
    const int warp_m_base1 = warp_id * 16 + 32;

    __shared__ __align__(128) half smem_B[128][128];
    __shared__ __align__(128) half smem_A[3][BM][BK_PAD];

    fragment<accumulator, 16, 16, 16, float> acc0[8], acc1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        fill_fragment(acc0[i], 0.0f);
        fill_fragment(acc1[i], 0.0f);
    }

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int flat = tid + i * 64;
        int br   = flat >> 4;
        int bc   = (flat & 15) << 3;
        cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int a_row0 = tid >> 1;
    const int a_row1 = (tid >> 1) + 32;
    const int a_col  = (tid & 1) << 3;

    cp_async_16(smem_u32addr(&smem_A[0][a_row0][a_col]),
                A + (bm_base + a_row0) * K + 0 * BK + a_col);
    cp_async_16(smem_u32addr(&smem_A[0][a_row1][a_col]),
                A + (bm_base + a_row1) * K + 0 * BK + a_col);
    cp_async_commit();

    cp_async_16(smem_u32addr(&smem_A[1][a_row0][a_col]),
                A + (bm_base + a_row0) * K + 1 * BK + a_col);
    cp_async_16(smem_u32addr(&smem_A[1][a_row1][a_col]),
                A + (bm_base + a_row1) * K + 1 * BK + a_col);
    cp_async_commit();

    #pragma unroll 8
    for (int k = 0; k < NUM_K; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;

        if (k + 2 < NUM_K) {
            cp_async_16(smem_u32addr(&smem_A[nxt][a_row0][a_col]),
                        A + (bm_base + a_row0) * K + (k + 2) * BK + a_col);
            cp_async_16(smem_u32addr(&smem_A[nxt][a_row1][a_col]),
                        A + (bm_base + a_row1) * K + (k + 2) * BK + a_col);
            cp_async_commit();
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag0, a_frag1;
        load_matrix_sync(a_frag0, &smem_A[cur][warp_m_base0][0], BK_PAD);
        load_matrix_sync(a_frag1, &smem_A[cur][warp_m_base1][0], BK_PAD);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
            load_matrix_sync(b_frag, &smem_B[k * BK][ni * 16], N);
            mma_sync(acc0[ni], a_frag0, b_frag, acc0[ni]);
            mma_sync(acc1[ni], a_frag1, b_frag, acc1[ni]);
        }
    }

    float* smem_float = reinterpret_cast<float*>(&smem_B[0][0]);
    float* wbuf0      = smem_float + warp_id * 4096;
    float* wbuf1      = smem_float + warp_id * 4096 + 2048;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        store_matrix_sync(wbuf0 + ni * 16, acc0[ni], N, mem_row_major);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        store_matrix_sync(wbuf1 + ni * 16, acc1[ni], N, mem_row_major);

    const int gm_r0_base = bm_base + warp_m_base0;
    const int gm_r1_base = bm_base + warp_m_base1;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat  = (lane_id << 3) + i * 256;
        int row   = flat >> 7;
        int col   = flat & 127;
        int gm_r0 = gm_r0_base + row;
        int gm_r1 = gm_r1_base + row;
        if (gm_r0 < M)
            store128_f2h(C + gm_r0 * N + col,
                         wbuf0[flat], wbuf0[flat+1], wbuf0[flat+2], wbuf0[flat+3],
                         wbuf0[flat+4], wbuf0[flat+5], wbuf0[flat+6], wbuf0[flat+7]);
        if (gm_r1 < M)
            store128_f2h(C + gm_r1 * N + col,
                         wbuf1[flat], wbuf1[flat+1], wbuf1[flat+2], wbuf1[flat+3],
                         wbuf1[flat+4], wbuf1[flat+5], wbuf1[flat+6], wbuf1[flat+7]);
    }
}

__global__ void __launch_bounds__(64, 2)
hgemm_v5_persistent_dual64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    constexpr int K       = 128;
    constexpr int N       = 128;
    constexpr int BM      = 64;
    constexpr int BK      = 16;
    constexpr int BK_PAD  = 24;
    constexpr int NUM_K   = K / BK;

    const int bid     = blockIdx.x;
    const int gdim    = gridDim.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_m_base0 = warp_id * 16;
    const int warp_m_base1 = warp_id * 16 + 32;

    __shared__ __align__(128) half smem_B[128][128];
    __shared__ __align__(128) half smem_A[3][BM][BK_PAD];

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int flat = tid + i * 64;
        int br   = flat >> 4;
        int bc   = (flat & 15) << 3;
        cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int a_row0 = tid >> 1;
    const int a_row1 = (tid >> 1) + 32;
    const int a_col  = (tid & 1) << 3;

    int num_tiles = (M + BM - 1) / BM;

    for (int bm_idx = bid; bm_idx < num_tiles; bm_idx += gdim) {
        const int bm_base = bm_idx * BM;

        fragment<accumulator, 16, 16, 16, float> acc0[8], acc1[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            fill_fragment(acc0[i], 0.0f);
            fill_fragment(acc1[i], 0.0f);
        }

        cp_async_16(smem_u32addr(&smem_A[0][a_row0][a_col]),
                    A + (bm_base + a_row0) * K + 0 * BK + a_col);
        cp_async_16(smem_u32addr(&smem_A[0][a_row1][a_col]),
                    A + (bm_base + a_row1) * K + 0 * BK + a_col);
        cp_async_commit();

        cp_async_16(smem_u32addr(&smem_A[1][a_row0][a_col]),
                    A + (bm_base + a_row0) * K + 1 * BK + a_col);
        cp_async_16(smem_u32addr(&smem_A[1][a_row1][a_col]),
                    A + (bm_base + a_row1) * K + 1 * BK + a_col);
        cp_async_commit();

        #pragma unroll 8
        for (int k = 0; k < NUM_K; k++) {
            const int cur = k % 3;
            const int nxt = (k + 2) % 3;

            if (k + 2 < NUM_K) {
                cp_async_16(smem_u32addr(&smem_A[nxt][a_row0][a_col]),
                            A + (bm_base + a_row0) * K + (k + 2) * BK + a_col);
                cp_async_16(smem_u32addr(&smem_A[nxt][a_row1][a_col]),
                            A + (bm_base + a_row1) * K + (k + 2) * BK + a_col);
                cp_async_commit();
                cp_async_wait_one();
            } else {
                cp_async_wait_all();
            }
            __syncthreads();

            fragment<matrix_a, 16, 16, 16, half, row_major> a_frag0, a_frag1;
            load_matrix_sync(a_frag0, &smem_A[cur][warp_m_base0][0], BK_PAD);
            load_matrix_sync(a_frag1, &smem_A[cur][warp_m_base1][0], BK_PAD);

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
                load_matrix_sync(b_frag, &smem_B[k * BK][ni * 16], N);
                mma_sync(acc0[ni], a_frag0, b_frag, acc0[ni]);
                mma_sync(acc1[ni], a_frag1, b_frag, acc1[ni]);
            }
        }

        float* smem_float = reinterpret_cast<float*>(&smem_B[0][0]);
        float* wbuf0      = smem_float + warp_id * 4096;
        float* wbuf1      = smem_float + warp_id * 4096 + 2048;

        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            store_matrix_sync(wbuf0 + ni * 16, acc0[ni], N, mem_row_major);
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            store_matrix_sync(wbuf1 + ni * 16, acc1[ni], N, mem_row_major);

        const int gm_r0_base = bm_base + warp_m_base0;
        const int gm_r1_base = bm_base + warp_m_base1;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int flat  = (lane_id << 3) + i * 256;
            int row   = flat >> 7;
            int col   = flat & 127;
            int gm_r0 = gm_r0_base + row;
            int gm_r1 = gm_r1_base + row;
            if (gm_r0 < M)
                store128_f2h(C + gm_r0 * N + col,
                             wbuf0[flat], wbuf0[flat+1], wbuf0[flat+2], wbuf0[flat+3],
                             wbuf0[flat+4], wbuf0[flat+5], wbuf0[flat+6], wbuf0[flat+7]);
            if (gm_r1 < M)
                store128_f2h(C + gm_r1 * N + col,
                             wbuf1[flat], wbuf1[flat+1], wbuf1[flat+2], wbuf1[flat+3],
                             wbuf1[flat+4], wbuf1[flat+5], wbuf1[flat+6], wbuf1[flat+7]);
        }

        if (bm_idx + gdim < num_tiles) {
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int flat = tid + i * 64;
                int br   = flat >> 4;
                int bc   = (flat & 15) << 3;
                cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
            }
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(64, 3)
hgemm_v5_tribuf32_padA(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    constexpr int K       = 128;
    constexpr int N       = 128;
    constexpr int BM      = 32;
    constexpr int BK      = 16;
    constexpr int BK_PAD  = 24;
    constexpr int NUM_K   = K / BK;

    const int bm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bm_base     = bm * BM;
    const int warp_m_base = warp_id * 16;

    __shared__ __align__(128) half smem_B[128][128];
    __shared__ __align__(128) half smem_A[3][BM][BK_PAD];

    fragment<accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) fill_fragment(acc[i], 0.0f);

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int flat = tid + i * 64;
        int br   = flat >> 4;
        int bc   = (flat & 15) << 3;
        cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 3;

    cp_async_16(smem_u32addr(&smem_A[0][a_row][a_col]),
                A + (bm_base + a_row) * K + 0 * BK + a_col);
    cp_async_commit();
    cp_async_16(smem_u32addr(&smem_A[1][a_row][a_col]),
                A + (bm_base + a_row) * K + 1 * BK + a_col);
    cp_async_commit();

    #pragma unroll 8
    for (int k = 0; k < NUM_K; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;

        if (k + 2 < NUM_K) {
            cp_async_16(smem_u32addr(&smem_A[nxt][a_row][a_col]),
                        A + (bm_base + a_row) * K + (k + 2) * BK + a_col);
            cp_async_commit();
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        load_matrix_sync(a_frag, &smem_A[cur][warp_m_base][0], BK_PAD);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
            load_matrix_sync(b_frag, &smem_B[k * BK][ni * 16], N);
            mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
        }
    }

    float* smem_float = reinterpret_cast<float*>(&smem_B[0][0]);
    float* warp_buf   = smem_float + warp_id * 2048;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        store_matrix_sync(warp_buf + ni * 16, acc[ni], N, mem_row_major);

    const int gm_row_base = bm_base + warp_m_base;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat   = (lane_id << 3) + i * 256;
        int row    = flat >> 7;
        int col    = flat & 127;
        int gm_row = gm_row_base + row;
        if (gm_row < M)
            store128_f2h(C + gm_row * N + col,
                         warp_buf[flat], warp_buf[flat+1], warp_buf[flat+2], warp_buf[flat+3],
                         warp_buf[flat+4], warp_buf[flat+5], warp_buf[flat+6], warp_buf[flat+7]);
    }
}

__global__ void __launch_bounds__(128, 2)
hgemm_v5_tribuf64_4warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    constexpr int K       = 128;
    constexpr int N       = 128;
    constexpr int BM      = 64;
    constexpr int BK      = 16;
    constexpr int BK_PAD  = 24;
    constexpr int NUM_K   = K / BK;

    const int bm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bm_base     = bm * BM;
    const int warp_m_base = warp_id * 16;

    __shared__ __align__(128) half smem_B[128][128];
    __shared__ __align__(128) half smem_A[3][BM][BK_PAD];

    fragment<accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) fill_fragment(acc[i], 0.0f);

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int flat = tid + i * 128;
        int br   = flat >> 4;
        int bc   = (flat & 15) << 3;
        cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 3;

    cp_async_16(smem_u32addr(&smem_A[0][a_row][a_col]),
                A + (bm_base + a_row) * K + 0 * BK + a_col);
    cp_async_commit();
    cp_async_16(smem_u32addr(&smem_A[1][a_row][a_col]),
                A + (bm_base + a_row) * K + 1 * BK + a_col);
    cp_async_commit();

    #pragma unroll 8
    for (int k = 0; k < NUM_K; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;

        if (k + 2 < NUM_K) {
            cp_async_16(smem_u32addr(&smem_A[nxt][a_row][a_col]),
                        A + (bm_base + a_row) * K + (k + 2) * BK + a_col);
            cp_async_commit();
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        load_matrix_sync(a_frag, &smem_A[cur][warp_m_base][0], BK_PAD);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
            load_matrix_sync(b_frag, &smem_B[k * BK][ni * 16], N);
            mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
        }
    }

    float* smem_float = reinterpret_cast<float*>(&smem_B[0][0]);
    float* warp_buf   = smem_float + warp_id * 2048;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        store_matrix_sync(warp_buf + ni * 16, acc[ni], N, mem_row_major);

    const int gm_row_base = bm_base + warp_m_base;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat   = (lane_id << 3) + i * 256;
        int row    = flat >> 7;
        int col    = flat & 127;
        int gm_row = gm_row_base + row;
        if (gm_row < M)
            store128_f2h(C + gm_row * N + col,
                         warp_buf[flat], warp_buf[flat+1], warp_buf[flat+2], warp_buf[flat+3],
                         warp_buf[flat+4], warp_buf[flat+5], warp_buf[flat+6], warp_buf[flat+7]);
    }
}

__global__ void __launch_bounds__(32, 6)
hgemm_v5_hiocc16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    constexpr int K       = 128;
    constexpr int N       = 128;
    constexpr int BM      = 16;
    constexpr int BK      = 16;
    constexpr int BK_PAD  = 24;
    constexpr int NUM_K   = K / BK;

    const int bm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane_id = tid;
    const int bm_base = bm * BM;

    __shared__ __align__(128) half smem_B[128][128];
    __shared__ __align__(128) half smem_A[2][BM][BK_PAD];

    fragment<accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) fill_fragment(acc[i], 0.0f);

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        int flat = tid + i * 32;
        int br   = flat >> 4;
        int bc   = (flat & 15) << 3;
        cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 3;

    cp_async_16(smem_u32addr(&smem_A[0][a_row][a_col]),
                A + (bm_base + a_row) * K + a_col);
    cp_async_commit();

    #pragma unroll 8
    for (int k = 0; k < NUM_K; k++) {
        const int cur = k & 1;
        const int nxt = 1 ^ cur;

        if (k + 1 < NUM_K) {
            cp_async_16(smem_u32addr(&smem_A[nxt][a_row][a_col]),
                        A + (bm_base + a_row) * K + (k + 1) * BK + a_col);
            cp_async_commit();
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        load_matrix_sync(a_frag, &smem_A[cur][0][0], BK_PAD);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
            load_matrix_sync(b_frag, &smem_B[k * BK][ni * 16], N);
            mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
        }
    }

    float* smem_float = reinterpret_cast<float*>(&smem_B[0][0]);

    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        store_matrix_sync(smem_float + ni * 16, acc[ni], N, mem_row_major);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat = (lane_id << 3) + i * 256;
        int row  = flat >> 7;
        int col  = flat & 127;
        int gm   = bm_base + row;
        if (gm < M)
            store128_f2h(C + gm * N + col,
                         smem_float[flat], smem_float[flat+1], smem_float[flat+2], smem_float[flat+3],
                         smem_float[flat+4], smem_float[flat+5], smem_float[flat+6], smem_float[flat+7]);
    }
}

__global__ void __launch_bounds__(64, 3)
hgemm_v5_dual64_noPad(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    constexpr int K     = 128;
    constexpr int N     = 128;
    constexpr int BM    = 64;
    constexpr int BK    = 16;
    constexpr int NUM_K = K / BK;

    const int bm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bm_base      = bm * BM;
    const int warp_m_base0 = warp_id * 16;
    const int warp_m_base1 = warp_id * 16 + 32;

    __shared__ __align__(128) half smem_B[128][128];
    __shared__ __align__(128) half smem_A[3][BM][BK];

    fragment<accumulator, 16, 16, 16, float> acc0[8], acc1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        fill_fragment(acc0[i], 0.0f);
        fill_fragment(acc1[i], 0.0f);
    }

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int flat = tid + i * 64;
        int br   = flat >> 4;
        int bc   = (flat & 15) << 3;
        cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int a_row0 = tid >> 1;
    const int a_row1 = (tid >> 1) + 32;
    const int a_col  = (tid & 1) << 3;

    cp_async_16(smem_u32addr(&smem_A[0][a_row0][a_col]),
                A + (bm_base + a_row0) * K + 0 * BK + a_col);
    cp_async_16(smem_u32addr(&smem_A[0][a_row1][a_col]),
                A + (bm_base + a_row1) * K + 0 * BK + a_col);
    cp_async_commit();

    cp_async_16(smem_u32addr(&smem_A[1][a_row0][a_col]),
                A + (bm_base + a_row0) * K + 1 * BK + a_col);
    cp_async_16(smem_u32addr(&smem_A[1][a_row1][a_col]),
                A + (bm_base + a_row1) * K + 1 * BK + a_col);
    cp_async_commit();

    #pragma unroll 8
    for (int k = 0; k < NUM_K; k++) {
        const int cur = k % 3;
        const int nxt = (k + 2) % 3;

        if (k + 2 < NUM_K) {
            cp_async_16(smem_u32addr(&smem_A[nxt][a_row0][a_col]),
                        A + (bm_base + a_row0) * K + (k + 2) * BK + a_col);
            cp_async_16(smem_u32addr(&smem_A[nxt][a_row1][a_col]),
                        A + (bm_base + a_row1) * K + (k + 2) * BK + a_col);
            cp_async_commit();
            cp_async_wait_one();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag0, a_frag1;
        load_matrix_sync(a_frag0, &smem_A[cur][warp_m_base0][0], BK);
        load_matrix_sync(a_frag1, &smem_A[cur][warp_m_base1][0], BK);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
            load_matrix_sync(b_frag, &smem_B[k * BK][ni * 16], N);
            mma_sync(acc0[ni], a_frag0, b_frag, acc0[ni]);
            mma_sync(acc1[ni], a_frag1, b_frag, acc1[ni]);
        }
    }

    float* smem_float = reinterpret_cast<float*>(&smem_B[0][0]);
    float* wbuf0      = smem_float + warp_id * 4096;
    float* wbuf1      = smem_float + warp_id * 4096 + 2048;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        store_matrix_sync(wbuf0 + ni * 16, acc0[ni], N, mem_row_major);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        store_matrix_sync(wbuf1 + ni * 16, acc1[ni], N, mem_row_major);

    const int gm_r0_base = bm_base + warp_m_base0;
    const int gm_r1_base = bm_base + warp_m_base1;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat  = (lane_id << 3) + i * 256;
        int row   = flat >> 7;
        int col   = flat & 127;
        int gm_r0 = gm_r0_base + row;
        int gm_r1 = gm_r1_base + row;
        if (gm_r0 < M)
            store128_f2h(C + gm_r0 * N + col,
                         wbuf0[flat], wbuf0[flat+1], wbuf0[flat+2], wbuf0[flat+3],
                         wbuf0[flat+4], wbuf0[flat+5], wbuf0[flat+6], wbuf0[flat+7]);
        if (gm_r1 < M)
            store128_f2h(C + gm_r1 * N + col,
                         wbuf1[flat], wbuf1[flat+1], wbuf1[flat+2], wbuf1[flat+3],
                         wbuf1[flat+4], wbuf1[flat+5], wbuf1[flat+6], wbuf1[flat+7]);
    }
}

__global__ void __launch_bounds__(64, 2)
hgemm_v5_persistent32_sepEpilogue(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    constexpr int K       = 128;
    constexpr int N       = 128;
    constexpr int BM      = 32;
    constexpr int BK      = 16;
    constexpr int BK_PAD  = 24;
    constexpr int NUM_K   = K / BK;

    const int bid     = blockIdx.x;
    const int gdim    = gridDim.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ __align__(128) half smem_B[128][128];
    __shared__ __align__(128) half smem_A[3][BM][BK_PAD];
    __shared__ __align__(128) float smem_C[2][16 * 128];

    const int warp_m_base = warp_id * 16;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int flat = tid + i * 64;
        int br   = flat >> 4;
        int bc   = (flat & 15) << 3;
        cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 3;

    int num_tiles = (M + BM - 1) / BM;

    for (int bm_idx = bid; bm_idx < num_tiles; bm_idx += gdim) {
        const int bm_base = bm_idx * BM;

        fragment<accumulator, 16, 16, 16, float> acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) fill_fragment(acc[i], 0.0f);

        cp_async_16(smem_u32addr(&smem_A[0][a_row][a_col]),
                    A + (bm_base + a_row) * K + 0 * BK + a_col);
        cp_async_commit();
        cp_async_16(smem_u32addr(&smem_A[1][a_row][a_col]),
                    A + (bm_base + a_row) * K + 1 * BK + a_col);
        cp_async_commit();

        #pragma unroll 8
        for (int k = 0; k < NUM_K; k++) {
            const int cur = k % 3;
            const int nxt = (k + 2) % 3;

            if (k + 2 < NUM_K) {
                cp_async_16(smem_u32addr(&smem_A[nxt][a_row][a_col]),
                            A + (bm_base + a_row) * K + (k + 2) * BK + a_col);
                cp_async_commit();
                cp_async_wait_one();
            } else {
                cp_async_wait_all();
            }
            __syncthreads();

            fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
            load_matrix_sync(a_frag, &smem_A[cur][warp_m_base][0], BK_PAD);

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
                load_matrix_sync(b_frag, &smem_B[k * BK][ni * 16], N);
                mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
            }
        }

        float* smem_float = reinterpret_cast<float*>(&smem_B[0][0]);
        float* warp_buf   = smem_float + warp_id * 2048;

        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            store_matrix_sync(warp_buf + ni * 16, acc[ni], N, mem_row_major);

        const int gm_row_base = bm_base + warp_m_base;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int flat   = (lane_id << 3) + i * 256;
            int row    = flat >> 7;
            int col    = flat & 127;
            int gm_row = gm_row_base + row;
            if (gm_row < M)
                store128_f2h(C + gm_row * N + col,
                             warp_buf[flat], warp_buf[flat+1], warp_buf[flat+2], warp_buf[flat+3],
                             warp_buf[flat+4], warp_buf[flat+5], warp_buf[flat+6], warp_buf[flat+7]);
        }

        if (bm_idx + gdim < num_tiles) {
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int flat = tid + i * 64;
                int br   = flat >> 4;
                int bc   = (flat & 15) << 3;
                cp_async_16(smem_u32addr(&smem_B[br][bc]), B + br * N + bc);
            }
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();
        }
    }
}

static bool g_initialized = false;
static int  g_best_kernel = 0;
static int  g_best_grid   = 0;
static int  g_best_block  = 64;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C       = reinterpret_cast<half*>(c.data_ptr());

    if (!g_initialized) {
        cudaFuncSetAttribute(hgemm_v5_dual64_padA,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(hgemm_v5_persistent_dual64,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(hgemm_v5_tribuf32_padA,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(hgemm_v5_tribuf64_4warp,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(hgemm_v5_hiocc16,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(hgemm_v5_dual64_noPad,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(hgemm_v5_persistent32_sepEpilogue,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int warmup = 10;
        const int iters  = 100;
        float best_time  = 1e10f;

        struct KConf { int id, grid, block; };

        int tiles64  = (M + 63) / 64;
        int tiles32  = (M + 31) / 32;
        int tiles16  = (M + 15) / 16;

        KConf configs[] = {
            {0, tiles64,  64},
            {1, 132,      64},
            {1, 264,      64},
            {1, tiles64,  64},
            {2, tiles32,  64},
            {3, tiles64, 128},
            {4, tiles16,  32},
            {5, tiles64,  64},
            {5, 132,      64},
            {5, 264,      64},
            {6, 132,      64},
            {6, 264,      64},
        };
        int nconf = (int)(sizeof(configs) / sizeof(configs[0]));

        auto run = [&](const KConf& cfg) {
            switch (cfg.id) {
                case 0: hgemm_v5_dual64_padA<<<cfg.grid, cfg.block>>>(A, B, C, M); break;
                case 1: hgemm_v5_persistent_dual64<<<cfg.grid, cfg.block>>>(A, B, C, M); break;
                case 2: hgemm_v5_tribuf32_padA<<<cfg.grid, cfg.block>>>(A, B, C, M); break;
                case 3: hgemm_v5_tribuf64_4warp<<<cfg.grid, cfg.block>>>(A, B, C, M); break;
                case 4: hgemm_v5_hiocc16<<<cfg.grid, cfg.block>>>(A, B, C, M); break;
                case 5: hgemm_v5_dual64_noPad<<<cfg.grid, cfg.block>>>(A, B, C, M); break;
                case 6: hgemm_v5_persistent32_sepEpilogue<<<cfg.grid, cfg.block>>>(A, B, C, M); break;
            }
        };

        for (int ci = 0; ci < nconf; ci++) {
            for (int i = 0; i < warmup; i++) run(configs[ci]);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            for (int i = 0; i < iters; i++) run(configs[ci]);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float t;
            cudaEventElapsedTime(&t, start, stop);
            if (t < best_time) {
                best_time     = t;
                g_best_kernel = configs[ci].id;
                g_best_grid   = configs[ci].grid;
                g_best_block  = configs[ci].block;
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        g_initialized = true;
    }

    switch (g_best_kernel) {
        case 0: hgemm_v5_dual64_padA<<<g_best_grid, g_best_block>>>(A, B, C, M); break;
        case 1: hgemm_v5_persistent_dual64<<<g_best_grid, g_best_block>>>(A, B, C, M); break;
        case 2: hgemm_v5_tribuf32_padA<<<g_best_grid, g_best_block>>>(A, B, C, M); break;
        case 3: hgemm_v5_tribuf64_4warp<<<g_best_grid, g_best_block>>>(A, B, C, M); break;
        case 4: hgemm_v5_hiocc16<<<g_best_grid, g_best_block>>>(A, B, C, M); break;
        case 5: hgemm_v5_dual64_noPad<<<g_best_grid, g_best_block>>>(A, B, C, M); break;
        case 6: hgemm_v5_persistent32_sepEpilogue<<<g_best_grid, g_best_block>>>(A, B, C, M); break;
    }
}