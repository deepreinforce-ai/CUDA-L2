#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <stdlib.h>
#include <stdint.h>

#define BLOCK_M       64
#define BLOCK_N       128
#define BLOCK_K       32
#define NUM_STAGES    4

#define WARP_M_TILES  4
#define WARP_N_TILES  2

#define SMEM_A_STRIDE 32
#define SMEM_B_STRIDE 128

static __device__ __forceinline__ void cp_async_ca_16(void* dst, const void* src) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src) : "memory");
}

static __device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src) : "memory");
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

static __device__ __forceinline__ void cp_async_wait_2() {
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
}

static __device__ __forceinline__ int swizzle_a_chunk(int row, int chunk) {
    return chunk ^ (row & 0x3);
}
static __device__ __forceinline__ int swizzle_b_chunk(int row, int chunk) {
    return chunk ^ (row & 0x7);
}

static __device__ __forceinline__
void ldmatrix_a_x4_swz(uint32_t ra[4], const half* sa, int m_base, int ki, int lane) {
    int row = m_base + (lane & 15);
    int chunk = ki * 2 + (lane >> 4);
    int schunk = swizzle_a_chunk(row, chunk);
    uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(sa + row * SMEM_A_STRIDE + schunk * 8)
    );
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3]) : "r"(addr)
    );
}

static __device__ __forceinline__
void ldmatrix_b_x2_trans_swz(uint32_t rb[2], const half* sb, int ki, int n_local_col, int lane) {
    int row = ki * 16 + (lane & 15);
    int col = n_local_col + ((lane >> 4) << 2);
    int chunk = col >> 3;
    int off   = col & 7;
    int schunk = swizzle_b_chunk(row, chunk);
    uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(sb + row * SMEM_B_STRIDE + schunk * 8 + off)
    );
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(rb[0]), "=r"(rb[1]) : "r"(addr)
    );
}

static __device__ __forceinline__
void mma_m16n8k16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
    );
}

__global__ void __launch_bounds__(256, 2)
hgemm_h100_optimized_swz4stage_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    constexpr int M = 64;
    constexpr int N = 512;
    constexpr int K = 128;
    constexpr int NUM_K_TILES = 4;
    constexpr int KI_STEPS = 2;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int n_block_start = blockIdx.x * BLOCK_N;
    const int warp_n_local  = warp_id * 16;
    const int warp_n_global = n_block_start + warp_n_local;

    float acc[WARP_M_TILES][WARP_N_TILES][4];
    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
        }
    }

    __shared__ __align__(128) half smem_A[NUM_STAGES][BLOCK_M * SMEM_A_STRIDE];
    __shared__ __align__(128) half smem_B[NUM_STAGES][BLOCK_K * SMEM_B_STRIDE];

    auto load_tile_async = [&](int k_tile, int stage) __attribute__((always_inline)) {
        int k_off = k_tile * BLOCK_K;

        {
            int elem = tid * 8;
            int row  = elem / BLOCK_K;
            int col  = elem % BLOCK_K;
            int chunk = col >> 3;
            int schunk = swizzle_a_chunk(row, chunk);

            half* dst = smem_A[stage] + row * SMEM_A_STRIDE + schunk * 8;
            const half* src = A + row * K + (k_off + col);
            cp_async_ca_16(dst, src);
        }

        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            int idx  = tid + it * 256;
            int elem = idx * 8;
            int row  = elem / BLOCK_N;
            int col  = elem % BLOCK_N;
            int chunk = col >> 3;
            int schunk = swizzle_b_chunk(row, chunk);

            half* dst = smem_B[stage] + row * SMEM_B_STRIDE + schunk * 8;
            const half* src = B + (k_off + row) * N + (n_block_start + col);
            cp_async_cg_16(dst, src);
        }

        cp_async_commit();
    };

    auto load_ki_frags = [&](uint32_t ra[][4], uint32_t rb[][2], const half* sa, const half* sb, int ki)
        __attribute__((always_inline)) {
        #pragma unroll
        for (int mi = 0; mi < WARP_M_TILES; ++mi) {
            ldmatrix_a_x4_swz(ra[mi], sa, mi * 16, ki, lane_id);
        }
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            ldmatrix_b_x2_trans_swz(rb[ni], sb, ki, warp_n_local + ni * 8, lane_id);
        }
    };

    load_tile_async(0, 0);
    load_tile_async(1, 1);
    load_tile_async(2, 2);
    cp_async_wait_2();
    __syncthreads();

    uint32_t ra_cur[WARP_M_TILES][4], ra_nxt[WARP_M_TILES][4];
    uint32_t rb_cur[WARP_N_TILES][2], rb_nxt[WARP_N_TILES][2];

    int cur_stage = 0;
    load_ki_frags(ra_cur, rb_cur, smem_A[cur_stage], smem_B[cur_stage], 0);

    #pragma unroll
    for (int kt = 0; kt < NUM_K_TILES; ++kt) {
        if (kt + 3 < NUM_K_TILES) {
            load_tile_async(kt + 3, (cur_stage + 3) % NUM_STAGES);
        } else {
            cp_async_commit();
        }

        const half* sa = smem_A[cur_stage];
        const half* sb = smem_B[cur_stage];

        #pragma unroll
        for (int ki = 0; ki < KI_STEPS; ++ki) {
            if (ki == 0) {
                load_ki_frags(ra_nxt, rb_nxt, sa, sb, 1);
            }

            #pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WARP_N_TILES; ++ni) {
                    mma_m16n8k16(acc[mi][ni], ra_cur[mi], rb_cur[ni]);
                }
            }

            if (ki == 0) {
                #pragma unroll
                for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) ra_cur[mi][r] = ra_nxt[mi][r];
                }
                #pragma unroll
                for (int ni = 0; ni < WARP_N_TILES; ++ni) {
                    rb_cur[ni][0] = rb_nxt[ni][0];
                    rb_cur[ni][1] = rb_nxt[ni][1];
                }
            }
        }

        cur_stage = (cur_stage + 1) % NUM_STAGES;

        if (kt != NUM_K_TILES - 1) {
            cp_async_wait_2();
            __syncthreads();
            load_ki_frags(ra_cur, rb_cur, smem_A[cur_stage], smem_B[cur_stage], 0);
        }
    }

    const int r0_base = lane_id >> 2;
    const int c0_base = (lane_id & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        int gr0 = mi * 16 + r0_base;
        int gr1 = gr0 + 8;

        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            int gc0 = warp_n_global + ni * 8 + c0_base;
            *reinterpret_cast<__half2*>(C + gr0 * N + gc0) =
                __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<__half2*>(C + gr1 * N + gc0) =
                __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C       = reinterpret_cast<half*>(c.data_ptr());

    hgemm_h100_optimized_swz4stage_kernel<<<dim3(4, 1, 1), dim3(256, 1, 1)>>>(A, B, C);
}