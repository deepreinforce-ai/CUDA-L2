#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

#define BM      16
#define BN      64
#define BK      16
#define STAGES   8
#define MMA_M   16
#define MMA_N    8
#define MMA_K   16

#define SA_STRIDE  24
#define SB_STRIDE  72

__global__ void __launch_bounds__(32, 10)
hgemm_persistent_bk16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K)
{
    const int lane_id   = threadIdx.x;
    const int N = 64;
    const int num_k_tiles = K / BK;

    __shared__ half smem_A[STAGES][BM][SA_STRIDE];
    __shared__ half smem_B[STAGES][BK][SB_STRIDE];

    for (int block_row = blockIdx.x * BM; block_row < M; block_row += gridDim.x * BM) {

        float acc[8][4];
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        const int ra0 = (lane_id * 8) / BK;
        const int ca0 = (lane_id * 8) % BK;

        const int rb0 = (lane_id * 8 +   0) / N, cb0 = (lane_id * 8 +   0) % N;
        const int rb1 = (lane_id * 8 + 256) / N, cb1 = (lane_id * 8 + 256) % N;
        const int rb2 = (lane_id * 8 + 512) / N, cb2 = (lane_id * 8 + 512) % N;
        const int rb3 = (lane_id * 8 + 768) / N, cb3 = (lane_id * 8 + 768) % N;

        const int gA0 = block_row + ra0;

        #pragma unroll
        for (int s = 0; s < STAGES - 1; s++) {
            const int k_off = s * BK;
            {
                uint32_t sa = __cvta_generic_to_shared(&smem_A[s][ra0][ca0]);
                if (gA0 < M)
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA0 * K + k_off + ca0));
                else
                    *reinterpret_cast<float4*>(&smem_A[s][ra0][ca0]) = make_float4(0,0,0,0);
            }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb0][cb0]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb0)*N + cb0)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb1][cb1]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb1)*N + cb1)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb2][cb2]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb2)*N + cb2)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb3][cb3]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb3)*N + cb3)); }
            asm volatile("cp.async.commit_group;\n");
        }

        #pragma unroll 1
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            const int pf = k_tile + STAGES - 1;
            if (pf < num_k_tiles) {
                const int ps    = pf % STAGES;
                const int k_off = pf * BK;
                {
                    uint32_t sa = __cvta_generic_to_shared(&smem_A[ps][ra0][ca0]);
                    if (gA0 < M)
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA0 * K + k_off + ca0));
                    else
                        *reinterpret_cast<float4*>(&smem_A[ps][ra0][ca0]) = make_float4(0,0,0,0);
                }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb0][cb0]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb0)*N + cb0)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb1][cb1]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb1)*N + cb1)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb2][cb2]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb2)*N + cb2)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb3][cb3]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb3)*N + cb3)); }
            }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 7;\n");

            const int cs = k_tile % STAGES;

            uint32_t a_frag[4];
            {
                const int lm_row = lane_id & 15;
                const int lm_col = (lane_id >> 4) << 3;
                uint32_t sa = __cvta_generic_to_shared(&smem_A[cs][lm_row][lm_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(sa));
            }

            uint32_t b_frag[8][2];
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                const int lm_row = lane_id & 15;
                const int lm_col = ni * MMA_N;
                uint32_t sb = __cvta_generic_to_shared(&smem_B[cs][lm_row][lm_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                    : "r"(sb));
            }

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                      "+f"(acc[ni][2]), "+f"(acc[ni][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[ni][0]), "r"(b_frag[ni][1]));
            }
        }

        asm volatile("cp.async.wait_all;\n");

        const int r0      = block_row + (lane_id >> 2);
        const int r1      = r0 + 8;
        const int col_off = (lane_id & 3) * 2;

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int c0 = ni * MMA_N + col_off;
            if (r0 < M)
                *reinterpret_cast<half2*>(&C[r0 * N + c0]) = __floats2half2_rn(acc[ni][0], acc[ni][1]);
            if (r1 < M)
                *reinterpret_cast<half2*>(&C[r1 * N + c0]) = __floats2half2_rn(acc[ni][2], acc[ni][3]);
        }
    }
}

#define BK32    32
#define STAGES4  4
#define SA32_S  40
#define SB32_S  72

__global__ void __launch_bounds__(32, 8)
hgemm_bk32_bm16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K)
{
    const int lane_id   = threadIdx.x;
    const int N = 64;

    for (int block_row = blockIdx.x * BM; block_row < M; block_row += gridDim.x * BM) {

        float acc[8][4];
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        __shared__ half smem_A[STAGES4][BM][SA32_S];
        __shared__ half smem_B[STAGES4][BK32][SB32_S];

        const int num_k_tiles = K / BK32;

        const int ra0 = (lane_id * 8)       / BK32, ca0 = (lane_id * 8)       % BK32;
        const int ra1 = (lane_id * 8 + 256) / BK32, ca1 = (lane_id * 8 + 256) % BK32;

        const int rb0 = (lane_id * 8 +    0) / N, cb0 = (lane_id * 8 +    0) % N;
        const int rb1 = (lane_id * 8 +  256) / N, cb1 = (lane_id * 8 +  256) % N;
        const int rb2 = (lane_id * 8 +  512) / N, cb2 = (lane_id * 8 +  512) % N;
        const int rb3 = (lane_id * 8 +  768) / N, cb3 = (lane_id * 8 +  768) % N;
        const int rb4 = (lane_id * 8 + 1024) / N, cb4 = (lane_id * 8 + 1024) % N;
        const int rb5 = (lane_id * 8 + 1280) / N, cb5 = (lane_id * 8 + 1280) % N;
        const int rb6 = (lane_id * 8 + 1536) / N, cb6 = (lane_id * 8 + 1536) % N;
        const int rb7 = (lane_id * 8 + 1792) / N, cb7 = (lane_id * 8 + 1792) % N;

        const int gA0 = block_row + ra0;
        const int gA1 = block_row + ra1;

        #pragma unroll
        for (int s = 0; s < STAGES4 - 1; s++) {
            const int k_off = s * BK32;
            { uint32_t sa = __cvta_generic_to_shared(&smem_A[s][ra0][ca0]); if (gA0 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA0*K + k_off + ca0)); else *reinterpret_cast<float4*>(&smem_A[s][ra0][ca0]) = make_float4(0,0,0,0); }
            { uint32_t sa = __cvta_generic_to_shared(&smem_A[s][ra1][ca1]); if (gA1 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA1*K + k_off + ca1)); else *reinterpret_cast<float4*>(&smem_A[s][ra1][ca1]) = make_float4(0,0,0,0); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb0][cb0]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb0)*N + cb0)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb1][cb1]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb1)*N + cb1)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb2][cb2]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb2)*N + cb2)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb3][cb3]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb3)*N + cb3)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb4][cb4]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb4)*N + cb4)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb5][cb5]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb5)*N + cb5)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb6][cb6]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb6)*N + cb6)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb7][cb7]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb7)*N + cb7)); }
            asm volatile("cp.async.commit_group;\n");
        }

        #pragma unroll 1
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            const int pf = k_tile + STAGES4 - 1;
            if (pf < num_k_tiles) {
                const int ps = pf % STAGES4, k_off = pf * BK32;
                { uint32_t sa = __cvta_generic_to_shared(&smem_A[ps][ra0][ca0]); if (gA0 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA0*K + k_off + ca0)); else *reinterpret_cast<float4*>(&smem_A[ps][ra0][ca0]) = make_float4(0,0,0,0); }
                { uint32_t sa = __cvta_generic_to_shared(&smem_A[ps][ra1][ca1]); if (gA1 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA1*K + k_off + ca1)); else *reinterpret_cast<float4*>(&smem_A[ps][ra1][ca1]) = make_float4(0,0,0,0); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb0][cb0]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb0)*N + cb0)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb1][cb1]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb1)*N + cb1)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb2][cb2]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb2)*N + cb2)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb3][cb3]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb3)*N + cb3)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb4][cb4]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb4)*N + cb4)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb5][cb5]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb5)*N + cb5)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb6][cb6]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb6)*N + cb6)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb7][cb7]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb7)*N + cb7)); }
            }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 3;\n");

            const int cs = k_tile % STAGES4;

            uint32_t a_frag[2][4];
            uint32_t b_frag[2][8][2];

            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                const int lm_row = lane_id & 15;
                const int lm_col = ki * MMA_K + ((lane_id >> 4) << 3);
                uint32_t sa = __cvta_generic_to_shared(&smem_A[cs][lm_row][lm_col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[ki][0]),"=r"(a_frag[ki][1]),"=r"(a_frag[ki][2]),"=r"(a_frag[ki][3])
                    : "r"(sa));
            }
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    const int lm_row = ki * MMA_K + (lane_id & 15);
                    const int lm_col = ni * MMA_N;
                    uint32_t sb = __cvta_generic_to_shared(&smem_B[cs][lm_row][lm_col]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(b_frag[ki][ni][0]),"=r"(b_frag[ki][ni][1])
                        : "r"(sb));
                }
            }
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[ni][0]),"+f"(acc[ni][1]),"+f"(acc[ni][2]),"+f"(acc[ni][3])
                        : "r"(a_frag[ki][0]),"r"(a_frag[ki][1]),"r"(a_frag[ki][2]),"r"(a_frag[ki][3]),
                          "r"(b_frag[ki][ni][0]),"r"(b_frag[ki][ni][1]));
                }
            }
        }

        asm volatile("cp.async.wait_all;\n");

        const int r0      = block_row + (lane_id >> 2);
        const int r1      = r0 + 8;
        const int col_off = (lane_id & 3) * 2;

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int c0 = ni * MMA_N + col_off;
            if (r0 < M) *reinterpret_cast<half2*>(&C[r0 * N + c0]) = __floats2half2_rn(acc[ni][0], acc[ni][1]);
            if (r1 < M) *reinterpret_cast<half2*>(&C[r1 * N + c0]) = __floats2half2_rn(acc[ni][2], acc[ni][3]);
        }
    }
}

__global__ void __launch_bounds__(64, 5)
hgemm_bm32_2w_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int N = 64;
    const int num_k_tiles = K / BK32;

    __shared__ half smem_A[STAGES4][32][SA32_S];
    __shared__ half smem_B[STAGES4][BK32][SB32_S];

    for (int block_row = blockIdx.x * 32; block_row < M; block_row += gridDim.x * 32) {

        float acc[8][4];
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        const int ra0 = (tid * 8)       / BK32, ca0 = (tid * 8)       % BK32;
        const int ra1 = (tid * 8 + 512) / BK32, ca1 = (tid * 8 + 512) % BK32;
        const int rb0 = (tid * 8 +    0) / N, cb0 = (tid * 8 +    0) % N;
        const int rb1 = (tid * 8 +  512) / N, cb1 = (tid * 8 +  512) % N;
        const int rb2 = (tid * 8 + 1024) / N, cb2 = (tid * 8 + 1024) % N;
        const int rb3 = (tid * 8 + 1536) / N, cb3 = (tid * 8 + 1536) % N;

        const int gA0 = block_row + ra0;
        const int gA1 = block_row + ra1;

        #pragma unroll
        for (int s = 0; s < STAGES4 - 1; s++) {
            const int k_off = s * BK32;
            { uint32_t sa = __cvta_generic_to_shared(&smem_A[s][ra0][ca0]); if (gA0 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA0*K + k_off + ca0)); else *reinterpret_cast<float4*>(&smem_A[s][ra0][ca0]) = make_float4(0,0,0,0); }
            { uint32_t sa = __cvta_generic_to_shared(&smem_A[s][ra1][ca1]); if (gA1 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA1*K + k_off + ca1)); else *reinterpret_cast<float4*>(&smem_A[s][ra1][ca1]) = make_float4(0,0,0,0); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb0][cb0]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb0)*N + cb0)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb1][cb1]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb1)*N + cb1)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb2][cb2]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb2)*N + cb2)); }
            { uint32_t sb = __cvta_generic_to_shared(&smem_B[s][rb3][cb3]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb3)*N + cb3)); }
            asm volatile("cp.async.commit_group;\n");
        }

        #pragma unroll 1
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            const int pf = k_tile + STAGES4 - 1;
            if (pf < num_k_tiles) {
                const int ps = pf % STAGES4, k_off = pf * BK32;
                { uint32_t sa = __cvta_generic_to_shared(&smem_A[ps][ra0][ca0]); if (gA0 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA0*K + k_off + ca0)); else *reinterpret_cast<float4*>(&smem_A[ps][ra0][ca0]) = make_float4(0,0,0,0); }
                { uint32_t sa = __cvta_generic_to_shared(&smem_A[ps][ra1][ca1]); if (gA1 < M) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(A + gA1*K + k_off + ca1)); else *reinterpret_cast<float4*>(&smem_A[ps][ra1][ca1]) = make_float4(0,0,0,0); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb0][cb0]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb0)*N + cb0)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb1][cb1]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb1)*N + cb1)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb2][cb2]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb2)*N + cb2)); }
                { uint32_t sb = __cvta_generic_to_shared(&smem_B[ps][rb3][cb3]); asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sb), "l"(B + (k_off+rb3)*N + cb3)); }
            }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 3;\n");
            __syncthreads();

            const int cs = k_tile % STAGES4;

            uint32_t a_frag[2][4];
            uint32_t b_frag[2][8][2];

            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                const int lm_row = warp_id * MMA_M + (lane_id & 15);
                const int lm_col = ki * MMA_K + ((lane_id >> 4) << 3);
                uint32_t sa = __cvta_generic_to_shared(&smem_A[cs][lm_row][lm_col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[ki][0]),"=r"(a_frag[ki][1]),"=r"(a_frag[ki][2]),"=r"(a_frag[ki][3])
                    : "r"(sa));
            }
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    const int lm_row = ki * MMA_K + (lane_id & 15);
                    const int lm_col = ni * MMA_N;
                    uint32_t sb = __cvta_generic_to_shared(&smem_B[cs][lm_row][lm_col]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(b_frag[ki][ni][0]),"=r"(b_frag[ki][ni][1])
                        : "r"(sb));
                }
            }
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[ni][0]),"+f"(acc[ni][1]),"+f"(acc[ni][2]),"+f"(acc[ni][3])
                        : "r"(a_frag[ki][0]),"r"(a_frag[ki][1]),"r"(a_frag[ki][2]),"r"(a_frag[ki][3]),
                          "r"(b_frag[ki][ni][0]),"r"(b_frag[ki][ni][1]));
                }
            }
        }

        asm volatile("cp.async.wait_all;\n");
        __syncthreads();

        const int base_row = block_row + warp_id * MMA_M;
        const int r0       = base_row + (lane_id >> 2);
        const int r1       = r0 + 8;
        const int col_off  = (lane_id & 3) * 2;

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int c0 = ni * MMA_N + col_off;
            if (r0 < M) *reinterpret_cast<half2*>(&C[r0 * N + c0]) = __floats2half2_rn(acc[ni][0], acc[ni][1]);
            if (r1 < M) *reinterpret_cast<half2*>(&C[r1 * N + c0]) = __floats2half2_rn(acc[ni][2], acc[ni][3]);
        }
    }
}

static int g_best_kernel = -1;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = a.size(0);
    const int K = a.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half*       C = reinterpret_cast<half*>(c.data_ptr());

    const int SM_COUNT = 132;
    const int grid_bk16 = min(SM_COUNT * 10, (M + BM - 1) / BM);
    const int grid_bk32 = min(SM_COUNT * 8,  (M + BM - 1) / BM);
    const int grid_bm32 = min(SM_COUNT * 5,  (M + 31)     / 32);

    auto launch = [&](int kid, half* Cout) {
        if (kid == 0)
            hgemm_persistent_bk16<<<grid_bk16, 32>>>(A, B, Cout, M, K);
        else if (kid == 1)
            hgemm_bk32_bm16<<<grid_bk32, 32>>>(A, B, Cout, M, K);
        else
            hgemm_bm32_2w_persistent<<<grid_bm32, 64>>>(A, B, Cout, M, K);
    };

    if (g_best_kernel < 0) {
        half* C_tmp = nullptr;
        cudaMalloc(&C_tmp, (size_t)M * 64 * sizeof(half));

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        const int WARMUP = 20, ITERS = 50;
        float best_ms = 1e30f;
        g_best_kernel = 0;

        for (int kid = 0; kid < 3; kid++) {
            for (int i = 0; i < WARMUP; i++) launch(kid, C_tmp);
            cudaDeviceSynchronize();
            cudaEventRecord(ev0);
            for (int i = 0; i < ITERS; i++) launch(kid, C_tmp);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            ms /= ITERS;
            if (ms < best_ms) { best_ms = ms; g_best_kernel = kid; }
        }

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaFree(C_tmp);
    }

    launch(g_best_kernel, C);
}