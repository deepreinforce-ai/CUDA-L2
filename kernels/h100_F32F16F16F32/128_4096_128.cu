#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <cstdint>
#include <cuda.h>

using namespace nvcuda;

#define SMEM_PTR(x) ((uint32_t)__cvta_generic_to_shared(x))

#define CP_ASYNC_CG_16(dst, src) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
        :: "r"(SMEM_PTR(dst)), "l"((const void*)(src)) : "memory")

#define CP_ASYNC_COMMIT() \
    asm volatile("cp.async.commit_group;\n" ::: "memory")

#define CP_ASYNC_WAIT(n) \
    asm volatile("cp.async.wait_group " #n ";\n" ::: "memory")

#define CP_ASYNC_WAIT_ALL() \
    asm volatile("cp.async.wait_all;\n" ::: "memory")

__device__ __forceinline__
void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

__device__ __forceinline__ int swizzle_col_bn32(int row, int col) {
    int chunk = col >> 3;
    int intra = col & 7;
    int swz_chunk = chunk ^ (row & 3);
    return (swz_chunk << 3) | intra;
}

__device__ __forceinline__ int swizzle_col_bn64(int row, int col) {
    int chunk = col >> 3;
    int intra = col & 7;
    int swz_chunk = chunk ^ (row & 7);
    return (swz_chunk << 3) | intra;
}

#define P1_BM    128
#define P1_BN    32
#define P1_BK    32
#define P1_TNUM  128
#define P1_PAD_A 8
#define P1_PAD_B 8
#define P1_STGS  3
#define P1_TM    2
#define P1_TN    4

__global__ __launch_bounds__(P1_TNUM, 4)
void hgemm_ptx_bn32_swz_rdb(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int gm = bm * P1_BM;
    const int gn = bn * P1_BN;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int warp_row_base = warp_id * 32;

    __shared__ __align__(128) half sA[P1_STGS][P1_BM][P1_BK + P1_PAD_A];
    __shared__ __align__(128) half sB[P1_STGS][P1_BK][P1_BN + P1_PAD_B];

    float acc[P1_TM][P1_TN][4];
    #pragma unroll
    for (int mi = 0; mi < P1_TM; mi++)
        #pragma unroll
        for (int ni = 0; ni < P1_TN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    const int a_row = tid;
    const int b_row = tid >> 2;
    const int b_col = (tid & 3) << 3;

    const int nkb = K / P1_BK;

    CP_ASYNC_CG_16(&sA[0][a_row][0],  A + (gm + a_row) * K + 0);
    CP_ASYNC_CG_16(&sA[0][a_row][8],  A + (gm + a_row) * K + 8);
    CP_ASYNC_CG_16(&sA[0][a_row][16], A + (gm + a_row) * K + 16);
    CP_ASYNC_CG_16(&sA[0][a_row][24], A + (gm + a_row) * K + 24);
    CP_ASYNC_CG_16(&sB[0][b_row][b_col], B + b_row * N + gn + b_col);
    CP_ASYNC_COMMIT();

    if (nkb > 1) {
        int k1 = P1_BK;
        CP_ASYNC_CG_16(&sA[1][a_row][0],  A + (gm + a_row) * K + k1);
        CP_ASYNC_CG_16(&sA[1][a_row][8],  A + (gm + a_row) * K + k1 + 8);
        CP_ASYNC_CG_16(&sA[1][a_row][16], A + (gm + a_row) * K + k1 + 16);
        CP_ASYNC_CG_16(&sA[1][a_row][24], A + (gm + a_row) * K + k1 + 24);
        CP_ASYNC_CG_16(&sB[1][b_row][b_col], B + (k1 + b_row) * N + gn + b_col);
        CP_ASYNC_COMMIT();
    }

    CP_ASYNC_WAIT(1);
    __syncthreads();

    #pragma unroll
    for (int kb = 0; kb < nkb; kb++) {
        const int cs = kb % P1_STGS;

        if (kb + 2 < nkb) {
            int ps = (kb + 2) % P1_STGS;
            int pk = (kb + 2) * P1_BK;
            CP_ASYNC_CG_16(&sA[ps][a_row][0],  A + (gm + a_row) * K + pk);
            CP_ASYNC_CG_16(&sA[ps][a_row][8],  A + (gm + a_row) * K + pk + 8);
            CP_ASYNC_CG_16(&sA[ps][a_row][16], A + (gm + a_row) * K + pk + 16);
            CP_ASYNC_CG_16(&sA[ps][a_row][24], A + (gm + a_row) * K + pk + 24);
            CP_ASYNC_CG_16(&sB[ps][b_row][b_col], B + (pk + b_row) * N + gn + b_col);
            CP_ASYNC_COMMIT();
        }

        uint32_t ra0_0, ra1_0, ra2_0, ra3_0;
        uint32_t ra4_0, ra5_0, ra6_0, ra7_0;
        uint32_t rb0_0, rb1_0, rb2_0, rb3_0;
        uint32_t rb4_0, rb5_0, rb6_0, rb7_0;

        uint32_t ra0_1, ra1_1, ra2_1, ra3_1;
        uint32_t ra4_1, ra5_1, ra6_1, ra7_1;
        uint32_t rb0_1, rb1_1, rb2_1, rb3_1;
        uint32_t rb4_1, rb5_1, rb6_1, rb7_1;

        {
            int row = warp_row_base + (lane_id & 15);
            int col = (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(ra0_0),"=r"(ra1_0),"=r"(ra2_0),"=r"(ra3_0)
                : "r"(SMEM_PTR(&sA[cs][row][col])));
        }
        {
            int row = warp_row_base + 16 + (lane_id & 15);
            int col = (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(ra4_0),"=r"(ra5_0),"=r"(ra6_0),"=r"(ra7_0)
                : "r"(SMEM_PTR(&sA[cs][row][col])));
        }
        {
            int row = (lane_id & 15);
            int col = (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(rb0_0),"=r"(rb1_0),"=r"(rb2_0),"=r"(rb3_0)
                : "r"(SMEM_PTR(&sB[cs][row][col])));
        }
        {
            int row = (lane_id & 15);
            int col = 16 + (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(rb4_0),"=r"(rb5_0),"=r"(rb6_0),"=r"(rb7_0)
                : "r"(SMEM_PTR(&sB[cs][row][col])));
        }

        {
            int row = warp_row_base + (lane_id & 15);
            int col = 16 + (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(ra0_1),"=r"(ra1_1),"=r"(ra2_1),"=r"(ra3_1)
                : "r"(SMEM_PTR(&sA[cs][row][col])));
        }
        {
            int row = warp_row_base + 16 + (lane_id & 15);
            int col = 16 + (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(ra4_1),"=r"(ra5_1),"=r"(ra6_1),"=r"(ra7_1)
                : "r"(SMEM_PTR(&sA[cs][row][col])));
        }
        {
            int row = 16 + (lane_id & 15);
            int col = (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(rb0_1),"=r"(rb1_1),"=r"(rb2_1),"=r"(rb3_1)
                : "r"(SMEM_PTR(&sB[cs][row][col])));
        }
        {
            int row = 16 + (lane_id & 15);
            int col = 16 + (lane_id >> 4) * 8;
            asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(rb4_1),"=r"(rb5_1),"=r"(rb6_1),"=r"(rb7_1)
                : "r"(SMEM_PTR(&sB[cs][row][col])));
        }

        mma_m16n8k16(acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3], ra0_0,ra1_0,ra2_0,ra3_0, rb0_0,rb1_0, acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3]);
        mma_m16n8k16(acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3], ra0_0,ra1_0,ra2_0,ra3_0, rb2_0,rb3_0, acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3]);
        mma_m16n8k16(acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3], ra0_0,ra1_0,ra2_0,ra3_0, rb4_0,rb5_0, acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3]);
        mma_m16n8k16(acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3], ra0_0,ra1_0,ra2_0,ra3_0, rb6_0,rb7_0, acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3]);
        mma_m16n8k16(acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3], ra4_0,ra5_0,ra6_0,ra7_0, rb0_0,rb1_0, acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3]);
        mma_m16n8k16(acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3], ra4_0,ra5_0,ra6_0,ra7_0, rb2_0,rb3_0, acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3]);
        mma_m16n8k16(acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3], ra4_0,ra5_0,ra6_0,ra7_0, rb4_0,rb5_0, acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3]);
        mma_m16n8k16(acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3], ra4_0,ra5_0,ra6_0,ra7_0, rb6_0,rb7_0, acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3]);

        mma_m16n8k16(acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3], ra0_1,ra1_1,ra2_1,ra3_1, rb0_1,rb1_1, acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3]);
        mma_m16n8k16(acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3], ra0_1,ra1_1,ra2_1,ra3_1, rb2_1,rb3_1, acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3]);
        mma_m16n8k16(acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3], ra0_1,ra1_1,ra2_1,ra3_1, rb4_1,rb5_1, acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3]);
        mma_m16n8k16(acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3], ra0_1,ra1_1,ra2_1,ra3_1, rb6_1,rb7_1, acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3]);
        mma_m16n8k16(acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3], ra4_1,ra5_1,ra6_1,ra7_1, rb0_1,rb1_1, acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3]);
        mma_m16n8k16(acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3], ra4_1,ra5_1,ra6_1,ra7_1, rb2_1,rb3_1, acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3]);
        mma_m16n8k16(acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3], ra4_1,ra5_1,ra6_1,ra7_1, rb4_1,rb5_1, acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3]);
        mma_m16n8k16(acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3], ra4_1,ra5_1,ra6_1,ra7_1, rb6_1,rb7_1, acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3]);

        if (kb + 1 < nkb) {
            if (kb + 2 < nkb) CP_ASYNC_WAIT(1);
            else               CP_ASYNC_WAIT_ALL();
            __syncthreads();
        }
    }

    const int mma_r = lane_id >> 2;
    const int mma_c = (lane_id & 3) << 1;
    #pragma unroll
    for (int mi = 0; mi < P1_TM; mi++) {
        const int row_base = gm + warp_row_base + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < P1_TN; ni++) {
            const int col_base = gn + ni * 8;
            int r0 = row_base + mma_r;
            int r1 = r0 + 8;
            int c0 = col_base + mma_c;
            *reinterpret_cast<__half2*>(C + r0 * N + c0) =
                __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<__half2*>(C + r1 * N + c0) =
                __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

#define P2_BM    128
#define P2_BN    64
#define P2_BK    32
#define P2_WM    4
#define P2_WN    2
#define P2_TNUM  (P2_WM * P2_WN * 32)
#define P2_PAD_A 8
#define P2_PAD_B 8
#define P2_STGS  3
#define P2_TM    2
#define P2_TN    4

__global__ __launch_bounds__(P2_TNUM, 2)
void hgemm_ptx_bn64_swz_rdb(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y, bn = blockIdx.x;
    const int gm = bm * P2_BM, gn = bn * P2_BN;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int warp_m = warp_id / P2_WN;
    const int warp_n = warp_id % P2_WN;
    const int warp_row_base = warp_m * 32;
    const int warp_col_base = warp_n * 32;

    __shared__ __align__(128) half sA[P2_STGS][P2_BM][P2_BK + P2_PAD_A];
    __shared__ __align__(128) half sB[P2_STGS][P2_BK][P2_BN + P2_PAD_B];

    float acc[P2_TM][P2_TN][4];
    #pragma unroll
    for (int mi = 0; mi < P2_TM; mi++)
        #pragma unroll
        for (int ni = 0; ni < P2_TN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    const int a_row0 = tid >> 2, a_col = (tid & 3) << 3, a_row1 = a_row0 + 64;
    const int b_row  = tid >> 3, b_col = (tid & 7) << 3;
    const int nkb = K / P2_BK;

    CP_ASYNC_CG_16(&sA[0][a_row0][a_col], A + (gm + a_row0) * K + a_col);
    CP_ASYNC_CG_16(&sA[0][a_row1][a_col], A + (gm + a_row1) * K + a_col);
    CP_ASYNC_CG_16(&sB[0][b_row][b_col],  B + b_row * N + gn + b_col);
    CP_ASYNC_COMMIT();

    if (nkb > 1) {
        int k1 = P2_BK;
        CP_ASYNC_CG_16(&sA[1][a_row0][a_col], A + (gm + a_row0) * K + k1 + a_col);
        CP_ASYNC_CG_16(&sA[1][a_row1][a_col], A + (gm + a_row1) * K + k1 + a_col);
        CP_ASYNC_CG_16(&sB[1][b_row][b_col],  B + (k1 + b_row) * N + gn + b_col);
        CP_ASYNC_COMMIT();
    }
    CP_ASYNC_WAIT(1);
    __syncthreads();

    #pragma unroll
    for (int kb = 0; kb < nkb; kb++) {
        const int cs = kb % P2_STGS;

        if (kb + 2 < nkb) {
            int ps = (kb + 2) % P2_STGS, pk = (kb + 2) * P2_BK;
            CP_ASYNC_CG_16(&sA[ps][a_row0][a_col], A + (gm + a_row0) * K + pk + a_col);
            CP_ASYNC_CG_16(&sA[ps][a_row1][a_col], A + (gm + a_row1) * K + pk + a_col);
            CP_ASYNC_CG_16(&sB[ps][b_row][b_col],  B + (pk + b_row) * N + gn + b_col);
            CP_ASYNC_COMMIT();
        }

        uint32_t ra0_0,ra1_0,ra2_0,ra3_0, ra4_0,ra5_0,ra6_0,ra7_0;
        uint32_t rb0_0,rb1_0,rb2_0,rb3_0, rb4_0,rb5_0,rb6_0,rb7_0;
        uint32_t ra0_1,ra1_1,ra2_1,ra3_1, ra4_1,ra5_1,ra6_1,ra7_1;
        uint32_t rb0_1,rb1_1,rb2_1,rb3_1, rb4_1,rb5_1,rb6_1,rb7_1;

        { int r=warp_row_base+(lane_id&15);    int c=(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(ra0_0),"=r"(ra1_0),"=r"(ra2_0),"=r"(ra3_0) : "r"(SMEM_PTR(&sA[cs][r][c]))); }
        { int r=warp_row_base+16+(lane_id&15); int c=(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(ra4_0),"=r"(ra5_0),"=r"(ra6_0),"=r"(ra7_0) : "r"(SMEM_PTR(&sA[cs][r][c]))); }
        { int r=(lane_id&15); int c=warp_col_base+(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(rb0_0),"=r"(rb1_0),"=r"(rb2_0),"=r"(rb3_0) : "r"(SMEM_PTR(&sB[cs][r][c]))); }
        { int r=(lane_id&15); int c=warp_col_base+16+(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(rb4_0),"=r"(rb5_0),"=r"(rb6_0),"=r"(rb7_0) : "r"(SMEM_PTR(&sB[cs][r][c]))); }

        { int r=warp_row_base+(lane_id&15);    int c=16+(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(ra0_1),"=r"(ra1_1),"=r"(ra2_1),"=r"(ra3_1) : "r"(SMEM_PTR(&sA[cs][r][c]))); }
        { int r=warp_row_base+16+(lane_id&15); int c=16+(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(ra4_1),"=r"(ra5_1),"=r"(ra6_1),"=r"(ra7_1) : "r"(SMEM_PTR(&sA[cs][r][c]))); }
        { int r=16+(lane_id&15); int c=warp_col_base+(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(rb0_1),"=r"(rb1_1),"=r"(rb2_1),"=r"(rb3_1) : "r"(SMEM_PTR(&sB[cs][r][c]))); }
        { int r=16+(lane_id&15); int c=warp_col_base+16+(lane_id>>4)*8;
          asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(rb4_1),"=r"(rb5_1),"=r"(rb6_1),"=r"(rb7_1) : "r"(SMEM_PTR(&sB[cs][r][c]))); }

        mma_m16n8k16(acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3], ra0_0,ra1_0,ra2_0,ra3_0, rb0_0,rb1_0, acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3]);
        mma_m16n8k16(acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3], ra0_0,ra1_0,ra2_0,ra3_0, rb2_0,rb3_0, acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3]);
        mma_m16n8k16(acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3], ra0_0,ra1_0,ra2_0,ra3_0, rb4_0,rb5_0, acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3]);
        mma_m16n8k16(acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3], ra0_0,ra1_0,ra2_0,ra3_0, rb6_0,rb7_0, acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3]);
        mma_m16n8k16(acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3], ra4_0,ra5_0,ra6_0,ra7_0, rb0_0,rb1_0, acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3]);
        mma_m16n8k16(acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3], ra4_0,ra5_0,ra6_0,ra7_0, rb2_0,rb3_0, acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3]);
        mma_m16n8k16(acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3], ra4_0,ra5_0,ra6_0,ra7_0, rb4_0,rb5_0, acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3]);
        mma_m16n8k16(acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3], ra4_0,ra5_0,ra6_0,ra7_0, rb6_0,rb7_0, acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3]);

        mma_m16n8k16(acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3], ra0_1,ra1_1,ra2_1,ra3_1, rb0_1,rb1_1, acc[0][0][0],acc[0][0][1],acc[0][0][2],acc[0][0][3]);
        mma_m16n8k16(acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3], ra0_1,ra1_1,ra2_1,ra3_1, rb2_1,rb3_1, acc[0][1][0],acc[0][1][1],acc[0][1][2],acc[0][1][3]);
        mma_m16n8k16(acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3], ra0_1,ra1_1,ra2_1,ra3_1, rb4_1,rb5_1, acc[0][2][0],acc[0][2][1],acc[0][2][2],acc[0][2][3]);
        mma_m16n8k16(acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3], ra0_1,ra1_1,ra2_1,ra3_1, rb6_1,rb7_1, acc[0][3][0],acc[0][3][1],acc[0][3][2],acc[0][3][3]);
        mma_m16n8k16(acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3], ra4_1,ra5_1,ra6_1,ra7_1, rb0_1,rb1_1, acc[1][0][0],acc[1][0][1],acc[1][0][2],acc[1][0][3]);
        mma_m16n8k16(acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3], ra4_1,ra5_1,ra6_1,ra7_1, rb2_1,rb3_1, acc[1][1][0],acc[1][1][1],acc[1][1][2],acc[1][1][3]);
        mma_m16n8k16(acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3], ra4_1,ra5_1,ra6_1,ra7_1, rb4_1,rb5_1, acc[1][2][0],acc[1][2][1],acc[1][2][2],acc[1][2][3]);
        mma_m16n8k16(acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3], ra4_1,ra5_1,ra6_1,ra7_1, rb6_1,rb7_1, acc[1][3][0],acc[1][3][1],acc[1][3][2],acc[1][3][3]);

        if (kb + 1 < nkb) {
            if (kb + 2 < nkb) CP_ASYNC_WAIT(1);
            else               CP_ASYNC_WAIT_ALL();
            __syncthreads();
        }
    }

    const int mma_r = lane_id >> 2;
    const int mma_c = (lane_id & 3) << 1;
    #pragma unroll
    for (int mi = 0; mi < P2_TM; mi++) {
        const int row_base = gm + warp_row_base + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < P2_TN; ni++) {
            const int col_base = gn + warp_col_base + ni * 8;
            int r0 = row_base + mma_r;
            int r1 = r0 + 8;
            int c0 = col_base + mma_c;
            *reinterpret_cast<__half2*>(C + r0 * N + c0) =
                __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<__half2*>(C + r1 * N + c0) =
                __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

#define P3_BM    128
#define P3_BN    64
#define P3_BK    32
#define P3_WM    4
#define P3_WN    2
#define P3_TNUM  (P3_WM * P3_WN * 32)
#define P3_PAD   8
#define P3_STGS  3
#define P3_TM    2
#define P3_TN    2

__global__ __launch_bounds__(P3_TNUM, 2)
void hgemm_wmma_bn64_3stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bn = blockIdx.x, bm = blockIdx.y;
    const int gn = bn * P3_BN, gm = bm * P3_BM;
    const int warp_id = threadIdx.x >> 5;
    const int warp_m  = warp_id / P3_WN;
    const int warp_n  = warp_id % P3_WN;
    const int tid     = threadIdx.x;

    __shared__ __align__(128) half sA[P3_STGS][P3_BM][P3_BK + P3_PAD];
    __shared__ __align__(128) half sB[P3_STGS][P3_BK][P3_BN + P3_PAD];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[P3_TM][P3_TN];
    #pragma unroll
    for (int i = 0; i < P3_TM; i++)
        #pragma unroll
        for (int j = 0; j < P3_TN; j++)
            wmma::fill_fragment(fc[i][j], 0.f);

    const int ar0 = tid >> 2, ac = (tid & 3) << 3, ar1 = ar0 + 64;
    const int br  = tid >> 3, bc = (tid & 7) << 3;
    const int nkb = K / P3_BK;

    CP_ASYNC_CG_16(&sA[0][ar0][ac], A + (gm+ar0)*K + ac);
    CP_ASYNC_CG_16(&sA[0][ar1][ac], A + (gm+ar1)*K + ac);
    CP_ASYNC_CG_16(&sB[0][br][bc],  B + br*N + gn+bc);
    CP_ASYNC_COMMIT();
    if (nkb > 1) {
        CP_ASYNC_CG_16(&sA[1][ar0][ac], A + (gm+ar0)*K + P3_BK+ac);
        CP_ASYNC_CG_16(&sA[1][ar1][ac], A + (gm+ar1)*K + P3_BK+ac);
        CP_ASYNC_CG_16(&sB[1][br][bc],  B + (P3_BK+br)*N + gn+bc);
        CP_ASYNC_COMMIT();
    }
    CP_ASYNC_WAIT(1);
    __syncthreads();

    #pragma unroll 4
    for (int kb = 0; kb < nkb; kb++) {
        int cs = kb % P3_STGS;
        if (kb + 2 < nkb) {
            int ps = (kb+2) % P3_STGS, pk = (kb+2)*P3_BK;
            CP_ASYNC_CG_16(&sA[ps][ar0][ac], A + (gm+ar0)*K + pk+ac);
            CP_ASYNC_CG_16(&sA[ps][ar1][ac], A + (gm+ar1)*K + pk+ac);
            CP_ASYNC_CG_16(&sB[ps][br][bc],  B + (pk+br)*N + gn+bc);
            CP_ASYNC_COMMIT();
        }

        wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> fa[P3_TM];
        wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> fb[P3_TN];
        #pragma unroll
        for (int wk = 0; wk < P3_BK/16; wk++) {
            #pragma unroll
            for (int ti = 0; ti < P3_TM; ti++)
                wmma::load_matrix_sync(fa[ti], &sA[cs][warp_m*P3_TM*16+ti*16][wk*16], P3_BK+P3_PAD);
            #pragma unroll
            for (int tj = 0; tj < P3_TN; tj++)
                wmma::load_matrix_sync(fb[tj], &sB[cs][wk*16][warp_n*P3_TN*16+tj*16], P3_BN+P3_PAD);
            #pragma unroll
            for (int ti = 0; ti < P3_TM; ti++)
                #pragma unroll
                for (int tj = 0; tj < P3_TN; tj++)
                    wmma::mma_sync(fc[ti][tj], fa[ti], fb[tj], fc[ti][tj]);
        }

        if (kb+1 < nkb) {
            if (kb+2 < nkb) CP_ASYNC_WAIT(1);
            else             CP_ASYNC_WAIT_ALL();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int ti = 0; ti < P3_TM; ti++)
        #pragma unroll
        for (int tj = 0; tj < P3_TN; tj++) {
            int or_ = gm+warp_m*P3_TM*16+ti*16, oc = gn+warp_n*P3_TN*16+tj*16;
            wmma::fragment<wmma::accumulator, 16,16,16, half> fo;
            #pragma unroll
            for (int i = 0; i < fc[ti][tj].num_elements; i++)
                fo.x[i] = __float2half(fc[ti][tj].x[i]);
            wmma::store_matrix_sync(C + or_*N + oc, fo, N, wmma::mem_row_major);
        }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    TORCH_CHECK(a.dtype() == torch::kHalf);
    TORCH_CHECK(b.dtype() == torch::kHalf);
    TORCH_CHECK(b_col_major.dtype() == torch::kHalf);
    TORCH_CHECK(c.dtype() == torch::kHalf);

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const half* pA = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* pB = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       pC = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (M % P1_BM == 0 && N % P1_BN == 0 && K % P1_BK == 0) {
        dim3 grid(N / P1_BN, M / P1_BM);
        hgemm_ptx_bn32_swz_rdb<<<grid, P1_TNUM>>>(pA, pB, pC, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    if (M % P2_BM == 0 && N % P2_BN == 0 && K % P2_BK == 0) {
        dim3 grid(N / P2_BN, M / P2_BM);
        hgemm_ptx_bn64_swz_rdb<<<grid, P2_TNUM>>>(pA, pB, pC, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid(N / P3_BN, M / P3_BM);
        hgemm_wmma_bn64_3stage<<<grid, P3_TNUM>>>(pA, pB, pC, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}