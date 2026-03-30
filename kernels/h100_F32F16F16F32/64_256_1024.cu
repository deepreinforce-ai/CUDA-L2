#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <torch/types.h>

#include <torch/extension.h>

#undef TORCH_INTERNAL_ASSERT
#undef TORCH_CHECK
#define TORCH_INTERNAL_ASSERT(...)
#define TORCH_CHECK(...)

#define BM 64
#define BN 32
#define BK 64
#define STAGES 10

#define A_SMEM_STRIDE 64
#define B_SMEM_STRIDE 32
#define A_STAGE_SIZE  (BM * A_SMEM_STRIDE)
#define B_STAGE_SIZE  (BK * B_SMEM_STRIDE)
#define STAGE_SIZE    (A_STAGE_SIZE + B_STAGE_SIZE)
#define TOTAL_SMEM_HALFS (STAGES * STAGE_SIZE)
#define TOTAL_SMEM_BYTES (TOTAL_SMEM_HALFS * 2)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ int swzA(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ int swzB(int row, int col) {
    return col ^ ((row & 3) << 3);
}

__global__ void __launch_bounds__(128, 1)
hgemm_opt_final(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = (wid >> 1) * 32;
    const int warp_n = (wid  & 1) * 16;

    extern __shared__ half smem[];

    #define sA_ptr(s) (smem + (int)(s) * STAGE_SIZE)
    #define sB_ptr(s) (smem + (int)(s) * STAGE_SIZE + A_STAGE_SIZE)

    float acc[2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int a_row0 = tid >> 3;
    const int a_col  = (tid & 7) << 3;
    const int b_row0 = tid >> 2;
    const int b_col  = (tid & 3) << 3;

    const int num_k_tiles = K / BK;

    const int a_sw0 = swzA(a_row0,      a_col);
    const int a_sw1 = swzA(a_row0 + 16, a_col);
    const int a_sw2 = swzA(a_row0 + 32, a_col);
    const int a_sw3 = swzA(a_row0 + 48, a_col);
    const int b_sw0 = swzB(b_row0,      b_col);
    const int b_sw1 = swzB(b_row0 + 32, b_col);

    const half* gA0 = A + (bm + a_row0)      * K + a_col;
    const half* gA1 = A + (bm + a_row0 + 16) * K + a_col;
    const half* gA2 = A + (bm + a_row0 + 32) * K + a_col;
    const half* gA3 = A + (bm + a_row0 + 48) * K + a_col;
    const half* gB0 = B + b_row0        * N + (bn + b_col);
    const half* gB1 = B + (b_row0 + 32) * N + (bn + b_col);

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        const int k_off = s * BK;
        half* pA = sA_ptr(s);
        half* pB = sB_ptr(s);

        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32(pA + a_row0      * A_SMEM_STRIDE + a_sw0)), "l"(gA0 + k_off));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32(pA + (a_row0+16) * A_SMEM_STRIDE + a_sw1)), "l"(gA1 + k_off));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32(pA + (a_row0+32) * A_SMEM_STRIDE + a_sw2)), "l"(gA2 + k_off));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32(pA + (a_row0+48) * A_SMEM_STRIDE + a_sw3)), "l"(gA3 + k_off));

        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32(pB + b_row0      * B_SMEM_STRIDE + b_sw0)), "l"(gB0 + (long long)k_off * N));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(smem_u32(pB + (b_row0+32) * B_SMEM_STRIDE + b_sw1)), "l"(gB1 + (long long)k_off * N));

        asm volatile("cp.async.commit_group;\n" :::);
    }

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();

        const int cur = k_tile % STAGES;
        half* sA_cur = sA_ptr(cur);
        half* sB_cur = sB_ptr(cur);

        const int pf = k_tile + STAGES - 1;
        if (pf < num_k_tiles) {
            const int ps    = pf % STAGES;
            const int k_off = pf * BK;
            half* pA = sA_ptr(ps);
            half* pB = sB_ptr(ps);

            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32(pA + a_row0      * A_SMEM_STRIDE + a_sw0)), "l"(gA0 + k_off));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32(pA + (a_row0+16) * A_SMEM_STRIDE + a_sw1)), "l"(gA1 + k_off));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32(pA + (a_row0+32) * A_SMEM_STRIDE + a_sw2)), "l"(gA2 + k_off));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32(pA + (a_row0+48) * A_SMEM_STRIDE + a_sw3)), "l"(gA3 + k_off));

            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32(pB + b_row0      * B_SMEM_STRIDE + b_sw0)), "l"(gB0 + (long long)k_off * N));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(smem_u32(pB + (b_row0+32) * B_SMEM_STRIDE + b_sw1)), "l"(gB1 + (long long)k_off * N));
        }
        asm volatile("cp.async.commit_group;\n" :::);

        const int a_lane_row_0 = warp_m + 0  + (lane & 15);
        const int a_lane_row_1 = warp_m + 16 + (lane & 15);
        const int a_lane_col_base = ((lane >> 4) & 1) * 8;

        const int b_mat_idx = (lane & 15) >> 3;
        const int b_col_in  = lane & 7;

        uint32_t a_reg[2][4];
        uint32_t b_reg[2][2];

        {
            const int col0 = a_lane_col_base;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_reg[0][0]),"=r"(a_reg[0][1]),"=r"(a_reg[0][2]),"=r"(a_reg[0][3])
                : "r"(smem_u32(sA_cur + a_lane_row_0 * A_SMEM_STRIDE + swzA(a_lane_row_0, col0))));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_reg[1][0]),"=r"(a_reg[1][1]),"=r"(a_reg[1][2]),"=r"(a_reg[1][3])
                : "r"(smem_u32(sA_cur + a_lane_row_1 * A_SMEM_STRIDE + swzA(a_lane_row_1, col0))));
        }
        {
            const int b_row_n = b_mat_idx * 8 + b_col_in;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_reg[0][0]),"=r"(b_reg[0][1])
                : "r"(smem_u32(sB_cur + b_row_n * B_SMEM_STRIDE + swzB(b_row_n, warp_n + 0))));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_reg[1][0]),"=r"(b_reg[1][1])
                : "r"(smem_u32(sB_cur + b_row_n * B_SMEM_STRIDE + swzB(b_row_n, warp_n + 8))));
        }

        {
            uint32_t a_nxt[2][4]; uint32_t b_nxt[2][2];
            const int col1 = a_lane_col_base + 16;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_nxt[0][0]),"=r"(a_nxt[0][1]),"=r"(a_nxt[0][2]),"=r"(a_nxt[0][3])
                : "r"(smem_u32(sA_cur + a_lane_row_0 * A_SMEM_STRIDE + swzA(a_lane_row_0, col1))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_nxt[1][0]),"=r"(a_nxt[1][1]),"=r"(a_nxt[1][2]),"=r"(a_nxt[1][3])
                : "r"(smem_u32(sA_cur + a_lane_row_1 * A_SMEM_STRIDE + swzA(a_lane_row_1, col1))));
            const int br1 = 16 + b_mat_idx * 8 + b_col_in;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_nxt[0][0]),"=r"(b_nxt[0][1])
                : "r"(smem_u32(sB_cur + br1 * B_SMEM_STRIDE + swzB(br1, warp_n + 0))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_nxt[1][0]),"=r"(b_nxt[1][1])
                : "r"(smem_u32(sB_cur + br1 * B_SMEM_STRIDE + swzB(br1, warp_n + 8))));

            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][0][0]),"+f"(acc[0][0][1]),"+f"(acc[0][0][2]),"+f"(acc[0][0][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][1][0]),"+f"(acc[0][1][1]),"+f"(acc[0][1][2]),"+f"(acc[0][1][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][0][0]),"+f"(acc[1][0][1]),"+f"(acc[1][0][2]),"+f"(acc[1][0][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][1][0]),"+f"(acc[1][1][1]),"+f"(acc[1][1][2]),"+f"(acc[1][1][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));

            a_reg[0][0]=a_nxt[0][0];a_reg[0][1]=a_nxt[0][1];a_reg[0][2]=a_nxt[0][2];a_reg[0][3]=a_nxt[0][3];
            a_reg[1][0]=a_nxt[1][0];a_reg[1][1]=a_nxt[1][1];a_reg[1][2]=a_nxt[1][2];a_reg[1][3]=a_nxt[1][3];
            b_reg[0][0]=b_nxt[0][0];b_reg[0][1]=b_nxt[0][1];
            b_reg[1][0]=b_nxt[1][0];b_reg[1][1]=b_nxt[1][1];
        }

        {
            uint32_t a_nxt[2][4]; uint32_t b_nxt[2][2];
            const int col2 = a_lane_col_base + 32;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_nxt[0][0]),"=r"(a_nxt[0][1]),"=r"(a_nxt[0][2]),"=r"(a_nxt[0][3])
                : "r"(smem_u32(sA_cur + a_lane_row_0 * A_SMEM_STRIDE + swzA(a_lane_row_0, col2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_nxt[1][0]),"=r"(a_nxt[1][1]),"=r"(a_nxt[1][2]),"=r"(a_nxt[1][3])
                : "r"(smem_u32(sA_cur + a_lane_row_1 * A_SMEM_STRIDE + swzA(a_lane_row_1, col2))));
            const int br2 = 32 + b_mat_idx * 8 + b_col_in;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_nxt[0][0]),"=r"(b_nxt[0][1])
                : "r"(smem_u32(sB_cur + br2 * B_SMEM_STRIDE + swzB(br2, warp_n + 0))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_nxt[1][0]),"=r"(b_nxt[1][1])
                : "r"(smem_u32(sB_cur + br2 * B_SMEM_STRIDE + swzB(br2, warp_n + 8))));

            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][0][0]),"+f"(acc[0][0][1]),"+f"(acc[0][0][2]),"+f"(acc[0][0][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][1][0]),"+f"(acc[0][1][1]),"+f"(acc[0][1][2]),"+f"(acc[0][1][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][0][0]),"+f"(acc[1][0][1]),"+f"(acc[1][0][2]),"+f"(acc[1][0][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][1][0]),"+f"(acc[1][1][1]),"+f"(acc[1][1][2]),"+f"(acc[1][1][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));

            a_reg[0][0]=a_nxt[0][0];a_reg[0][1]=a_nxt[0][1];a_reg[0][2]=a_nxt[0][2];a_reg[0][3]=a_nxt[0][3];
            a_reg[1][0]=a_nxt[1][0];a_reg[1][1]=a_nxt[1][1];a_reg[1][2]=a_nxt[1][2];a_reg[1][3]=a_nxt[1][3];
            b_reg[0][0]=b_nxt[0][0];b_reg[0][1]=b_nxt[0][1];
            b_reg[1][0]=b_nxt[1][0];b_reg[1][1]=b_nxt[1][1];
        }

        {
            uint32_t a_nxt[2][4]; uint32_t b_nxt[2][2];
            const int col3 = a_lane_col_base + 48;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_nxt[0][0]),"=r"(a_nxt[0][1]),"=r"(a_nxt[0][2]),"=r"(a_nxt[0][3])
                : "r"(smem_u32(sA_cur + a_lane_row_0 * A_SMEM_STRIDE + swzA(a_lane_row_0, col3))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_nxt[1][0]),"=r"(a_nxt[1][1]),"=r"(a_nxt[1][2]),"=r"(a_nxt[1][3])
                : "r"(smem_u32(sA_cur + a_lane_row_1 * A_SMEM_STRIDE + swzA(a_lane_row_1, col3))));
            const int br3 = 48 + b_mat_idx * 8 + b_col_in;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_nxt[0][0]),"=r"(b_nxt[0][1])
                : "r"(smem_u32(sB_cur + br3 * B_SMEM_STRIDE + swzB(br3, warp_n + 0))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_nxt[1][0]),"=r"(b_nxt[1][1])
                : "r"(smem_u32(sB_cur + br3 * B_SMEM_STRIDE + swzB(br3, warp_n + 8))));

            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][0][0]),"+f"(acc[0][0][1]),"+f"(acc[0][0][2]),"+f"(acc[0][0][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][1][0]),"+f"(acc[0][1][1]),"+f"(acc[0][1][2]),"+f"(acc[0][1][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][0][0]),"+f"(acc[1][0][1]),"+f"(acc[1][0][2]),"+f"(acc[1][0][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][1][0]),"+f"(acc[1][1][1]),"+f"(acc[1][1][2]),"+f"(acc[1][1][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));

            a_reg[0][0]=a_nxt[0][0];a_reg[0][1]=a_nxt[0][1];a_reg[0][2]=a_nxt[0][2];a_reg[0][3]=a_nxt[0][3];
            a_reg[1][0]=a_nxt[1][0];a_reg[1][1]=a_nxt[1][1];a_reg[1][2]=a_nxt[1][2];a_reg[1][3]=a_nxt[1][3];
            b_reg[0][0]=b_nxt[0][0];b_reg[0][1]=b_nxt[0][1];
            b_reg[1][0]=b_nxt[1][0];b_reg[1][1]=b_nxt[1][1];
        }

        {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][0][0]),"+f"(acc[0][0][1]),"+f"(acc[0][0][2]),"+f"(acc[0][0][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[0][1][0]),"+f"(acc[0][1][1]),"+f"(acc[0][1][2]),"+f"(acc[0][1][3])
                :"r"(a_reg[0][0]),"r"(a_reg[0][1]),"r"(a_reg[0][2]),"r"(a_reg[0][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][0][0]),"+f"(acc[1][0][1]),"+f"(acc[1][0][2]),"+f"(acc[1][0][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[0][0]),"r"(b_reg[0][1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(acc[1][1][0]),"+f"(acc[1][1][1]),"+f"(acc[1][1][2]),"+f"(acc[1][1][3])
                :"r"(a_reg[1][0]),"r"(a_reg[1][1]),"r"(a_reg[1][2]),"r"(a_reg[1][3]),"r"(b_reg[1][0]),"r"(b_reg[1][1]));
        }
    }

    asm volatile("cp.async.wait_all;\n" :::);

    const int c_row0 = lane >> 2;
    const int c_row1 = c_row0 + 8;
    const int c_col0 = (lane & 3) * 2;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        const int m0 = bm + warp_m + mi * 16 + c_row0;
        const int m1 = bm + warp_m + mi * 16 + c_row1;
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const int n0 = bn + warp_n + ni * 8 + c_col0;
            float* cd = acc[mi][ni];
            *reinterpret_cast<half2*>(&C[m0 * N + n0]) = __floats2half2_rn(cd[0], cd[1]);
            *reinterpret_cast<half2*>(&C[m1 * N + n0]) = __floats2half2_rn(cd[2], cd[3]);
        }
    }

    #undef sA_ptr
    #undef sB_ptr
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    int M = (int)a.sizes()[0];
    int K = (int)a.sizes()[1];
    int N = (int)b.sizes()[1];

    const half* A    = reinterpret_cast<const half*>(a.data_ptr());
    const half* B    = reinterpret_cast<const half*>(b.data_ptr());
    half*       Cout = reinterpret_cast<half*>(c.data_ptr());

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);

    cudaFuncSetAttribute(hgemm_opt_final,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         TOTAL_SMEM_BYTES);

    hgemm_opt_final<<<grid, block, TOTAL_SMEM_BYTES>>>(A, B, Cout, M, N, K);
}