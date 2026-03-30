#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

static __device__ __forceinline__
uint32_t cvt_smem(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__
int swA(int row, int col) {
    return (((col >> 3) ^ ((row >> 1) & 3)) << 3) | (col & 7);
}

static __device__ __forceinline__
int swB(int row, int col) {
    return (((col >> 3) ^ ((row >> 1) & 3)) << 3) | (col & 7);
}

__global__ void __launch_bounds__(128, 8)
hgemm_optimized_v13(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_row = blockIdx.y * 64;
    const int block_col = blockIdx.x * 32;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m_base = warp_row * 32;
    const int warp_n_base = warp_col * 16;

    __shared__ half smemA[4][64][40];
    __shared__ half smemB[4][32][40];

    float acc00[4] = {0,0,0,0};
    float acc01[4] = {0,0,0,0};
    float acc10[4] = {0,0,0,0};
    float acc11[4] = {0,0,0,0};

#define ASYNC_LDA(k_start, buf) \
{ \
    int _r = tid >> 1; \
    int _c0 = (tid & 1) << 4; \
    const half* _src = A + (block_row + _r) * K + (k_start) + _c0; \
    { uint32_t _addr = cvt_smem(&smemA[buf][_r][swA(_r, _c0)]); \
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
          :: "r"(_addr), "l"((const void*)_src) : "memory"); } \
    { uint32_t _addr = cvt_smem(&smemA[buf][_r][swA(_r, _c0+8)]); \
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
          :: "r"(_addr), "l"((const void*)(_src+8)) : "memory"); } \
}

#define ASYNC_LDB(k_start, buf) \
{ \
    int _r = tid >> 2; \
    int _c = (tid & 3) << 3; \
    const half* _src = B + ((k_start) + _r) * N + block_col + _c; \
    uint32_t _addr = cvt_smem(&smemB[buf][_r][swB(_r, _c)]); \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" \
        :: "r"(_addr), "l"((const void*)_src) : "memory"); \
}

#define COMMIT asm volatile("cp.async.commit_group;\n")
#define WAIT2  asm volatile("cp.async.wait_group 2;\n" ::: "memory")
#define WAIT1  asm volatile("cp.async.wait_group 1;\n" ::: "memory")
#define WAIT0  asm volatile("cp.async.wait_group 0;\n" ::: "memory")

#define DECL_REGS() \
    uint32_t ra0c0[4], ra0c1[4]; \
    uint32_t ra1c0[4], ra1c1[4]; \
    uint32_t rb0c0[2], rb0c1[2]; \
    uint32_t rb1c0[2], rb1c1[2]; \
    uint32_t ra0n0[4], ra0n1[4]; \
    uint32_t ra1n0[4], ra1n1[4]; \
    uint32_t rb0n0[2], rb0n1[2]; \
    uint32_t rb1n0[2], rb1n1[2];

#define LDA4(buf, rbase, koff, d0,d1,d2,d3) \
{ \
    int _r = (rbase) + (lane_id & 15); \
    int _c = (koff) + (((lane_id>>4)&1)<<3); \
    uint32_t _a = cvt_smem(&smemA[buf][_r][swA(_r,_c)]); \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n" \
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(_a)); \
}

#define LDB2T(buf, koff, cbase, d0,d1) \
{ \
    int _r = (koff) + (lane_id & 15); \
    uint32_t _a = cvt_smem(&smemB[buf][_r][swB(_r, cbase)]); \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n" \
        :"=r"(d0),"=r"(d1):"r"(_a)); \
}

#define LOAD_NXT_KM0(nxt_buf) \
{ \
    LDA4(nxt_buf, warp_m_base,    0,  ra0n0[0],ra0n0[1],ra0n0[2],ra0n0[3]); \
    LDA4(nxt_buf, warp_m_base+16, 0,  ra0n1[0],ra0n1[1],ra0n1[2],ra0n1[3]); \
    LDB2T(nxt_buf, 0,  warp_n_base,   rb0n0[0],rb0n0[1]); \
    LDB2T(nxt_buf, 0,  warp_n_base+8, rb0n1[0],rb0n1[1]); \
}

#define LOAD_NXT_KM1(nxt_buf) \
{ \
    LDA4(nxt_buf, warp_m_base,    16, ra1n0[0],ra1n0[1],ra1n0[2],ra1n0[3]); \
    LDA4(nxt_buf, warp_m_base+16, 16, ra1n1[0],ra1n1[1],ra1n1[2],ra1n1[3]); \
    LDB2T(nxt_buf, 16, warp_n_base,   rb1n0[0],rb1n0[1]); \
    LDB2T(nxt_buf, 16, warp_n_base+8, rb1n1[0],rb1n1[1]); \
}

#define MMA_KM0_CUR() \
{ \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc00[0]),"+f"(acc00[1]),"+f"(acc00[2]),"+f"(acc00[3]) \
        :"r"(ra0c0[0]),"r"(ra0c0[1]),"r"(ra0c0[2]),"r"(ra0c0[3]),"r"(rb0c0[0]),"r"(rb0c0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc01[0]),"+f"(acc01[1]),"+f"(acc01[2]),"+f"(acc01[3]) \
        :"r"(ra0c0[0]),"r"(ra0c0[1]),"r"(ra0c0[2]),"r"(ra0c0[3]),"r"(rb0c1[0]),"r"(rb0c1[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc10[0]),"+f"(acc10[1]),"+f"(acc10[2]),"+f"(acc10[3]) \
        :"r"(ra0c1[0]),"r"(ra0c1[1]),"r"(ra0c1[2]),"r"(ra0c1[3]),"r"(rb0c0[0]),"r"(rb0c0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc11[0]),"+f"(acc11[1]),"+f"(acc11[2]),"+f"(acc11[3]) \
        :"r"(ra0c1[0]),"r"(ra0c1[1]),"r"(ra0c1[2]),"r"(ra0c1[3]),"r"(rb0c1[0]),"r"(rb0c1[1])); \
}

#define MMA_KM1_CUR() \
{ \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc00[0]),"+f"(acc00[1]),"+f"(acc00[2]),"+f"(acc00[3]) \
        :"r"(ra1c0[0]),"r"(ra1c0[1]),"r"(ra1c0[2]),"r"(ra1c0[3]),"r"(rb1c0[0]),"r"(rb1c0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc01[0]),"+f"(acc01[1]),"+f"(acc01[2]),"+f"(acc01[3]) \
        :"r"(ra1c0[0]),"r"(ra1c0[1]),"r"(ra1c0[2]),"r"(ra1c0[3]),"r"(rb1c1[0]),"r"(rb1c1[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc10[0]),"+f"(acc10[1]),"+f"(acc10[2]),"+f"(acc10[3]) \
        :"r"(ra1c1[0]),"r"(ra1c1[1]),"r"(ra1c1[2]),"r"(ra1c1[3]),"r"(rb1c0[0]),"r"(rb1c0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc11[0]),"+f"(acc11[1]),"+f"(acc11[2]),"+f"(acc11[3]) \
        :"r"(ra1c1[0]),"r"(ra1c1[1]),"r"(ra1c1[2]),"r"(ra1c1[3]),"r"(rb1c1[0]),"r"(rb1c1[1])); \
}

#define MMA_KM0_NXT() \
{ \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc00[0]),"+f"(acc00[1]),"+f"(acc00[2]),"+f"(acc00[3]) \
        :"r"(ra0n0[0]),"r"(ra0n0[1]),"r"(ra0n0[2]),"r"(ra0n0[3]),"r"(rb0n0[0]),"r"(rb0n0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc01[0]),"+f"(acc01[1]),"+f"(acc01[2]),"+f"(acc01[3]) \
        :"r"(ra0n0[0]),"r"(ra0n0[1]),"r"(ra0n0[2]),"r"(ra0n0[3]),"r"(rb0n1[0]),"r"(rb0n1[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc10[0]),"+f"(acc10[1]),"+f"(acc10[2]),"+f"(acc10[3]) \
        :"r"(ra0n1[0]),"r"(ra0n1[1]),"r"(ra0n1[2]),"r"(ra0n1[3]),"r"(rb0n0[0]),"r"(rb0n0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc11[0]),"+f"(acc11[1]),"+f"(acc11[2]),"+f"(acc11[3]) \
        :"r"(ra0n1[0]),"r"(ra0n1[1]),"r"(ra0n1[2]),"r"(ra0n1[3]),"r"(rb0n1[0]),"r"(rb0n1[1])); \
}

#define MMA_KM1_NXT() \
{ \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc00[0]),"+f"(acc00[1]),"+f"(acc00[2]),"+f"(acc00[3]) \
        :"r"(ra1n0[0]),"r"(ra1n0[1]),"r"(ra1n0[2]),"r"(ra1n0[3]),"r"(rb1n0[0]),"r"(rb1n0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc01[0]),"+f"(acc01[1]),"+f"(acc01[2]),"+f"(acc01[3]) \
        :"r"(ra1n0[0]),"r"(ra1n0[1]),"r"(ra1n0[2]),"r"(ra1n0[3]),"r"(rb1n1[0]),"r"(rb1n1[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc10[0]),"+f"(acc10[1]),"+f"(acc10[2]),"+f"(acc10[3]) \
        :"r"(ra1n1[0]),"r"(ra1n1[1]),"r"(ra1n1[2]),"r"(ra1n1[3]),"r"(rb1n0[0]),"r"(rb1n0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc11[0]),"+f"(acc11[1]),"+f"(acc11[2]),"+f"(acc11[3]) \
        :"r"(ra1n1[0]),"r"(ra1n1[1]),"r"(ra1n1[2]),"r"(ra1n1[3]),"r"(rb1n1[0]),"r"(rb1n1[1])); \
}

    DECL_REGS();

    ASYNC_LDA(0,  0); ASYNC_LDB(0,  0); COMMIT;
    ASYNC_LDA(32, 1); ASYNC_LDB(32, 1); COMMIT;
    ASYNC_LDA(64, 2); ASYNC_LDB(64, 2); COMMIT;
    WAIT2; __syncthreads();

    ASYNC_LDA(96, 3); ASYNC_LDB(96, 3); COMMIT;

    LDA4(0, warp_m_base,    0,  ra0c0[0],ra0c0[1],ra0c0[2],ra0c0[3]);
    LDA4(0, warp_m_base+16, 0,  ra0c1[0],ra0c1[1],ra0c1[2],ra0c1[3]);
    LDB2T(0, 0,  warp_n_base,   rb0c0[0],rb0c0[1]);
    LDB2T(0, 0,  warp_n_base+8, rb0c1[0],rb0c1[1]);
    LDA4(0, warp_m_base,    16, ra1c0[0],ra1c0[1],ra1c0[2],ra1c0[3]);
    LDA4(0, warp_m_base+16, 16, ra1c1[0],ra1c1[1],ra1c1[2],ra1c1[3]);
    LDB2T(0, 16, warp_n_base,   rb1c0[0],rb1c0[1]);
    LDB2T(0, 16, warp_n_base+8, rb1c1[0],rb1c1[1]);

#define STAGE_PF_CC(nxt_buf, pf_k, pf_buf, wait_op) \
{ \
    wait_op; __syncthreads(); \
    ASYNC_LDA(pf_k, pf_buf); ASYNC_LDB(pf_k, pf_buf); COMMIT; \
    LOAD_NXT_KM0(nxt_buf); \
    MMA_KM0_CUR(); \
    LOAD_NXT_KM1(nxt_buf); \
    MMA_KM1_CUR(); \
    ra0c0[0]=ra0n0[0]; ra0c0[1]=ra0n0[1]; ra0c0[2]=ra0n0[2]; ra0c0[3]=ra0n0[3]; \
    ra0c1[0]=ra0n1[0]; ra0c1[1]=ra0n1[1]; ra0c1[2]=ra0n1[2]; ra0c1[3]=ra0n1[3]; \
    ra1c0[0]=ra1n0[0]; ra1c0[1]=ra1n0[1]; ra1c0[2]=ra1n0[2]; ra1c0[3]=ra1n0[3]; \
    ra1c1[0]=ra1n1[0]; ra1c1[1]=ra1n1[1]; ra1c1[2]=ra1n1[2]; ra1c1[3]=ra1n1[3]; \
    rb0c0[0]=rb0n0[0]; rb0c0[1]=rb0n0[1]; \
    rb0c1[0]=rb0n1[0]; rb0c1[1]=rb0n1[1]; \
    rb1c0[0]=rb1n0[0]; rb1c0[1]=rb1n0[1]; \
    rb1c1[0]=rb1n1[0]; rb1c1[1]=rb1n1[1]; \
}

#define STAGE_PF_NN(nxt_buf, pf_k, pf_buf, wait_op) \
{ \
    wait_op; __syncthreads(); \
    ASYNC_LDA(pf_k, pf_buf); ASYNC_LDB(pf_k, pf_buf); COMMIT; \
    LOAD_NXT_KM0(nxt_buf); \
    MMA_KM0_NXT(); \
    LOAD_NXT_KM1(nxt_buf); \
    MMA_KM1_NXT(); \
    ra0c0[0]=ra0n0[0]; ra0c0[1]=ra0n0[1]; ra0c0[2]=ra0n0[2]; ra0c0[3]=ra0n0[3]; \
    ra0c1[0]=ra0n1[0]; ra0c1[1]=ra0n1[1]; ra0c1[2]=ra0n1[2]; ra0c1[3]=ra0n1[3]; \
    ra1c0[0]=ra1n0[0]; ra1c0[1]=ra1n0[1]; ra1c0[2]=ra1n0[2]; ra1c0[3]=ra1n0[3]; \
    ra1c1[0]=ra1n1[0]; ra1c1[1]=ra1n1[1]; ra1c1[2]=ra1n1[2]; ra1c1[3]=ra1n1[3]; \
    rb0c0[0]=rb0n0[0]; rb0c0[1]=rb0n0[1]; \
    rb0c1[0]=rb0n1[0]; rb0c1[1]=rb0n1[1]; \
    rb1c0[0]=rb1n0[0]; rb1c0[1]=rb1n0[1]; \
    rb1c1[0]=rb1n1[0]; rb1c1[1]=rb1n1[1]; \
}

#define STAGE_NOPF_CC(nxt_buf, wait_op) \
{ \
    wait_op; __syncthreads(); \
    LOAD_NXT_KM0(nxt_buf); \
    MMA_KM0_CUR(); \
    LOAD_NXT_KM1(nxt_buf); \
    MMA_KM1_CUR(); \
    ra0c0[0]=ra0n0[0]; ra0c0[1]=ra0n0[1]; ra0c0[2]=ra0n0[2]; ra0c0[3]=ra0n0[3]; \
    ra0c1[0]=ra0n1[0]; ra0c1[1]=ra0n1[1]; ra0c1[2]=ra0n1[2]; ra0c1[3]=ra0n1[3]; \
    ra1c0[0]=ra1n0[0]; ra1c0[1]=ra1n0[1]; ra1c0[2]=ra1n0[2]; ra1c0[3]=ra1n0[3]; \
    ra1c1[0]=ra1n1[0]; ra1c1[1]=ra1n1[1]; ra1c1[2]=ra1n1[2]; ra1c1[3]=ra1n1[3]; \
    rb0c0[0]=rb0n0[0]; rb0c0[1]=rb0n0[1]; \
    rb0c1[0]=rb0n1[0]; rb0c1[1]=rb0n1[1]; \
    rb1c0[0]=rb1n0[0]; rb1c0[1]=rb1n0[1]; \
    rb1c1[0]=rb1n1[0]; rb1c1[1]=rb1n1[1]; \
}

    STAGE_PF_CC(1, 128, 0, WAIT2);
    STAGE_PF_CC(2, 160, 1, WAIT2);
    STAGE_PF_CC(3, 192, 2, WAIT2);
    STAGE_PF_CC(0, 224, 3, WAIT2);

    STAGE_NOPF_CC(1, WAIT2);
    STAGE_NOPF_CC(2, WAIT1);
    STAGE_NOPF_CC(3, WAIT0);

    MMA_KM0_CUR();
    MMA_KM1_CUR();

#undef ASYNC_LDA
#undef ASYNC_LDB
#undef COMMIT
#undef WAIT2
#undef WAIT1
#undef WAIT0
#undef DECL_REGS
#undef LDA4
#undef LDB2T
#undef LOAD_NXT_KM0
#undef LOAD_NXT_KM1
#undef MMA_KM0_CUR
#undef MMA_KM1_CUR
#undef MMA_KM0_NXT
#undef MMA_KM1_NXT
#undef STAGE_PF_CC
#undef STAGE_PF_NN
#undef STAGE_NOPF_CC

    half* smemC = reinterpret_cast<half*>(smemA);

    const int c_row0 = lane_id >> 2;
    const int c_col0 = (lane_id & 3) << 1;

    {
        int r0 = warp_m_base + c_row0;
        int r1 = r0 + 8;
        int c0 = warp_n_base + c_col0;
        smemC[r0 * 32 + c0]     = __float2half(acc00[0]);
        smemC[r0 * 32 + c0 + 1] = __float2half(acc00[1]);
        smemC[r1 * 32 + c0]     = __float2half(acc00[2]);
        smemC[r1 * 32 + c0 + 1] = __float2half(acc00[3]);
    }
    {
        int r0 = warp_m_base + c_row0;
        int r1 = r0 + 8;
        int c0 = warp_n_base + 8 + c_col0;
        smemC[r0 * 32 + c0]     = __float2half(acc01[0]);
        smemC[r0 * 32 + c0 + 1] = __float2half(acc01[1]);
        smemC[r1 * 32 + c0]     = __float2half(acc01[2]);
        smemC[r1 * 32 + c0 + 1] = __float2half(acc01[3]);
    }
    {
        int r0 = warp_m_base + 16 + c_row0;
        int r1 = r0 + 8;
        int c0 = warp_n_base + c_col0;
        smemC[r0 * 32 + c0]     = __float2half(acc10[0]);
        smemC[r0 * 32 + c0 + 1] = __float2half(acc10[1]);
        smemC[r1 * 32 + c0]     = __float2half(acc10[2]);
        smemC[r1 * 32 + c0 + 1] = __float2half(acc10[3]);
    }
    {
        int r0 = warp_m_base + 16 + c_row0;
        int r1 = r0 + 8;
        int c0 = warp_n_base + 8 + c_col0;
        smemC[r0 * 32 + c0]     = __float2half(acc11[0]);
        smemC[r0 * 32 + c0 + 1] = __float2half(acc11[1]);
        smemC[r1 * 32 + c0]     = __float2half(acc11[2]);
        smemC[r1 * 32 + c0 + 1] = __float2half(acc11[3]);
    }

    __syncthreads();

    {
        int idx = tid;
        int row0 = (idx * 2) >> 5;
        int col0 = ((idx * 2) & 31) << 3;

        int r = tid >> 1;
        int half_chunk = (tid & 1) << 4;
        int col_start = (tid & 1) << 4;

        uint4 v0 = *reinterpret_cast<const uint4*>(&smemC[r * 32 + col_start]);
        uint4 v1 = *reinterpret_cast<const uint4*>(&smemC[r * 32 + col_start + 8]);

        int gm_row = block_row + r;
        int gm_col = block_col + col_start;
        *reinterpret_cast<uint4*>(&C[gm_row * N + gm_col])     = v0;
        *reinterpret_cast<uint4*>(&C[gm_row * N + gm_col + 8]) = v1;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C = reinterpret_cast<half*>(c.data_ptr());

    dim3 grid(4, 2);
    dim3 block(128);
    hgemm_optimized_v13<<<grid, block>>>(A, B, C, M, N, K);
}