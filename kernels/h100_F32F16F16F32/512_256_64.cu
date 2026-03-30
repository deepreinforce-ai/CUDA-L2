#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <float.h>

__global__ __launch_bounds__(256, 2)
void kern_bm16_preload(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int bm      = blockIdx.y * 16;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int wn_base = warp_id * 32;

    __shared__ half smem_A[16][72];
    __shared__ half smem_B[64][264];

    {
        const int row = tid >> 4;
        const int col = (tid & 15) * 4;
        *reinterpret_cast<uint2*>(&smem_A[row][col]) =
            *reinterpret_cast<const uint2*>(A + (bm + row) * 64 + col);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = (i * 256 + tid) * 8;
        const int brow = linear >> 8;
        const int bcol = linear & 255;
        *reinterpret_cast<uint4*>(&smem_B[brow][bcol]) =
            *reinterpret_cast<const uint4*>(B + brow * 256 + bcol);
    }
    __syncthreads();

    uint32_t a[4][4];
    uint32_t bv[4][8];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        {
            const int mr = lane_id & 15;
            const int mc = ki * 16 + ((lane_id >> 4) * 8);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a[ki][0]),"=r"(a[ki][1]),"=r"(a[ki][2]),"=r"(a[ki][3]):"r"(addr));
        }
        {
            const int br = ki * 16 + (lane_id & 15);
            uint32_t p0=__cvta_generic_to_shared(&smem_B[br][wn_base+0]);
            uint32_t p1=__cvta_generic_to_shared(&smem_B[br][wn_base+8]);
            uint32_t p2=__cvta_generic_to_shared(&smem_B[br][wn_base+16]);
            uint32_t p3=__cvta_generic_to_shared(&smem_B[br][wn_base+24]);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][0]),"=r"(bv[ki][1]):"r"(p0));
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][2]),"=r"(bv[ki][3]):"r"(p1));
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][4]),"=r"(bv[ki][5]):"r"(p2));
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][6]),"=r"(bv[ki][7]):"r"(p3));
        }
    }

    float d0=0,d1=0,d2=0,d3=0, e0=0,e1=0,e2=0,e3=0;
    float f0=0,f1=0,f2=0,f3=0, g0=0,g1=0,g2=0,g3=0;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3):"r"(a[ki][0]),"r"(a[ki][1]),"r"(a[ki][2]),"r"(a[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d0),"f"(d1),"f"(d2),"f"(d3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(e0),"=f"(e1),"=f"(e2),"=f"(e3):"r"(a[ki][0]),"r"(a[ki][1]),"r"(a[ki][2]),"r"(a[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(e0),"f"(e1),"f"(e2),"f"(e3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(f0),"=f"(f1),"=f"(f2),"=f"(f3):"r"(a[ki][0]),"r"(a[ki][1]),"r"(a[ki][2]),"r"(a[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(f0),"f"(f1),"f"(f2),"f"(f3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(g0),"=f"(g1),"=f"(g2),"=f"(g3):"r"(a[ki][0]),"r"(a[ki][1]),"r"(a[ki][2]),"r"(a[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(g0),"f"(g1),"f"(g2),"f"(g3));
    }

    {
        const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
        const int gm0=bm+r0, gm1=bm+r1;
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(d0,d1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(e0,e1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(f0,f1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(g0,g1);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(d2,d3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(e2,e3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(f2,f3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(g2,g3);
    }
}

__global__ __launch_bounds__(256, 2)
void kern_bm32_dual(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int bm_base = blockIdx.y * 32;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int wn_base = warp_id * 32;

    __shared__ half smem_A0[16][72];
    __shared__ half smem_A1[16][72];
    __shared__ half smem_B[64][264];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = (i * 256 + tid) * 8;
        const int brow = linear >> 8;
        const int bcol = linear & 255;
        *reinterpret_cast<uint4*>(&smem_B[brow][bcol]) =
            *reinterpret_cast<const uint4*>(B + brow * 256 + bcol);
    }
    {
        const int row = tid >> 4;
        const int col = (tid & 15) * 4;
        *reinterpret_cast<uint2*>(&smem_A0[row][col]) =
            *reinterpret_cast<const uint2*>(A + (bm_base + row) * 64 + col);
        *reinterpret_cast<uint2*>(&smem_A1[row][col]) =
            *reinterpret_cast<const uint2*>(A + (bm_base + 16 + row) * 64 + col);
    }
    __syncthreads();

    uint32_t a0[4][4], a1[4][4], bv[4][8];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        {
            const int mr = lane_id & 15;
            const int mc = ki * 16 + ((lane_id >> 4) * 8);
            uint32_t addr0 = __cvta_generic_to_shared(&smem_A0[mr][mc]);
            uint32_t addr1 = __cvta_generic_to_shared(&smem_A1[mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a0[ki][0]),"=r"(a0[ki][1]),"=r"(a0[ki][2]),"=r"(a0[ki][3]):"r"(addr0));
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a1[ki][0]),"=r"(a1[ki][1]),"=r"(a1[ki][2]),"=r"(a1[ki][3]):"r"(addr1));
        }
        {
            const int br = ki * 16 + (lane_id & 15);
            uint32_t p0=__cvta_generic_to_shared(&smem_B[br][wn_base+0]);
            uint32_t p1=__cvta_generic_to_shared(&smem_B[br][wn_base+8]);
            uint32_t p2=__cvta_generic_to_shared(&smem_B[br][wn_base+16]);
            uint32_t p3=__cvta_generic_to_shared(&smem_B[br][wn_base+24]);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][0]),"=r"(bv[ki][1]):"r"(p0));
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][2]),"=r"(bv[ki][3]):"r"(p1));
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][4]),"=r"(bv[ki][5]):"r"(p2));
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][6]),"=r"(bv[ki][7]):"r"(p3));
        }
    }

    float s0d0=0,s0d1=0,s0d2=0,s0d3=0;
    float s0e0=0,s0e1=0,s0e2=0,s0e3=0;
    float s0f0=0,s0f1=0,s0f2=0,s0f3=0;
    float s0g0=0,s0g1=0,s0g2=0,s0g3=0;
    float s1d0=0,s1d1=0,s1d2=0,s1d3=0;
    float s1e0=0,s1e1=0,s1e2=0,s1e3=0;
    float s1f0=0,s1f1=0,s1f2=0,s1f3=0;
    float s1g0=0,s1g1=0,s1g2=0,s1g3=0;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s0d0),"=f"(s0d1),"=f"(s0d2),"=f"(s0d3):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(s0d0),"f"(s0d1),"f"(s0d2),"f"(s0d3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s0e0),"=f"(s0e1),"=f"(s0e2),"=f"(s0e3):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(s0e0),"f"(s0e1),"f"(s0e2),"f"(s0e3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s0f0),"=f"(s0f1),"=f"(s0f2),"=f"(s0f3):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(s0f0),"f"(s0f1),"f"(s0f2),"f"(s0f3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s0g0),"=f"(s0g1),"=f"(s0g2),"=f"(s0g3):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(s0g0),"f"(s0g1),"f"(s0g2),"f"(s0g3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s1d0),"=f"(s1d1),"=f"(s1d2),"=f"(s1d3):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(s1d0),"f"(s1d1),"f"(s1d2),"f"(s1d3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s1e0),"=f"(s1e1),"=f"(s1e2),"=f"(s1e3):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(s1e0),"f"(s1e1),"f"(s1e2),"f"(s1e3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s1f0),"=f"(s1f1),"=f"(s1f2),"=f"(s1f3):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(s1f0),"f"(s1f1),"f"(s1f2),"f"(s1f3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(s1g0),"=f"(s1g1),"=f"(s1g2),"=f"(s1g3):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(s1g0),"f"(s1g1),"f"(s1g2),"f"(s1g3));
    }

    {
        const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
        const int gm0=bm_base+r0, gm1=bm_base+r1;
        const int gm2=bm_base+16+r0, gm3=bm_base+16+r1;
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(s0d0,s0d1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(s0e0,s0e1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(s0f0,s0f1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(s0g0,s0g1);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(s0d2,s0d3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(s0e2,s0e3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(s0f2,s0f3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(s0g2,s0g3);
        *reinterpret_cast<half2*>(&C[gm2*256+wn_base+cp])    =__floats2half2_rn(s1d0,s1d1);
        *reinterpret_cast<half2*>(&C[gm2*256+wn_base+8+cp])  =__floats2half2_rn(s1e0,s1e1);
        *reinterpret_cast<half2*>(&C[gm2*256+wn_base+16+cp]) =__floats2half2_rn(s1f0,s1f1);
        *reinterpret_cast<half2*>(&C[gm2*256+wn_base+24+cp]) =__floats2half2_rn(s1g0,s1g1);
        *reinterpret_cast<half2*>(&C[gm3*256+wn_base+cp])    =__floats2half2_rn(s1d2,s1d3);
        *reinterpret_cast<half2*>(&C[gm3*256+wn_base+8+cp])  =__floats2half2_rn(s1e2,s1e3);
        *reinterpret_cast<half2*>(&C[gm3*256+wn_base+16+cp]) =__floats2half2_rn(s1f2,s1f3);
        *reinterpret_cast<half2*>(&C[gm3*256+wn_base+24+cp]) =__floats2half2_rn(s1g2,s1g3);
    }
}

__global__ __launch_bounds__(256, 2)
void kern_bm48_tri(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int blk     = blockIdx.y;
    const int bm_base = blk * 48;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int wn_base = warp_id * 32;

    int remaining = M - bm_base;
    int num_strips = (remaining >= 48) ? 3 : (remaining >= 32 ? 2 : 1);

    __shared__ half smem_A[3][16][72];
    __shared__ half smem_B[64][264];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = (i * 256 + tid) * 8;
        const int brow = linear >> 8;
        const int bcol = linear & 255;
        *reinterpret_cast<uint4*>(&smem_B[brow][bcol]) =
            *reinterpret_cast<const uint4*>(B + brow * 256 + bcol);
    }

    for (int s = 0; s < num_strips; s++) {
        const int bm = bm_base + s * 16;
        const int row = tid >> 4;
        const int col = (tid & 15) * 4;
        *reinterpret_cast<uint2*>(&smem_A[s][row][col]) =
            *reinterpret_cast<const uint2*>(A + (bm + row) * 64 + col);
    }
    __syncthreads();

    uint32_t bv[4][8];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int br = ki * 16 + (lane_id & 15);
        uint32_t p0=__cvta_generic_to_shared(&smem_B[br][wn_base+0]);
        uint32_t p1=__cvta_generic_to_shared(&smem_B[br][wn_base+8]);
        uint32_t p2=__cvta_generic_to_shared(&smem_B[br][wn_base+16]);
        uint32_t p3=__cvta_generic_to_shared(&smem_B[br][wn_base+24]);
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][0]),"=r"(bv[ki][1]):"r"(p0));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][2]),"=r"(bv[ki][3]):"r"(p1));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][4]),"=r"(bv[ki][5]):"r"(p2));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][6]),"=r"(bv[ki][7]):"r"(p3));
    }

    float d0_0=0,d0_1=0,d0_2=0,d0_3=0;
    float d0_4=0,d0_5=0,d0_6=0,d0_7=0;
    float d0_8=0,d0_9=0,d0_10=0,d0_11=0;
    float d0_12=0,d0_13=0,d0_14=0,d0_15=0;

    float d1_0=0,d1_1=0,d1_2=0,d1_3=0;
    float d1_4=0,d1_5=0,d1_6=0,d1_7=0;
    float d1_8=0,d1_9=0,d1_10=0,d1_11=0;
    float d1_12=0,d1_13=0,d1_14=0,d1_15=0;

    float d2_0=0,d2_1=0,d2_2=0,d2_3=0;
    float d2_4=0,d2_5=0,d2_6=0,d2_7=0;
    float d2_8=0,d2_9=0,d2_10=0,d2_11=0;
    float d2_12=0,d2_13=0,d2_14=0,d2_15=0;

    uint32_t a0[4][4], a1[4][4], a2[4][4];

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int mr = lane_id & 15;
        const int mc = ki * 16 + ((lane_id >> 4) * 8);
        {
            uint32_t addr = __cvta_generic_to_shared(&smem_A[0][mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a0[ki][0]),"=r"(a0[ki][1]),"=r"(a0[ki][2]),"=r"(a0[ki][3]):"r"(addr));
        }
        if (num_strips >= 2) {
            uint32_t addr = __cvta_generic_to_shared(&smem_A[1][mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a1[ki][0]),"=r"(a1[ki][1]),"=r"(a1[ki][2]),"=r"(a1[ki][3]):"r"(addr));
        }
        if (num_strips >= 3) {
            uint32_t addr = __cvta_generic_to_shared(&smem_A[2][mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a2[ki][0]),"=r"(a2[ki][1]),"=r"(a2[ki][2]),"=r"(a2[ki][3]):"r"(addr));
        }
    }

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(d0_0),"=f"(d0_1),"=f"(d0_2),"=f"(d0_3):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d0_0),"f"(d0_1),"f"(d0_2),"f"(d0_3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(d0_4),"=f"(d0_5),"=f"(d0_6),"=f"(d0_7):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(d0_4),"f"(d0_5),"f"(d0_6),"f"(d0_7));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(d0_8),"=f"(d0_9),"=f"(d0_10),"=f"(d0_11):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(d0_8),"f"(d0_9),"f"(d0_10),"f"(d0_11));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(d0_12),"=f"(d0_13),"=f"(d0_14),"=f"(d0_15):"r"(a0[ki][0]),"r"(a0[ki][1]),"r"(a0[ki][2]),"r"(a0[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(d0_12),"f"(d0_13),"f"(d0_14),"f"(d0_15));
        if (num_strips >= 2) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d1_0),"=f"(d1_1),"=f"(d1_2),"=f"(d1_3):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d1_0),"f"(d1_1),"f"(d1_2),"f"(d1_3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d1_4),"=f"(d1_5),"=f"(d1_6),"=f"(d1_7):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(d1_4),"f"(d1_5),"f"(d1_6),"f"(d1_7));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d1_8),"=f"(d1_9),"=f"(d1_10),"=f"(d1_11):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(d1_8),"f"(d1_9),"f"(d1_10),"f"(d1_11));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d1_12),"=f"(d1_13),"=f"(d1_14),"=f"(d1_15):"r"(a1[ki][0]),"r"(a1[ki][1]),"r"(a1[ki][2]),"r"(a1[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(d1_12),"f"(d1_13),"f"(d1_14),"f"(d1_15));
        }
        if (num_strips >= 3) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d2_0),"=f"(d2_1),"=f"(d2_2),"=f"(d2_3):"r"(a2[ki][0]),"r"(a2[ki][1]),"r"(a2[ki][2]),"r"(a2[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d2_0),"f"(d2_1),"f"(d2_2),"f"(d2_3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d2_4),"=f"(d2_5),"=f"(d2_6),"=f"(d2_7):"r"(a2[ki][0]),"r"(a2[ki][1]),"r"(a2[ki][2]),"r"(a2[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(d2_4),"f"(d2_5),"f"(d2_6),"f"(d2_7));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d2_8),"=f"(d2_9),"=f"(d2_10),"=f"(d2_11):"r"(a2[ki][0]),"r"(a2[ki][1]),"r"(a2[ki][2]),"r"(a2[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(d2_8),"f"(d2_9),"f"(d2_10),"f"(d2_11));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d2_12),"=f"(d2_13),"=f"(d2_14),"=f"(d2_15):"r"(a2[ki][0]),"r"(a2[ki][1]),"r"(a2[ki][2]),"r"(a2[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(d2_12),"f"(d2_13),"f"(d2_14),"f"(d2_15));
        }
    }

    {
        const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
        const int gm0=bm_base+r0, gm1=bm_base+r1;
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(d0_0,d0_1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(d0_4,d0_5);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(d0_8,d0_9);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(d0_12,d0_13);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(d0_2,d0_3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(d0_6,d0_7);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(d0_10,d0_11);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(d0_14,d0_15);
    }
    if (num_strips >= 2) {
        const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
        const int gm0=bm_base+16+r0, gm1=bm_base+16+r1;
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(d1_0,d1_1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(d1_4,d1_5);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(d1_8,d1_9);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(d1_12,d1_13);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(d1_2,d1_3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(d1_6,d1_7);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(d1_10,d1_11);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(d1_14,d1_15);
    }
    if (num_strips >= 3) {
        const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
        const int gm0=bm_base+32+r0, gm1=bm_base+32+r1;
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(d2_0,d2_1);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(d2_4,d2_5);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(d2_8,d2_9);
        *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(d2_12,d2_13);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(d2_2,d2_3);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(d2_6,d2_7);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(d2_10,d2_11);
        *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(d2_14,d2_15);
    }
}

__global__ __launch_bounds__(256, 2)
void kern_bm64_fullN(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int bm      = blockIdx.y * 64;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id >> 1;
    const int warp_n  = warp_id & 1;
    const int wm_off  = warp_m * 16;
    const int wn_base = warp_n * 128;

    __shared__ half smem_A[64][72];
    __shared__ half smem_B[64][264];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = (i*256+tid)*8;
        const int brow = linear >> 8;
        const int bcol = linear & 255;
        if (brow < 64)
            *reinterpret_cast<uint4*>(&smem_B[brow][bcol]) =
                *reinterpret_cast<const uint4*>(B + brow*256 + bcol);
    }
    {
        const int row = tid >> 2;
        const int col = (tid & 3) * 16;
        *reinterpret_cast<uint4*>(&smem_A[row][col])   =
            *reinterpret_cast<const uint4*>(A + (bm+row)*64 + col);
        *reinterpret_cast<uint4*>(&smem_A[row][col+8]) =
            *reinterpret_cast<const uint4*>(A + (bm+row)*64 + col+8);
    }
    __syncthreads();

    float d0=0,d1=0,d2=0,d3=0,   d4=0,d5=0,d6=0,d7=0;
    float d8=0,d9=0,d10=0,d11=0,  d12=0,d13=0,d14=0,d15=0;
    float d16=0,d17=0,d18=0,d19=0, d20=0,d21=0,d22=0,d23=0;
    float d24=0,d25=0,d26=0,d27=0, d28=0,d29=0,d30=0,d31=0;
    float d32=0,d33=0,d34=0,d35=0, d36=0,d37=0,d38=0,d39=0;
    float d40=0,d41=0,d42=0,d43=0, d44=0,d45=0,d46=0,d47=0;
    float d48=0,d49=0,d50=0,d51=0, d52=0,d53=0,d54=0,d55=0;
    float d56=0,d57=0,d58=0,d59=0, d60=0,d61=0,d62=0,d63=0;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        uint32_t a0,a1,a2,a3;
        {
            const int mr=wm_off+(lane_id&15), mc=ki*16+((lane_id>>4)*8);
            uint32_t addr=__cvta_generic_to_shared(&smem_A[mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(addr));
        }
        uint32_t b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;
        uint32_t b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31;
        {
            const int br=ki*16+(lane_id&15);
            #define LDB(r0,r1,off) { uint32_t addr=__cvta_generic_to_shared(&smem_B[br][wn_base+(off)]); \
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(r0),"=r"(r1):"r"(addr)); }
            LDB(b0,b1,0)    LDB(b2,b3,8)    LDB(b4,b5,16)   LDB(b6,b7,24)
            LDB(b8,b9,32)   LDB(b10,b11,40) LDB(b12,b13,48) LDB(b14,b15,56)
            LDB(b16,b17,64) LDB(b18,b19,72) LDB(b20,b21,80) LDB(b22,b23,88)
            LDB(b24,b25,96) LDB(b26,b27,104) LDB(b28,b29,112) LDB(b30,b31,120)
            #undef LDB
        }
        #define MMA(d0,d1,d2,d3,bx,by) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
            :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bx),"r"(by),"f"(d0),"f"(d1),"f"(d2),"f"(d3));
        MMA(d0,d1,d2,d3,b0,b1)      MMA(d4,d5,d6,d7,b2,b3)
        MMA(d8,d9,d10,d11,b4,b5)    MMA(d12,d13,d14,d15,b6,b7)
        MMA(d16,d17,d18,d19,b8,b9)  MMA(d20,d21,d22,d23,b10,b11)
        MMA(d24,d25,d26,d27,b12,b13) MMA(d28,d29,d30,d31,b14,b15)
        MMA(d32,d33,d34,d35,b16,b17) MMA(d36,d37,d38,d39,b18,b19)
        MMA(d40,d41,d42,d43,b20,b21) MMA(d44,d45,d46,d47,b22,b23)
        MMA(d48,d49,d50,d51,b24,b25) MMA(d52,d53,d54,d55,b26,b27)
        MMA(d56,d57,d58,d59,b28,b29) MMA(d60,d61,d62,d63,b30,b31)
        #undef MMA
    }

    {
        const int r0=wm_off+(lane_id>>2), r1=r0+8, cp=(lane_id&3)*2;
        const int gm0=bm+r0, gm1=bm+r1;
        #define ST(row,x,y,off) *reinterpret_cast<half2*>(&C[(row)*256+wn_base+(off)+cp])=__floats2half2_rn(x,y);
        ST(gm0,d0,d1,0)    ST(gm0,d4,d5,8)    ST(gm0,d8,d9,16)   ST(gm0,d12,d13,24)
        ST(gm0,d16,d17,32) ST(gm0,d20,d21,40) ST(gm0,d24,d25,48) ST(gm0,d28,d29,56)
        ST(gm0,d32,d33,64) ST(gm0,d36,d37,72) ST(gm0,d40,d41,80) ST(gm0,d44,d45,88)
        ST(gm0,d48,d49,96) ST(gm0,d52,d53,104) ST(gm0,d56,d57,112) ST(gm0,d60,d61,120)
        ST(gm1,d2,d3,0)    ST(gm1,d6,d7,8)    ST(gm1,d10,d11,16) ST(gm1,d14,d15,24)
        ST(gm1,d18,d19,32) ST(gm1,d22,d23,40) ST(gm1,d26,d27,48) ST(gm1,d30,d31,56)
        ST(gm1,d34,d35,64) ST(gm1,d38,d39,72) ST(gm1,d42,d43,80) ST(gm1,d46,d47,88)
        ST(gm1,d50,d51,96) ST(gm1,d54,d55,104) ST(gm1,d58,d59,112) ST(gm1,d62,d63,120)
        #undef ST
    }
}

__global__ __launch_bounds__(128, 8)
void kern_bm16_bn64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int bm      = blockIdx.y * 16;
    const int bn      = blockIdx.x * 64;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int wn_base = warp_id * 16;

    __shared__ half smem_A[16][72];
    __shared__ half smem_B[64][72];

    {
        const int row = tid >> 3;
        const int col = (tid & 7) * 8;
        *reinterpret_cast<uint4*>(&smem_A[row][col]) =
            *reinterpret_cast<const uint4*>(A + (bm+row)*64 + col);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int lin = (tid*4+i)*8;
        const int row = lin >> 6;
        const int col = lin & 63;
        *reinterpret_cast<uint4*>(&smem_B[row][col]) =
            *reinterpret_cast<const uint4*>(B + row*256 + bn + col);
    }
    __syncthreads();

    uint32_t a[4][4], bv[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        {
            const int mr=lane_id&15, mc=ki*16+((lane_id>>4)*8);
            uint32_t addr=__cvta_generic_to_shared(&smem_A[mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(a[ki][0]),"=r"(a[ki][1]),"=r"(a[ki][2]),"=r"(a[ki][3]):"r"(addr));
        }
        {
            const int br=ki*16+(lane_id&15);
            uint32_t p0=__cvta_generic_to_shared(&smem_B[br][wn_base]);
            uint32_t p1=__cvta_generic_to_shared(&smem_B[br][wn_base+8]);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][0]),"=r"(bv[ki][1]):"r"(p0));
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][2]),"=r"(bv[ki][3]):"r"(p1));
        }
    }

    float d0=0,d1=0,d2=0,d3=0, e0=0,e1=0,e2=0,e3=0;
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3):"r"(a[ki][0]),"r"(a[ki][1]),"r"(a[ki][2]),"r"(a[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d0),"f"(d1),"f"(d2),"f"(d3));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            :"=f"(e0),"=f"(e1),"=f"(e2),"=f"(e3):"r"(a[ki][0]),"r"(a[ki][1]),"r"(a[ki][2]),"r"(a[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(e0),"f"(e1),"f"(e2),"f"(e3));
    }

    {
        const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
        *reinterpret_cast<half2*>(&C[(bm+r0)*256+bn+wn_base+cp])   =__floats2half2_rn(d0,d1);
        *reinterpret_cast<half2*>(&C[(bm+r0)*256+bn+wn_base+8+cp]) =__floats2half2_rn(e0,e1);
        *reinterpret_cast<half2*>(&C[(bm+r1)*256+bn+wn_base+cp])   =__floats2half2_rn(d2,d3);
        *reinterpret_cast<half2*>(&C[(bm+r1)*256+bn+wn_base+8+cp]) =__floats2half2_rn(e2,e3);
    }
}

__global__ __launch_bounds__(256, 2)
void kern_b_reg_pinned(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int wn_base = warp_id * 32;

    __shared__ half smem_B[64][264];
    __shared__ half smem_A[16][72];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = (i * 256 + tid) * 8;
        const int brow = linear >> 8;
        const int bcol = linear & 255;
        *reinterpret_cast<uint4*>(&smem_B[brow][bcol]) =
            *reinterpret_cast<const uint4*>(B + brow * 256 + bcol);
    }
    __syncthreads();

    uint32_t bv[4][8];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int br = ki * 16 + (lane_id & 15);
        uint32_t p0=__cvta_generic_to_shared(&smem_B[br][wn_base+0]);
        uint32_t p1=__cvta_generic_to_shared(&smem_B[br][wn_base+8]);
        uint32_t p2=__cvta_generic_to_shared(&smem_B[br][wn_base+16]);
        uint32_t p3=__cvta_generic_to_shared(&smem_B[br][wn_base+24]);
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][0]),"=r"(bv[ki][1]):"r"(p0));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][2]),"=r"(bv[ki][3]):"r"(p1));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][4]),"=r"(bv[ki][5]):"r"(p2));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][6]),"=r"(bv[ki][7]):"r"(p3));
    }

    #pragma unroll 4
    for (int strip = 0; strip < 32; strip++) {
        const int bm = strip * 16;

        {
            const int row = tid >> 4;
            const int col = (tid & 15) * 4;
            *reinterpret_cast<uint2*>(&smem_A[row][col]) =
                *reinterpret_cast<const uint2*>(A + (bm + row) * 64 + col);
        }
        __syncthreads();

        uint32_t av[4][4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int mr = lane_id & 15;
            const int mc = ki * 16 + ((lane_id >> 4) * 8);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(av[ki][0]),"=r"(av[ki][1]),"=r"(av[ki][2]),"=r"(av[ki][3]):"r"(addr));
        }

        float d0=0,d1=0,d2=0,d3=0, e0=0,e1=0,e2=0,e3=0;
        float f0=0,f1=0,f2=0,f3=0, g0=0,g1=0,g2=0,g3=0;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d0),"f"(d1),"f"(d2),"f"(d3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(e0),"=f"(e1),"=f"(e2),"=f"(e3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(e0),"f"(e1),"f"(e2),"f"(e3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(f0),"=f"(f1),"=f"(f2),"=f"(f3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(f0),"f"(f1),"f"(f2),"f"(f3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(g0),"=f"(g1),"=f"(g2),"=f"(g3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(g0),"f"(g1),"f"(g2),"f"(g3));
        }

        {
            const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
            const int gm0=bm+r0, gm1=bm+r1;
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(d0,d1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(e0,e1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(f0,f1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(g0,g1);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(d2,d3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(e2,e3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(f2,f3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(g2,g3);
        }

        __syncthreads();
    }
}

__global__ __launch_bounds__(256, 2)
void kern_b_reg_pinned_2cta(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int cta_m   = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int wn_base = warp_id * 32;
    const int m_start = cta_m * 256;

    __shared__ half smem_B[64][264];
    __shared__ half smem_A[16][72];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = (i * 256 + tid) * 8;
        const int brow = linear >> 8;
        const int bcol = linear & 255;
        *reinterpret_cast<uint4*>(&smem_B[brow][bcol]) =
            *reinterpret_cast<const uint4*>(B + brow * 256 + bcol);
    }
    __syncthreads();

    uint32_t bv[4][8];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int br = ki * 16 + (lane_id & 15);
        uint32_t p0=__cvta_generic_to_shared(&smem_B[br][wn_base+0]);
        uint32_t p1=__cvta_generic_to_shared(&smem_B[br][wn_base+8]);
        uint32_t p2=__cvta_generic_to_shared(&smem_B[br][wn_base+16]);
        uint32_t p3=__cvta_generic_to_shared(&smem_B[br][wn_base+24]);
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][0]),"=r"(bv[ki][1]):"r"(p0));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][2]),"=r"(bv[ki][3]):"r"(p1));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][4]),"=r"(bv[ki][5]):"r"(p2));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][6]),"=r"(bv[ki][7]):"r"(p3));
    }

    #pragma unroll 4
    for (int strip = 0; strip < 16; strip++) {
        const int bm = m_start + strip * 16;

        {
            const int row = tid >> 4;
            const int col = (tid & 15) * 4;
            *reinterpret_cast<uint2*>(&smem_A[row][col]) =
                *reinterpret_cast<const uint2*>(A + (bm + row) * 64 + col);
        }
        __syncthreads();

        uint32_t av[4][4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int mr = lane_id & 15;
            const int mc = ki * 16 + ((lane_id >> 4) * 8);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(av[ki][0]),"=r"(av[ki][1]),"=r"(av[ki][2]),"=r"(av[ki][3]):"r"(addr));
        }

        float d0=0,d1=0,d2=0,d3=0, e0=0,e1=0,e2=0,e3=0;
        float f0=0,f1=0,f2=0,f3=0, g0=0,g1=0,g2=0,g3=0;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d0),"f"(d1),"f"(d2),"f"(d3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(e0),"=f"(e1),"=f"(e2),"=f"(e3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(e0),"f"(e1),"f"(e2),"f"(e3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(f0),"=f"(f1),"=f"(f2),"=f"(f3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(f0),"f"(f1),"f"(f2),"f"(f3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(g0),"=f"(g1),"=f"(g2),"=f"(g3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(g0),"f"(g1),"f"(g2),"f"(g3));
        }

        {
            const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
            const int gm0=bm+r0, gm1=bm+r1;
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(d0,d1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(e0,e1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(f0,f1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(g0,g1);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(d2,d3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(e2,e3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(f2,f3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(g2,g3);
        }

        __syncthreads();
    }
}

__global__ __launch_bounds__(256, 2)
void kern_b_reg_pinned_4cta(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int cta_m   = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int wn_base = warp_id * 32;
    const int m_start = cta_m * 128;

    __shared__ half smem_B[64][264];
    __shared__ half smem_A[16][72];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int linear = (i * 256 + tid) * 8;
        const int brow = linear >> 8;
        const int bcol = linear & 255;
        *reinterpret_cast<uint4*>(&smem_B[brow][bcol]) =
            *reinterpret_cast<const uint4*>(B + brow * 256 + bcol);
    }
    __syncthreads();

    uint32_t bv[4][8];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int br = ki * 16 + (lane_id & 15);
        uint32_t p0=__cvta_generic_to_shared(&smem_B[br][wn_base+0]);
        uint32_t p1=__cvta_generic_to_shared(&smem_B[br][wn_base+8]);
        uint32_t p2=__cvta_generic_to_shared(&smem_B[br][wn_base+16]);
        uint32_t p3=__cvta_generic_to_shared(&smem_B[br][wn_base+24]);
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][0]),"=r"(bv[ki][1]):"r"(p0));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][2]),"=r"(bv[ki][3]):"r"(p1));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][4]),"=r"(bv[ki][5]):"r"(p2));
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ki][6]),"=r"(bv[ki][7]):"r"(p3));
    }

    #pragma unroll 4
    for (int strip = 0; strip < 8; strip++) {
        const int bm = m_start + strip * 16;

        {
            const int row = tid >> 4;
            const int col = (tid & 15) * 4;
            *reinterpret_cast<uint2*>(&smem_A[row][col]) =
                *reinterpret_cast<const uint2*>(A + (bm + row) * 64 + col);
        }
        __syncthreads();

        uint32_t av[4][4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int mr = lane_id & 15;
            const int mc = ki * 16 + ((lane_id >> 4) * 8);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[mr][mc]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                :"=r"(av[ki][0]),"=r"(av[ki][1]),"=r"(av[ki][2]),"=r"(av[ki][3]):"r"(addr));
        }

        float d0=0,d1=0,d2=0,d3=0, e0=0,e1=0,e2=0,e3=0;
        float f0=0,f1=0,f2=0,f3=0, g0=0,g1=0,g2=0,g3=0;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][0]),"r"(bv[ki][1]),"f"(d0),"f"(d1),"f"(d2),"f"(d3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(e0),"=f"(e1),"=f"(e2),"=f"(e3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][2]),"r"(bv[ki][3]),"f"(e0),"f"(e1),"f"(e2),"f"(e3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(f0),"=f"(f1),"=f"(f2),"=f"(f3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][4]),"r"(bv[ki][5]),"f"(f0),"f"(f1),"f"(f2),"f"(f3));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                :"=f"(g0),"=f"(g1),"=f"(g2),"=f"(g3):"r"(av[ki][0]),"r"(av[ki][1]),"r"(av[ki][2]),"r"(av[ki][3]),"r"(bv[ki][6]),"r"(bv[ki][7]),"f"(g0),"f"(g1),"f"(g2),"f"(g3));
        }

        {
            const int r0=lane_id>>2, r1=r0+8, cp=(lane_id&3)*2;
            const int gm0=bm+r0, gm1=bm+r1;
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+cp])    =__floats2half2_rn(d0,d1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+8+cp])  =__floats2half2_rn(e0,e1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+16+cp]) =__floats2half2_rn(f0,f1);
            *reinterpret_cast<half2*>(&C[gm0*256+wn_base+24+cp]) =__floats2half2_rn(g0,g1);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+cp])    =__floats2half2_rn(d2,d3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+8+cp])  =__floats2half2_rn(e2,e3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+16+cp]) =__floats2half2_rn(f2,f3);
            *reinterpret_cast<half2*>(&C[gm1*256+wn_base+24+cp]) =__floats2half2_rn(g2,g3);
        }

        __syncthreads();
    }
}

static int g_best_kid = -1;

static void do_autotune(const half* A, const half* B, half* C) {
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    const int WU = 40, RN = 300;
    const int NKERN = 8;
    float times[NKERN];

    auto bench = [&](int kid, auto fn) {
        cudaDeviceSynchronize();
        cudaEventRecord(ev0);
        for (int r = 0; r < RN; r++) fn();
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&times[kid], ev0, ev1);
    };

    for (int w = 0; w < WU; w++) {
        kern_bm16_preload<<<dim3(1,32),256>>>(A,B,C);
        kern_bm32_dual<<<dim3(1,16),256>>>(A,B,C);
        kern_bm48_tri<<<dim3(1,11),256>>>(A,B,C,512);
        kern_bm64_fullN<<<dim3(1,8),256>>>(A,B,C);
        kern_bm16_bn64<<<dim3(4,32),128>>>(A,B,C);
        kern_b_reg_pinned<<<dim3(1,1),256>>>(A,B,C);
        kern_b_reg_pinned_2cta<<<dim3(1,2),256>>>(A,B,C);
        kern_b_reg_pinned_4cta<<<dim3(1,4),256>>>(A,B,C);
    }
    cudaDeviceSynchronize();

    bench(0, [&]{ kern_bm16_preload<<<dim3(1,32),256>>>(A,B,C); });
    bench(1, [&]{ kern_bm32_dual<<<dim3(1,16),256>>>(A,B,C); });
    bench(2, [&]{ kern_bm48_tri<<<dim3(1,11),256>>>(A,B,C,512); });
    bench(3, [&]{ kern_bm64_fullN<<<dim3(1,8),256>>>(A,B,C); });
    bench(4, [&]{ kern_bm16_bn64<<<dim3(4,32),128>>>(A,B,C); });
    bench(5, [&]{ kern_b_reg_pinned<<<dim3(1,1),256>>>(A,B,C); });
    bench(6, [&]{ kern_b_reg_pinned_2cta<<<dim3(1,2),256>>>(A,B,C); });
    bench(7, [&]{ kern_b_reg_pinned_4cta<<<dim3(1,4),256>>>(A,B,C); });

    g_best_kid = 0;
    float best = times[0];
    for (int i = 1; i < NKERN; i++) {
        if (times[i] < best) { best = times[i]; g_best_kid = i; }
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c
) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

    if (M == 512 && N == 256 && K == 64) {
        if (g_best_kid < 0) {
            do_autotune(ptr_A, ptr_B, ptr_C);
        }
        switch (g_best_kid) {
            case 0: kern_bm16_preload<<<dim3(1,32),256>>>(ptr_A,ptr_B,ptr_C); break;
            case 1: kern_bm32_dual<<<dim3(1,16),256>>>(ptr_A,ptr_B,ptr_C); break;
            case 2: kern_bm48_tri<<<dim3(1,11),256>>>(ptr_A,ptr_B,ptr_C,M); break;
            case 3: kern_bm64_fullN<<<dim3(1,8),256>>>(ptr_A,ptr_B,ptr_C); break;
            case 4: kern_bm16_bn64<<<dim3(4,32),128>>>(ptr_A,ptr_B,ptr_C); break;
            case 5: kern_b_reg_pinned<<<dim3(1,1),256>>>(ptr_A,ptr_B,ptr_C); break;
            case 6: kern_b_reg_pinned_2cta<<<dim3(1,2),256>>>(ptr_A,ptr_B,ptr_C); break;
            case 7: kern_b_reg_pinned_4cta<<<dim3(1,4),256>>>(ptr_A,ptr_B,ptr_C); break;
        }
        return;
    }

    {
        dim3 grid((N+63)/64, (M+15)/16);
        kern_bm16_bn64<<<grid,128>>>(ptr_A,ptr_B,ptr_C);
    }
}