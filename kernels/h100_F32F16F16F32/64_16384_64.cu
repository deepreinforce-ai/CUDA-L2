#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t pack_half2(half a, half b) {
    uint32_t v;
    asm volatile("{ .reg .b16 lo, hi; mov.b16 lo, %1; mov.b16 hi, %2; mov.b32 %0, {lo,hi}; }"
                 : "=r"(v) : "h"(__half_as_ushort(a)), "h"(__half_as_ushort(b)));
    return v;
}

__device__ __forceinline__ void mma_m16n8k16(
    float &c0, float &c1, float &c2, float &c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    uint32_t d = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" : : "r"(d),"l"((const char*)src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n"); }

__global__ void __launch_bounds__(128, 8)
gemm_v13_primary(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int N)
{
    __shared__ __align__(128) half sh_A[64][72];
    __shared__ __align__(128) half sh_B[128][72];

    const int cta_n = blockIdx.x * 128;
    const int tid   = threadIdx.x;
    const int wid   = tid >> 5;
    const int lid   = tid & 31;
    const int warp_n_base = wid * 32;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 128;
        int row = idx >> 3;
        int k8  = (idx & 7) * 8;
        *reinterpret_cast<float4*>(&sh_A[row][k8]) =
            *reinterpret_cast<const float4*>(&A[row * 64 + k8]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx      = tid + i * 128;
        int n_local  = idx >> 3;
        int k8       = (idx & 7) * 8;
        int n_global = cta_n + n_local;
        cp_async16(&sh_B[n_local][k8], &B_col[(int64_t)n_global * 64 + k8]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int row_in_warp = lid >> 2;
    const int kc          = (lid & 3) * 2;

    float acc[4][4][4];
    #pragma unroll
    for (int wm = 0; wm < 4; wm++)
        #pragma unroll
        for (int wn = 0; wn < 4; wn++)
            acc[wm][wn][0] = acc[wm][wn][1] = acc[wm][wn][2] = acc[wm][wn][3] = 0.f;

    uint32_t af_cur[4][4], af_nxt[4][4];
    uint32_t bf_cur[4][2], bf_nxt[4][2];

    #pragma unroll
    for (int wm = 0; wm < 4; wm++) {
        int r0 = wm*16 + row_in_warp, r1 = r0 + 8;
        af_cur[wm][0] = pack_half2(sh_A[r0][kc],   sh_A[r0][kc+1]);
        af_cur[wm][1] = pack_half2(sh_A[r1][kc],   sh_A[r1][kc+1]);
        af_cur[wm][2] = pack_half2(sh_A[r0][8+kc], sh_A[r0][8+kc+1]);
        af_cur[wm][3] = pack_half2(sh_A[r1][8+kc], sh_A[r1][8+kc+1]);
    }
    #pragma unroll
    for (int wn = 0; wn < 4; wn++) {
        int bn = warp_n_base + wn*8 + row_in_warp;
        bf_cur[wn][0] = pack_half2(sh_B[bn][kc],   sh_B[bn][kc+1]);
        bf_cur[wn][1] = pack_half2(sh_B[bn][8+kc], sh_B[bn][8+kc+1]);
    }

    #pragma unroll
    for (int wm = 0; wm < 4; wm++) {
        int r0 = wm*16 + row_in_warp, r1 = r0 + 8;
        af_nxt[wm][0] = pack_half2(sh_A[r0][16+kc], sh_A[r0][16+kc+1]);
        af_nxt[wm][1] = pack_half2(sh_A[r1][16+kc], sh_A[r1][16+kc+1]);
        af_nxt[wm][2] = pack_half2(sh_A[r0][24+kc], sh_A[r0][24+kc+1]);
        af_nxt[wm][3] = pack_half2(sh_A[r1][24+kc], sh_A[r1][24+kc+1]);
    }
    #pragma unroll
    for (int wn = 0; wn < 4; wn++) {
        int bn = warp_n_base + wn*8 + row_in_warp;
        bf_nxt[wn][0] = pack_half2(sh_B[bn][16+kc], sh_B[bn][16+kc+1]);
        bf_nxt[wn][1] = pack_half2(sh_B[bn][24+kc], sh_B[bn][24+kc+1]);
    }

    #pragma unroll
    for (int wn = 0; wn < 4; wn++)
        #pragma unroll
        for (int wm = 0; wm < 4; wm++)
            mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                         af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],
                         bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wm = 0; wm < 4; wm++) {
        af_cur[wm][0]=af_nxt[wm][0]; af_cur[wm][1]=af_nxt[wm][1];
        af_cur[wm][2]=af_nxt[wm][2]; af_cur[wm][3]=af_nxt[wm][3];
        int r0=wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][32+kc], sh_A[r0][32+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][32+kc], sh_A[r1][32+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][40+kc], sh_A[r0][40+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][40+kc], sh_A[r1][40+kc+1]);
    }
    #pragma unroll
    for (int wn = 0; wn < 4; wn++) {
        bf_cur[wn][0]=bf_nxt[wn][0]; bf_cur[wn][1]=bf_nxt[wn][1];
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][32+kc], sh_B[bn][32+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][40+kc], sh_B[bn][40+kc+1]);
    }
    #pragma unroll
    for (int wn = 0; wn < 4; wn++)
        #pragma unroll
        for (int wm = 0; wm < 4; wm++)
            mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                         af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],
                         bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wm = 0; wm < 4; wm++) {
        af_cur[wm][0]=af_nxt[wm][0]; af_cur[wm][1]=af_nxt[wm][1];
        af_cur[wm][2]=af_nxt[wm][2]; af_cur[wm][3]=af_nxt[wm][3];
        int r0=wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][48+kc], sh_A[r0][48+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][48+kc], sh_A[r1][48+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][56+kc], sh_A[r0][56+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][56+kc], sh_A[r1][56+kc+1]);
    }
    #pragma unroll
    for (int wn = 0; wn < 4; wn++) {
        bf_cur[wn][0]=bf_nxt[wn][0]; bf_cur[wn][1]=bf_nxt[wn][1];
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][48+kc], sh_B[bn][48+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][56+kc], sh_B[bn][56+kc+1]);
    }
    #pragma unroll
    for (int wn = 0; wn < 4; wn++)
        #pragma unroll
        for (int wm = 0; wm < 4; wm++)
            mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                         af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],
                         bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wn = 0; wn < 4; wn++)
        #pragma unroll
        for (int wm = 0; wm < 4; wm++)
            mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                         af_nxt[wm][0],af_nxt[wm][1],af_nxt[wm][2],af_nxt[wm][3],
                         bf_nxt[wn][0],bf_nxt[wn][1]);

    const int lane_col_off = (lid & 3) * 2;
    #pragma unroll
    for (int wm = 0; wm < 4; wm++) {
        int row0 = wm*16 + row_in_warp;
        int row1 = row0 + 8;
        #pragma unroll
        for (int wn = 0; wn < 4; wn++) {
            int col = cta_n + warp_n_base + wn*8 + lane_col_off;
            *reinterpret_cast<__half2*>(&C[(int64_t)row0*N + col]) =
                __floats2half2_rn(acc[wm][wn][0], acc[wm][wn][1]);
            *reinterpret_cast<__half2*>(&C[(int64_t)row1*N + col]) =
                __floats2half2_rn(acc[wm][wn][2], acc[wm][wn][3]);
        }
    }
}

__global__ void __launch_bounds__(256, 2)
gemm_v13_n256(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int N)
{
    __shared__ __align__(128) half sh_A[64][72];
    __shared__ __align__(128) half sh_B[256][72];

    const int cta_n = blockIdx.x * 256;
    const int tid   = threadIdx.x;
    const int wid   = tid >> 5;
    const int lid   = tid & 31;
    const int warp_n_base = wid * 32;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid + i*256, row = idx >> 3, k8 = (idx & 7)*8;
        *reinterpret_cast<float4*>(&sh_A[row][k8]) =
            *reinterpret_cast<const float4*>(&A[row*64+k8]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx=tid+i*256, n_local=idx>>3, k8=(idx&7)*8;
        cp_async16(&sh_B[n_local][k8], &B_col[(int64_t)(cta_n+n_local)*64+k8]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int row_in_warp = lid >> 2;
    const int kc          = (lid & 3) * 2;

    float acc[4][4][4];
    #pragma unroll
    for (int wm=0;wm<4;wm++) for (int wn=0;wn<4;wn++)
        acc[wm][wn][0]=acc[wm][wn][1]=acc[wm][wn][2]=acc[wm][wn][3]=0.f;

    uint32_t af_cur[4][4], af_nxt[4][4];
    uint32_t bf_cur[4][2], bf_nxt[4][2];

    #pragma unroll
    for (int wm=0;wm<4;wm++) {
        int r0=wm*16+row_in_warp, r1=r0+8;
        af_cur[wm][0]=pack_half2(sh_A[r0][kc],   sh_A[r0][kc+1]);
        af_cur[wm][1]=pack_half2(sh_A[r1][kc],   sh_A[r1][kc+1]);
        af_cur[wm][2]=pack_half2(sh_A[r0][8+kc], sh_A[r0][8+kc+1]);
        af_cur[wm][3]=pack_half2(sh_A[r1][8+kc], sh_A[r1][8+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_cur[wn][0]=pack_half2(sh_B[bn][kc],   sh_B[bn][kc+1]);
        bf_cur[wn][1]=pack_half2(sh_B[bn][8+kc], sh_B[bn][8+kc+1]);
    }
    #pragma unroll
    for (int wm=0;wm<4;wm++) {
        int r0=wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][16+kc], sh_A[r0][16+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][16+kc], sh_A[r1][16+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][24+kc], sh_A[r0][24+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][24+kc], sh_A[r1][24+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][16+kc], sh_B[bn][16+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][24+kc], sh_B[bn][24+kc+1]);
    }

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<4;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wm=0;wm<4;wm++) {
        af_cur[wm][0]=af_nxt[wm][0]; af_cur[wm][1]=af_nxt[wm][1];
        af_cur[wm][2]=af_nxt[wm][2]; af_cur[wm][3]=af_nxt[wm][3];
        int r0=wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][32+kc], sh_A[r0][32+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][32+kc], sh_A[r1][32+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][40+kc], sh_A[r0][40+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][40+kc], sh_A[r1][40+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        bf_cur[wn][0]=bf_nxt[wn][0]; bf_cur[wn][1]=bf_nxt[wn][1];
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][32+kc], sh_B[bn][32+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][40+kc], sh_B[bn][40+kc+1]);
    }

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<4;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wm=0;wm<4;wm++) {
        af_cur[wm][0]=af_nxt[wm][0]; af_cur[wm][1]=af_nxt[wm][1];
        af_cur[wm][2]=af_nxt[wm][2]; af_cur[wm][3]=af_nxt[wm][3];
        int r0=wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][48+kc], sh_A[r0][48+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][48+kc], sh_A[r1][48+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][56+kc], sh_A[r0][56+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][56+kc], sh_A[r1][56+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        bf_cur[wn][0]=bf_nxt[wn][0]; bf_cur[wn][1]=bf_nxt[wn][1];
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][48+kc], sh_B[bn][48+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][56+kc], sh_B[bn][56+kc+1]);
    }

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<4;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<4;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_nxt[wm][0],af_nxt[wm][1],af_nxt[wm][2],af_nxt[wm][3],bf_nxt[wn][0],bf_nxt[wn][1]);

    const int lane_col_off = (lid & 3) * 2;
    #pragma unroll
    for (int wm=0;wm<4;wm++) {
        int row0=wm*16+row_in_warp, row1=row0+8;
        #pragma unroll
        for (int wn=0;wn<4;wn++) {
            int col=cta_n+warp_n_base+wn*8+lane_col_off;
            *reinterpret_cast<__half2*>(&C[(int64_t)row0*N+col]) =
                __floats2half2_rn(acc[wm][wn][0],acc[wm][wn][1]);
            *reinterpret_cast<__half2*>(&C[(int64_t)row1*N+col]) =
                __floats2half2_rn(acc[wm][wn][2],acc[wm][wn][3]);
        }
    }
}

__global__ void __launch_bounds__(256, 4)
gemm_v13_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __align__(128) half sh_A[64][72];
    __shared__ __align__(128) half sh_B[128][72];

    const int cta_n = blockIdx.x * 128;
    const int cta_m = blockIdx.y * 64;
    const int tid   = threadIdx.x;
    const int wid   = tid >> 5;
    const int lid   = tid & 31;

    const int warp_row    = wid >> 2;
    const int warp_col    = wid & 3;
    const int warp_m_base = warp_row * 32;
    const int warp_n_base = warp_col * 32;

    #pragma unroll
    for (int i=0;i<2;i++) {
        int idx=tid+i*256, m_local=idx>>3, k8=(idx&7)*8;
        int m_global=cta_m+m_local;
        float4 val=make_float4(0,0,0,0);
        if (m_local<64 && m_global<M && k8+8<=K)
            val=*reinterpret_cast<const float4*>(&A[m_global*K+k8]);
        if (m_local<64)
            *reinterpret_cast<float4*>(&sh_A[m_local][k8])=val;
    }
    #pragma unroll
    for (int i=0;i<4;i++) {
        int idx=tid+i*256, n_local=idx>>3, k8=(idx&7)*8;
        int n_global=cta_n+n_local;
        if (n_local<128) {
            if (n_global<N && k8+8<=K)
                cp_async16(&sh_B[n_local][k8], &B_col[(int64_t)n_global*K+k8]);
            else
                *reinterpret_cast<float4*>(&sh_B[n_local][k8])=make_float4(0,0,0,0);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    const int row_in_warp = lid >> 2;
    const int kc          = (lid & 3) * 2;

    float acc[2][4][4];
    #pragma unroll
    for (int wm=0;wm<2;wm++) for (int wn=0;wn<4;wn++)
        acc[wm][wn][0]=acc[wm][wn][1]=acc[wm][wn][2]=acc[wm][wn][3]=0.f;

    uint32_t af_cur[2][4], af_nxt[2][4];
    uint32_t bf_cur[4][2], bf_nxt[4][2];

    #pragma unroll
    for (int wm=0;wm<2;wm++) {
        int r0=warp_m_base+wm*16+row_in_warp, r1=r0+8;
        af_cur[wm][0]=pack_half2(sh_A[r0][kc],   sh_A[r0][kc+1]);
        af_cur[wm][1]=pack_half2(sh_A[r1][kc],   sh_A[r1][kc+1]);
        af_cur[wm][2]=pack_half2(sh_A[r0][8+kc], sh_A[r0][8+kc+1]);
        af_cur[wm][3]=pack_half2(sh_A[r1][8+kc], sh_A[r1][8+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_cur[wn][0]=pack_half2(sh_B[bn][kc],   sh_B[bn][kc+1]);
        bf_cur[wn][1]=pack_half2(sh_B[bn][8+kc], sh_B[bn][8+kc+1]);
    }
    #pragma unroll
    for (int wm=0;wm<2;wm++) {
        int r0=warp_m_base+wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][16+kc], sh_A[r0][16+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][16+kc], sh_A[r1][16+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][24+kc], sh_A[r0][24+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][24+kc], sh_A[r1][24+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][16+kc], sh_B[bn][16+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][24+kc], sh_B[bn][24+kc+1]);
    }

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<2;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wm=0;wm<2;wm++) {
        af_cur[wm][0]=af_nxt[wm][0]; af_cur[wm][1]=af_nxt[wm][1];
        af_cur[wm][2]=af_nxt[wm][2]; af_cur[wm][3]=af_nxt[wm][3];
        int r0=warp_m_base+wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][32+kc], sh_A[r0][32+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][32+kc], sh_A[r1][32+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][40+kc], sh_A[r0][40+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][40+kc], sh_A[r1][40+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        bf_cur[wn][0]=bf_nxt[wn][0]; bf_cur[wn][1]=bf_nxt[wn][1];
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][32+kc], sh_B[bn][32+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][40+kc], sh_B[bn][40+kc+1]);
    }

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<2;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wm=0;wm<2;wm++) {
        af_cur[wm][0]=af_nxt[wm][0]; af_cur[wm][1]=af_nxt[wm][1];
        af_cur[wm][2]=af_nxt[wm][2]; af_cur[wm][3]=af_nxt[wm][3];
        int r0=warp_m_base+wm*16+row_in_warp, r1=r0+8;
        af_nxt[wm][0]=pack_half2(sh_A[r0][48+kc], sh_A[r0][48+kc+1]);
        af_nxt[wm][1]=pack_half2(sh_A[r1][48+kc], sh_A[r1][48+kc+1]);
        af_nxt[wm][2]=pack_half2(sh_A[r0][56+kc], sh_A[r0][56+kc+1]);
        af_nxt[wm][3]=pack_half2(sh_A[r1][56+kc], sh_A[r1][56+kc+1]);
    }
    #pragma unroll
    for (int wn=0;wn<4;wn++) {
        bf_cur[wn][0]=bf_nxt[wn][0]; bf_cur[wn][1]=bf_nxt[wn][1];
        int bn=warp_n_base+wn*8+row_in_warp;
        bf_nxt[wn][0]=pack_half2(sh_B[bn][48+kc], sh_B[bn][48+kc+1]);
        bf_nxt[wn][1]=pack_half2(sh_B[bn][56+kc], sh_B[bn][56+kc+1]);
    }

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<2;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_cur[wm][0],af_cur[wm][1],af_cur[wm][2],af_cur[wm][3],bf_cur[wn][0],bf_cur[wn][1]);

    #pragma unroll
    for (int wn=0;wn<4;wn++) for (int wm=0;wm<2;wm++)
        mma_m16n8k16(acc[wm][wn][0],acc[wm][wn][1],acc[wm][wn][2],acc[wm][wn][3],
                     af_nxt[wm][0],af_nxt[wm][1],af_nxt[wm][2],af_nxt[wm][3],bf_nxt[wn][0],bf_nxt[wn][1]);

    const int lane_col_off = (lid & 3) * 2;
    #pragma unroll
    for (int wm=0;wm<2;wm++) {
        int row0=cta_m+warp_m_base+wm*16+row_in_warp;
        int row1=row0+8;
        #pragma unroll
        for (int wn=0;wn<4;wn++) {
            int col=cta_n+warp_n_base+wn*8+lane_col_off;
            if (row0<M) {
                if (col  <N) C[(int64_t)row0*N+col]   = __float2half(acc[wm][wn][0]);
                if (col+1<N) C[(int64_t)row0*N+col+1] = __float2half(acc[wm][wn][1]);
            }
            if (row1<M) {
                if (col  <N) C[(int64_t)row1*N+col]   = __float2half(acc[wm][wn][2]);
                if (col+1<N) C[(int64_t)row1*N+col+1] = __float2half(acc[wm][wn][3]);
            }
        }
    }
}

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr());

    if (M == 64 && K == 64) {
        if (N % 128 == 0) {
            dim3 grid(N / 128, 1);
            gemm_v13_primary<<<grid, 128>>>(A_ptr, B_ptr, C_ptr, N);
        } else if (N % 256 == 0) {
            dim3 grid(N / 256, 1);
            gemm_v13_n256<<<grid, 256>>>(A_ptr, B_ptr, C_ptr, N);
        } else {
            dim3 grid((N+127)/128, 1);
            gemm_v13_fallback<<<grid, 256>>>(A_ptr, B_ptr, C_ptr, M, N, K);
        }
    } else {
        dim3 grid((N+127)/128, (M+63)/64);
        gemm_v13_fallback<<<grid, 256>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    }
}