#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <cuda_pipeline_primitives.h>

#define BM_A 64
#define BN_A 64
#define BK_A 32
#define STAGES_A 4
#define SA_STRIDE (BK_A + 8)
#define SB_STRIDE (BK_A + 8)

static __device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }"
                 : "=r"(addr) : "l"(ptr));
    return addr;
}

static __device__ __forceinline__ void async16(void* d, const void* s) {
    uint32_t da = smem_addr(d);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(da), "l"(s) : "memory");
}

static __device__ __forceinline__ void commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N> static __device__ __forceinline__ void wait_n() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

static __device__ __forceinline__ void wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

static __device__ __forceinline__ void mma16x8x16(
    float* d, const uint32_t* a, const uint32_t* b, const float* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
          "r"(b[0]),"r"(b[1]),
          "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

static __device__ __forceinline__ uint32_t ph2(half a, half b) {
    uint32_t r;
    asm volatile("{ .reg .b16 x,y; mov.b16 x,%1; mov.b16 y,%2; mov.b32 %0,{x,y}; }"
                 : "=r"(r) : "h"(__half_as_ushort(a)), "h"(__half_as_ushort(b)));
    return r;
}

static __device__ __forceinline__ void ldmx4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    uint32_t a = smem_addr(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}

static __device__ __forceinline__ void zero8(half* dst) {
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = __float2half(0.f);
}

static __device__ __forceinline__ void load8_safe(half* dst, const half* src, int gm, int gk, int M, int K) {
    if (gm < M) {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            dst[i] = (gk + i < K) ? src[gm * K + gk + i] : __float2half(0.f);
    } else {
        zero8(dst);
    }
}

static __device__ __forceinline__ void loadB8_safe(half* dst, const half* B_col, int n, int gk, int N, int K) {
    if (n < N) {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            dst[i] = (gk + i < K) ? B_col[n * K + gk + i] : __float2half(0.f);
    }
}

__global__ __launch_bounds__(256, 4)
void hgemm_kernel_v4(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];
    const int smA_sz = BM_A * SA_STRIDE;
    const int smB_sz = BN_A * SB_STRIDE;
    half* smA = smem;
    half* smB = smem + STAGES_A * smA_sz;

    int bm   = blockIdx.x;
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;
    int row_base = bm * BM_A;

    int warp_m   = wid >> 2;
    int warp_n   = wid & 3;
    int warp_row = warp_m * 32;
    int warp_col = warp_n * 16;

    float acc[2][2][4];
    #pragma unroll
    for (int mi=0;mi<2;mi++) for(int ni=0;ni<2;ni++) for(int f=0;f<4;f++) acc[mi][ni][f]=0.f;

    int nkt = K / BK_A;

    auto issue = [&](int st, int kb) {
        int k_off = kb * BK_A;
        half* dA = smA + st * smA_sz;
        half* dB = smB + st * smB_sz;
        {
            int row = tid >> 2, ksub = (tid & 3) << 3;
            int gm = row_base + row, gk = k_off + ksub;
            half* dst = dA + row * SA_STRIDE + ksub;
            if (gm < M && gk + 7 < K)
                async16(dst, A + gm * K + gk);
            else
                load8_safe(dst, A, gm, gk, M, K);
        }
        {
            int n = tid >> 2, ksub = (tid & 3) << 3;
            int gk = k_off + ksub;
            half* dst = dB + n * SB_STRIDE + ksub;
            if (n < N && gk + 7 < K)
                async16(dst, B_col + n * K + gk);
            else
                loadB8_safe(dst, B_col, n, gk, N, K);
        }
        commit();
    };

    #pragma unroll
    for (int s=0;s<STAGES_A-1;s++) {
        if (s < nkt) issue(s, s); else commit();
    }

    #pragma unroll 1
    for (int kb=0; kb<nkt; kb++) {
        int stage = kb % STAGES_A;
        int nkb   = kb + STAGES_A - 1;
        if (nkb < nkt) issue(nkb % STAGES_A, nkb);
        else commit();

        wait_n<STAGES_A-1>();
        __syncthreads();

        half* cA = smA + stage * smA_sz;
        half* cB = smB + stage * smB_sz;

        #pragma unroll
        for (int kk=0; kk<BK_A; kk+=16) {
            uint32_t af[2][4];
            #pragma unroll
            for (int mi=0; mi<2; mi++) {
                int tr  = warp_row + mi * 16;
                int grp = lane >> 3;
                int li  = lane & 7;
                int row_off = (grp & 1) * 8 + li;
                int k_off2  = (grp >> 1) * 8;
                const half* ptr = cA + (tr + row_off) * SA_STRIDE + kk + k_off2;
                ldmx4(af[mi][0], af[mi][1], af[mi][2], af[mi][3], ptr);
            }

            uint32_t bf[2][2];
            #pragma unroll
            for (int ni=0; ni<2; ni++) {
                int nc = warp_col + ni*8 + (lane>>2);
                int k0=(lane&3)<<1, k1=k0+1, k8=k0+8, k9=k8+1;
                bf[ni][0] = ph2(cB[nc*SB_STRIDE+kk+k0], cB[nc*SB_STRIDE+kk+k1]);
                bf[ni][1] = ph2(cB[nc*SB_STRIDE+kk+k8], cB[nc*SB_STRIDE+kk+k9]);
            }

            #pragma unroll
            for (int mi=0;mi<2;mi++)
                #pragma unroll
                for (int ni=0;ni<2;ni++)
                    mma16x8x16(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }
    }

    wait_all();
    __syncthreads();

    int r0 = lane >> 2, r1 = r0+8, c0 = (lane&3)<<1;
    #pragma unroll
    for (int mi=0;mi<2;mi++) {
        int gr0 = row_base + warp_row + mi*16 + r0;
        int gr1 = row_base + warp_row + mi*16 + r1;
        #pragma unroll
        for (int ni=0;ni<2;ni++) {
            int gc = warp_col + ni*8 + c0;
            if (gr0<M && gc+1<=N)
                *reinterpret_cast<__half2*>(C+gr0*N+gc)=__floats2half2_rn(acc[mi][ni][0],acc[mi][ni][1]);
            else if (gr0<M) {
                if (gc<N)   C[gr0*N+gc]  =__float2half(acc[mi][ni][0]);
                if (gc+1<N) C[gr0*N+gc+1]=__float2half(acc[mi][ni][1]);
            }
            if (gr1<M && gc+1<=N)
                *reinterpret_cast<__half2*>(C+gr1*N+gc)=__floats2half2_rn(acc[mi][ni][2],acc[mi][ni][3]);
            else if (gr1<M) {
                if (gc<N)   C[gr1*N+gc]  =__float2half(acc[mi][ni][2]);
                if (gc+1<N) C[gr1*N+gc+1]=__float2half(acc[mi][ni][3]);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    const half* A     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C           = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    int gridM = (M + BM_A - 1) / BM_A;
    dim3 grid(gridM), block(256);
    int smem = STAGES_A * (BM_A * SA_STRIDE + BN_A * SB_STRIDE) * (int)sizeof(half);
    cudaFuncSetAttribute(hgemm_kernel_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    hgemm_kernel_v4<<<grid, block, smem>>>(A, B_col, C, M, N, K);
}