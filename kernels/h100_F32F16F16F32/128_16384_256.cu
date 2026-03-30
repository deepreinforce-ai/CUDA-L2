#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <cuda.h>

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"((uint64_t)ptr));
    return addr;
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
        :: "r"(dst), "l"((uint64_t)src));
}

#define BM 128
#define BN 128
#define BK 32
#define ST 8
#define SA 32
#define SB 128
#define WM 64
#define WN 64
#define WMT 4
#define WNT 8
#define WKT 2

__global__ void __launch_bounds__(128, 2)
kern_main(const half* __restrict__ A,
          const half* __restrict__ B,
          half* __restrict__ C,
          int N, int K)
{
    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + ST * BM * SA;

    const int bx = blockIdx.x;
    const int bn0 = bx * BN;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lid = tid & 31;
    const int wm = wid >> 1, wn = wid & 1;
    const int nk = K / BK;

    float acc[WMT][WNT][4];
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++)
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto ldA = [&](int s, int kb) __attribute__((always_inline)) {
        half* sa = sA + s * BM * SA;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 128;
            int r = f >> 2, c = (f & 3) << 3;
            int sc = c ^ ((r & 3) << 3);
            cp16(smem_u32(&sa[r * SA + sc]), A + r * K + kb + c);
        }
    };

    auto ldB = [&](int s, int kb) __attribute__((always_inline)) {
        half* sb = sB + s * BK * SB;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 128;
            int r = f >> 4, c = (f & 15) << 3;
            int sc = c ^ ((r & 7) << 3);
            cp16(smem_u32(&sb[r * SB + sc]), B + (kb + r) * N + bn0 + c);
        }
    };

    #pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < nk) { ldA(s, s * BK); ldB(s, s * BK); }
        asm volatile("cp.async.commit_group;");
    }

    int sr = 0;
    #pragma unroll 2
    for (int kt = 0; kt < nk; kt++) {
        int sw = (kt + ST - 1) % ST;
        if (kt + ST - 1 < nk) {
            ldA(sw, (kt + ST - 1) * BK);
            ldB(sw, (kt + ST - 1) * BK);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1));
        __syncthreads();

        half* sa = sA + sr * BM * SA;
        half* sb = sB + sr * BK * SB;

        uint32_t af0[WMT][4], bf0[WNT][2];
        uint32_t af1[WMT][4], bf1[WNT][2];

        #pragma unroll
        for (int mi = 0; mi < WMT; mi++) {
            int row = wm * WM + mi * 16 + (lid & 15);
            int col = 0 * 16 + ((lid >> 4) << 3);
            int sc = col ^ ((row & 3) << 3);
            uint32_t addr = smem_u32(&sa[row * SA + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                : "=r"(af0[mi][0]),"=r"(af0[mi][1]),"=r"(af0[mi][2]),"=r"(af0[mi][3])
                : "r"(addr));
        }
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++) {
            int row = 0 * 16 + (lid & 15);
            int col = wn * WN + ni * 8;
            int sc = col ^ ((row & 7) << 3);
            uint32_t addr = smem_u32(&sb[row * SB + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                : "=r"(bf0[ni][0]),"=r"(bf0[ni][1]) : "r"(addr));
        }

        #pragma unroll
        for (int mi = 0; mi < WMT; mi++) {
            int row = wm * WM + mi * 16 + (lid & 15);
            int col = 1 * 16 + ((lid >> 4) << 3);
            int sc = col ^ ((row & 3) << 3);
            uint32_t addr = smem_u32(&sa[row * SA + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                : "=r"(af1[mi][0]),"=r"(af1[mi][1]),"=r"(af1[mi][2]),"=r"(af1[mi][3])
                : "r"(addr));
        }
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++) {
            int row = 1 * 16 + (lid & 15);
            int col = wn * WN + ni * 8;
            int sc = col ^ ((row & 7) << 3);
            uint32_t addr = smem_u32(&sb[row * SB + sc]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                : "=r"(bf1[ni][0]),"=r"(bf1[ni][1]) : "r"(addr));
        }

        #pragma unroll
        for (int mi = 0; mi < WMT; mi++)
            #pragma unroll
            for (int ni = 0; ni < WNT; ni++)
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                    : "r"(af0[mi][0]),"r"(af0[mi][1]),"r"(af0[mi][2]),"r"(af0[mi][3]),
                      "r"(bf0[ni][0]),"r"(bf0[ni][1]),
                      "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));

        #pragma unroll
        for (int mi = 0; mi < WMT; mi++)
            #pragma unroll
            for (int ni = 0; ni < WNT; ni++)
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                    : "r"(af1[mi][0]),"r"(af1[mi][1]),"r"(af1[mi][2]),"r"(af1[mi][3]),
                      "r"(bf1[ni][0]),"r"(bf1[ni][1]),
                      "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));

        sr = (sr + 1) % ST;
    }

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    half* Cout = C + bn0;
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++) {
        int r0 = wm * WM + mi * 16 + (lid >> 2);
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++) {
            int c0 = wn * WN + ni * 8 + (lid & 3) * 2;
            *reinterpret_cast<half2*>(&Cout[r0 * N + c0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&Cout[r1 * N + c0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(128, 2)
kern_smem_epi(const half* __restrict__ A,
              const half* __restrict__ B,
              half* __restrict__ C,
              int N, int K)
{
    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + ST * BM * SA;

    const int bx = blockIdx.x;
    const int bn0 = bx * BN;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lid = tid & 31;
    const int wm = wid >> 1, wn = wid & 1;
    const int nk = K / BK;

    float acc[WMT][WNT][4];
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++)
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto ldA = [&](int s, int kb) __attribute__((always_inline)) {
        half* sa = sA + s * BM * SA;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 128;
            int r = f >> 2, c = (f & 3) << 3;
            int sc = c ^ ((r & 3) << 3);
            cp16(smem_u32(&sa[r * SA + sc]), A + r * K + kb + c);
        }
    };

    auto ldB = [&](int s, int kb) __attribute__((always_inline)) {
        half* sb = sB + s * BK * SB;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 128;
            int r = f >> 4, c = (f & 15) << 3;
            int sc = c ^ ((r & 7) << 3);
            cp16(smem_u32(&sb[r * SB + sc]), B + (kb + r) * N + bn0 + c);
        }
    };

    #pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < nk) { ldA(s, s * BK); ldB(s, s * BK); }
        asm volatile("cp.async.commit_group;");
    }

    int sr = 0;
    #pragma unroll 2
    for (int kt = 0; kt < nk; kt++) {
        int sw = (kt + ST - 1) % ST;
        if (kt + ST - 1 < nk) {
            ldA(sw, (kt + ST - 1) * BK);
            ldB(sw, (kt + ST - 1) * BK);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1));
        __syncthreads();

        half* sa = sA + sr * BM * SA;
        half* sb = sB + sr * BK * SB;

        uint32_t af[WKT][WMT][4], bf[WKT][WNT][2];
        #pragma unroll
        for (int kk = 0; kk < WKT; kk++) {
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++) {
                int row = wm * WM + mi * 16 + (lid & 15);
                int col = kk * 16 + ((lid >> 4) << 3);
                int sc = col ^ ((row & 3) << 3);
                uint32_t addr = smem_u32(&sa[row * SA + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(af[kk][mi][0]),"=r"(af[kk][mi][1]),"=r"(af[kk][mi][2]),"=r"(af[kk][mi][3])
                    : "r"(addr));
            }
            #pragma unroll
            for (int ni = 0; ni < WNT; ni++) {
                int row = kk * 16 + (lid & 15);
                int col = wn * WN + ni * 8;
                int sc = col ^ ((row & 7) << 3);
                uint32_t addr = smem_u32(&sb[row * SB + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                    : "=r"(bf[kk][ni][0]),"=r"(bf[kk][ni][1]) : "r"(addr));
            }
        }

        #pragma unroll
        for (int kk = 0; kk < WKT; kk++)
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++)
                #pragma unroll
                for (int ni = 0; ni < WNT; ni++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                        : "r"(af[kk][mi][0]),"r"(af[kk][mi][1]),"r"(af[kk][mi][2]),"r"(af[kk][mi][3]),
                          "r"(bf[kk][ni][0]),"r"(bf[kk][ni][1]),
                          "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));

        sr = (sr + 1) % ST;
    }

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    half* so = reinterpret_cast<half*>(smem_raw);
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++) {
        int r0 = wm * WM + mi * 16 + (lid >> 2);
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++) {
            int c0 = wn * WN + ni * 8 + (lid & 3) * 2;
            so[r0 * BN + c0]     = __float2half(acc[mi][ni][0]);
            so[r0 * BN + c0 + 1] = __float2half(acc[mi][ni][1]);
            so[r1 * BN + c0]     = __float2half(acc[mi][ni][2]);
            so[r1 * BN + c0 + 1] = __float2half(acc[mi][ni][3]);
        }
    }
    __syncthreads();

    half* Cout = C + bn0;
    #pragma unroll
    for (int p = 0; p < 16; p++) {
        int f = tid + p * 128;
        int row = f >> 4, col = (f & 15) << 3;
        *reinterpret_cast<float4*>(&Cout[row * N + col]) =
            *reinterpret_cast<const float4*>(&so[row * BN + col]);
    }
}

#define BN3 256
#define SB3 256
#define WN3 64
#define WNT3 8

__global__ void __launch_bounds__(256, 1)
kern_bn256(const half* __restrict__ A,
           const half* __restrict__ B,
           half* __restrict__ C,
           int N, int K)
{
    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + ST * BM * SA;

    const int bx = blockIdx.x;
    const int bn0 = bx * BN3;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lid = tid & 31;
    const int wm = wid >> 2, wn = wid & 3;
    const int nk = K / BK;

    float acc[WMT][WNT3][4];
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++)
        #pragma unroll
        for (int ni = 0; ni < WNT3; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto ldA = [&](int s, int kb) __attribute__((always_inline)) {
        half* sa = sA + s * BM * SA;
        #pragma unroll
        for (int p = 0; p < 2; p++) {
            int f = tid + p * 256;
            int r = f >> 2, c = (f & 3) << 3;
            int sc = c ^ ((r & 3) << 3);
            cp16(smem_u32(&sa[r * SA + sc]), A + r * K + kb + c);
        }
    };

    auto ldB = [&](int s, int kb) __attribute__((always_inline)) {
        half* sb = sB + s * BK * SB3;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 256;
            int r = f >> 5, c = (f & 31) << 3;
            int sc = c ^ ((r & 7) << 3);
            cp16(smem_u32(&sb[r * SB3 + sc]), B + (kb + r) * N + bn0 + c);
        }
    };

    #pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < nk) { ldA(s, s * BK); ldB(s, s * BK); }
        asm volatile("cp.async.commit_group;");
    }

    int sr = 0;
    #pragma unroll 2
    for (int kt = 0; kt < nk; kt++) {
        int sw = (kt + ST - 1) % ST;
        if (kt + ST - 1 < nk) {
            ldA(sw, (kt + ST - 1) * BK);
            ldB(sw, (kt + ST - 1) * BK);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(ST - 1));
        __syncthreads();

        half* sa = sA + sr * BM * SA;
        half* sb = sB + sr * BK * SB3;

        uint32_t af[WKT][WMT][4], bf[WKT][WNT3][2];
        #pragma unroll
        for (int kk = 0; kk < WKT; kk++) {
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++) {
                int row = wm * WM + mi * 16 + (lid & 15);
                int col = kk * 16 + ((lid >> 4) << 3);
                int sc = col ^ ((row & 3) << 3);
                uint32_t addr = smem_u32(&sa[row * SA + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(af[kk][mi][0]),"=r"(af[kk][mi][1]),"=r"(af[kk][mi][2]),"=r"(af[kk][mi][3])
                    : "r"(addr));
            }
            #pragma unroll
            for (int ni = 0; ni < WNT3; ni++) {
                int row = kk * 16 + (lid & 15);
                int col = wn * WN3 + ni * 8;
                int sc = col ^ ((row & 7) << 3);
                uint32_t addr = smem_u32(&sb[row * SB3 + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                    : "=r"(bf[kk][ni][0]),"=r"(bf[kk][ni][1]) : "r"(addr));
            }
        }

        #pragma unroll
        for (int kk = 0; kk < WKT; kk++)
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++)
                #pragma unroll
                for (int ni = 0; ni < WNT3; ni++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                        : "r"(af[kk][mi][0]),"r"(af[kk][mi][1]),"r"(af[kk][mi][2]),"r"(af[kk][mi][3]),
                          "r"(bf[kk][ni][0]),"r"(bf[kk][ni][1]),
                          "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));

        sr = (sr + 1) % ST;
    }

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    half* Cout = C + bn0;
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++) {
        int r0 = wm * WM + mi * 16 + (lid >> 2);
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WNT3; ni++) {
            int c0 = wn * WN3 + ni * 8 + (lid & 3) * 2;
            *reinterpret_cast<half2*>(&Cout[r0 * N + c0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&Cout[r1 * N + c0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

#define ST4 4

__global__ void __launch_bounds__(128, 4)
kern_4stage(const half* __restrict__ A,
            const half* __restrict__ B,
            half* __restrict__ C,
            int N, int K)
{
    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + ST4 * BM * SA;

    const int bx = blockIdx.x;
    const int bn0 = bx * BN;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lid = tid & 31;
    const int wm = wid >> 1, wn = wid & 1;
    const int nk = K / BK;

    float acc[WMT][WNT][4];
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++)
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto ldA = [&](int s, int kb) __attribute__((always_inline)) {
        half* sa = sA + s * BM * SA;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 128;
            int r = f >> 2, c = (f & 3) << 3;
            int sc = c ^ ((r & 3) << 3);
            cp16(smem_u32(&sa[r * SA + sc]), A + r * K + kb + c);
        }
    };

    auto ldB = [&](int s, int kb) __attribute__((always_inline)) {
        half* sb = sB + s * BK * SB;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 128;
            int r = f >> 4, c = (f & 15) << 3;
            int sc = c ^ ((r & 7) << 3);
            cp16(smem_u32(&sb[r * SB + sc]), B + (kb + r) * N + bn0 + c);
        }
    };

    #pragma unroll
    for (int s = 0; s < ST4 - 1; s++) {
        if (s < nk) { ldA(s, s * BK); ldB(s, s * BK); }
        asm volatile("cp.async.commit_group;");
    }

    int sr = 0;
    #pragma unroll 4
    for (int kt = 0; kt < nk; kt++) {
        int sw = (kt + ST4 - 1) % ST4;
        if (kt + ST4 - 1 < nk) {
            ldA(sw, (kt + ST4 - 1) * BK);
            ldB(sw, (kt + ST4 - 1) * BK);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(ST4 - 1));
        __syncthreads();

        half* sa = sA + sr * BM * SA;
        half* sb = sB + sr * BK * SB;

        uint32_t af[WKT][WMT][4], bf[WKT][WNT][2];
        #pragma unroll
        for (int kk = 0; kk < WKT; kk++) {
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++) {
                int row = wm * WM + mi * 16 + (lid & 15);
                int col = kk * 16 + ((lid >> 4) << 3);
                int sc = col ^ ((row & 3) << 3);
                uint32_t addr = smem_u32(&sa[row * SA + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(af[kk][mi][0]),"=r"(af[kk][mi][1]),"=r"(af[kk][mi][2]),"=r"(af[kk][mi][3])
                    : "r"(addr));
            }
            #pragma unroll
            for (int ni = 0; ni < WNT; ni++) {
                int row = kk * 16 + (lid & 15);
                int col = wn * WN + ni * 8;
                int sc = col ^ ((row & 7) << 3);
                uint32_t addr = smem_u32(&sb[row * SB + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                    : "=r"(bf[kk][ni][0]),"=r"(bf[kk][ni][1]) : "r"(addr));
            }
        }

        #pragma unroll
        for (int kk = 0; kk < WKT; kk++)
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++)
                #pragma unroll
                for (int ni = 0; ni < WNT; ni++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                        : "r"(af[kk][mi][0]),"r"(af[kk][mi][1]),"r"(af[kk][mi][2]),"r"(af[kk][mi][3]),
                          "r"(bf[kk][ni][0]),"r"(bf[kk][ni][1]),
                          "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));

        sr = (sr + 1) % ST4;
    }

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    half* Cout = C + bn0;
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++) {
        int r0 = wm * WM + mi * 16 + (lid >> 2);
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++) {
            int c0 = wn * WN + ni * 8 + (lid & 3) * 2;
            *reinterpret_cast<half2*>(&Cout[r0 * N + c0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&Cout[r1 * N + c0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

#define BK5 64
#define SA5 64
#define SB5 128
#define ST5 4
#define WKT5 4

__global__ void __launch_bounds__(128, 2)
kern_bk64(const half* __restrict__ A,
          const half* __restrict__ B,
          half* __restrict__ C,
          int N, int K)
{
    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + ST5 * BM * SA5;

    const int bx = blockIdx.x;
    const int bn0 = bx * BN;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lid = tid & 31;
    const int wm = wid >> 1, wn = wid & 1;
    const int nk = K / BK5;

    float acc[WMT][WNT][4];
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++)
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto ldA = [&](int s, int kb) __attribute__((always_inline)) {
        half* sa = sA + s * BM * SA5;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int f = tid + p * 128;
            int r = f >> 3, c = (f & 7) << 3;
            int sc = c ^ ((r & 3) << 3);
            cp16(smem_u32(&sa[r * SA5 + sc]), A + r * K + kb + c);
        }
    };

    auto ldB = [&](int s, int kb) __attribute__((always_inline)) {
        half* sb = sB + s * BK5 * SB5;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int f = tid + p * 128;
            int r = f >> 4, c = (f & 15) << 3;
            int sc = c ^ ((r & 7) << 3);
            cp16(smem_u32(&sb[r * SB5 + sc]), B + (kb + r) * N + bn0 + c);
        }
    };

    #pragma unroll
    for (int s = 0; s < ST5 - 1; s++) {
        if (s < nk) { ldA(s, s * BK5); ldB(s, s * BK5); }
        asm volatile("cp.async.commit_group;");
    }

    int sr = 0;
    for (int kt = 0; kt < nk; kt++) {
        int sw = (kt + ST5 - 1) % ST5;
        if (kt + ST5 - 1 < nk) {
            ldA(sw, (kt + ST5 - 1) * BK5);
            ldB(sw, (kt + ST5 - 1) * BK5);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(ST5 - 1));
        __syncthreads();

        half* sa = sA + sr * BM * SA5;
        half* sb = sB + sr * BK5 * SB5;

        uint32_t af[WKT5][WMT][4], bf[WKT5][WNT][2];
        #pragma unroll
        for (int kk = 0; kk < WKT5; kk++) {
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++) {
                int row = wm * WM + mi * 16 + (lid & 15);
                int col = kk * 16 + ((lid >> 4) << 3);
                int sc = col ^ ((row & 3) << 3);
                uint32_t addr = smem_u32(&sa[row * SA5 + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(af[kk][mi][0]),"=r"(af[kk][mi][1]),"=r"(af[kk][mi][2]),"=r"(af[kk][mi][3])
                    : "r"(addr));
            }
            #pragma unroll
            for (int ni = 0; ni < WNT; ni++) {
                int row = kk * 16 + (lid & 15);
                int col = wn * WN + ni * 8;
                int sc = col ^ ((row & 7) << 3);
                uint32_t addr = smem_u32(&sb[row * SB5 + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                    : "=r"(bf[kk][ni][0]),"=r"(bf[kk][ni][1]) : "r"(addr));
            }
        }

        #pragma unroll
        for (int kk = 0; kk < WKT5; kk++)
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++)
                #pragma unroll
                for (int ni = 0; ni < WNT; ni++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                        : "r"(af[kk][mi][0]),"r"(af[kk][mi][1]),"r"(af[kk][mi][2]),"r"(af[kk][mi][3]),
                          "r"(bf[kk][ni][0]),"r"(bf[kk][ni][1]),
                          "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));

        sr = (sr + 1) % ST5;
    }

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    half* Cout = C + bn0;
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++) {
        int r0 = wm * WM + mi * 16 + (lid >> 2);
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WNT; ni++) {
            int c0 = wn * WN + ni * 8 + (lid & 3) * 2;
            *reinterpret_cast<half2*>(&Cout[r0 * N + c0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&Cout[r1 * N + c0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

#define BN6 256
#define SA6 64
#define SB6 256
#define ST6 4
#define WKT6 4
#define WN6 64
#define WNT6 8

__global__ void __launch_bounds__(256, 1)
kern_bn256_bk64(const half* __restrict__ A,
                const half* __restrict__ B,
                half* __restrict__ C,
                int N, int K)
{
    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + ST6 * BM * SA6;

    const int bx = blockIdx.x;
    const int bn0 = bx * BN6;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lid = tid & 31;
    const int wm = wid >> 2, wn = wid & 3;
    const int nk = K / BK5;

    float acc[WMT][WNT6][4];
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++)
        #pragma unroll
        for (int ni = 0; ni < WNT6; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto ldA = [&](int s, int kb) __attribute__((always_inline)) {
        half* sa = sA + s * BM * SA6;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f = tid + p * 256;
            int r = f >> 3, c = (f & 7) << 3;
            int sc = c ^ ((r & 3) << 3);
            cp16(smem_u32(&sa[r * SA6 + sc]), A + r * K + kb + c);
        }
    };

    auto ldB = [&](int s, int kb) __attribute__((always_inline)) {
        half* sb = sB + s * BK5 * SB6;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int f = tid + p * 256;
            int r = f >> 5, c = (f & 31) << 3;
            int sc = c ^ ((r & 7) << 3);
            cp16(smem_u32(&sb[r * SB6 + sc]), B + (kb + r) * N + bn0 + c);
        }
    };

    #pragma unroll
    for (int s = 0; s < ST6 - 1; s++) {
        if (s < nk) { ldA(s, s * BK5); ldB(s, s * BK5); }
        asm volatile("cp.async.commit_group;");
    }

    int sr = 0;
    for (int kt = 0; kt < nk; kt++) {
        int sw = (kt + ST6 - 1) % ST6;
        if (kt + ST6 - 1 < nk) {
            ldA(sw, (kt + ST6 - 1) * BK5);
            ldB(sw, (kt + ST6 - 1) * BK5);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" :: "n"(ST6 - 1));
        __syncthreads();

        half* sa = sA + sr * BM * SA6;
        half* sb = sB + sr * BK5 * SB6;

        uint32_t af[WKT6][WMT][4], bf[WKT6][WNT6][2];
        #pragma unroll
        for (int kk = 0; kk < WKT6; kk++) {
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++) {
                int row = wm * WM + mi * 16 + (lid & 15);
                int col = kk * 16 + ((lid >> 4) << 3);
                int sc = col ^ ((row & 3) << 3);
                uint32_t addr = smem_u32(&sa[row * SA6 + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(af[kk][mi][0]),"=r"(af[kk][mi][1]),"=r"(af[kk][mi][2]),"=r"(af[kk][mi][3])
                    : "r"(addr));
            }
            #pragma unroll
            for (int ni = 0; ni < WNT6; ni++) {
                int row = kk * 16 + (lid & 15);
                int col = wn * WN6 + ni * 8;
                int sc = col ^ ((row & 7) << 3);
                uint32_t addr = smem_u32(&sb[row * SB6 + sc]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                    : "=r"(bf[kk][ni][0]),"=r"(bf[kk][ni][1]) : "r"(addr));
            }
        }

        #pragma unroll
        for (int kk = 0; kk < WKT6; kk++)
            #pragma unroll
            for (int mi = 0; mi < WMT; mi++)
                #pragma unroll
                for (int ni = 0; ni < WNT6; ni++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                        : "r"(af[kk][mi][0]),"r"(af[kk][mi][1]),"r"(af[kk][mi][2]),"r"(af[kk][mi][3]),
                          "r"(bf[kk][ni][0]),"r"(bf[kk][ni][1]),
                          "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));

        sr = (sr + 1) % ST6;
    }

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    half* Cout = C + bn0;
    #pragma unroll
    for (int mi = 0; mi < WMT; mi++) {
        int r0 = wm * WM + mi * 16 + (lid >> 2);
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WNT6; ni++) {
            int c0 = wn * WN6 + ni * 8 + (lid & 3) * 2;
            *reinterpret_cast<half2*>(&Cout[r0 * N + c0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&Cout[r1 * N + c0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

static int g_best = -1;

static size_t smem_k1() { return (size_t)ST  * (BM * SA  + BK  * SB ) * 2; }
static size_t smem_k2() { return (size_t)ST  * (BM * SA  + BK  * SB ) * 2; }
static size_t smem_k3() { return (size_t)ST  * (BM * SA  + BK  * SB3) * 2; }
static size_t smem_k4() { return (size_t)ST4 * (BM * SA  + BK  * SB ) * 2; }
static size_t smem_k5() { return (size_t)ST5 * (BM * SA5 + BK5 * SB5) * 2; }
static size_t smem_k6() { return (size_t)ST6 * (BM * SA6 + BK5 * SB6) * 2; }

static void setup_kernels() {
    static bool done = false;
    if (done) return;
    done = true;
    auto s = [](auto* f, size_t sz) {
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sz);
        cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    };
    s(kern_main,     smem_k1());
    s(kern_smem_epi, smem_k2());
    s(kern_bn256,    smem_k3());
    s(kern_4stage,   smem_k4());
    s(kern_bk64,     smem_k5());
    s(kern_bn256_bk64, smem_k6());
}

static void run_k(int kid, const half* A, const half* B, half* C, int N, int K) {
    int g128 = (N + BN  - 1) / BN;
    int g256_bn256 = (N + BN3 - 1) / BN3;
    int g256_bn6  = (N + BN6 - 1) / BN6;
    switch (kid) {
        case 0: kern_main    <<<g128,      128, smem_k1()>>>(A, B, C, N, K); break;
        case 1: kern_smem_epi<<<g128,      128, smem_k2()>>>(A, B, C, N, K); break;
        case 2: kern_bn256   <<<g256_bn256,256, smem_k3()>>>(A, B, C, N, K); break;
        case 3: kern_4stage  <<<g128,      128, smem_k4()>>>(A, B, C, N, K); break;
        case 4: kern_bk64    <<<g128,      128, smem_k5()>>>(A, B, C, N, K); break;
        case 5: kern_bn256_bk64<<<g256_bn6,256, smem_k6()>>>(A, B, C, N, K); break;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);
    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    setup_kernels();

    if (g_best < 0) {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        float best_ms = 1e18f;
        g_best = 0;
        const int NKERNELS = 6;
        for (int kid = 0; kid < NKERNELS; kid++) {
            for (int i = 0; i < 5; i++) run_k(kid, A, B, C, N, K);
            cudaDeviceSynchronize();
            cudaEventRecord(t0);
            for (int i = 0; i < 30; i++) run_k(kid, A, B, C, N, K);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, t0, t1);
            if (ms < best_ms) { best_ms = ms; g_best = kid; }
        }
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    run_k(g_best, A, B, C, N, K);
}