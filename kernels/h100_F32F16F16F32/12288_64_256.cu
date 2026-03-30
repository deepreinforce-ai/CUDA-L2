#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

static constexpr int KDIM       = 256;
static constexpr int NDIM       = 64;
static constexpr int WMMA_K     = 16;
static constexpr int MMA_N8     = 8;
static constexpr int NT8        = NDIM / MMA_N8;
static constexpr int K_TILES    = KDIM / WMMA_K;
static constexpr int B_PAD      = 8;
static constexpr int B_STRIDE   = NDIM + B_PAD;

struct FA { uint32_t r[4]; };
struct FB { uint32_t r[2]; };
struct FC { float    r[4]; };

__device__ __forceinline__ void mma16x8x16(FC& d, const FA& a, const FB& b, const FC& c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d.r[0]),"=f"(d.r[1]),"=f"(d.r[2]),"=f"(d.r[3])
        : "r"(a.r[0]),"r"(a.r[1]),"r"(a.r[2]),"r"(a.r[3]),
          "r"(b.r[0]),"r"(b.r[1]),
          "f"(c.r[0]),"f"(c.r[1]),"f"(c.r[2]),"f"(c.r[3])
    );
}

__device__ __forceinline__ FA load_A_ldg(const __half* base, int lane) {
    FA fa;
    int r0 = lane >> 2;
    int r1 = r0 + 8;
    int c0 = (lane & 3) << 1;
    int c1 = c0 + 8;
    fa.r[0] = __ldg(reinterpret_cast<const uint32_t*>(base + r0 * KDIM + c0));
    fa.r[1] = __ldg(reinterpret_cast<const uint32_t*>(base + r1 * KDIM + c0));
    fa.r[2] = __ldg(reinterpret_cast<const uint32_t*>(base + r0 * KDIM + c1));
    fa.r[3] = __ldg(reinterpret_cast<const uint32_t*>(base + r1 * KDIM + c1));
    return fa;
}

__device__ __forceinline__ FB load_B_ldmatrix(const __half* smB_kt, int nt, int lane) {
    FB fb;
    int k_row = lane & 15;
    const __half* ptr = smB_kt + k_row * B_STRIDE + nt * MMA_N8;
    uint32_t sp;
    asm volatile("{ .reg .u64 s64; cvta.to.shared.u64 s64, %1; cvt.u32.u64 %0, s64; }"
                 : "=r"(sp) : "l"(ptr));
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(fb.r[0]),"=r"(fb.r[1]) : "r"(sp));
    return fb;
}

__device__ __forceinline__ FB load_B_nopad(const __half* smB_kt, int nt, int lane) {
    FB fb;
    int k_row = lane & 15;
    const __half* ptr = smB_kt + k_row * NDIM + nt * MMA_N8;
    uint32_t sp;
    asm volatile("{ .reg .u64 s64; cvta.to.shared.u64 s64, %1; cvt.u32.u64 %0, s64; }"
                 : "=r"(sp) : "l"(ptr));
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(fb.r[0]),"=r"(fb.r[1]) : "r"(sp));
    return fb;
}

static constexpr int K1_W  = 4;
static constexpr int K1_RT = 3;
static constexpr int K1_BR = K1_W * K1_RT * 16;

__global__ void __launch_bounds__(128, 3)
hgemm_192_nt_outer(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * B_STRIDE];

    const int brs = blockIdx.x * K1_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K1_RT * 16;

    #pragma unroll 2
    for (int row_base = 0; row_base < KDIM; row_base += 128) {
        int row = row_base + threadIdx.x;
        if (row < KDIM) {
            const __half* src = B + row * NDIM;
            __half*       dst = smB + row * B_STRIDE;
            float4 t0 = reinterpret_cast<const float4*>(src)[0];
            float4 t1 = reinterpret_cast<const float4*>(src)[1];
            float4 t2 = reinterpret_cast<const float4*>(src)[2];
            float4 t3 = reinterpret_cast<const float4*>(src)[3];
            float4 t4 = reinterpret_cast<const float4*>(src)[4];
            float4 t5 = reinterpret_cast<const float4*>(src)[5];
            float4 t6 = reinterpret_cast<const float4*>(src)[6];
            float4 t7 = reinterpret_cast<const float4*>(src)[7];
            reinterpret_cast<float4*>(dst)[0] = t0;
            reinterpret_cast<float4*>(dst)[1] = t1;
            reinterpret_cast<float4*>(dst)[2] = t2;
            reinterpret_cast<float4*>(dst)[3] = t3;
            reinterpret_cast<float4*>(dst)[4] = t4;
            reinterpret_cast<float4*>(dst)[5] = t5;
            reinterpret_cast<float4*>(dst)[6] = t6;
            reinterpret_cast<float4*>(dst)[7] = t7;
        }
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K1_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K1_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K1_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K1_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K1_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int nt = 0; nt < NT8; nt++) {
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            FB fb = load_B_ldmatrix(smB + kt * 16 * B_STRIDE, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt], fb, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt], fb, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt], fb, acc[2][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K1_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static constexpr int K2_W  = 4;
static constexpr int K2_RT = 3;
static constexpr int K2_BR = K2_W * K2_RT * 16;

__global__ void __launch_bounds__(128, 3)
hgemm_192_kt_outer(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * B_STRIDE];

    const int brs = blockIdx.x * K2_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K2_RT * 16;

    #pragma unroll 2
    for (int row_base = 0; row_base < KDIM; row_base += 128) {
        int row = row_base + threadIdx.x;
        if (row < KDIM) {
            const __half* src = B + row * NDIM;
            __half*       dst = smB + row * B_STRIDE;
            float4 t0 = reinterpret_cast<const float4*>(src)[0];
            float4 t1 = reinterpret_cast<const float4*>(src)[1];
            float4 t2 = reinterpret_cast<const float4*>(src)[2];
            float4 t3 = reinterpret_cast<const float4*>(src)[3];
            float4 t4 = reinterpret_cast<const float4*>(src)[4];
            float4 t5 = reinterpret_cast<const float4*>(src)[5];
            float4 t6 = reinterpret_cast<const float4*>(src)[6];
            float4 t7 = reinterpret_cast<const float4*>(src)[7];
            reinterpret_cast<float4*>(dst)[0] = t0;
            reinterpret_cast<float4*>(dst)[1] = t1;
            reinterpret_cast<float4*>(dst)[2] = t2;
            reinterpret_cast<float4*>(dst)[3] = t3;
            reinterpret_cast<float4*>(dst)[4] = t4;
            reinterpret_cast<float4*>(dst)[5] = t5;
            reinterpret_cast<float4*>(dst)[6] = t6;
            reinterpret_cast<float4*>(dst)[7] = t7;
        }
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K2_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K2_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K2_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K2_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K2_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int kt = 0; kt < K_TILES; kt++) {
        const __half* Bkt = smB + kt * 16 * B_STRIDE;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            FB fb = load_B_ldmatrix(Bkt, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt], fb, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt], fb, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt], fb, acc[2][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K2_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static constexpr int K3_W  = 4;
static constexpr int K3_RT = 2;
static constexpr int K3_BR = K3_W * K3_RT * 16;

__global__ void __launch_bounds__(128, 4)
hgemm_128_nt_outer(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * B_STRIDE];

    const int brs = blockIdx.x * K3_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K3_RT * 16;

    #pragma unroll 2
    for (int row_base = 0; row_base < KDIM; row_base += 128) {
        int row = row_base + threadIdx.x;
        if (row < KDIM) {
            const __half* src = B + row * NDIM;
            __half*       dst = smB + row * B_STRIDE;
            float4 t0 = reinterpret_cast<const float4*>(src)[0];
            float4 t1 = reinterpret_cast<const float4*>(src)[1];
            float4 t2 = reinterpret_cast<const float4*>(src)[2];
            float4 t3 = reinterpret_cast<const float4*>(src)[3];
            float4 t4 = reinterpret_cast<const float4*>(src)[4];
            float4 t5 = reinterpret_cast<const float4*>(src)[5];
            float4 t6 = reinterpret_cast<const float4*>(src)[6];
            float4 t7 = reinterpret_cast<const float4*>(src)[7];
            reinterpret_cast<float4*>(dst)[0] = t0;
            reinterpret_cast<float4*>(dst)[1] = t1;
            reinterpret_cast<float4*>(dst)[2] = t2;
            reinterpret_cast<float4*>(dst)[3] = t3;
            reinterpret_cast<float4*>(dst)[4] = t4;
            reinterpret_cast<float4*>(dst)[5] = t5;
            reinterpret_cast<float4*>(dst)[6] = t6;
            reinterpret_cast<float4*>(dst)[7] = t7;
        }
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K3_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        asm volatile("prefetch.global.L1 [%0];\n" :: "l"(p)       : "memory");
        asm volatile("prefetch.global.L1 [%0];\n" :: "l"(p+128)   : "memory");
        #pragma unroll
        for (int kt = 1; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K3_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K3_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K3_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K3_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int nt = 0; nt < NT8; nt++) {
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            FB fb = load_B_ldmatrix(smB + kt * 16 * B_STRIDE, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt], fb, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt], fb, acc[1][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K3_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static constexpr int K4_W  = 4;
static constexpr int K4_RT = 4;
static constexpr int K4_BR = K4_W * K4_RT * 16;

__global__ void __launch_bounds__(128, 3)
hgemm_256_nt_outer(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * B_STRIDE];

    const int brs = blockIdx.x * K4_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K4_RT * 16;

    #pragma unroll 2
    for (int row_base = 0; row_base < KDIM; row_base += 128) {
        int row = row_base + threadIdx.x;
        if (row < KDIM) {
            const __half* src = B + row * NDIM;
            __half*       dst = smB + row * B_STRIDE;
            float4 t0 = reinterpret_cast<const float4*>(src)[0];
            float4 t1 = reinterpret_cast<const float4*>(src)[1];
            float4 t2 = reinterpret_cast<const float4*>(src)[2];
            float4 t3 = reinterpret_cast<const float4*>(src)[3];
            float4 t4 = reinterpret_cast<const float4*>(src)[4];
            float4 t5 = reinterpret_cast<const float4*>(src)[5];
            float4 t6 = reinterpret_cast<const float4*>(src)[6];
            float4 t7 = reinterpret_cast<const float4*>(src)[7];
            reinterpret_cast<float4*>(dst)[0] = t0;
            reinterpret_cast<float4*>(dst)[1] = t1;
            reinterpret_cast<float4*>(dst)[2] = t2;
            reinterpret_cast<float4*>(dst)[3] = t3;
            reinterpret_cast<float4*>(dst)[4] = t4;
            reinterpret_cast<float4*>(dst)[5] = t5;
            reinterpret_cast<float4*>(dst)[6] = t6;
            reinterpret_cast<float4*>(dst)[7] = t7;
        }
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K4_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K4_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K4_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K4_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K4_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int nt = 0; nt < NT8; nt++) {
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            FB fb = load_B_ldmatrix(smB + kt * 16 * B_STRIDE, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt], fb, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt], fb, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt], fb, acc[2][nt]);
            mma16x8x16(acc[3][nt], fa[3][kt], fb, acc[3][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K4_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static constexpr int K5_W  = 4;
static constexpr int K5_RT = 3;
static constexpr int K5_BR = K5_W * K5_RT * 16;

__global__ void __launch_bounds__(128, 4)
hgemm_192_nopad_nt_outer(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * NDIM];

    const int brs = blockIdx.x * K5_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K5_RT * 16;

    {
        float4*       dst = reinterpret_cast<float4*>(smB);
        const float4* src = reinterpret_cast<const float4*>(B);
        #pragma unroll 16
        for (int i = threadIdx.x; i < KDIM * NDIM / 8; i += 128)
            dst[i] = src[i];
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K5_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K5_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K5_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K5_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K5_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int nt = 0; nt < NT8; nt++) {
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            FB fb = load_B_nopad(smB + kt * 16 * NDIM, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt], fb, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt], fb, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt], fb, acc[2][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K5_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static constexpr int K6_W  = 4;
static constexpr int K6_RT = 3;
static constexpr int K6_BR = K6_W * K6_RT * 16;

__global__ void __launch_bounds__(128, 3)
hgemm_192_kpair_nt(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * B_STRIDE];

    const int brs = blockIdx.x * K6_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K6_RT * 16;

    #pragma unroll 2
    for (int row_base = 0; row_base < KDIM; row_base += 128) {
        int row = row_base + threadIdx.x;
        if (row < KDIM) {
            const __half* src = B + row * NDIM;
            __half*       dst = smB + row * B_STRIDE;
            float4 t0 = reinterpret_cast<const float4*>(src)[0];
            float4 t1 = reinterpret_cast<const float4*>(src)[1];
            float4 t2 = reinterpret_cast<const float4*>(src)[2];
            float4 t3 = reinterpret_cast<const float4*>(src)[3];
            float4 t4 = reinterpret_cast<const float4*>(src)[4];
            float4 t5 = reinterpret_cast<const float4*>(src)[5];
            float4 t6 = reinterpret_cast<const float4*>(src)[6];
            float4 t7 = reinterpret_cast<const float4*>(src)[7];
            reinterpret_cast<float4*>(dst)[0] = t0;
            reinterpret_cast<float4*>(dst)[1] = t1;
            reinterpret_cast<float4*>(dst)[2] = t2;
            reinterpret_cast<float4*>(dst)[3] = t3;
            reinterpret_cast<float4*>(dst)[4] = t4;
            reinterpret_cast<float4*>(dst)[5] = t5;
            reinterpret_cast<float4*>(dst)[6] = t6;
            reinterpret_cast<float4*>(dst)[7] = t7;
        }
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K6_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K6_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K6_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K6_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K6_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int nt = 0; nt < NT8; nt++) {
        #pragma unroll
        for (int kp = 0; kp < K_TILES / 2; kp++) {
            int kt0 = 2 * kp;
            int kt1 = 2 * kp + 1;
            FB fb0 = load_B_ldmatrix(smB + kt0 * 16 * B_STRIDE, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt0], fb0, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt0], fb0, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt0], fb0, acc[2][nt]);
            FB fb1 = load_B_ldmatrix(smB + kt1 * 16 * B_STRIDE, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt1], fb1, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt1], fb1, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt1], fb1, acc[2][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K6_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static constexpr int K7_W  = 4;
static constexpr int K7_RT = 3;
static constexpr int K7_BR = K7_W * K7_RT * 16;

__global__ void __launch_bounds__(128, 3)
hgemm_192_kt4_unroll(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * B_STRIDE];

    const int brs = blockIdx.x * K7_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K7_RT * 16;

    #pragma unroll 2
    for (int row_base = 0; row_base < KDIM; row_base += 128) {
        int row = row_base + threadIdx.x;
        if (row < KDIM) {
            const __half* src = B + row * NDIM;
            __half*       dst = smB + row * B_STRIDE;
            float4 t0 = reinterpret_cast<const float4*>(src)[0];
            float4 t1 = reinterpret_cast<const float4*>(src)[1];
            float4 t2 = reinterpret_cast<const float4*>(src)[2];
            float4 t3 = reinterpret_cast<const float4*>(src)[3];
            float4 t4 = reinterpret_cast<const float4*>(src)[4];
            float4 t5 = reinterpret_cast<const float4*>(src)[5];
            float4 t6 = reinterpret_cast<const float4*>(src)[6];
            float4 t7 = reinterpret_cast<const float4*>(src)[7];
            reinterpret_cast<float4*>(dst)[0] = t0;
            reinterpret_cast<float4*>(dst)[1] = t1;
            reinterpret_cast<float4*>(dst)[2] = t2;
            reinterpret_cast<float4*>(dst)[3] = t3;
            reinterpret_cast<float4*>(dst)[4] = t4;
            reinterpret_cast<float4*>(dst)[5] = t5;
            reinterpret_cast<float4*>(dst)[6] = t6;
            reinterpret_cast<float4*>(dst)[7] = t7;
        }
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K7_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K7_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K7_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K7_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K7_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int kg = 0; kg < K_TILES / 4; kg++) {
        int kt0 = kg * 4 + 0;
        int kt1 = kg * 4 + 1;
        int kt2 = kg * 4 + 2;
        int kt3 = kg * 4 + 3;
        const __half* B0 = smB + kt0 * 16 * B_STRIDE;
        const __half* B1 = smB + kt1 * 16 * B_STRIDE;
        const __half* B2 = smB + kt2 * 16 * B_STRIDE;
        const __half* B3 = smB + kt3 * 16 * B_STRIDE;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            FB fb0 = load_B_ldmatrix(B0, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt0], fb0, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt0], fb0, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt0], fb0, acc[2][nt]);
            FB fb1 = load_B_ldmatrix(B1, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt1], fb1, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt1], fb1, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt1], fb1, acc[2][nt]);
            FB fb2 = load_B_ldmatrix(B2, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt2], fb2, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt2], fb2, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt2], fb2, acc[2][nt]);
            FB fb3 = load_B_ldmatrix(B3, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt3], fb3, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt3], fb3, acc[1][nt]);
            mma16x8x16(acc[2][nt], fa[2][kt3], fb3, acc[2][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K7_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static constexpr int K8_W  = 4;
static constexpr int K8_RT = 2;
static constexpr int K8_BR = K8_W * K8_RT * 16;

__global__ void __launch_bounds__(128, 4)
hgemm_128_kt_outer(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M
) {
    __shared__ __align__(128) __half smB[KDIM * B_STRIDE];

    const int brs = blockIdx.x * K8_BR;
    const int wid = threadIdx.x >> 5;
    const int lid = threadIdx.x & 31;
    const int wrs = brs + wid * K8_RT * 16;

    #pragma unroll 2
    for (int row_base = 0; row_base < KDIM; row_base += 128) {
        int row = row_base + threadIdx.x;
        if (row < KDIM) {
            const __half* src = B + row * NDIM;
            __half*       dst = smB + row * B_STRIDE;
            float4 t0 = reinterpret_cast<const float4*>(src)[0];
            float4 t1 = reinterpret_cast<const float4*>(src)[1];
            float4 t2 = reinterpret_cast<const float4*>(src)[2];
            float4 t3 = reinterpret_cast<const float4*>(src)[3];
            float4 t4 = reinterpret_cast<const float4*>(src)[4];
            float4 t5 = reinterpret_cast<const float4*>(src)[5];
            float4 t6 = reinterpret_cast<const float4*>(src)[6];
            float4 t7 = reinterpret_cast<const float4*>(src)[7];
            reinterpret_cast<float4*>(dst)[0] = t0;
            reinterpret_cast<float4*>(dst)[1] = t1;
            reinterpret_cast<float4*>(dst)[2] = t2;
            reinterpret_cast<float4*>(dst)[3] = t3;
            reinterpret_cast<float4*>(dst)[4] = t4;
            reinterpret_cast<float4*>(dst)[5] = t5;
            reinterpret_cast<float4*>(dst)[6] = t6;
            reinterpret_cast<float4*>(dst)[7] = t7;
        }
    }
    __syncthreads();

    if (wrs >= M) return;

    const __half* Aw = A + wrs * KDIM;

    #pragma unroll
    for (int wr = 0; wr < K8_RT; wr++) {
        const __half* p = Aw + wr * 16 * KDIM;
        asm volatile("prefetch.global.L1 [%0];\n" :: "l"(p)       : "memory");
        asm volatile("prefetch.global.L1 [%0];\n" :: "l"(p+128)   : "memory");
        #pragma unroll
        for (int kt = 1; kt < K_TILES; kt++) {
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16)     : "memory");
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + kt*16+128) : "memory");
        }
    }

    FA fa[K8_RT][K_TILES];
    #pragma unroll
    for (int wr = 0; wr < K8_RT; wr++) {
        const __half* bwr = Aw + wr * 16 * KDIM;
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++)
            fa[wr][kt] = load_A_ldg(bwr + kt * 16, lid);
    }

    FC acc[K8_RT][NT8];
    #pragma unroll
    for (int wr = 0; wr < K8_RT; wr++)
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++)
            acc[wr][nt] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int kt = 0; kt < K_TILES; kt++) {
        const __half* Bkt = smB + kt * 16 * B_STRIDE;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            FB fb = load_B_ldmatrix(Bkt, nt, lid);
            mma16x8x16(acc[0][nt], fa[0][kt], fb, acc[0][nt]);
            mma16x8x16(acc[1][nt], fa[1][kt], fb, acc[1][nt]);
        }
    }

    #pragma unroll
    for (int wr = 0; wr < K8_RT; wr++) {
        int r0 = wrs + wr * 16 + (lid >> 2);
        int r1 = r0 + 8;
        int cn = (lid & 3) << 1;
        #pragma unroll
        for (int nt = 0; nt < NT8; nt++) {
            int no = nt * MMA_N8 + cn;
            if (r0 < M)
                *reinterpret_cast<__half2*>(&C[r0 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[0], acc[wr][nt].r[1]);
            if (r1 < M)
                *reinterpret_cast<__half2*>(&C[r1 * NDIM + no]) =
                    __floats2half2_rn(acc[wr][nt].r[2], acc[wr][nt].r[3]);
        }
    }
}

static int   g_best    = -1;
static float g_best_ms = 1e9f;
static constexpr int N_KERNELS = 8;

static void dispatch(int kid, const __half* A, const __half* B, __half* C, int M) {
    switch (kid) {
    case 0: hgemm_192_nt_outer       <<<(M+K1_BR-1)/K1_BR, 128>>>(A,B,C,M); break;
    case 1: hgemm_192_kt_outer       <<<(M+K2_BR-1)/K2_BR, 128>>>(A,B,C,M); break;
    case 2: hgemm_128_nt_outer       <<<(M+K3_BR-1)/K3_BR, 128>>>(A,B,C,M); break;
    case 3: hgemm_256_nt_outer       <<<(M+K4_BR-1)/K4_BR, 128>>>(A,B,C,M); break;
    case 4: hgemm_192_nopad_nt_outer <<<(M+K5_BR-1)/K5_BR, 128>>>(A,B,C,M); break;
    case 5: hgemm_192_kpair_nt       <<<(M+K6_BR-1)/K6_BR, 128>>>(A,B,C,M); break;
    case 6: hgemm_192_kt4_unroll     <<<(M+K7_BR-1)/K7_BR, 128>>>(A,B,C,M); break;
    case 7: hgemm_128_kt_outer       <<<(M+K8_BR-1)/K8_BR, 128>>>(A,B,C,M); break;
    default: hgemm_192_kt_outer      <<<(M+K2_BR-1)/K2_BR, 128>>>(A,B,C,M); break;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    int M = a.size(0);
    const __half* A_ptr = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    __half*       C_ptr = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    static bool attrs_set = false;
    if (!attrs_set) {
        auto sa = [](const void* fn) {
            cudaFuncSetAttribute(fn,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxShared);
        };
        sa((const void*)hgemm_192_nt_outer);
        sa((const void*)hgemm_192_kt_outer);
        sa((const void*)hgemm_128_nt_outer);
        sa((const void*)hgemm_256_nt_outer);
        sa((const void*)hgemm_192_nopad_nt_outer);
        sa((const void*)hgemm_192_kpair_nt);
        sa((const void*)hgemm_192_kt4_unroll);
        sa((const void*)hgemm_128_kt_outer);
        attrs_set = true;
    }

    if (g_best < 0) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int WARMUP = 20, TRIALS = 200;

        g_best    = 0;
        g_best_ms = 1e9f;

        for (int kid = 0; kid < N_KERNELS; kid++) {
            for (int t = 0; t < WARMUP; t++)
                dispatch(kid, A_ptr, B_ptr, C_ptr, M);
            cudaDeviceSynchronize();

            cudaEventRecord(e0);
            for (int t = 0; t < TRIALS; t++)
                dispatch(kid, A_ptr, B_ptr, C_ptr, M);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);

            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ms < g_best_ms) { g_best_ms = ms; g_best = kid; }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }

    dispatch(g_best, A_ptr, B_ptr, C_ptr, M);
}