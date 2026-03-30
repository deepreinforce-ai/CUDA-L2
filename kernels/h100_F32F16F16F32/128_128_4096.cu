#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm volatile("{.reg .u64 u64addr;\n"
                 " cvta.to.shared.u64 u64addr, %1;\n"
                 " cvt.u32.u64 %0, u64addr;}\n"
                 : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__device__ __forceinline__ void cp_async16_ca(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void cp_async16_cg(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<int N_GROUPS>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_GROUPS) : "memory");
}

__device__ __forceinline__ void mma_m16n8k16(
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

static constexpr int BM        = 64;
static constexpr int BN        = 64;
static constexpr int BK        = 32;
static constexpr int K_SPLIT   = 32;
static constexpr int N_STAGES  = 4;
static constexpr int NTHREADS  = 128;

static constexpr int SA_STRIDE = 40;
static constexpr int SB_STRIDE = 72;

static constexpr int WM = 2;
static constexpr int WN = 4;
static constexpr int WK = 2;

static float* g_partial      = nullptr;
static size_t g_partial_size = 0;

static float* ensure_partial(size_t bytes) {
    if (!g_partial || g_partial_size < bytes) {
        if (g_partial) cudaFree(g_partial);
        cudaMalloc(&g_partial, bytes);
        g_partial_size = bytes;
    }
    return g_partial;
}

__global__ void __launch_bounds__(128, 2)
hgemm_optimized_v5(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C_partial,
    __half*       __restrict__ C_out,
    int M, int N, int K)
{
    const int bid_m  = blockIdx.x;
    const int bid_n  = blockIdx.y;
    const int bid_ks = blockIdx.z;

    const int row_base    = bid_m * BM;
    const int col_base    = bid_n * BN;
    const int k_per_split = K / K_SPLIT;
    const int k_start     = bid_ks * k_per_split;
    const int k_tiles     = k_per_split / BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int wy = warp_id >> 1;
    const int wx = warp_id & 1;

    __shared__ __align__(128) __half sA[N_STAGES][BM][SA_STRIDE];
    __shared__ __align__(128) __half sB[N_STAGES][BK][SB_STRIDE];

    const int aLd_row0 = tid >> 2;
    const int aLd_row1 = aLd_row0 + 32;
    const int aLd_col  = (tid & 3) << 3;

    const int bLd_row0 = tid >> 3;
    const int bLd_row1 = bLd_row0 + 16;
    const int bLd_col  = (tid & 7) << 3;

    float acc[WM][WN][4];
    #pragma unroll
    for (int mi = 0; mi < WM; mi++)
        #pragma unroll
        for (int ni = 0; ni < WN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] =
            acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    #pragma unroll
    for (int s = 0; s < N_STAGES - 1; s++) {
        if (s < k_tiles) {
            const int kb = k_start + s * BK;
            cp_async16_ca(smem_u32addr(&sA[s][aLd_row0][aLd_col]),
                          &A[(row_base + aLd_row0) * K + kb + aLd_col]);
            cp_async16_ca(smem_u32addr(&sA[s][aLd_row1][aLd_col]),
                          &A[(row_base + aLd_row1) * K + kb + aLd_col]);
            cp_async16_cg(smem_u32addr(&sB[s][bLd_row0][bLd_col]),
                          &B[(kb + bLd_row0) * N + col_base + bLd_col]);
            cp_async16_cg(smem_u32addr(&sB[s][bLd_row1][bLd_col]),
                          &B[(kb + bLd_row1) * N + col_base + bLd_col]);
        }
        cp_async_commit();
    }

    #pragma unroll 1
    for (int kt = 0; kt < k_tiles; kt++) {
        cp_async_wait<N_STAGES - 2>();
        __syncthreads();

        const int cur_s = kt & (N_STAGES - 1);

        const int pf = kt + N_STAGES - 1;
        if (pf < k_tiles) {
            const int ns = pf & (N_STAGES - 1);
            const int kb = k_start + pf * BK;
            cp_async16_ca(smem_u32addr(&sA[ns][aLd_row0][aLd_col]),
                          &A[(row_base + aLd_row0) * K + kb + aLd_col]);
            cp_async16_ca(smem_u32addr(&sA[ns][aLd_row1][aLd_col]),
                          &A[(row_base + aLd_row1) * K + kb + aLd_col]);
            cp_async16_cg(smem_u32addr(&sB[ns][bLd_row0][bLd_col]),
                          &B[(kb + bLd_row0) * N + col_base + bLd_col]);
            cp_async16_cg(smem_u32addr(&sB[ns][bLd_row1][bLd_col]),
                          &B[(kb + bLd_row1) * N + col_base + bLd_col]);
        }
        cp_async_commit();

        const __half* csA = &sA[cur_s][0][0];
        const __half* csB = &sB[cur_s][0][0];

        const int warp_row = wy * (WM * 16);
        const int warp_col = wx * (WN * 8);

        uint32_t a_cur[WM][4], b_cur[WN][2];

        #pragma unroll
        for (int mi = 0; mi < WM; mi++) {
            const int row = warp_row + mi * 16;
            const uint32_t sp = smem_u32addr(
                csA + (row + (lane & 15)) * SA_STRIDE + (lane >> 4) * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_cur[mi][0]),"=r"(a_cur[mi][1]),
                  "=r"(a_cur[mi][2]),"=r"(a_cur[mi][3])
                : "r"(sp));
        }
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            const int col = warp_col + ni * 8;
            const uint32_t sp = smem_u32addr(
                csB + (lane & 15) * SB_STRIDE + col + (lane >> 4) * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_cur[ni][0]),"=r"(b_cur[ni][1])
                : "r"(sp));
        }

        #pragma unroll
        for (int ki = 0; ki < WK; ki++) {
            uint32_t a_nxt[WM][4], b_nxt[WN][2];

            if (ki + 1 < WK) {
                const int koff = (ki + 1) * 16;
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    const int row = warp_row + mi * 16;
                    const uint32_t sp = smem_u32addr(
                        csA + (row + (lane & 15)) * SA_STRIDE + koff + (lane >> 4) * 8);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a_nxt[mi][0]),"=r"(a_nxt[mi][1]),
                          "=r"(a_nxt[mi][2]),"=r"(a_nxt[mi][3])
                        : "r"(sp));
                }
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    const int col = warp_col + ni * 8;
                    const uint32_t sp = smem_u32addr(
                        csB + (koff + (lane & 15)) * SB_STRIDE + col + (lane >> 4) * 8);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(b_nxt[ni][0]),"=r"(b_nxt[ni][1])
                        : "r"(sp));
                }
            }

            #pragma unroll
            for (int mi = 0; mi < WM; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    mma_m16n8k16(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a_cur[mi][0], a_cur[mi][1],
                        a_cur[mi][2], a_cur[mi][3],
                        b_cur[ni][0], b_cur[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
                }
            }

            if (ki + 1 < WK) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++)
                    #pragma unroll
                    for (int r = 0; r < 4; r++) a_cur[mi][r] = a_nxt[mi][r];
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    b_cur[ni][0] = b_nxt[ni][0];
                    b_cur[ni][1] = b_nxt[ni][1];
                }
            }
        }
    }

    cp_async_wait<0>();
    __syncthreads();

    float* partial = C_partial + (size_t)bid_ks * M * N;
    const int out_row_lo = lane >> 2;
    const int out_row_hi = out_row_lo + 8;
    const int out_col2   = (lane & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < WM; mi++) {
        const int base_row = row_base + wy * (WM * 16) + mi * 16;
        const int gr0 = base_row + out_row_lo;
        const int gr1 = base_row + out_row_hi;
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            const int gc = col_base + wx * (WN * 8) + ni * 8 + out_col2;
            *reinterpret_cast<float2*>(&partial[gr0 * N + gc]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&partial[gr1 * N + gc]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }

    cg::this_grid().sync();

    const int MN      = M * N;
    const int cta_lin = bid_ks * (gridDim.x * gridDim.y) + bid_m * gridDim.y + bid_n;
    const int elem    = cta_lin * NTHREADS + tid;

    if (elem < MN) {
        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
        float s4=0.f, s5=0.f, s6=0.f, s7=0.f;

        #pragma unroll
        for (int ks = 0; ks < K_SPLIT; ks += 8) {
            s0 += __ldg(&C_partial[(size_t)(ks+0) * MN + elem]);
            s1 += __ldg(&C_partial[(size_t)(ks+1) * MN + elem]);
            s2 += __ldg(&C_partial[(size_t)(ks+2) * MN + elem]);
            s3 += __ldg(&C_partial[(size_t)(ks+3) * MN + elem]);
            s4 += __ldg(&C_partial[(size_t)(ks+4) * MN + elem]);
            s5 += __ldg(&C_partial[(size_t)(ks+5) * MN + elem]);
            s6 += __ldg(&C_partial[(size_t)(ks+6) * MN + elem]);
            s7 += __ldg(&C_partial[(size_t)(ks+7) * MN + elem]);
        }
        C_out[elem] = __float2half((s0+s1) + (s2+s3) + (s4+s5) + (s6+s7));
    }
}

__global__ void __launch_bounds__(128, 2)
hgemm_splitk_nc(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C_partial,
    int M, int N, int K)
{
    const int bid_m  = blockIdx.x;
    const int bid_n  = blockIdx.y;
    const int bid_ks = blockIdx.z;

    const int row_base    = bid_m * BM;
    const int col_base    = bid_n * BN;
    const int k_per_split = K / K_SPLIT;
    const int k_start     = bid_ks * k_per_split;
    const int k_tiles     = k_per_split / BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int wy = warp_id >> 1;
    const int wx = warp_id & 1;

    __shared__ __align__(128) __half sA[N_STAGES][BM][SA_STRIDE];
    __shared__ __align__(128) __half sB[N_STAGES][BK][SB_STRIDE];

    const int aLd_row0 = tid >> 2;
    const int aLd_row1 = aLd_row0 + 32;
    const int aLd_col  = (tid & 3) << 3;
    const int bLd_row0 = tid >> 3;
    const int bLd_row1 = bLd_row0 + 16;
    const int bLd_col  = (tid & 7) << 3;

    float acc[WM][WN][4];
    #pragma unroll
    for (int mi = 0; mi < WM; mi++)
        #pragma unroll
        for (int ni = 0; ni < WN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] =
            acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    #pragma unroll
    for (int s = 0; s < N_STAGES - 1; s++) {
        if (s < k_tiles) {
            const int kb = k_start + s * BK;
            cp_async16_ca(smem_u32addr(&sA[s][aLd_row0][aLd_col]),
                          &A[(row_base + aLd_row0) * K + kb + aLd_col]);
            cp_async16_ca(smem_u32addr(&sA[s][aLd_row1][aLd_col]),
                          &A[(row_base + aLd_row1) * K + kb + aLd_col]);
            cp_async16_cg(smem_u32addr(&sB[s][bLd_row0][bLd_col]),
                          &B[(kb + bLd_row0) * N + col_base + bLd_col]);
            cp_async16_cg(smem_u32addr(&sB[s][bLd_row1][bLd_col]),
                          &B[(kb + bLd_row1) * N + col_base + bLd_col]);
        }
        cp_async_commit();
    }

    #pragma unroll 1
    for (int kt = 0; kt < k_tiles; kt++) {
        cp_async_wait<N_STAGES - 2>();
        __syncthreads();

        const int cur_s = kt & (N_STAGES - 1);
        const int pf = kt + N_STAGES - 1;
        if (pf < k_tiles) {
            const int ns = pf & (N_STAGES - 1);
            const int kb = k_start + pf * BK;
            cp_async16_ca(smem_u32addr(&sA[ns][aLd_row0][aLd_col]),
                          &A[(row_base + aLd_row0) * K + kb + aLd_col]);
            cp_async16_ca(smem_u32addr(&sA[ns][aLd_row1][aLd_col]),
                          &A[(row_base + aLd_row1) * K + kb + aLd_col]);
            cp_async16_cg(smem_u32addr(&sB[ns][bLd_row0][bLd_col]),
                          &B[(kb + bLd_row0) * N + col_base + bLd_col]);
            cp_async16_cg(smem_u32addr(&sB[ns][bLd_row1][bLd_col]),
                          &B[(kb + bLd_row1) * N + col_base + bLd_col]);
        }
        cp_async_commit();

        const __half* csA = &sA[cur_s][0][0];
        const __half* csB = &sB[cur_s][0][0];
        const int warp_row = wy * (WM * 16);
        const int warp_col = wx * (WN * 8);

        uint32_t a_cur[WM][4], b_cur[WN][2];

        #pragma unroll
        for (int mi = 0; mi < WM; mi++) {
            const int row = warp_row + mi * 16;
            const uint32_t sp = smem_u32addr(
                csA + (row + (lane & 15)) * SA_STRIDE + (lane >> 4) * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_cur[mi][0]),"=r"(a_cur[mi][1]),
                  "=r"(a_cur[mi][2]),"=r"(a_cur[mi][3])
                : "r"(sp));
        }
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            const int col = warp_col + ni * 8;
            const uint32_t sp = smem_u32addr(
                csB + (lane & 15) * SB_STRIDE + col + (lane >> 4) * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_cur[ni][0]),"=r"(b_cur[ni][1])
                : "r"(sp));
        }

        #pragma unroll
        for (int ki = 0; ki < WK; ki++) {
            uint32_t a_nxt[WM][4], b_nxt[WN][2];
            if (ki + 1 < WK) {
                const int koff = (ki + 1) * 16;
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    const int row = warp_row + mi * 16;
                    const uint32_t sp = smem_u32addr(
                        csA + (row + (lane & 15)) * SA_STRIDE + koff + (lane >> 4) * 8);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a_nxt[mi][0]),"=r"(a_nxt[mi][1]),
                          "=r"(a_nxt[mi][2]),"=r"(a_nxt[mi][3])
                        : "r"(sp));
                }
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    const int col = warp_col + ni * 8;
                    const uint32_t sp = smem_u32addr(
                        csB + (koff + (lane & 15)) * SB_STRIDE + col + (lane >> 4) * 8);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(b_nxt[ni][0]),"=r"(b_nxt[ni][1])
                        : "r"(sp));
                }
            }

            #pragma unroll
            for (int mi = 0; mi < WM; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    mma_m16n8k16(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a_cur[mi][0], a_cur[mi][1],
                        a_cur[mi][2], a_cur[mi][3],
                        b_cur[ni][0], b_cur[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
                }
            }

            if (ki + 1 < WK) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++)
                    #pragma unroll
                    for (int r = 0; r < 4; r++) a_cur[mi][r] = a_nxt[mi][r];
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    b_cur[ni][0] = b_nxt[ni][0];
                    b_cur[ni][1] = b_nxt[ni][1];
                }
            }
        }
    }

    cp_async_wait<0>();
    __syncthreads();

    float* partial = C_partial + (size_t)bid_ks * M * N;
    const int out_row_lo = lane >> 2;
    const int out_row_hi = out_row_lo + 8;
    const int out_col2   = (lane & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < WM; mi++) {
        const int base_row = row_base + wy * (WM * 16) + mi * 16;
        const int gr0 = base_row + out_row_lo;
        const int gr1 = base_row + out_row_hi;
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            const int gc = col_base + wx * (WN * 8) + ni * 8 + out_col2;
            *reinterpret_cast<float2*>(&partial[gr0 * N + gc]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&partial[gr1 * N + gc]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(256, 4)
hgemm_reduce_v5(
    const float* __restrict__ C_partial,
    __half*      __restrict__ C_out,
    int MN)
{
    const int base = (blockIdx.x * 256 + threadIdx.x) * 8;
    if (base >= MN) return;

    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;

    #pragma unroll
    for (int ks = 0; ks < K_SPLIT; ks += 4) {
        const float* p0 = C_partial + (size_t)(ks+0) * MN + base;
        const float* p1 = C_partial + (size_t)(ks+1) * MN + base;
        const float* p2 = C_partial + (size_t)(ks+2) * MN + base;
        const float* p3 = C_partial + (size_t)(ks+3) * MN + base;

        float4 v00 = __ldg(reinterpret_cast<const float4*>(p0+0));
        float4 v01 = __ldg(reinterpret_cast<const float4*>(p0+4));
        float4 v10 = __ldg(reinterpret_cast<const float4*>(p1+0));
        float4 v11 = __ldg(reinterpret_cast<const float4*>(p1+4));
        float4 v20 = __ldg(reinterpret_cast<const float4*>(p2+0));
        float4 v21 = __ldg(reinterpret_cast<const float4*>(p2+4));
        float4 v30 = __ldg(reinterpret_cast<const float4*>(p3+0));
        float4 v31 = __ldg(reinterpret_cast<const float4*>(p3+4));

        s0 += v00.x + v10.x + v20.x + v30.x;
        s1 += v00.y + v10.y + v20.y + v30.y;
        s2 += v00.z + v10.z + v20.z + v30.z;
        s3 += v00.w + v10.w + v20.w + v30.w;
        s4 += v01.x + v11.x + v21.x + v31.x;
        s5 += v01.y + v11.y + v21.y + v31.y;
        s6 += v01.z + v11.z + v21.z + v31.z;
        s7 += v01.w + v11.w + v21.w + v31.w;
    }

    *reinterpret_cast<__half2*>(&C_out[base+0]) = __float22half2_rn(make_float2(s0,s1));
    *reinterpret_cast<__half2*>(&C_out[base+2]) = __float22half2_rn(make_float2(s2,s3));
    *reinterpret_cast<__half2*>(&C_out[base+4]) = __float22half2_rn(make_float2(s4,s5));
    *reinterpret_cast<__half2*>(&C_out[base+6]) = __float22half2_rn(make_float2(s6,s7));
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* A = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B = reinterpret_cast<const __half*>(b.data_ptr());
    __half*       C = reinterpret_cast<__half*>(c.data_ptr());

    const size_t partial_bytes = (size_t)K_SPLIT * M * N * sizeof(float);
    float* C_partial = ensure_partial(partial_bytes);

    dim3 grid(M / BM, N / BN, K_SPLIT);
    dim3 block(NTHREADS);

    static int coop_ok = -1;
    if (coop_ok < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        int val = 0;
        cudaDeviceGetAttribute(&val, cudaDevAttrCooperativeLaunch, dev);
        coop_ok = val;
    }

    if (coop_ok) {
        void* args[] = {
            (void*)&A, (void*)&B,
            (void*)&C_partial, (void*)&C,
            (void*)&M, (void*)&N, (void*)&K
        };
        cudaLaunchCooperativeKernel(
            (void*)hgemm_optimized_v5,
            grid, block, args,
            0, nullptr);
    } else {
        hgemm_splitk_nc<<<grid, block>>>(A, B, C_partial, M, N, K);
        const int MN = M * N;
        const int rb = (MN / 8 + 255) / 256;
        hgemm_reduce_v5<<<rb, 256>>>(C_partial, C, MN);
    }
}