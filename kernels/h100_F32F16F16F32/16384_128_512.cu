#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int NW>
__device__ __forceinline__ void async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NW) : "memory");
}

__device__ __forceinline__ int swzA(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int swzB(int row, int col) {
    return (((col >> 3) ^ (row & 15)) << 3) | (col & 7);
}

static const int K1_BM     = 64;
static const int K1_BN     = 128;
static const int K1_BK     = 64;
static const int K1_STAGES = 3;
static const int K1_NTH    = 128;
static const int K1_WMT    = 4;
static const int K1_WNT    = 4;
static const int K1_KSTEPS = 4;
static const int K1_SA_EL  = K1_BM * K1_BK;
static const int K1_SB_EL  = K1_BK * K1_BN;
static const int K1_SMEM   = K1_STAGES * (K1_SA_EL + K1_SB_EL) * 2;

__global__ void __launch_bounds__(K1_NTH, 2)
hgemm_k1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K)
{
    const int N      = K1_BN;
    const int ntiles = M / K1_BM;

    extern __shared__ half smem1[];
    half* sA = smem1;
    half* sB = smem1 + K1_STAGES * K1_SA_EL;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int w_col   = warp_id;

    for (int tile_m = blockIdx.x; tile_m < ntiles; tile_m += gridDim.x) {
        const int bm     = tile_m * K1_BM;
        const int num_kt = K / K1_BK;

        float acc[K1_WMT][K1_WNT][4];
        #pragma unroll
        for (int mi = 0; mi < K1_WMT; mi++)
            #pragma unroll
            for (int ni = 0; ni < K1_WNT; ni++)
                acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

        #pragma unroll
        for (int s = 0; s < K1_STAGES - 1; s++) {
            if (s < num_kt) {
                const int kb = s * K1_BK;
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K1_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[s * K1_SA_EL + row * K1_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K1_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[s * K1_SB_EL + row * K1_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();
        }

        uint32_t af[2][K1_WMT][4];
        uint32_t bf[2][K1_WNT][2];

        #pragma unroll 1
        for (int ki = 0; ki < num_kt; ki++) {
            const int cur = ki % K1_STAGES;

            async_wait<K1_STAGES - 2>();
            __syncthreads();

            const int nxt = ki + (K1_STAGES - 1);
            if (nxt < num_kt) {
                const int ns = nxt % K1_STAGES;
                const int kb = nxt * K1_BK;
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K1_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[ns * K1_SA_EL + row * K1_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K1_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[ns * K1_SB_EL + row * K1_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();

            const half* cA = &sA[cur * K1_SA_EL];
            const half* cB = &sB[cur * K1_SB_EL];

            #pragma unroll
            for (int mi = 0; mi < K1_WMT; mi++) {
                int srow = mi * 16 + (lane_id & 15);
                int sc   = swzA(srow, (lane_id >> 4) * 8);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    :"=r"(af[0][mi][0]),"=r"(af[0][mi][1]),"=r"(af[0][mi][2]),"=r"(af[0][mi][3])
                    :"r"(smem_u32(&cA[srow * K1_BK + sc])));
            }
            #pragma unroll
            for (int ni = 0; ni < K1_WNT; ni++) {
                int rc   = w_col * (K1_WNT * 8) + ni * 8;
                int srow = lane_id & 15;
                int sc   = swzB(srow, rc);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    :"=r"(bf[0][ni][0]),"=r"(bf[0][ni][1])
                    :"r"(smem_u32(&cB[srow * K1_BN + sc])));
            }

            #pragma unroll
            for (int step = 0; step < K1_KSTEPS; step++) {
                const int cb = step & 1;
                const int nb = 1 - cb;

                if (step < K1_KSTEPS - 1) {
                    const int koff = (step + 1) * 16;
                    #pragma unroll
                    for (int mi = 0; mi < K1_WMT; mi++) {
                        int srow = mi * 16 + (lane_id & 15);
                        int sc   = swzA(srow, koff + (lane_id >> 4) * 8);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                            :"=r"(af[nb][mi][0]),"=r"(af[nb][mi][1]),"=r"(af[nb][mi][2]),"=r"(af[nb][mi][3])
                            :"r"(smem_u32(&cA[srow * K1_BK + sc])));
                    }
                    #pragma unroll
                    for (int ni = 0; ni < K1_WNT; ni++) {
                        int rc   = w_col * (K1_WNT * 8) + ni * 8;
                        int srow = koff + (lane_id & 15);
                        int sc   = swzB(srow, rc);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                            :"=r"(bf[nb][ni][0]),"=r"(bf[nb][ni][1])
                            :"r"(smem_u32(&cB[srow * K1_BN + sc])));
                    }
                }

                #pragma unroll
                for (int mi = 0; mi < K1_WMT; mi++) {
                    #pragma unroll
                    for (int ni = 0; ni < K1_WNT; ni++) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            :"=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                            :"r"(af[cb][mi][0]),"r"(af[cb][mi][1]),"r"(af[cb][mi][2]),"r"(af[cb][mi][3]),
                             "r"(bf[cb][ni][0]),"r"(bf[cb][ni][1]),
                             "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));
                    }
                }
            }
        }

        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();

        #pragma unroll
        for (int mi = 0; mi < K1_WMT; mi++) {
            const int r0 = bm + mi * 16 + (lane_id >> 2);
            const int r1 = r0 + 8;
            #pragma unroll
            for (int ni = 0; ni < K1_WNT; ni++) {
                const int c0 = w_col * (K1_WNT * 8) + ni * 8 + (lane_id & 3) * 2;
                *(half2*)&C[r0 * N + c0] = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
                *(half2*)&C[r1 * N + c0] = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }
}

static const int K2_BM     = 128;
static const int K2_BN     = 128;
static const int K2_BK     = 64;
static const int K2_STAGES = 5;
static const int K2_NTH    = 256;
static const int K2_WMT    = 4;
static const int K2_WNT    = 4;
static const int K2_KSTEPS = 4;
static const int K2_SA_EL  = K2_BM * K2_BK;
static const int K2_SB_EL  = K2_BK * K2_BN;
static const int K2_SMEM   = K2_STAGES * (K2_SA_EL + K2_SB_EL) * 2;

__global__ void __launch_bounds__(K2_NTH, 1)
hgemm_k2(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K)
{
    const int N      = K2_BN;
    const int ntiles = M / K2_BM;

    extern __shared__ half smem2[];
    half* sA = smem2;
    half* sB = smem2 + K2_STAGES * K2_SA_EL;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int w_row   = warp_id >> 2;
    const int w_col   = warp_id & 3;

    for (int tile_m = blockIdx.x; tile_m < ntiles; tile_m += gridDim.x) {
        const int bm     = tile_m * K2_BM;
        const int num_kt = K / K2_BK;

        float acc[K2_WMT][K2_WNT][4];
        #pragma unroll
        for (int mi = 0; mi < K2_WMT; mi++)
            #pragma unroll
            for (int ni = 0; ni < K2_WNT; ni++)
                acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

        #pragma unroll
        for (int s = 0; s < K2_STAGES - 1; s++) {
            if (s < num_kt) {
                const int kb = s * K2_BK;
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K2_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[s * K2_SA_EL + row * K2_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K2_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[s * K2_SB_EL + row * K2_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();
        }

        uint32_t af[2][K2_WMT][4];
        uint32_t bf[2][K2_WNT][2];

        #pragma unroll 1
        for (int ki = 0; ki < num_kt; ki++) {
            const int cur = ki % K2_STAGES;

            async_wait<K2_STAGES - 2>();
            __syncthreads();

            const int nxt = ki + (K2_STAGES - 1);
            if (nxt < num_kt) {
                const int ns = nxt % K2_STAGES;
                const int kb = nxt * K2_BK;
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K2_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[ns * K2_SA_EL + row * K2_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K2_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[ns * K2_SB_EL + row * K2_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();

            const half* cA = &sA[cur * K2_SA_EL];
            const half* cB = &sB[cur * K2_SB_EL];

            #pragma unroll
            for (int mi = 0; mi < K2_WMT; mi++) {
                int srow = w_row * (K2_WMT * 16) + mi * 16 + (lane_id & 15);
                int sc   = swzA(srow, (lane_id >> 4) * 8);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    :"=r"(af[0][mi][0]),"=r"(af[0][mi][1]),"=r"(af[0][mi][2]),"=r"(af[0][mi][3])
                    :"r"(smem_u32(&cA[srow * K2_BK + sc])));
            }
            #pragma unroll
            for (int ni = 0; ni < K2_WNT; ni++) {
                int rc   = w_col * (K2_WNT * 8) + ni * 8;
                int srow = lane_id & 15;
                int sc   = swzB(srow, rc);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    :"=r"(bf[0][ni][0]),"=r"(bf[0][ni][1])
                    :"r"(smem_u32(&cB[srow * K2_BN + sc])));
            }

            #pragma unroll
            for (int step = 0; step < K2_KSTEPS; step++) {
                const int cb = step & 1;
                const int nb = 1 - cb;

                if (step < K2_KSTEPS - 1) {
                    const int koff = (step + 1) * 16;
                    #pragma unroll
                    for (int mi = 0; mi < K2_WMT; mi++) {
                        int srow = w_row * (K2_WMT * 16) + mi * 16 + (lane_id & 15);
                        int sc   = swzA(srow, koff + (lane_id >> 4) * 8);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                            :"=r"(af[nb][mi][0]),"=r"(af[nb][mi][1]),"=r"(af[nb][mi][2]),"=r"(af[nb][mi][3])
                            :"r"(smem_u32(&cA[srow * K2_BK + sc])));
                    }
                    #pragma unroll
                    for (int ni = 0; ni < K2_WNT; ni++) {
                        int rc   = w_col * (K2_WNT * 8) + ni * 8;
                        int srow = koff + (lane_id & 15);
                        int sc   = swzB(srow, rc);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                            :"=r"(bf[nb][ni][0]),"=r"(bf[nb][ni][1])
                            :"r"(smem_u32(&cB[srow * K2_BN + sc])));
                    }
                }

                #pragma unroll
                for (int mi = 0; mi < K2_WMT; mi++) {
                    #pragma unroll
                    for (int ni = 0; ni < K2_WNT; ni++) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            :"=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                            :"r"(af[cb][mi][0]),"r"(af[cb][mi][1]),"r"(af[cb][mi][2]),"r"(af[cb][mi][3]),
                             "r"(bf[cb][ni][0]),"r"(bf[cb][ni][1]),
                             "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));
                    }
                }
            }
        }

        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();

        #pragma unroll
        for (int mi = 0; mi < K2_WMT; mi++) {
            const int r0 = bm + w_row * (K2_WMT * 16) + mi * 16 + (lane_id >> 2);
            const int r1 = r0 + 8;
            #pragma unroll
            for (int ni = 0; ni < K2_WNT; ni++) {
                const int c0 = w_col * (K2_WNT * 8) + ni * 8 + (lane_id & 3) * 2;
                *(half2*)&C[r0 * N + c0] = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
                *(half2*)&C[r1 * N + c0] = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }
}

static const int K3_BM     = 128;
static const int K3_BN     = 128;
static const int K3_BK     = 64;
static const int K3_STAGES = 4;
static const int K3_NTH    = 128;
static const int K3_WMT    = 8;
static const int K3_WNT    = 4;
static const int K3_KSTEPS = 4;
static const int K3_SA_EL  = K3_BM * K3_BK;
static const int K3_SB_EL  = K3_BK * K3_BN;
static const int K3_SMEM   = K3_STAGES * (K3_SA_EL + K3_SB_EL) * 2;

__global__ void __launch_bounds__(K3_NTH, 1)
hgemm_k3(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K)
{
    const int N      = K3_BN;
    const int ntiles = M / K3_BM;

    extern __shared__ half smem3[];
    half* sA = smem3;
    half* sB = smem3 + K3_STAGES * K3_SA_EL;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int w_col   = warp_id;

    for (int tile_m = blockIdx.x; tile_m < ntiles; tile_m += gridDim.x) {
        const int bm     = tile_m * K3_BM;
        const int num_kt = K / K3_BK;

        float acc[K3_WMT][K3_WNT][4];
        #pragma unroll
        for (int mi = 0; mi < K3_WMT; mi++)
            #pragma unroll
            for (int ni = 0; ni < K3_WNT; ni++)
                acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

        #pragma unroll
        for (int s = 0; s < K3_STAGES - 1; s++) {
            if (s < num_kt) {
                const int kb = s * K3_BK;
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K3_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[s * K3_SA_EL + row * K3_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K3_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[s * K3_SB_EL + row * K3_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();
        }

        uint32_t af[2][K3_WMT][4];
        uint32_t bf[2][K3_WNT][2];

        #pragma unroll 1
        for (int ki = 0; ki < num_kt; ki++) {
            const int cur = ki % K3_STAGES;

            async_wait<K3_STAGES - 2>();
            __syncthreads();

            const int nxt = ki + (K3_STAGES - 1);
            if (nxt < num_kt) {
                const int ns = nxt % K3_STAGES;
                const int kb = nxt * K3_BK;
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K3_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[ns * K3_SA_EL + row * K3_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K3_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[ns * K3_SB_EL + row * K3_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();

            const half* cA = &sA[cur * K3_SA_EL];
            const half* cB = &sB[cur * K3_SB_EL];

            #pragma unroll
            for (int mi = 0; mi < K3_WMT; mi++) {
                int srow = mi * 16 + (lane_id & 15);
                int sc   = swzA(srow, (lane_id >> 4) * 8);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    :"=r"(af[0][mi][0]),"=r"(af[0][mi][1]),"=r"(af[0][mi][2]),"=r"(af[0][mi][3])
                    :"r"(smem_u32(&cA[srow * K3_BK + sc])));
            }
            #pragma unroll
            for (int ni = 0; ni < K3_WNT; ni++) {
                int rc   = w_col * (K3_WNT * 8) + ni * 8;
                int srow = lane_id & 15;
                int sc   = swzB(srow, rc);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    :"=r"(bf[0][ni][0]),"=r"(bf[0][ni][1])
                    :"r"(smem_u32(&cB[srow * K3_BN + sc])));
            }

            #pragma unroll
            for (int step = 0; step < K3_KSTEPS; step++) {
                const int cb = step & 1;
                const int nb = 1 - cb;

                if (step < K3_KSTEPS - 1) {
                    const int koff = (step + 1) * 16;
                    #pragma unroll
                    for (int mi = 0; mi < K3_WMT; mi++) {
                        int srow = mi * 16 + (lane_id & 15);
                        int sc   = swzA(srow, koff + (lane_id >> 4) * 8);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                            :"=r"(af[nb][mi][0]),"=r"(af[nb][mi][1]),"=r"(af[nb][mi][2]),"=r"(af[nb][mi][3])
                            :"r"(smem_u32(&cA[srow * K3_BK + sc])));
                    }
                    #pragma unroll
                    for (int ni = 0; ni < K3_WNT; ni++) {
                        int rc   = w_col * (K3_WNT * 8) + ni * 8;
                        int srow = koff + (lane_id & 15);
                        int sc   = swzB(srow, rc);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                            :"=r"(bf[nb][ni][0]),"=r"(bf[nb][ni][1])
                            :"r"(smem_u32(&cB[srow * K3_BN + sc])));
                    }
                }

                #pragma unroll
                for (int mi = 0; mi < K3_WMT; mi++) {
                    #pragma unroll
                    for (int ni = 0; ni < K3_WNT; ni++) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            :"=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                            :"r"(af[cb][mi][0]),"r"(af[cb][mi][1]),"r"(af[cb][mi][2]),"r"(af[cb][mi][3]),
                             "r"(bf[cb][ni][0]),"r"(bf[cb][ni][1]),
                             "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));
                    }
                }
            }
        }

        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();

        #pragma unroll
        for (int mi = 0; mi < K3_WMT; mi++) {
            const int r0 = bm + mi * 16 + (lane_id >> 2);
            const int r1 = r0 + 8;
            #pragma unroll
            for (int ni = 0; ni < K3_WNT; ni++) {
                const int c0 = w_col * (K3_WNT * 8) + ni * 8 + (lane_id & 3) * 2;
                *(half2*)&C[r0 * N + c0] = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
                *(half2*)&C[r1 * N + c0] = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }
}

static const int K4_BM     = 64;
static const int K4_BN     = 128;
static const int K4_BK     = 64;
static const int K4_STAGES = 5;
static const int K4_NTH    = 128;
static const int K4_WMT    = 4;
static const int K4_WNT    = 4;
static const int K4_KSTEPS = 4;
static const int K4_SA_EL  = K4_BM * K4_BK;
static const int K4_SB_EL  = K4_BK * K4_BN;
static const int K4_SMEM   = K4_STAGES * (K4_SA_EL + K4_SB_EL) * 2;

__global__ void __launch_bounds__(K4_NTH, 2)
hgemm_k4(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K)
{
    const int N      = K4_BN;
    const int ntiles = M / K4_BM;

    extern __shared__ half smem4[];
    half* sA = smem4;
    half* sB = smem4 + K4_STAGES * K4_SA_EL;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int w_col   = warp_id;

    for (int tile_m = blockIdx.x; tile_m < ntiles; tile_m += gridDim.x) {
        const int bm     = tile_m * K4_BM;
        const int num_kt = K / K4_BK;

        float acc[K4_WMT][K4_WNT][4];
        #pragma unroll
        for (int mi = 0; mi < K4_WMT; mi++)
            #pragma unroll
            for (int ni = 0; ni < K4_WNT; ni++)
                acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

        #pragma unroll
        for (int s = 0; s < K4_STAGES - 1; s++) {
            if (s < num_kt) {
                const int kb = s * K4_BK;
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K4_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[s * K4_SA_EL + row * K4_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K4_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[s * K4_SB_EL + row * K4_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();
        }

        uint32_t af[2][K4_WMT][4];
        uint32_t bf[2][K4_WNT][2];

        #pragma unroll 1
        for (int ki = 0; ki < num_kt; ki++) {
            const int cur = ki % K4_STAGES;

            async_wait<K4_STAGES - 2>();
            __syncthreads();

            const int nxt = ki + (K4_STAGES - 1);
            if (nxt < num_kt) {
                const int ns = nxt % K4_STAGES;
                const int kb = nxt * K4_BK;
                #pragma unroll
                for (int li = 0; li < 4; li++) {
                    int idx = tid + li * K4_NTH;
                    int row = idx >> 3;
                    int col = (idx & 7) << 3;
                    int sc  = swzA(row, col);
                    cp_async16(smem_u32(&sA[ns * K4_SA_EL + row * K4_BK + sc]),
                               &A[(bm + row) * K + kb + col]);
                }
                #pragma unroll
                for (int li = 0; li < 8; li++) {
                    int idx = tid + li * K4_NTH;
                    int row = idx >> 4;
                    int col = (idx & 15) << 3;
                    int sc  = swzB(row, col);
                    cp_async16(smem_u32(&sB[ns * K4_SB_EL + row * K4_BN + sc]),
                               &B[(kb + row) * N + col]);
                }
            }
            async_commit();

            const half* cA = &sA[cur * K4_SA_EL];
            const half* cB = &sB[cur * K4_SB_EL];

            #pragma unroll
            for (int mi = 0; mi < K4_WMT; mi++) {
                int srow = mi * 16 + (lane_id & 15);
                int sc   = swzA(srow, (lane_id >> 4) * 8);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    :"=r"(af[0][mi][0]),"=r"(af[0][mi][1]),"=r"(af[0][mi][2]),"=r"(af[0][mi][3])
                    :"r"(smem_u32(&cA[srow * K4_BK + sc])));
            }
            #pragma unroll
            for (int ni = 0; ni < K4_WNT; ni++) {
                int rc   = w_col * (K4_WNT * 8) + ni * 8;
                int srow = lane_id & 15;
                int sc   = swzB(srow, rc);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    :"=r"(bf[0][ni][0]),"=r"(bf[0][ni][1])
                    :"r"(smem_u32(&cB[srow * K4_BN + sc])));
            }

            #pragma unroll
            for (int step = 0; step < K4_KSTEPS; step++) {
                const int cb = step & 1;
                const int nb = 1 - cb;

                if (step < K4_KSTEPS - 1) {
                    const int koff = (step + 1) * 16;
                    #pragma unroll
                    for (int mi = 0; mi < K4_WMT; mi++) {
                        int srow = mi * 16 + (lane_id & 15);
                        int sc   = swzA(srow, koff + (lane_id >> 4) * 8);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                            :"=r"(af[nb][mi][0]),"=r"(af[nb][mi][1]),"=r"(af[nb][mi][2]),"=r"(af[nb][mi][3])
                            :"r"(smem_u32(&cA[srow * K4_BK + sc])));
                    }
                    #pragma unroll
                    for (int ni = 0; ni < K4_WNT; ni++) {
                        int rc   = w_col * (K4_WNT * 8) + ni * 8;
                        int srow = koff + (lane_id & 15);
                        int sc   = swzB(srow, rc);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                            :"=r"(bf[nb][ni][0]),"=r"(bf[nb][ni][1])
                            :"r"(smem_u32(&cB[srow * K4_BN + sc])));
                    }
                }

                #pragma unroll
                for (int mi = 0; mi < K4_WMT; mi++) {
                    #pragma unroll
                    for (int ni = 0; ni < K4_WNT; ni++) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            :"=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                            :"r"(af[cb][mi][0]),"r"(af[cb][mi][1]),"r"(af[cb][mi][2]),"r"(af[cb][mi][3]),
                             "r"(bf[cb][ni][0]),"r"(bf[cb][ni][1]),
                             "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3]));
                    }
                }
            }
        }

        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();

        #pragma unroll
        for (int mi = 0; mi < K4_WMT; mi++) {
            const int r0 = bm + mi * 16 + (lane_id >> 2);
            const int r1 = r0 + 8;
            #pragma unroll
            for (int ni = 0; ni < K4_WNT; ni++) {
                const int c0 = w_col * (K4_WNT * 8) + ni * 8 + (lane_id & 3) * 2;
                *(half2*)&C[r0 * N + c0] = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
                *(half2*)&C[r1 * N + c0] = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }
}

static bool g_tuned = false;
static int  g_best  = 0;
static int  g_nb[4] = {0,0,0,0};

static void set_smem_attr(const void* fn, int sz) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sz);
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);

    const half* pa = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* pb = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       pc = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (!g_tuned) {
        int dev = 0, nsms = 132;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsms, cudaDevAttrMultiProcessorCount, dev);

        set_smem_attr((const void*)hgemm_k1, K1_SMEM);
        set_smem_attr((const void*)hgemm_k2, K2_SMEM);
        set_smem_attr((const void*)hgemm_k3, K3_SMEM);
        set_smem_attr((const void*)hgemm_k4, K4_SMEM);

        const int nt1 = M / K1_BM;
        const int nt2 = M / K2_BM;
        const int nt3 = M / K3_BM;
        const int nt4 = M / K4_BM;

        g_nb[0] = min(nt1, 2 * nsms);
        g_nb[1] = min(nt2, nsms);
        g_nb[2] = min(nt3, nsms);
        g_nb[3] = min(nt4, 2 * nsms);

        for (int i = 0; i < 5; i++) {
            hgemm_k1<<<g_nb[0], K1_NTH, K1_SMEM>>>(pa, pb, pc, M, K);
            hgemm_k2<<<g_nb[1], K2_NTH, K2_SMEM>>>(pa, pb, pc, M, K);
            hgemm_k3<<<g_nb[2], K3_NTH, K3_SMEM>>>(pa, pb, pc, M, K);
            hgemm_k4<<<g_nb[3], K4_NTH, K4_SMEM>>>(pa, pb, pc, M, K);
        }
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float t[4] = {1e9f, 1e9f, 1e9f, 1e9f};
        const int R = 200;

        auto bench = [&](int i) {
            cudaEventRecord(e0);
            for (int r = 0; r < R; r++) {
                if      (i==0) hgemm_k1<<<g_nb[0], K1_NTH, K1_SMEM>>>(pa, pb, pc, M, K);
                else if (i==1) hgemm_k2<<<g_nb[1], K2_NTH, K2_SMEM>>>(pa, pb, pc, M, K);
                else if (i==2) hgemm_k3<<<g_nb[2], K3_NTH, K3_SMEM>>>(pa, pb, pc, M, K);
                else           hgemm_k4<<<g_nb[3], K4_NTH, K4_SMEM>>>(pa, pb, pc, M, K);
            }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t[i], e0, e1);
        };

        bench(0); bench(1); bench(2); bench(3);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        g_best = 0;
        for (int i = 1; i < 4; i++) if (t[i] < t[g_best]) g_best = i;
        g_tuned = true;
    }

    if      (g_best==0) hgemm_k1<<<g_nb[0], K1_NTH, K1_SMEM>>>(pa, pb, pc, M, K);
    else if (g_best==1) hgemm_k2<<<g_nb[1], K2_NTH, K2_SMEM>>>(pa, pb, pc, M, K);
    else if (g_best==2) hgemm_k3<<<g_nb[2], K3_NTH, K3_SMEM>>>(pa, pb, pc, M, K);
    else                hgemm_k4<<<g_nb[3], K4_NTH, K4_SMEM>>>(pa, pb, pc, M, K);
}