#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <stdint.h>

#define CP_ASYNC_CG_16(dst, src) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), \
           "l"(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src))))

#define CP_ASYNC_COMMIT()  asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT(n)   asm volatile("cp.async.wait_group %0;\n" :: "n"(n) : "memory")

__device__ __forceinline__
void mma_m16n8k16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
          "r"(b[0]),"r"(b[1]),
          "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

__device__ __forceinline__
void ldmatrix_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t r[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__device__ __forceinline__ int swz64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

static constexpr int M_FIX = 4096;
static constexpr int N_FIX = 64;
static constexpr int K_FIX = 1024;

static constexpr int BM      = 32;
static constexpr int BN      = 64;
static constexpr int BK      = 128;
static constexpr int NTH     = 64;

static constexpr int MMA_M   = 16;
static constexpr int MMA_N   = 8;
static constexpr int MMA_K   = 16;
static constexpr int WM      = 1;
static constexpr int WN      = BN / MMA_N;
static constexpr int INNER_K = BK / MMA_K;
static constexpr int STAGES  = 3;

static constexpr int TILE_BYTES = (BM * BK + BK * BN) * 2;
static constexpr int SMEM_BYTES = STAGES * TILE_BYTES;

__global__ __launch_bounds__(NTH, 6)
void hgemm_fixed_m32n64k128_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C)
{
    const int bm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int blk_row = bm * BM;
    const int num_kt  = K_FIX / BK;

    extern __shared__ uint16_t smem_raw[];
    half* As = reinterpret_cast<half*>(smem_raw);
    half* Bs = As + STAGES * BM * BK;

    const int warp_row = warp_id * MMA_M;

    float acc[WM][WN][4];
    #pragma unroll
    for (int mi = 0; mi < WM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            acc[mi][ni][0] = 0.f;
            acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f;
            acc[mi][ni][3] = 0.f;
        }
    }

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_kt) {
            const int k0 = s * BK;

            {
                half* Ap = As + s * BM * BK;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int lin = tid + i * NTH;
                    const int row = lin >> 4;
                    const int col = (lin & 15) << 3;
                    half* dst = Ap + row * BK + swz64(row, col);
                    const half* src = &A[(blk_row + row) * K_FIX + (k0 + col)];
                    CP_ASYNC_CG_16(dst, src);
                }
            }

            {
                half* Bp = Bs + s * BK * BN;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int lin = tid + i * NTH;
                    const int row = lin >> 3;
                    const int col = (lin & 7) << 3;
                    half* dst = Bp + row * BN + swz64(row, col);
                    const half* src = &B[(k0 + row) * N_FIX + col];
                    CP_ASYNC_CG_16(dst, src);
                }
            }
        }
        CP_ASYNC_COMMIT();
    }

    #pragma unroll 1
    for (int kt = 0; kt < num_kt; ++kt) {
        const int pf = kt + STAGES - 1;
        if (pf < num_kt) {
            const int k0 = pf * BK;

            {
                half* Ap = As + (pf % STAGES) * BM * BK;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int lin = tid + i * NTH;
                    const int row = lin >> 4;
                    const int col = (lin & 15) << 3;
                    half* dst = Ap + row * BK + swz64(row, col);
                    const half* src = &A[(blk_row + row) * K_FIX + (k0 + col)];
                    CP_ASYNC_CG_16(dst, src);
                }
            }

            {
                half* Bp = Bs + (pf % STAGES) * BK * BN;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int lin = tid + i * NTH;
                    const int row = lin >> 3;
                    const int col = (lin & 7) << 3;
                    half* dst = Bp + row * BN + swz64(row, col);
                    const half* src = &B[(k0 + row) * N_FIX + col];
                    CP_ASYNC_CG_16(dst, src);
                }
            }
        }

        CP_ASYNC_COMMIT();
        CP_ASYNC_WAIT(STAGES - 2);
        __syncthreads();

        const int cs   = kt % STAGES;
        const half* Ac = As + cs * BM * BK;
        const half* Bc = Bs + cs * BK * BN;

        uint32_t a_cur[WM][4];
        uint32_t b_cur[WN][2];

        #pragma unroll
        for (int mi = 0; mi < WM; ++mi) {
            const int pr = warp_row + mi * MMA_M + (lane_id & 15);
            const int pc = (lane_id >> 4) << 3;
            ldmatrix_x4(a_cur[mi], __cvta_generic_to_shared(Ac + pr * BK + swz64(pr, pc)));
        }

        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            const int pr = lane_id & 15;
            const int pc = ni * MMA_N;
            ldmatrix_x2_trans(b_cur[ni], __cvta_generic_to_shared(Bc + pr * BN + swz64(pr, pc)));
        }

        #pragma unroll
        for (int ki = 0; ki < INNER_K; ++ki) {
            uint32_t a_nxt[WM][4];
            uint32_t b_nxt[WN][2];
            const bool has_next = (ki + 1 < INNER_K);

            if (has_next) {
                const int nki = ki + 1;

                #pragma unroll
                for (int mi = 0; mi < WM; ++mi) {
                    const int pr = warp_row + mi * MMA_M + (lane_id & 15);
                    const int pc = nki * MMA_K + ((lane_id >> 4) << 3);
                    ldmatrix_x4(a_nxt[mi], __cvta_generic_to_shared(Ac + pr * BK + swz64(pr, pc)));
                }

                #pragma unroll
                for (int ni = 0; ni < WN; ++ni) {
                    const int pr = nki * MMA_K + (lane_id & 15);
                    const int pc = ni * MMA_N;
                    ldmatrix_x2_trans(b_nxt[ni], __cvta_generic_to_shared(Bc + pr * BN + swz64(pr, pc)));
                }
            }

            #pragma unroll
            for (int mi = 0; mi < WM; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WN; ++ni) {
                    mma_m16n8k16(acc[mi][ni], a_cur[mi], b_cur[ni], acc[mi][ni]);
                }
            }

            if (has_next) {
                #pragma unroll
                for (int mi = 0; mi < WM; ++mi) {
                    a_cur[mi][0] = a_nxt[mi][0];
                    a_cur[mi][1] = a_nxt[mi][1];
                    a_cur[mi][2] = a_nxt[mi][2];
                    a_cur[mi][3] = a_nxt[mi][3];
                }
                #pragma unroll
                for (int ni = 0; ni < WN; ++ni) {
                    b_cur[ni][0] = b_nxt[ni][0];
                    b_cur[ni][1] = b_nxt[ni][1];
                }
            }
        }

        __syncthreads();
    }

    const int lr = lane_id >> 2;
    const int lc = (lane_id & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < WM; ++mi) {
        const int r0 = blk_row + warp_row + mi * MMA_M + lr;
        const int r1 = r0 + 8;

        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            const int col = ni * MMA_N + lc;
            *reinterpret_cast<half2*>(&C[r0 * N_FIX + col]) =
                __float22half2_rn(make_float2(acc[mi][ni][0], acc[mi][ni][1]));
            *reinterpret_cast<half2*>(&C[r1 * N_FIX + col]) =
                __float22half2_rn(make_float2(acc[mi][ni][2], acc[mi][ni][3]));
        }
    }
}

static bool g_init = false;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    (void)b_col_major;

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr());
    half* C_ptr       = reinterpret_cast<half*>(c.data_ptr());

    if (!g_init) {
        cudaFuncSetAttribute(
            hgemm_fixed_m32n64k128_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_BYTES);
        g_init = true;
    }

    constexpr int grid_m = M_FIX / BM;
    hgemm_fixed_m32n64k128_kernel<<<grid_m, NTH, SMEM_BYTES>>>(A_ptr, B_ptr, C_ptr);
}