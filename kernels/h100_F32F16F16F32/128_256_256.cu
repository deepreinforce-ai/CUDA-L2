#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("values must be " #th_type); \
  }

static constexpr int BM  = 64;
static constexpr int BN  = 64;
static constexpr int BK  = 32;
static constexpr int NS  = 5;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

static constexpr int WR  = 2;
static constexpr int WC  = 2;
static constexpr int WTM = 2;
static constexpr int WTN = 4;

static constexpr int A_STR = 32;
static constexpr int B_STR = 64;
static constexpr int A_SZ  = BM * A_STR;
static constexpr int B_SZ  = BK * B_STR;
static constexpr int SMEM  = NS * (A_SZ + B_SZ) * (int)sizeof(half);

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    unsigned d = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(src) : "memory");
}

__global__ void __launch_bounds__(128, 4)
hgemm_64x64_v5(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N, int K)
{
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int wr = warp_id / WC;
    const int wc = warp_id % WC;
    const int wm = wr * (BM / WR);
    const int wn = wc * (BN / WC);

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + NS * A_SZ;

    float acc[WTM][WTN][4];
    #pragma unroll
    for (int i = 0; i < WTM; i++)
        #pragma unroll
        for (int j = 0; j < WTN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int grp    = lane_id >> 3;
    const int in_grp = lane_id & 7;

    const int al_row = tid >> 2;
    const int al_col = (tid & 3) << 3;

    const int bl_row = tid >> 3;
    const int bl_col = (tid & 7) << 3;

    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        const int ko = s * BK;
        half* pA = sA + s * A_SZ;
        #pragma unroll
        for (int dr = 0; dr < 2; dr++) {
            const int row  = al_row + dr * 32;
            const int swiz = al_col ^ ((row & 3) << 3);
            cp16(pA + row * A_STR + swiz, &A[(bm + row) * K + ko + al_col]);
        }
        half* pB = sB + s * B_SZ;
        #pragma unroll
        for (int dr = 0; dr < 2; dr++) {
            const int row  = bl_row + dr * 16;
            const int swiz = bl_col ^ ((row & 7) << 3);
            cp16(pB + row * B_STR + swiz, &B[(ko + row) * N + bn + bl_col]);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    #pragma unroll 1
    for (int kt = 0; kt < 8; kt++) {
        const int cs = kt % NS;
        const int ns = (kt + NS - 1) % NS;
        const int pf = kt + (NS - 1);

        if (pf < 8) {
            const int ko = pf * BK;
            half* pA = sA + ns * A_SZ;
            #pragma unroll
            for (int dr = 0; dr < 2; dr++) {
                const int row  = al_row + dr * 32;
                const int swiz = al_col ^ ((row & 3) << 3);
                cp16(pA + row * A_STR + swiz, &A[(bm + row) * K + ko + al_col]);
            }
            half* pB = sB + ns * B_SZ;
            #pragma unroll
            for (int dr = 0; dr < 2; dr++) {
                const int row  = bl_row + dr * 16;
                const int swiz = bl_col ^ ((row & 7) << 3);
                cp16(pB + row * B_STR + swiz, &B[(ko + row) * N + bn + bl_col]);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 4;\n" ::: "memory");
        __syncthreads();

        const half* cA = sA + cs * A_SZ;
        const half* cB = sB + cs * B_SZ;

        uint32_t fA0[WTM][4], fA1[WTM][4];
        uint32_t fB0[WTN][2], fB1[WTN][2];

        #pragma unroll
        for (int tm = 0; tm < WTM; tm++) {
            const int ar    = wm + tm * MMA_M + (grp & 1) * 8 + in_grp;
            const int ac    = (grp >> 1) * 8;
            const int aswiz = ac ^ ((ar & 3) << 3);
            unsigned addr = __cvta_generic_to_shared(&cA[ar * A_STR + aswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(fA0[tm][0]), "=r"(fA0[tm][1]), "=r"(fA0[tm][2]), "=r"(fA0[tm][3])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int tn = 0; tn < WTN; tn++) {
            const int bk    = lane_id & 15;
            const int bc    = wn + tn * MMA_N;
            const int bswiz = bc ^ ((bk & 7) << 3);
            unsigned addr = __cvta_generic_to_shared(&cB[bk * B_STR + bswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared::cta.b16 {%0,%1}, [%2];\n"
                : "=r"(fB0[tn][0]), "=r"(fB0[tn][1])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int tm = 0; tm < WTM; tm++) {
            const int ar    = wm + tm * MMA_M + (grp & 1) * 8 + in_grp;
            const int ac    = 16 + (grp >> 1) * 8;
            const int aswiz = ac ^ ((ar & 3) << 3);
            unsigned addr = __cvta_generic_to_shared(&cA[ar * A_STR + aswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(fA1[tm][0]), "=r"(fA1[tm][1]), "=r"(fA1[tm][2]), "=r"(fA1[tm][3])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int tn = 0; tn < WTN; tn++) {
            const int bk    = 16 + (lane_id & 15);
            const int bc    = wn + tn * MMA_N;
            const int bswiz = bc ^ ((bk & 7) << 3);
            unsigned addr = __cvta_generic_to_shared(&cB[bk * B_STR + bswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared::cta.b16 {%0,%1}, [%2];\n"
                : "=r"(fB1[tn][0]), "=r"(fB1[tn][1])
                : "r"(addr)
            );
        }

        #pragma unroll
        for (int tn = 0; tn < WTN; tn++) {
            #pragma unroll
            for (int tm = 0; tm < WTM; tm++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(acc[tm][tn][0]), "=f"(acc[tm][tn][1]),
                      "=f"(acc[tm][tn][2]), "=f"(acc[tm][tn][3])
                    : "r"(fA0[tm][0]), "r"(fA0[tm][1]), "r"(fA0[tm][2]), "r"(fA0[tm][3]),
                      "r"(fB0[tn][0]), "r"(fB0[tn][1]),
                      "f"(acc[tm][tn][0]), "f"(acc[tm][tn][1]),
                      "f"(acc[tm][tn][2]), "f"(acc[tm][tn][3])
                );
            }
        }

        #pragma unroll
        for (int tn = 0; tn < WTN; tn++) {
            #pragma unroll
            for (int tm = 0; tm < WTM; tm++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(acc[tm][tn][0]), "=f"(acc[tm][tn][1]),
                      "=f"(acc[tm][tn][2]), "=f"(acc[tm][tn][3])
                    : "r"(fA1[tm][0]), "r"(fA1[tm][1]), "r"(fA1[tm][2]), "r"(fA1[tm][3]),
                      "r"(fB1[tn][0]), "r"(fB1[tn][1]),
                      "f"(acc[tm][tn][0]), "f"(acc[tm][tn][1]),
                      "f"(acc[tm][tn][2]), "f"(acc[tm][tn][3])
                );
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int ep_row = lane_id >> 2;
    const int ep_col = (lane_id & 3) << 1;

    #pragma unroll
    for (int tm = 0; tm < WTM; tm++) {
        const int r0 = bm + wm + tm * MMA_M + ep_row;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int tn = 0; tn < WTN; tn++) {
            const int c0 = bn + wn + tn * MMA_N + ep_col;
            *reinterpret_cast<half2*>(&C[r0 * N + c0]) =
                __floats2half2_rn(acc[tm][tn][0], acc[tm][tn][1]);
            *reinterpret_cast<half2*>(&C[r1 * N + c0]) =
                __floats2half2_rn(acc[tm][tn][2], acc[tm][tn][3]);
        }
    }
}

static constexpr int BM2  = 128;
static constexpr int BN2  = 64;
static constexpr int BK2  = 32;
static constexpr int NS2  = 4;
static constexpr int WR2  = 2, WC2 = 4;
static constexpr int WTM2 = 4, WTN2 = 2;
static constexpr int A2S  = 32, B2S = 64;
static constexpr int A2SZ = BM2 * A2S;
static constexpr int B2SZ = BK2 * B2S;
static constexpr int SMEM2 = NS2 * (A2SZ + B2SZ) * (int)sizeof(half);

__global__ void __launch_bounds__(256, 3)
hgemm_128x64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N, int K)
{
    const int bm = blockIdx.y * BM2;
    const int bn = blockIdx.x * BN2;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int wr = warp_id / WC2;
    const int wc = warp_id % WC2;
    const int wm = wr * (BM2 / WR2);
    const int wn = wc * (BN2 / WC2);

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + NS2 * A2SZ;

    float acc[WTM2][WTN2][4];
    #pragma unroll
    for (int i = 0; i < WTM2; i++)
        #pragma unroll
        for (int j = 0; j < WTN2; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int grp    = lane_id >> 3;
    const int in_grp = lane_id & 7;

    const int al_row = tid >> 2;
    const int al_col = (tid & 3) << 3;
    const int bl_row = tid >> 3;
    const int bl_col = (tid & 7) << 3;

    #pragma unroll
    for (int s = 0; s < NS2 - 1; s++) {
        const int ko = s * BK2;
        half* pA = sA + s * A2SZ;
        #pragma unroll
        for (int dr = 0; dr < 2; dr++) {
            const int row  = al_row + dr * 64;
            const int swiz = al_col ^ ((row & 3) << 3);
            unsigned d = __cvta_generic_to_shared(pA + row * A2S + swiz);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(d), "l"(&A[(bm + row) * K + ko + al_col]) : "memory");
        }
        half* pB = sB + s * B2SZ;
        {
            const int swiz = bl_col ^ ((bl_row & 7) << 3);
            unsigned d = __cvta_generic_to_shared(pB + bl_row * B2S + swiz);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(d), "l"(&B[(ko + bl_row) * N + bn + bl_col]) : "memory");
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    #pragma unroll 1
    for (int kt = 0; kt < 8; kt++) {
        const int cs = kt % NS2;
        const int ns = (kt + NS2 - 1) % NS2;
        const int pf = kt + (NS2 - 1);

        if (pf < 8) {
            const int ko = pf * BK2;
            half* pA = sA + ns * A2SZ;
            #pragma unroll
            for (int dr = 0; dr < 2; dr++) {
                const int row  = al_row + dr * 64;
                const int swiz = al_col ^ ((row & 3) << 3);
                unsigned d = __cvta_generic_to_shared(pA + row * A2S + swiz);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(d), "l"(&A[(bm + row) * K + ko + al_col]) : "memory");
            }
            half* pB = sB + ns * B2SZ;
            {
                const int swiz = bl_col ^ ((bl_row & 7) << 3);
                unsigned d = __cvta_generic_to_shared(pB + bl_row * B2S + swiz);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(d), "l"(&B[(pf * BK2 + bl_row) * N + bn + bl_col]) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 3;\n" ::: "memory");
        __syncthreads();

        const half* cA = sA + cs * A2SZ;
        const half* cB = sB + cs * B2SZ;

        uint32_t fA0[WTM2][4], fA1[WTM2][4];
        uint32_t fB0[WTN2][2], fB1[WTN2][2];

        #pragma unroll
        for (int tm = 0; tm < WTM2; tm++) {
            const int ar    = wm + tm * 16 + (grp & 1) * 8 + in_grp;
            const int ac    = (grp >> 1) * 8;
            const int aswiz = ac ^ ((ar & 3) << 3);
            unsigned addr = __cvta_generic_to_shared(&cA[ar * A2S + aswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(fA0[tm][0]), "=r"(fA0[tm][1]), "=r"(fA0[tm][2]), "=r"(fA0[tm][3])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int tn = 0; tn < WTN2; tn++) {
            const int bk    = lane_id & 15;
            const int bc    = wn + tn * 8;
            const int bswiz = bc ^ ((bk & 7) << 3);
            unsigned addr = __cvta_generic_to_shared(&cB[bk * B2S + bswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared::cta.b16 {%0,%1}, [%2];\n"
                : "=r"(fB0[tn][0]), "=r"(fB0[tn][1])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int tm = 0; tm < WTM2; tm++) {
            const int ar    = wm + tm * 16 + (grp & 1) * 8 + in_grp;
            const int ac    = 16 + (grp >> 1) * 8;
            const int aswiz = ac ^ ((ar & 3) << 3);
            unsigned addr = __cvta_generic_to_shared(&cA[ar * A2S + aswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(fA1[tm][0]), "=r"(fA1[tm][1]), "=r"(fA1[tm][2]), "=r"(fA1[tm][3])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int tn = 0; tn < WTN2; tn++) {
            const int bk    = 16 + (lane_id & 15);
            const int bc    = wn + tn * 8;
            const int bswiz = bc ^ ((bk & 7) << 3);
            unsigned addr = __cvta_generic_to_shared(&cB[bk * B2S + bswiz]);
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared::cta.b16 {%0,%1}, [%2];\n"
                : "=r"(fB1[tn][0]), "=r"(fB1[tn][1])
                : "r"(addr)
            );
        }

        #pragma unroll
        for (int tn = 0; tn < WTN2; tn++) {
            #pragma unroll
            for (int tm = 0; tm < WTM2; tm++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(acc[tm][tn][0]), "=f"(acc[tm][tn][1]),
                      "=f"(acc[tm][tn][2]), "=f"(acc[tm][tn][3])
                    : "r"(fA0[tm][0]), "r"(fA0[tm][1]), "r"(fA0[tm][2]), "r"(fA0[tm][3]),
                      "r"(fB0[tn][0]), "r"(fB0[tn][1]),
                      "f"(acc[tm][tn][0]), "f"(acc[tm][tn][1]),
                      "f"(acc[tm][tn][2]), "f"(acc[tm][tn][3])
                );
            }
        }
        #pragma unroll
        for (int tn = 0; tn < WTN2; tn++) {
            #pragma unroll
            for (int tm = 0; tm < WTM2; tm++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(acc[tm][tn][0]), "=f"(acc[tm][tn][1]),
                      "=f"(acc[tm][tn][2]), "=f"(acc[tm][tn][3])
                    : "r"(fA1[tm][0]), "r"(fA1[tm][1]), "r"(fA1[tm][2]), "r"(fA1[tm][3]),
                      "r"(fB1[tn][0]), "r"(fB1[tn][1]),
                      "f"(acc[tm][tn][0]), "f"(acc[tm][tn][1]),
                      "f"(acc[tm][tn][2]), "f"(acc[tm][tn][3])
                );
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int ep_row = lane_id >> 2;
    const int ep_col = (lane_id & 3) << 1;

    #pragma unroll
    for (int tm = 0; tm < WTM2; tm++) {
        const int r0 = bm + wm + tm * 16 + ep_row;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int tn = 0; tn < WTN2; tn++) {
            const int c0 = bn + wn + tn * 8 + ep_col;
            *reinterpret_cast<half2*>(&C[r0 * N + c0]) =
                __floats2half2_rn(acc[tm][tn][0], acc[tm][tn][1]);
            *reinterpret_cast<half2*>(&C[r1 * N + c0]) =
                __floats2half2_rn(acc[tm][tn][2], acc[tm][tn][3]);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A    = reinterpret_cast<const half*>(a.data_ptr());
    const half* B    = reinterpret_cast<const half*>(b.data_ptr());
    half*       Cptr = reinterpret_cast<half*>(c.data_ptr());

    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(hgemm_64x64_v5,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(hgemm_128x64,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2);
        init = true;
    }

    {
        dim3 grid(N / BN, M / BM);
        dim3 block(128);
        hgemm_64x64_v5<<<grid, block, SMEM>>>(A, B, Cptr, N, K);
    }
}