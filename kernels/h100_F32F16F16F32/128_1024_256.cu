#ifndef _GLIBCXX_USE_CXX11_ABI
#define _GLIBCXX_USE_CXX11_ABI 0
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

static constexpr int M_DIM = 128;
static constexpr int N_DIM = 1024;
static constexpr int K_DIM = 256;

static constexpr int BM = 32;
static constexpr int BN = 64;
static constexpr int BK = 64;
static constexpr int STAGES = 3;
static constexpr int THREADS = 128;
static constexpr int K_TILES = K_DIM / BK;

__global__ void __launch_bounds__(THREADS, 2)
hgemm_optimized_h100_bk64(const half* __restrict__ A,
                           const half* __restrict__ B,
                           half* __restrict__ C) {
    const int bm = blockIdx.y;
    const int bn = blockIdx.x ^ (bm & 1);

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int m_off = bm * BM;
    const int n_off = bn * BN;

    __shared__ half sA[STAGES][BM][BK + 8];
    __shared__ half sB[STAGES][BK][BN + 8];

    float acc[4][4];
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        acc[ni][0] = 0.f; acc[ni][1] = 0.f; acc[ni][2] = 0.f; acc[ni][3] = 0.f;
    }

    auto g2s = [&](int stage, int kt) __attribute__((always_inline)) {
        const int ko = kt * BK;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * THREADS;
            const int row = idx >> 3;
            const int col8 = idx & 7;

            uint32_t smem_addr = __cvta_generic_to_shared(&sA[stage][row][col8 * 8]);
            const void* gmem_ptr = (const void*)(A + (m_off + row) * K_DIM + ko + col8 * 8);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(gmem_ptr));
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + i * THREADS;
            const int row = idx >> 3;
            const int col8 = idx & 7;

            uint32_t smem_addr = __cvta_generic_to_shared(&sB[stage][row][col8 * 8]);
            const void* gmem_ptr = (const void*)(B + (ko + row) * N_DIM + n_off + col8 * 8);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(gmem_ptr));
        }
    };

    g2s(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    g2s(1, 1);
    asm volatile("cp.async.commit_group;\n" ::);

    #pragma unroll
    for (int kt = 0; kt < K_TILES; ++kt) {
        const int sc = kt % STAGES;
        const int kn = kt + (STAGES - 1);

        if (kn < K_TILES) g2s(kn % STAGES, kn);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        auto ld_kk = [&](int kk, uint32_t fA[4], uint32_t fB[4][2]) __attribute__((always_inline)) {
            {
                const int sr = warp_m * 16 + (lane & 15);
                const int sc2 = kk * 16 + ((lane >> 4) << 3);
                const uint32_t addr = __cvta_generic_to_shared(&sA[sc][sr][sc2]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(fA[0]), "=r"(fA[1]), "=r"(fA[2]), "=r"(fA[3]) : "r"(addr));
            }

            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int sr = kk * 16 + (lane & 15);
                const int sc2 = warp_n * 32 + ni * 8 + ((lane >> 4) << 3);
                const uint32_t addr = __cvta_generic_to_shared(&sB[sc][sr][sc2]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(fB[ni][0]), "=r"(fB[ni][1]) : "r"(addr));
            }
        };

        uint32_t curA[4], nxtA[4];
        uint32_t curB[4][2], nxtB[4][2];

        ld_kk(0, curA, curB);

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            if (kk < 3) ld_kk(kk + 1, nxtA, nxtB);

            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(acc[ni][0]), "=f"(acc[ni][1]), "=f"(acc[ni][2]), "=f"(acc[ni][3])
                    : "r"(curA[0]), "r"(curA[1]), "r"(curA[2]), "r"(curA[3]),
                      "r"(curB[ni][0]), "r"(curB[ni][1]),
                      "f"(acc[ni][0]), "f"(acc[ni][1]), "f"(acc[ni][2]), "f"(acc[ni][3]));
            }

            if (kk < 3) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) curA[i] = nxtA[i];
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    curB[ni][0] = nxtB[ni][0];
                    curB[ni][1] = nxtB[ni][1];
                }
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);

    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    const int c0 = (lane & 3) << 1;

    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const int tm = m_off + warp_m * 16;
        const int tn = n_off + warp_n * 32 + ni * 8;

        *reinterpret_cast<half2*>(&C[(tm + r0) * N_DIM + tn + c0]) =
            __floats2half2_rn(acc[ni][0], acc[ni][1]);
        *reinterpret_cast<half2*>(&C[(tm + r1) * N_DIM + tn + c0]) =
            __floats2half2_rn(acc[ni][2], acc[ni][3]);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a,
                                torch::Tensor b,
                                torch::Tensor b_col_major,
                                torch::Tensor c)
{
    (void)b_col_major;

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C       = reinterpret_cast<half*>(c.data_ptr());

    const dim3 grid(16, 4);
    hgemm_optimized_h100_bk64<<<grid, THREADS>>>(A, B, C);
}