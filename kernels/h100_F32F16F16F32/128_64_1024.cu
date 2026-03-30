#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <stdexcept>
#include <functional>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

static constexpr int M_GLOBAL = 128;
static constexpr int N_GLOBAL = 64;
static constexpr int K_GLOBAL = 1024;
static constexpr int MMA_M    = 16;
static constexpr int MMA_N    = 8;
static constexpr int MMA_K    = 16;

extern __shared__ char smem_raw[];

__global__ void __launch_bounds__(128, 4)
kernel_prime(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C) {
    static constexpr int CTA_M   = 16;
    static constexpr int NT      = 128;
    static constexpr int BK      = 128;
    static constexpr int NS      = 4;
    static constexpr int WARP_N  = 16;
    static constexpr int WMMA_M  = 1;
    static constexpr int WMMA_N  = 2;
    static constexpr int WMMA_K  = 8;
    static constexpr int A_MASK  = 15;
    static constexpr int B_MASK  = 7;

    const int cta_m_base = blockIdx.x * CTA_M;
    half* smA = reinterpret_cast<half*>(smem_raw);
    half* smB = smA + NS * CTA_M * BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_col_base = warp_id * WARP_N;

    float acc[WMMA_M][WMMA_N][4];
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto load_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smA + stage * CTA_M * BK;
        #pragma unroll 2
        for (int i = tid; i < CTA_M * (BK / 8); i += NT) {
            const int row = i / (BK / 8);
            const int lc  = i % (BK / 8);
            const int sc  = lc ^ (row & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * BK + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(A + (cta_m_base + row) * K_GLOBAL + k_off + lc * 8)), "n"(16));
        }
    };

    auto load_B = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smB + stage * BK * N_GLOBAL;
        #pragma unroll 4
        for (int i = tid; i < BK * (N_GLOBAL / 8); i += NT) {
            const int row = i / (N_GLOBAL / 8);
            const int lc  = i % (N_GLOBAL / 8);
            const int sc  = lc ^ (row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * N_GLOBAL + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(B + (k_off + row) * N_GLOBAL + lc * 8)), "n"(16));
        }
    };

    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    #pragma unroll 1
    for (int kt = 0; kt < K_GLOBAL / BK; kt++) {
        const int cs = kt % NS;
        const int pk = kt + (NS - 1);
        if (pk < K_GLOBAL / BK) {
            load_A(pk % NS, pk * BK);
            load_B(pk % NS, pk * BK);
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2));
        __syncthreads();

        const half* cA = smA + cs * CTA_M * BK;
        const half* cB = smB + cs * BK * N_GLOBAL;

        uint32_t ra_cur[WMMA_M][4], rb_cur[WMMA_N][2];
        uint32_t ra_next[WMMA_M][4], rb_next[WMMA_N][2];

        #pragma unroll
        for (int mi = 0; mi < WMMA_M; mi++) {
            const int lm_row = lane_id & 15;
            const int lm_cg  = lane_id >> 4;
            const int arow   = mi * MMA_M + lm_row;
            const int sc     = lm_cg ^ (arow & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra_cur[mi][0]), "=r"(ra_cur[mi][1]), "=r"(ra_cur[mi][2]), "=r"(ra_cur[mi][3]) : "r"(sa));
        }
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int col_b  = warp_col_base + ni * MMA_N;
            const int lm_row = lane_id & 15;
            const int lc     = col_b >> 3;
            const int sc     = lc ^ (lm_row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(cB + lm_row * N_GLOBAL + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(rb_cur[ni][0]), "=r"(rb_cur[ni][1]) : "r"(sa));
        }

        #pragma unroll
        for (int ki = 0; ki < WMMA_K; ki++) {
            if (ki + 1 < WMMA_K) {
                const int k_off_next = (ki + 1) * MMA_K;
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    const int lm_row = lane_id & 15;
                    const int lm_cg  = lane_id >> 4;
                    const int arow   = mi * MMA_M + lm_row;
                    const int lc     = (k_off_next >> 3) + lm_cg;
                    const int sc     = lc ^ (arow & A_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(ra_next[mi][0]), "=r"(ra_next[mi][1]), "=r"(ra_next[mi][2]), "=r"(ra_next[mi][3]) : "r"(sa));
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    const int col_b  = warp_col_base + ni * MMA_N;
                    const int lm_row = lane_id & 15;
                    const int ak     = k_off_next + lm_row;
                    const int lc     = col_b >> 3;
                    const int sc     = lc ^ (ak & B_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cB + ak * N_GLOBAL + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb_next[ni][0]), "=r"(rb_next[ni][1]) : "r"(sa));
                }
            }

            #pragma unroll
            for (int ni = 0; ni < WMMA_N; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(ra_cur[mi][0]), "r"(ra_cur[mi][1]), "r"(ra_cur[mi][2]), "r"(ra_cur[mi][3]),
                          "r"(rb_cur[ni][0]), "r"(rb_cur[ni][1]));
                }
            }

            if (ki + 1 < WMMA_K) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    ra_cur[mi][0]=ra_next[mi][0]; ra_cur[mi][1]=ra_next[mi][1];
                    ra_cur[mi][2]=ra_next[mi][2]; ra_cur[mi][3]=ra_next[mi][3];
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    rb_cur[ni][0]=rb_next[ni][0]; rb_cur[ni][1]=rb_next[ni][1];
                }
            }
        }
    }

    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    const int lane_row = lane_id >> 2;
    const int lane_col = (lane_id & 3) << 1;
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++) {
        const int r0 = cta_m_base + mi * MMA_M + lane_row;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int c0 = warp_col_base + ni * MMA_N + lane_col;
            *reinterpret_cast<half2*>(C + r0 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(C + r1 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(128, 3)
kernel_wm2_regdb(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C) {
    static constexpr int CTA_M   = 32;
    static constexpr int NT      = 128;
    static constexpr int BK      = 128;
    static constexpr int NS      = 4;
    static constexpr int WARP_N  = 16;
    static constexpr int WMMA_M  = 2;
    static constexpr int WMMA_N  = 2;
    static constexpr int WMMA_K  = 8;
    static constexpr int A_MASK  = 15;
    static constexpr int B_MASK  = 7;

    const int cta_m_base = blockIdx.x * CTA_M;
    half* smA = reinterpret_cast<half*>(smem_raw);
    half* smB = smA + NS * CTA_M * BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_col_base = warp_id * WARP_N;

    float acc[WMMA_M][WMMA_N][4];
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto load_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smA + stage * CTA_M * BK;
        #pragma unroll 4
        for (int i = tid; i < CTA_M * (BK / 8); i += NT) {
            const int row = i / (BK / 8);
            const int lc  = i % (BK / 8);
            const int sc  = lc ^ (row & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * BK + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(A + (cta_m_base + row) * K_GLOBAL + k_off + lc * 8)), "n"(16));
        }
    };

    auto load_B = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smB + stage * BK * N_GLOBAL;
        #pragma unroll 4
        for (int i = tid; i < BK * (N_GLOBAL / 8); i += NT) {
            const int row = i / (N_GLOBAL / 8);
            const int lc  = i % (N_GLOBAL / 8);
            const int sc  = lc ^ (row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * N_GLOBAL + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(B + (k_off + row) * N_GLOBAL + lc * 8)), "n"(16));
        }
    };

    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    #pragma unroll 1
    for (int kt = 0; kt < K_GLOBAL / BK; kt++) {
        const int cs = kt % NS;
        const int pk = kt + (NS - 1);
        if (pk < K_GLOBAL / BK) {
            load_A(pk % NS, pk * BK);
            load_B(pk % NS, pk * BK);
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2));
        __syncthreads();

        const half* cA = smA + cs * CTA_M * BK;
        const half* cB = smB + cs * BK * N_GLOBAL;

        uint32_t ra_cur[WMMA_M][4], rb_cur[WMMA_N][2];
        uint32_t ra_next[WMMA_M][4], rb_next[WMMA_N][2];

        #pragma unroll
        for (int mi = 0; mi < WMMA_M; mi++) {
            const int lm_row = lane_id & 15;
            const int lm_cg  = lane_id >> 4;
            const int arow   = mi * MMA_M + lm_row;
            const int sc     = lm_cg ^ (arow & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra_cur[mi][0]), "=r"(ra_cur[mi][1]), "=r"(ra_cur[mi][2]), "=r"(ra_cur[mi][3]) : "r"(sa));
        }
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int col_b  = warp_col_base + ni * MMA_N;
            const int lm_row = lane_id & 15;
            const int lc     = col_b >> 3;
            const int sc     = lc ^ (lm_row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(cB + lm_row * N_GLOBAL + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(rb_cur[ni][0]), "=r"(rb_cur[ni][1]) : "r"(sa));
        }

        #pragma unroll
        for (int ki = 0; ki < WMMA_K; ki++) {
            if (ki + 1 < WMMA_K) {
                const int k_off_next = (ki + 1) * MMA_K;
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    const int lm_row = lane_id & 15;
                    const int lm_cg  = lane_id >> 4;
                    const int arow   = mi * MMA_M + lm_row;
                    const int lc     = (k_off_next >> 3) + lm_cg;
                    const int sc     = lc ^ (arow & A_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(ra_next[mi][0]), "=r"(ra_next[mi][1]), "=r"(ra_next[mi][2]), "=r"(ra_next[mi][3]) : "r"(sa));
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    const int col_b  = warp_col_base + ni * MMA_N;
                    const int lm_row = lane_id & 15;
                    const int ak     = k_off_next + lm_row;
                    const int lc     = col_b >> 3;
                    const int sc     = lc ^ (ak & B_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cB + ak * N_GLOBAL + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb_next[ni][0]), "=r"(rb_next[ni][1]) : "r"(sa));
                }
            }

            #pragma unroll
            for (int ni = 0; ni < WMMA_N; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(ra_cur[mi][0]), "r"(ra_cur[mi][1]), "r"(ra_cur[mi][2]), "r"(ra_cur[mi][3]),
                          "r"(rb_cur[ni][0]), "r"(rb_cur[ni][1]));
                }
            }

            if (ki + 1 < WMMA_K) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    ra_cur[mi][0]=ra_next[mi][0]; ra_cur[mi][1]=ra_next[mi][1];
                    ra_cur[mi][2]=ra_next[mi][2]; ra_cur[mi][3]=ra_next[mi][3];
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    rb_cur[ni][0]=rb_next[ni][0]; rb_cur[ni][1]=rb_next[ni][1];
                }
            }
        }
    }

    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    const int lane_row = lane_id >> 2;
    const int lane_col = (lane_id & 3) << 1;
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++) {
        const int r0 = cta_m_base + mi * MMA_M + lane_row;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int c0 = warp_col_base + ni * MMA_N + lane_col;
            *reinterpret_cast<half2*>(C + r0 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(C + r1 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(128, 1)
kernel_single_cta(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C) {
    static constexpr int CTA_M   = 128;
    static constexpr int NT      = 128;
    static constexpr int BK      = 128;
    static constexpr int NS      = 4;
    static constexpr int WARP_M  = 32;
    static constexpr int WARP_N  = 16;
    static constexpr int WMMA_M  = 2;
    static constexpr int WMMA_N  = 2;
    static constexpr int WMMA_K  = 8;
    static constexpr int A_MASK  = 15;
    static constexpr int B_MASK  = 7;

    half* smA = reinterpret_cast<half*>(smem_raw);
    half* smB = smA + NS * CTA_M * BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_row_base = warp_id * WARP_M;
    const int warp_col_base = 0;

    float acc[WMMA_M][WMMA_N][4];
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto load_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smA + stage * CTA_M * BK;
        #pragma unroll 8
        for (int i = tid; i < CTA_M * (BK / 8); i += NT) {
            const int row = i / (BK / 8);
            const int lc  = i % (BK / 8);
            const int sc  = lc ^ (row & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * BK + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(A + row * K_GLOBAL + k_off + lc * 8)), "n"(16));
        }
    };

    auto load_B = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smB + stage * BK * N_GLOBAL;
        #pragma unroll 4
        for (int i = tid; i < BK * (N_GLOBAL / 8); i += NT) {
            const int row = i / (N_GLOBAL / 8);
            const int lc  = i % (N_GLOBAL / 8);
            const int sc  = lc ^ (row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * N_GLOBAL + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(B + (k_off + row) * N_GLOBAL + lc * 8)), "n"(16));
        }
    };

    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    (void)load_A; (void)load_B;
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();
}

__global__ void __launch_bounds__(128, 1)
kernel_single_cta_v2(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C) {
    static constexpr int CTA_M   = 128;
    static constexpr int NT      = 128;
    static constexpr int BK      = 128;
    static constexpr int NS      = 4;
    static constexpr int WARP_M  = 32;
    static constexpr int WMMA_M  = 2;
    static constexpr int WMMA_N  = 8;
    static constexpr int WMMA_K  = 8;
    static constexpr int A_MASK  = 15;
    static constexpr int B_MASK  = 7;

    half* smA = reinterpret_cast<half*>(smem_raw);
    half* smB = smA + NS * CTA_M * BK;

    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int warp_row_base = warp_id * WARP_M;

    float acc[WMMA_M][WMMA_N][4];
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto load_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smA + stage * CTA_M * BK;
        #pragma unroll 8
        for (int i = tid; i < CTA_M * (BK / 8); i += NT) {
            const int row = i / (BK / 8);
            const int lc  = i % (BK / 8);
            const int sc  = lc ^ (row & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * BK + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(A + row * K_GLOBAL + k_off + lc * 8)), "n"(16));
        }
    };

    auto load_B = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smB + stage * BK * N_GLOBAL;
        #pragma unroll 4
        for (int i = tid; i < BK * (N_GLOBAL / 8); i += NT) {
            const int row = i / (N_GLOBAL / 8);
            const int lc  = i % (N_GLOBAL / 8);
            const int sc  = lc ^ (row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * N_GLOBAL + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(B + (k_off + row) * N_GLOBAL + lc * 8)), "n"(16));
        }
    };

    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    #pragma unroll 1
    for (int kt = 0; kt < K_GLOBAL / BK; kt++) {
        const int cs = kt % NS;
        const int pk = kt + (NS - 1);
        if (pk < K_GLOBAL / BK) {
            load_A(pk % NS, pk * BK);
            load_B(pk % NS, pk * BK);
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2));
        __syncthreads();

        const half* cA = smA + cs * CTA_M * BK;
        const half* cB = smB + cs * BK * N_GLOBAL;

        uint32_t ra_cur[WMMA_M][4], rb_cur[WMMA_N][2];
        uint32_t ra_next[WMMA_M][4], rb_next[WMMA_N][2];

        #pragma unroll
        for (int mi = 0; mi < WMMA_M; mi++) {
            const int lm_row = lane_id & 15;
            const int lm_cg  = lane_id >> 4;
            const int arow   = warp_row_base + mi * MMA_M + lm_row;
            const int sc     = lm_cg ^ (arow & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra_cur[mi][0]), "=r"(ra_cur[mi][1]), "=r"(ra_cur[mi][2]), "=r"(ra_cur[mi][3]) : "r"(sa));
        }
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int col_b  = ni * MMA_N;
            const int lm_row = lane_id & 15;
            const int lc     = col_b >> 3;
            const int sc     = lc ^ (lm_row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(cB + lm_row * N_GLOBAL + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(rb_cur[ni][0]), "=r"(rb_cur[ni][1]) : "r"(sa));
        }

        #pragma unroll
        for (int ki = 0; ki < WMMA_K; ki++) {
            if (ki + 1 < WMMA_K) {
                const int k_off_next = (ki + 1) * MMA_K;
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    const int lm_row = lane_id & 15;
                    const int lm_cg  = lane_id >> 4;
                    const int arow   = warp_row_base + mi * MMA_M + lm_row;
                    const int lc     = (k_off_next >> 3) + lm_cg;
                    const int sc     = lc ^ (arow & A_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(ra_next[mi][0]), "=r"(ra_next[mi][1]), "=r"(ra_next[mi][2]), "=r"(ra_next[mi][3]) : "r"(sa));
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    const int col_b  = ni * MMA_N;
                    const int lm_row = lane_id & 15;
                    const int ak     = k_off_next + lm_row;
                    const int lc     = col_b >> 3;
                    const int sc     = lc ^ (ak & B_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cB + ak * N_GLOBAL + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb_next[ni][0]), "=r"(rb_next[ni][1]) : "r"(sa));
                }
            }

            #pragma unroll
            for (int ni = 0; ni < WMMA_N; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(ra_cur[mi][0]), "r"(ra_cur[mi][1]), "r"(ra_cur[mi][2]), "r"(ra_cur[mi][3]),
                          "r"(rb_cur[ni][0]), "r"(rb_cur[ni][1]));
                }
            }

            if (ki + 1 < WMMA_K) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    ra_cur[mi][0]=ra_next[mi][0]; ra_cur[mi][1]=ra_next[mi][1];
                    ra_cur[mi][2]=ra_next[mi][2]; ra_cur[mi][3]=ra_next[mi][3];
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    rb_cur[ni][0]=rb_next[ni][0]; rb_cur[ni][1]=rb_next[ni][1];
                }
            }
        }
    }

    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    const int lane_row = lane_id >> 2;
    const int lane_col = (lane_id & 3) << 1;
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++) {
        const int r0 = warp_row_base + mi * MMA_M + lane_row;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int c0 = ni * MMA_N + lane_col;
            *reinterpret_cast<half2*>(C + r0 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(C + r1 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(128, 5)
kernel_ns3_regdb(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C) {
    static constexpr int CTA_M   = 16;
    static constexpr int NT      = 128;
    static constexpr int BK      = 128;
    static constexpr int NS      = 3;
    static constexpr int WARP_N  = 16;
    static constexpr int WMMA_M  = 1;
    static constexpr int WMMA_N  = 2;
    static constexpr int WMMA_K  = 8;
    static constexpr int A_MASK  = 15;
    static constexpr int B_MASK  = 7;

    const int cta_m_base = blockIdx.x * CTA_M;
    half* smA = reinterpret_cast<half*>(smem_raw);
    half* smB = smA + NS * CTA_M * BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_col_base = warp_id * WARP_N;

    float acc[WMMA_M][WMMA_N][4];
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto load_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smA + stage * CTA_M * BK;
        #pragma unroll 2
        for (int i = tid; i < CTA_M * (BK / 8); i += NT) {
            const int row = i / (BK / 8);
            const int lc  = i % (BK / 8);
            const int sc  = lc ^ (row & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * BK + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(A + (cta_m_base + row) * K_GLOBAL + k_off + lc * 8)), "n"(16));
        }
    };

    auto load_B = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smB + stage * BK * N_GLOBAL;
        #pragma unroll 4
        for (int i = tid; i < BK * (N_GLOBAL / 8); i += NT) {
            const int row = i / (N_GLOBAL / 8);
            const int lc  = i % (N_GLOBAL / 8);
            const int sc  = lc ^ (row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * N_GLOBAL + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(B + (k_off + row) * N_GLOBAL + lc * 8)), "n"(16));
        }
    };

    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    #pragma unroll 1
    for (int kt = 0; kt < K_GLOBAL / BK; kt++) {
        const int cs = kt % NS;
        const int pk = kt + (NS - 1);
        if (pk < K_GLOBAL / BK) {
            load_A(pk % NS, pk * BK);
            load_B(pk % NS, pk * BK);
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2));
        __syncthreads();

        const half* cA = smA + cs * CTA_M * BK;
        const half* cB = smB + cs * BK * N_GLOBAL;

        uint32_t ra_cur[WMMA_M][4], rb_cur[WMMA_N][2];
        uint32_t ra_next[WMMA_M][4], rb_next[WMMA_N][2];

        #pragma unroll
        for (int mi = 0; mi < WMMA_M; mi++) {
            const int lm_row = lane_id & 15;
            const int lm_cg  = lane_id >> 4;
            const int arow   = mi * MMA_M + lm_row;
            const int sc     = lm_cg ^ (arow & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra_cur[mi][0]), "=r"(ra_cur[mi][1]), "=r"(ra_cur[mi][2]), "=r"(ra_cur[mi][3]) : "r"(sa));
        }
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int col_b  = warp_col_base + ni * MMA_N;
            const int lm_row = lane_id & 15;
            const int lc     = col_b >> 3;
            const int sc     = lc ^ (lm_row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(cB + lm_row * N_GLOBAL + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(rb_cur[ni][0]), "=r"(rb_cur[ni][1]) : "r"(sa));
        }

        #pragma unroll
        for (int ki = 0; ki < WMMA_K; ki++) {
            if (ki + 1 < WMMA_K) {
                const int k_off_next = (ki + 1) * MMA_K;
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    const int lm_row = lane_id & 15;
                    const int lm_cg  = lane_id >> 4;
                    const int arow   = mi * MMA_M + lm_row;
                    const int lc     = (k_off_next >> 3) + lm_cg;
                    const int sc     = lc ^ (arow & A_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(ra_next[mi][0]), "=r"(ra_next[mi][1]), "=r"(ra_next[mi][2]), "=r"(ra_next[mi][3]) : "r"(sa));
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    const int col_b  = warp_col_base + ni * MMA_N;
                    const int lm_row = lane_id & 15;
                    const int ak     = k_off_next + lm_row;
                    const int lc     = col_b >> 3;
                    const int sc     = lc ^ (ak & B_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cB + ak * N_GLOBAL + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb_next[ni][0]), "=r"(rb_next[ni][1]) : "r"(sa));
                }
            }

            #pragma unroll
            for (int ni = 0; ni < WMMA_N; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(ra_cur[mi][0]), "r"(ra_cur[mi][1]), "r"(ra_cur[mi][2]), "r"(ra_cur[mi][3]),
                          "r"(rb_cur[ni][0]), "r"(rb_cur[ni][1]));
                }
            }

            if (ki + 1 < WMMA_K) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    ra_cur[mi][0]=ra_next[mi][0]; ra_cur[mi][1]=ra_next[mi][1];
                    ra_cur[mi][2]=ra_next[mi][2]; ra_cur[mi][3]=ra_next[mi][3];
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    rb_cur[ni][0]=rb_next[ni][0]; rb_cur[ni][1]=rb_next[ni][1];
                }
            }
        }
    }

    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    const int lane_row = lane_id >> 2;
    const int lane_col = (lane_id & 3) << 1;
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++) {
        const int r0 = cta_m_base + mi * MMA_M + lane_row;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int c0 = warp_col_base + ni * MMA_N + lane_col;
            *reinterpret_cast<half2*>(C + r0 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(C + r1 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(128, 2)
kernel_2cta_regdb(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C) {
    static constexpr int CTA_M   = 64;
    static constexpr int NT      = 128;
    static constexpr int BK      = 128;
    static constexpr int NS      = 4;
    static constexpr int WARP_N  = 16;
    static constexpr int WMMA_M  = 4;
    static constexpr int WMMA_N  = 2;
    static constexpr int WMMA_K  = 8;
    static constexpr int A_MASK  = 15;
    static constexpr int B_MASK  = 7;

    const int cta_m_base = blockIdx.x * CTA_M;
    half* smA = reinterpret_cast<half*>(smem_raw);
    half* smB = smA + NS * CTA_M * BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_col_base = warp_id * WARP_N;

    float acc[WMMA_M][WMMA_N][4];
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    auto load_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smA + stage * CTA_M * BK;
        #pragma unroll 4
        for (int i = tid; i < CTA_M * (BK / 8); i += NT) {
            const int row = i / (BK / 8);
            const int lc  = i % (BK / 8);
            const int sc  = lc ^ (row & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * BK + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(A + (cta_m_base + row) * K_GLOBAL + k_off + lc * 8)), "n"(16));
        }
    };

    auto load_B = [&](int stage, int k_off) __attribute__((always_inline)) {
        half* dst = smB + stage * BK * N_GLOBAL;
        #pragma unroll 4
        for (int i = tid; i < BK * (N_GLOBAL / 8); i += NT) {
            const int row = i / (N_GLOBAL / 8);
            const int lc  = i % (N_GLOBAL / 8);
            const int sc  = lc ^ (row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(dst + row * N_GLOBAL + sc * 8);
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(sa), "l"((unsigned long long)(B + (k_off + row) * N_GLOBAL + lc * 8)), "n"(16));
        }
    };

    #pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    #pragma unroll 1
    for (int kt = 0; kt < K_GLOBAL / BK; kt++) {
        const int cs = kt % NS;
        const int pk = kt + (NS - 1);
        if (pk < K_GLOBAL / BK) {
            load_A(pk % NS, pk * BK);
            load_B(pk % NS, pk * BK);
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2));
        __syncthreads();

        const half* cA = smA + cs * CTA_M * BK;
        const half* cB = smB + cs * BK * N_GLOBAL;

        uint32_t ra_cur[WMMA_M][4], rb_cur[WMMA_N][2];
        uint32_t ra_next[WMMA_M][4], rb_next[WMMA_N][2];

        #pragma unroll
        for (int mi = 0; mi < WMMA_M; mi++) {
            const int lm_row = lane_id & 15;
            const int lm_cg  = lane_id >> 4;
            const int arow   = mi * MMA_M + lm_row;
            const int sc     = lm_cg ^ (arow & A_MASK);
            uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra_cur[mi][0]), "=r"(ra_cur[mi][1]), "=r"(ra_cur[mi][2]), "=r"(ra_cur[mi][3]) : "r"(sa));
        }
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int col_b  = warp_col_base + ni * MMA_N;
            const int lm_row = lane_id & 15;
            const int lc     = col_b >> 3;
            const int sc     = lc ^ (lm_row & B_MASK);
            uint32_t sa = __cvta_generic_to_shared(cB + lm_row * N_GLOBAL + sc * 8);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(rb_cur[ni][0]), "=r"(rb_cur[ni][1]) : "r"(sa));
        }

        #pragma unroll
        for (int ki = 0; ki < WMMA_K; ki++) {
            if (ki + 1 < WMMA_K) {
                const int k_off_next = (ki + 1) * MMA_K;
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    const int lm_row = lane_id & 15;
                    const int lm_cg  = lane_id >> 4;
                    const int arow   = mi * MMA_M + lm_row;
                    const int lc     = (k_off_next >> 3) + lm_cg;
                    const int sc     = lc ^ (arow & A_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cA + arow * BK + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(ra_next[mi][0]), "=r"(ra_next[mi][1]), "=r"(ra_next[mi][2]), "=r"(ra_next[mi][3]) : "r"(sa));
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    const int col_b  = warp_col_base + ni * MMA_N;
                    const int lm_row = lane_id & 15;
                    const int ak     = k_off_next + lm_row;
                    const int lc     = col_b >> 3;
                    const int sc     = lc ^ (ak & B_MASK);
                    uint32_t sa = __cvta_generic_to_shared(cB + ak * N_GLOBAL + sc * 8);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb_next[ni][0]), "=r"(rb_next[ni][1]) : "r"(sa));
                }
            }

            #pragma unroll
            for (int ni = 0; ni < WMMA_N; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(ra_cur[mi][0]), "r"(ra_cur[mi][1]), "r"(ra_cur[mi][2]), "r"(ra_cur[mi][3]),
                          "r"(rb_cur[ni][0]), "r"(rb_cur[ni][1]));
                }
            }

            if (ki + 1 < WMMA_K) {
                #pragma unroll
                for (int mi = 0; mi < WMMA_M; mi++) {
                    ra_cur[mi][0]=ra_next[mi][0]; ra_cur[mi][1]=ra_next[mi][1];
                    ra_cur[mi][2]=ra_next[mi][2]; ra_cur[mi][3]=ra_next[mi][3];
                }
                #pragma unroll
                for (int ni = 0; ni < WMMA_N; ni++) {
                    rb_cur[ni][0]=rb_next[ni][0]; rb_cur[ni][1]=rb_next[ni][1];
                }
            }
        }
    }

    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    const int lane_row = lane_id >> 2;
    const int lane_col = (lane_id & 3) << 1;
    #pragma unroll
    for (int mi = 0; mi < WMMA_M; mi++) {
        const int r0 = cta_m_base + mi * MMA_M + lane_row;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WMMA_N; ni++) {
            const int c0 = warp_col_base + ni * MMA_N + lane_col;
            *reinterpret_cast<half2*>(C + r0 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(C + r1 * N_GLOBAL + c0) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

static constexpr size_t S_PRIME    = (size_t)4*16*128*2  + (size_t)4*128*64*2;
static constexpr size_t S_WM2      = (size_t)4*32*128*2  + (size_t)4*128*64*2;
static constexpr size_t S_SINGLE   = (size_t)4*128*128*2 + (size_t)4*128*64*2;
static constexpr size_t S_NS3      = (size_t)3*16*128*2  + (size_t)3*128*64*2;
static constexpr size_t S_2CTA     = (size_t)4*64*128*2  + (size_t)4*128*64*2;

static float bench_fn(cudaEvent_t ev0, cudaEvent_t ev1, std::function<void()> fn,
                      int warmup=8, int iters=50) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0;
    cudaEventElapsedTime(&ms, ev0, ev1);
    return ms / iters;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half*       C = reinterpret_cast<half*>(c.data_ptr());

    static bool attrs_set = false;
    if (!attrs_set) {
        cudaFuncSetAttribute(kernel_prime,          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S_PRIME);
        cudaFuncSetAttribute(kernel_wm2_regdb,      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S_WM2);
        cudaFuncSetAttribute(kernel_single_cta_v2,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S_SINGLE);
        cudaFuncSetAttribute(kernel_ns3_regdb,      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S_NS3);
        cudaFuncSetAttribute(kernel_2cta_regdb,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S_2CTA);
        attrs_set = true;
    }

    static int best = -1;
    if (best < 0) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        float t[5];
        t[0] = bench_fn(ev0, ev1, [&](){ kernel_prime<<<8,128,S_PRIME>>>(A,B,C); });
        t[1] = bench_fn(ev0, ev1, [&](){ kernel_wm2_regdb<<<4,128,S_WM2>>>(A,B,C); });
        t[2] = bench_fn(ev0, ev1, [&](){ kernel_single_cta_v2<<<1,128,S_SINGLE>>>(A,B,C); });
        t[3] = bench_fn(ev0, ev1, [&](){ kernel_ns3_regdb<<<8,128,S_NS3>>>(A,B,C); });
        t[4] = bench_fn(ev0, ev1, [&](){ kernel_2cta_regdb<<<2,128,S_2CTA>>>(A,B,C); });

        best = 0;
        for (int i = 1; i < 5; i++) if (t[i] < t[best]) best = i;

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }

    switch (best) {
        case 0: kernel_prime<<<8,128,S_PRIME>>>(A,B,C); break;
        case 1: kernel_wm2_regdb<<<4,128,S_WM2>>>(A,B,C); break;
        case 2: kernel_single_cta_v2<<<1,128,S_SINGLE>>>(A,B,C); break;
        case 3: kernel_ns3_regdb<<<8,128,S_NS3>>>(A,B,C); break;
        case 4: kernel_2cta_regdb<<<2,128,S_2CTA>>>(A,B,C); break;
        default: kernel_prime<<<8,128,S_PRIME>>>(A,B,C); break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}