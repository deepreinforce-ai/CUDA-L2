#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

namespace cg = cooperative_groups;

__device__ __forceinline__
uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__device__ __forceinline__
void cp_async16_ca(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__
void cp_async16_cg(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
}

__device__ __forceinline__
void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

static constexpr int M_DIM    = 128;
static constexpr int N_DIM    = 128;
static constexpr int K_DIM    = 16384;
static constexpr int NTHREADS = 256;
static constexpr int WM       = 2;
static constexpr int WN       = 8;
static constexpr int N_ELEMS  = M_DIM * N_DIM;

static constexpr int NS   = 64;
static constexpr int KP   = K_DIM / NS;
static constexpr int BK   = 64;
static constexpr int ST   = 4;
static constexpr int KT   = KP / BK;
static constexpr int ASTR = BK + 8;
static constexpr int BSTR = N_DIM + 8;
static constexpr int ASZ  = M_DIM * ASTR;
static constexpr int BSZ  = BK * BSTR;
static constexpr int SSTSZ= ASZ + BSZ;
static constexpr int SMEM = ST * SSTSZ * (int)sizeof(__half);

static constexpr int NS2   = 128;
static constexpr int KP2   = K_DIM / NS2;
static constexpr int BK2   = 64;
static constexpr int ST2   = 3;
static constexpr int KT2   = KP2 / BK2;
static constexpr int ASTR2 = BK2 + 8;
static constexpr int BSTR2 = N_DIM + 8;
static constexpr int ASZ2  = M_DIM * ASTR2;
static constexpr int SSTSZ2= ASZ2 + BK2 * BSTR2;
static constexpr int SMEM2 = ST2 * SSTSZ2 * (int)sizeof(__half);

__global__ void __launch_bounds__(NTHREADS, 1)
hgemm_pass1_64(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ SplitBuf
) {
    const int split_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m   = warp_row * 32;
    const int warp_n   = warp_col * 64;

    float acc[WM][WN][4];
    #pragma unroll
    for (int mi = 0; mi < WM; mi++)
        #pragma unroll
        for (int ni = 0; ni < WN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] =
            acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    extern __shared__ __half smem[];
    const int k_start = split_id * KP;

    #pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        const int k_off = k_start + s * BK;
        __half* dA = smem + s * SSTSZ;
        __half* dB = smem + s * SSTSZ + ASZ;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx  = tid + i * NTHREADS;
            int row  = idx >> 3;
            int col8 = idx & 7;
            int phys = col8 ^ ((row >> 2) & 7);
            uint32_t dst = smem_u32addr(dA + row * ASTR + phys * 8);
            cp_async16_ca(dst, A + (size_t)row * K_DIM + k_off + col8 * 8);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx   = tid + i * NTHREADS;
            int k_row = idx >> 4;
            int col8  = idx & 15;
            int phys  = col8 ^ ((k_row >> 1) & 15);
            uint32_t dst = smem_u32addr(dB + k_row * BSTR + phys * 8);
            cp_async16_cg(dst, B + (size_t)(k_off + k_row) * N_DIM + col8 * 8);
        }
        cp_async_commit();
    }

    int cur = 0;
    #pragma unroll 1
    for (int kt = 0; kt < KT; kt++) {
        const int next_tile  = kt + ST - 1;
        const int next_stage = (cur + ST - 1) % ST;
        if (next_tile < KT) {
            const int k_off = k_start + next_tile * BK;
            __half* dA = smem + next_stage * SSTSZ;
            __half* dB = smem + next_stage * SSTSZ + ASZ;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx  = tid + i * NTHREADS;
                int row  = idx >> 3;
                int col8 = idx & 7;
                int phys = col8 ^ ((row >> 2) & 7);
                uint32_t dst = smem_u32addr(dA + row * ASTR + phys * 8);
                cp_async16_ca(dst, A + (size_t)row * K_DIM + k_off + col8 * 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx   = tid + i * NTHREADS;
                int k_row = idx >> 4;
                int col8  = idx & 15;
                int phys  = col8 ^ ((k_row >> 1) & 15);
                uint32_t dst = smem_u32addr(dB + k_row * BSTR + phys * 8);
                cp_async16_cg(dst, B + (size_t)(k_off + k_row) * N_DIM + col8 * 8);
            }
        }
        cp_async_commit();
        cp_async_wait<ST - 2>();
        __syncthreads();

        __half* cA = smem + cur * SSTSZ;
        __half* cB = smem + cur * SSTSZ + ASZ;

        uint32_t a_frag[WM][4];
        #pragma unroll
        for (int mi = 0; mi < WM; mi++) {
            const int smem_row  = warp_m + mi * 16 + (lane_id & 15);
            const int col8_idx  = (lane_id >> 4);
            const int phys_col8 = col8_idx ^ ((smem_row >> 2) & 7);
            uint32_t sptr = smem_u32addr(cA + smem_row * ASTR + phys_col8 * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                  "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                : "r"(sptr));
        }

        constexpr int KK_ITERS = BK / 16;
        #pragma unroll
        for (int kk = 0; kk < KK_ITERS; kk++) {
            const int k16 = kk * 16;
            uint32_t a_frag_next[WM][4];
            if (kk + 1 < KK_ITERS) {
                const int k16_next = (kk + 1) * 16;
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    const int smem_row  = warp_m + mi * 16 + (lane_id & 15);
                    const int col8_base = (k16_next >> 3) + (lane_id >> 4);
                    const int phys_col8 = col8_base ^ ((smem_row >> 2) & 7);
                    uint32_t sptr = smem_u32addr(cA + smem_row * ASTR + phys_col8 * 8);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a_frag_next[mi][0]), "=r"(a_frag_next[mi][1]),
                          "=r"(a_frag_next[mi][2]), "=r"(a_frag_next[mi][3])
                        : "r"(sptr));
                }
            }
            uint32_t b_frags[WN][2];
            #pragma unroll
            for (int ni = 0; ni < WN; ni++) {
                const int n_base      = warp_n + ni * 8;
                const int b_k_row     = k16 + (lane_id & 15);
                const int b_col8      = n_base >> 3;
                const int phys_b_col8 = b_col8 ^ ((b_k_row >> 1) & 15);
                uint32_t b_sptr = smem_u32addr(cB + b_k_row * BSTR + phys_b_col8 * 8);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frags[ni][0]), "=r"(b_frags[ni][1])
                    : "r"(b_sptr));
            }
            #pragma unroll
            for (int ni = 0; ni < WN; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    mma_m16n8k16(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a_frag[mi][0], a_frag[mi][1],
                        a_frag[mi][2], a_frag[mi][3],
                        b_frags[ni][0], b_frags[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
                }
            }
            if (kk + 1 < KK_ITERS) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++)
                    #pragma unroll
                    for (int r = 0; r < 4; r++)
                        a_frag[mi][r] = a_frag_next[mi][r];
            }
        }
        cur = (cur + 1) % ST;
    }
    cp_async_wait<0>();

    float* my_buf = SplitBuf + (size_t)split_id * N_ELEMS;
    #pragma unroll
    for (int mi = 0; mi < WM; mi++) {
        const int row0 = warp_m + mi * 16 + (lane_id >> 2);
        const int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            const int col0 = warp_n + ni * 8 + (lane_id & 3) * 2;
            *reinterpret_cast<float2*>(&my_buf[row0 * N_DIM + col0]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&my_buf[row1 * N_DIM + col0]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(NTHREADS, 1)
hgemm_pass1_128(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ SplitBuf
) {
    const int split_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m   = warp_row * 32;
    const int warp_n   = warp_col * 64;

    float acc[WM][WN][4];
    #pragma unroll
    for (int mi = 0; mi < WM; mi++)
        #pragma unroll
        for (int ni = 0; ni < WN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] =
            acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    extern __shared__ __half smem[];
    const int k_start = split_id * KP2;

    #pragma unroll
    for (int s = 0; s < ST2 - 1; s++) {
        if (s < KT2) {
            const int k_off = k_start + s * BK2;
            __half* dA = smem + s * SSTSZ2;
            __half* dB = smem + s * SSTSZ2 + ASZ2;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx  = tid + i * NTHREADS;
                int row  = idx >> 3;
                int col8 = idx & 7;
                int phys = col8 ^ ((row >> 2) & 7);
                uint32_t dst = smem_u32addr(dA + row * ASTR2 + phys * 8);
                cp_async16_ca(dst, A + (size_t)row * K_DIM + k_off + col8 * 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx   = tid + i * NTHREADS;
                int k_row = idx >> 4;
                int col8  = idx & 15;
                int phys  = col8 ^ ((k_row >> 1) & 15);
                uint32_t dst = smem_u32addr(dB + k_row * BSTR2 + phys * 8);
                cp_async16_cg(dst, B + (size_t)(k_off + k_row) * N_DIM + col8 * 8);
            }
        }
        cp_async_commit();
    }

    int cur = 0;
    #pragma unroll 1
    for (int kt = 0; kt < KT2; kt++) {
        const int next_tile  = kt + ST2 - 1;
        const int next_stage = (cur + ST2 - 1) % ST2;
        if (next_tile < KT2) {
            const int k_off = k_start + next_tile * BK2;
            __half* dA = smem + next_stage * SSTSZ2;
            __half* dB = smem + next_stage * SSTSZ2 + ASZ2;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx  = tid + i * NTHREADS;
                int row  = idx >> 3;
                int col8 = idx & 7;
                int phys = col8 ^ ((row >> 2) & 7);
                uint32_t dst = smem_u32addr(dA + row * ASTR2 + phys * 8);
                cp_async16_ca(dst, A + (size_t)row * K_DIM + k_off + col8 * 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx   = tid + i * NTHREADS;
                int k_row = idx >> 4;
                int col8  = idx & 15;
                int phys  = col8 ^ ((k_row >> 1) & 15);
                uint32_t dst = smem_u32addr(dB + k_row * BSTR2 + phys * 8);
                cp_async16_cg(dst, B + (size_t)(k_off + k_row) * N_DIM + col8 * 8);
            }
        }
        cp_async_commit();
        cp_async_wait<ST2 - 2>();
        __syncthreads();

        __half* cA = smem + cur * SSTSZ2;
        __half* cB = smem + cur * SSTSZ2 + ASZ2;

        uint32_t a_frag[WM][4];
        #pragma unroll
        for (int mi = 0; mi < WM; mi++) {
            const int smem_row  = warp_m + mi * 16 + (lane_id & 15);
            const int col8_idx  = (lane_id >> 4);
            const int phys_col8 = col8_idx ^ ((smem_row >> 2) & 7);
            uint32_t sptr = smem_u32addr(cA + smem_row * ASTR2 + phys_col8 * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                  "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                : "r"(sptr));
        }

        constexpr int KK_ITERS = BK2 / 16;
        #pragma unroll
        for (int kk = 0; kk < KK_ITERS; kk++) {
            const int k16 = kk * 16;
            uint32_t a_frag_next[WM][4];
            if (kk + 1 < KK_ITERS) {
                const int k16_next = (kk + 1) * 16;
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    const int smem_row  = warp_m + mi * 16 + (lane_id & 15);
                    const int col8_base = (k16_next >> 3) + (lane_id >> 4);
                    const int phys_col8 = col8_base ^ ((smem_row >> 2) & 7);
                    uint32_t sptr = smem_u32addr(cA + smem_row * ASTR2 + phys_col8 * 8);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a_frag_next[mi][0]), "=r"(a_frag_next[mi][1]),
                          "=r"(a_frag_next[mi][2]), "=r"(a_frag_next[mi][3])
                        : "r"(sptr));
                }
            }
            uint32_t b_frags[WN][2];
            #pragma unroll
            for (int ni = 0; ni < WN; ni++) {
                const int n_base      = warp_n + ni * 8;
                const int b_k_row     = k16 + (lane_id & 15);
                const int b_col8      = n_base >> 3;
                const int phys_b_col8 = b_col8 ^ ((b_k_row >> 1) & 15);
                uint32_t b_sptr = smem_u32addr(cB + b_k_row * BSTR2 + phys_b_col8 * 8);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frags[ni][0]), "=r"(b_frags[ni][1])
                    : "r"(b_sptr));
            }
            #pragma unroll
            for (int ni = 0; ni < WN; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    mma_m16n8k16(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a_frag[mi][0], a_frag[mi][1],
                        a_frag[mi][2], a_frag[mi][3],
                        b_frags[ni][0], b_frags[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
                }
            }
            if (kk + 1 < KK_ITERS) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++)
                    #pragma unroll
                    for (int r = 0; r < 4; r++)
                        a_frag[mi][r] = a_frag_next[mi][r];
            }
        }
        cur = (cur + 1) % ST2;
    }
    cp_async_wait<0>();

    float* my_buf = SplitBuf + (size_t)split_id * N_ELEMS;
    #pragma unroll
    for (int mi = 0; mi < WM; mi++) {
        const int row0 = warp_m + mi * 16 + (lane_id >> 2);
        const int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            const int col0 = warp_n + ni * 8 + (lane_id & 3) * 2;
            *reinterpret_cast<float2*>(&my_buf[row0 * N_DIM + col0]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&my_buf[row1 * N_DIM + col0]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(1024, 2)
hgemm_reduce_f4_64(
    const float* __restrict__ SplitBuf,
    __half*      __restrict__ C
) {
    const int elem_idx = blockIdx.x * 1024 + threadIdx.x;
    if (elem_idx >= N_ELEMS) return;
    const float* base = SplitBuf + elem_idx;

    float s0=__ldg(base+(size_t)0*N_ELEMS),   s1=__ldg(base+(size_t)1*N_ELEMS);
    float s2=__ldg(base+(size_t)2*N_ELEMS),   s3=__ldg(base+(size_t)3*N_ELEMS);
    float s4=__ldg(base+(size_t)4*N_ELEMS),   s5=__ldg(base+(size_t)5*N_ELEMS);
    float s6=__ldg(base+(size_t)6*N_ELEMS),   s7=__ldg(base+(size_t)7*N_ELEMS);
    float s8=__ldg(base+(size_t)8*N_ELEMS),   s9=__ldg(base+(size_t)9*N_ELEMS);
    float s10=__ldg(base+(size_t)10*N_ELEMS), s11=__ldg(base+(size_t)11*N_ELEMS);
    float s12=__ldg(base+(size_t)12*N_ELEMS), s13=__ldg(base+(size_t)13*N_ELEMS);
    float s14=__ldg(base+(size_t)14*N_ELEMS), s15=__ldg(base+(size_t)15*N_ELEMS);
    float s16=__ldg(base+(size_t)16*N_ELEMS), s17=__ldg(base+(size_t)17*N_ELEMS);
    float s18=__ldg(base+(size_t)18*N_ELEMS), s19=__ldg(base+(size_t)19*N_ELEMS);
    float s20=__ldg(base+(size_t)20*N_ELEMS), s21=__ldg(base+(size_t)21*N_ELEMS);
    float s22=__ldg(base+(size_t)22*N_ELEMS), s23=__ldg(base+(size_t)23*N_ELEMS);
    float s24=__ldg(base+(size_t)24*N_ELEMS), s25=__ldg(base+(size_t)25*N_ELEMS);
    float s26=__ldg(base+(size_t)26*N_ELEMS), s27=__ldg(base+(size_t)27*N_ELEMS);
    float s28=__ldg(base+(size_t)28*N_ELEMS), s29=__ldg(base+(size_t)29*N_ELEMS);
    float s30=__ldg(base+(size_t)30*N_ELEMS), s31=__ldg(base+(size_t)31*N_ELEMS);
    float s32=__ldg(base+(size_t)32*N_ELEMS), s33=__ldg(base+(size_t)33*N_ELEMS);
    float s34=__ldg(base+(size_t)34*N_ELEMS), s35=__ldg(base+(size_t)35*N_ELEMS);
    float s36=__ldg(base+(size_t)36*N_ELEMS), s37=__ldg(base+(size_t)37*N_ELEMS);
    float s38=__ldg(base+(size_t)38*N_ELEMS), s39=__ldg(base+(size_t)39*N_ELEMS);
    float s40=__ldg(base+(size_t)40*N_ELEMS), s41=__ldg(base+(size_t)41*N_ELEMS);
    float s42=__ldg(base+(size_t)42*N_ELEMS), s43=__ldg(base+(size_t)43*N_ELEMS);
    float s44=__ldg(base+(size_t)44*N_ELEMS), s45=__ldg(base+(size_t)45*N_ELEMS);
    float s46=__ldg(base+(size_t)46*N_ELEMS), s47=__ldg(base+(size_t)47*N_ELEMS);
    float s48=__ldg(base+(size_t)48*N_ELEMS), s49=__ldg(base+(size_t)49*N_ELEMS);
    float s50=__ldg(base+(size_t)50*N_ELEMS), s51=__ldg(base+(size_t)51*N_ELEMS);
    float s52=__ldg(base+(size_t)52*N_ELEMS), s53=__ldg(base+(size_t)53*N_ELEMS);
    float s54=__ldg(base+(size_t)54*N_ELEMS), s55=__ldg(base+(size_t)55*N_ELEMS);
    float s56=__ldg(base+(size_t)56*N_ELEMS), s57=__ldg(base+(size_t)57*N_ELEMS);
    float s58=__ldg(base+(size_t)58*N_ELEMS), s59=__ldg(base+(size_t)59*N_ELEMS);
    float s60=__ldg(base+(size_t)60*N_ELEMS), s61=__ldg(base+(size_t)61*N_ELEMS);
    float s62=__ldg(base+(size_t)62*N_ELEMS), s63=__ldg(base+(size_t)63*N_ELEMS);

    float a0=s0+s1,a1=s2+s3,a2=s4+s5,a3=s6+s7;
    float a4=s8+s9,a5=s10+s11,a6=s12+s13,a7=s14+s15;
    float a8=s16+s17,a9=s18+s19,a10=s20+s21,a11=s22+s23;
    float a12=s24+s25,a13=s26+s27,a14=s28+s29,a15=s30+s31;
    float a16=s32+s33,a17=s34+s35,a18=s36+s37,a19=s38+s39;
    float a20=s40+s41,a21=s42+s43,a22=s44+s45,a23=s46+s47;
    float a24=s48+s49,a25=s50+s51,a26=s52+s53,a27=s54+s55;
    float a28=s56+s57,a29=s58+s59,a30=s60+s61,a31=s62+s63;
    float b0=a0+a1,b1=a2+a3,b2=a4+a5,b3=a6+a7;
    float b4=a8+a9,b5=a10+a11,b6=a12+a13,b7=a14+a15;
    float b8=a16+a17,b9=a18+a19,b10=a20+a21,b11=a22+a23;
    float b12=a24+a25,b13=a26+a27,b14=a28+a29,b15=a30+a31;
    float c0=b0+b1,c1=b2+b3,c2=b4+b5,c3=b6+b7;
    float c4=b8+b9,c5=b10+b11,c6=b12+b13,c7=b14+b15;
    float d0=c0+c1,d1=c2+c3,d2=c4+c5,d3=c6+c7;
    float e0=d0+d1,e1=d2+d3;
    C[elem_idx] = __float2half(e0+e1);
}

__global__ void __launch_bounds__(256, 8)
hgemm_reduce_f2_64(
    const float* __restrict__ SplitBuf,
    __half*      __restrict__ C
) {
    const int elem2_idx = blockIdx.x * 256 + threadIdx.x;
    if (elem2_idx >= N_ELEMS / 2) return;
    const int base_idx = elem2_idx * 2;

    float s0=0,s1=0,t0=0,t1=0,u0=0,u1=0,v0=0,v1=0;
    float w0=0,w1=0,x0=0,x1=0,y0=0,y1=0,z0=0,z1=0;

    #pragma unroll
    for (int s = 0; s < NS; s += 16) {
        float2 r0  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 0)*N_ELEMS+base_idx);
        float2 r1  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 1)*N_ELEMS+base_idx);
        float2 r2  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 2)*N_ELEMS+base_idx);
        float2 r3  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 3)*N_ELEMS+base_idx);
        float2 r4  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 4)*N_ELEMS+base_idx);
        float2 r5  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 5)*N_ELEMS+base_idx);
        float2 r6  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 6)*N_ELEMS+base_idx);
        float2 r7  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 7)*N_ELEMS+base_idx);
        float2 r8  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 8)*N_ELEMS+base_idx);
        float2 r9  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 9)*N_ELEMS+base_idx);
        float2 r10 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+10)*N_ELEMS+base_idx);
        float2 r11 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+11)*N_ELEMS+base_idx);
        float2 r12 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+12)*N_ELEMS+base_idx);
        float2 r13 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+13)*N_ELEMS+base_idx);
        float2 r14 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+14)*N_ELEMS+base_idx);
        float2 r15 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+15)*N_ELEMS+base_idx);
        s0+=r0.x+r1.x; s1+=r0.y+r1.y;
        t0+=r2.x+r3.x; t1+=r2.y+r3.y;
        u0+=r4.x+r5.x; u1+=r4.y+r5.y;
        v0+=r6.x+r7.x; v1+=r6.y+r7.y;
        w0+=r8.x+r9.x; w1+=r8.y+r9.y;
        x0+=r10.x+r11.x; x1+=r10.y+r11.y;
        y0+=r12.x+r13.x; y1+=r12.y+r13.y;
        z0+=r14.x+r15.x; z1+=r14.y+r15.y;
    }
    C[base_idx+0] = __float2half(((s0+t0)+(u0+v0))+((w0+x0)+(y0+z0)));
    C[base_idx+1] = __float2half(((s1+t1)+(u1+v1))+((w1+x1)+(y1+z1)));
}

__global__ void __launch_bounds__(256, 8)
hgemm_reduce_f2_128(
    const float* __restrict__ SplitBuf,
    __half*      __restrict__ C
) {
    const int elem2_idx = blockIdx.x * 256 + threadIdx.x;
    if (elem2_idx >= N_ELEMS / 2) return;
    const int base_idx = elem2_idx * 2;

    float s0=0,s1=0,t0=0,t1=0,u0=0,u1=0,v0=0,v1=0;
    float w0=0,w1=0,x0=0,x1=0,y0=0,y1=0,z0=0,z1=0;

    #pragma unroll
    for (int s = 0; s < NS2; s += 16) {
        float2 r0  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 0)*N_ELEMS+base_idx);
        float2 r1  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 1)*N_ELEMS+base_idx);
        float2 r2  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 2)*N_ELEMS+base_idx);
        float2 r3  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 3)*N_ELEMS+base_idx);
        float2 r4  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 4)*N_ELEMS+base_idx);
        float2 r5  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 5)*N_ELEMS+base_idx);
        float2 r6  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 6)*N_ELEMS+base_idx);
        float2 r7  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 7)*N_ELEMS+base_idx);
        float2 r8  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 8)*N_ELEMS+base_idx);
        float2 r9  = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+ 9)*N_ELEMS+base_idx);
        float2 r10 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+10)*N_ELEMS+base_idx);
        float2 r11 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+11)*N_ELEMS+base_idx);
        float2 r12 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+12)*N_ELEMS+base_idx);
        float2 r13 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+13)*N_ELEMS+base_idx);
        float2 r14 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+14)*N_ELEMS+base_idx);
        float2 r15 = *reinterpret_cast<const float2*>(SplitBuf+(size_t)(s+15)*N_ELEMS+base_idx);
        s0+=r0.x+r1.x; s1+=r0.y+r1.y;
        t0+=r2.x+r3.x; t1+=r2.y+r3.y;
        u0+=r4.x+r5.x; u1+=r4.y+r5.y;
        v0+=r6.x+r7.x; v1+=r6.y+r7.y;
        w0+=r8.x+r9.x; w1+=r8.y+r9.y;
        x0+=r10.x+r11.x; x1+=r10.y+r11.y;
        y0+=r12.x+r13.x; y1+=r12.y+r13.y;
        z0+=r14.x+r15.x; z1+=r14.y+r15.y;
    }
    C[base_idx+0] = __float2half(((s0+t0)+(u0+v0))+((w0+x0)+(y0+z0)));
    C[base_idx+1] = __float2half(((s1+t1)+(u1+v1))+((w1+x1)+(y1+z1)));
}

__global__ void __launch_bounds__(256, 4)
hgemm_reduce_4elem_64(
    const float* __restrict__ SplitBuf,
    __half*      __restrict__ C
) {
    const int tid4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = tid4 * 4;
    if (base >= N_ELEMS) return;

    float s0=0,s1=0,s2=0,s3=0;
    float t0=0,t1=0,t2=0,t3=0;
    float u0=0,u1=0,u2=0,u3=0;
    float v0=0,v1=0,v2=0,v3=0;
    float w0=0,w1=0,w2=0,w3=0;
    float x0=0,x1=0,x2=0,x3=0;
    float y0=0,y1=0,y2=0,y3=0;
    float z0=0,z1=0,z2=0,z3=0;
    float A0=0,A1=0,A2=0,A3=0;
    float B0=0,B1=0,B2=0,B3=0;
    float Cc0=0,Cc1=0,Cc2=0,Cc3=0;
    float D0=0,D1=0,D2=0,D3=0;
    float E0=0,E1=0,E2=0,E3=0;
    float F0=0,F1=0,F2=0,F3=0;
    float G0=0,G1=0,G2=0,G3=0;
    float H0=0,H1=0,H2=0,H3=0;

    #define LOAD4(dst0,dst1,dst2,dst3, sp) do { \
        float4 rr = *reinterpret_cast<const float4*>(SplitBuf+(size_t)(sp)*N_ELEMS+base); \
        dst0+=rr.x; dst1+=rr.y; dst2+=rr.z; dst3+=rr.w; \
    } while(0)

    LOAD4(s0,s1,s2,s3,  0); LOAD4(t0,t1,t2,t3,  1);
    LOAD4(u0,u1,u2,u3,  2); LOAD4(v0,v1,v2,v3,  3);
    LOAD4(w0,w1,w2,w3,  4); LOAD4(x0,x1,x2,x3,  5);
    LOAD4(y0,y1,y2,y3,  6); LOAD4(z0,z1,z2,z3,  7);
    LOAD4(A0,A1,A2,A3,  8); LOAD4(B0,B1,B2,B3,  9);
    LOAD4(Cc0,Cc1,Cc2,Cc3,10); LOAD4(D0,D1,D2,D3,11);
    LOAD4(E0,E1,E2,E3, 12); LOAD4(F0,F1,F2,F3, 13);
    LOAD4(G0,G1,G2,G3, 14); LOAD4(H0,H1,H2,H3, 15);
    LOAD4(s0,s1,s2,s3, 16); LOAD4(t0,t1,t2,t3, 17);
    LOAD4(u0,u1,u2,u3, 18); LOAD4(v0,v1,v2,v3, 19);
    LOAD4(w0,w1,w2,w3, 20); LOAD4(x0,x1,x2,x3, 21);
    LOAD4(y0,y1,y2,y3, 22); LOAD4(z0,z1,z2,z3, 23);
    LOAD4(A0,A1,A2,A3, 24); LOAD4(B0,B1,B2,B3, 25);
    LOAD4(Cc0,Cc1,Cc2,Cc3,26); LOAD4(D0,D1,D2,D3,27);
    LOAD4(E0,E1,E2,E3, 28); LOAD4(F0,F1,F2,F3, 29);
    LOAD4(G0,G1,G2,G3, 30); LOAD4(H0,H1,H2,H3, 31);
    LOAD4(s0,s1,s2,s3, 32); LOAD4(t0,t1,t2,t3, 33);
    LOAD4(u0,u1,u2,u3, 34); LOAD4(v0,v1,v2,v3, 35);
    LOAD4(w0,w1,w2,w3, 36); LOAD4(x0,x1,x2,x3, 37);
    LOAD4(y0,y1,y2,y3, 38); LOAD4(z0,z1,z2,z3, 39);
    LOAD4(A0,A1,A2,A3, 40); LOAD4(B0,B1,B2,B3, 41);
    LOAD4(Cc0,Cc1,Cc2,Cc3,42); LOAD4(D0,D1,D2,D3,43);
    LOAD4(E0,E1,E2,E3, 44); LOAD4(F0,F1,F2,F3, 45);
    LOAD4(G0,G1,G2,G3, 46); LOAD4(H0,H1,H2,H3, 47);
    LOAD4(s0,s1,s2,s3, 48); LOAD4(t0,t1,t2,t3, 49);
    LOAD4(u0,u1,u2,u3, 50); LOAD4(v0,v1,v2,v3, 51);
    LOAD4(w0,w1,w2,w3, 52); LOAD4(x0,x1,x2,x3, 53);
    LOAD4(y0,y1,y2,y3, 54); LOAD4(z0,z1,z2,z3, 55);
    LOAD4(A0,A1,A2,A3, 56); LOAD4(B0,B1,B2,B3, 57);
    LOAD4(Cc0,Cc1,Cc2,Cc3,58); LOAD4(D0,D1,D2,D3,59);
    LOAD4(E0,E1,E2,E3, 60); LOAD4(F0,F1,F2,F3, 61);
    LOAD4(G0,G1,G2,G3, 62); LOAD4(H0,H1,H2,H3, 63);

    #undef LOAD4

    C[base+0] = __float2half(((s0+t0)+(u0+v0))+((w0+x0)+(y0+z0))+((A0+B0)+(Cc0+D0))+((E0+F0)+(G0+H0)));
    C[base+1] = __float2half(((s1+t1)+(u1+v1))+((w1+x1)+(y1+z1))+((A1+B1)+(Cc1+D1))+((E1+F1)+(G1+H1)));
    C[base+2] = __float2half(((s2+t2)+(u2+v2))+((w2+x2)+(y2+z2))+((A2+B2)+(Cc2+D2))+((E2+F2)+(G2+H2)));
    C[base+3] = __float2half(((s3+t3)+(u3+v3))+((w3+x3)+(y3+z3))+((A3+B3)+(Cc3+D3))+((E3+F3)+(G3+H3)));
}

__global__ void __launch_bounds__(NTHREADS, 1)
hgemm_fused_64(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ SplitBuf,
    __half*       __restrict__ C
) {
    cg::grid_group grid = cg::this_grid();

    const int split_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int warp_m   = warp_row * 32;
    const int warp_n   = warp_col * 64;

    float acc[WM][WN][4];
    #pragma unroll
    for (int mi = 0; mi < WM; mi++)
        #pragma unroll
        for (int ni = 0; ni < WN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] =
            acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    extern __shared__ __half smem[];
    const int k_start = split_id * KP;

    #pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        const int k_off = k_start + s * BK;
        __half* dA = smem + s * SSTSZ;
        __half* dB = smem + s * SSTSZ + ASZ;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx  = tid + i * NTHREADS;
            int row  = idx >> 3;
            int col8 = idx & 7;
            int phys = col8 ^ ((row >> 2) & 7);
            uint32_t dst = smem_u32addr(dA + row * ASTR + phys * 8);
            cp_async16_ca(dst, A + (size_t)row * K_DIM + k_off + col8 * 8);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx   = tid + i * NTHREADS;
            int k_row = idx >> 4;
            int col8  = idx & 15;
            int phys  = col8 ^ ((k_row >> 1) & 15);
            uint32_t dst = smem_u32addr(dB + k_row * BSTR + phys * 8);
            cp_async16_cg(dst, B + (size_t)(k_off + k_row) * N_DIM + col8 * 8);
        }
        cp_async_commit();
    }

    int cur = 0;
    #pragma unroll 1
    for (int kt = 0; kt < KT; kt++) {
        const int next_tile  = kt + ST - 1;
        const int next_stage = (cur + ST - 1) % ST;
        if (next_tile < KT) {
            const int k_off = k_start + next_tile * BK;
            __half* dA = smem + next_stage * SSTSZ;
            __half* dB = smem + next_stage * SSTSZ + ASZ;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx  = tid + i * NTHREADS;
                int row  = idx >> 3;
                int col8 = idx & 7;
                int phys = col8 ^ ((row >> 2) & 7);
                uint32_t dst = smem_u32addr(dA + row * ASTR + phys * 8);
                cp_async16_ca(dst, A + (size_t)row * K_DIM + k_off + col8 * 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx   = tid + i * NTHREADS;
                int k_row = idx >> 4;
                int col8  = idx & 15;
                int phys  = col8 ^ ((k_row >> 1) & 15);
                uint32_t dst = smem_u32addr(dB + k_row * BSTR + phys * 8);
                cp_async16_cg(dst, B + (size_t)(k_off + k_row) * N_DIM + col8 * 8);
            }
        }
        cp_async_commit();
        cp_async_wait<ST - 2>();
        __syncthreads();

        __half* cA = smem + cur * SSTSZ;
        __half* cB = smem + cur * SSTSZ + ASZ;

        uint32_t a_frag[WM][4];
        #pragma unroll
        for (int mi = 0; mi < WM; mi++) {
            const int smem_row  = warp_m + mi * 16 + (lane_id & 15);
            const int col8_idx  = (lane_id >> 4);
            const int phys_col8 = col8_idx ^ ((smem_row >> 2) & 7);
            uint32_t sptr = smem_u32addr(cA + smem_row * ASTR + phys_col8 * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                  "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                : "r"(sptr));
        }

        constexpr int KK_ITERS = BK / 16;
        #pragma unroll
        for (int kk = 0; kk < KK_ITERS; kk++) {
            const int k16 = kk * 16;
            uint32_t a_frag_next[WM][4];
            if (kk + 1 < KK_ITERS) {
                const int k16_next = (kk + 1) * 16;
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    const int smem_row  = warp_m + mi * 16 + (lane_id & 15);
                    const int col8_base = (k16_next >> 3) + (lane_id >> 4);
                    const int phys_col8 = col8_base ^ ((smem_row >> 2) & 7);
                    uint32_t sptr = smem_u32addr(cA + smem_row * ASTR + phys_col8 * 8);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(a_frag_next[mi][0]), "=r"(a_frag_next[mi][1]),
                          "=r"(a_frag_next[mi][2]), "=r"(a_frag_next[mi][3])
                        : "r"(sptr));
                }
            }
            uint32_t b_frags[WN][2];
            #pragma unroll
            for (int ni = 0; ni < WN; ni++) {
                const int n_base      = warp_n + ni * 8;
                const int b_k_row     = k16 + (lane_id & 15);
                const int b_col8      = n_base >> 3;
                const int phys_b_col8 = b_col8 ^ ((b_k_row >> 1) & 15);
                uint32_t b_sptr = smem_u32addr(cB + b_k_row * BSTR + phys_b_col8 * 8);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frags[ni][0]), "=r"(b_frags[ni][1])
                    : "r"(b_sptr));
            }
            #pragma unroll
            for (int ni = 0; ni < WN; ni++) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    mma_m16n8k16(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a_frag[mi][0], a_frag[mi][1],
                        a_frag[mi][2], a_frag[mi][3],
                        b_frags[ni][0], b_frags[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
                }
            }
            if (kk + 1 < KK_ITERS) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++)
                    #pragma unroll
                    for (int r = 0; r < 4; r++)
                        a_frag[mi][r] = a_frag_next[mi][r];
            }
        }
        cur = (cur + 1) % ST;
    }
    cp_async_wait<0>();

    float* my_buf = SplitBuf + (size_t)split_id * N_ELEMS;
    #pragma unroll
    for (int mi = 0; mi < WM; mi++) {
        const int row0 = warp_m + mi * 16 + (lane_id >> 2);
        const int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            const int col0 = warp_n + ni * 8 + (lane_id & 3) * 2;
            *reinterpret_cast<float2*>(&my_buf[row0 * N_DIM + col0]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&my_buf[row1 * N_DIM + col0]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }

    grid.sync();

    const int global_tid = blockIdx.x * NTHREADS + tid;
    if (global_tid < N_ELEMS) {
        const int elem_idx = global_tid;
        const float* bptr = SplitBuf + elem_idx;
        float s0=__ldg(bptr+(size_t)0*N_ELEMS),   s1=__ldg(bptr+(size_t)1*N_ELEMS);
        float s2=__ldg(bptr+(size_t)2*N_ELEMS),   s3=__ldg(bptr+(size_t)3*N_ELEMS);
        float s4=__ldg(bptr+(size_t)4*N_ELEMS),   s5=__ldg(bptr+(size_t)5*N_ELEMS);
        float s6=__ldg(bptr+(size_t)6*N_ELEMS),   s7=__ldg(bptr+(size_t)7*N_ELEMS);
        float s8=__ldg(bptr+(size_t)8*N_ELEMS),   s9=__ldg(bptr+(size_t)9*N_ELEMS);
        float s10=__ldg(bptr+(size_t)10*N_ELEMS), s11=__ldg(bptr+(size_t)11*N_ELEMS);
        float s12=__ldg(bptr+(size_t)12*N_ELEMS), s13=__ldg(bptr+(size_t)13*N_ELEMS);
        float s14=__ldg(bptr+(size_t)14*N_ELEMS), s15=__ldg(bptr+(size_t)15*N_ELEMS);
        float s16=__ldg(bptr+(size_t)16*N_ELEMS), s17=__ldg(bptr+(size_t)17*N_ELEMS);
        float s18=__ldg(bptr+(size_t)18*N_ELEMS), s19=__ldg(bptr+(size_t)19*N_ELEMS);
        float s20=__ldg(bptr+(size_t)20*N_ELEMS), s21=__ldg(bptr+(size_t)21*N_ELEMS);
        float s22=__ldg(bptr+(size_t)22*N_ELEMS), s23=__ldg(bptr+(size_t)23*N_ELEMS);
        float s24=__ldg(bptr+(size_t)24*N_ELEMS), s25=__ldg(bptr+(size_t)25*N_ELEMS);
        float s26=__ldg(bptr+(size_t)26*N_ELEMS), s27=__ldg(bptr+(size_t)27*N_ELEMS);
        float s28=__ldg(bptr+(size_t)28*N_ELEMS), s29=__ldg(bptr+(size_t)29*N_ELEMS);
        float s30=__ldg(bptr+(size_t)30*N_ELEMS), s31=__ldg(bptr+(size_t)31*N_ELEMS);
        float s32=__ldg(bptr+(size_t)32*N_ELEMS), s33=__ldg(bptr+(size_t)33*N_ELEMS);
        float s34=__ldg(bptr+(size_t)34*N_ELEMS), s35=__ldg(bptr+(size_t)35*N_ELEMS);
        float s36=__ldg(bptr+(size_t)36*N_ELEMS), s37=__ldg(bptr+(size_t)37*N_ELEMS);
        float s38=__ldg(bptr+(size_t)38*N_ELEMS), s39=__ldg(bptr+(size_t)39*N_ELEMS);
        float s40=__ldg(bptr+(size_t)40*N_ELEMS), s41=__ldg(bptr+(size_t)41*N_ELEMS);
        float s42=__ldg(bptr+(size_t)42*N_ELEMS), s43=__ldg(bptr+(size_t)43*N_ELEMS);
        float s44=__ldg(bptr+(size_t)44*N_ELEMS), s45=__ldg(bptr+(size_t)45*N_ELEMS);
        float s46=__ldg(bptr+(size_t)46*N_ELEMS), s47=__ldg(bptr+(size_t)47*N_ELEMS);
        float s48=__ldg(bptr+(size_t)48*N_ELEMS), s49=__ldg(bptr+(size_t)49*N_ELEMS);
        float s50=__ldg(bptr+(size_t)50*N_ELEMS), s51=__ldg(bptr+(size_t)51*N_ELEMS);
        float s52=__ldg(bptr+(size_t)52*N_ELEMS), s53=__ldg(bptr+(size_t)53*N_ELEMS);
        float s54=__ldg(bptr+(size_t)54*N_ELEMS), s55=__ldg(bptr+(size_t)55*N_ELEMS);
        float s56=__ldg(bptr+(size_t)56*N_ELEMS), s57=__ldg(bptr+(size_t)57*N_ELEMS);
        float s58=__ldg(bptr+(size_t)58*N_ELEMS), s59=__ldg(bptr+(size_t)59*N_ELEMS);
        float s60=__ldg(bptr+(size_t)60*N_ELEMS), s61=__ldg(bptr+(size_t)61*N_ELEMS);
        float s62=__ldg(bptr+(size_t)62*N_ELEMS), s63=__ldg(bptr+(size_t)63*N_ELEMS);
        float a0=s0+s1,a1=s2+s3,a2=s4+s5,a3=s6+s7;
        float a4=s8+s9,a5=s10+s11,a6=s12+s13,a7=s14+s15;
        float a8=s16+s17,a9=s18+s19,a10=s20+s21,a11=s22+s23;
        float a12=s24+s25,a13=s26+s27,a14=s28+s29,a15=s30+s31;
        float a16=s32+s33,a17=s34+s35,a18=s36+s37,a19=s38+s39;
        float a20=s40+s41,a21=s42+s43,a22=s44+s45,a23=s46+s47;
        float a24=s48+s49,a25=s50+s51,a26=s52+s53,a27=s54+s55;
        float a28=s56+s57,a29=s58+s59,a30=s60+s61,a31=s62+s63;
        float b0=a0+a1,b1=a2+a3,b2=a4+a5,b3=a6+a7;
        float b4=a8+a9,b5=a10+a11,b6=a12+a13,b7=a14+a15;
        float b8=a16+a17,b9=a18+a19,b10=a20+a21,b11=a22+a23;
        float b12=a24+a25,b13=a26+a27,b14=a28+a29,b15=a30+a31;
        float c0=b0+b1,c1=b2+b3,c2=b4+b5,c3=b6+b7;
        float c4=b8+b9,c5=b10+b11,c6=b12+b13,c7=b14+b15;
        float d0=c0+c1,d1=c2+c3,d2=c4+c5,d3=c6+c7;
        float e0=d0+d1,e1=d2+d3;
        C[elem_idx] = __float2half(e0+e1);
    }
}

static float* d_buf64  = nullptr;
static float* d_buf128 = nullptr;
static bool   res_init  = false;
static int    best_var  = -1;
static bool   coop_ok  = false;

static void ensure_resources() {
    if (res_init) return;
    cudaMalloc(&d_buf64,  (size_t)NS  * N_ELEMS * sizeof(float));
    cudaMalloc(&d_buf128, (size_t)NS2 * N_ELEMS * sizeof(float));

    cudaFuncSetAttribute(hgemm_pass1_64,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(hgemm_pass1_128,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2);
    cudaFuncSetAttribute(hgemm_fused_64,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    if (prop.cooperativeLaunch) {
        int occ = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, hgemm_fused_64, NTHREADS, SMEM);
        int coop_max = occ * prop.multiProcessorCount;
        if (coop_max >= NS) coop_ok = true;
    }
    res_init = true;
}

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor b_col_major, torch::Tensor c)
{
    const __half* A_ptr = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B_ptr = reinterpret_cast<const __half*>(b.data_ptr());
    __half*       C_ptr = reinterpret_cast<__half*>(c.data_ptr());

    ensure_resources();

    const int RED_F2_BLOCKS  = N_ELEMS / 2 / 256;
    const int RED_F4_BLOCKS  = N_ELEMS / 1024;
    const int RED_4X_BLOCKS  = N_ELEMS / 4 / 256;

    if (best_var == -1) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        float t[5] = {1e9f, 1e9f, 1e9f, 1e9f, 1e9f};

        for (int w = 0; w < 5; w++) {
            hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
            hgemm_reduce_f2_64<<<RED_F2_BLOCKS, 256>>>(d_buf64, C_ptr);
            hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
            hgemm_reduce_f4_64<<<RED_F4_BLOCKS, 1024>>>(d_buf64, C_ptr);
            hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
            hgemm_reduce_4elem_64<<<RED_4X_BLOCKS, 256>>>(d_buf64, C_ptr);
            hgemm_pass1_128<<<NS2, NTHREADS, SMEM2>>>(A_ptr, B_ptr, d_buf128);
            hgemm_reduce_f2_128<<<RED_F2_BLOCKS, 256>>>(d_buf128, C_ptr);
        }
        if (coop_ok) {
            void* args[] = {(void*)&A_ptr,(void*)&B_ptr,(void*)&d_buf64,(void*)&C_ptr};
            for (int w = 0; w < 5; w++)
                cudaLaunchCooperativeKernel((void*)hgemm_fused_64,
                    dim3(NS), dim3(NTHREADS), args, SMEM, nullptr);
        }
        cudaDeviceSynchronize();

        constexpr int BENCH = 80;

        if (coop_ok) {
            void* args[] = {(void*)&A_ptr,(void*)&B_ptr,(void*)&d_buf64,(void*)&C_ptr};
            cudaEventRecord(ev0);
            for (int i = 0; i < BENCH; i++)
                cudaLaunchCooperativeKernel((void*)hgemm_fused_64,
                    dim3(NS), dim3(NTHREADS), args, SMEM, nullptr);
            cudaEventRecord(ev1); cudaEventSynchronize(ev1);
            cudaEventElapsedTime(&t[0], ev0, ev1);
        }

        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) {
            hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
            hgemm_reduce_f2_64<<<RED_F2_BLOCKS, 256>>>(d_buf64, C_ptr);
        }
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[1], ev0, ev1);

        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) {
            hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
            hgemm_reduce_f4_64<<<RED_F4_BLOCKS, 1024>>>(d_buf64, C_ptr);
        }
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[2], ev0, ev1);

        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) {
            hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
            hgemm_reduce_4elem_64<<<RED_4X_BLOCKS, 256>>>(d_buf64, C_ptr);
        }
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[3], ev0, ev1);

        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) {
            hgemm_pass1_128<<<NS2, NTHREADS, SMEM2>>>(A_ptr, B_ptr, d_buf128);
            hgemm_reduce_f2_128<<<RED_F2_BLOCKS, 256>>>(d_buf128, C_ptr);
        }
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[4], ev0, ev1);

        best_var = 1;
        for (int i = 0; i < 5; i++)
            if (t[i] < t[best_var]) best_var = i;
        if (!coop_ok && best_var == 0) best_var = 1;

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }

    const int rf2 = N_ELEMS / 2 / 256;
    const int rf4 = N_ELEMS / 1024;
    const int r4x = N_ELEMS / 4 / 256;

    if (best_var == 0 && coop_ok) {
        void* args[] = {(void*)&A_ptr,(void*)&B_ptr,(void*)&d_buf64,(void*)&C_ptr};
        cudaLaunchCooperativeKernel((void*)hgemm_fused_64,
            dim3(NS), dim3(NTHREADS), args, SMEM, nullptr);
    } else if (best_var == 1) {
        hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
        hgemm_reduce_f2_64<<<rf2, 256>>>(d_buf64, C_ptr);
    } else if (best_var == 2) {
        hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
        hgemm_reduce_f4_64<<<rf4, 1024>>>(d_buf64, C_ptr);
    } else if (best_var == 3) {
        hgemm_pass1_64<<<NS, NTHREADS, SMEM>>>(A_ptr, B_ptr, d_buf64);
        hgemm_reduce_4elem_64<<<r4x, 256>>>(d_buf64, C_ptr);
    } else {
        hgemm_pass1_128<<<NS2, NTHREADS, SMEM2>>>(A_ptr, B_ptr, d_buf128);
        hgemm_reduce_f2_128<<<rf2, 256>>>(d_buf128, C_ptr);
    }
}