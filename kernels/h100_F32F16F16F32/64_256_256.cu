#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda::wmma;

static constexpr int A_BM   = 64;
static constexpr int A_BN   = 32;
static constexpr int A_BK   = 32;
static constexpr int A_AST  = 40;
static constexpr int A_BST  = 40;
static constexpr int A_STAGES = 4;

__global__ __launch_bounds__(256, 6)
void hgemm_8cta_4stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int bx      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int wm      = warp_id >> 1;
    const int wn      = warp_id & 1;

    const int block_col = bx * A_BN;

    __shared__ half smA[A_STAGES][A_BM * A_AST];
    __shared__ half smB[A_STAGES][A_BK * A_BST];

    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.f);

    const int a_r = tid >> 2;
    const int a_c = (tid & 3) * 8;

    const int b_r = tid >> 3;
    const int b_c = (tid & 7) * 4;

    const half* A_base = A + a_r * 256 + a_c;
    const half* B_base = B + b_r * 256 + block_col + b_c;

    #define PREF_A(kt, buf) do { \
        uint32_t _a = __cvta_generic_to_shared(&smA[(buf)][a_r * A_AST + a_c]); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                     :: "r"(_a), "l"(A_base + (kt) * A_BK) : "memory"); \
    } while(0)

    #define PREF_B(kt, buf) do { \
        uint32_t _b = __cvta_generic_to_shared(&smB[(buf)][b_r * A_BST + b_c]); \
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" \
                     :: "r"(_b), "l"(B_base + (kt) * (long)A_BK * 256) : "memory"); \
    } while(0)

    PREF_A(0, 0); PREF_B(0, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    PREF_A(1, 1); PREF_B(1, 1);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    PREF_A(2, 2); PREF_B(2, 2);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    #pragma unroll 8
    for (int kt = 0; kt < 8; kt++) {
        const int cur = kt % A_STAGES;

        if (kt + 3 < 8) {
            PREF_A(kt + 3, (kt + 3) % A_STAGES);
            PREF_B(kt + 3, (kt + 3) % A_STAGES);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads();

        const half* sa = smA[cur] + wm * 16 * A_AST;
        const half* sb = smB[cur] + wn * 16;

        fragment<matrix_a, 16, 16, 16, half, row_major> fa0, fa1;
        fragment<matrix_b, 16, 16, 16, half, row_major> fb0, fb1;

        load_matrix_sync(fa0, sa,                A_AST);
        load_matrix_sync(fa1, sa + 16,           A_AST);
        load_matrix_sync(fb0, sb,                A_BST);
        load_matrix_sync(fb1, sb + 16 * A_BST,  A_BST);

        mma_sync(acc, fa0, fb0, acc);
        mma_sync(acc, fa1, fb1, acc);
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    #undef PREF_A
    #undef PREF_B

    const int base_row = wm * 16;
    const int base_col = block_col + wn * 16;
    const int r0    = base_row + (lane >> 2);
    const int r1    = r0 + 8;
    const int c_off = (lane & 3) * 2;

    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + c_off])     = __floats2half2_rn(acc.x[0], acc.x[1]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + c_off])     = __floats2half2_rn(acc.x[2], acc.x[3]);
    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + c_off + 8]) = __floats2half2_rn(acc.x[4], acc.x[5]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + c_off + 8]) = __floats2half2_rn(acc.x[6], acc.x[7]);
}

static constexpr int B_BM   = 64;
static constexpr int B_BN   = 64;
static constexpr int B_BK   = 32;
static constexpr int B_AST  = 40;
static constexpr int B_BST  = 72;
static constexpr int B_STAGES = 4;

__global__ __launch_bounds__(256, 4)
void hgemm_4cta_4stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int bx      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int wm      = warp_id >> 1;
    const int wn      = warp_id & 1;

    const int block_col = bx * B_BN;

    __shared__ half smA[B_STAGES][B_BM * B_AST];
    __shared__ half smB[B_STAGES][B_BK * B_BST];

    fragment<accumulator, 16, 16, 16, float> acc0, acc1;
    fill_fragment(acc0, 0.f);
    fill_fragment(acc1, 0.f);

    const int a_r = tid >> 2;
    const int a_c = (tid & 3) * 8;

    const int b_r = tid >> 3;
    const int b_c = (tid & 7) * 8;

    const half* A_base = A + a_r * 256 + a_c;
    const half* B_base = B + b_r * 256 + block_col + b_c;

    #define PREF2_A(kt, buf) do { \
        uint32_t _a = __cvta_generic_to_shared(&smA[(buf)][a_r * B_AST + a_c]); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                     :: "r"(_a), "l"(A_base + (kt) * B_BK) : "memory"); \
    } while(0)

    #define PREF2_B(kt, buf) do { \
        uint32_t _b = __cvta_generic_to_shared(&smB[(buf)][b_r * B_BST + b_c]); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                     :: "r"(_b), "l"(B_base + (kt) * (long)B_BK * 256) : "memory"); \
    } while(0)

    PREF2_A(0, 0); PREF2_B(0, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    PREF2_A(1, 1); PREF2_B(1, 1);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    PREF2_A(2, 2); PREF2_B(2, 2);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    #pragma unroll 8
    for (int kt = 0; kt < 8; kt++) {
        const int cur = kt % B_STAGES;

        if (kt + 3 < 8) {
            PREF2_A(kt + 3, (kt + 3) % B_STAGES);
            PREF2_B(kt + 3, (kt + 3) % B_STAGES);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads();

        const half* sa = smA[cur] + wm * 16 * B_AST;
        const half* sb = smB[cur] + wn * 32;

        fragment<matrix_a, 16, 16, 16, half, row_major> fa0, fa1;
        fragment<matrix_b, 16, 16, 16, half, row_major> fb00, fb01, fb10, fb11;

        load_matrix_sync(fa0,  sa,                   B_AST);
        load_matrix_sync(fa1,  sa + 16,              B_AST);
        load_matrix_sync(fb00, sb,                   B_BST);
        load_matrix_sync(fb01, sb + 16,              B_BST);
        load_matrix_sync(fb10, sb + 16 * B_BST,      B_BST);
        load_matrix_sync(fb11, sb + 16 * B_BST + 16, B_BST);

        mma_sync(acc0, fa0, fb00, acc0);
        mma_sync(acc1, fa0, fb01, acc1);
        mma_sync(acc0, fa1, fb10, acc0);
        mma_sync(acc1, fa1, fb11, acc1);
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    #undef PREF2_A
    #undef PREF2_B

    const int base_row = wm * 16;
    const int base_col = block_col + wn * 32;
    const int r0    = base_row + (lane >> 2);
    const int r1    = r0 + 8;
    const int c_off = (lane & 3) * 2;

    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + c_off])          = __floats2half2_rn(acc0.x[0], acc0.x[1]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + c_off])          = __floats2half2_rn(acc0.x[2], acc0.x[3]);
    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + c_off + 8])      = __floats2half2_rn(acc0.x[4], acc0.x[5]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + c_off + 8])      = __floats2half2_rn(acc0.x[6], acc0.x[7]);
    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + 16 + c_off])     = __floats2half2_rn(acc1.x[0], acc1.x[1]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + 16 + c_off])     = __floats2half2_rn(acc1.x[2], acc1.x[3]);
    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + 16 + c_off + 8]) = __floats2half2_rn(acc1.x[4], acc1.x[5]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + 16 + c_off + 8]) = __floats2half2_rn(acc1.x[6], acc1.x[7]);
}

static constexpr int C_BM   = 32;
static constexpr int C_BN   = 32;
static constexpr int C_BK   = 32;
static constexpr int C_AST  = 40;
static constexpr int C_BST  = 40;
static constexpr int C_STAGES = 4;

__global__ __launch_bounds__(128, 8)
void hgemm_16cta_4stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int bx      = blockIdx.x;
    const int by      = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int wm      = warp_id >> 1;
    const int wn      = warp_id & 1;

    const int block_row = by * C_BM;
    const int block_col = bx * C_BN;

    __shared__ half smA[C_STAGES][C_BM * C_AST];
    __shared__ half smB[C_STAGES][C_BK * C_BST];

    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.f);

    const int a_r = tid >> 2;
    const int a_c = (tid & 3) * 8;

    const int b_r = tid >> 2;
    const int b_c = (tid & 3) * 8;

    const half* A_base = A + (block_row + a_r) * 256 + a_c;
    const half* B_base = B + b_r * 256 + block_col + b_c;

    #define PREF3_A(kt, buf) do { \
        uint32_t _a = __cvta_generic_to_shared(&smA[(buf)][a_r * C_AST + a_c]); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                     :: "r"(_a), "l"(A_base + (kt) * C_BK) : "memory"); \
    } while(0)

    #define PREF3_B(kt, buf) do { \
        uint32_t _b = __cvta_generic_to_shared(&smB[(buf)][b_r * C_BST + b_c]); \
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
                     :: "r"(_b), "l"(B_base + (kt) * (long)C_BK * 256) : "memory"); \
    } while(0)

    PREF3_A(0, 0); PREF3_B(0, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    PREF3_A(1, 1); PREF3_B(1, 1);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    PREF3_A(2, 2); PREF3_B(2, 2);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    #pragma unroll 8
    for (int kt = 0; kt < 8; kt++) {
        const int cur = kt % C_STAGES;

        if (kt + 3 < 8) {
            PREF3_A(kt + 3, (kt + 3) % C_STAGES);
            PREF3_B(kt + 3, (kt + 3) % C_STAGES);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads();

        const half* sa = smA[cur] + wm * 16 * C_AST;
        const half* sb = smB[cur] + wn * 16;

        fragment<matrix_a, 16, 16, 16, half, row_major> fa0, fa1;
        fragment<matrix_b, 16, 16, 16, half, row_major> fb0, fb1;

        load_matrix_sync(fa0, sa,              C_AST);
        load_matrix_sync(fa1, sa + 16,         C_AST);
        load_matrix_sync(fb0, sb,              C_BST);
        load_matrix_sync(fb1, sb + 16*C_BST,   C_BST);

        mma_sync(acc, fa0, fb0, acc);
        mma_sync(acc, fa1, fb1, acc);
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    #undef PREF3_A
    #undef PREF3_B

    const int base_row = block_row + wm * 16;
    const int base_col = block_col + wn * 16;
    const int r0    = base_row + (lane >> 2);
    const int r1    = r0 + 8;
    const int c_off = (lane & 3) * 2;

    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + c_off])     = __floats2half2_rn(acc.x[0], acc.x[1]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + c_off])     = __floats2half2_rn(acc.x[2], acc.x[3]);
    *reinterpret_cast<half2*>(&C[r0 * 256 + base_col + c_off + 8]) = __floats2half2_rn(acc.x[4], acc.x[5]);
    *reinterpret_cast<half2*>(&C[r1 * 256 + base_col + c_off + 8]) = __floats2half2_rn(acc.x[6], acc.x[7]);
}

static constexpr int FB_BM   = 64;
static constexpr int FB_BN   = 64;
static constexpr int FB_BK   = 32;
static constexpr int FB_AST  = 40;
static constexpr int FB_BST  = 72;

__global__ __launch_bounds__(256, 3)
void hgemm_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5, lane = tid & 31;
    const int wm = warp_id >> 1, wn = warp_id & 1;

    const int block_row = by * FB_BM, block_col = bx * FB_BN;

    __shared__ half smA[3][FB_BM * FB_AST];
    __shared__ half smB[3][FB_BK * FB_BST];

    fragment<accumulator, 16, 16, 16, float> acc0, acc1;
    fill_fragment(acc0, 0.f); fill_fragment(acc1, 0.f);

    const int nk = (K + FB_BK - 1) / FB_BK;
    const int ar = tid >> 2, ac = (tid & 3) * 8;
    const int br = tid >> 3, bc = (tid & 7) * 8;

    auto pf = [&](int kt, int buf) __attribute__((always_inline)) {
        int gr_a = block_row + ar, gc_a = kt * FB_BK + ac;
        if (gr_a < M && gc_a + 8 <= K) {
            uint32_t addr = __cvta_generic_to_shared(&smA[buf][ar * FB_AST + ac]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addr), "l"(&A[gr_a * K + gc_a]) : "memory");
        } else {
            for (int i = 0; i < 8; i++)
                smA[buf][ar * FB_AST + ac + i] = (gr_a < M && gc_a+i < K) ? A[gr_a*K+gc_a+i] : __float2half(0.f);
        }
        int gr_b = kt * FB_BK + br, gc_b = block_col + bc;
        if (gr_b < K && gc_b + 8 <= N) {
            uint32_t addr = __cvta_generic_to_shared(&smB[buf][br * FB_BST + bc]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addr), "l"(&B[gr_b * N + gc_b]) : "memory");
        } else {
            for (int i = 0; i < 8; i++)
                smB[buf][br * FB_BST + bc + i] = (gr_b < K && gc_b+i < N) ? B[gr_b*N+gc_b+i] : __float2half(0.f);
        }
    };

    pf(0, 0); asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (nk > 1) pf(1, 1);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    for (int kt = 0; kt < nk; kt++) {
        int cur = kt % 3;
        if (kt + 2 < nk) pf(kt + 2, (kt + 2) % 3);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();

        const half* sa = smA[cur] + wm * 16 * FB_AST;
        const half* sb = smB[cur] + wn * 32;

        fragment<matrix_a,16,16,16,half,row_major> fa0,fa1;
        fragment<matrix_b,16,16,16,half,row_major> fb00,fb01,fb10,fb11;
        load_matrix_sync(fa0,  sa,                   FB_AST);
        load_matrix_sync(fa1,  sa+16,                FB_AST);
        load_matrix_sync(fb00, sb,                   FB_BST);
        load_matrix_sync(fb01, sb+16,                FB_BST);
        load_matrix_sync(fb10, sb+16*FB_BST,         FB_BST);
        load_matrix_sync(fb11, sb+16*FB_BST+16,      FB_BST);
        mma_sync(acc0, fa0, fb00, acc0); mma_sync(acc1, fa0, fb01, acc1);
        mma_sync(acc0, fa1, fb10, acc0); mma_sync(acc1, fa1, fb11, acc1);
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int base_row = block_row + wm * 16, base_col = block_col + wn * 32;
    const int r0 = base_row + (lane >> 2), r1 = r0 + 8;
    const int c_off = (lane & 3) * 2;

#define WR2G(v0,v1,r,c) \
    if((r)<M&&(c)+1<=N)*reinterpret_cast<half2*>(&C[(r)*N+(c)])=__floats2half2_rn(v0,v1); \
    else if((r)<M&&(c)<N)C[(r)*N+(c)]=__float2half(v0);
    WR2G(acc0.x[0],acc0.x[1],r0,base_col+c_off)
    WR2G(acc0.x[2],acc0.x[3],r1,base_col+c_off)
    WR2G(acc0.x[4],acc0.x[5],r0,base_col+c_off+8)
    WR2G(acc0.x[6],acc0.x[7],r1,base_col+c_off+8)
    WR2G(acc1.x[0],acc1.x[1],r0,base_col+16+c_off)
    WR2G(acc1.x[2],acc1.x[3],r1,base_col+16+c_off)
    WR2G(acc1.x[4],acc1.x[5],r0,base_col+16+c_off+8)
    WR2G(acc1.x[6],acc1.x[7],r1,base_col+16+c_off+8)
#undef WR2G
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (M == 64 && N == 256 && K == 256) {
        hgemm_8cta_4stage<<<dim3(8, 1), dim3(256)>>>(A, B, C);
    } else {
        dim3 grid((N + FB_BN - 1) / FB_BN, (M + FB_BM - 1) / FB_BM);
        hgemm_fallback<<<grid, dim3(256)>>>(A, B, C, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}