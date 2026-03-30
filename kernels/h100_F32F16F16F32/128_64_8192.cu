#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>

using namespace nvcuda;

#define M_DIM   128
#define N_DIM   64
#define K_TOTAL 8192
#define MN_SIZE (M_DIM * N_DIM)

#define SPLIT_K   128
#define K_CHUNK   (K_TOTAL / SPLIT_K)
#define N_STAGES  3
#define NUM_ITERS (K_CHUNK / 16)

#define SMEM_A_STRIDE 16
#define SMEM_B_STRIDE 16

__global__ __launch_bounds__(256, 4)
void hgemm_mma_sk128(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_partial
) {
    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_CHUNK;
    const int warp_id  = (threadIdx.x >> 5);
    const int lane_id  = (threadIdx.x & 31);
    const int tid      = threadIdx.x;
    const int m_base   = warp_id * 16;

    __shared__ __align__(128) half smem_A[N_STAGES][128 * 16];
    __shared__ __align__(128) half smem_B[N_STAGES][64 * 16];

    float acc[8][4];
    #pragma unroll
    for (int t = 0; t < 8; t++)
        for (int r = 0; r < 4; r++)
            acc[t][r] = 0.f;

    auto async_load_A = [&](int stage, int k_off) {
        int flat = tid * 8;
        int row  = flat >> 4;
        int col  = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_A[stage][row * SMEM_A_STRIDE + col]);
        const void* src = A + (size_t)row * K_TOTAL + k_off + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"(src) : "memory");
    };

    auto async_load_B = [&](int stage, int k_off) {
        int flat    = tid * 4;
        int n_col   = flat >> 4;
        int k_inner = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[stage][n_col * SMEM_B_STRIDE + k_inner]);
        const void* src = B_col + (size_t)n_col * K_TOTAL + k_off + k_inner;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(dst), "l"(src) : "memory");
    };

    int fill_iters = min(N_STAGES - 1, NUM_ITERS);
    for (int s = 0; s < fill_iters; s++) {
        async_load_A(s, k_start + s * 16);
        async_load_B(s, k_start + s * 16);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        int fetch_iter = iter + N_STAGES - 1;
        if (fetch_iter < NUM_ITERS) {
            int stage = fetch_iter % N_STAGES;
            async_load_A(stage, k_start + fetch_iter * 16);
            async_load_B(stage, k_start + fetch_iter * 16);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES - 2) : "memory");
        __syncthreads();

        int cur_stage = iter % N_STAGES;

        uint32_t a_frag[4];
        {
            uint32_t smem_ptr = __cvta_generic_to_shared(smem_A[cur_stage] + m_base * 16 + (lane_id % 16) * 16);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(smem_ptr)
            );
        }

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            uint32_t b_frag[2];
            {
                uint32_t smem_ptr_b = __cvta_generic_to_shared(
                    smem_B[cur_stage] + (nt * 8 + (lane_id % 8)) * 16
                );
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1])
                    : "r"(smem_ptr_b)
                );
            }

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, "
                "{%4,%5,%6,%7}, "
                "{%8,%9}, "
                "{%0,%1,%2,%3};\n"
                : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1])
            );
        }
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    float* out = C_partial + (size_t)split_id * MN_SIZE;

    int row0 = m_base + (lane_id >> 2);
    int row1 = row0 + 8;
    int col_base = (lane_id & 3) * 2;

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int n_off = nt * 8 + col_base;
        if (row0 < M_DIM && n_off < N_DIM) {
            out[row0 * N_DIM + n_off]     = acc[nt][0];
            out[row0 * N_DIM + n_off + 1] = acc[nt][1];
        }
        if (row1 < M_DIM && n_off < N_DIM) {
            out[row1 * N_DIM + n_off]     = acc[nt][2];
            out[row1 * N_DIM + n_off + 1] = acc[nt][3];
        }
    }
}

__global__ __launch_bounds__(256, 8)
void reduce_sk128_fast(
    const float* __restrict__ C_partial,
    half* __restrict__ C,
    int MN
) {
    int elem = blockIdx.x * 32 + threadIdx.x;
    if (elem >= MN) return;

    const float* col = C_partial + elem;
    int ty = threadIdx.y;
    const int CHUNK = SPLIT_K / 8;

    float sum = 0.f;
    int s_base = ty * CHUNK;
    #pragma unroll 4
    for (int j = 0; j < CHUNK; j += 4) {
        sum += col[(size_t)(s_base+j+0) * MN]
             + col[(size_t)(s_base+j+1) * MN]
             + col[(size_t)(s_base+j+2) * MN]
             + col[(size_t)(s_base+j+3) * MN];
    }

    __shared__ float smem[8][32];
    smem[ty][threadIdx.x] = sum;
    __syncthreads();

    if (ty == 0) {
        float total = smem[0][threadIdx.x] + smem[1][threadIdx.x]
                    + smem[2][threadIdx.x] + smem[3][threadIdx.x]
                    + smem[4][threadIdx.x] + smem[5][threadIdx.x]
                    + smem[6][threadIdx.x] + smem[7][threadIdx.x];
        C[elem] = __float2half(total);
    }
}

__global__ __launch_bounds__(256, 4)
void hgemm_wmma_sk128(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_partial
) {
    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_CHUNK;
    const int warp_id  = threadIdx.x >> 5;
    const int tid      = threadIdx.x;
    const int m_base   = warp_id * 16;

    __shared__ __align__(128) half smem_A[2][128 * 16];
    __shared__ __align__(128) half smem_B[2][64 * 16];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[t], 0.0f);

    auto async_load_A = [&](int buf, int k_off) {
        int flat = tid * 8;
        int row  = flat >> 4;
        int col  = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_A[buf][row * 16 + col]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"((const void*)(A + (size_t)row * K_TOTAL + k_off + col)) : "memory");
    };

    auto async_load_B = [&](int buf, int k_off) {
        int flat    = tid * 4;
        int n_col   = flat >> 4;
        int k_inner = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[buf][n_col * 16 + k_inner]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(dst), "l"((const void*)(B_col + (size_t)n_col * K_TOTAL + k_off + k_inner)) : "memory");
    };

    async_load_A(0, k_start);
    async_load_B(0, k_start);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    async_load_A(1, k_start + 16);
    async_load_B(1, k_start + 16);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A[0] + m_base * 16, 16);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, smem_B[0] + nt * 256, 16);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    async_load_A(0, k_start + 32);
    async_load_B(0, k_start + 32);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A[1] + m_base * 16, 16);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, smem_B[1] + nt * 256, 16);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    async_load_A(1, k_start + 48);
    async_load_B(1, k_start + 48);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A[0] + m_base * 16, 16);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, smem_B[0] + nt * 256, 16);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A[1] + m_base * 16, 16);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, smem_B[1] + nt * 256, 16);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }

    float* out = C_partial + (size_t)split_id * MN_SIZE;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        wmma::store_matrix_sync(out + m_base * N_DIM + nt * 16, acc[nt],
                                N_DIM, wmma::mem_row_major);
    }
}

#define SPLIT_K2  256
#define K_CHUNK2  (K_TOTAL / SPLIT_K2)
#define NUM_ITERS2 (K_CHUNK2 / 16)

__global__ __launch_bounds__(256, 4)
void hgemm_wmma_sk256(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_partial
) {
    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_CHUNK2;
    const int warp_id  = threadIdx.x >> 5;
    const int tid      = threadIdx.x;
    const int m_base   = warp_id * 16;

    __shared__ __align__(128) half smem_A[2][128 * 16];
    __shared__ __align__(128) half smem_B[2][64 * 16];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int t = 0; t < 4; t++) wmma::fill_fragment(acc[t], 0.0f);

    auto async_load_A = [&](int buf, int k_off) {
        int flat = tid * 8;
        int row  = flat >> 4;
        int col  = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_A[buf][row * 16 + col]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"((const void*)(A + (size_t)row * K_TOTAL + k_off + col)) : "memory");
    };

    auto async_load_B = [&](int buf, int k_off) {
        int flat    = tid * 4;
        int n_col   = flat >> 4;
        int k_inner = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[buf][n_col * 16 + k_inner]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(dst), "l"((const void*)(B_col + (size_t)n_col * K_TOTAL + k_off + k_inner)) : "memory");
    };

    async_load_A(0, k_start);
    async_load_B(0, k_start);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    async_load_A(1, k_start + 16);
    async_load_B(1, k_start + 16);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A[0] + m_base * 16, 16);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, smem_B[0] + nt * 256, 16);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A[1] + m_base * 16, 16);
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, smem_B[1] + nt * 256, 16);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }

    float* out = C_partial + (size_t)split_id * MN_SIZE;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        wmma::store_matrix_sync(out + m_base * N_DIM + nt * 16, acc[nt],
                                N_DIM, wmma::mem_row_major);
    }
}

__global__ __launch_bounds__(256, 8)
void reduce_sk256_fast(
    const float* __restrict__ C_partial,
    half* __restrict__ C,
    int MN
) {
    int elem = blockIdx.x * 32 + threadIdx.x;
    if (elem >= MN) return;

    const float* col = C_partial + elem;
    int ty = threadIdx.y;
    const int CHUNK = SPLIT_K2 / 8;

    float sum = 0.f;
    int s_base = ty * CHUNK;
    #pragma unroll 8
    for (int j = 0; j < CHUNK; j += 4) {
        sum += col[(size_t)(s_base+j+0) * MN]
             + col[(size_t)(s_base+j+1) * MN]
             + col[(size_t)(s_base+j+2) * MN]
             + col[(size_t)(s_base+j+3) * MN];
    }

    __shared__ float smem[8][32];
    smem[ty][threadIdx.x] = sum;
    __syncthreads();

    if (ty == 0) {
        float total = smem[0][threadIdx.x] + smem[1][threadIdx.x]
                    + smem[2][threadIdx.x] + smem[3][threadIdx.x]
                    + smem[4][threadIdx.x] + smem[5][threadIdx.x]
                    + smem[6][threadIdx.x] + smem[7][threadIdx.x];
        C[elem] = __float2half(total);
    }
}

__global__ __launch_bounds__(256, 4)
void hgemm_ptx_mma_sk128(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ C_partial
) {
    const int split_id = blockIdx.x;
    const int k_start  = split_id * K_CHUNK;
    const int warp_id  = threadIdx.x >> 5;
    const int lane_id  = threadIdx.x & 31;
    const int tid      = threadIdx.x;
    const int m_base   = warp_id * 16;

    __shared__ __align__(128) half smem_A[2][128 * 16];
    __shared__ __align__(128) half smem_B[2][64 * 16];

    float acc[8][4];
    #pragma unroll
    for (int t = 0; t < 8; t++)
        for (int r = 0; r < 4; r++)
            acc[t][r] = 0.f;

    auto async_load_A = [&](int buf, int k_off) {
        int flat = tid * 8;
        int row  = flat >> 4;
        int col  = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_A[buf][row * 16 + col]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(dst), "l"((const void*)(A + (size_t)row * K_TOTAL + k_off + col)) : "memory");
    };

    auto async_load_B = [&](int buf, int k_off) {
        int flat    = tid * 4;
        int n_col   = flat >> 4;
        int k_inner = flat & 15;
        uint32_t dst = __cvta_generic_to_shared(&smem_B[buf][n_col * 16 + k_inner]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(dst), "l"((const void*)(B_col + (size_t)n_col * K_TOTAL + k_off + k_inner)) : "memory");
    };

    async_load_A(0, k_start);
    async_load_B(0, k_start);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        int cur = iter & 1;
        int nxt = 1 - cur;
        bool has_nxt = (iter + 1 < NUM_ITERS);

        if (has_nxt) {
            async_load_A(nxt, k_start + (iter+1)*16);
            async_load_B(nxt, k_start + (iter+1)*16);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        uint32_t a_reg[4];
        {
            int r = m_base + (lane_id & 15);
            int c = (lane_id >= 16) ? 8 : 0;
            uint32_t smem_ptr = __cvta_generic_to_shared(&smem_A[cur][r * 16 + c]);

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
                : "r"(smem_ptr)
            );
        }

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            uint32_t b_reg[2];
            {
                int b_row = lane_id & 7;
                int b_col = ((lane_id >> 3) & 1) * 8;
                uint32_t smem_ptr_b = __cvta_generic_to_shared(
                    &smem_B[cur][(nt * 8 + b_row) * 16 + b_col]
                );

                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_reg[0]), "=r"(b_reg[1])
                    : "r"(smem_ptr_b)
                );
            }

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, "
                "{%4,%5,%6,%7}, "
                "{%8,%9}, "
                "{%0,%1,%2,%3};\n"
                : "+f"(acc[nt][0]), "+f"(acc[nt][1]),
                  "+f"(acc[nt][2]), "+f"(acc[nt][3])
                : "r"(a_reg[0]), "r"(a_reg[1]),
                  "r"(a_reg[2]), "r"(a_reg[3]),
                  "r"(b_reg[0]), "r"(b_reg[1])
            );
        }

        if (has_nxt) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
        }
    }

    float* out = C_partial + (size_t)split_id * MN_SIZE;

    int row0 = m_base + (lane_id >> 2);
    int row1 = row0 + 8;

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int col0 = nt * 8 + (lane_id & 3) * 2;
        int col1 = col0 + 1;

        out[row0 * N_DIM + col0] = acc[nt][0];
        out[row0 * N_DIM + col1] = acc[nt][1];
        out[row1 * N_DIM + col0] = acc[nt][2];
        out[row1 * N_DIM + col1] = acc[nt][3];
    }
}

static float* g_partial    = nullptr;
static size_t g_partial_sz = 0;

static void ensure_buf(size_t bytes) {
    if (bytes > g_partial_sz) {
        if (g_partial) cudaFree(g_partial);
        cudaMalloc(&g_partial, bytes);
        g_partial_sz = bytes;
    }
}

static int g_best_cfg = -1;

static void run_cfg(int cfg,
                    const half* A, const half* Bcol,
                    float* partial, half* C)
{
    const int MN = MN_SIZE;
    if (cfg == 0) {
        hgemm_wmma_sk128<<<SPLIT_K, 256>>>(A, Bcol, partial);
        reduce_sk128_fast<<<MN / 32, dim3(32, 8)>>>(partial, C, MN);
    } else if (cfg == 1) {
        hgemm_wmma_sk256<<<SPLIT_K2, 256>>>(A, Bcol, partial);
        reduce_sk256_fast<<<MN / 32, dim3(32, 8)>>>(partial, C, MN);
    } else {
        hgemm_ptx_mma_sk128<<<SPLIT_K, 256>>>(A, Bcol, partial);
        reduce_sk128_fast<<<MN / 32, dim3(32, 8)>>>(partial, C, MN);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const half* ptr_A    = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_Bcol = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       ptr_C    = reinterpret_cast<half*>(c.data_ptr());

    const int MN = MN_SIZE;
    const size_t buf_needed = (size_t)SPLIT_K2 * MN * sizeof(float);
    ensure_buf(buf_needed);

    if (g_best_cfg < 0) {
        for (int w = 0; w < 5; w++) {
            for (int ci = 0; ci < 3; ci++) {
                run_cfg(ci, ptr_A, ptr_Bcol, g_partial, ptr_C);
            }
        }
        cudaDeviceSynchronize();

        float best_t = 1e18f;
        int   best_c = 0;
        const int NREP = 300;

        for (int ci = 0; ci < 3; ci++) {
            cudaEvent_t s, e;
            cudaEventCreate(&s);
            cudaEventCreate(&e);
            cudaEventRecord(s);
            for (int i = 0; i < NREP; i++) {
                run_cfg(ci, ptr_A, ptr_Bcol, g_partial, ptr_C);
            }
            cudaEventRecord(e);
            cudaDeviceSynchronize();
            float t = 0;
            cudaEventElapsedTime(&t, s, e);
            if (t < best_t) { best_t = t; best_c = ci; }
            cudaEventDestroy(s);
            cudaEventDestroy(e);
        }
        g_best_cfg = best_c;
    }

    run_cfg(g_best_cfg, ptr_A, ptr_Bcol, g_partial, ptr_C);
}