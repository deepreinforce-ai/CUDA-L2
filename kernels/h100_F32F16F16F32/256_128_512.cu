#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda::wmma;

static constexpr int P_BM      = 32;
static constexpr int P_BN      = 64;
static constexpr int P_BK      = 64;
static constexpr int P_STAGES  = 3;
static constexpr int P_NTHREADS = 128;
static constexpr int P_A_STRIDE = P_BK + 8;
static constexpr int P_B_STRIDE = P_BN + 8;
static constexpr int P_WARPS_M = 2;
static constexpr int P_WARPS_N = 2;
static constexpr int P_WARP_N_TILES = 2;

__global__ __launch_bounds__(128, 10)
void hgemm_primary_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.x * P_BM;
    const int block_n = blockIdx.y * P_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_row    = warp_id / P_WARPS_N;
    const int warp_col    = warp_id % P_WARPS_N;
    const int warp_m_base = warp_row * 16;
    const int warp_n_base = warp_col * P_WARP_N_TILES * 16;

    __shared__ half smem_A[P_STAGES][P_BM][P_A_STRIDE];
    __shared__ half smem_B[P_STAGES][P_BK][P_B_STRIDE];

    fragment<accumulator, 16, 16, 16, float> frag_c[P_WARP_N_TILES];
    fill_fragment(frag_c[0], 0.0f);
    fill_fragment(frag_c[1], 0.0f);

    const int num_k_tiles = (K + P_BK - 1) / P_BK;

    const int a_lin   = tid * 16;
    const int a_lrow  = a_lin / P_BK;
    const int a_lcol  = a_lin % P_BK;
    const int ga_row  = block_m + a_lrow;
    const bool a_valid = (ga_row < M);

    const int b_lin0  = tid * 32;
    const int b_lin1  = b_lin0 + 16;
    const int b0_row  = b_lin0 / P_BN;
    const int b0_col  = b_lin0 % P_BN;
    const int b1_row  = b_lin1 / P_BN;
    const int b1_col  = b_lin1 % P_BN;
    const int gb0_col = block_n + b0_col;
    const int gb1_col = block_n + b1_col;

#define P_SMPTR(p) static_cast<uint32_t>(__cvta_generic_to_shared(p))

#define P_ASYNC_A(s, ko) \
    do { \
        const int _kc = (ko) + a_lcol; \
        if (a_valid && _kc + 15 <= K) { \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(P_SMPTR(&smem_A[(s)][a_lrow][a_lcol])), \
                   "l"(&A[ga_row * K + _kc]) : "memory"); \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(P_SMPTR(&smem_A[(s)][a_lrow][a_lcol + 8])), \
                   "l"(&A[ga_row * K + _kc + 8]) : "memory"); \
        } else { \
            *reinterpret_cast<float4*>(&smem_A[(s)][a_lrow][a_lcol])     = make_float4(0,0,0,0); \
            *reinterpret_cast<float4*>(&smem_A[(s)][a_lrow][a_lcol + 8]) = make_float4(0,0,0,0); \
        } \
    } while(0)

#define P_ASYNC_B0(s, ko) \
    do { \
        const int _br = (ko) + b0_row; \
        if (_br < K && gb0_col + 7 < N) { \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(P_SMPTR(&smem_B[(s)][b0_row][b0_col])), \
                   "l"(&B[_br * N + gb0_col]) : "memory"); \
        } else { \
            *reinterpret_cast<float4*>(&smem_B[(s)][b0_row][b0_col]) = make_float4(0,0,0,0); \
        } \
    } while(0)

#define P_ASYNC_B1(s, ko) \
    do { \
        const int _br = (ko) + b1_row; \
        if (_br < K && gb1_col + 7 < N) { \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(P_SMPTR(&smem_B[(s)][b1_row][b1_col])), \
                   "l"(&B[_br * N + gb1_col]) : "memory"); \
        } else { \
            *reinterpret_cast<float4*>(&smem_B[(s)][b1_row][b1_col]) = make_float4(0,0,0,0); \
        } \
    } while(0)

    P_ASYNC_A(0, 0); P_ASYNC_B0(0, 0); P_ASYNC_B1(0, 0);
    asm volatile("cp.async.commit_group;\n"::);
    if (num_k_tiles > 1) {
        P_ASYNC_A(1, P_BK); P_ASYNC_B0(1, P_BK); P_ASYNC_B1(1, P_BK);
        asm volatile("cp.async.commit_group;\n"::);
    }
    asm volatile("cp.async.wait_group 1;\n"::);
    __syncthreads();

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int cur  = k_tile % P_STAGES;
        const int nxt2 = (k_tile + 2) % P_STAGES;

        if (k_tile + 2 < num_k_tiles) {
            const int k2 = (k_tile + 2) * P_BK;
            P_ASYNC_A(nxt2, k2); P_ASYNC_B0(nxt2, k2); P_ASYNC_B1(nxt2, k2);
            asm volatile("cp.async.commit_group;\n"::);
        }

        fragment<matrix_a, 16,16,16, half, row_major> fa0, fa1;
        fragment<matrix_b, 16,16,16, half, row_major> fb00, fb01, fb10, fb11;

        load_matrix_sync(fa0,  &smem_A[cur][warp_m_base][0],         P_A_STRIDE);
        load_matrix_sync(fb00, &smem_B[cur][0][warp_n_base],          P_B_STRIDE);
        load_matrix_sync(fb01, &smem_B[cur][0][warp_n_base + 16],     P_B_STRIDE);

        load_matrix_sync(fa1,  &smem_A[cur][warp_m_base][16],         P_A_STRIDE);
        load_matrix_sync(fb10, &smem_B[cur][16][warp_n_base],         P_B_STRIDE);
        load_matrix_sync(fb11, &smem_B[cur][16][warp_n_base + 16],    P_B_STRIDE);
        mma_sync(frag_c[0], fa0, fb00, frag_c[0]);
        mma_sync(frag_c[1], fa0, fb01, frag_c[1]);

        load_matrix_sync(fa0,  &smem_A[cur][warp_m_base][32],         P_A_STRIDE);
        load_matrix_sync(fb00, &smem_B[cur][32][warp_n_base],         P_B_STRIDE);
        load_matrix_sync(fb01, &smem_B[cur][32][warp_n_base + 16],    P_B_STRIDE);
        mma_sync(frag_c[0], fa1, fb10, frag_c[0]);
        mma_sync(frag_c[1], fa1, fb11, frag_c[1]);

        load_matrix_sync(fa1,  &smem_A[cur][warp_m_base][48],         P_A_STRIDE);
        load_matrix_sync(fb10, &smem_B[cur][48][warp_n_base],         P_B_STRIDE);
        load_matrix_sync(fb11, &smem_B[cur][48][warp_n_base + 16],    P_B_STRIDE);
        mma_sync(frag_c[0], fa0, fb00, frag_c[0]);
        mma_sync(frag_c[1], fa0, fb01, frag_c[1]);

        mma_sync(frag_c[0], fa1, fb10, frag_c[0]);
        mma_sync(frag_c[1], fa1, fb11, frag_c[1]);

        if (k_tile + 1 < num_k_tiles) {
            if (k_tile + 2 < num_k_tiles)
                asm volatile("cp.async.wait_group 1;\n"::);
            else
                asm volatile("cp.async.wait_group 0;\n"::);
            __syncthreads();
        }
    }

#undef P_ASYNC_A
#undef P_ASYNC_B0
#undef P_ASYNC_B1
#undef P_SMPTR

    const int lrow = lane_id >> 2;
    const int lcol = (lane_id & 3) << 1;

    #pragma unroll
    for (int t = 0; t < P_WARP_N_TILES; t++) {
        const int base_n = block_n + warp_n_base + t * 16;

        const int gr0 = block_m + warp_m_base + lrow;
        const int gr8 = gr0 + 8;
        const int gc0 = base_n + lcol;
        const int gc8 = base_n + lcol + 8;

        if (gr0 < M) {
            if (gc0 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr0 * N + gc0]) =
                    __floats2half2_rn(frag_c[t].x[0], frag_c[t].x[1]);
            else if (gc0 < N)
                C[gr0 * N + gc0] = __float2half(frag_c[t].x[0]);

            if (gc8 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr0 * N + gc8]) =
                    __floats2half2_rn(frag_c[t].x[4], frag_c[t].x[5]);
            else if (gc8 < N)
                C[gr0 * N + gc8] = __float2half(frag_c[t].x[4]);
        }
        if (gr8 < M) {
            if (gc0 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr8 * N + gc0]) =
                    __floats2half2_rn(frag_c[t].x[2], frag_c[t].x[3]);
            else if (gc0 < N)
                C[gr8 * N + gc0] = __float2half(frag_c[t].x[2]);

            if (gc8 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr8 * N + gc8]) =
                    __floats2half2_rn(frag_c[t].x[6], frag_c[t].x[7]);
            else if (gc8 < N)
                C[gr8 * N + gc8] = __float2half(frag_c[t].x[6]);
        }
    }
}

static constexpr int G_BM      = 32;
static constexpr int G_BN      = 32;
static constexpr int G_BK      = 64;
static constexpr int G_STAGES  = 4;
static constexpr int G_NTHREADS = 128;
static constexpr int G_A_STRIDE = G_BK + 8;
static constexpr int G_B_STRIDE = G_BN + 8;

__global__ __launch_bounds__(128, 8)
void hgemm_secondary_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.x * G_BM;
    const int block_n = blockIdx.y * G_BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_row    = warp_id >> 1;
    const int warp_col    = warp_id & 1;
    const int warp_m_base = warp_row * 16;
    const int warp_n_base = warp_col * 16;

    __shared__ half smem_A[G_STAGES][G_BM][G_A_STRIDE];
    __shared__ half smem_B[G_STAGES][G_BK][G_B_STRIDE];

    fragment<accumulator, 16, 16, 16, float> frag_c;
    fill_fragment(frag_c, 0.0f);

    const int num_k_tiles = (K + G_BK - 1) / G_BK;

    const int a_lin   = tid * 16;
    const int a_lrow  = a_lin / G_BK;
    const int a_lcol0 = a_lin % G_BK;
    const int a_lcol1 = a_lcol0 + 8;
    const int ga_row  = block_m + a_lrow;
    const bool a_valid = (ga_row < M);

    const int b_lin   = tid * 16;
    const int b_lrow0 = b_lin / G_BN;
    const int b_lcol0 = b_lin % G_BN;
    const int b_lin1  = b_lin + 8;
    const int b_lrow1 = b_lin1 / G_BN;
    const int b_lcol1 = b_lin1 % G_BN;
    const int gb_col0 = block_n + b_lcol0;
    const int gb_col1 = block_n + b_lcol1;

#define G_PTR(p) static_cast<uint32_t>(__cvta_generic_to_shared(p))

#define G_ASYNC_A(s, ko) \
    do { \
        const int _kc0 = (ko) + a_lcol0; \
        const int _kc1 = (ko) + a_lcol1; \
        if (a_valid && _kc0 + 7 < K) { \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(G_PTR(&smem_A[(s)][a_lrow][a_lcol0])), \
                   "l"(&A[ga_row * K + _kc0]) : "memory"); \
        } else { \
            *reinterpret_cast<float4*>(&smem_A[(s)][a_lrow][a_lcol0]) = make_float4(0,0,0,0); \
        } \
        if (a_valid && _kc1 + 7 < K) { \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(G_PTR(&smem_A[(s)][a_lrow][a_lcol1])), \
                   "l"(&A[ga_row * K + _kc1]) : "memory"); \
        } else { \
            *reinterpret_cast<float4*>(&smem_A[(s)][a_lrow][a_lcol1]) = make_float4(0,0,0,0); \
        } \
    } while(0)

#define G_ASYNC_B(s, ko) \
    do { \
        const int _br0 = (ko) + b_lrow0; \
        const int _br1 = (ko) + b_lrow1; \
        if (_br0 < K && gb_col0 + 7 < N) { \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(G_PTR(&smem_B[(s)][b_lrow0][b_lcol0])), \
                   "l"(&B[_br0 * N + gb_col0]) : "memory"); \
        } else { \
            *reinterpret_cast<float4*>(&smem_B[(s)][b_lrow0][b_lcol0]) = make_float4(0,0,0,0); \
        } \
        if (_br1 < K && gb_col1 + 7 < N) { \
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                :: "r"(G_PTR(&smem_B[(s)][b_lrow1][b_lcol1])), \
                   "l"(&B[_br1 * N + gb_col1]) : "memory"); \
        } else { \
            *reinterpret_cast<float4*>(&smem_B[(s)][b_lrow1][b_lcol1]) = make_float4(0,0,0,0); \
        } \
    } while(0)

    G_ASYNC_A(0, 0); G_ASYNC_B(0, 0);
    asm volatile("cp.async.commit_group;\n"::);
    if (num_k_tiles > 1) { G_ASYNC_A(1, G_BK); G_ASYNC_B(1, G_BK); asm volatile("cp.async.commit_group;\n"::); }
    if (num_k_tiles > 2) { G_ASYNC_A(2, 2*G_BK); G_ASYNC_B(2, 2*G_BK); asm volatile("cp.async.commit_group;\n"::); }
    asm volatile("cp.async.wait_group 2;\n"::);
    __syncthreads();

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int cur  = k_tile & 3;
        const int nxt3 = (k_tile + 3) & 3;

        if (k_tile + 3 < num_k_tiles) {
            G_ASYNC_A(nxt3, (k_tile + 3) * G_BK);
            G_ASYNC_B(nxt3, (k_tile + 3) * G_BK);
            asm volatile("cp.async.commit_group;\n"::);
        }

        fragment<matrix_a, 16,16,16, half, row_major> fa0, fa1, fa2, fa3;
        fragment<matrix_b, 16,16,16, half, row_major> fb0, fb1, fb2, fb3;

        load_matrix_sync(fa0, &smem_A[cur][warp_m_base][0],    G_A_STRIDE);
        load_matrix_sync(fb0, &smem_B[cur][0][warp_n_base],     G_B_STRIDE);
        load_matrix_sync(fa1, &smem_A[cur][warp_m_base][16],   G_A_STRIDE);
        load_matrix_sync(fb1, &smem_B[cur][16][warp_n_base],   G_B_STRIDE);
        mma_sync(frag_c, fa0, fb0, frag_c);

        load_matrix_sync(fa2, &smem_A[cur][warp_m_base][32],   G_A_STRIDE);
        load_matrix_sync(fb2, &smem_B[cur][32][warp_n_base],   G_B_STRIDE);
        mma_sync(frag_c, fa1, fb1, frag_c);

        load_matrix_sync(fa3, &smem_A[cur][warp_m_base][48],   G_A_STRIDE);
        load_matrix_sync(fb3, &smem_B[cur][48][warp_n_base],   G_B_STRIDE);
        mma_sync(frag_c, fa2, fb2, frag_c);

        mma_sync(frag_c, fa3, fb3, frag_c);

        if (k_tile + 1 < num_k_tiles) {
            if (k_tile + 3 < num_k_tiles)
                asm volatile("cp.async.wait_group 2;\n"::);
            else
                asm volatile("cp.async.wait_group 0;\n"::);
            __syncthreads();
        }
    }

#undef G_ASYNC_A
#undef G_ASYNC_B
#undef G_PTR

    const int lrow = lane_id >> 2;
    const int lcol = (lane_id & 3) << 1;

    const int gr0 = block_m + warp_m_base + lrow;
    const int gr8 = gr0 + 8;
    const int gc0 = block_n + warp_n_base + lcol;
    const int gc8 = gc0 + 8;

    if (gr0 < M) {
        if (gc0 + 1 <= N)
            *reinterpret_cast<half2*>(&C[gr0 * N + gc0]) = __floats2half2_rn(frag_c.x[0], frag_c.x[1]);
        else if (gc0 < N) C[gr0 * N + gc0] = __float2half(frag_c.x[0]);
        if (gc8 + 1 <= N)
            *reinterpret_cast<half2*>(&C[gr0 * N + gc8]) = __floats2half2_rn(frag_c.x[4], frag_c.x[5]);
        else if (gc8 < N) C[gr0 * N + gc8] = __float2half(frag_c.x[4]);
    }
    if (gr8 < M) {
        if (gc0 + 1 <= N)
            *reinterpret_cast<half2*>(&C[gr8 * N + gc0]) = __floats2half2_rn(frag_c.x[2], frag_c.x[3]);
        else if (gc0 < N) C[gr8 * N + gc0] = __float2half(frag_c.x[2]);
        if (gc8 + 1 <= N)
            *reinterpret_cast<half2*>(&C[gr8 * N + gc8]) = __floats2half2_rn(frag_c.x[6], frag_c.x[7]);
        else if (gc8 < N) C[gr8 * N + gc8] = __float2half(frag_c.x[6]);
    }
}

static constexpr int T_BM      = 32;
static constexpr int T_BN      = 128;
static constexpr int T_BK      = 32;
static constexpr int T_STAGES  = 4;
static constexpr int T_NTHREADS = 256;
static constexpr int T_A_STRIDE = T_BK + 8;
static constexpr int T_B_STRIDE = T_BN + 8;
static constexpr int T_WARPS_M = 2;
static constexpr int T_WARPS_N = 4;
static constexpr int T_WARP_N_TILES = 2;

__global__ __launch_bounds__(256, 4)
void hgemm_tertiary_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.x * T_BM;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_row    = warp_id / T_WARPS_N;
    const int warp_col    = warp_id % T_WARPS_N;
    const int warp_m_base = warp_row * 16;
    const int warp_n_base = warp_col * T_WARP_N_TILES * 16;

    __shared__ half smem_A[T_STAGES][T_BM][T_A_STRIDE];
    __shared__ half smem_B[T_STAGES][T_BK][T_B_STRIDE];

    fragment<accumulator, 16,16,16, float> frag_c[T_WARP_N_TILES];
    fill_fragment(frag_c[0], 0.0f);
    fill_fragment(frag_c[1], 0.0f);

    const int num_k_tiles = (K + T_BK - 1) / T_BK;

    const int a_lin   = tid * 4;
    const int a_lrow  = a_lin / T_BK;
    const int a_lcol  = a_lin % T_BK;
    const int ga_row  = block_m + a_lrow;
    const bool a_valid = (ga_row < M);

    const int b_lin   = tid * 16;
    const int b_lrow  = b_lin / T_BN;
    const int b_lcol0 = b_lin % T_BN;
    const int b_lcol1 = b_lcol0 + 8;

    const int b_lrow2 = tid >> 3;
    const int b_lcol_base = (tid & 7) << 4;
    const int b_lcol_0 = b_lcol_base;
    const int b_lcol_8 = b_lcol_base + 8;

#define T_SMPTR(p) static_cast<uint32_t>(__cvta_generic_to_shared(p))

#define T_ASYNC_A(s, ko) \
    do { \
        const int _kc = (ko) + a_lcol; \
        if (a_valid && _kc + 3 < K) { \
            asm volatile("cp.async.ca.shared.global [%0],[%1],8;\n" \
                :: "r"(T_SMPTR(&smem_A[(s)][a_lrow][a_lcol])), \
                   "l"(&A[ga_row * K + _kc]) : "memory"); \
        } else { \
            *reinterpret_cast<float2*>(&smem_A[(s)][a_lrow][a_lcol]) = make_float2(0,0); \
        } \
    } while(0)

#define T_ASYNC_B(s, ko) \
    do { \
        const int _br = (ko) + b_lrow2; \
        if (_br < K) { \
            if (b_lcol_0 + 7 < N) { \
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                    :: "r"(T_SMPTR(&smem_B[(s)][b_lrow2][b_lcol_0])), \
                       "l"(&B[_br * N + b_lcol_0]) : "memory"); \
            } else { \
                *reinterpret_cast<float4*>(&smem_B[(s)][b_lrow2][b_lcol_0]) = make_float4(0,0,0,0); \
            } \
            if (b_lcol_8 + 7 < N) { \
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" \
                    :: "r"(T_SMPTR(&smem_B[(s)][b_lrow2][b_lcol_8])), \
                       "l"(&B[_br * N + b_lcol_8]) : "memory"); \
            } else { \
                *reinterpret_cast<float4*>(&smem_B[(s)][b_lrow2][b_lcol_8]) = make_float4(0,0,0,0); \
            } \
        } else { \
            *reinterpret_cast<float4*>(&smem_B[(s)][b_lrow2][b_lcol_0]) = make_float4(0,0,0,0); \
            *reinterpret_cast<float4*>(&smem_B[(s)][b_lrow2][b_lcol_8]) = make_float4(0,0,0,0); \
        } \
    } while(0)

    T_ASYNC_A(0, 0); T_ASYNC_B(0, 0);
    asm volatile("cp.async.commit_group;\n"::);
    if (num_k_tiles > 1) { T_ASYNC_A(1, T_BK); T_ASYNC_B(1, T_BK); asm volatile("cp.async.commit_group;\n"::); }
    if (num_k_tiles > 2) { T_ASYNC_A(2, 2*T_BK); T_ASYNC_B(2, 2*T_BK); asm volatile("cp.async.commit_group;\n"::); }
    asm volatile("cp.async.wait_group 2;\n"::);
    __syncthreads();

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int cur  = k_tile & 3;
        const int nxt3 = (k_tile + 3) & 3;

        if (k_tile + 3 < num_k_tiles) {
            T_ASYNC_A(nxt3, (k_tile + 3) * T_BK);
            T_ASYNC_B(nxt3, (k_tile + 3) * T_BK);
            asm volatile("cp.async.commit_group;\n"::);
        }

        fragment<matrix_a, 16,16,16, half, row_major> fa0, fa1;
        fragment<matrix_b, 16,16,16, half, row_major> fb00, fb01, fb10, fb11;

        load_matrix_sync(fa0,  &smem_A[cur][warp_m_base][0],           T_A_STRIDE);
        load_matrix_sync(fb00, &smem_B[cur][0][warp_n_base],            T_B_STRIDE);
        load_matrix_sync(fb01, &smem_B[cur][0][warp_n_base + 16],       T_B_STRIDE);
        load_matrix_sync(fa1,  &smem_A[cur][warp_m_base][16],           T_A_STRIDE);
        load_matrix_sync(fb10, &smem_B[cur][16][warp_n_base],           T_B_STRIDE);
        load_matrix_sync(fb11, &smem_B[cur][16][warp_n_base + 16],      T_B_STRIDE);
        mma_sync(frag_c[0], fa0, fb00, frag_c[0]);
        mma_sync(frag_c[1], fa0, fb01, frag_c[1]);
        mma_sync(frag_c[0], fa1, fb10, frag_c[0]);
        mma_sync(frag_c[1], fa1, fb11, frag_c[1]);

        if (k_tile + 1 < num_k_tiles) {
            if (k_tile + 3 < num_k_tiles)
                asm volatile("cp.async.wait_group 2;\n"::);
            else
                asm volatile("cp.async.wait_group 0;\n"::);
            __syncthreads();
        }
    }

#undef T_ASYNC_A
#undef T_ASYNC_B
#undef T_SMPTR

    const int lrow = lane_id >> 2;
    const int lcol = (lane_id & 3) << 1;

    #pragma unroll
    for (int t = 0; t < T_WARP_N_TILES; t++) {
        const int base_n = warp_n_base + t * 16;
        const int gr0 = block_m + warp_m_base + lrow;
        const int gr8 = gr0 + 8;
        const int gc0 = base_n + lcol;
        const int gc8 = base_n + lcol + 8;

        if (gr0 < M) {
            if (gc0 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr0 * N + gc0]) =
                    __floats2half2_rn(frag_c[t].x[0], frag_c[t].x[1]);
            else if (gc0 < N) C[gr0 * N + gc0] = __float2half(frag_c[t].x[0]);
            if (gc8 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr0 * N + gc8]) =
                    __floats2half2_rn(frag_c[t].x[4], frag_c[t].x[5]);
            else if (gc8 < N) C[gr0 * N + gc8] = __float2half(frag_c[t].x[4]);
        }
        if (gr8 < M) {
            if (gc0 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr8 * N + gc0]) =
                    __floats2half2_rn(frag_c[t].x[2], frag_c[t].x[3]);
            else if (gc0 < N) C[gr8 * N + gc0] = __float2half(frag_c[t].x[2]);
            if (gc8 + 1 <= N)
                *reinterpret_cast<half2*>(&C[gr8 * N + gc8]) =
                    __floats2half2_rn(frag_c[t].x[6], frag_c[t].x[7]);
            else if (gc8 < N) C[gr8 * N + gc8] = __float2half(frag_c[t].x[6]);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C       = reinterpret_cast<half*>(c.data_ptr());

    {
        dim3 grid((M + G_BM - 1) / G_BM, (N + G_BN - 1) / G_BN);
        dim3 block(G_NTHREADS);
        hgemm_secondary_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
}