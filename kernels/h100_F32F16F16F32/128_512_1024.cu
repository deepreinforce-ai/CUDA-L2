#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

__device__ __forceinline__
uint32_t smem_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__
void ldmatrix_x4(uint32_t (&R)[4], const void* ptr) {
    uint32_t addr = smem_u32(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
        : "r"(addr));
}

__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t (&R)[2], const void* ptr) {
    uint32_t addr = smem_u32(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(R[0]), "=r"(R[1])
        : "r"(addr));
}

__device__ __forceinline__
void mma_m16n8k16(const uint32_t (&A)[4], const uint32_t (&B)[2], float (&D)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

__device__ __forceinline__
void cp_async16_ca_zfill(void* dst, const void* src, bool valid) {
    uint32_t dst_addr = smem_u32(dst);
    uint32_t sz = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst_addr), "l"(src), "r"(sz) : "memory");
}

__device__ __forceinline__
void cp_async16_cg_zfill(void* dst, const void* src, bool valid) {
    uint32_t dst_addr = smem_u32(dst);
    uint32_t sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :: "r"(dst_addr), "l"(src), "r"(sz) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int REMAIN>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(REMAIN) : "memory");
}

__device__ __forceinline__ int sw64(int row, int col) {
    int chunk  = (col >> 3) & 7;
    int offset = col & 7;
    int phys   = chunk ^ (row & 7);
    return (phys << 3) | offset;
}

__device__ __forceinline__ int sw128(int row, int col) {
    int chunk  = (col >> 3) & 15;
    int offset = col & 7;
    int phys   = chunk ^ (row & 15);
    return (phys << 3) | offset;
}

__device__ __forceinline__ int sw256(int row, int col) {
    int chunk  = (col >> 3) & 31;
    int offset = col & 7;
    int phys   = chunk ^ (row & 31);
    return (phys << 3) | offset;
}

static constexpr int K1_BM       = 64;
static constexpr int K1_BN       = 64;
static constexpr int K1_BK       = 64;
static constexpr int K1_STAGES   = 4;
static constexpr int K1_WARPS_M  = 2;
static constexpr int K1_WARPS_N  = 4;
static constexpr int K1_NWARPS   = K1_WARPS_M * K1_WARPS_N;
static constexpr int K1_NTHREADS = K1_NWARPS * 32;
static constexpr int K1_WARP_M   = K1_BM / K1_WARPS_M;
static constexpr int K1_WARP_N   = K1_BN / K1_WARPS_N;
static constexpr int K1_MMA_M    = 16;
static constexpr int K1_MMA_N    = 8;
static constexpr int K1_MMA_K    = 16;
static constexpr int K1_WARP_TM  = K1_WARP_M / K1_MMA_M;
static constexpr int K1_WARP_TN  = K1_WARP_N / K1_MMA_N;
static constexpr int K1_K_TILES  = K1_BK / K1_MMA_K;

static constexpr int K1_A_STRIDE = K1_BK;
static constexpr int K1_B_STRIDE = K1_BN;
static constexpr int K1_A_STAGE  = K1_BM * K1_A_STRIDE;
static constexpr int K1_B_STAGE  = K1_BK * K1_B_STRIDE;
static constexpr int K1_STAGE_SZ = K1_A_STAGE + K1_B_STAGE;
static constexpr int K1_SMEM_BYTES = K1_STAGES * K1_STAGE_SZ * 2;

__global__ __launch_bounds__(K1_NTHREADS, 2)
void hgemm_k1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int cta_m = blockIdx.y * K1_BM;
    const int cta_n = blockIdx.x * K1_BN;

    const int warp_row = warp_id / K1_WARPS_N;
    const int warp_col = warp_id % K1_WARPS_N;
    const int wm_base  = warp_row * K1_WARP_M;
    const int wn_base  = warp_col * K1_WARP_N;

    float acc[K1_WARP_TM][K1_WARP_TN][4];
    #pragma unroll
    for (int i = 0; i < K1_WARP_TM; i++)
        #pragma unroll
        for (int j = 0; j < K1_WARP_TN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int a_tpr  = K1_BK / 8;
    const int a_row0 = tid / a_tpr;
    const int a_col  = (tid % a_tpr) * 8;
    const int a_row1 = a_row0 + 32;

    const int b_tpr  = K1_BN / 8;
    const int b_row0 = tid / b_tpr;
    const int b_col  = (tid % b_tpr) * 8;
    const int b_row1 = b_row0 + 32;

    const int num_k_tiles = (K + K1_BK - 1) / K1_BK;

    #pragma unroll
    for (int s = 0; s < K1_STAGES - 1; s++) {
        half* sA = smem + s * K1_STAGE_SZ;
        half* sB = sA + K1_A_STAGE;
        int k_off = s * K1_BK;
        bool vs = (s < num_k_tiles);
        { int gr = cta_m + a_row0, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row0 * K1_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gr = cta_m + a_row1, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row1 * K1_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gk = k_off + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K1_B_STRIDE + sw64(b_row0, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K1_B_STRIDE + sw64(b_row1, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        cp_async_commit();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int cur   = k_tile % K1_STAGES;
        int nxt   = (k_tile + K1_STAGES - 1) % K1_STAGES;
        int k_nxt = (k_tile + K1_STAGES - 1) * K1_BK;
        {
            half* sA = smem + nxt * K1_STAGE_SZ;
            half* sB = sA + K1_A_STAGE;
            bool vn = (k_nxt < K);
            { int gr = cta_m + a_row0, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row0 * K1_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gr = cta_m + a_row1, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row1 * K1_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gk = k_nxt + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K1_B_STRIDE + sw64(b_row0, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K1_B_STRIDE + sw64(b_row1, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            cp_async_commit();
        }
        cp_async_wait<K1_STAGES - 2>();
        __syncthreads();

        const half* sA_cur = smem + cur * K1_STAGE_SZ;
        const half* sB_cur = sA_cur + K1_A_STAGE;

        uint32_t rA[2][K1_WARP_TM][4];
        uint32_t rB[2][K1_WARP_TN][2];

        #pragma unroll
        for (int mi = 0; mi < K1_WARP_TM; mi++) {
            int row = wm_base + mi * K1_MMA_M + (lane_id & 15);
            int col = ((lane_id >> 4) & 1) * 8;
            ldmatrix_x4(rA[0][mi], &sA_cur[row * K1_A_STRIDE + sw64(row, col)]);
        }
        #pragma unroll
        for (int ni = 0; ni < K1_WARP_TN; ni++) {
            int row = lane_id & 15;
            int col = wn_base + ni * K1_MMA_N;
            ldmatrix_x2_trans(rB[0][ni], &sB_cur[row * K1_B_STRIDE + sw64(row, col)]);
        }

        #pragma unroll
        for (int ki = 0; ki < K1_K_TILES; ki++) {
            int cb = ki & 1, nb = cb ^ 1;
            if (ki + 1 < K1_K_TILES) {
                #pragma unroll
                for (int mi = 0; mi < K1_WARP_TM; mi++) {
                    int row = wm_base + mi * K1_MMA_M + (lane_id & 15);
                    int col = (ki + 1) * K1_MMA_K + ((lane_id >> 4) & 1) * 8;
                    ldmatrix_x4(rA[nb][mi], &sA_cur[row * K1_A_STRIDE + sw64(row, col)]);
                }
                #pragma unroll
                for (int ni = 0; ni < K1_WARP_TN; ni++) {
                    int row = (ki + 1) * K1_MMA_K + (lane_id & 15);
                    int col = wn_base + ni * K1_MMA_N;
                    ldmatrix_x2_trans(rB[nb][ni], &sB_cur[row * K1_B_STRIDE + sw64(row, col)]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < K1_WARP_TM; mi++)
                #pragma unroll
                for (int ni = 0; ni < K1_WARP_TN; ni++)
                    mma_m16n8k16(rA[cb][mi], rB[cb][ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < K1_WARP_TM; mi++) {
        int base_row = cta_m + wm_base + mi * K1_MMA_M;
        int row0 = base_row + (lane_id >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < K1_WARP_TN; ni++) {
            int col0 = cta_n + wn_base + ni * K1_MMA_N + (lane_id & 3) * 2;
            if (row0 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row0 * N + col0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            else if (row0 < M && col0 < N)
                C[row0 * N + col0] = __float2half(acc[mi][ni][0]);
            if (row1 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row1 * N + col0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            else if (row1 < M && col0 < N)
                C[row1 * N + col0] = __float2half(acc[mi][ni][2]);
        }
    }
}

static constexpr int K2_BM       = 32;
static constexpr int K2_BN       = 64;
static constexpr int K2_BK       = 64;
static constexpr int K2_STAGES   = 4;
static constexpr int K2_WARPS_M  = 1;
static constexpr int K2_WARPS_N  = 4;
static constexpr int K2_NWARPS   = K2_WARPS_M * K2_WARPS_N;
static constexpr int K2_NTHREADS = K2_NWARPS * 32;
static constexpr int K2_WARP_M   = K2_BM / K2_WARPS_M;
static constexpr int K2_WARP_N   = K2_BN / K2_WARPS_N;
static constexpr int K2_MMA_M    = 16;
static constexpr int K2_MMA_N    = 8;
static constexpr int K2_MMA_K    = 16;
static constexpr int K2_WARP_TM  = K2_WARP_M / K2_MMA_M;
static constexpr int K2_WARP_TN  = K2_WARP_N / K2_MMA_N;
static constexpr int K2_K_TILES  = K2_BK / K2_MMA_K;

static constexpr int K2_A_STRIDE = K2_BK;
static constexpr int K2_B_STRIDE = K2_BN;
static constexpr int K2_A_STAGE  = K2_BM * K2_A_STRIDE;
static constexpr int K2_B_STAGE  = K2_BK * K2_B_STRIDE;
static constexpr int K2_STAGE_SZ = K2_A_STAGE + K2_B_STAGE;
static constexpr int K2_SMEM_BYTES = K2_STAGES * K2_STAGE_SZ * 2;

__global__ __launch_bounds__(K2_NTHREADS, 4)
void hgemm_k2(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int cta_m = blockIdx.y * K2_BM;
    const int cta_n = blockIdx.x * K2_BN;

    const int warp_col = warp_id;
    const int wm_base  = 0;
    const int wn_base  = warp_col * K2_WARP_N;

    float acc[K2_WARP_TM][K2_WARP_TN][4];
    #pragma unroll
    for (int i = 0; i < K2_WARP_TM; i++)
        #pragma unroll
        for (int j = 0; j < K2_WARP_TN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int a_tpr  = K2_BK / 8;
    const int a_row0 = tid / a_tpr;
    const int a_col  = (tid % a_tpr) * 8;
    const int a_row1 = a_row0 + 16;

    const int b_tpr  = K2_BN / 8;
    const int b_row0 = tid / b_tpr;
    const int b_col  = (tid % b_tpr) * 8;
    const int b_row1 = b_row0 + 16;
    const int b_row2 = b_row0 + 32;
    const int b_row3 = b_row0 + 48;

    const int num_k_tiles = (K + K2_BK - 1) / K2_BK;

    #pragma unroll
    for (int s = 0; s < K2_STAGES - 1; s++) {
        half* sA = smem + s * K2_STAGE_SZ;
        half* sB = sA + K2_A_STAGE;
        int k_off = s * K2_BK;
        bool vs = (s < num_k_tiles);
        { int gr = cta_m + a_row0, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row0 * K2_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gr = cta_m + a_row1, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row1 * K2_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gk = k_off + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K2_B_STRIDE + sw64(b_row0, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K2_B_STRIDE + sw64(b_row1, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row2, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row2 * K2_B_STRIDE + sw64(b_row2, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row3, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row3 * K2_B_STRIDE + sw64(b_row3, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        cp_async_commit();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int cur   = k_tile % K2_STAGES;
        int nxt   = (k_tile + K2_STAGES - 1) % K2_STAGES;
        int k_nxt = (k_tile + K2_STAGES - 1) * K2_BK;
        {
            half* sA = smem + nxt * K2_STAGE_SZ;
            half* sB = sA + K2_A_STAGE;
            bool vn = (k_nxt < K);
            { int gr = cta_m + a_row0, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row0 * K2_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gr = cta_m + a_row1, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row1 * K2_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gk = k_nxt + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K2_B_STRIDE + sw64(b_row0, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K2_B_STRIDE + sw64(b_row1, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row2, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row2 * K2_B_STRIDE + sw64(b_row2, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row3, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row3 * K2_B_STRIDE + sw64(b_row3, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            cp_async_commit();
        }
        cp_async_wait<K2_STAGES - 2>();
        __syncthreads();

        const half* sA_cur = smem + cur * K2_STAGE_SZ;
        const half* sB_cur = sA_cur + K2_A_STAGE;

        uint32_t rA[2][K2_WARP_TM][4];
        uint32_t rB[2][K2_WARP_TN][2];

        #pragma unroll
        for (int mi = 0; mi < K2_WARP_TM; mi++) {
            int row = wm_base + mi * K2_MMA_M + (lane_id & 15);
            int col = ((lane_id >> 4) & 1) * 8;
            ldmatrix_x4(rA[0][mi], &sA_cur[row * K2_A_STRIDE + sw64(row, col)]);
        }
        #pragma unroll
        for (int ni = 0; ni < K2_WARP_TN; ni++) {
            int row = lane_id & 15;
            int col = wn_base + ni * K2_MMA_N;
            ldmatrix_x2_trans(rB[0][ni], &sB_cur[row * K2_B_STRIDE + sw64(row, col)]);
        }

        #pragma unroll
        for (int ki = 0; ki < K2_K_TILES; ki++) {
            int cb = ki & 1, nb = cb ^ 1;
            if (ki + 1 < K2_K_TILES) {
                #pragma unroll
                for (int mi = 0; mi < K2_WARP_TM; mi++) {
                    int row = wm_base + mi * K2_MMA_M + (lane_id & 15);
                    int col = (ki + 1) * K2_MMA_K + ((lane_id >> 4) & 1) * 8;
                    ldmatrix_x4(rA[nb][mi], &sA_cur[row * K2_A_STRIDE + sw64(row, col)]);
                }
                #pragma unroll
                for (int ni = 0; ni < K2_WARP_TN; ni++) {
                    int row = (ki + 1) * K2_MMA_K + (lane_id & 15);
                    int col = wn_base + ni * K2_MMA_N;
                    ldmatrix_x2_trans(rB[nb][ni], &sB_cur[row * K2_B_STRIDE + sw64(row, col)]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < K2_WARP_TM; mi++)
                #pragma unroll
                for (int ni = 0; ni < K2_WARP_TN; ni++)
                    mma_m16n8k16(rA[cb][mi], rB[cb][ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < K2_WARP_TM; mi++) {
        int base_row = cta_m + wm_base + mi * K2_MMA_M;
        int row0 = base_row + (lane_id >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < K2_WARP_TN; ni++) {
            int col0 = cta_n + wn_base + ni * K2_MMA_N + (lane_id & 3) * 2;
            if (row0 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row0 * N + col0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            else if (row0 < M && col0 < N)
                C[row0 * N + col0] = __float2half(acc[mi][ni][0]);
            if (row1 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row1 * N + col0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            else if (row1 < M && col0 < N)
                C[row1 * N + col0] = __float2half(acc[mi][ni][2]);
        }
    }
}

static constexpr int K3_BM       = 64;
static constexpr int K3_BN       = 128;
static constexpr int K3_BK       = 64;
static constexpr int K3_STAGES   = 4;
static constexpr int K3_WARPS_M  = 2;
static constexpr int K3_WARPS_N  = 4;
static constexpr int K3_NWARPS   = K3_WARPS_M * K3_WARPS_N;
static constexpr int K3_NTHREADS = K3_NWARPS * 32;
static constexpr int K3_WARP_M   = K3_BM / K3_WARPS_M;
static constexpr int K3_WARP_N   = K3_BN / K3_WARPS_N;
static constexpr int K3_MMA_M    = 16;
static constexpr int K3_MMA_N    = 8;
static constexpr int K3_MMA_K    = 16;
static constexpr int K3_WARP_TM  = K3_WARP_M / K3_MMA_M;
static constexpr int K3_WARP_TN  = K3_WARP_N / K3_MMA_N;
static constexpr int K3_K_TILES  = K3_BK / K3_MMA_K;

static constexpr int K3_A_STRIDE = K3_BK;
static constexpr int K3_B_STRIDE = K3_BN;
static constexpr int K3_A_STAGE  = K3_BM * K3_A_STRIDE;
static constexpr int K3_B_STAGE  = K3_BK * K3_B_STRIDE;
static constexpr int K3_STAGE_SZ = K3_A_STAGE + K3_B_STAGE;
static constexpr int K3_SMEM_BYTES = K3_STAGES * K3_STAGE_SZ * 2;

__global__ __launch_bounds__(K3_NTHREADS, 1)
void hgemm_k3(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int cta_m = blockIdx.y * K3_BM;
    const int cta_n = blockIdx.x * K3_BN;

    const int warp_row = warp_id / K3_WARPS_N;
    const int warp_col = warp_id % K3_WARPS_N;
    const int wm_base  = warp_row * K3_WARP_M;
    const int wn_base  = warp_col * K3_WARP_N;

    float acc[K3_WARP_TM][K3_WARP_TN][4];
    #pragma unroll
    for (int i = 0; i < K3_WARP_TM; i++)
        #pragma unroll
        for (int j = 0; j < K3_WARP_TN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int a_tpr  = K3_BK / 8;
    const int a_row0 = tid / a_tpr;
    const int a_col  = (tid % a_tpr) * 8;
    const int a_row1 = a_row0 + 32;

    const int b_tpr  = K3_BN / 8;
    const int b_row0 = tid / b_tpr;
    const int b_col  = (tid % b_tpr) * 8;
    const int b_row1 = b_row0 + 16;
    const int b_row2 = b_row0 + 32;
    const int b_row3 = b_row0 + 48;

    const int num_k_tiles = (K + K3_BK - 1) / K3_BK;

    #pragma unroll
    for (int s = 0; s < K3_STAGES - 1; s++) {
        half* sA = smem + s * K3_STAGE_SZ;
        half* sB = sA + K3_A_STAGE;
        int k_off = s * K3_BK;
        bool vs = (s < num_k_tiles);
        { int gr = cta_m + a_row0, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row0 * K3_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gr = cta_m + a_row1, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row1 * K3_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gk = k_off + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K3_B_STRIDE + sw128(b_row0, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K3_B_STRIDE + sw128(b_row1, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row2, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row2 * K3_B_STRIDE + sw128(b_row2, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row3, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row3 * K3_B_STRIDE + sw128(b_row3, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        cp_async_commit();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int cur   = k_tile % K3_STAGES;
        int nxt   = (k_tile + K3_STAGES - 1) % K3_STAGES;
        int k_nxt = (k_tile + K3_STAGES - 1) * K3_BK;
        {
            half* sA = smem + nxt * K3_STAGE_SZ;
            half* sB = sA + K3_A_STAGE;
            bool vn = (k_nxt < K);
            { int gr = cta_m + a_row0, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row0 * K3_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gr = cta_m + a_row1, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row1 * K3_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gk = k_nxt + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K3_B_STRIDE + sw128(b_row0, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K3_B_STRIDE + sw128(b_row1, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row2, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row2 * K3_B_STRIDE + sw128(b_row2, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row3, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row3 * K3_B_STRIDE + sw128(b_row3, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            cp_async_commit();
        }
        cp_async_wait<K3_STAGES - 2>();
        __syncthreads();

        const half* sA_cur = smem + cur * K3_STAGE_SZ;
        const half* sB_cur = sA_cur + K3_A_STAGE;

        uint32_t rA[2][K3_WARP_TM][4];
        uint32_t rB[2][K3_WARP_TN][2];

        #pragma unroll
        for (int mi = 0; mi < K3_WARP_TM; mi++) {
            int row = wm_base + mi * K3_MMA_M + (lane_id & 15);
            int col = ((lane_id >> 4) & 1) * 8;
            ldmatrix_x4(rA[0][mi], &sA_cur[row * K3_A_STRIDE + sw64(row, col)]);
        }
        #pragma unroll
        for (int ni = 0; ni < K3_WARP_TN; ni++) {
            int row = lane_id & 15;
            int col = wn_base + ni * K3_MMA_N;
            ldmatrix_x2_trans(rB[0][ni], &sB_cur[row * K3_B_STRIDE + sw128(row, col)]);
        }

        #pragma unroll
        for (int ki = 0; ki < K3_K_TILES; ki++) {
            int cb = ki & 1, nb = cb ^ 1;
            if (ki + 1 < K3_K_TILES) {
                #pragma unroll
                for (int mi = 0; mi < K3_WARP_TM; mi++) {
                    int row = wm_base + mi * K3_MMA_M + (lane_id & 15);
                    int col = (ki + 1) * K3_MMA_K + ((lane_id >> 4) & 1) * 8;
                    ldmatrix_x4(rA[nb][mi], &sA_cur[row * K3_A_STRIDE + sw64(row, col)]);
                }
                #pragma unroll
                for (int ni = 0; ni < K3_WARP_TN; ni++) {
                    int row = (ki + 1) * K3_MMA_K + (lane_id & 15);
                    int col = wn_base + ni * K3_MMA_N;
                    ldmatrix_x2_trans(rB[nb][ni], &sB_cur[row * K3_B_STRIDE + sw128(row, col)]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < K3_WARP_TM; mi++)
                #pragma unroll
                for (int ni = 0; ni < K3_WARP_TN; ni++)
                    mma_m16n8k16(rA[cb][mi], rB[cb][ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < K3_WARP_TM; mi++) {
        int base_row = cta_m + wm_base + mi * K3_MMA_M;
        int row0 = base_row + (lane_id >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < K3_WARP_TN; ni++) {
            int col0 = cta_n + wn_base + ni * K3_MMA_N + (lane_id & 3) * 2;
            if (row0 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row0 * N + col0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            else if (row0 < M && col0 < N)
                C[row0 * N + col0] = __float2half(acc[mi][ni][0]);
            if (row1 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row1 * N + col0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            else if (row1 < M && col0 < N)
                C[row1 * N + col0] = __float2half(acc[mi][ni][2]);
        }
    }
}

static constexpr int K4_BM       = 128;
static constexpr int K4_BN       = 128;
static constexpr int K4_BK       = 64;
static constexpr int K4_STAGES   = 3;
static constexpr int K4_WARPS_M  = 4;
static constexpr int K4_WARPS_N  = 4;
static constexpr int K4_NWARPS   = K4_WARPS_M * K4_WARPS_N;
static constexpr int K4_NTHREADS = K4_NWARPS * 32;
static constexpr int K4_WARP_M   = K4_BM / K4_WARPS_M;
static constexpr int K4_WARP_N   = K4_BN / K4_WARPS_N;
static constexpr int K4_MMA_M    = 16;
static constexpr int K4_MMA_N    = 8;
static constexpr int K4_MMA_K    = 16;
static constexpr int K4_WARP_TM  = K4_WARP_M / K4_MMA_M;
static constexpr int K4_WARP_TN  = K4_WARP_N / K4_MMA_N;
static constexpr int K4_K_TILES  = K4_BK / K4_MMA_K;

static constexpr int K4_A_STRIDE = K4_BK;
static constexpr int K4_B_STRIDE = K4_BN;
static constexpr int K4_A_STAGE  = K4_BM * K4_A_STRIDE;
static constexpr int K4_B_STAGE  = K4_BK * K4_B_STRIDE;
static constexpr int K4_STAGE_SZ = K4_A_STAGE + K4_B_STAGE;
static constexpr int K4_SMEM_BYTES = K4_STAGES * K4_STAGE_SZ * 2;

__global__ __launch_bounds__(K4_NTHREADS, 1)
void hgemm_k4(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int cta_m = blockIdx.y * K4_BM;
    const int cta_n = blockIdx.x * K4_BN;

    const int warp_row = warp_id / K4_WARPS_N;
    const int warp_col = warp_id % K4_WARPS_N;
    const int wm_base  = warp_row * K4_WARP_M;
    const int wn_base  = warp_col * K4_WARP_N;

    float acc[K4_WARP_TM][K4_WARP_TN][4];
    #pragma unroll
    for (int i = 0; i < K4_WARP_TM; i++)
        #pragma unroll
        for (int j = 0; j < K4_WARP_TN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int a_tpr  = K4_BK / 8;
    const int a_row0 = tid / a_tpr;
    const int a_col  = (tid % a_tpr) * 8;
    const int a_row1 = a_row0 + 64;

    const int b_tpr  = K4_BN / 8;
    const int b_row0 = tid / b_tpr;
    const int b_col  = (tid % b_tpr) * 8;
    const int b_row1 = b_row0 + 32;

    const int num_k_tiles = (K + K4_BK - 1) / K4_BK;

    #pragma unroll
    for (int s = 0; s < K4_STAGES - 1; s++) {
        half* sA = smem + s * K4_STAGE_SZ;
        half* sB = sA + K4_A_STAGE;
        int k_off = s * K4_BK;
        bool vs = (s < num_k_tiles);
        { int gr = cta_m + a_row0, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row0 * K4_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        if (a_row1 < K4_BM) { int gr = cta_m + a_row1, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row1 * K4_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gk = k_off + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K4_B_STRIDE + sw128(b_row0, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        if (b_row1 < K4_BK) { int gk = k_off + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K4_B_STRIDE + sw128(b_row1, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        cp_async_commit();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int cur   = k_tile % K4_STAGES;
        int nxt   = (k_tile + K4_STAGES - 1) % K4_STAGES;
        int k_nxt = (k_tile + K4_STAGES - 1) * K4_BK;
        {
            half* sA = smem + nxt * K4_STAGE_SZ;
            half* sB = sA + K4_A_STAGE;
            bool vn = (k_nxt < K);
            { int gr = cta_m + a_row0, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row0 * K4_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            if (a_row1 < K4_BM) { int gr = cta_m + a_row1, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row1 * K4_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gk = k_nxt + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K4_B_STRIDE + sw128(b_row0, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            if (b_row1 < K4_BK) { int gk = k_nxt + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K4_B_STRIDE + sw128(b_row1, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            cp_async_commit();
        }
        cp_async_wait<K4_STAGES - 2>();
        __syncthreads();

        const half* sA_cur = smem + cur * K4_STAGE_SZ;
        const half* sB_cur = sA_cur + K4_A_STAGE;

        uint32_t rA[2][K4_WARP_TM][4];
        uint32_t rB[2][K4_WARP_TN][2];

        #pragma unroll
        for (int mi = 0; mi < K4_WARP_TM; mi++) {
            int row = wm_base + mi * K4_MMA_M + (lane_id & 15);
            int col = ((lane_id >> 4) & 1) * 8;
            ldmatrix_x4(rA[0][mi], &sA_cur[row * K4_A_STRIDE + sw64(row, col)]);
        }
        #pragma unroll
        for (int ni = 0; ni < K4_WARP_TN; ni++) {
            int row = lane_id & 15;
            int col = wn_base + ni * K4_MMA_N;
            ldmatrix_x2_trans(rB[0][ni], &sB_cur[row * K4_B_STRIDE + sw128(row, col)]);
        }

        #pragma unroll
        for (int ki = 0; ki < K4_K_TILES; ki++) {
            int cb = ki & 1, nb = cb ^ 1;
            if (ki + 1 < K4_K_TILES) {
                #pragma unroll
                for (int mi = 0; mi < K4_WARP_TM; mi++) {
                    int row = wm_base + mi * K4_MMA_M + (lane_id & 15);
                    int col = (ki + 1) * K4_MMA_K + ((lane_id >> 4) & 1) * 8;
                    ldmatrix_x4(rA[nb][mi], &sA_cur[row * K4_A_STRIDE + sw64(row, col)]);
                }
                #pragma unroll
                for (int ni = 0; ni < K4_WARP_TN; ni++) {
                    int row = (ki + 1) * K4_MMA_K + (lane_id & 15);
                    int col = wn_base + ni * K4_MMA_N;
                    ldmatrix_x2_trans(rB[nb][ni], &sB_cur[row * K4_B_STRIDE + sw128(row, col)]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < K4_WARP_TM; mi++)
                #pragma unroll
                for (int ni = 0; ni < K4_WARP_TN; ni++)
                    mma_m16n8k16(rA[cb][mi], rB[cb][ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < K4_WARP_TM; mi++) {
        int base_row = cta_m + wm_base + mi * K4_MMA_M;
        int row0 = base_row + (lane_id >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < K4_WARP_TN; ni++) {
            int col0 = cta_n + wn_base + ni * K4_MMA_N + (lane_id & 3) * 2;
            if (row0 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row0 * N + col0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            else if (row0 < M && col0 < N)
                C[row0 * N + col0] = __float2half(acc[mi][ni][0]);
            if (row1 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row1 * N + col0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            else if (row1 < M && col0 < N)
                C[row1 * N + col0] = __float2half(acc[mi][ni][2]);
        }
    }
}

static constexpr int K5_BM       = 128;
static constexpr int K5_BN       = 256;
static constexpr int K5_BK       = 64;
static constexpr int K5_STAGES   = 3;
static constexpr int K5_WARPS_M  = 4;
static constexpr int K5_WARPS_N  = 4;
static constexpr int K5_NWARPS   = K5_WARPS_M * K5_WARPS_N;
static constexpr int K5_NTHREADS = K5_NWARPS * 32;
static constexpr int K5_WARP_M   = K5_BM / K5_WARPS_M;
static constexpr int K5_WARP_N   = K5_BN / K5_WARPS_N;
static constexpr int K5_MMA_M    = 16;
static constexpr int K5_MMA_N    = 8;
static constexpr int K5_MMA_K    = 16;
static constexpr int K5_WARP_TM  = K5_WARP_M / K5_MMA_M;
static constexpr int K5_WARP_TN  = K5_WARP_N / K5_MMA_N;
static constexpr int K5_K_TILES  = K5_BK / K5_MMA_K;

static constexpr int K5_A_STRIDE = K5_BK;
static constexpr int K5_B_STRIDE = K5_BN;
static constexpr int K5_A_STAGE  = K5_BM * K5_A_STRIDE;
static constexpr int K5_B_STAGE  = K5_BK * K5_B_STRIDE;
static constexpr int K5_STAGE_SZ = K5_A_STAGE + K5_B_STAGE;
static constexpr int K5_SMEM_BYTES = K5_STAGES * K5_STAGE_SZ * 2;

__global__ __launch_bounds__(K5_NTHREADS, 1)
void hgemm_k5(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int cta_m = blockIdx.y * K5_BM;
    const int cta_n = blockIdx.x * K5_BN;

    const int warp_row = warp_id / K5_WARPS_N;
    const int warp_col = warp_id % K5_WARPS_N;
    const int wm_base  = warp_row * K5_WARP_M;
    const int wn_base  = warp_col * K5_WARP_N;

    float acc[K5_WARP_TM][K5_WARP_TN][4];
    #pragma unroll
    for (int i = 0; i < K5_WARP_TM; i++)
        #pragma unroll
        for (int j = 0; j < K5_WARP_TN; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int a_tpr  = K5_BK / 8;
    const int a_row0 = tid / a_tpr;
    const int a_col  = (tid % a_tpr) * 8;
    const int a_row1 = a_row0 + 64;

    const int b_tpr  = K5_BN / 8;
    const int b_row0 = tid / b_tpr;
    const int b_col  = (tid % b_tpr) * 8;
    const int b_row1 = b_row0 + 16;
    const int b_row2 = b_row0 + 32;
    const int b_row3 = b_row0 + 48;

    const int num_k_tiles = (K + K5_BK - 1) / K5_BK;

    #pragma unroll
    for (int s = 0; s < K5_STAGES - 1; s++) {
        half* sA = smem + s * K5_STAGE_SZ;
        half* sB = sA + K5_A_STAGE;
        int k_off = s * K5_BK;
        bool vs = (s < num_k_tiles);
        { int gr = cta_m + a_row0, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row0 * K5_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        if (a_row1 < K5_BM) { int gr = cta_m + a_row1, gc = k_off + a_col; cp_async16_ca_zfill(&sA[a_row1 * K5_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vs && gr < M && gc < K); }
        { int gk = k_off + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K5_B_STRIDE + sw256(b_row0, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K5_B_STRIDE + sw256(b_row1, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row2, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row2 * K5_B_STRIDE + sw256(b_row2, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        { int gk = k_off + b_row3, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row3 * K5_B_STRIDE + sw256(b_row3, b_col)], &B[gk * N + gn], vs && gk < K && gn < N); }
        cp_async_commit();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int cur   = k_tile % K5_STAGES;
        int nxt   = (k_tile + K5_STAGES - 1) % K5_STAGES;
        int k_nxt = (k_tile + K5_STAGES - 1) * K5_BK;
        {
            half* sA = smem + nxt * K5_STAGE_SZ;
            half* sB = sA + K5_A_STAGE;
            bool vn = (k_nxt < K);
            { int gr = cta_m + a_row0, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row0 * K5_A_STRIDE + sw64(a_row0, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            if (a_row1 < K5_BM) { int gr = cta_m + a_row1, gc = k_nxt + a_col; cp_async16_ca_zfill(&sA[a_row1 * K5_A_STRIDE + sw64(a_row1, a_col)], &A[gr * K + gc], vn && gr < M && gc < K); }
            { int gk = k_nxt + b_row0, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row0 * K5_B_STRIDE + sw256(b_row0, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row1, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row1 * K5_B_STRIDE + sw256(b_row1, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row2, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row2 * K5_B_STRIDE + sw256(b_row2, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            { int gk = k_nxt + b_row3, gn = cta_n + b_col; cp_async16_cg_zfill(&sB[b_row3 * K5_B_STRIDE + sw256(b_row3, b_col)], &B[gk * N + gn], vn && gk < K && gn < N); }
            cp_async_commit();
        }
        cp_async_wait<K5_STAGES - 2>();
        __syncthreads();

        const half* sA_cur = smem + cur * K5_STAGE_SZ;
        const half* sB_cur = sA_cur + K5_A_STAGE;

        uint32_t rA[2][K5_WARP_TM][4];
        uint32_t rB[2][K5_WARP_TN][2];

        #pragma unroll
        for (int mi = 0; mi < K5_WARP_TM; mi++) {
            int row = wm_base + mi * K5_MMA_M + (lane_id & 15);
            int col = ((lane_id >> 4) & 1) * 8;
            ldmatrix_x4(rA[0][mi], &sA_cur[row * K5_A_STRIDE + sw64(row, col)]);
        }
        #pragma unroll
        for (int ni = 0; ni < K5_WARP_TN; ni++) {
            int row = lane_id & 15;
            int col = wn_base + ni * K5_MMA_N;
            ldmatrix_x2_trans(rB[0][ni], &sB_cur[row * K5_B_STRIDE + sw256(row, col)]);
        }

        #pragma unroll
        for (int ki = 0; ki < K5_K_TILES; ki++) {
            int cb = ki & 1, nb = cb ^ 1;
            if (ki + 1 < K5_K_TILES) {
                #pragma unroll
                for (int mi = 0; mi < K5_WARP_TM; mi++) {
                    int row = wm_base + mi * K5_MMA_M + (lane_id & 15);
                    int col = (ki + 1) * K5_MMA_K + ((lane_id >> 4) & 1) * 8;
                    ldmatrix_x4(rA[nb][mi], &sA_cur[row * K5_A_STRIDE + sw64(row, col)]);
                }
                #pragma unroll
                for (int ni = 0; ni < K5_WARP_TN; ni++) {
                    int row = (ki + 1) * K5_MMA_K + (lane_id & 15);
                    int col = wn_base + ni * K5_MMA_N;
                    ldmatrix_x2_trans(rB[nb][ni], &sB_cur[row * K5_B_STRIDE + sw256(row, col)]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < K5_WARP_TM; mi++)
                #pragma unroll
                for (int ni = 0; ni < K5_WARP_TN; ni++)
                    mma_m16n8k16(rA[cb][mi], rB[cb][ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < K5_WARP_TM; mi++) {
        int base_row = cta_m + wm_base + mi * K5_MMA_M;
        int row0 = base_row + (lane_id >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < K5_WARP_TN; ni++) {
            int col0 = cta_n + wn_base + ni * K5_MMA_N + (lane_id & 3) * 2;
            if (row0 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row0 * N + col0]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            else if (row0 < M && col0 < N)
                C[row0 * N + col0] = __float2half(acc[mi][ni][0]);
            if (row1 < M && col0 + 1 < N)
                *reinterpret_cast<__half2*>(&C[row1 * N + col0]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            else if (row1 < M && col0 < N)
                C[row1 * N + col0] = __float2half(acc[mi][ni][2]);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    static bool attr_set = false;
    static int best_kernel = -1;

    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, K1_SMEM_BYTES);
        cudaFuncSetAttribute(hgemm_k2, cudaFuncAttributeMaxDynamicSharedMemorySize, K2_SMEM_BYTES);
        cudaFuncSetAttribute(hgemm_k3, cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM_BYTES);
        cudaFuncSetAttribute(hgemm_k4, cudaFuncAttributeMaxDynamicSharedMemorySize, K4_SMEM_BYTES);
        cudaFuncSetAttribute(hgemm_k5, cudaFuncAttributeMaxDynamicSharedMemorySize, K5_SMEM_BYTES);
        attr_set = true;
    }

    dim3 grid1((N + K1_BN - 1) / K1_BN, (M + K1_BM - 1) / K1_BM);
    dim3 grid2((N + K2_BN - 1) / K2_BN, (M + K2_BM - 1) / K2_BM);
    dim3 grid3((N + K3_BN - 1) / K3_BN, (M + K3_BM - 1) / K3_BM);
    dim3 grid4((N + K4_BN - 1) / K4_BN, (M + K4_BM - 1) / K4_BM);
    dim3 grid5((N + K5_BN - 1) / K5_BN, (M + K5_BM - 1) / K5_BM);

    if (best_kernel == -1) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);

        const int WARMUP = 5, BENCH = 20;
        float t[5] = {0.f, 0.f, 0.f, 0.f, 0.f};

        for (int i = 0; i < WARMUP; i++) hgemm_k1<<<grid1, K1_NTHREADS, K1_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) hgemm_k1<<<grid1, K1_NTHREADS, K1_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[0], ev0, ev1); t[0] /= BENCH;

        for (int i = 0; i < WARMUP; i++) hgemm_k2<<<grid2, K2_NTHREADS, K2_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) hgemm_k2<<<grid2, K2_NTHREADS, K2_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[1], ev0, ev1); t[1] /= BENCH;

        for (int i = 0; i < WARMUP; i++) hgemm_k3<<<grid3, K3_NTHREADS, K3_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) hgemm_k3<<<grid3, K3_NTHREADS, K3_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[2], ev0, ev1); t[2] /= BENCH;

        for (int i = 0; i < WARMUP; i++) hgemm_k4<<<grid4, K4_NTHREADS, K4_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) hgemm_k4<<<grid4, K4_NTHREADS, K4_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[3], ev0, ev1); t[3] /= BENCH;

        for (int i = 0; i < WARMUP; i++) hgemm_k5<<<grid5, K5_NTHREADS, K5_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(ev0);
        for (int i = 0; i < BENCH; i++) hgemm_k5<<<grid5, K5_NTHREADS, K5_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&t[4], ev0, ev1); t[4] /= BENCH;

        best_kernel = 0;
        for (int i = 1; i < 5; i++)
            if (t[i] < t[best_kernel]) best_kernel = i;

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }

    switch (best_kernel) {
        case 0: hgemm_k1<<<grid1, K1_NTHREADS, K1_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K); break;
        case 1: hgemm_k2<<<grid2, K2_NTHREADS, K2_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K); break;
        case 2: hgemm_k3<<<grid3, K3_NTHREADS, K3_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K); break;
        case 3: hgemm_k4<<<grid4, K4_NTHREADS, K4_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K); break;
        case 4: hgemm_k5<<<grid5, K5_NTHREADS, K5_SMEM_BYTES>>>(ptr_A, ptr_B, ptr_C, M, N, K); break;
    }
}