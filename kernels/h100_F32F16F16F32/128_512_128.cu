#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm("{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__device__ __forceinline__ void cp_async16_ca(void* dst, const void* src) {
    uint32_t addr = smem_u32addr(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

#define BM 64
#define BN 64
#define BK 64
#define WROW_TILES 2
#define WCOL_TILES 2
#define WARP_ROWS 2
#define WARP_COLS 2
#define THREADS 128

#define PAD_A 8
#define PAD_B 8
#define A_STR (BK + PAD_A)
#define B_STR (BN + PAD_B)

__global__ void __launch_bounds__(THREADS, 4)
hgemm_64x64_opt(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    if (block_row >= M || block_col >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_row = warp_id >> 1;
    const int warp_col = warp_id & 1;
    const int wt_row = warp_row * (WROW_TILES * WMMA_M);
    const int wt_col = warp_col * (WCOL_TILES * WMMA_N);

    __shared__ half smA[2][BM * A_STR];
    __shared__ half smB[2][BK * B_STR];

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WROW_TILES][WCOL_TILES];
    #pragma unroll
    for (int i = 0; i < WROW_TILES; i++)
        #pragma unroll
        for (int j = 0; j < WCOL_TILES; j++)
            fill_fragment(acc[i][j], 0.0f);

    const int num_tiles = (K + BK - 1) / BK;

    auto load_A_tile = [&](int buf, int tk) __attribute__((always_inline)) {
        const int per = (BM * BK) / THREADS;
        #pragma unroll
        for (int v = 0; v < per; v += 8) {
            int lin = tid * per + v;
            int row = lin >> 6;
            int col = lin & 63;
            int gr = block_row + row;
            int gc = tk * BK + col;
            half* dst = smA[buf] + row * A_STR + col;
            if (__builtin_expect(gr < M && gc + 7 < K, 1))
                cp_async16_ca(dst, &A[gr * K + gc]);
            else
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    dst[x] = (gr < M && gc + x < K) ? __ldg(&A[gr * K + gc + x]) : __float2half(0.f);
        }
    };

    auto load_B_tile = [&](int buf, int tk) __attribute__((always_inline)) {
        const int per = (BK * BN) / THREADS;
        #pragma unroll
        for (int v = 0; v < per; v += 8) {
            int lin = tid * per + v;
            int kl = lin >> 6;
            int nl = lin & 63;
            int gk = tk * BK + kl;
            int gn = block_col + nl;
            half* dst = smB[buf] + kl * B_STR + nl;
            if (__builtin_expect(gk < K && gn + 7 < N, 1))
                cp_async16_ca(dst, &B[gk * N + gn]);
            else
                #pragma unroll
                for (int x = 0; x < 8; x++) {
                    int gnn = gn + x;
                    dst[x] = (gk < K && gnn < N) ? __ldg(&B[gk * N + gnn]) : __float2half(0.f);
                }
        }
    };

    load_A_tile(0, 0);
    load_B_tile(0, 0);
    cp_async_commit();

    if (num_tiles > 1) {
        load_A_tile(1, 1);
        load_B_tile(1, 1);
        cp_async_commit();
    }

    #pragma unroll 1
    for (int t = 0; t < num_tiles; t++) {
        if (t < num_tiles - 1) cp_async_wait_one();
        else cp_async_wait_all();
        __syncthreads();

        int cur = t & 1;
        int nxt2 = t + 2;
        if (nxt2 < num_tiles) {
            load_A_tile(nxt2 & 1, nxt2);
            load_B_tile(nxt2 & 1, nxt2);
            cp_async_commit();
        }

        #pragma unroll
        for (int ki = 0; ki < BK / WMMA_K; ki++) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[WROW_TILES];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag[WCOL_TILES];

            #pragma unroll
            for (int ti = 0; ti < WROW_TILES; ti++)
                load_matrix_sync(a_frag[ti],
                    smA[cur] + (wt_row + ti * WMMA_M) * A_STR + ki * WMMA_K, A_STR);
            #pragma unroll
            for (int tj = 0; tj < WCOL_TILES; tj++)
                load_matrix_sync(b_frag[tj],
                    smB[cur] + ki * WMMA_K * B_STR + (wt_col + tj * WMMA_N), B_STR);
            #pragma unroll
            for (int ti = 0; ti < WROW_TILES; ti++)
                #pragma unroll
                for (int tj = 0; tj < WCOL_TILES; tj++)
                    mma_sync(acc[ti][tj], a_frag[ti], b_frag[tj], acc[ti][tj]);
        }
    }

    cp_async_wait_all();
    __syncthreads();

    float* warp_scratch = reinterpret_cast<float*>(smA[0]) + warp_id * (WMMA_M * WMMA_N);

    #pragma unroll
    for (int ti = 0; ti < WROW_TILES; ti++) {
        int c_row = block_row + wt_row + ti * WMMA_M;
        #pragma unroll
        for (int tj = 0; tj < WCOL_TILES; tj++) {
            int c_col = block_col + wt_col + tj * WMMA_N;
            store_matrix_sync(warp_scratch, acc[ti][tj], WMMA_N, mem_row_major);
            __syncwarp();
            if (c_row < M && c_col < N) {
                int r = lane_id >> 1;
                int c_off = (lane_id & 1) << 3;
                int out_r = c_row + r;
                int out_c = c_col + c_off;
                if (out_r < M && out_c + 7 < N) {
                    float* src = warp_scratch + r * WMMA_N + c_off;
                    __half2 h01 = __float22half2_rn(make_float2(src[0], src[1]));
                    __half2 h23 = __float22half2_rn(make_float2(src[2], src[3]));
                    __half2 h45 = __float22half2_rn(make_float2(src[4], src[5]));
                    __half2 h67 = __float22half2_rn(make_float2(src[6], src[7]));
                    half* dst = &C[out_r * N + out_c];
                    *reinterpret_cast<__half2*>(dst + 0) = h01;
                    *reinterpret_cast<__half2*>(dst + 2) = h23;
                    *reinterpret_cast<__half2*>(dst + 4) = h45;
                    *reinterpret_cast<__half2*>(dst + 6) = h67;
                } else if (out_r < M) {
                    float* src = warp_scratch + r * WMMA_N + c_off;
                    #pragma unroll
                    for (int x = 0; x < 8; x++)
                        if (out_c + x < N)
                            C[out_r * N + out_c + x] = __float2half(src[x]);
                }
            }
        }
    }
}

#define BM2 128
#define BN2 128
#define BK2 128
#define WROW2 4
#define WCOL2 2
#define WR2 2
#define WC2 4
#define THREADS2 256
#define PAD_A2 8
#define PAD_B2 8
#define A_STR2 (BK2 + PAD_A2)
#define B_STR2 (BN2 + PAD_B2)

__global__ void __launch_bounds__(THREADS2, 1)
hgemm_128x128_single(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ char smem2_raw[];

    const int block_row = blockIdx.y * BM2;
    const int block_col = blockIdx.x * BN2;
    if (block_row >= M || block_col >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_row = warp_id / WC2;
    const int warp_col = warp_id % WC2;
    const int wt_row = warp_row * (WROW2 * WMMA_M);
    const int wt_col = warp_col * (WCOL2 * WMMA_N);

    half* smA2 = reinterpret_cast<half*>(smem2_raw);
    half* smB2 = smA2 + BM2 * A_STR2;

    {
        const int per = (BM2 * BK2) / THREADS2;
        #pragma unroll
        for (int v = 0; v < per; v += 8) {
            int lin = tid * per + v;
            int row = lin / BK2;
            int col = lin % BK2;
            int gr = block_row + row;
            int gc = col;
            half* dst = smA2 + row * A_STR2 + col;
            if (__builtin_expect(gr < M && gc + 7 < K, 1))
                cp_async16_ca(dst, &A[gr * K + gc]);
            else
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    dst[x] = (gr < M && gc + x < K) ? __ldg(&A[gr * K + gc + x]) : __float2half(0.f);
        }
    }
    {
        const int per = (BK2 * BN2) / THREADS2;
        #pragma unroll
        for (int v = 0; v < per; v += 8) {
            int lin = tid * per + v;
            int kl = lin / BN2;
            int nl = lin % BN2;
            int gk = kl;
            int gn = block_col + nl;
            half* dst = smB2 + kl * B_STR2 + nl;
            if (__builtin_expect(gk < K && gn + 7 < N, 1))
                cp_async16_ca(dst, &B[gk * N + gn]);
            else
                #pragma unroll
                for (int x = 0; x < 8; x++) {
                    int gnn = gn + x;
                    dst[x] = (gk < K && gnn < N) ? __ldg(&B[gk * N + gnn]) : __float2half(0.f);
                }
        }
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WROW2][WCOL2];
    #pragma unroll
    for (int i = 0; i < WROW2; i++)
        #pragma unroll
        for (int j = 0; j < WCOL2; j++)
            fill_fragment(acc[i][j], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < BK2 / WMMA_K; ki++) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[WROW2];
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag[WCOL2];
        #pragma unroll
        for (int ti = 0; ti < WROW2; ti++)
            load_matrix_sync(a_frag[ti],
                smA2 + (wt_row + ti * WMMA_M) * A_STR2 + ki * WMMA_K, A_STR2);
        #pragma unroll
        for (int tj = 0; tj < WCOL2; tj++)
            load_matrix_sync(b_frag[tj],
                smB2 + ki * WMMA_K * B_STR2 + (wt_col + tj * WMMA_N), B_STR2);
        #pragma unroll
        for (int ti = 0; ti < WROW2; ti++)
            #pragma unroll
            for (int tj = 0; tj < WCOL2; tj++)
                mma_sync(acc[ti][tj], a_frag[ti], b_frag[tj], acc[ti][tj]);
    }

    __syncthreads();

    float* scratch2 = reinterpret_cast<float*>(smA2);
    float* ws2 = scratch2 + warp_id * (WMMA_M * WMMA_N);

    #pragma unroll
    for (int ti = 0; ti < WROW2; ti++) {
        int c_row = block_row + wt_row + ti * WMMA_M;
        #pragma unroll
        for (int tj = 0; tj < WCOL2; tj++) {
            int c_col = block_col + wt_col + tj * WMMA_N;
            store_matrix_sync(ws2, acc[ti][tj], WMMA_N, mem_row_major);
            __syncwarp();
            if (c_row < M && c_col < N) {
                int r = lane_id >> 1;
                int c_off = (lane_id & 1) << 3;
                int out_r = c_row + r;
                int out_c = c_col + c_off;
                if (out_r < M && out_c + 7 < N) {
                    float* src = ws2 + r * WMMA_N + c_off;
                    __half2 h01 = __float22half2_rn(make_float2(src[0], src[1]));
                    __half2 h23 = __float22half2_rn(make_float2(src[2], src[3]));
                    __half2 h45 = __float22half2_rn(make_float2(src[4], src[5]));
                    __half2 h67 = __float22half2_rn(make_float2(src[6], src[7]));
                    half* dst = &C[out_r * N + out_c];
                    *reinterpret_cast<__half2*>(dst + 0) = h01;
                    *reinterpret_cast<__half2*>(dst + 2) = h23;
                    *reinterpret_cast<__half2*>(dst + 4) = h45;
                    *reinterpret_cast<__half2*>(dst + 6) = h67;
                } else if (out_r < M) {
                    float* src = ws2 + r * WMMA_N + c_off;
                    #pragma unroll
                    for (int x = 0; x < 8; x++)
                        if (out_c + x < N)
                            C[out_r * N + out_c + x] = __float2half(src[x]);
                }
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    int M = (int)a.size(0);
    int K = (int)a.size(1);
    int N = (int)b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

    {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(THREADS);
        hgemm_64x64_opt<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        return;
    }

    {
        size_t smem_sz = (size_t)(BM2 * A_STR2 + BK2 * B_STR2) * sizeof(half);
        cudaFuncSetAttribute(hgemm_128x128_single,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_sz);
        dim3 grid2((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        dim3 block2(THREADS2);
        hgemm_128x128_single<<<grid2, block2, smem_sz>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    }
}