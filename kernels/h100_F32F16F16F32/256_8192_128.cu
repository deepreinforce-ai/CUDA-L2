#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

#define BM 64
#define BN 128
#define BK 32
#define NUM_WARPS 4
#define BLOCK_THREADS 128
#define WARPS_M 2
#define WARPS_N 2
#define WARP_M 32
#define WARP_N 64
#define N_MMA_M 2
#define N_MMA_N 4
#define N_MMA_K 2
#define SMEM_A_STRIDE (BK + 8)
#define SMEM_B_STRIDE (BN + 8)
#define NUM_STAGES 3

__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    uint32_t dst_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"(src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ void __launch_bounds__(128, 4)
hgemm_pipeline_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K)
{
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    const int warp_m = wid / WARPS_N;
    const int warp_n = wid % WARPS_N;

    __shared__ half smA[NUM_STAGES][BM][SMEM_A_STRIDE];
    __shared__ half smB[NUM_STAGES][BK][SMEM_B_STRIDE];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC[N_MMA_M][N_MMA_N];
    #pragma unroll
    for (int mi = 0; mi < N_MMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < N_MMA_N; ni++)
            wmma::fill_fragment(fragC[mi][ni], 0.f);

    const int A_row_base = bm * BM;
    const int B_col_base = bn * BN;
    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A_async = [&](int s, int kt) {
        const int k_start = kt * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int slot = tid + i * 128;
            int row  = slot >> 2;
            int col8 = slot & 3;
            int gr = A_row_base + row;
            int gk = k_start + col8 * 8;
            if (gr < M && gk + 7 < K) {
                cp_async_cg_16(&smA[s][row][col8*8], &A[gr*K+gk]);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    smA[s][row][col8*8+e] = (gr<M && gk+e<K) ? A[gr*K+gk+e] : __float2half(0.f);
            }
        }
    };

    auto load_B_async = [&](int s, int kt) {
        const int k_start = kt * BK;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int slot    = tid + i * 128;
            int k_local = slot >> 4;
            int n8      = slot & 15;
            int gk = k_start + k_local;
            int gn = B_col_base + n8 * 8;
            if (gk < K && gn + 7 < N) {
                cp_async_cg_16(&smB[s][k_local][n8*8], &B[gk*N+gn]);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    smB[s][k_local][n8*8+e] = (gk<K && gn+e<N) ? B[gk*N+gn+e] : __float2half(0.f);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < num_k_tiles) {
            load_A_async(s, s);
            load_B_async(s, s);
        }
        cp_async_commit();
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        cp_async_wait<NUM_STAGES - 2>();
        __syncthreads();

        int cur_s = kt % NUM_STAGES;

        int nxt_kt = kt + (NUM_STAGES - 1);
        if (nxt_kt < num_k_tiles) {
            int nxt_s = nxt_kt % NUM_STAGES;
            load_A_async(nxt_s, nxt_kt);
            load_B_async(nxt_s, nxt_kt);
        }
        cp_async_commit();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fragA[N_MMA_M];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fragB_w[N_MMA_N];

        const int warp_row_base = warp_m * WARP_M;
        const int warp_col_base = warp_n * WARP_N;

        #pragma unroll
        for (int ks = 0; ks < N_MMA_K; ks++) {
            int k_off = ks * 16;
            #pragma unroll
            for (int mi = 0; mi < N_MMA_M; mi++)
                wmma::load_matrix_sync(fragA[mi],
                    &smA[cur_s][warp_row_base + mi*16][k_off],
                    SMEM_A_STRIDE);
            #pragma unroll
            for (int ni = 0; ni < N_MMA_N; ni++)
                wmma::load_matrix_sync(fragB_w[ni],
                    &smB[cur_s][k_off][warp_col_base + ni*16],
                    SMEM_B_STRIDE);
            #pragma unroll
            for (int mi = 0; mi < N_MMA_M; mi++)
                #pragma unroll
                for (int ni = 0; ni < N_MMA_N; ni++)
                    wmma::mma_sync(fragC[mi][ni], fragA[mi], fragB_w[ni], fragC[mi][ni]);
        }
    }

    cp_async_wait<0>();
    __syncthreads();

    __shared__ float smTmp[NUM_WARPS][16][16];

    const int c_row_base = bm * BM + warp_m * WARP_M;
    const int c_col_base = bn * BN + warp_n * WARP_N;

    #pragma unroll
    for (int mi = 0; mi < N_MMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N_MMA_N; ni++) {
            wmma::store_matrix_sync(&smTmp[wid][0][0], fragC[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();

            int out_row_base = c_row_base + mi * 16;
            int out_col_base = c_col_base + ni * 16;

            #pragma unroll
            for (int e = 0; e < 8; e++) {
                int idx = lane + e * 32;
                int r   = idx >> 4;
                int col = idx & 15;
                int gr  = out_row_base + r;
                int gc  = out_col_base + col;
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(smTmp[wid][r][col]);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BLOCK_THREADS);

    hgemm_pipeline_kernel<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
}