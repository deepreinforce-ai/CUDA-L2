#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define BM 64
#define BN 64
#define BK 32
#define SPLIT_K 16
#define BLOCK_THREADS 128

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define SMEM_A_STRIDE (BK + 8)
#define SMEM_B_STRIDE (BN + 8)

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
hgemm_splitk_v4(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ workspace,
    int M, int N, int K, int k_per_split
) {
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int bk_split = blockIdx.z;

    const int bm_start = bm * BM;
    const int bn_start = bn * BN;
    const int k_start = bk_split * k_per_split;
    const int k_end = min(k_start + k_per_split, K);

    if (bm_start >= M || bn_start >= N || k_start >= K) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    const int wm_start = warp_row * 32;
    const int wn_start = warp_col * 32;

    __shared__ half smem_A[2][BM][SMEM_A_STRIDE];
    __shared__ half smem_B[2][BK][SMEM_B_STRIDE];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b[2];

    wmma::fill_fragment(acc[0][0], 0.f);
    wmma::fill_fragment(acc[0][1], 0.f);
    wmma::fill_fragment(acc[1][0], 0.f);
    wmma::fill_fragment(acc[1][1], 0.f);

    const int a_f4_0 = tid;
    const int a_f4_1 = tid + 128;
    const int a_r0 = a_f4_0 / 4,  a_c0 = (a_f4_0 % 4) * 8;
    const int a_r1 = a_f4_1 / 4,  a_c1 = (a_f4_1 % 4) * 8;

    const int b_f4_0 = tid;
    const int b_f4_1 = tid + 128;
    const int b_r0 = b_f4_0 / 8,  b_c0 = (b_f4_0 % 8) * 8;
    const int b_r1 = b_f4_1 / 8,  b_c1 = (b_f4_1 % 8) * 8;

    auto load_tile = [&](int k_tile, int st) {
        {
            int gm0 = bm_start + a_r0, gk0 = k_tile + a_c0;
            int gm1 = bm_start + a_r1, gk1 = k_tile + a_c1;

            if (gm0 < M && gk0 + 7 < K) {
                *reinterpret_cast<float4*>(&smem_A[st][a_r0][a_c0]) =
                    *reinterpret_cast<const float4*>(&A[gm0 * K + gk0]);
            } else {
                half* dst = &smem_A[st][a_r0][a_c0];
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    dst[i] = (gm0 < M && gk0 + i < K) ? A[gm0 * K + gk0 + i] : __float2half(0.f);
            }
            if (gm1 < M && gk1 + 7 < K) {
                *reinterpret_cast<float4*>(&smem_A[st][a_r1][a_c1]) =
                    *reinterpret_cast<const float4*>(&A[gm1 * K + gk1]);
            } else {
                half* dst = &smem_A[st][a_r1][a_c1];
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    dst[i] = (gm1 < M && gk1 + i < K) ? A[gm1 * K + gk1 + i] : __float2half(0.f);
            }
        }
        {
            int gk0 = k_tile + b_r0, gn0 = bn_start + b_c0;
            int gk1 = k_tile + b_r1, gn1 = bn_start + b_c1;

            if (gk0 < K && gn0 + 7 < N) {
                *reinterpret_cast<float4*>(&smem_B[st][b_r0][b_c0]) =
                    *reinterpret_cast<const float4*>(&B[gk0 * N + gn0]);
            } else {
                half* dst = &smem_B[st][b_r0][b_c0];
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    dst[i] = (gk0 < K && gn0 + i < N) ? B[gk0 * N + gn0 + i] : __float2half(0.f);
            }
            if (gk1 < K && gn1 + 7 < N) {
                *reinterpret_cast<float4*>(&smem_B[st][b_r1][b_c1]) =
                    *reinterpret_cast<const float4*>(&B[gk1 * N + gn1]);
            } else {
                half* dst = &smem_B[st][b_r1][b_c1];
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    dst[i] = (gk1 < K && gn1 + i < N) ? B[gk1 * N + gn1 + i] : __float2half(0.f);
            }
        }
    };

    int num_steps = (k_end - k_start + BK - 1) / BK;
    if (num_steps == 0) return;

    load_tile(k_start, 0);
    __syncthreads();

    for (int step = 0; step < num_steps; step++) {
        int cur_st = step & 1;
        int nxt_st = 1 - cur_st;
        int k_tile = k_start + step * BK;

        if (step + 1 < num_steps) {
            load_tile(k_tile + BK, nxt_st);
        }

        #pragma unroll
        for (int ki = 0; ki < BK; ki += WMMA_K) {
            wmma::load_matrix_sync(frag_a[0], &smem_A[cur_st][wm_start][ki],      SMEM_A_STRIDE);
            wmma::load_matrix_sync(frag_a[1], &smem_A[cur_st][wm_start+16][ki],   SMEM_A_STRIDE);
            wmma::load_matrix_sync(frag_b[0], &smem_B[cur_st][ki][wn_start],      SMEM_B_STRIDE);
            wmma::load_matrix_sync(frag_b[1], &smem_B[cur_st][ki][wn_start+16],   SMEM_B_STRIDE);

            wmma::mma_sync(acc[0][0], frag_a[0], frag_b[0], acc[0][0]);
            wmma::mma_sync(acc[0][1], frag_a[0], frag_b[1], acc[0][1]);
            wmma::mma_sync(acc[1][0], frag_a[1], frag_b[0], acc[1][0]);
            wmma::mma_sync(acc[1][1], frag_a[1], frag_b[1], acc[1][1]);
        }

        __syncthreads();
    }

    __shared__ float smem_out[BM][BN];

    wmma::store_matrix_sync(&smem_out[wm_start][wn_start],       acc[0][0], BN, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_out[wm_start][wn_start+16],    acc[0][1], BN, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_out[wm_start+16][wn_start],    acc[1][0], BN, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_out[wm_start+16][wn_start+16], acc[1][1], BN, wmma::mem_row_major);
    __syncthreads();

    const long ws_base = (long)bk_split * M * N;
    for (int idx = tid; idx < (BM * BN / 2); idx += BLOCK_THREADS) {
        int elem = idx * 2;
        int r = elem / BN;
        int c = elem % BN;
        int gm = bm_start + r;
        int gn = bn_start + c;
        if (gm < M && gn + 1 <= N) {
            float2 val = make_float2(smem_out[r][c], smem_out[r][c+1]);
            *reinterpret_cast<float2*>(&workspace[ws_base + (long)gm * N + gn]) = val;
        }
    }
}

__global__ void reduce_splitk_v4(
    const float* __restrict__ workspace,
    half* __restrict__ C,
    int MN, int split_k
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 >= MN) {
        for (int i = 0; i < 4 && idx + i < MN; i++) {
            float sum = 0.f;
            for (int s = 0; s < split_k; s++)
                sum += workspace[(long)s * MN + idx + i];
            C[idx + i] = __float2half(sum);
        }
        return;
    }

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    #pragma unroll
    for (int s = 0; s < SPLIT_K; s++) {
        long base = (long)s * MN + idx;
        float4 v = *reinterpret_cast<const float4*>(&workspace[base]);
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }

    half2 h01 = __float22half2_rn(make_float2(s0, s1));
    half2 h23 = __float22half2_rn(make_float2(s2, s3));
    *reinterpret_cast<half2*>(&C[idx])   = h01;
    *reinterpret_cast<half2*>(&C[idx+2]) = h23;
}

static float* g_workspace = nullptr;
static size_t g_workspace_bytes = 0;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int split_k = SPLIT_K;
    const int k_per_split = (K + split_k - 1) / split_k;

    size_t ws_bytes = (size_t)split_k * M * N * sizeof(float);
    if (g_workspace == nullptr || g_workspace_bytes < ws_bytes) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, ws_bytes);
        g_workspace_bytes = ws_bytes;
    }

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, split_k);
    dim3 block(BLOCK_THREADS);

    hgemm_splitk_v4<<<grid, block>>>(
        ptr_A, ptr_B, g_workspace, M, N, K, k_per_split
    );

    const int MN = M * N;
    const int reduce_threads = 256;
    const int reduce_blocks = (MN / 4 + reduce_threads - 1) / reduce_threads;
    reduce_splitk_v4<<<reduce_blocks, reduce_threads>>>(g_workspace, ptr_C, MN, split_k);
}