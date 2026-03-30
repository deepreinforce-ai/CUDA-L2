#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 32
#define BN 64
#define BK 64
#define WARPS_M 2
#define WARPS_N 4
#define NUM_WARPS (WARPS_M * WARPS_N)
#define BLOCK_THREADS (NUM_WARPS * 32)
#define WMMA_STEPS (BK / WMMA_K)

#define PAD 8
#define SMEM_A_COLS (BK + PAD)
#define SMEM_B_COLS (BK + PAD)

__global__ __launch_bounds__(BLOCK_THREADS, 6)
void hgemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    float* __restrict__ D,
    int M, int N, int K,
    int k_per_split)
{
    const int split_idx = blockIdx.z;
    const int block_m   = blockIdx.x * BM;
    const int block_n   = blockIdx.y * BN;

    if (block_m >= M || block_n >= N) return;

    const int k_start = split_idx * k_per_split;
    const int k_end   = min(k_start + k_per_split, K);

    const int warp_id    = threadIdx.x >> 5;
    const int lane_id    = threadIdx.x & 31;
    const int warp_m_idx = warp_id / WARPS_N;
    const int warp_n_idx = warp_id % WARPS_N;
    const int warp_row   = warp_m_idx * WMMA_M;
    const int warp_col   = warp_n_idx * WMMA_N;
    const int tid        = threadIdx.x;

    const int global_row = block_m + warp_row;
    const int global_col = block_n + warp_col;

    __shared__ half smem_A[2][BM * SMEM_A_COLS];
    __shared__ half smem_B[2][BN * SMEM_B_COLS];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WMMA_STEPS];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[WMMA_STEPS];

    auto load_A = [&](int buf, int k_base) {
        const int linear = tid * 8;
        const int r  = linear / BK;
        const int kk = linear % BK;
        const int gr = block_m + r;
        const int gk = k_base + kk;
        if (r < BM) {
            if (gr < M && gk + 7 < K) {
                float4 v = *reinterpret_cast<const float4*>(A + gr * K + gk);
                *reinterpret_cast<float4*>(&smem_A[buf][r * SMEM_A_COLS + kk]) = v;
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++) {
                    smem_A[buf][r * SMEM_A_COLS + kk + x] =
                        (gr < M && gk + x < K) ? __ldg(A + gr * K + gk + x) : __float2half(0.f);
                }
            }
        }
    };

    auto load_B = [&](int buf, int k_base) {
        #pragma unroll
        for (int iter = 0; iter < 2; iter++) {
            const int linear = (tid + iter * BLOCK_THREADS) * 8;
            const int nn = linear / BK;
            const int kk = linear % BK;
            const int gn = block_n + nn;
            const int gk = k_base + kk;
            if (nn < BN) {
                if (gn < N && gk + 7 < K) {
                    float4 v = *reinterpret_cast<const float4*>(B_col + gn * K + gk);
                    *reinterpret_cast<float4*>(&smem_B[buf][nn * SMEM_B_COLS + kk]) = v;
                } else {
                    #pragma unroll
                    for (int x = 0; x < 8; x++) {
                        smem_B[buf][nn * SMEM_B_COLS + kk + x] =
                            (gn < N && gk + x < K) ? __ldg(B_col + gn * K + gk + x) : __float2half(0.f);
                    }
                }
            }
        }
    };

    load_A(0, k_start);
    load_B(0, k_start);
    __syncthreads();

    int buf = 0;
    const bool active = (global_row < M && global_col < N);

    for (int k = k_start; k < k_end; k += BK) {
        const int next_buf = 1 - buf;

        if (k + BK < k_end) {
            load_A(next_buf, k + BK);
            load_B(next_buf, k + BK);
        }

        if (active) {
            #pragma unroll
            for (int s = 0; s < WMMA_STEPS; s++) {
                wmma::load_matrix_sync(a_frag[s],
                    smem_A[buf] + warp_row * SMEM_A_COLS + s * WMMA_K,
                    SMEM_A_COLS);
                wmma::load_matrix_sync(b_frag[s],
                    smem_B[buf] + warp_col * SMEM_B_COLS + s * WMMA_K,
                    SMEM_B_COLS);
                wmma::mma_sync(c_frag, a_frag[s], b_frag[s], c_frag);
            }
        }

        __syncthreads();
        buf = next_buf;
    }

    if (active) {
        __shared__ float frag_smem[NUM_WARPS * WMMA_M * WMMA_N];
        float* warp_out = frag_smem + warp_id * WMMA_M * WMMA_N;

        wmma::store_matrix_sync(warp_out, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            const int r = global_row + i / WMMA_N;
            const int c_ = global_col + i % WMMA_N;
            if (r < M && c_ < N)
                atomicAdd(&D[r * N + c_], warp_out[i]);
        }
    }
}

__global__ void fused_convert_zero(
    float* __restrict__ src,
    half*  __restrict__ dst,
    int count)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < count) {
        float4 v = *reinterpret_cast<float4*>(src + idx);
        *reinterpret_cast<half2*>(dst + idx)     = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<half2*>(dst + idx + 2) = __floats2half2_rn(v.z, v.w);
        *reinterpret_cast<float4*>(src + idx)    = make_float4(0.f, 0.f, 0.f, 0.f);
    } else {
        for (int i = idx; i < count && i < idx + 4; i++) {
            dst[i] = __float2half(src[i]);
            src[i] = 0.f;
        }
    }
}

static float* g_accum  = nullptr;
static size_t g_accsz  = 0;
static bool   g_zeroed = false;

static float* get_accum(size_t bytes) {
    if (bytes > g_accsz) {
        if (g_accum) cudaFree(g_accum);
        cudaMalloc(&g_accum, bytes);
        g_accsz  = bytes;
        g_zeroed = false;
    }
    return g_accum;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* A     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C           = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int MN = M * N;
    float* D_accum = get_accum((size_t)MN * sizeof(float));

    if (!g_zeroed) {
        cudaMemsetAsync(D_accum, 0, (size_t)MN * sizeof(float), 0);
        g_zeroed = true;
    }

    const int SPLIT_K = 64;
    const int k_per   = K / SPLIT_K;

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
    dim3 block(BLOCK_THREADS);

    hgemm_kernel<<<grid, block>>>(A, B_col, D_accum, M, N, K, k_per);

    const int cvt_threads = 256;
    const int cvt_blocks  = (MN / 4 + cvt_threads - 1) / cvt_threads;
    fused_convert_zero<<<cvt_blocks, cvt_threads>>>(D_accum, C, MN);
}