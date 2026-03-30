#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>

using namespace nvcuda;

#define SPLIT_K 32
#define BM 64
#define BN 128
#define BK 32
#define NWARPS 8
#define BK_PAD (BK + 8)
#define BN_PAD (BN + 8)

#define STAGES 2

static float* g_workspace = nullptr;
static size_t g_workspace_size = 0;

static void ensure_workspace(size_t size) {
    if (size > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, size);
        g_workspace_size = size;
    }
}

__global__ __launch_bounds__(256, 2)
void hgemm_splitk_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ workspace,
    int M, int N, int K,
    int K_per_split
) {
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int bk = blockIdx.z;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int k_start = bk * K_per_split;
    const int k_end   = min(k_start + K_per_split, K);

    const int warp_m   = warp_id / 2;
    const int warp_n   = warp_id % 2;
    const int warp_row = warp_m * 16;
    const int warp_col = warp_n * 64;

    __shared__ half smA[STAGES][BM][BK_PAD];
    __shared__ half smB[STAGES][BK][BN_PAD];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(acc[i], 0.0f);

    const int row_base = bm * BM;
    const int col_base = bn * BN;

    int num_tiles = (k_end - k_start + BK - 1) / BK;

    auto load_tile = [&](int tile_idx, int stage) {
        int k_off = k_start + tile_idx * BK;

        {
            int r = tid / 4;
            int c = (tid % 4) * 8;
            int gr = row_base + r;
            int gk = k_off + c;
            uint32_t dst = __cvta_generic_to_shared(&smA[stage][r][c]);
            if (gr < M && gk + 7 < K) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst), "l"((const void*)(A + gr * K + gk)));
            } else {
                if (gr < M) {
                    for (int x = 0; x < 8; x++)
                        smA[stage][r][c+x] = (gk+x < K) ? A[gr*K + gk+x] : __float2half(0.f);
                } else {
                    *reinterpret_cast<float4*>(&smA[stage][r][c]) = make_float4(0,0,0,0);
                }
            }
        }

        {
            int r = tid / 8;
            int c = (tid % 8) * 16;
            int gk = k_off + r;
            int gn0 = col_base + c;
            int gn1 = gn0 + 8;

            uint32_t dst0 = __cvta_generic_to_shared(&smB[stage][r][c]);
            uint32_t dst1 = __cvta_generic_to_shared(&smB[stage][r][c+8]);

            if (gk < K && gn0 + 7 < N) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst0), "l"((const void*)(B + gk * N + gn0)));
            } else {
                if (gk < K) {
                    for (int x = 0; x < 8; x++)
                        smB[stage][r][c+x] = (gn0+x < N) ? B[gk*N + gn0+x] : __float2half(0.f);
                } else {
                    *reinterpret_cast<float4*>(&smB[stage][r][c]) = make_float4(0,0,0,0);
                }
            }

            if (gk < K && gn1 + 7 < N) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst1), "l"((const void*)(B + gk * N + gn1)));
            } else {
                if (gk < K) {
                    for (int x = 0; x < 8; x++)
                        smB[stage][r][c+8+x] = (gn1+x < N) ? B[gk*N + gn1+x] : __float2half(0.f);
                } else {
                    *reinterpret_cast<float4*>(&smB[stage][r][c+8]) = make_float4(0,0,0,0);
                }
            }
        }
    };

    if (num_tiles > 0) {
        load_tile(0, 0);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int tile = 0; tile < num_tiles; tile++) {
        int next_stage = (tile + 1) % STAGES;
        int cur_stage  = tile % STAGES;

        if (tile + 1 < num_tiles) {
            load_tile(tile + 1, next_stage);
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, &smA[cur_stage][warp_row][ks * 16], BK_PAD);

            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                wmma::load_matrix_sync(fb, &smB[cur_stage][ks * 16][warp_col + ni * 16], BN_PAD);
                wmma::mma_sync(acc[ni], fa, fb, acc[ni]);
            }
        }
    }

    float* ws = workspace + (size_t)bk * M * N;
    int out_row = row_base + warp_row;
    int out_col = col_base + warp_col;

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        if (out_row < M && out_col + ni * 16 < N) {
            wmma::store_matrix_sync(
                ws + out_row * N + out_col + ni * 16,
                acc[ni],
                N,
                wmma::mem_row_major
            );
        }
    }
}

__global__ void splitk_reduce_kernel(
    const float* __restrict__ workspace,
    half* __restrict__ C,
    int total,
    int split_k
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= total) return;

    float4 sums = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < split_k; k++) {
        const float* base = workspace + (size_t)k * total + idx;
        float4 v = *reinterpret_cast<const float4*>(base);
        sums.x += v.x;
        sums.y += v.y;
        sums.z += v.z;
        sums.w += v.w;
    }

    half2 h01 = __float22half2_rn(make_float2(sums.x, sums.y));
    half2 h23 = __float22half2_rn(make_float2(sums.z, sums.w));
    *reinterpret_cast<half2*>(C + idx)     = h01;
    *reinterpret_cast<half2*>(C + idx + 2) = h23;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int split_k    = SPLIT_K;
    const int K_per_split = (K + split_k - 1) / split_k;

    size_t ws_bytes = (size_t)split_k * M * N * sizeof(float);
    ensure_workspace(ws_bytes);

    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, split_k);
    dim3 block(256);

    hgemm_splitk_kernel<<<grid, block>>>(
        ptr_A, ptr_B, g_workspace,
        M, N, K, K_per_split
    );

    int total_elems = M * N;
    int red_threads = 256;
    int red_blocks  = (total_elems / 4 + red_threads - 1) / red_threads;

    splitk_reduce_kernel<<<red_blocks, red_threads>>>(
        g_workspace, ptr_C, total_elems, split_k
    );
}