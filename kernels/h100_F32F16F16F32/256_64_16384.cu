#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

#define BM          32
#define BN          64
#define BK_STEP     128
#define WMMA_M      16
#define WMMA_N      16
#define WMMA_K      16
#define K_STEP_TILES (BK_STEP / WMMA_K)
#define WARP_TILE_M  16
#define WARP_TILE_N  32
#define WARP_TILES_N  2
#define NUM_WARPS    4
#define BLOCK_THREADS 128
#define K_SPLITS     64
#define SMEM_A_ROWS  BM
#define SMEM_A_COLS  (BK_STEP + 8)
#define SMEM_B_ROWS  BK_STEP
#define SMEM_B_COLS  (BN + 8)

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    uint32_t dst_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"(src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__global__ __launch_bounds__(BLOCK_THREADS, 3)
void hgemm_phase1(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ workspace,
    int M, int N, int K, int k_per_split)
{
    const int bm = blockIdx.x;
    const int bk = blockIdx.y;

    const int m_off   = bm * BM;
    const int k_start = bk * k_per_split;
    const int k_end   = min(k_start + k_per_split, K);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int warp_m     = warp_id >> 1;
    const int warp_n     = warp_id & 1;
    const int warp_m_off = warp_m * WARP_TILE_M;
    const int warp_n_off = warp_n * WARP_TILE_N;

    __shared__ __half smA[BM][BK_STEP + 8];
    __shared__ __half smB[BK_STEP][BN + 8];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILES_N];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    for (int k_cur = k_start; k_cur < k_end; k_cur += BK_STEP) {

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear = tid + i * BLOCK_THREADS;
            int row    = linear / (BK_STEP / 8);
            int cg     = linear % (BK_STEP / 8);
            int gm = m_off + row;
            int gk = k_cur + cg * 8;
            __half* dst = &smA[row][cg * 8];
            if (gm < M && gk + 7 < K) {
                cp_async16(dst, A + gm * K + gk);
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    dst[x] = (gm < M && gk + x < K) ? A[gm * K + gk + x] : __float2half(0.f);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int linear  = tid + i * BLOCK_THREADS;
            int k_local = linear / (BN / 8);
            int cg      = linear % (BN / 8);
            int gk = k_cur + k_local;
            int gn = cg * 8;
            __half* dst = &smB[k_local][gn];
            if (gk < K && gn + 7 < N) {
                cp_async16(dst, B + gk * N + gn);
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    dst[x] = (gk < K && gn + x < N) ? B[gk * N + gn + x] : __float2half(0.f);
            }
        }

        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < K_STEP_TILES; k++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> fb0, fb1;

            wmma::load_matrix_sync(fa,  &smA[warp_m_off][k * WMMA_K],           BK_STEP + 8);
            wmma::load_matrix_sync(fb0, &smB[k * WMMA_K][warp_n_off],           BN + 8);
            wmma::load_matrix_sync(fb1, &smB[k * WMMA_K][warp_n_off + WMMA_N],  BN + 8);

            wmma::mma_sync(acc[0], fa, fb0, acc[0]);
            wmma::mma_sync(acc[1], fa, fb1, acc[1]);
        }

        __syncthreads();
    }

    __shared__ float smOut[NUM_WARPS][WARP_TILE_M][WARP_TILE_N + 4];

    wmma::store_matrix_sync(&smOut[warp_id][0][0],      acc[0], WARP_TILE_N + 4, wmma::mem_row_major);
    wmma::store_matrix_sync(&smOut[warp_id][0][WMMA_N], acc[1], WARP_TILE_N + 4, wmma::mem_row_major);
    __syncthreads();

    float* ws_slice = workspace + (size_t)bk * M * N;

    #pragma unroll
    for (int i = lane; i < WARP_TILE_M * WARP_TILE_N; i += 32) {
        int lm = i / WARP_TILE_N;
        int ln = i % WARP_TILE_N;
        int gm = m_off + warp_m_off + lm;
        int gn = warp_n_off + ln;
        if (gm < M && gn < N) {
            ws_slice[gm * N + gn] = smOut[warp_id][lm][ln];
        }
    }
}

__global__ __launch_bounds__(256)
void hgemm_phase2(
    const float* __restrict__ workspace,
    __half*      __restrict__ C,
    int M, int N, int k_splits)
{
    const int total = M * N;
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float sum = 0.f;
    const size_t stride = total;

    #pragma unroll 8
    for (int s = 0; s < k_splits; s++) {
        sum += workspace[s * stride + idx];
    }

    C[idx] = __float2half(sum);
}

static float* g_ws    = nullptr;
static size_t g_ws_sz = 0;

static float* get_workspace(size_t bytes) {
    if (bytes > g_ws_sz) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, bytes);
        g_ws_sz = bytes;
    }
    return g_ws;
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* ptr_A = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* ptr_B = reinterpret_cast<const __half*>(b.data_ptr());
    __half*       ptr_C = reinterpret_cast<__half*>(c.data_ptr());

    const int total       = M * N;
    const int k_splits    = K_SPLITS;
    const int k_per_split = (K + k_splits - 1) / k_splits;

    float* workspace = get_workspace((size_t)k_splits * total * sizeof(float));

    const int m_tiles = (M + BM - 1) / BM;
    dim3 grid_p1(m_tiles, k_splits);
    dim3 block_p1(BLOCK_THREADS);

    hgemm_phase1<<<grid_p1, block_p1>>>(ptr_A, ptr_B, workspace, M, N, K, k_per_split);

    dim3 grid_p2((total + 255) / 256);
    dim3 block_p2(256);

    hgemm_phase2<<<grid_p2, block_p2>>>(workspace, ptr_C, M, N, k_splits);
}