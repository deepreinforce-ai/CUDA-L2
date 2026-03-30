#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

#define K1_BM  64
#define K1_BN  64
#define K1_BK  64
#define K1_SA  72
#define K1_SB  72

__global__ void __launch_bounds__(128, 4)
hgemm_64x64_bk64(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * K1_BM;
    const int bn = blockIdx.x * K1_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int wm_id = warp_id >> 1;
    const int wn_id = warp_id & 1;
    const int wm_off = wm_id * 32;
    const int wn_off = wn_id * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    __shared__ __align__(128) half smA[2][K1_BM * K1_SA];
    __shared__ __align__(128) half smB[2][K1_BN * K1_SB];

    const int k_tiles = K / K1_BK;

    auto ldA = [&](int buf, int k_off) {
        #pragma unroll 4
        for (int idx = tid; idx < (K1_BM * K1_BK / 8); idx += 128) {
            const int row = (idx << 3) / K1_BK;
            const int col = (idx << 3) % K1_BK;
            __pipeline_memcpy_async(
                &smA[buf][row * K1_SA + col],
                &A[(bm + row) * K + k_off + col], 16);
        }
    };

    auto ldB = [&](int buf, int k_off) {
        #pragma unroll 4
        for (int idx = tid; idx < (K1_BN * K1_BK / 8); idx += 128) {
            const int row = (idx << 3) / K1_BK;
            const int col = (idx << 3) % K1_BK;
            __pipeline_memcpy_async(
                &smB[buf][row * K1_SB + col],
                &B_col[(bn + row) * K + k_off + col], 16);
        }
    };

    ldA(0, 0); ldB(0, 0);
    __pipeline_commit();

    for (int ki = 0; ki < k_tiles; ki++) {
        const int cur = ki & 1;
        const int nxt = 1 - cur;

        if (ki + 1 < k_tiles) {
            ldA(nxt, (ki + 1) * K1_BK);
            ldB(nxt, (ki + 1) * K1_BK);
            __pipeline_commit();
        }
        __pipeline_wait_prior(ki + 1 < k_tiles ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kf = 0; kf < K1_BK / 16; kf++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(af[mi], &smA[cur][(wm_off + mi * 16) * K1_SA + kf * 16], K1_SA);
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(bf[ni], &smB[cur][(wn_off + ni * 16) * K1_SB + kf * 16], K1_SB);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 2; ni++)
                    wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }

        if (ki + 1 < k_tiles) __syncthreads();
    }

    __shared__ __align__(128) float stg[4 * 256];
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            wmma::store_matrix_sync(stg + warp_id * 256, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();
            const int c_row = bm + wm_off + mi * 16;
            const int c_col = bn + wn_off + ni * 16;
            #pragma unroll
            for (int e = lane_id; e < 256; e += 32) {
                const int gr = c_row + (e >> 4);
                const int gc = c_col + (e & 15);
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(stg[warp_id * 256 + e]);
            }
            __syncwarp();
        }
    }
}

#define K2_BM  128
#define K2_BN  64
#define K2_BK  32
#define K2_SA  40
#define K2_SB  40
#define K2_STAGES 3

__global__ void __launch_bounds__(128, 2)
hgemm_128x64_3stage(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * K2_BM;
    const int bn = blockIdx.x * K2_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int wm_id = warp_id >> 1;
    const int wn_id = warp_id & 1;
    const int wm_off = wm_id * 64;
    const int wn_off = wn_id * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    __shared__ __align__(128) half smA[K2_STAGES][K2_BM * K2_SA];
    __shared__ __align__(128) half smB[K2_STAGES][K2_BN * K2_SB];

    const int k_tiles = K / K2_BK;

    auto ldA = [&](int stage, int k_off) {
        #pragma unroll 4
        for (int idx = tid; idx < (K2_BM * K2_BK / 8); idx += 128) {
            const int row = (idx << 3) / K2_BK;
            const int col = (idx << 3) % K2_BK;
            __pipeline_memcpy_async(
                &smA[stage][row * K2_SA + col],
                &A[(bm + row) * K + k_off + col], 16);
        }
    };

    auto ldB = [&](int stage, int k_off) {
        #pragma unroll 2
        for (int idx = tid; idx < (K2_BN * K2_BK / 8); idx += 128) {
            const int row = (idx << 3) / K2_BK;
            const int col = (idx << 3) % K2_BK;
            __pipeline_memcpy_async(
                &smB[stage][row * K2_SB + col],
                &B_col[(bn + row) * K + k_off + col], 16);
        }
    };

    ldA(0, 0); ldB(0, 0); __pipeline_commit();
    if (k_tiles > 1) { ldA(1, K2_BK); ldB(1, K2_BK); __pipeline_commit(); }

    for (int ki = 0; ki < k_tiles; ki++) {
        const int cur = ki % K2_STAGES;
        const int load_ki = ki + K2_STAGES - 1;

        if (load_ki < k_tiles) {
            int ns = load_ki % K2_STAGES;
            ldA(ns, load_ki * K2_BK);
            ldB(ns, load_ki * K2_BK);
            __pipeline_commit();
        }
        __pipeline_wait_prior(K2_STAGES - 2);
        __syncthreads();

        #pragma unroll
        for (int kf = 0; kf < K2_BK / 16; kf++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                wmma::load_matrix_sync(af[mi], &smA[cur][(wm_off + mi * 16) * K2_SA + kf * 16], K2_SA);
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(bf[ni], &smB[cur][(wn_off + ni * 16) * K2_SB + kf * 16], K2_SB);
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                #pragma unroll
                for (int ni = 0; ni < 2; ni++)
                    wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }

        __syncthreads();
    }

    float* stg = reinterpret_cast<float*>(smA[0]);
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            wmma::store_matrix_sync(stg + warp_id * 256, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();
            const int c_row = bm + wm_off + mi * 16;
            const int c_col = bn + wn_off + ni * 16;
            #pragma unroll
            for (int e = lane_id; e < 256; e += 32) {
                const int gr = c_row + (e >> 4);
                const int gc = c_col + (e & 15);
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(stg[warp_id * 256 + e]);
            }
            __syncwarp();
        }
    }
}

#define K3_BM  64
#define K3_BN  64
#define K3_BK  32
#define K3_SA  40
#define K3_SB  40
#define K3_STAGES 3

__global__ void __launch_bounds__(128, 5)
hgemm_64x64_3stage(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * K3_BM;
    const int bn = blockIdx.x * K3_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int wm_id = warp_id >> 1;
    const int wn_id = warp_id & 1;
    const int wm_off = wm_id * 32;
    const int wn_off = wn_id * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    __shared__ __align__(128) half smA[K3_STAGES][K3_BM * K3_SA];
    __shared__ __align__(128) half smB[K3_STAGES][K3_BN * K3_SB];

    const int k_tiles = K / K3_BK;

    auto ldA = [&](int stage, int k_off) {
        #pragma unroll 2
        for (int idx = tid; idx < (K3_BM * K3_BK / 8); idx += 128) {
            const int row = (idx << 3) / K3_BK;
            const int col = (idx << 3) % K3_BK;
            __pipeline_memcpy_async(
                &smA[stage][row * K3_SA + col],
                &A[(bm + row) * K + k_off + col], 16);
        }
    };

    auto ldB = [&](int stage, int k_off) {
        #pragma unroll 2
        for (int idx = tid; idx < (K3_BN * K3_BK / 8); idx += 128) {
            const int row = (idx << 3) / K3_BK;
            const int col = (idx << 3) % K3_BK;
            __pipeline_memcpy_async(
                &smB[stage][row * K3_SB + col],
                &B_col[(bn + row) * K + k_off + col], 16);
        }
    };

    ldA(0, 0); ldB(0, 0); __pipeline_commit();
    if (k_tiles > 1) { ldA(1, K3_BK); ldB(1, K3_BK); __pipeline_commit(); }

    for (int ki = 0; ki < k_tiles; ki++) {
        const int cur = ki % K3_STAGES;
        const int load_ki = ki + K3_STAGES - 1;

        if (load_ki < k_tiles) {
            int ns = load_ki % K3_STAGES;
            ldA(ns, load_ki * K3_BK);
            ldB(ns, load_ki * K3_BK);
            __pipeline_commit();
        }
        __pipeline_wait_prior(K3_STAGES - 2);
        __syncthreads();

        #pragma unroll
        for (int kf = 0; kf < K3_BK / 16; kf++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(af[mi], &smA[cur][(wm_off + mi * 16) * K3_SA + kf * 16], K3_SA);
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(bf[ni], &smB[cur][(wn_off + ni * 16) * K3_SB + kf * 16], K3_SB);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 2; ni++)
                    wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }

        __syncthreads();
    }

    float* stg = reinterpret_cast<float*>(smA[0]);
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            wmma::store_matrix_sync(stg + warp_id * 256, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();
            const int c_row = bm + wm_off + mi * 16;
            const int c_col = bn + wn_off + ni * 16;
            #pragma unroll
            for (int e = lane_id; e < 256; e += 32) {
                const int gr = c_row + (e >> 4);
                const int gc = c_col + (e & 15);
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(stg[warp_id * 256 + e]);
            }
            __syncwarp();
        }
    }
}

#define K4_BM  128
#define K4_BN  128
#define K4_BK  32
#define K4_SA  40
#define K4_SB  40

__global__ void __launch_bounds__(256, 2)
hgemm_128x128(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * K4_BM;
    const int bn = blockIdx.x * K4_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int wm_id = warp_id >> 1;
    const int wn_id = warp_id & 1;
    const int wm_off = wm_id * 32;
    const int wn_off = wn_id * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    __shared__ __align__(128) half smA[2][K4_BM * K4_SA];
    __shared__ __align__(128) half smB[2][K4_BN * K4_SB];

    const int k_tiles = K / K4_BK;

    auto ldA = [&](int buf, int k_off) {
        #pragma unroll 2
        for (int idx = tid; idx < (K4_BM * K4_BK / 8); idx += 256) {
            const int row = (idx << 3) / K4_BK;
            const int col = (idx << 3) % K4_BK;
            __pipeline_memcpy_async(
                &smA[buf][row * K4_SA + col],
                &A[(bm + row) * K + k_off + col], 16);
        }
    };

    auto ldB = [&](int buf, int k_off) {
        #pragma unroll 2
        for (int idx = tid; idx < (K4_BN * K4_BK / 8); idx += 256) {
            const int row = (idx << 3) / K4_BK;
            const int col = (idx << 3) % K4_BK;
            __pipeline_memcpy_async(
                &smB[buf][row * K4_SB + col],
                &B_col[(bn + row) * K + k_off + col], 16);
        }
    };

    ldA(0, 0); ldB(0, 0);
    __pipeline_commit();

    for (int ki = 0; ki < k_tiles; ki++) {
        const int cur = ki & 1;
        const int nxt = 1 - cur;
        if (ki + 1 < k_tiles) {
            ldA(nxt, (ki + 1) * K4_BK);
            ldB(nxt, (ki + 1) * K4_BK);
            __pipeline_commit();
        }
        __pipeline_wait_prior(ki + 1 < k_tiles ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kf = 0; kf < K4_BK / 16; kf++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(af[mi], &smA[cur][(wm_off + mi * 16) * K4_SA + kf * 16], K4_SA);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(bf[ni], &smB[cur][(wn_off + ni * 16) * K4_SB + kf * 16], K4_SB);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }
        if (ki + 1 < k_tiles) __syncthreads();
    }

    float* stg = reinterpret_cast<float*>(smA[0]);
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::store_matrix_sync(stg + warp_id * 256, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();
            const int c_row = bm + wm_off + mi * 16;
            const int c_col = bn + wn_off + ni * 16;
            #pragma unroll
            for (int e = lane_id; e < 256; e += 32) {
                const int gr = c_row + (e >> 4);
                const int gc = c_col + (e & 15);
                if (gr < M && gc < N)
                    C[gr * N + gc] = __float2half(stg[warp_id * 256 + e]);
            }
            __syncwarp();
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    TORCH_CHECK(a.dtype() == torch::kHalf);
    TORCH_CHECK(b_col_major.dtype() == torch::kHalf);
    TORCH_CHECK(c.dtype() == torch::kHalf);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C           = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        dim3 grid((N + K1_BN - 1) / K1_BN, (M + K1_BM - 1) / K1_BM);
        dim3 block(128);
        hgemm_64x64_bk64<<<grid, block>>>(A, B_col, C, M, N, K);
    }
}