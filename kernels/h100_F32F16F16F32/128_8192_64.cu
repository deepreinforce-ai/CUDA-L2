#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ __launch_bounds__(256, 4)
void hgemm_wmma_bn64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half sA[128][72];
    __shared__ __align__(128) half sB[64][72];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int block_n = blockIdx.x * 64;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int fi  = tid + i * 256;
        int row = fi >> 3;
        int col = (fi & 7) << 3;
        *reinterpret_cast<float4*>(&sA[row][col]) =
            *reinterpret_cast<const float4*>(&A[row * K + col]);
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int fi    = tid + i * 256;
        int n_loc = fi >> 3;
        int k_loc = (fi & 7) << 3;
        int gn    = block_n + n_loc;
        if (gn < N) {
            *reinterpret_cast<float4*>(&sB[n_loc][k_loc]) =
                *reinterpret_cast<const float4*>(&B_col[gn * K + k_loc]);
        } else {
            *reinterpret_cast<float4*>(&sB[n_loc][k_loc]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    __syncthreads();

    const int warp_row_base = warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::fill_fragment(acc[ni], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[4];

        wmma::load_matrix_sync(fa, &sA[warp_row_base][ki * 16], 72);

        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(fb[ni], &sB[ni * 16][ki * 16], 72);

        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(acc[ni], fa, fb[ni], acc[ni]);
    }

    __syncthreads();
    float* fscratch = reinterpret_cast<float*>(sA);
    float* wbuf = fscratch + warp_id * 256;

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        wmma::store_matrix_sync(wbuf, acc[ni], 16, wmma::mem_row_major);
        __syncwarp();

        int base_r = warp_row_base;
        int base_c = block_n + ni * 16;

        #pragma unroll
        for (int idx = lane; idx < 128; idx += 32) {
            int r  = (idx * 2) >> 4;
            int cc = (idx * 2) & 15;
            int gr = base_r + r;
            int gc = base_c + cc;
            if (gr < M) {
                if (gc + 1 < N) {
                    half2 val = __floats2half2_rn(wbuf[r * 16 + cc], wbuf[r * 16 + cc + 1]);
                    *reinterpret_cast<half2*>(&C[gr * N + gc]) = val;
                } else if (gc < N) {
                    C[gr * N + gc] = __float2half(wbuf[r * 16 + cc]);
                }
            }
        }
    }
}

__global__ __launch_bounds__(256, 3)
void hgemm_wmma_bn128_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half sA[128][72];
    __shared__ __align__(128) half sB[128][72];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int warp_m  = warp_id >> 1;
    const int warp_n  = warp_id & 1;
    const int block_n = blockIdx.x * 128;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int fi  = tid + i * 256;
        int row = fi >> 3;
        int col = (fi & 7) << 3;
        *reinterpret_cast<float4*>(&sA[row][col]) =
            *reinterpret_cast<const float4*>(&A[row * K + col]);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int fi    = tid + i * 256;
        int n_loc = fi >> 3;
        int k_loc = (fi & 7) << 3;
        int gn    = block_n + n_loc;
        if (gn < N) {
            *reinterpret_cast<float4*>(&sB[n_loc][k_loc]) =
                *reinterpret_cast<const float4*>(&B_col[gn * K + k_loc]);
        } else {
            *reinterpret_cast<float4*>(&sB[n_loc][k_loc]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    __syncthreads();

    const int warp_row_base = warp_m * 32;
    const int warp_col_base = warp_n * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[4];

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(fa[mi], &sA[warp_row_base + mi * 16][ki * 16], 72);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(fb[ni], &sB[warp_col_base + ni * 16][ki * 16], 72);

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(acc[mi][ni], fa[mi], fb[ni], acc[mi][ni]);
    }

    __syncthreads();
    float* fscratch = reinterpret_cast<float*>(sA);
    float* wbuf = fscratch + warp_id * 256;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::store_matrix_sync(wbuf, acc[mi][ni], 16, wmma::mem_row_major);
            __syncwarp();

            int base_r = warp_row_base + mi * 16;
            int base_c = block_n + warp_col_base + ni * 16;

            #pragma unroll
            for (int idx = lane; idx < 128; idx += 32) {
                int r  = (idx * 2) >> 4;
                int cc = (idx * 2) & 15;
                int gr = base_r + r;
                int gc = base_c + cc;
                if (gr < M) {
                    if (gc + 1 < N) {
                        half2 val = __floats2half2_rn(wbuf[r * 16 + cc], wbuf[r * 16 + cc + 1]);
                        *reinterpret_cast<half2*>(&C[gr * N + gc]) = val;
                    } else if (gc < N) {
                        C[gr * N + gc] = __float2half(wbuf[r * 16 + cc]);
                    }
                }
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* pA    = reinterpret_cast<const half*>(a.data_ptr());
    const half* pBcol = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* pC          = reinterpret_cast<half*>(c.data_ptr());

    {
        dim3 grid((N + 63) / 64);
        dim3 block(256);
        hgemm_wmma_bn64_kernel<<<grid, block>>>(pA, pBcol, pC, M, N, K);
    }
}