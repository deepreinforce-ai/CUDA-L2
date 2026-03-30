#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace nvcuda;

#define BM 16
#define BN 128
#define BK 128
#define NW  8
#define NT  (NW * 32)
#define PAD 8

__global__ __launch_bounds__(NT, 2)
void hgemm_kernel_v1(
    const __half* __restrict__ A,
    const __half* __restrict__ B_cm,
    __half* __restrict__ C,
    int M, int N, int K
) {
    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int gm_base = bm * BM;
    int gn_base = bn * BN;

    __shared__ __half smem_A[BM][BK + PAD];
    __shared__ __half smem_B[BN][BK + PAD];
    __shared__ float  smem_out[NW][256];

    {
        int row = tid >> 4;
        int col = (tid & 15) << 3;
        int gm  = gm_base + row;
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (gm < M && col + 7 < K)
            v = *reinterpret_cast<const float4*>(&A[gm * K + col]);
        *reinterpret_cast<float4*>(&smem_A[row][col]) = v;
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int eg      = i * NT + tid;
        int n_local = eg >> 4;
        int k_local = (eg & 15) << 3;
        int gn = gn_base + n_local;
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (gn < N && k_local + 7 < K)
            v = *reinterpret_cast<const float4*>(&B_cm[gn * K + k_local]);
        *reinterpret_cast<float4*>(&smem_B[n_local][k_local]) = v;
    }

    __syncthreads();

    int wn_base = warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, &smem_A[0][k],        BK + PAD);
        wmma::load_matrix_sync(b_frag, &smem_B[wn_base][k],  BK + PAD);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    wmma::store_matrix_sync(smem_out[warp_id], acc, 16, wmma::mem_row_major);
    __syncwarp();

    int gm_row  = gm_base;
    int gn_tile = gn_base + wn_base;

    #pragma unroll
    for (int e = 0; e < 4; e++) {
        int idx    = lane_id * 4 + e;
        int r      = idx >> 3;
        int c      = (idx & 7) << 1;
        int gm_out = gm_row  + r;
        int gn_out = gn_tile + c;
        if (gm_out < M) {
            float f0 = smem_out[warp_id][r * 16 + c];
            float f1 = smem_out[warp_id][r * 16 + c + 1];
            __half2 h2 = __floats2half2_rn(f0, f1);
            if (gn_out + 1 <= N)
                *reinterpret_cast<__half2*>(&C[gm_out * N + gn_out]) = h2;
            else if (gn_out < N)
                C[gm_out * N + gn_out] = __float2half(f0);
        }
    }
}

#define BM2  16
#define BN2  64
#define BK2  128
#define NW2  4
#define NT2  (NW2 * 32)

__global__ __launch_bounds__(NT2, 4)
void hgemm_kernel_v2(
    const __half* __restrict__ A,
    const __half* __restrict__ B_cm,
    __half* __restrict__ C,
    int M, int N, int K
) {
    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int gm_base = bm * BM2;
    int gn_base = bn * BN2;

    __shared__ __half smem_A[BM2][BK2 + PAD];
    __shared__ __half smem_B[BN2][BK2 + PAD];
    __shared__ float  smem_out[NW2][256];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int eg  = i * NT2 + tid;
        int row = eg >> 4;
        int col = (eg & 15) << 3;
        if (row < BM2) {
            int gm = gm_base + row;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gm < M && col + 7 < K)
                v = *reinterpret_cast<const float4*>(&A[gm * K + col]);
            *reinterpret_cast<float4*>(&smem_A[row][col]) = v;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int eg      = i * NT2 + tid;
        int n_local = eg >> 4;
        int k_local = (eg & 15) << 3;
        if (n_local < BN2) {
            int gn = gn_base + n_local;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gn < N && k_local + 7 < K)
                v = *reinterpret_cast<const float4*>(&B_cm[gn * K + k_local]);
            *reinterpret_cast<float4*>(&smem_B[n_local][k_local]) = v;
        }
    }

    __syncthreads();

    int wn_base = warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int k = 0; k < BK2; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, &smem_A[0][k],       BK2 + PAD);
        wmma::load_matrix_sync(b_frag, &smem_B[wn_base][k], BK2 + PAD);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    wmma::store_matrix_sync(smem_out[warp_id], acc, 16, wmma::mem_row_major);
    __syncwarp();

    int gm_row  = gm_base;
    int gn_tile = gn_base + wn_base;

    #pragma unroll
    for (int e = 0; e < 4; e++) {
        int idx    = lane_id * 4 + e;
        int r      = idx >> 3;
        int c      = (idx & 7) << 1;
        int gm_out = gm_row  + r;
        int gn_out = gn_tile + c;
        if (gm_out < M) {
            float f0 = smem_out[warp_id][r * 16 + c];
            float f1 = smem_out[warp_id][r * 16 + c + 1];
            __half2 h2 = __floats2half2_rn(f0, f1);
            if (gn_out + 1 <= N)
                *reinterpret_cast<__half2*>(&C[gm_out * N + gn_out]) = h2;
            else if (gn_out < N)
                C[gm_out * N + gn_out] = __float2half(f0);
        }
    }
}

#define BM3  32
#define BN3  128
#define BK3  128
#define NW3  4
#define NT3  (NW3 * 32)

__global__ __launch_bounds__(NT3, 3)
void hgemm_kernel_v3(
    const __half* __restrict__ A,
    const __half* __restrict__ B_cm,
    __half* __restrict__ C,
    int M, int N, int K
) {
    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int gm_base = bm * BM3;
    int gn_base = bn * BN3;

    __shared__ __half smem_A[BM3][BK3 + PAD];
    __shared__ __half smem_B[BN3][BK3 + PAD];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int eg  = i * NT3 + tid;
        int row = eg >> 4;
        int col = (eg & 15) << 3;
        if (row < BM3) {
            int gm = gm_base + row;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gm < M && col + 7 < K)
                v = *reinterpret_cast<const float4*>(&A[gm * K + col]);
            *reinterpret_cast<float4*>(&smem_A[row][col]) = v;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eg      = i * NT3 + tid;
        int n_local = eg >> 4;
        int k_local = (eg & 15) << 3;
        if (n_local < BN3) {
            int gn = gn_base + n_local;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gn < N && k_local + 7 < K)
                v = *reinterpret_cast<const float4*>(&B_cm[gn * K + k_local]);
            *reinterpret_cast<float4*>(&smem_B[n_local][k_local]) = v;
        }
    }

    __syncthreads();

    int warp_row = warp_id >> 1;
    int warp_col = warp_id & 1;
    int wm_base  = warp_row * 16;
    int wn_base  = warp_col * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) wmma::fill_fragment(acc[ni], 0.0f);

    #pragma unroll
    for (int k = 0; k < BK3; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &smem_A[wm_base][k], BK3 + PAD);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, &smem_B[wn_base + ni * 16][k], BK3 + PAD);
            wmma::mma_sync(acc[ni], a_frag, b_frag, acc[ni]);
        }
    }

    __syncthreads();
    float* smem_float = reinterpret_cast<float*>(&smem_A[0][0]);
    float* warp_buf   = smem_float + warp_id * 256;

    int gm_warp = gm_base + wm_base;

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int gn_tile = gn_base + wn_base + ni * 16;
        wmma::store_matrix_sync(warp_buf, acc[ni], 16, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int idx    = lane_id * 4 + e;
            int r      = idx >> 3;
            int c      = (idx & 7) << 1;
            int gm_out = gm_warp  + r;
            int gn_out = gn_tile + c;
            if (gm_out < M) {
                float f0 = warp_buf[r * 16 + c];
                float f1 = warp_buf[r * 16 + c + 1];
                __half2 h2 = __floats2half2_rn(f0, f1);
                if (gn_out + 1 <= N)
                    *reinterpret_cast<__half2*>(&C[gm_out * N + gn_out]) = h2;
                else if (gn_out < N)
                    C[gm_out * N + gn_out] = __float2half(f0);
            }
        }
        __syncwarp();
    }
}

#define BM4  8
#define BN4  128
#define BK4  128
#define NW4  8
#define NT4  (NW4 * 32)

#define BM5  16
#define BN5  128
#define BK5  128
#define NW5  4
#define NT5  (NW5 * 32)

__global__ __launch_bounds__(NT5, 3)
void hgemm_kernel_v5(
    const __half* __restrict__ A,
    const __half* __restrict__ B_cm,
    __half* __restrict__ C,
    int M, int N, int K
) {
    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int gm_base = bm * BM5;
    int gn_base = bn * BN5;

    __shared__ __half smem_A[BM5][BK5 + PAD];
    __shared__ __half smem_B[BN5][BK5 + PAD];
    __shared__ float  smem_out[NW5][2][256];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int eg  = i * NT5 + tid;
        int row = eg >> 4;
        int col = (eg & 15) << 3;
        if (row < BM5) {
            int gm = gm_base + row;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gm < M && col + 7 < K)
                v = *reinterpret_cast<const float4*>(&A[gm * K + col]);
            *reinterpret_cast<float4*>(&smem_A[row][col]) = v;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eg      = i * NT5 + tid;
        int n_local = eg >> 4;
        int k_local = (eg & 15) << 3;
        if (n_local < BN5) {
            int gn = gn_base + n_local;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gn < N && k_local + 7 < K)
                v = *reinterpret_cast<const float4*>(&B_cm[gn * K + k_local]);
            *reinterpret_cast<float4*>(&smem_B[n_local][k_local]) = v;
        }
    }

    __syncthreads();

    int wn_base = warp_id * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    #pragma unroll
    for (int k = 0; k < BK5; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &smem_A[0][k], BK5 + PAD);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag0, b_frag1;
        wmma::load_matrix_sync(b_frag0, &smem_B[wn_base][k],      BK5 + PAD);
        wmma::load_matrix_sync(b_frag1, &smem_B[wn_base + 16][k], BK5 + PAD);
        wmma::mma_sync(acc[0], a_frag, b_frag0, acc[0]);
        wmma::mma_sync(acc[1], a_frag, b_frag1, acc[1]);
    }

    int gm_row  = gm_base;
    int gn_tile0 = gn_base + wn_base;
    int gn_tile1 = gn_base + wn_base + 16;

    wmma::store_matrix_sync(smem_out[warp_id][0], acc[0], 16, wmma::mem_row_major);
    wmma::store_matrix_sync(smem_out[warp_id][1], acc[1], 16, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int t = 0; t < 2; t++) {
        int gn_tile = (t == 0) ? gn_tile0 : gn_tile1;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int idx    = lane_id * 4 + e;
            int r      = idx >> 3;
            int c      = (idx & 7) << 1;
            int gm_out = gm_row  + r;
            int gn_out = gn_tile + c;
            if (gm_out < M) {
                float f0 = smem_out[warp_id][t][r * 16 + c];
                float f1 = smem_out[warp_id][t][r * 16 + c + 1];
                __half2 h2 = __floats2half2_rn(f0, f1);
                if (gn_out + 1 <= N)
                    *reinterpret_cast<__half2*>(&C[gm_out * N + gn_out]) = h2;
                else if (gn_out < N)
                    C[gm_out * N + gn_out] = __float2half(f0);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    const __half* pA  = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* pBc = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half* pC        = reinterpret_cast<__half*>(c.data_ptr());

    {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NT);
        hgemm_kernel_v1<<<grid, block>>>(pA, pBc, pC, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
        dim3 block(NT2);
        hgemm_kernel_v2<<<grid, block>>>(pA, pBc, pC, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((N + BN5 - 1) / BN5, (M + BM5 - 1) / BM5);
        dim3 block(NT5);
        hgemm_kernel_v5<<<grid, block>>>(pA, pBc, pC, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((N + BN3 - 1) / BN3, (M + BM3 - 1) / BM3);
        dim3 block(NT3);
        hgemm_kernel_v3<<<grid, block>>>(pA, pBc, pC, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}