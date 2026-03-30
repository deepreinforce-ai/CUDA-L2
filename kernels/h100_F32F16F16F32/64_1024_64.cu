#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;
using namespace nvcuda::wmma;

#define TN_BLOCK_A  16
#define NUM_WARPS_A 4
#define BLOCK_DIM_A 128

#define SA_STRIDE_A 72
#define SB_STRIDE_A 24

__global__ __launch_bounds__(BLOCK_DIM_A, 8)
void hgemm_v3_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int N
) {
    __shared__ __align__(128) half sA[64 * SA_STRIDE_A];
    __shared__ __align__(128) half sB[64 * SB_STRIDE_A];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int col_base = blockIdx.x * TN_BLOCK_A;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int fid  = tid * 4 + i;
        int row  = fid >> 3;
        int gcol = (fid & 7) << 3;
        *reinterpret_cast<float4*>(sA + row * SA_STRIDE_A + gcol) =
            *reinterpret_cast<const float4*>(A + row * 64 + gcol);
    }

    {
        int k_row   = tid >> 1;
        int n_local = (tid & 1) << 3;
        int n_global = col_base + n_local;
        *reinterpret_cast<float4*>(sB + k_row * SB_STRIDE_A + n_local) =
            *reinterpret_cast<const float4*>(B + k_row * N + n_global);
    }

    __syncthreads();

    const int warp_row = warp_id << 4;

    fragment<matrix_a, 16, 16, 16, half, row_major> fa[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fa[ki],
            sA + warp_row * SA_STRIDE_A + ki * 16,
            SA_STRIDE_A);
    }

    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        fragment<matrix_b, 16, 16, 16, half, row_major> fb;
        load_matrix_sync(fb,
            sB + ki * 16 * SB_STRIDE_A,
            SB_STRIDE_A);
        mma_sync(acc, fa[ki], fb, acc);
    }

    __syncthreads();

    float* fscratch = reinterpret_cast<float*>(sA);
    float* wfs      = fscratch + warp_id * 256;

    store_matrix_sync(wfs, acc, 16, mem_row_major);

    const int n_global_base = col_base;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        int flat  = lane_id * 8 + e;
        int lrow  = flat >> 4;
        int lcol  = flat & 15;
        int grow  = warp_row + lrow;
        int gcol  = n_global_base + lcol;
        C[grow * N + gcol] = __float2half(wfs[flat]);
    }
}


#define TN_BLOCK_B  32
#define NUM_WARPS_B 8
#define BLOCK_DIM_B 256

#define SA_STRIDE_B 72
#define SB_STRIDE_B 40

__global__ __launch_bounds__(BLOCK_DIM_B, 4)
void hgemm_v3b_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int N
) {
    __shared__ __align__(128) half sA[64 * SA_STRIDE_B];
    __shared__ __align__(128) half sB[64 * SB_STRIDE_B];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int col_base = blockIdx.x * TN_BLOCK_B;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int fid  = tid * 2 + i;
        int row  = fid >> 3;
        int gcol = (fid & 7) << 3;
        *reinterpret_cast<float4*>(sA + row * SA_STRIDE_B + gcol) =
            *reinterpret_cast<const float4*>(A + row * 64 + gcol);
    }

    {
        int k_row   = tid >> 2;
        int n_local = (tid & 3) << 3;
        int n_global = col_base + n_local;
        *reinterpret_cast<float4*>(sB + k_row * SB_STRIDE_B + n_local) =
            *reinterpret_cast<const float4*>(B + k_row * N + n_global);
    }

    __syncthreads();

    const int warp_row_id = warp_id & 3;
    const int warp_col_id = warp_id >> 2;
    const int warp_row    = warp_row_id << 4;
    const int n_off       = warp_col_id << 4;

    fragment<matrix_a, 16, 16, 16, half, row_major> fa[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fa[ki],
            sA + warp_row * SA_STRIDE_B + ki * 16,
            SA_STRIDE_B);
    }

    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        fragment<matrix_b, 16, 16, 16, half, row_major> fb;
        load_matrix_sync(fb,
            sB + ki * 16 * SB_STRIDE_B + n_off,
            SB_STRIDE_B);
        mma_sync(acc, fa[ki], fb, acc);
    }

    __syncthreads();

    float* fscratch = reinterpret_cast<float*>(sA);
    float* wfs      = fscratch + warp_id * 256;

    store_matrix_sync(wfs, acc, 16, mem_row_major);

    const int n_global_base = col_base + n_off;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        int flat = lane_id * 8 + e;
        int lrow = flat >> 4;
        int lcol = flat & 15;
        int grow = warp_row + lrow;
        int gcol = n_global_base + lcol;
        C[grow * N + gcol] = __float2half(wfs[flat]);
    }
}


#define TN_BLOCK_C  64
#define BLOCK_DIM_C 128

#define SA_STRIDE_C 72
#define SB_STRIDE_C 72

__global__ __launch_bounds__(BLOCK_DIM_C, 4)
void hgemm_v3c_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int N
) {
    __shared__ __align__(128) half sA[64 * SA_STRIDE_C];
    __shared__ __align__(128) half sB[64 * SB_STRIDE_C];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int col_base = blockIdx.x * TN_BLOCK_C;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int fid  = tid * 4 + i;
        int row  = fid >> 3;
        int gcol = (fid & 7) << 3;
        *reinterpret_cast<float4*>(sA + row * SA_STRIDE_C + gcol) =
            *reinterpret_cast<const float4*>(A + row * 64 + gcol);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int fid     = tid * 4 + i;
        int k_row   = fid >> 3;
        int n_local = (fid & 7) << 3;
        int n_global = col_base + n_local;
        *reinterpret_cast<float4*>(sB + k_row * SB_STRIDE_C + n_local) =
            *reinterpret_cast<const float4*>(B + k_row * N + n_global);
    }

    __syncthreads();

    const int warp_row = warp_id << 4;

    fragment<matrix_a, 16, 16, 16, half, row_major> fa[4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        load_matrix_sync(fa[ki],
            sA + warp_row * SA_STRIDE_C + ki * 16,
            SA_STRIDE_C);
    }

    fragment<accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) fill_fragment(acc[nt], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> fb;
            load_matrix_sync(fb,
                sB + ki * 16 * SB_STRIDE_C + nt * 16,
                SB_STRIDE_C);
            mma_sync(acc[nt], fa[ki], fb, acc[nt]);
        }
    }

    __syncthreads();

    float* fscratch = reinterpret_cast<float*>(sA);
    float* wfs      = fscratch + warp_id * 256;

    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        store_matrix_sync(wfs, acc[nt], 16, mem_row_major);

        const int n_global_base = col_base + nt * 16;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int flat = lane_id * 8 + e;
            int lrow = flat >> 4;
            int lcol = flat & 15;
            int grow = warp_row + lrow;
            int gcol = n_global_base + lcol;
            C[grow * N + gcol] = __float2half(wfs[flat]);
        }
    }
}


void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C_ptr       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (M == 64 && K == 64 && N == 1024) {
        dim3 grid_a(N / TN_BLOCK_A);
        dim3 block_a(BLOCK_DIM_A);

        hgemm_v3_kernel<<<grid_a, block_a>>>(A_ptr, B_ptr, C_ptr, N);
    } else {
        dim3 grid_c((N + TN_BLOCK_C - 1) / TN_BLOCK_C, 1);
        dim3 block_c(BLOCK_DIM_C);
        hgemm_v3c_kernel<<<grid_c, block_c>>>(A_ptr, B_ptr, C_ptr, N);
    }
}