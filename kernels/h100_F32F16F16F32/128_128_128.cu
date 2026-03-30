#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda;

__global__ void __launch_bounds__(128, 8)
hgemm_main_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C
) {
    const int by = blockIdx.x;
    const int m_base = by * 16;

    __shared__ __half smem_A[2][16][24];
    __shared__ __half smem_B[2][16][136];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int col_base_warp = warp_id * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    {
        const int elem = tid * 2;
        const int r = elem >> 4;
        const int c = elem & 15;
        __pipeline_memcpy_async(
            &smem_A[0][r][c],
            &A[(m_base + r) * 128 + c],
            4);
    }
    {
        const int row = tid >> 3;
        const int col = (tid & 7) << 4;
        __pipeline_memcpy_async(
            &smem_B[0][row][col],
            &B[row * 128 + col],
            16);
        __pipeline_memcpy_async(
            &smem_B[0][row][col + 8],
            &B[row * 128 + col + 8],
            16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < 8; k++) {
        const int cur = k & 1;
        const int nxt = cur ^ 1;

        if (k < 7) {
            const int kn = (k + 1) * 16;
            {
                const int elem = tid * 2;
                const int r = elem >> 4;
                const int c = elem & 15;
                __pipeline_memcpy_async(
                    &smem_A[nxt][r][c],
                    &A[(m_base + r) * 128 + kn + c],
                    4);
            }
            {
                const int row = tid >> 3;
                const int col = (tid & 7) << 4;
                __pipeline_memcpy_async(
                    &smem_B[nxt][row][col],
                    &B[(kn + row) * 128 + col],
                    16);
                __pipeline_memcpy_async(
                    &smem_B[nxt][row][col + 8],
                    &B[(kn + row) * 128 + col + 8],
                    16);
            }
            __pipeline_commit();
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb0, fb1;

        wmma::load_matrix_sync(fa,  &smem_A[cur][0][0],                   24);
        wmma::load_matrix_sync(fb0, &smem_B[cur][0][col_base_warp],       136);
        wmma::load_matrix_sync(fb1, &smem_B[cur][0][col_base_warp + 16],  136);
        wmma::mma_sync(acc0, fa, fb0, acc0);
        wmma::mma_sync(acc1, fa, fb1, acc1);

        if (k < 7) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    float* fsmem = reinterpret_cast<float*>(smem_A);
    float* warp_buf = fsmem + warp_id * 256;

    wmma::store_matrix_sync(warp_buf, acc0, 16, wmma::mem_row_major);
    __syncwarp();
    {
        const int cb = col_base_warp;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx0 = lane * 8 + e * 2;
            const int idx1 = idx0 + 1;
            const int r0 = idx0 >> 4, c0 = idx0 & 15;
            const int r1 = idx1 >> 4, c1 = idx1 & 15;
            if (r0 == r1) {
                *reinterpret_cast<half2*>(&C[(m_base + r0) * 128 + cb + c0]) =
                    __floats2half2_rn(warp_buf[idx0], warp_buf[idx1]);
            } else {
                C[(m_base + r0) * 128 + cb + c0] = __float2half(warp_buf[idx0]);
                C[(m_base + r1) * 128 + cb + c1] = __float2half(warp_buf[idx1]);
            }
        }
    }

    wmma::store_matrix_sync(warp_buf, acc1, 16, wmma::mem_row_major);
    __syncwarp();
    {
        const int cb = col_base_warp + 16;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx0 = lane * 8 + e * 2;
            const int idx1 = idx0 + 1;
            const int r0 = idx0 >> 4, c0 = idx0 & 15;
            const int r1 = idx1 >> 4, c1 = idx1 & 15;
            if (r0 == r1) {
                *reinterpret_cast<half2*>(&C[(m_base + r0) * 128 + cb + c0]) =
                    __floats2half2_rn(warp_buf[idx0], warp_buf[idx1]);
            } else {
                C[(m_base + r0) * 128 + cb + c0] = __float2half(warp_buf[idx0]);
                C[(m_base + r1) * 128 + cb + c1] = __float2half(warp_buf[idx1]);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4)
hgemm_4cta_256t(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C
) {
    const int by = blockIdx.x;
    const int m_base = by * 32;

    __shared__ __half smem_A[2][32][24];
    __shared__ __half smem_B[2][16][136];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id >> 2;
    const int warp_n = warp_id & 3;
    const int col_base_warp = warp_n * 32;
    const int row_base_warp = warp_m * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    {
        const int elem = tid * 2;
        const int r = elem >> 4;
        const int c = elem & 15;
        __pipeline_memcpy_async(
            &smem_A[0][r][c],
            &A[(m_base + r) * 128 + c],
            4);
    }
    {
        const int row = tid >> 4;
        const int col = (tid & 15) << 3;
        __pipeline_memcpy_async(
            &smem_B[0][row][col],
            &B[row * 128 + col],
            16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < 8; k++) {
        const int cur = k & 1;
        const int nxt = cur ^ 1;

        if (k < 7) {
            const int kn = (k + 1) * 16;
            {
                const int elem = tid * 2;
                const int r = elem >> 4;
                const int c = elem & 15;
                __pipeline_memcpy_async(
                    &smem_A[nxt][r][c],
                    &A[(m_base + r) * 128 + kn + c],
                    4);
            }
            {
                const int row = tid >> 4;
                const int col = (tid & 15) << 3;
                __pipeline_memcpy_async(
                    &smem_B[nxt][row][col],
                    &B[(kn + row) * 128 + col],
                    16);
            }
            __pipeline_commit();
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb0, fb1;

        wmma::load_matrix_sync(fa,  &smem_A[cur][row_base_warp][0],       24);
        wmma::load_matrix_sync(fb0, &smem_B[cur][0][col_base_warp],       136);
        wmma::load_matrix_sync(fb1, &smem_B[cur][0][col_base_warp + 16],  136);
        wmma::mma_sync(acc0, fa, fb0, acc0);
        wmma::mma_sync(acc1, fa, fb1, acc1);

        if (k < 7) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    float* fsmem = reinterpret_cast<float*>(&smem_B[0][0][0]);
    float* warp_buf = fsmem + warp_id * 256;

    const int out_row_base = m_base + row_base_warp;

    wmma::store_matrix_sync(warp_buf, acc0, 16, wmma::mem_row_major);
    __syncwarp();
    {
        const int cb = col_base_warp;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx0 = lane * 8 + e * 2;
            const int idx1 = idx0 + 1;
            const int r0 = idx0 >> 4, c0 = idx0 & 15;
            const int r1 = idx1 >> 4, c1 = idx1 & 15;
            if (r0 == r1) {
                *reinterpret_cast<half2*>(&C[(out_row_base + r0) * 128 + cb + c0]) =
                    __floats2half2_rn(warp_buf[idx0], warp_buf[idx1]);
            } else {
                C[(out_row_base + r0) * 128 + cb + c0] = __float2half(warp_buf[idx0]);
                C[(out_row_base + r1) * 128 + cb + c1] = __float2half(warp_buf[idx1]);
            }
        }
    }

    wmma::store_matrix_sync(warp_buf, acc1, 16, wmma::mem_row_major);
    __syncwarp();
    {
        const int cb = col_base_warp + 16;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx0 = lane * 8 + e * 2;
            const int idx1 = idx0 + 1;
            const int r0 = idx0 >> 4, c0 = idx0 & 15;
            const int r1 = idx1 >> 4, c1 = idx1 & 15;
            if (r0 == r1) {
                *reinterpret_cast<half2*>(&C[(out_row_base + r0) * 128 + cb + c0]) =
                    __floats2half2_rn(warp_buf[idx0], warp_buf[idx1]);
            } else {
                C[(out_row_base + r0) * 128 + cb + c0] = __float2half(warp_buf[idx0]);
                C[(out_row_base + r1) * 128 + cb + c1] = __float2half(warp_buf[idx1]);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const __half* A_ptr = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B_ptr = reinterpret_cast<const __half*>(b.data_ptr());
    __half*       C_ptr = reinterpret_cast<__half*>(c.data_ptr());

    hgemm_main_kernel<<<8, 128>>>(A_ptr, B_ptr, C_ptr);
}