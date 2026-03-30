#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(128, 8)
hgemm_async_v1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int block_m = blockIdx.y * 16;

    if (block_m >= M) return;

    __shared__ __align__(128) half smA[16][72];
    __shared__ __align__(128) half smB[64][136];

    {
        int m_l = tid >> 3;
        int k8  = (tid & 7) << 3;
        int mg  = block_m + m_l;
        half* dst = &smA[m_l][k8];
        if (mg < M) {
            const half* src = &A[mg * K + k8];
            uint32_t dst_addr = __cvta_generic_to_shared(dst);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(dst_addr), "l"((const void*)src)
            );
        } else {
            *reinterpret_cast<float4*>(dst) = make_float4(0,0,0,0);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int k_l = idx >> 4;
        int n8  = (idx & 15) << 3;
        half* dst = &smB[k_l][n8];
        if (k_l < K) {
            const half* src = &B[k_l * N + n8];
            uint32_t dst_addr = __cvta_generic_to_shared(dst);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(dst_addr), "l"((const void*)src)
            );
        } else {
            *reinterpret_cast<float4*>(dst) = make_float4(0,0,0,0);
        }
    }

    asm volatile("cp.async.commit_group;\n" :::);
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    const int wn_base = warp_id * 32;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb0, fb1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1;

    wmma::fill_fragment(fc0, 0.0f);
    wmma::fill_fragment(fc1, 0.0f);

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        wmma::load_matrix_sync(fa,  &smA[0][kt * 16],          72);
        wmma::load_matrix_sync(fb0, &smB[kt * 16][wn_base],    136);
        wmma::load_matrix_sync(fb1, &smB[kt * 16][wn_base+16], 136);
        wmma::mma_sync(fc0, fa, fb0, fc0);
        wmma::mma_sync(fc1, fa, fb1, fc1);
    }

    __syncthreads();
    float* smCf = reinterpret_cast<float*>(smB);

    wmma::store_matrix_sync(smCf + wn_base,      fc0, 128, wmma::mem_row_major);
    wmma::store_matrix_sync(smCf + wn_base + 16, fc1, 128, wmma::mem_row_major);

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int flat = tid + i * 128;
        int m_l  = flat >> 6;
        int n_l  = (flat & 63) * 2;
        int mg   = block_m + m_l;
        if (mg < M) {
            half2 h2 = __floats2half2_rn(smCf[m_l * 128 + n_l], smCf[m_l * 128 + n_l + 1]);
            *reinterpret_cast<half2*>(&C[mg * N + n_l]) = h2;
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* pA = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* pB = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* pC       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    dim3 grid(1, (M + 15) / 16);
    dim3 block(128);
    hgemm_async_v1<<<grid, block>>>(pA, pB, pC, M, N, K);
}