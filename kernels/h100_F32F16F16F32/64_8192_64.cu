#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace nvcuda;

static constexpr int PERS_BN       = 64;
static constexpr int PERS_BM       = 64;
static constexpr int PERS_BK       = 64;
static constexpr int PERS_NWARPS   = 4;
static constexpr int PERS_NTHREADS = PERS_NWARPS * 32;
static constexpr int PERS_WM       = 16;
static constexpr int PERS_WN       = 16;
static constexpr int PERS_WK       = 16;
static constexpr int PERS_KTILES   = PERS_BK / PERS_WK;
static constexpr int PERS_NTILES   = PERS_BN / PERS_WN;
static constexpr int PERS_SA_STRIDE= PERS_BK + 8;
static constexpr int PERS_SMEM     = PERS_BM * PERS_BN * (int)sizeof(float);

__global__ void __launch_bounds__(PERS_NTHREADS, 8)
hgemm_persistent_bn64(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half*       __restrict__ C,
    int N, int K)
{
    const int warp_id = threadIdx.x >> 5;
    const int warp_m  = warp_id * PERS_WM;

    extern __shared__ char smem_raw[];
    half* smem_A = reinterpret_cast<half*>(smem_raw);

    {
        const int total_f4 = (PERS_BM * PERS_BK) / 8;
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < total_f4; idx += PERS_NTHREADS) {
            int row = idx / (PERS_BK / 8);
            int col = (idx % (PERS_BK / 8)) * 8;
            half* dst = &smem_A[row * PERS_SA_STRIDE + col];
            uint32_t smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(smem_addr), "l"((uint64_t)(&A[row * PERS_BK + col])));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }

    wmma::fragment<wmma::matrix_a, PERS_WM, PERS_WN, PERS_WK, half, wmma::row_major> frag_A[PERS_KTILES];
    #pragma unroll
    for (int kw = 0; kw < PERS_KTILES; kw++) {
        wmma::load_matrix_sync(frag_A[kw],
            smem_A + warp_m * PERS_SA_STRIDE + kw * PERS_WK,
            PERS_SA_STRIDE);
    }

    const int n_tiles_total = N / PERS_BN;
    for (int tile_n = blockIdx.x; tile_n < n_tiles_total; tile_n += gridDim.x) {
        const int n_start = tile_n * PERS_BN;

        wmma::fragment<wmma::accumulator, PERS_WM, PERS_WN, PERS_WK, float> frag_C[PERS_NTILES];
        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::fill_fragment(frag_C[nw], 0.0f);

        wmma::fragment<wmma::matrix_b, PERS_WM, PERS_WN, PERS_WK, half, wmma::col_major> frag_B_a[PERS_NTILES];
        wmma::fragment<wmma::matrix_b, PERS_WM, PERS_WN, PERS_WK, half, wmma::col_major> frag_B_b[PERS_NTILES];

        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::load_matrix_sync(frag_B_a[nw],
                B_col + (int64_t)(n_start + nw * PERS_WN) * K + 0, K);

        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::load_matrix_sync(frag_B_b[nw],
                B_col + (int64_t)(n_start + nw * PERS_WN) * K + PERS_WK, K);
        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::mma_sync(frag_C[nw], frag_A[0], frag_B_a[nw], frag_C[nw]);

        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::load_matrix_sync(frag_B_a[nw],
                B_col + (int64_t)(n_start + nw * PERS_WN) * K + 2*PERS_WK, K);
        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::mma_sync(frag_C[nw], frag_A[1], frag_B_b[nw], frag_C[nw]);

        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::load_matrix_sync(frag_B_b[nw],
                B_col + (int64_t)(n_start + nw * PERS_WN) * K + 3*PERS_WK, K);
        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::mma_sync(frag_C[nw], frag_A[2], frag_B_a[nw], frag_C[nw]);

        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++)
            wmma::mma_sync(frag_C[nw], frag_A[3], frag_B_b[nw], frag_C[nw]);

        float* smem_C = reinterpret_cast<float*>(smem_raw);

        #pragma unroll
        for (int nw = 0; nw < PERS_NTILES; nw++) {
            wmma::store_matrix_sync(
                smem_C + warp_m * PERS_BN + nw * PERS_WN,
                frag_C[nw], PERS_BN, wmma::mem_row_major);
        }
        __syncthreads();

        {
            half2* C_h2    = reinterpret_cast<half2*>(C);
            const int N_h2 = N >> 1;
            const int tot  = (PERS_BM * PERS_BN) >> 1;
            #pragma unroll 4
            for (int i = threadIdx.x; i < tot; i += PERS_NTHREADS) {
                int m  = i / (PERS_BN / 2);
                int n2 = (i % (PERS_BN / 2)) * 2;
                float v0 = smem_C[m * PERS_BN + n2];
                float v1 = smem_C[m * PERS_BN + n2 + 1];
                C_h2[(int64_t)m * N_h2 + ((n_start + n2) >> 1)] =
                    __float22half2_rn(make_float2(v0, v1));
            }
        }
        __syncthreads();
    }
}

static constexpr int A_BM       = 64;
static constexpr int A_BN       = 64;
static constexpr int A_BK       = 64;
static constexpr int A_NWARPS   = 4;
static constexpr int A_NTHREADS = A_NWARPS * 32;
static constexpr int A_WM       = 16;
static constexpr int A_WN       = 16;
static constexpr int A_WK       = 16;
static constexpr int A_KTILES   = A_BK / A_WK;
static constexpr int A_NTILES   = A_BN / A_WN;
static constexpr int A_SA_STRIDE= A_BK + 8;
static constexpr int A_SMEM     = A_BM * A_BN * (int)sizeof(float);

__global__ void __launch_bounds__(A_NTHREADS, 10)
hgemm_bn64_4warp(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half*       __restrict__ C,
    int N, int K)
{
    const int n_start = blockIdx.x * A_BN;
    const int warp_id = threadIdx.x >> 5;
    const int warp_m  = warp_id * A_WM;

    extern __shared__ char smem_raw[];
    half* smem_A = reinterpret_cast<half*>(smem_raw);

    {
        const int total_f4 = (A_BM * A_BK) / 8;
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < total_f4; idx += A_NTHREADS) {
            int row = idx / (A_BK / 8);
            int col = (idx % (A_BK / 8)) * 8;
            half* dst = &smem_A[row * A_SA_STRIDE + col];
            uint32_t smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(smem_addr), "l"((uint64_t)(&A[row * A_BK + col])));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }

    wmma::fragment<wmma::matrix_a, A_WM, A_WN, A_WK, half, wmma::row_major> frag_A[A_KTILES];
    #pragma unroll
    for (int kw = 0; kw < A_KTILES; kw++)
        wmma::load_matrix_sync(frag_A[kw], smem_A + warp_m * A_SA_STRIDE + kw * A_WK, A_SA_STRIDE);

    wmma::fragment<wmma::accumulator, A_WM, A_WN, A_WK, float> frag_C[A_NTILES];
    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::fill_fragment(frag_C[nw], 0.0f);

    wmma::fragment<wmma::matrix_b, A_WM, A_WN, A_WK, half, wmma::col_major> frag_B_a[A_NTILES];
    wmma::fragment<wmma::matrix_b, A_WM, A_WN, A_WK, half, wmma::col_major> frag_B_b[A_NTILES];

    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + nw * A_WN) * K + 0, K);
    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + nw * A_WN) * K + A_WK, K);
    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[0], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + nw * A_WN) * K + 2*A_WK, K);
    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[1], frag_B_b[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + nw * A_WN) * K + 3*A_WK, K);
    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[2], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[3], frag_B_b[nw], frag_C[nw]);

    float* smem_C = reinterpret_cast<float*>(smem_raw);
    #pragma unroll
    for (int nw = 0; nw < A_NTILES; nw++)
        wmma::store_matrix_sync(smem_C + warp_m * A_BN + nw * A_WN, frag_C[nw], A_BN, wmma::mem_row_major);
    __syncthreads();

    {
        half2* C_h2    = reinterpret_cast<half2*>(C);
        const int N_h2 = N >> 1;
        const int tot  = (A_BM * A_BN) >> 1;
        #pragma unroll 4
        for (int i = threadIdx.x; i < tot; i += A_NTHREADS) {
            int m  = i / (A_BN / 2);
            int n2 = (i % (A_BN / 2)) * 2;
            float v0 = smem_C[m * A_BN + n2];
            float v1 = smem_C[m * A_BN + n2 + 1];
            C_h2[(int64_t)m * N_h2 + ((n_start + n2) >> 1)] = __float22half2_rn(make_float2(v0, v1));
        }
    }
}

static constexpr int B_BM       = 64;
static constexpr int B_BN       = 128;
static constexpr int B_BK       = 64;
static constexpr int B_NWARPS   = 4;
static constexpr int B_NTHREADS = B_NWARPS * 32;
static constexpr int B_WM       = 16;
static constexpr int B_WN       = 16;
static constexpr int B_WK       = 16;
static constexpr int B_KTILES   = B_BK / B_WK;
static constexpr int B_NTILES   = B_BN / B_WN;
static constexpr int B_SA_STRIDE= B_BK + 8;
static constexpr int B_SMEM     = B_BM * B_BN * (int)sizeof(float);

__global__ void __launch_bounds__(B_NTHREADS, 4)
hgemm_bn128_4warp(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half*       __restrict__ C,
    int N, int K)
{
    const int n_start = blockIdx.x * B_BN;
    const int warp_id = threadIdx.x >> 5;
    const int warp_m  = warp_id * B_WM;

    extern __shared__ char smem_raw[];
    half* smem_A = reinterpret_cast<half*>(smem_raw);

    {
        const int total_f4 = (B_BM * B_BK) / 8;
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < total_f4; idx += B_NTHREADS) {
            int row = idx / (B_BK / 8);
            int col = (idx % (B_BK / 8)) * 8;
            half* dst = &smem_A[row * B_SA_STRIDE + col];
            uint32_t smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(smem_addr), "l"((uint64_t)(&A[row * B_BK + col])));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }

    wmma::fragment<wmma::matrix_a, B_WM, B_WN, B_WK, half, wmma::row_major> frag_A[B_KTILES];
    #pragma unroll
    for (int kw = 0; kw < B_KTILES; kw++)
        wmma::load_matrix_sync(frag_A[kw], smem_A + warp_m * B_SA_STRIDE + kw * B_WK, B_SA_STRIDE);

    wmma::fragment<wmma::accumulator, B_WM, B_WN, B_WK, float> frag_C[B_NTILES];
    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::fill_fragment(frag_C[nw], 0.0f);

    wmma::fragment<wmma::matrix_b, B_WM, B_WN, B_WK, half, wmma::col_major> frag_B_a[B_NTILES];
    wmma::fragment<wmma::matrix_b, B_WM, B_WN, B_WK, half, wmma::col_major> frag_B_b[B_NTILES];

    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + nw * B_WN) * K + 0, K);
    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + nw * B_WN) * K + B_WK, K);
    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[0], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + nw * B_WN) * K + 2*B_WK, K);
    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[1], frag_B_b[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + nw * B_WN) * K + 3*B_WK, K);
    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[2], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[3], frag_B_b[nw], frag_C[nw]);

    float* smem_C = reinterpret_cast<float*>(smem_raw);
    #pragma unroll
    for (int nw = 0; nw < B_NTILES; nw++)
        wmma::store_matrix_sync(smem_C + warp_m * B_BN + nw * B_WN, frag_C[nw], B_BN, wmma::mem_row_major);
    __syncthreads();

    {
        half2* C_h2    = reinterpret_cast<half2*>(C);
        const int N_h2 = N >> 1;
        const int tot  = (B_BM * B_BN) >> 1;
        #pragma unroll 8
        for (int i = threadIdx.x; i < tot; i += B_NTHREADS) {
            int m  = i / (B_BN / 2);
            int n2 = (i % (B_BN / 2)) * 2;
            float v0 = smem_C[m * B_BN + n2];
            float v1 = smem_C[m * B_BN + n2 + 1];
            C_h2[(int64_t)m * N_h2 + ((n_start + n2) >> 1)] = __float22half2_rn(make_float2(v0, v1));
        }
    }
}

static constexpr int C_BM       = 64;
static constexpr int C_BN       = 64;
static constexpr int C_BK       = 64;
static constexpr int C_NWARPS   = 8;
static constexpr int C_NTHREADS = C_NWARPS * 32;
static constexpr int C_WM       = 16;
static constexpr int C_WN       = 16;
static constexpr int C_WK       = 16;
static constexpr int C_KTILES   = C_BK / C_WK;
static constexpr int C_NTW      = 2;
static constexpr int C_SA_STRIDE= C_BK + 8;
static constexpr int C_SMEM     = C_BM * C_BN * (int)sizeof(float);

__global__ void __launch_bounds__(C_NTHREADS, 6)
hgemm_bn64_8warp(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half*       __restrict__ C,
    int N, int K)
{
    const int n_start = blockIdx.x * C_BN;
    const int warp_id = threadIdx.x >> 5;
    const int warp_m  = (warp_id >> 1) * C_WM;
    const int warp_n  = (warp_id & 1) * (C_NTW * C_WN);

    extern __shared__ char smem_raw[];
    half* smem_A = reinterpret_cast<half*>(smem_raw);

    {
        const int total_f4 = (C_BM * C_BK) / 8;
        #pragma unroll 2
        for (int idx = threadIdx.x; idx < total_f4; idx += C_NTHREADS) {
            int row = idx / (C_BK / 8);
            int col = (idx % (C_BK / 8)) * 8;
            half* dst = &smem_A[row * C_SA_STRIDE + col];
            uint32_t smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(smem_addr), "l"((uint64_t)(&A[row * C_BK + col])));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }

    wmma::fragment<wmma::matrix_a, C_WM, C_WN, C_WK, half, wmma::row_major> frag_A[C_KTILES];
    #pragma unroll
    for (int kw = 0; kw < C_KTILES; kw++)
        wmma::load_matrix_sync(frag_A[kw], smem_A + warp_m * C_SA_STRIDE + kw * C_WK, C_SA_STRIDE);

    wmma::fragment<wmma::accumulator, C_WM, C_WN, C_WK, float> frag_C[C_NTW];
    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::fill_fragment(frag_C[nw], 0.0f);

    wmma::fragment<wmma::matrix_b, C_WM, C_WN, C_WK, half, wmma::col_major> frag_B_a[C_NTW];
    wmma::fragment<wmma::matrix_b, C_WM, C_WN, C_WK, half, wmma::col_major> frag_B_b[C_NTW];

    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + warp_n + nw * C_WN) * K + 0, K);
    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + warp_n + nw * C_WN) * K + C_WK, K);
    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[0], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + warp_n + nw * C_WN) * K + 2*C_WK, K);
    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[1], frag_B_b[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + warp_n + nw * C_WN) * K + 3*C_WK, K);
    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[2], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[3], frag_B_b[nw], frag_C[nw]);

    float* smem_C = reinterpret_cast<float*>(smem_raw);
    #pragma unroll
    for (int nw = 0; nw < C_NTW; nw++)
        wmma::store_matrix_sync(smem_C + warp_m * C_BN + warp_n + nw * C_WN, frag_C[nw], C_BN, wmma::mem_row_major);
    __syncthreads();

    {
        half2* C_h2    = reinterpret_cast<half2*>(C);
        const int N_h2 = N >> 1;
        const int tot  = (C_BM * C_BN) >> 1;
        #pragma unroll 2
        for (int i = threadIdx.x; i < tot; i += C_NTHREADS) {
            int m  = i / (C_BN / 2);
            int n2 = (i % (C_BN / 2)) * 2;
            float v0 = smem_C[m * C_BN + n2];
            float v1 = smem_C[m * C_BN + n2 + 1];
            C_h2[(int64_t)m * N_h2 + ((n_start + n2) >> 1)] = __float22half2_rn(make_float2(v0, v1));
        }
    }
}

static constexpr int D_BM       = 64;
static constexpr int D_BN       = 256;
static constexpr int D_BK       = 64;
static constexpr int D_NWARPS   = 4;
static constexpr int D_NTHREADS = D_NWARPS * 32;
static constexpr int D_WM       = 16;
static constexpr int D_WN       = 16;
static constexpr int D_WK       = 16;
static constexpr int D_KTILES   = D_BK / D_WK;
static constexpr int D_NTILES   = D_BN / D_WN;
static constexpr int D_SA_STRIDE= D_BK + 8;
static constexpr int D_SMEM     = D_BM * D_BN * (int)sizeof(float);

__global__ void __launch_bounds__(D_NTHREADS, 2)
hgemm_bn256_4warp(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half*       __restrict__ C,
    int N, int K)
{
    const int n_start = blockIdx.x * D_BN;
    const int warp_id = threadIdx.x >> 5;
    const int warp_m  = warp_id * D_WM;

    extern __shared__ char smem_raw[];
    half* smem_A = reinterpret_cast<half*>(smem_raw);

    {
        const int total_f4 = (D_BM * D_BK) / 8;
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < total_f4; idx += D_NTHREADS) {
            int row = idx / (D_BK / 8);
            int col = (idx % (D_BK / 8)) * 8;
            half* dst = &smem_A[row * D_SA_STRIDE + col];
            uint32_t smem_addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(smem_addr), "l"((uint64_t)(&A[row * D_BK + col])));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }

    wmma::fragment<wmma::matrix_a, D_WM, D_WN, D_WK, half, wmma::row_major> frag_A[D_KTILES];
    #pragma unroll
    for (int kw = 0; kw < D_KTILES; kw++)
        wmma::load_matrix_sync(frag_A[kw], smem_A + warp_m * D_SA_STRIDE + kw * D_WK, D_SA_STRIDE);

    wmma::fragment<wmma::accumulator, D_WM, D_WN, D_WK, float> frag_C[D_NTILES];
    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::fill_fragment(frag_C[nw], 0.0f);

    wmma::fragment<wmma::matrix_b, D_WM, D_WN, D_WK, half, wmma::col_major> frag_B_a[D_NTILES];
    wmma::fragment<wmma::matrix_b, D_WM, D_WN, D_WK, half, wmma::col_major> frag_B_b[D_NTILES];

    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + nw * D_WN) * K + 0, K);
    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + nw * D_WN) * K + D_WK, K);
    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[0], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_a[nw], B_col + (int64_t)(n_start + nw * D_WN) * K + 2*D_WK, K);
    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[1], frag_B_b[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::load_matrix_sync(frag_B_b[nw], B_col + (int64_t)(n_start + nw * D_WN) * K + 3*D_WK, K);
    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[2], frag_B_a[nw], frag_C[nw]);

    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::mma_sync(frag_C[nw], frag_A[3], frag_B_b[nw], frag_C[nw]);

    float* smem_C = reinterpret_cast<float*>(smem_raw);
    #pragma unroll
    for (int nw = 0; nw < D_NTILES; nw++)
        wmma::store_matrix_sync(smem_C + warp_m * D_BN + nw * D_WN, frag_C[nw], D_BN, wmma::mem_row_major);
    __syncthreads();

    {
        half2* C_h2    = reinterpret_cast<half2*>(C);
        const int N_h2 = N >> 1;
        const int tot  = (D_BM * D_BN) >> 1;
        #pragma unroll 16
        for (int i = threadIdx.x; i < tot; i += D_NTHREADS) {
            int m  = i / (D_BN / 2);
            int n2 = (i % (D_BN / 2)) * 2;
            float v0 = smem_C[m * D_BN + n2];
            float v1 = smem_C[m * D_BN + n2 + 1];
            C_h2[(int64_t)m * N_h2 + ((n_start + n2) >> 1)] = __float22half2_rn(make_float2(v0, v1));
        }
    }
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template<typename TM_, typename TN_, typename TK_, typename Policy_, int Stages_>
struct CutGemmCfg {
    using EA = cutlass::half_t; using EB = cutlass::half_t;
    using EC = cutlass::half_t; using ED = cutlass::half_t;
    using EAcc = float; using ECmp = float;
    using LA = cutlass::layout::RowMajor; using LB = cutlass::layout::ColumnMajor;
    using LC = cutlass::layout::RowMajor; using LD = cutlass::layout::RowMajor;
    static constexpr int Align = 8;
    using TileShape    = cute::Shape<TM_, TN_, TK_>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ED, ECmp, EC, ECmp, cutlass::FloatRoundStyle::round_to_nearest>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        EAcc, ECmp, EC, LC, Align, ED, LD, Align,
        cutlass::epilogue::NoSmemWarpSpecialized, EpilogueOp>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        EA, LA, Align, EB, LB, Align, EAcc,
        TileShape, ClusterShape, cute::Int<Stages_>, Policy_>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using CutPP128 = CutGemmCfg<cute::_64, cute::_128, cute::_64,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong, 3>;
using CutPP64  = CutGemmCfg<cute::_64, cute::_64,  cute::_64,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong, 3>;

template<typename Cfg>
static bool run_cut(const cutlass::half_t* A, const cutlass::half_t* B,
                    cutlass::half_t* C, int M, int N, int K)
{
    using Gemm = typename Cfg::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    int dev = 0; cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
        {A, sA, B, sB}, {{1.0f, 0.0f}, C, sC, C, sD}, hw
    };
    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> wsbuf(ws);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    if (gemm.initialize(args, wsbuf.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}
#endif

static void init_attrs() {
    static bool done = false;
    if (done) return;
    done = true;
    cudaFuncSetAttribute(hgemm_persistent_bn64, cudaFuncAttributeMaxDynamicSharedMemorySize, PERS_SMEM);
    cudaFuncSetAttribute(hgemm_bn64_4warp,      cudaFuncAttributeMaxDynamicSharedMemorySize, A_SMEM);
    cudaFuncSetAttribute(hgemm_bn128_4warp,     cudaFuncAttributeMaxDynamicSharedMemorySize, B_SMEM);
    cudaFuncSetAttribute(hgemm_bn64_8warp,      cudaFuncAttributeMaxDynamicSharedMemorySize, C_SMEM);
    cudaFuncSetAttribute(hgemm_bn256_4warp,     cudaFuncAttributeMaxDynamicSharedMemorySize, D_SMEM);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* ptr_A     = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B_col = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       ptr_C     = reinterpret_cast<half*>(c.data_ptr());

    init_attrs();

    if (M == PERS_BM && K == PERS_BK && (N % PERS_BN == 0) && (N % 2 == 0)) {
        int sm_count = 132;
        int n_tiles  = N / PERS_BN;
        int grid_x   = (n_tiles < sm_count) ? n_tiles : sm_count;
        dim3 grid(grid_x);
        dim3 block(PERS_NTHREADS);
        hgemm_persistent_bn64<<<grid, block, PERS_SMEM>>>(ptr_A, ptr_B_col, ptr_C, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

    if (M == A_BM && K == A_BK && (N % A_BN == 0) && (N % 2 == 0)) {
        dim3 grid(N / A_BN);
        dim3 block(A_NTHREADS);
        hgemm_bn64_4warp<<<grid, block, A_SMEM>>>(ptr_A, ptr_B_col, ptr_C, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

    if (M == B_BM && K == B_BK && (N % B_BN == 0) && (N % 2 == 0)) {
        dim3 grid(N / B_BN);
        dim3 block(B_NTHREADS);
        hgemm_bn128_4warp<<<grid, block, B_SMEM>>>(ptr_A, ptr_B_col, ptr_C, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

    if (M == C_BM && K == C_BK && (N % C_BN == 0) && (N % 2 == 0)) {
        dim3 grid(N / C_BN);
        dim3 block(C_NTHREADS);
        hgemm_bn64_8warp<<<grid, block, C_SMEM>>>(ptr_A, ptr_B_col, ptr_C, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

    if (M == D_BM && K == D_BK && (N % D_BN == 0) && (N % 2 == 0)) {
        dim3 grid(N / D_BN);
        dim3 block(D_NTHREADS);
        hgemm_bn256_4warp<<<grid, block, D_SMEM>>>(ptr_A, ptr_B_col, ptr_C, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    {
        const auto* cA = reinterpret_cast<const cutlass::half_t*>(ptr_A);
        const auto* cB = reinterpret_cast<const cutlass::half_t*>(ptr_B_col);
        auto*       cC = reinterpret_cast<cutlass::half_t*>(ptr_C);
        if (run_cut<CutPP128>(cA, cB, cC, M, N, K)) return;
        if (run_cut<CutPP64> (cA, cB, cC, M, N, K)) return;
    }
#endif

    throw std::runtime_error("cuda_l2_h100_fp32: no suitable kernel found");
}