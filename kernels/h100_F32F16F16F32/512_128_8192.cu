#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define BM 64
#define BN 128
#define BK 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M 4
#define WARPS_N 2
#define NUM_WARPS 8
#define BLOCK_THREADS (NUM_WARPS * 32)

#define WARP_TILES_M 1
#define WARP_TILES_N 4

#define SMA_S (BK + 8)
#define SMB_S (BN + 8)

__device__ __forceinline__ void cp_async_16b(void* dst, const void* src) {
    unsigned dst_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        : : "r"(dst_addr), "l"((unsigned long long)(uintptr_t)src)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" : : : "memory");
}

__device__ __forceinline__ void cp_async_wait1() {
    asm volatile("cp.async.wait_group 1;\n" : : : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" : : : "memory");
}

__device__ __forceinline__ void load_A_async(
    const half* __restrict__ A,
    half smA[][SMA_S],
    int m_off, int kk, int M, int K, int tid
) {
    const int num = (BM * BK) / 8;
    for (int i = tid; i < num; i += BLOCK_THREADS) {
        int elem = i * 8;
        int r = elem / BK;
        int c = elem % BK;
        int gr = m_off + r;
        int gc = kk + c;
        half* dst = &smA[r][c];
        if (__builtin_expect(gr < M && gc + 7 < K, 1)) {
            cp_async_16b(dst, &A[gr * K + gc]);
        } else {
            half tmp[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                tmp[e] = (gr < M && gc + e < K) ? A[gr * K + gc + e] : __float2half(0.f);
            *reinterpret_cast<float4*>(dst) = *reinterpret_cast<float4*>(tmp);
        }
    }
}

__device__ __forceinline__ void load_B_async(
    const half* __restrict__ B,
    half smB[][SMB_S],
    int kk, int N, int K, int tid
) {
    const int num = (BK * BN) / 8;
    for (int i = tid; i < num; i += BLOCK_THREADS) {
        int elem = i * 8;
        int r = elem / BN;
        int c = elem % BN;
        int gr = kk + r;
        half* dst = &smB[r][c];
        if (__builtin_expect(gr < K && c + 7 < N, 1)) {
            cp_async_16b(dst, &B[gr * N + c]);
        } else {
            half tmp[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                tmp[e] = (gr < K && c + e < N) ? B[gr * N + c + e] : __float2half(0.f);
            *reinterpret_cast<float4*>(dst) = *reinterpret_cast<float4*>(tmp);
        }
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
hgemm_main_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C_partial,
    int M, int N, int K,
    int k_per_split
) {
    __shared__ half smA[2][BM][SMA_S];
    __shared__ half smB[2][BK][SMB_S];
    __shared__ float warp_stg[4][WMMA_M * WMMA_N];

    const int bm    = blockIdx.x;
    const int split = blockIdx.y;
    const int m_off = bm * BM;
    const int k_off = split * k_per_split;
    const int k_end = min(k_off + k_per_split, K);

    if (m_off >= M || k_off >= k_end) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int warp_r = warp_id / WARPS_N;
    const int warp_c = warp_id % WARPS_N;

    const int warp_m_off = warp_r * WMMA_M;
    const int warp_n_off = warp_c * (WARP_TILES_N * WMMA_N);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILES_N];
    #pragma unroll
    for (int j = 0; j < WARP_TILES_N; j++)
        wmma::fill_fragment(acc[j], 0.f);

    const int num_iters = (k_end - k_off + BK - 1) / BK;

    load_A_async(A, smA[0], m_off, k_off, M, K, tid);
    load_B_async(B, smB[0], k_off, N, K, tid);
    cp_async_commit();

    int buf = 0;

    for (int iter = 0; iter < num_iters; iter++) {
        int nb = 1 - buf;
        int next_kk = k_off + (iter + 1) * BK;

        if (iter + 1 < num_iters) {
            load_A_async(A, smA[nb], m_off, next_kk, M, K, tid);
            load_B_async(B, smB[nb], next_kk, N, K, tid);
            cp_async_commit();
            cp_async_wait1();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fb[WARP_TILES_N];

        #pragma unroll 2
        for (int ki = 0; ki < BK; ki += WMMA_K) {
            wmma::load_matrix_sync(fa, &smA[buf][warp_m_off][ki], SMA_S);
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++) {
                wmma::load_matrix_sync(fb[j], &smB[buf][ki][warp_n_off + j * WMMA_N], SMB_S);
            }
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++) {
                wmma::mma_sync(acc[j], fa, fb[j], acc[j]);
            }
        }

        buf = nb;
    }

    float* C_split = C_partial + (long long)split * M * N;
    const int crow = m_off + warp_m_off;

    if (warp_id < 4) {
        float* slot = warp_stg[warp_id];
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            int ccol = warp_n_off + j * WMMA_N;
            wmma::store_matrix_sync(slot, acc[j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = lane_id; e < WMMA_M * WMMA_N; e += 32) {
                int r = e / WMMA_N;
                int c = e % WMMA_N;
                int gr = crow + r;
                int gc = ccol + c;
                if (gr < M && gc < N)
                    C_split[(long long)gr * N + gc] = slot[e];
            }
        }
    }

    __syncthreads();

    if (warp_id >= 4) {
        float* slot = warp_stg[warp_id - 4];
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            int ccol = warp_n_off + j * WMMA_N;
            wmma::store_matrix_sync(slot, acc[j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = lane_id; e < WMMA_M * WMMA_N; e += 32) {
                int r = e / WMMA_N;
                int c = e % WMMA_N;
                int gr = crow + r;
                int gc = ccol + c;
                if (gr < M && gc < N)
                    C_split[(long long)gr * N + gc] = slot[e];
            }
        }
    }
}

__global__ void hgemm_reduce_kernel(
    const float* __restrict__ partial,
    half* __restrict__ C,
    int M, int N, int num_splits
) {
    const int warps_per_block = blockDim.x / 32;
    const int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= M) return;

    const int lane = threadIdx.x % 32;
    const int col = lane * 4;
    if (col + 3 >= N) return;

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const long long mn = (long long)M * N;
    const long long base_off = (long long)row * N + col;

    #pragma unroll 4
    for (int sp = 0; sp < num_splits; sp++) {
        float4 v = *reinterpret_cast<const float4*>(&partial[sp * mn + base_off]);
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }

    *reinterpret_cast<half2*>(&C[row * N + col])     = __floats2half2_rn(s0, s1);
    *reinterpret_cast<half2*>(&C[row * N + col + 2]) = __floats2half2_rn(s2, s3);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C_ptr       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int num_splits = 16;
    int k_per_split = (K + num_splits - 1) / num_splits;
    k_per_split = ((k_per_split + BK - 1) / BK) * BK;
    const int actual_splits = (K + k_per_split - 1) / k_per_split;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    auto C_partial = torch::empty({(long long)actual_splits * M * N}, opts);
    float* partial_ptr = C_partial.data_ptr<float>();

    const int num_m_tiles = (M + BM - 1) / BM;
    dim3 grid(num_m_tiles, actual_splits);
    dim3 block(BLOCK_THREADS);

    hgemm_main_kernel<<<grid, block>>>(
        A_ptr, B_ptr, partial_ptr, M, N, K, k_per_split);

    const int warps_per_block = 16;
    dim3 red_grid((M + warps_per_block - 1) / warps_per_block);
    dim3 red_block(warps_per_block * 32);
    hgemm_reduce_kernel<<<red_grid, red_block>>>(
        partial_ptr, C_ptr, M, N, actual_splits);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}