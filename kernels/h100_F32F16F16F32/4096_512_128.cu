#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

using namespace nvcuda::wmma;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

static constexpr int BM = 128;
static constexpr int BN = 64;
static constexpr int BK = 64;
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WARPS_M = 4;
static constexpr int WARPS_N = 2;
static constexpr int THREADS = WARPS_M * WARPS_N * 32;
static constexpr int WT_M = 2;
static constexpr int WT_N = 2;
static constexpr int K_STEPS = BK / WMMA_K;
static constexpr int SMA_PAD = 8;
static constexpr int SMB_PAD = 8;
static constexpr int SMA_STRIDE = BK + SMA_PAD;
static constexpr int SMB_STRIDE = BK + SMB_PAD;
static constexpr int STAGES = 2;
static constexpr int SMEM_A_SIZE = STAGES * BM * SMA_STRIDE;
static constexpr int SMEM_B_SIZE = STAGES * BN * SMB_STRIDE;

__device__ __forceinline__ uint32_t smem_ptr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ void __launch_bounds__(256, 2)
hgemm_kernel_128x64_db(
    const half* __restrict__ A,
    const half* __restrict__ Bcm,
    half* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smem + SMEM_A_SIZE;

    int bm = blockIdx.y * BM;
    int bn = blockIdx.x * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int wm_base = warp_m * (WT_M * WMMA_M);
    int wn_base = warp_n * (WT_N * WMMA_N);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WT_M][WT_N];
    #pragma unroll
    for (int mi = 0; mi < WT_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WT_N; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    int num_k_tiles = (K + BK - 1) / BK;

    auto async_load_A = [&](int stage, int k_off) {
        const half* Abase = A + bm * K + k_off;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int f4_idx = tid + i * THREADS;
            int row = f4_idx / 8;
            int col = (f4_idx % 8) * 8;
            int grow = bm + row;
            half* dst = &smA[stage * BM * SMA_STRIDE + row * SMA_STRIDE + col];
            if (grow < M && (k_off + col) < K) {
                uint32_t dst_ptr = smem_ptr(dst);
                const half* src = Abase + row * K + col;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst_ptr), "l"(src));
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0.f,0.f,0.f,0.f);
            }
        }
    };

    auto async_load_B = [&](int stage, int k_off) {
        const half* Bbase = Bcm + bn * K + k_off;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int f4_idx = tid + i * THREADS;
            int row = f4_idx / 8;
            int col = (f4_idx % 8) * 8;
            int gn = bn + row;
            half* dst = &smB[stage * BN * SMB_STRIDE + row * SMB_STRIDE + col];
            if (gn < N && (k_off + col) < K) {
                uint32_t dst_ptr = smem_ptr(dst);
                const half* src = Bbase + row * K + col;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst_ptr), "l"(src));
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0.f,0.f,0.f,0.f);
            }
        }
    };

    async_load_A(0, 0);
    async_load_B(0, 0);
    asm volatile("cp.async.commit_group;\n"::);

    if (num_k_tiles > 1) {
        async_load_A(1, BK);
        async_load_B(1, BK);
    }
    asm volatile("cp.async.commit_group;\n"::);

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % STAGES;

        asm volatile("cp.async.wait_group 1;\n"::);
        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA[WT_M];
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB[WT_N];

        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            int ko = k * WMMA_K;
            #pragma unroll
            for (int mi = 0; mi < WT_M; mi++)
                load_matrix_sync(fragA[mi],
                    &smA[stage * BM * SMA_STRIDE + (wm_base + mi*WMMA_M)*SMA_STRIDE + ko],
                    SMA_STRIDE);
            #pragma unroll
            for (int ni = 0; ni < WT_N; ni++)
                load_matrix_sync(fragB[ni],
                    &smB[stage * BN * SMB_STRIDE + (wn_base + ni*WMMA_N)*SMB_STRIDE + ko],
                    SMB_STRIDE);
            #pragma unroll
            for (int mi = 0; mi < WT_M; mi++)
                #pragma unroll
                for (int ni = 0; ni < WT_N; ni++)
                    mma_sync(acc[mi][ni], fragA[mi], fragB[ni], acc[mi][ni]);
        }

        int next2 = k_tile + 2;
        if (next2 < num_k_tiles) {
            int ns = next2 % STAGES;
            async_load_A(ns, next2 * BK);
            async_load_B(ns, next2 * BK);
        }
        asm volatile("cp.async.commit_group;\n"::);
    }

    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    float* smOut = reinterpret_cast<float*>(smem);

    #pragma unroll
    for (int mi = 0; mi < WT_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WT_N; ni++)
            store_matrix_sync(smOut + (wm_base + mi*WMMA_M)*BN + wn_base + ni*WMMA_N,
                              acc[mi][ni], BN, mem_row_major);

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int idx = tid + i * THREADS;
        int row = idx / BN;
        int col = idx % BN;
        int grow = bm + row;
        int gcol = bn + col;
        if (grow < M && gcol < N) {
            C[grow * N + gcol] = __float2half_rn(smOut[idx]);
        }
    }
}

static constexpr int BN128 = 128;
static constexpr int WT_N128 = 4;
static constexpr int SMEM_A128 = STAGES * BM * SMA_STRIDE;
static constexpr int SMEM_B128 = STAGES * BN128 * SMB_STRIDE;

__global__ void __launch_bounds__(256, 1)
hgemm_kernel_128x128_db(
    const half* __restrict__ A,
    const half* __restrict__ Bcm,
    half* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ half smem[];
    half* smA = smem;
    half* smB = smem + SMEM_A128;

    int bm = blockIdx.y * BM;
    int bn = blockIdx.x * BN128;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int wm_base = warp_m * (WT_M * WMMA_M);
    int wn_base = warp_n * (WT_N128 * WMMA_N);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WT_M][WT_N128];
    #pragma unroll
    for (int mi = 0; mi < WT_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WT_N128; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    int num_k_tiles = (K + BK - 1) / BK;

    auto async_load_A128 = [&](int stage, int k_off) {
        const half* Abase = A + bm * K + k_off;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int f4_idx = tid + i * THREADS;
            int row = f4_idx / 8;
            int col = (f4_idx % 8) * 8;
            int grow = bm + row;
            half* dst = &smA[stage * BM * SMA_STRIDE + row * SMA_STRIDE + col];
            if (grow < M && (k_off + col) < K) {
                uint32_t dst_ptr = smem_ptr(dst);
                const half* src = Abase + row * K + col;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst_ptr), "l"(src));
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0.f,0.f,0.f,0.f);
            }
        }
    };

    auto async_load_B128 = [&](int stage, int k_off) {
        const half* Bbase = Bcm + bn * K + k_off;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int f4_idx = tid + i * THREADS;
            int row = f4_idx / 8;
            int col = (f4_idx % 8) * 8;
            int gn = bn + row;
            half* dst = &smB[stage * BN128 * SMB_STRIDE + row * SMB_STRIDE + col];
            if (gn < N && (k_off + col) < K) {
                uint32_t dst_ptr = smem_ptr(dst);
                const half* src = Bbase + row * K + col;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(dst_ptr), "l"(src));
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0.f,0.f,0.f,0.f);
            }
        }
    };

    async_load_A128(0, 0);
    async_load_B128(0, 0);
    asm volatile("cp.async.commit_group;\n"::);

    if (num_k_tiles > 1) {
        async_load_A128(1, BK);
        async_load_B128(1, BK);
    }
    asm volatile("cp.async.commit_group;\n"::);

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % STAGES;

        asm volatile("cp.async.wait_group 1;\n"::);
        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA[WT_M];
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB[WT_N128];

        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            int ko = k * WMMA_K;
            #pragma unroll
            for (int mi = 0; mi < WT_M; mi++)
                load_matrix_sync(fragA[mi],
                    &smA[stage * BM * SMA_STRIDE + (wm_base+mi*WMMA_M)*SMA_STRIDE + ko],
                    SMA_STRIDE);
            #pragma unroll
            for (int ni = 0; ni < WT_N128; ni++)
                load_matrix_sync(fragB[ni],
                    &smB[stage * BN128 * SMB_STRIDE + (wn_base+ni*WMMA_N)*SMB_STRIDE + ko],
                    SMB_STRIDE);
            #pragma unroll
            for (int mi = 0; mi < WT_M; mi++)
                #pragma unroll
                for (int ni = 0; ni < WT_N128; ni++)
                    mma_sync(acc[mi][ni], fragA[mi], fragB[ni], acc[mi][ni]);
        }

        int next2 = k_tile + 2;
        if (next2 < num_k_tiles) {
            int ns = next2 % STAGES;
            async_load_A128(ns, next2*BK);
            async_load_B128(ns, next2*BK);
        }
        asm volatile("cp.async.commit_group;\n"::);
    }

    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    float* smOut = reinterpret_cast<float*>(smem);

    #pragma unroll
    for (int mi = 0; mi < WT_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WT_N128; ni++)
            store_matrix_sync(
                smOut + (wm_base+mi*WMMA_M)*BN128 + wn_base+ni*WMMA_N,
                acc[mi][ni], BN128, mem_row_major);

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        int idx = tid + i * THREADS;
        int row = idx / BN128;
        int col = idx % BN128;
        int grow = bm + row;
        int gcol = bn + col;
        if (grow < M && gcol < N)
            C[grow * N + gcol] = __float2half_rn(smOut[idx]);
    }
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
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (N % BN128 == 0) {
        constexpr int SMEM_128 = (SMEM_A128 + SMEM_B128) * sizeof(half);
        cudaFuncSetAttribute(hgemm_kernel_128x128_db,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        dim3 grid((N + BN128 - 1) / BN128, (M + BM - 1) / BM);
        hgemm_kernel_128x128_db<<<grid, dim3(THREADS), SMEM_128>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    } else {
        constexpr int SMEM_64 = (SMEM_A_SIZE + SMEM_B_SIZE) * sizeof(half);
        cudaFuncSetAttribute(hgemm_kernel_128x64_db,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        hgemm_kernel_128x64_db<<<grid, dim3(THREADS), SMEM_64>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
    }
}