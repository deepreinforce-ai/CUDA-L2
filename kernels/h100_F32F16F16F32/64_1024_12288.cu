#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 0

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int M_T = 64;
constexpr int N_T = 1024;
constexpr int K_T = 12288;

constexpr int WMMA_K = 16;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;

constexpr int THREADS = 256;
constexpr int SPLIT_K = 24;

static_assert(K_T % SPLIT_K == 0, "K_T must be divisible by SPLIT_K");
static_assert((K_T / SPLIT_K) % BK == 0, "k_per_split must be divisible by BK");
static_assert((BK % WMMA_K) == 0, "BK must be multiple of WMMA_K");

static float* g_partial = nullptr;
static size_t g_partial_bytes = 0;

inline float* get_partial_buffer(size_t bytes_needed) {
    if (bytes_needed > g_partial_bytes) {
        if (g_partial) {
            cudaFree(g_partial);
            g_partial = nullptr;
            g_partial_bytes = 0;
        }
        if (cudaMalloc(&g_partial, bytes_needed) != cudaSuccess) {
            g_partial = nullptr;
            g_partial_bytes = 0;
            return nullptr;
        }
        g_partial_bytes = bytes_needed;
    }
    return g_partial;
}

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32), "l"(gmem_ptr));
#else
    *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__global__ void __launch_bounds__(THREADS, 2)
wmma_splitk_64x64x64_cpasync_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B_row,
    float* __restrict__ Partial)
{
    const int bn = blockIdx.x;
    const int ks = blockIdx.z;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;

    const int warp_m  = wid & 3;
    const int warp_ng = wid >> 2;

    const int n0 = bn * BN;

    constexpr int k_per_split = K_T / SPLIT_K;
    const int k_begin = ks * k_per_split;
    const int k_end   = k_begin + k_per_split;

    __shared__ __half smA[2][BM][BK + 8];
    __shared__ __half smB[2][BK][BN + 8];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag0, b_frag1;

    int buf = 0;

    {
        const int gk0 = k_begin;

        for (int v = tid; v < (BM * BK) / 8; v += THREADS) {
            int e = v * 8;
            int r = e / BK;
            int c = e % BK;
            const int4* gmem_src = reinterpret_cast<const int4*>(A + r * K_T + (gk0 + c));
            int4* sptr = reinterpret_cast<int4*>(&smA[buf][r][c]);
            cp_async_16B(sptr, gmem_src);
        }

        for (int v = tid; v < (BK * BN) / 8; v += THREADS) {
            int e = v * 8;
            int r = e / BN;
            int c = e % BN;
            const int4* gmem_src = reinterpret_cast<const int4*>(B_row + (gk0 + r) * N_T + (n0 + c));
            int4* sptr = reinterpret_cast<int4*>(&smB[buf][r][c]);
            cp_async_16B(sptr, gmem_src);
        }

        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
        const int next_k = k0 + BK;
        const int next_buf = 1 - buf;

        if (next_k < k_end) {
            for (int v = tid; v < (BM * BK) / 8; v += THREADS) {
                int e = v * 8;
                int r = e / BK;
                int c = e % BK;
                const int4* gmem_src = reinterpret_cast<const int4*>(A + r * K_T + (next_k + c));
                int4* sptr = reinterpret_cast<int4*>(&smA[next_buf][r][c]);
                cp_async_16B(sptr, gmem_src);
            }

            for (int v = tid; v < (BK * BN) / 8; v += THREADS) {
                int e = v * 8;
                int r = e / BN;
                int c = e % BN;
                const int4* gmem_src = reinterpret_cast<const int4*>(B_row + (next_k + r) * N_T + (n0 + c));
                int4* sptr = reinterpret_cast<int4*>(&smB[next_buf][r][c]);
                cp_async_16B(sptr, gmem_src);
            }
            cp_async_commit();
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            const int a_row  = warp_m * 16;
            const int b_col0 = warp_ng * 32 + 0;
            const int b_col1 = warp_ng * 32 + 16;

            wmma::load_matrix_sync(a_frag,  &smA[buf][a_row][kk], BK + 8);
            wmma::load_matrix_sync(b_frag0, &smB[buf][kk][b_col0], BN + 8);
            wmma::load_matrix_sync(b_frag1, &smB[buf][kk][b_col1], BN + 8);

            wmma::mma_sync(acc0, a_frag, b_frag0, acc0);
            wmma::mma_sync(acc1, a_frag, b_frag1, acc1);
        }

        if (next_k < k_end) cp_async_wait_all();
        __syncthreads();
        buf = next_buf;
    }

    float* split_base = Partial + static_cast<size_t>(ks) * (M_T * N_T);

    const int out_row  = warp_m * 16;
    const int out_col0 = n0 + warp_ng * 32 + 0;
    const int out_col1 = n0 + warp_ng * 32 + 16;

    wmma::store_matrix_sync(split_base + static_cast<size_t>(out_row) * N_T + out_col0,
                            acc0, N_T, wmma::mem_row_major);
    wmma::store_matrix_sync(split_base + static_cast<size_t>(out_row) * N_T + out_col1,
                            acc1, N_T, wmma::mem_row_major);
}

__global__ void reduce_splitk_fp32_to_fp16_vec4_kernel(
    const float* __restrict__ Partial,
    __half* __restrict__ C)
{
    constexpr int total = M_T * N_T;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base = t * 4;
    if (base >= total) return;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        const float4 v = *reinterpret_cast<const float4*>(
            Partial + static_cast<size_t>(s) * total + base
        );
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }

    C[base + 0] = __float2half_rn(acc.x);
    C[base + 1] = __float2half_rn(acc.y);
    C[base + 2] = __float2half_rn(acc.z);
    C[base + 3] = __float2half_rn(acc.w);
}

__global__ void fallback_gemm_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
    }
    C[m * N + n] = __float2half_rn(acc);
}

} // namespace

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)

{
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const __half* A = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* B = reinterpret_cast<const __half*>(b.data_ptr());
    __half* C = reinterpret_cast<__half*>(c.data_ptr());

    (void)b_col_major;

    if (M == M_T && N == N_T && K == K_T) {
        const size_t partial_elems = static_cast<size_t>(SPLIT_K) * M_T * N_T;
        float* partial = get_partial_buffer(partial_elems * sizeof(float));
        if (partial != nullptr) {
            dim3 grid(N_T / BN, 1, SPLIT_K);
            dim3 block(THREADS);
            wmma_splitk_64x64x64_cpasync_kernel<<<grid, block>>>(A, B, partial);

            constexpr int total = M_T * N_T;
            const int threads = 256;
            const int blocks = ((total / 4) + threads - 1) / threads;
            reduce_splitk_fp32_to_fp16_vec4_kernel<<<blocks, threads>>>(partial, C);
            return;
        }
    }

    dim3 block(256, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, M, 1);
    fallback_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}