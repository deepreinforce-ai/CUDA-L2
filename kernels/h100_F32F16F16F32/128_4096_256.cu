#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

using namespace nvcuda;

static constexpr int BM = 32;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int NUM_WARPS = 4;
static constexpr int THREADS = 128;
static constexpr int WN = BN / NUM_WARPS;
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 16;
static constexpr int MMA_K = 16;
static constexpr int WARP_M = BM / MMA_M;
static constexpr int WARP_N = WN / MMA_N;
static constexpr int WARP_K = BK / MMA_K;
static constexpr int K_STEPS = 256 / BK;

static constexpr int SA = BK + 8;
static constexpr int SB = BN + 8;

__device__ __forceinline__
void cp16(void* dst, const void* src) {
    uint32_t addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(addr), "l"(src) : "memory");
}

__global__ __launch_bounds__(THREADS, 8)
void hgemm_optimized(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int warp_id = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int warp_n_base = warp_id * WN;

    __shared__ __align__(128) __half smA[2][BM * SA];
    __shared__ __align__(128) __half smB[2][BK * SB];

    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> acc[WARP_M][WARP_N];
    #pragma unroll
    for (int mi = 0; mi < WARP_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < WARP_N; ++ni)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    const __half* A_base = A + bm * K;
    const __half* B_base = B + bn;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int lin = tid * 16 + i * 8;
        int row = lin / BK;
        int col = lin % BK;
        cp16(&smA[0][row * SA + col], &A_base[row * K + col]);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int lin = tid * 64 + i * 8;
        int row = lin / BN;
        int col = lin % BN;
        cp16(&smB[0][row * SB + col], &B_base[row * N + col]);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    #pragma unroll 1
    for (int ks = 0; ks < K_STEPS; ++ks) {
        const int cur = ks & 1;
        const int nxt = cur ^ 1;
        const int k_next = (ks + 1) * BK;

        if (ks + 1 < K_STEPS) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int lin = tid * 16 + i * 8;
                int row = lin / BK;
                int col = lin % BK;
                cp16(&smA[nxt][row * SA + col], &A_base[row * K + k_next + col]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int lin = tid * 64 + i * 8;
                int row = lin / BN;
                int col = lin % BN;
                cp16(&smB[nxt][row * SB + col], &B_base[(k_next + row) * N + col]);
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        #pragma unroll
        for (int ki = 0; ki < WARP_K; ++ki) {
            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __half, wmma::row_major> af[WARP_M];
            #pragma unroll
            for (int mi = 0; mi < WARP_M; ++mi) {
                wmma::load_matrix_sync(
                    af[mi],
                    &smA[cur][(mi * MMA_M) * SA + ki * MMA_K],
                    SA
                );
            }
            #pragma unroll
            for (int ni = 0; ni < WARP_N; ++ni) {
                wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __half, wmma::row_major> bf;
                wmma::load_matrix_sync(
                    bf,
                    &smB[cur][(ki * MMA_K) * SB + warp_n_base + ni * MMA_N],
                    SB
                );
                #pragma unroll
                for (int mi = 0; mi < WARP_M; ++mi) {
                    wmma::mma_sync(acc[mi][ni], af[mi], bf, acc[mi][ni]);
                }
            }
        }

        if (ks + 1 < K_STEPS) {
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            __syncthreads();
        }
    }

    __syncthreads();

    float* so = reinterpret_cast<float*>(smB);

    #pragma unroll
    for (int mi = 0; mi < WARP_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_N; ++ni) {
            wmma::store_matrix_sync(
                &so[(mi * MMA_M) * BN + warp_n_base + ni * MMA_N],
                acc[mi][ni],
                BN,
                wmma::mem_row_major
            );
        }
    }

    __syncthreads();

    #pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        int base = tid * 8 + iter * 1024;
        int row = base / BN;
        int col = base % BN;
        __half tmp[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            tmp[e] = __float2half(so[base + e]);
        }
        *reinterpret_cast<float4*>(&C[(bm + row) * N + bn + col]) =
            *reinterpret_cast<const float4*>(tmp);
    }
}

#define CHECK_DTYPE(T, th_type) \
    if ((T).options().dtype() != (th_type)) throw std::runtime_error("dtype mismatch");

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_DTYPE(a, torch::kHalf)
    CHECK_DTYPE(b, torch::kHalf)
    CHECK_DTYPE(b_col_major, torch::kHalf)
    CHECK_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const __half* A_ptr = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    __half*       C_ptr = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    const dim3 grid(N / BN, M / BM);
    const dim3 block(THREADS);

    hgemm_optimized<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("HGEMM failed: ") + cudaGetErrorString(err));
    }
}