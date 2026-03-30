#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

__global__ void __launch_bounds__(128, 6)
hgemm_optimized_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 64;

    const int cta_m = blockIdx.x * BM;
    if (cta_m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;

    __shared__ __align__(128) half sA[BM][BK + 8];

    {
        const int load_row = tid >> 2;
        const int load_col = (tid & 3) << 4;

        if (cta_m + load_row < M) {
            const half* src = A + (cta_m + load_row) * K + load_col;
            *reinterpret_cast<float4*>(&sA[load_row][load_col]) =
                *reinterpret_cast<const float4*>(src);
            *reinterpret_cast<float4*>(&sA[load_row][load_col + 8]) =
                *reinterpret_cast<const float4*>(src + 8);
        } else {
            *reinterpret_cast<float4*>(&sA[load_row][load_col]) = 
                make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(&sA[load_row][load_col + 8]) = 
                make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    __syncthreads();

    const int warp_m = (warp_id >> 1) * 16;
    const int warp_n = (warp_id & 1) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    const half* B0 = B_col_major + (warp_n +  0) * K;
    const half* B1 = B_col_major + (warp_n + 16) * K;

    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b0, frag_b1;

        wmma::load_matrix_sync(frag_a, &sA[warp_m][kk * 16], BK + 8);
        wmma::load_matrix_sync(frag_b0, B0 + kk * 16, K);
        wmma::load_matrix_sync(frag_b1, B1 + kk * 16, K);

        wmma::mma_sync(acc0, frag_a, frag_b0, acc0);
        wmma::mma_sync(acc1, frag_a, frag_b1, acc1);
    }

    __shared__ __align__(128) float sC[BM][BN + 8];
    
    wmma::store_matrix_sync(&sC[warp_m][warp_n], acc0, BN + 8, wmma::mem_row_major);
    wmma::store_matrix_sync(&sC[warp_m][warp_n + 16], acc1, BN + 8, wmma::mem_row_major);

    __syncthreads();

    {
        const int out_row = tid >> 3;
        const int out_col = (tid & 7) << 3;

        if (cta_m + out_row < M && out_col < N) {
            const float* src = &sC[out_row][out_col];
            half* dst = C + (cta_m + out_row) * N + out_col;

            const half2 h0 = __float22half2_rn(make_float2(src[0], src[1]));
            const half2 h1 = __float22half2_rn(make_float2(src[2], src[3]));
            const half2 h2 = __float22half2_rn(make_float2(src[4], src[5]));
            const half2 h3 = __float22half2_rn(make_float2(src[6], src[7]));

            float4 packed;
            reinterpret_cast<half2*>(&packed)[0] = h0;
            reinterpret_cast<half2*>(&packed)[1] = h1;
            reinterpret_cast<half2*>(&packed)[2] = h2;
            reinterpret_cast<half2*>(&packed)[3] = h3;

            *reinterpret_cast<float4*>(dst) = packed;
        }

        const int out_row2 = out_row + 16;
        if (cta_m + out_row2 < M && out_col < N) {
            const float* src = &sC[out_row2][out_col];
            half* dst = C + (cta_m + out_row2) * N + out_col;

            const half2 h0 = __float22half2_rn(make_float2(src[0], src[1]));
            const half2 h1 = __float22half2_rn(make_float2(src[2], src[3]));
            const half2 h2 = __float22half2_rn(make_float2(src[4], src[5]));
            const half2 h3 = __float22half2_rn(make_float2(src[6], src[7]));

            float4 packed;
            reinterpret_cast<half2*>(&packed)[0] = h0;
            reinterpret_cast<half2*>(&packed)[1] = h1;
            reinterpret_cast<half2*>(&packed)[2] = h2;
            reinterpret_cast<half2*>(&packed)[3] = h3;

            *reinterpret_cast<float4*>(dst) = packed;
        }
    }
}

__global__ void __launch_bounds__(128, 8)
hgemm_fallback_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int BM = 16;
    constexpr int BN = 64;
    constexpr int BK = 64;

    const int cta_m = blockIdx.x * BM;
    if (cta_m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;

    __shared__ __align__(128) half sA[BM][BK + 8];
    __shared__ __align__(128) float sC[BM][BN + 8];

    {
        const int load_row = tid >> 3;
        const int load_col = (tid & 7) << 3;

        if (load_row < BM) {
            if (cta_m + load_row < M) {
                const half* src = A + (cta_m + load_row) * K + load_col;
                *reinterpret_cast<float4*>(&sA[load_row][load_col]) =
                    *reinterpret_cast<const float4*>(src);
            } else {
                *reinterpret_cast<float4*>(&sA[load_row][load_col]) = 
                    make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }

    __syncthreads();

    const int warp_n = warp_id << 4;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const half* B_ptr = B_col_major + warp_n * K;

    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;

        wmma::load_matrix_sync(frag_a, &sA[0][kk * 16], BK + 8);
        wmma::load_matrix_sync(frag_b, B_ptr + kk * 16, K);

        wmma::mma_sync(acc, frag_a, frag_b, acc);
    }

    wmma::store_matrix_sync(&sC[0][warp_n], acc, BN + 8, wmma::mem_row_major);

    __syncthreads();

    {
        const int out_row = tid >> 3;
        const int out_col = (tid & 7) << 3;

        if (out_row < BM && cta_m + out_row < M && out_col < N) {
            const float* src = &sC[out_row][out_col];
            half* dst = C + (cta_m + out_row) * N + out_col;

            const half2 h0 = __float22half2_rn(make_float2(src[0], src[1]));
            const half2 h1 = __float22half2_rn(make_float2(src[2], src[3]));
            const half2 h2 = __float22half2_rn(make_float2(src[4], src[5]));
            const half2 h3 = __float22half2_rn(make_float2(src[6], src[7]));

            float4 packed;
            reinterpret_cast<half2*>(&packed)[0] = h0;
            reinterpret_cast<half2*>(&packed)[1] = h1;
            reinterpret_cast<half2*>(&packed)[2] = h2;
            reinterpret_cast<half2*>(&packed)[3] = h3;

            *reinterpret_cast<float4*>(dst) = packed;
        }
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if (((T).options().dtype() != (th_type))) {                                \
        std::cout << "Tensor Info:" << (T).options() << std::endl;             \
        throw std::runtime_error("values must be " #th_type);                 \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                      \
        throw std::runtime_error("Tensor size mismatch!");                     \
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

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const half* a_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* b_col_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* c_ptr = reinterpret_cast<half*>(c.data_ptr());

    if (M % 32 == 0) {
        const int num_blocks = M / 32;
        hgemm_optimized_kernel<<<num_blocks, 128>>>(a_ptr, b_col_ptr, c_ptr, M, N, K);
    } else {
        const int num_blocks = (M + 15) / 16;
        hgemm_fallback_kernel<<<num_blocks, 128>>>(a_ptr, b_col_ptr, c_ptr, M, N, K);
    }
}