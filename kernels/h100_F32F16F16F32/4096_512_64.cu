#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <stdexcept>
#include <string>

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int P_BM = 128;
static constexpr int P_BN = 64;
static constexpr int P_BK = 64;
static constexpr int P_WARPS = 4;
static constexpr int P_THREADS = P_WARPS * 32;
static constexpr int P_WM = 2;
static constexpr int P_WN = 2;
static constexpr int P_WTM = 4;
static constexpr int P_WTN = 2;
static constexpr int P_KITER = P_BK / WMMA_K;
static constexpr int P_PADA = 8;
static constexpr int P_PADB = 8;

__global__ void __launch_bounds__(P_THREADS, 4)
hgemm_kernel_128x64_4w(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half smem_A[P_BM][P_BK + P_PADA];
    __shared__ half smem_B[P_BN][P_BK + P_PADB];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_row = warp_id / P_WN;
    const int warp_col = warp_id % P_WN;

    const int bm = blockIdx.x * P_BM;
    const int bn = blockIdx.y * P_BN;
    const int wm0 = warp_row * (P_WTM * WMMA_M);
    const int wn0 = warp_col * (P_WTN * WMMA_N);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int chunk  = tid * 8 + i;
        const int m_loc  = chunk / (P_BK / 8);
        const int k_loc  = (chunk % (P_BK / 8)) * 8;
        const int gm     = bm + m_loc;
        if (gm < M) {
            *reinterpret_cast<float4*>(&smem_A[m_loc][k_loc]) =
                *reinterpret_cast<const float4*>(A + gm * K + k_loc);
        } else {
            *reinterpret_cast<float4*>(&smem_A[m_loc][k_loc]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int chunk  = tid * 4 + i;
        const int n_loc  = chunk / (P_BK / 8);
        const int k_loc  = (chunk % (P_BK / 8)) * 8;
        const int gn     = bn + n_loc;
        if (gn < N) {
            *reinterpret_cast<float4*>(&smem_B[n_loc][k_loc]) =
                *reinterpret_cast<const float4*>(B_col + gn * K + k_loc);
        } else {
            *reinterpret_cast<float4*>(&smem_B[n_loc][k_loc]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[P_WTM][P_WTN];
    #pragma unroll
    for (int i = 0; i < P_WTM; i++)
        #pragma unroll
        for (int j = 0; j < P_WTN; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < P_KITER; ki++) {
        const int k_off = ki * WMMA_K;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> af[P_WTM];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bf[P_WTN];

        #pragma unroll
        for (int i = 0; i < P_WTM; i++)
            wmma::load_matrix_sync(af[i], &smem_A[wm0 + i * WMMA_M][k_off], P_BK + P_PADA);

        #pragma unroll
        for (int j = 0; j < P_WTN; j++)
            wmma::load_matrix_sync(bf[j], &smem_B[wn0 + j * WMMA_N][k_off], P_BK + P_PADB);

        #pragma unroll
        for (int i = 0; i < P_WTM; i++)
            #pragma unroll
            for (int j = 0; j < P_WTN; j++)
                wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }

    #pragma unroll
    for (int i = 0; i < P_WTM; i++) {
        #pragma unroll
        for (int j = 0; j < P_WTN; j++) {
            const int gm = bm + wm0 + i * WMMA_M;
            const int gn = bn + wn0 + j * WMMA_N;
            if (gm + WMMA_M <= M && gn + WMMA_N <= N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out;
                #pragma unroll
                for (int t = 0; t < acc[i][j].num_elements; t++)
                    out.x[t] = __float2half(acc[i][j].x[t]);
                wmma::store_matrix_sync(C + gm * N + gn, out, N, wmma::mem_row_major);
            }
        }
    }
}

static constexpr int S_BM = 128;
static constexpr int S_BN = 64;
static constexpr int S_BK = 64;
static constexpr int S_WARPS = 8;
static constexpr int S_THREADS = S_WARPS * 32;
static constexpr int S_WM = 4;
static constexpr int S_WN = 2;
static constexpr int S_WTM = 2;
static constexpr int S_WTN = 2;
static constexpr int S_KITER = 4;
static constexpr int S_PADA = 8;
static constexpr int S_PADB = 8;

__global__ void __launch_bounds__(S_THREADS, 3)
hgemm_kernel_128x64_8w(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half smem_A[S_BM][S_BK + S_PADA];
    __shared__ half smem_B[S_BN][S_BK + S_PADB];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_row = warp_id / S_WN;
    const int warp_col = warp_id % S_WN;

    const int bm = blockIdx.x * S_BM;
    const int bn = blockIdx.y * S_BN;
    const int wm0 = warp_row * (S_WTM * WMMA_M);
    const int wn0 = warp_col * (S_WTN * WMMA_N);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int chunk = tid * 4 + i;
        const int m_loc = chunk / (S_BK / 8);
        const int k_loc = (chunk % (S_BK / 8)) * 8;
        const int gm    = bm + m_loc;
        if (gm < M) {
            *reinterpret_cast<float4*>(&smem_A[m_loc][k_loc]) =
                *reinterpret_cast<const float4*>(A + gm * K + k_loc);
        } else {
            *reinterpret_cast<float4*>(&smem_A[m_loc][k_loc]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int chunk = tid * 2 + i;
        const int n_loc = chunk / (S_BK / 8);
        const int k_loc = (chunk % (S_BK / 8)) * 8;
        const int gn    = bn + n_loc;
        if (gn < N) {
            *reinterpret_cast<float4*>(&smem_B[n_loc][k_loc]) =
                *reinterpret_cast<const float4*>(B_col + gn * K + k_loc);
        } else {
            *reinterpret_cast<float4*>(&smem_B[n_loc][k_loc]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[S_WTM][S_WTN];
    #pragma unroll
    for (int i = 0; i < S_WTM; i++)
        #pragma unroll
        for (int j = 0; j < S_WTN; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < S_KITER; ki++) {
        const int k_off = ki * WMMA_K;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> af[S_WTM];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bf[S_WTN];
        #pragma unroll
        for (int i = 0; i < S_WTM; i++)
            wmma::load_matrix_sync(af[i], &smem_A[wm0 + i * WMMA_M][k_off], S_BK + S_PADA);
        #pragma unroll
        for (int j = 0; j < S_WTN; j++)
            wmma::load_matrix_sync(bf[j], &smem_B[wn0 + j * WMMA_N][k_off], S_BK + S_PADB);
        #pragma unroll
        for (int i = 0; i < S_WTM; i++)
            #pragma unroll
            for (int j = 0; j < S_WTN; j++)
                wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }

    #pragma unroll
    for (int i = 0; i < S_WTM; i++) {
        #pragma unroll
        for (int j = 0; j < S_WTN; j++) {
            const int gm = bm + wm0 + i * WMMA_M;
            const int gn = bn + wn0 + j * WMMA_N;
            if (gm + WMMA_M <= M && gn + WMMA_N <= N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out;
                #pragma unroll
                for (int t = 0; t < acc[i][j].num_elements; t++)
                    out.x[t] = __float2half(acc[i][j].x[t]);
                wmma::store_matrix_sync(C + gm * N + gn, out, N, wmma::mem_row_major);
            }
        }
    }
}

static constexpr int T_BM = 64;
static constexpr int T_BN = 128;
static constexpr int T_BK = 64;
static constexpr int T_WARPS = 8;
static constexpr int T_THREADS = 256;
static constexpr int T_WM = 2;
static constexpr int T_WN = 4;
static constexpr int T_WTM = 2;
static constexpr int T_WTN = 2;
static constexpr int T_KITER = 4;
static constexpr int T_PADA = 8;
static constexpr int T_PADB = 8;

__global__ void __launch_bounds__(T_THREADS, 3)
hgemm_kernel_64x128_8w(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half smem_A[T_BM][T_BK + T_PADA];
    __shared__ half smem_B[T_BN][T_BK + T_PADB];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_row = warp_id / T_WN;
    const int warp_col = warp_id % T_WN;

    const int bm = blockIdx.x * T_BM;
    const int bn = blockIdx.y * T_BN;
    const int wm0 = warp_row * (T_WTM * WMMA_M);
    const int wn0 = warp_col * (T_WTN * WMMA_N);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int chunk = tid * 2 + i;
        const int m_loc = chunk / (T_BK / 8);
        const int k_loc = (chunk % (T_BK / 8)) * 8;
        const int gm    = bm + m_loc;
        if (gm < M) {
            *reinterpret_cast<float4*>(&smem_A[m_loc][k_loc]) =
                *reinterpret_cast<const float4*>(A + gm * K + k_loc);
        } else {
            *reinterpret_cast<float4*>(&smem_A[m_loc][k_loc]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int chunk = tid * 4 + i;
        const int n_loc = chunk / (T_BK / 8);
        const int k_loc = (chunk % (T_BK / 8)) * 8;
        const int gn    = bn + n_loc;
        if (gn < N) {
            *reinterpret_cast<float4*>(&smem_B[n_loc][k_loc]) =
                *reinterpret_cast<const float4*>(B_col + gn * K + k_loc);
        } else {
            *reinterpret_cast<float4*>(&smem_B[n_loc][k_loc]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[T_WTM][T_WTN];
    #pragma unroll
    for (int i = 0; i < T_WTM; i++)
        #pragma unroll
        for (int j = 0; j < T_WTN; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    #pragma unroll
    for (int ki = 0; ki < T_KITER; ki++) {
        const int k_off = ki * WMMA_K;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> af[T_WTM];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bf[T_WTN];
        #pragma unroll
        for (int i = 0; i < T_WTM; i++)
            wmma::load_matrix_sync(af[i], &smem_A[wm0 + i * WMMA_M][k_off], T_BK + T_PADA);
        #pragma unroll
        for (int j = 0; j < T_WTN; j++)
            wmma::load_matrix_sync(bf[j], &smem_B[wn0 + j * WMMA_N][k_off], T_BK + T_PADB);
        #pragma unroll
        for (int i = 0; i < T_WTM; i++)
            #pragma unroll
            for (int j = 0; j < T_WTN; j++)
                wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }

    #pragma unroll
    for (int i = 0; i < T_WTM; i++) {
        #pragma unroll
        for (int j = 0; j < T_WTN; j++) {
            const int gm = bm + wm0 + i * WMMA_M;
            const int gn = bn + wn0 + j * WMMA_N;
            if (gm + WMMA_M <= M && gn + WMMA_N <= N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out;
                #pragma unroll
                for (int t = 0; t < acc[i][j].num_elements; t++)
                    out.x[t] = __float2half(acc[i][j].x[t]);
                wmma::store_matrix_sync(C + gm * N + gn, out, N, wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4)
hgemm_kernel_direct(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int warp_id  = threadIdx.x >> 5;
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int bm  = blockIdx.x * 64;
    const int bn  = blockIdx.y * 128;
    const int wm  = bm + warp_row * 32;
    const int wn  = bn + warp_col * 32;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    if (wm + 32 <= M && wn + 32 <= N) {
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k = ki * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(af[i], A + (wm+i*16)*K + k, K);
            #pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(bf[j], B_col + (wn+j*16)*K + k, K);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out;
                #pragma unroll
                for (int t = 0; t < out.num_elements; t++)
                    out.x[t] = __float2half(acc[i][j].x[t]);
                wmma::store_matrix_sync(C + (wm+i*16)*N + (wn+j*16), out, N, wmma::mem_row_major);
            }
        }
    }
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const half* A     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       C     = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        dim3 grid((M + P_BM - 1) / P_BM, (N + P_BN - 1) / P_BN);
        dim3 block(P_THREADS);
        hgemm_kernel_128x64_4w<<<grid, block>>>(A, B_col, C, M, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((M + S_BM - 1) / S_BM, (N + S_BN - 1) / S_BN);
        dim3 block(S_THREADS);
        hgemm_kernel_128x64_8w<<<grid, block>>>(A, B_col, C, M, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((M + T_BM - 1) / T_BM, (N + T_BN - 1) / T_BN);
        dim3 block(T_THREADS);
        hgemm_kernel_64x128_8w<<<grid, block>>>(A, B_col, C, M, N, K);
        if (cudaGetLastError() == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid((M + 63) / 64, (N + 127) / 128);
        dim3 block(256);
        hgemm_kernel_direct<<<grid, block>>>(A, B_col, C, M, N, K);
        if (cudaGetLastError() != cudaSuccess)
            throw std::runtime_error("All HGEMM kernels failed");
    }
}