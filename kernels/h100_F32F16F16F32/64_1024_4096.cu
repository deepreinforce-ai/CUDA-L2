#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

using namespace nvcuda;

constexpr int SPEC_M = 64;
constexpr int SPEC_N = 1024;
constexpr int SPEC_K = 4096;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int KSTEP   = 64;
constexpr int SPLIT_K = 8;

constexpr int A_PAD = 8;
constexpr int B_PAD = 8;
constexpr int LDS_A = KSTEP + A_PAD;
constexpr int LDS_B = BLOCK_N + B_PAD;

static float* g_partial = nullptr;
static size_t g_partial_elems = 0;

static inline void ensure_partial_buffer(size_t elems) {
    if (g_partial != nullptr && g_partial_elems >= elems) return;
    if (g_partial) {
        cudaFree(g_partial);
        g_partial = nullptr;
        g_partial_elems = 0;
    }
    cudaMalloc(&g_partial, elems * sizeof(float));
    g_partial_elems = elems;
}

__device__ __forceinline__ void copy_tile_A_vec8x2(
    half* __restrict__ dst, const half* __restrict__ src, int K, int k0, int tid)
{
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int vec_idx = tid + it * 256;
        int elem = vec_idx * 8;
        int r = elem / KSTEP;
        int c = elem - r * KSTEP;

        int64_t g_off = (int64_t)r * K + (k0 + c);
        int64_t s_off = (int64_t)r * LDS_A + c;

        *reinterpret_cast<int4*>(dst + s_off) = *reinterpret_cast<const int4*>(src + g_off);
    }
}

__device__ __forceinline__ void copy_tile_B_vec8x2(
    half* __restrict__ dst, const half* __restrict__ src, int N, int k0, int n_base, int tid)
{
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int vec_idx = tid + it * 256;
        int elem = vec_idx * 8;
        int r = elem / BLOCK_N;
        int c = elem - r * BLOCK_N;

        int64_t g_off = (int64_t)(k0 + r) * N + (n_base + c);
        int64_t s_off = (int64_t)r * LDS_B + c;

        *reinterpret_cast<int4*>(dst + s_off) = *reinterpret_cast<const int4*>(src + g_off);
    }
}

__global__ __launch_bounds__(256, 2)
void hgemm_splitk_wmma_k64_s8_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ Partial,
    int N, int K)
{
    __shared__ half As[2][BLOCK_M * LDS_A];
    __shared__ half Bs[2][KSTEP   * LDS_B];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;

    const int n_tile = blockIdx.x;
    const int split  = blockIdx.y;
    const int n_base = n_tile * BLOCK_N;

    const int k_tiles_total   = K / KSTEP;
    const int tiles_per_split = k_tiles_total / SPLIT_K;
    const int k_tile_begin    = split * tiles_per_split;

    const int m_group = warp_id & 3;
    const int n_group = warp_id >> 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    int stage = 0;

    {
        int k0 = k_tile_begin * KSTEP;
        copy_tile_A_vec8x2(&As[stage][0], A, K, k0, tid);
        copy_tile_B_vec8x2(&Bs[stage][0], B, N, k0, n_base, tid);
        __syncthreads();
    }

    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        const int global_kt = k_tile_begin + kt;
        const int next_stage = stage ^ 1;

        if (kt + 1 < 8) {
            int k_next = (global_kt + 1) * KSTEP;
            copy_tile_A_vec8x2(&As[next_stage][0], A, K, k_next, tid);
            copy_tile_B_vec8x2(&Bs[next_stage][0], B, N, k_next, n_base, tid);
        }

        const int a_row  = m_group * 16;
        const int b_col0 = n_group * 32;
        const int b_col1 = b_col0 + 16;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0, a1, a2, a3;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b00, b01, b10, b11, b20, b21, b30, b31;

        wmma::load_matrix_sync(a0,  &As[stage][a_row * LDS_A + 0],   LDS_A);
        wmma::load_matrix_sync(b00, &Bs[stage][0  * LDS_B + b_col0], LDS_B);
        wmma::load_matrix_sync(b01, &Bs[stage][0  * LDS_B + b_col1], LDS_B);
        wmma::mma_sync(acc0, a0, b00, acc0);
        wmma::mma_sync(acc1, a0, b01, acc1);

        wmma::load_matrix_sync(a1,  &As[stage][a_row * LDS_A + 16],  LDS_A);
        wmma::load_matrix_sync(b10, &Bs[stage][16 * LDS_B + b_col0], LDS_B);
        wmma::load_matrix_sync(b11, &Bs[stage][16 * LDS_B + b_col1], LDS_B);
        wmma::mma_sync(acc0, a1, b10, acc0);
        wmma::mma_sync(acc1, a1, b11, acc1);

        wmma::load_matrix_sync(a2,  &As[stage][a_row * LDS_A + 32],  LDS_A);
        wmma::load_matrix_sync(b20, &Bs[stage][32 * LDS_B + b_col0], LDS_B);
        wmma::load_matrix_sync(b21, &Bs[stage][32 * LDS_B + b_col1], LDS_B);
        wmma::mma_sync(acc0, a2, b20, acc0);
        wmma::mma_sync(acc1, a2, b21, acc1);

        wmma::load_matrix_sync(a3,  &As[stage][a_row * LDS_A + 48],  LDS_A);
        wmma::load_matrix_sync(b30, &Bs[stage][48 * LDS_B + b_col0], LDS_B);
        wmma::load_matrix_sync(b31, &Bs[stage][48 * LDS_B + b_col1], LDS_B);
        wmma::mma_sync(acc0, a3, b30, acc0);
        wmma::mma_sync(acc1, a3, b31, acc1);

        __syncthreads();
        stage = next_stage;
    }

    {
        const int c_row  = m_group * 16;
        const int c_col0 = n_group * 32;
        const int c_col1 = c_col0 + 16;

        float* base = Partial + (int64_t)split * SPEC_M * N;
        float* p0 = base + (int64_t)c_row * N + (n_base + c_col0);
        float* p1 = base + (int64_t)c_row * N + (n_base + c_col1);

        wmma::store_matrix_sync(p0, acc0, N, wmma::mem_row_major);
        wmma::store_matrix_sync(p1, acc1, N, wmma::mem_row_major);
    }
}

__global__ void reduce_splitk8_to_half_vec4_kernel(
    const float* __restrict__ Partial,
    half* __restrict__ C,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vec4 = SPEC_M * (N >> 2);
    if (idx >= total_vec4) return;

    int row = idx / (N >> 2);
    int n4  = (idx - row * (N >> 2)) << 2;

    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        int64_t off = ((int64_t)s * SPEC_M + row) * N + n4;
        float4 v = *reinterpret_cast<const float4*>(Partial + off);
        sum.x += v.x;
        sum.y += v.y;
        sum.z += v.z;
        sum.w += v.w;
    }

    int64_t c_off = (int64_t)row * N + n4;
    half2 h01 = __floats2half2_rn(sum.x, sum.y);
    half2 h23 = __floats2half2_rn(sum.z, sum.w);
    *reinterpret_cast<half2*>(C + c_off + 0) = h01;
    *reinterpret_cast<half2*>(C + c_off + 2) = h23;
}

void cuda_l2_h100_fp32(torch::Tensor a,
                                torch::Tensor b,
                                torch::Tensor b_col_major,
                                torch::Tensor c)
{
    (void)b_col_major;

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr());
    half* C_ptr       = reinterpret_cast<half*>(c.data_ptr());

    size_t partial_elems = (size_t)SPLIT_K * (size_t)SPEC_M * (size_t)SPEC_N;
    ensure_partial_buffer(partial_elems);

    cudaStream_t stream = 0;

    dim3 block_main(256, 1, 1);
    dim3 grid_main((unsigned)(SPEC_N / BLOCK_N), (unsigned)SPLIT_K, 1);
    hgemm_splitk_wmma_k64_s8_kernel<<<grid_main, block_main, 0, stream>>>(
        A_ptr, B_ptr, g_partial, SPEC_N, SPEC_K);

    int total_vec4 = SPEC_M * (SPEC_N >> 2);
    dim3 block_red(256, 1, 1);
    dim3 grid_red((unsigned)((total_vec4 + 255) / 256), 1, 1);
    reduce_splitk8_to_half_vec4_kernel<<<grid_red, block_red, 0, stream>>>(
        g_partial, C_ptr, SPEC_N);
}