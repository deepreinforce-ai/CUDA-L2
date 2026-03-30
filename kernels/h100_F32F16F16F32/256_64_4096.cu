#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

#define NS_ATOM    128
#define BM_ATOM    64
#define BN_ATOM    64
#define BK_ATOM    32
#define NT_ATOM    128
#define NW_ATOM    4
#define SA_STR_A   40
#define SB_STR_A   72

#define NS_128     32
#define BK_128     128
#define SA_STR_128 136
#define SB_STR_128 72

static float* g_accum    = nullptr;
static size_t g_accum_sz = 0;

static void ensure_accum(size_t n) {
    if (n > g_accum_sz) {
        if (g_accum) cudaFree(g_accum);
        cudaMalloc(&g_accum, n);
        g_accum_sz = n;
    }
}

__global__ void __launch_bounds__(NT_ATOM, 6)
splitk_atom_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ accum,
    int M, int N, int K
) {
    const int m_block  = blockIdx.x;
    const int split_id = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;

    const int gm_base = m_block * BM_ATOM;
    const int gk_base = split_id * BK_ATOM;

    __shared__ __align__(128) half smA[BM_ATOM][SA_STR_A];
    __shared__ __align__(128) half smB[BK_ATOM][SB_STR_A];

    #pragma unroll
    for (int i = tid; i < (BM_ATOM * BK_ATOM) / 8; i += NT_ATOM) {
        int r = (i * 8) / BK_ATOM;
        int c = (i * 8) % BK_ATOM;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"((uint32_t)__cvta_generic_to_shared(&smA[r][c])),
            "l"((uint64_t)&A[(gm_base + r) * K + gk_base + c]));
    }

    #pragma unroll
    for (int i = tid; i < (BK_ATOM * BN_ATOM) / 8; i += NT_ATOM) {
        int r = (i * 8) / BN_ATOM;
        int c = (i * 8) % BN_ATOM;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"((uint32_t)__cvta_generic_to_shared(&smB[r][c])),
            "l"((uint64_t)&B[(gk_base + r) * N + c]));
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int w_row = warp_id * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> wacc[4];
    wmma::fill_fragment(wacc[0], 0.f);
    wmma::fill_fragment(wacc[1], 0.f);
    wmma::fill_fragment(wacc[2], 0.f);
    wmma::fill_fragment(wacc[3], 0.f);

    #pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb0, fb1, fb2, fb3;
        wmma::load_matrix_sync(fa,  &smA[w_row][kk * 16],   SA_STR_A);
        wmma::load_matrix_sync(fb0, &smB[kk * 16][0],       SB_STR_A);
        wmma::load_matrix_sync(fb1, &smB[kk * 16][16],      SB_STR_A);
        wmma::load_matrix_sync(fb2, &smB[kk * 16][32],      SB_STR_A);
        wmma::load_matrix_sync(fb3, &smB[kk * 16][48],      SB_STR_A);
        wmma::mma_sync(wacc[0], fa, fb0, wacc[0]);
        wmma::mma_sync(wacc[1], fa, fb1, wacc[1]);
        wmma::mma_sync(wacc[2], fa, fb2, wacc[2]);
        wmma::mma_sync(wacc[3], fa, fb3, wacc[3]);
    }

    __syncthreads();

    float* fbuf = reinterpret_cast<float*>(&smB[0][0]);
    float* wbuf = &fbuf[warp_id * 256];

    wmma::store_matrix_sync(&wbuf[0],   wacc[0], BN_ATOM, wmma::mem_row_major);
    wmma::store_matrix_sync(&wbuf[16],  wacc[1], BN_ATOM, wmma::mem_row_major);
    wmma::store_matrix_sync(&wbuf[32],  wacc[2], BN_ATOM, wmma::mem_row_major);
    wmma::store_matrix_sync(&wbuf[48],  wacc[3], BN_ATOM, wmma::mem_row_major);

    __syncthreads();

    float* out = &accum[gm_base * N];

    #pragma unroll 8
    for (int idx = tid; idx < BM_ATOM * BN_ATOM; idx += NT_ATOM) {
        int r   = idx / BN_ATOM;
        int c   = idx % BN_ATOM;
        int w   = r >> 4;
        int lr  = r & 15;
        float val = fbuf[w * 256 + lr * BN_ATOM + c];
        atomicAdd(&out[r * N + c], val);
    }
}

__global__ void __launch_bounds__(256)
convert_fp32_to_fp16(
    const float* __restrict__ accum,
    half*        __restrict__ C,
    int MN
) {
    int base = (blockIdx.x * 256 + threadIdx.x) * 2;
    if (base + 1 < MN) {
        *reinterpret_cast<half2*>(&C[base]) =
            __floats2half2_rn(accum[base], accum[base + 1]);
    } else if (base < MN) {
        C[base] = __float2half(accum[base]);
    }
}

__global__ void __launch_bounds__(256, 3)
splitk_bk128_atom_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ accum,
    int M, int N, int K
) {
    const int m_block  = blockIdx.x;
    const int split_id = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;

    const int gm_base = m_block * BM_ATOM;
    const int gk_base = split_id * BK_128;

    __shared__ __align__(128) half smA[BM_ATOM][SA_STR_128];
    __shared__ __align__(128) half smB[BK_128][SB_STR_128];

    for (int i = tid; i < (BM_ATOM * BK_128) / 8; i += 256) {
        int r = (i * 8) / BK_128;
        int c = (i * 8) % BK_128;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"((uint32_t)__cvta_generic_to_shared(&smA[r][c])),
            "l"((uint64_t)&A[(gm_base + r) * K + gk_base + c]));
    }
    for (int i = tid; i < (BK_128 * BN_ATOM) / 8; i += 256) {
        int r = (i * 8) / BN_ATOM;
        int c = (i * 8) % BN_ATOM;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"((uint32_t)__cvta_generic_to_shared(&smB[r][c])),
            "l"((uint64_t)&B[(gk_base + r) * N + c]));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int w_row  = (warp_id >> 1) * 16;
    const int w_ncol = (warp_id & 1) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> wacc[2];
    wmma::fill_fragment(wacc[0], 0.f);
    wmma::fill_fragment(wacc[1], 0.f);

    #pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb0, fb1;
        wmma::load_matrix_sync(fa,  &smA[w_row][kk * 16],        SA_STR_128);
        wmma::load_matrix_sync(fb0, &smB[kk * 16][w_ncol],      SB_STR_128);
        wmma::load_matrix_sync(fb1, &smB[kk * 16][w_ncol + 16], SB_STR_128);
        wmma::mma_sync(wacc[0], fa, fb0, wacc[0]);
        wmma::mma_sync(wacc[1], fa, fb1, wacc[1]);
    }

    __syncthreads();

    float* fbuf = reinterpret_cast<float*>(&smB[0][0]);
    float* wbuf = &fbuf[warp_id * 512];

    wmma::store_matrix_sync(&wbuf[0],  wacc[0], 32, wmma::mem_row_major);
    wmma::store_matrix_sync(&wbuf[16], wacc[1], 32, wmma::mem_row_major);

    __syncthreads();

    float* out = &accum[gm_base * N];

    #pragma unroll 4
    for (int idx = tid; idx < BM_ATOM * BN_ATOM; idx += 256) {
        int r   = idx / BN_ATOM;
        int c   = idx % BN_ATOM;
        int wr  = r >> 4;
        int wc  = c >> 5;
        int w   = (wr << 1) | wc;
        int lr  = r & 15;
        int lc  = c & 31;
        float val = fbuf[w * 512 + lr * 32 + lc];
        atomicAdd(&out[r * N + c], val);
    }
}

static float* g_partial    = nullptr;
static size_t g_partial_sz = 0;

static void ensure_partial(size_t n) {
    if (n > g_partial_sz) {
        if (g_partial) cudaFree(g_partial);
        cudaMalloc(&g_partial, n);
        g_partial_sz = n;
    }
}

__global__ void __launch_bounds__(256, 2)
splitk_bk128_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ partial,
    int M, int N, int K
) {
    const int m_block  = blockIdx.x;
    const int split_id = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;

    const int gm_base = m_block * BM_ATOM;
    const int gk_base = split_id * BK_128;

    __shared__ __align__(128) half smA[BM_ATOM][SA_STR_128];
    __shared__ __align__(128) half smB[BK_128][SB_STR_128];

    for (int i = tid; i < (BM_ATOM * BK_128) / 8; i += 256) {
        int r = (i * 8) / BK_128;
        int c = (i * 8) % BK_128;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
            "r"((uint32_t)__cvta_generic_to_shared(&smA[r][c])),
            "l"((uint64_t)&A[(gm_base + r) * K + gk_base + c]));
    }
    for (int i = tid; i < (BK_128 * BN_ATOM) / 8; i += 256) {
        int r = (i * 8) / BN_ATOM;
        int c = (i * 8) % BN_ATOM;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"((uint32_t)__cvta_generic_to_shared(&smB[r][c])),
            "l"((uint64_t)&B[(gk_base + r) * N + c]));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    const int w_row  = (warp_id >> 1) * 16;
    const int w_ncol = (warp_id & 1) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> wacc[2];
    wmma::fill_fragment(wacc[0], 0.f);
    wmma::fill_fragment(wacc[1], 0.f);

    #pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb0, fb1;
        wmma::load_matrix_sync(fa,  &smA[w_row][kk * 16],        SA_STR_128);
        wmma::load_matrix_sync(fb0, &smB[kk * 16][w_ncol],      SB_STR_128);
        wmma::load_matrix_sync(fb1, &smB[kk * 16][w_ncol + 16], SB_STR_128);
        wmma::mma_sync(wacc[0], fa, fb0, wacc[0]);
        wmma::mma_sync(wacc[1], fa, fb1, wacc[1]);
    }

    __syncthreads();

    float* fbuf = reinterpret_cast<float*>(&smB[0][0]);
    float* wbuf = &fbuf[warp_id * 512];

    wmma::store_matrix_sync(&wbuf[0],  wacc[0], 32, wmma::mem_row_major);
    wmma::store_matrix_sync(&wbuf[16], wacc[1], 32, wmma::mem_row_major);

    __syncthreads();

    float* out_base = partial + (size_t)split_id * M * N + (size_t)gm_base * N;

    #pragma unroll 4
    for (int idx = tid; idx < (BM_ATOM * BN_ATOM) / 4; idx += 256) {
        int lin = idx * 4;
        int r   = lin / BN_ATOM;
        int c   = lin % BN_ATOM;

        int wr  = r >> 4;
        int wc  = c >> 5;
        int w   = (wr << 1) | wc;
        int lr  = r & 15;
        int lc  = c & 31;

        const float* src = &fbuf[w * 512 + lr * 32 + lc];
        float4 v = {src[0], src[1], src[2], src[3]};
        *reinterpret_cast<float4*>(&out_base[r * N + c]) = v;
    }
}

__global__ void __launch_bounds__(256)
reduce_f4_kernel(
    const float* __restrict__ partial,
    half*        __restrict__ C,
    int MN
) {
    const int base = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (base >= MN) return;

    float4 acc = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int s = 0; s < NS_128; s++) {
        float4 v = *reinterpret_cast<const float4*>(&partial[(size_t)s * MN + base]);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }

    *reinterpret_cast<half2*>(&C[base + 0]) = __floats2half2_rn(acc.x, acc.y);
    *reinterpret_cast<half2*>(&C[base + 2]) = __floats2half2_rn(acc.z, acc.w);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M  = a.size(0);
    const int K  = a.size(1);
    const int N  = b.size(1);
    const int MN = M * N;

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C_ptr       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        size_t psz = (size_t)NS_128 * MN * sizeof(float);
        ensure_partial(psz);

        dim3 grid(M / BM_ATOM, NS_128);
        splitk_bk128_kernel<<<grid, 256>>>(A_ptr, B_ptr, g_partial, M, N, K);

        int red_blocks = (MN / 4 + 255) / 256;
        reduce_f4_kernel<<<red_blocks, 256>>>(g_partial, C_ptr, MN);
    }
}