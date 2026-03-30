#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

static constexpr int BM = 64;
static constexpr int BN = 128;
static constexpr int BK = 16;
static constexpr int K_FIXED = 128;
static constexpr int N_FIXED = 16384;

static constexpr int BLD = 24;

static constexpr int WARPS = 8;
static constexpr int THREADS = WARPS * 32;
static constexpr int WS_ELEMS = 16 * 16;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ unsigned smem_u32addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_ca_16B(void* smem_ptr, const void* gmem_ptr) {
    unsigned s = smem_u32addr(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

__device__ __forceinline__ void store_16x16_fp32_to_half2(
    const float* __restrict__ ws,
    half* __restrict__ C,
    int row_base,
    int col_base,
    int lane_id)
{
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        int idx = lane_id * 2 + t * 64;
        int rr = idx >> 4;
        int cc = idx & 15;

        half h0 = __float2half_rn(ws[idx]);
        half h1 = __float2half_rn(ws[idx + 1]);
        half2 hv = __halves2half2(h0, h1);

        half* dst = C + (row_base + rr) * N_FIXED + (col_base + cc);
        *reinterpret_cast<half2*>(dst) = hv;
    }
}

__global__ __launch_bounds__(THREADS, 3)
void hgemm_64x128_kernel(
    const half* __restrict__ A,
    const half* __restrict__ Bcol,
    half* __restrict__ C)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* As = reinterpret_cast<half*>(smem_raw);
    half* Bs0 = As + (BM * K_FIXED);
    half* Bs1 = Bs0 + (BN * BLD);
    float* WarpScratch = reinterpret_cast<float*>(Bs1 + (BN * BLD));

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int block_n = blockIdx.x * BN;

    {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
        uint4* As4 = reinterpret_cast<uint4*>(As);
        constexpr int A_VEC4 = (BM * K_FIXED) / 8;
        for (int i = tid; i < A_VEC4; i += THREADS) {
            As4[i] = A4[i];
        }
    }

    {
        int n = tid >> 1;
        int kk_base = (tid & 1) * 8;
        const half* gmem_src = Bcol + (block_n + n) * K_FIXED + kk_base;
        half* sptr = Bs0 + n * BLD + kk_base;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        cp_async_ca_16B(sptr, gmem_src);
        cp_async_commit();
        cp_async_wait_all();
#else
        *reinterpret_cast<uint4*>(sptr) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
    }

    __syncthreads();

    const int warp_m = warp_id >> 1;
    const int warp_ng = warp_id & 1;
    const int m0 = warp_m * 16;
    const int n_base = warp_ng * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1, acc2, acc3;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);
    wmma::fill_fragment(acc3, 0.0f);

    int stage = 0;

#pragma unroll
    for (int k0 = 0; k0 < K_FIXED; k0 += BK) {
        half* Bs = (stage == 0) ? Bs0 : Bs1;

        int next_k0 = k0 + BK;
        if (next_k0 < K_FIXED) {
            half* Bs_next = (stage == 0) ? Bs1 : Bs0;
            int n = tid >> 1;
            int kk_base = (tid & 1) * 8;
            const half* gmem_src = Bcol + (block_n + n) * K_FIXED + next_k0 + kk_base;
            half* sptr = Bs_next + n * BLD + kk_base;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            cp_async_ca_16B(sptr, gmem_src);
            cp_async_commit();
#else
            *reinterpret_cast<uint4*>(sptr) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag0, b_frag1, b_frag2, b_frag3;

        wmma::load_matrix_sync(a_frag, As + m0 * K_FIXED + k0, K_FIXED);
        wmma::load_matrix_sync(b_frag0, Bs + (n_base +  0) * BLD, BLD);
        wmma::load_matrix_sync(b_frag1, Bs + (n_base + 16) * BLD, BLD);
        wmma::load_matrix_sync(b_frag2, Bs + (n_base + 32) * BLD, BLD);
        wmma::load_matrix_sync(b_frag3, Bs + (n_base + 48) * BLD, BLD);

        wmma::mma_sync(acc0, a_frag, b_frag0, acc0);
        wmma::mma_sync(acc1, a_frag, b_frag1, acc1);
        wmma::mma_sync(acc2, a_frag, b_frag2, acc2);
        wmma::mma_sync(acc3, a_frag, b_frag3, acc3);

        if (next_k0 < K_FIXED) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            cp_async_wait_all();
#endif
            __syncthreads();
            stage ^= 1;
        }
    }

    float* ws = WarpScratch + warp_id * WS_ELEMS;

    wmma::store_matrix_sync(ws, acc0, 16, wmma::mem_row_major);
    store_16x16_fp32_to_half2(ws, C, m0, block_n + n_base + 0, lane_id);

    wmma::store_matrix_sync(ws, acc1, 16, wmma::mem_row_major);
    store_16x16_fp32_to_half2(ws, C, m0, block_n + n_base + 16, lane_id);

    wmma::store_matrix_sync(ws, acc2, 16, wmma::mem_row_major);
    store_16x16_fp32_to_half2(ws, C, m0, block_n + n_base + 32, lane_id);

    wmma::store_matrix_sync(ws, acc3, 16, wmma::mem_row_major);
    store_16x16_fp32_to_half2(ws, C, m0, block_n + n_base + 48, lane_id);
}

void cuda_l2_h100_fp32(torch::Tensor a,
                                torch::Tensor b,
                                torch::Tensor b_col_major,
                                torch::Tensor c)
{
    (void)b;

    const half* A_ptr  = reinterpret_cast<const half*>(a.data_ptr());
    const half* Bc_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* C_ptr        = reinterpret_cast<half*>(c.data_ptr());

    cudaStream_t stream = 0;

    dim3 block(THREADS);
    dim3 grid(N_FIXED / BN, 1, 1);

    size_t smem_bytes =
        (BM * K_FIXED) * sizeof(half) +
        (2 * BN * BLD) * sizeof(half) +
        (WARPS * WS_ELEMS) * sizeof(float);

    hgemm_64x128_kernel<<<grid, block, smem_bytes, stream>>>(A_ptr, Bc_ptr, C_ptr);
}