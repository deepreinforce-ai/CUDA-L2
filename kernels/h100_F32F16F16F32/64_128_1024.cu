#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

static __device__ __forceinline__ uint32_t smem_u32addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t s = smem_u32addr(smem_ptr);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr));
#else
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

static __device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__global__ __launch_bounds__(128, 8)
void hgemm_64x128x1024_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C) {
  constexpr int N = 128;
  constexpr int K = 1024;

  constexpr int CTA_M = 32;
  constexpr int CTA_N = 32;
  constexpr int BK    = 128;

  constexpr int WARPS   = 4;
  constexpr int THREADS = WARPS * 32;

  constexpr int SA_STRIDE = BK + 8;
  constexpr int SB_STRIDE = BK + 8;
  constexpr int A_STAGE_ELEMS = CTA_M * SA_STRIDE;
  constexpr int B_STAGE_ELEMS = CTA_N * SB_STRIDE;

  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int tid     = warp_id * 32 + lane_id;

  const int block_m0 = blockIdx.y * CTA_M;
  const int block_n0 = blockIdx.x * CTA_N;

  const int warp_m = warp_id >> 1;
  const int warp_n = warp_id & 1;

  const int c_row_local = warp_m * 16;
  const int c_col_local = warp_n * 16;

  __shared__ half  sA[2][A_STAGE_ELEMS];
  __shared__ half  sB[2][B_STAGE_ELEMS];
  __shared__ float sWarpOut[WARPS][16 * 16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

  int stage = 0;
  {
    constexpr int A_CHUNKS_PER_ROW = BK / 8;
    constexpr int A_TOTAL_CHUNKS   = CTA_M * A_CHUNKS_PER_ROW;
    for (int ch = tid; ch < A_TOTAL_CHUNKS; ch += THREADS) {
      int r = ch / A_CHUNKS_PER_ROW;
      int c8 = ch % A_CHUNKS_PER_ROW;
      int k_off = c8 * 8;
      const half* mem_ptr = A + (block_m0 + r) * K + k_off;
      half* sptr = &sA[stage][r * SA_STRIDE + k_off];
      cp_async_16B(sptr, mem_ptr);
    }

    constexpr int B_CHUNKS_PER_ROW = BK / 8;
    constexpr int B_TOTAL_CHUNKS   = CTA_N * B_CHUNKS_PER_ROW;
    for (int ch = tid; ch < B_TOTAL_CHUNKS; ch += THREADS) {
      int n_local = ch / B_CHUNKS_PER_ROW;
      int c8      = ch % B_CHUNKS_PER_ROW;
      int k_off   = c8 * 8;
      int gcol    = block_n0 + n_local;
      const half* mem_ptr = B_col + gcol * K + k_off;
      half* sptr = &sB[stage][n_local * SB_STRIDE + k_off];
      cp_async_16B(sptr, mem_ptr);
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
  }

  #pragma unroll 1
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int read_stage = stage;
    const int next_k = k0 + BK;

    if (next_k < K) {
      const int next_stage = read_stage ^ 1;

      constexpr int A_CHUNKS_PER_ROW = BK / 8;
      constexpr int A_TOTAL_CHUNKS   = CTA_M * A_CHUNKS_PER_ROW;
      for (int ch = tid; ch < A_TOTAL_CHUNKS; ch += THREADS) {
        int r = ch / A_CHUNKS_PER_ROW;
        int c8 = ch % A_CHUNKS_PER_ROW;
        int k_off = c8 * 8;
        const half* mem_ptr = A + (block_m0 + r) * K + (next_k + k_off);
        half* sptr = &sA[next_stage][r * SA_STRIDE + k_off];
        cp_async_16B(sptr, mem_ptr);
      }

      constexpr int B_CHUNKS_PER_ROW = BK / 8;
      constexpr int B_TOTAL_CHUNKS   = CTA_N * B_CHUNKS_PER_ROW;
      for (int ch = tid; ch < B_TOTAL_CHUNKS; ch += THREADS) {
        int n_local = ch / B_CHUNKS_PER_ROW;
        int c8      = ch % B_CHUNKS_PER_ROW;
        int k_off   = c8 * 8;
        int gcol    = block_n0 + n_local;
        const half* mem_ptr = B_col + gcol * K + (next_k + k_off);
        half* sptr = &sB[next_stage][n_local * SB_STRIDE + k_off];
        cp_async_16B(sptr, mem_ptr);
      }

      cp_async_commit();
    }

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      const half* a_ptr = &sA[read_stage][c_row_local * SA_STRIDE + kk];
      const half* b_ptr = &sB[read_stage][c_col_local * SB_STRIDE + kk];
      wmma::load_matrix_sync(a_frag, a_ptr, SA_STRIDE);
      wmma::load_matrix_sync(b_frag, b_ptr, SB_STRIDE);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    if (next_k < K) {
      cp_async_wait_all();
      __syncthreads();
      stage ^= 1;
    }
  }

  float* warp_buf = &sWarpOut[warp_id][0];
  wmma::store_matrix_sync(warp_buf, acc, 16, wmma::mem_row_major);
  __syncwarp();

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int pair_id = lane_id + i * 32;
    int e0 = pair_id * 2;

    int rr = e0 >> 4;
    int cc = e0 & 15;

    float f0 = warp_buf[rr * 16 + cc];
    float f1 = warp_buf[rr * 16 + (cc + 1)];
    half2 h2 = __halves2half2(__float2half_rn(f0), __float2half_rn(f1));

    int g_row = block_m0 + c_row_local + rr;
    int g_col = block_n0 + c_col_local + cc;
    reinterpret_cast<half2*>(C + g_row * N + g_col)[0] = h2;
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
  const half* A_ptr  = reinterpret_cast<const half*>(a.data_ptr());
  const half* Bc_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* C_ptr        = reinterpret_cast<half*>(c.data_ptr());
  (void)b;

  dim3 block(32, 4, 1);
  dim3 grid(4, 2, 1);

  hgemm_64x128x1024_kernel<<<grid, block, 0, 0>>>(A_ptr, Bc_ptr, C_ptr);
}