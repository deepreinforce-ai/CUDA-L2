#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/types.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda::wmma;

namespace c10 {
namespace detail {
__attribute__((used)) void torchInternalAssertFail(
    const char* expr,
    const char* file,
    uint32_t line,
    const char* func,
    const std::string& msg) {
  fprintf(stderr,
          "torchInternalAssertFail: expr=%s file=%s line=%u func=%s msg=%s\n",
          expr ? expr : "(null)",
          file ? file : "(null)",
          line,
          func ? func : "(null)",
          msg.c_str());
  abort();
}
} // namespace detail
} // namespace c10

static constexpr int M_FIXED  = 2048;
static constexpr int K_FIXED  = 128;
static constexpr int N_FIXED  = 64;

static constexpr int BLOCK_M  = 64;
static constexpr int BLOCK_N  = 64;
static constexpr int BK       = 16;
static constexpr int NUM_BK   = 8;
static constexpr int B_LD     = 136;

__device__ __forceinline__ uint32_t smem_u32addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t s = smem_u32addr(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__global__ void __launch_bounds__(128, 8)
hgemm_h100_fast_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C)
{
  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int warp_m  = warp_id >> 1;
  const int warp_n  = warp_id & 1;

  const int block_m = blockIdx.x;
  const int row_base = block_m * BLOCK_M;

  __shared__ __align__(128) half smem_B[BLOCK_N][B_LD];
  __shared__ __align__(128) half smem_A[2][BLOCK_M][BK];

  {
    const int4* __restrict__ gB = reinterpret_cast<const int4*>(B_col_major);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      int t  = tid + i * 128;
      int n  = t >> 4;
      int v4 = t & 15;
      reinterpret_cast<int4*>(&smem_B[n][0])[v4] = gB[t];
    }
  }

  {
    int row = tid >> 1;
    int c8  = (tid & 1) * 8;
    int gr  = row_base + row;
    cp_async_16B(&smem_A[0][row][c8], &A[gr * K_FIXED + c8]);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  fragment<accumulator, 16, 16, 16, float> acc[2][2];
  #pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) fill_fragment(acc[mi][ni], 0.0f);
  }

  #pragma unroll
  for (int kb = 0; kb < NUM_BK; ++kb) {
    const int cur = kb & 1;
    const int nxt = cur ^ 1;

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      load_matrix_sync(a_frag[mi], &smem_A[cur][warp_m * 32 + mi * 16][0], BK);
    }

    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[2];
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
      load_matrix_sync(b_frag[ni], &smem_B[warp_n * 32 + ni * 16][kb * BK], B_LD);
    }

    if (kb + 1 < NUM_BK) {
      int row = tid >> 1;
      int c8  = (tid & 1) * 8;
      int gr  = row_base + row;
      int nk  = (kb + 1) * BK;
      cp_async_16B(&smem_A[nxt][row][c8], &A[gr * K_FIXED + nk + c8]);
      cp_async_commit();
    }

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      #pragma unroll
      for (int ni = 0; ni < 2; ++ni) {
        mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
      }
    }

    if (kb + 1 < NUM_BK) {
      cp_async_wait<0>();
      __syncthreads();
    }
  }

  #pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
      const int base_r = warp_m * 32 + mi * 16 + (lane_id >> 2);
      const int base_c = warp_n * 32 + ni * 16 + ((lane_id & 3) << 1);

      const int gr0 = row_base + base_r;
      const int gr1 = gr0 + 8;

      const float* f = reinterpret_cast<const float*>(&acc[mi][ni]);

      half2* c0 = reinterpret_cast<half2*>(&C[gr0 * N_FIXED + base_c]);
      c0[0] = __floats2half2_rn(f[0], f[1]);
      c0[4] = __floats2half2_rn(f[4], f[5]);

      half2* c1 = reinterpret_cast<half2*>(&C[gr1 * N_FIXED + base_c]);
      c1[0] = __floats2half2_rn(f[2], f[3]);
      c1[4] = __floats2half2_rn(f[6], f[7]);
    }
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
  (void)b;

  const half* A_ptr  = reinterpret_cast<const half*>(a.data_ptr());
  const half* Bc_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* C_ptr        = reinterpret_cast<half*>(c.data_ptr());

  dim3 block(128, 1, 1);
  dim3 grid(M_FIXED / BLOCK_M, 1, 1);

  hgemm_h100_fast_kernel<<<grid, block>>>(A_ptr, Bc_ptr, C_ptr);
}