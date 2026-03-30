#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <stdexcept>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

namespace c10 {
namespace detail {
[[noreturn]] void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const std::string& userMsg) {
  std::fprintf(stderr,
               "torchInternalAssertFail shim called.\n"
               "func: %s\nfile: %s\nline: %u\ncond: %s\nmsg: %s\n",
               func ? func : "(null)",
               file ? file : "(null)",
               line,
               condMsg ? condMsg : "(null)",
               userMsg.c_str());
  std::abort();
}
} // namespace detail
} // namespace c10

#define CHECK_CUDA(call)                                                                 \
  do {                                                                                   \
    cudaError_t _e = (call);                                                             \
    if (_e != cudaSuccess) {                                                             \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e));   \
    }                                                                                    \
  } while (0)

static constexpr int M_FIXED = 128;
static constexpr int N_FIXED = 256;
static constexpr int K_FIXED = 12288;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int BLOCK_K = 128;

static constexpr int THREADS = 256;
static constexpr int SPLIT_K = 32;

static constexpr int AS_LD = BLOCK_K + 8;
static constexpr int BS_LD = BLOCK_K + 8;
static constexpr int STAGES = 2;

static_assert(M_FIXED % BLOCK_M == 0, "M_FIXED must be divisible by BLOCK_M");
static_assert(N_FIXED % BLOCK_N == 0, "N_FIXED must be divisible by BLOCK_N");
static_assert(K_FIXED % SPLIT_K == 0, "K_FIXED must be divisible by SPLIT_K");
static_assert((K_FIXED / SPLIT_K) % BLOCK_K == 0, "K_PER_SPLIT must be divisible by BLOCK_K");
static_assert((M_FIXED * N_FIXED) % 4 == 0, "TOTAL must be divisible by 4");

__device__ __forceinline__ uint32_t smem_u32addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_ca_16B(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  uint32_t s = smem_u32addr(smem_ptr);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr));
#else
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_cg_16B(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  uint32_t s = smem_u32addr(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr));
#else
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__global__ __launch_bounds__(THREADS, 2)
void hgemm_splitk_wmma_cpasync_kernel(
    const half* __restrict__ A,
    const half* __restrict__ Bcol,
    float* __restrict__ partial) {

  extern __shared__ __align__(16) unsigned char smem_raw[];
  half* smem = reinterpret_cast<half*>(smem_raw);

  half* As0 = smem;
  half* Bs0 = As0 + BLOCK_M * AS_LD;
  half* As1 = Bs0 + BLOCK_N * BS_LD;
  half* Bs1 = As1 + BLOCK_M * AS_LD;

  const int tid      = threadIdx.x;
  const int warp_id  = tid >> 5;

  const int block_m  = blockIdx.y * BLOCK_M;
  const int block_n  = blockIdx.x * BLOCK_N;
  const int split_id = blockIdx.z;

  constexpr int K_PER_SPLIT = K_FIXED / SPLIT_K;
  constexpr int KTILES      = K_PER_SPLIT / BLOCK_K;
  const int k_begin = split_id * K_PER_SPLIT;

  const int warp_row = warp_id >> 1;
  const int warp_col = warp_id & 1;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    wmma::fill_fragment(c_frag[j], 0.0f);
  }

  auto issue_stage_copies = [&](half* As, half* Bs, int kk_base) {
    for (int vec = tid; vec < BLOCK_M * (BLOCK_K / 8); vec += THREADS) {
      int r  = vec / (BLOCK_K / 8);
      int kc = (vec % (BLOCK_K / 8)) * 8;
      const half* mem_ptr = A + (block_m + r) * K_FIXED + (kk_base + kc);
      half* sptr          = As + r * AS_LD + kc;
      cp_async_ca_16B(sptr, mem_ptr);
    }

    for (int vec = tid; vec < BLOCK_N * (BLOCK_K / 8); vec += THREADS) {
      int n  = vec / (BLOCK_K / 8);
      int kc = (vec % (BLOCK_K / 8)) * 8;
      const half* mem_ptr = Bcol + (block_n + n) * K_FIXED + (kk_base + kc);
      half* sptr          = Bs + n * BS_LD + kc;
      cp_async_cg_16B(sptr, mem_ptr);
    }

    cp_async_commit();
  };

  issue_stage_copies(As0, Bs0, k_begin);
  cp_async_wait_all();
  __syncthreads();

#pragma unroll 1
  for (int tile = 0; tile < KTILES; ++tile) {
    const bool curr0 = ((tile & 1) == 0);
    half* As_cur = curr0 ? As0 : As1;
    half* Bs_cur = curr0 ? Bs0 : Bs1;

    if (tile + 1 < KTILES) {
      const int kk_next = k_begin + (tile + 1) * BLOCK_K;
      half* As_nxt = curr0 ? As1 : As0;
      half* Bs_nxt = curr0 ? Bs1 : Bs0;
      issue_stage_copies(As_nxt, Bs_nxt, kk_next);
    }

    const int m_local = warp_row * WMMA_M;
    const int n_local = warp_col * 32;

#pragma unroll
    for (int ks = 0; ks < BLOCK_K; ks += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, &As_cur[m_local * AS_LD + ks], AS_LD);

      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b0, b1;
      wmma::load_matrix_sync(b0, &Bs_cur[(n_local + 0)  * BS_LD + ks], BS_LD);
      wmma::load_matrix_sync(b1, &Bs_cur[(n_local + 16) * BS_LD + ks], BS_LD);

      wmma::mma_sync(c_frag[0], a_frag, b0, c_frag[0]);
      wmma::mma_sync(c_frag[1], a_frag, b1, c_frag[1]);
    }

    if (tile + 1 < KTILES) {
      cp_async_wait_all();
      __syncthreads();
    }
  }

  const int split_base = split_id * (M_FIXED * N_FIXED);
  const int m0 = block_m + warp_row * WMMA_M;
  const int n0 = block_n + warp_col * 32;

  float* out0 = partial + split_base + m0 * N_FIXED + (n0 + 0);
  float* out1 = partial + split_base + m0 * N_FIXED + (n0 + 16);

  wmma::store_matrix_sync(out0, c_frag[0], N_FIXED, wmma::mem_row_major);
  wmma::store_matrix_sync(out1, c_frag[1], N_FIXED, wmma::mem_row_major);
}

__global__ __launch_bounds__(256, 2)
void reduce_splitk32_fp32_to_half_vec4_kernel(
    const float* __restrict__ partial,
    half* __restrict__ C) {

  constexpr int TOTAL = M_FIXED * N_FIXED;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = t * 4;
  if (idx >= TOTAL) return;

  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
  for (int s = 0; s < 32; ++s) {
    const float4 v = *reinterpret_cast<const float4*>(partial + s * TOTAL + idx);
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }

  reinterpret_cast<half2*>(C + idx)[0]     = __floats2half2_rn(acc.x, acc.y);
  reinterpret_cast<half2*>(C + idx + 2)[0] = __floats2half2_rn(acc.z, acc.w);
}

static float* g_partial = nullptr;
static size_t g_partial_elems = 0;

static void ensure_partial_workspace(size_t elems) {
  if (g_partial && g_partial_elems >= elems) return;
  if (g_partial) {
    cudaFree(g_partial);
    g_partial = nullptr;
    g_partial_elems = 0;
  }
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&g_partial), elems * sizeof(float)));
  g_partial_elems = elems;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
  const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
  const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* C_ptr       = reinterpret_cast<half*>(c.data_ptr());
  (void)b;

  const size_t partial_elems = (size_t)SPLIT_K * (size_t)M_FIXED * (size_t)N_FIXED;
  ensure_partial_workspace(partial_elems);

  dim3 grid(N_FIXED / BLOCK_N, M_FIXED / BLOCK_M, SPLIT_K);
  dim3 block(THREADS);

  const size_t smem_bytes =
      (size_t)STAGES * (size_t)(BLOCK_M * AS_LD + BLOCK_N * BS_LD) * sizeof(half);

  CHECK_CUDA(cudaFuncSetAttribute(
      hgemm_splitk_wmma_cpasync_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes));

  hgemm_splitk_wmma_cpasync_kernel<<<grid, block, smem_bytes>>>(A_ptr, B_ptr, g_partial);
  CHECK_CUDA(cudaGetLastError());

  constexpr int TOTAL = M_FIXED * N_FIXED;
  constexpr int RED_THREADS = 256;
  const int blocks = ((TOTAL / 4) + RED_THREADS - 1) / RED_THREADS;
  reduce_splitk32_fp32_to_half_vec4_kernel<<<blocks, RED_THREADS>>>(g_partial, C_ptr);
  CHECK_CUDA(cudaGetLastError());
}