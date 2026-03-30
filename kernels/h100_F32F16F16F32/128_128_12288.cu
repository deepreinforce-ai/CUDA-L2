#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int SPEC_M = 128;
constexpr int SPEC_N = 128;
constexpr int SPEC_K = 12288;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int SPLIT_K = 24;
constexpr int K_PER_SPLIT = SPEC_K / SPLIT_K;
static_assert(K_PER_SPLIT * SPLIT_K == SPEC_K, "K must be divisible by SPLIT_K");
static_assert((K_PER_SPLIT % WMMA_K) == 0, "K_PER_SPLIT must align with WMMA_K");

float* g_partial = nullptr;
size_t g_partial_elems = 0;

inline void ensure_partial_buffer(size_t elems) {
  if (g_partial_elems < elems) {
    if (g_partial) {
      cudaFree(g_partial);
      g_partial = nullptr;
      g_partial_elems = 0;
    }
    cudaMalloc(&g_partial, elems * sizeof(float));
    g_partial_elems = elems;
  }
}

__device__ __forceinline__ void load_stage_64x16_A_16x64_B(
    int tid,
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    int tile_m,
    int tile_n,
    int kk,
    half* sA_stage,
    half* sB_stage)
{
  {
    const int row = tid >> 1;
    const int col = (tid & 1) * 8;

    const half* mem_ptr = A + (tile_m + row) * SPEC_K + kk + col;
    half* sptr = sA_stage + row * 16 + col;

    *reinterpret_cast<int4*>(sptr) = *reinterpret_cast<const int4*>(mem_ptr);
  }

  {
    const int col = tid >> 1;
    const int row = (tid & 1) * 8;

    const half* mem_ptr = B_col_major + (tile_n + col) * SPEC_K + kk + row;
    half* sptr = sB_stage + col * 16 + row;

    *reinterpret_cast<int4*>(sptr) = *reinterpret_cast<const int4*>(mem_ptr);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void splitk_wmma_64x64_smem_pingpong_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    float* __restrict__ Partial)
{
  const int tile_n = blockIdx.x * BLOCK_N;
  const int tile_m = blockIdx.y * BLOCK_M;
  const int sk     = blockIdx.z;

  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;
  const int warp_row = warp_id >> 1;
  const int warp_col = warp_id & 1;

  const int m0 = tile_m + warp_row * 32;
  const int n0 = tile_n + warp_col * 32;

  const int k_begin = sk * K_PER_SPLIT;
  constexpr int ITERS = K_PER_SPLIT / WMMA_K;

  extern __shared__ half smem[];
  half* sA = smem;
  half* sB = sA + 2 * (64 * 16);

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::fill_fragment(c[i][j], 0.0f);
    }
  }

  load_stage_64x16_A_16x64_B(
      tid, A, B_col_major, tile_m, tile_n, k_begin,
      sA + 0 * (64 * 16), sB + 0 * (16 * 64));
  __syncthreads();

#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int cur = it & 1;
    const int nxt = cur ^ 1;

    half* sA_cur = sA + cur * (64 * 16);
    half* sB_cur = sB + cur * (16 * 64);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const half* a_ptr = sA_cur + (warp_row * 32 + i * 16) * 16;
      wmma::load_matrix_sync(a_frag[i], a_ptr, 16);
    }

#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const half* b_ptr = sB_cur + (warp_col * 32 + j * 16) * 16;
      wmma::load_matrix_sync(b_frag[j], b_ptr, 16);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        wmma::mma_sync(c[i][j], a_frag[i], b_frag[j], c[i][j]);
      }
    }

    if (it + 1 < ITERS) {
      const int kk_next = k_begin + (it + 1) * WMMA_K;
      load_stage_64x16_A_16x64_B(
          tid, A, B_col_major, tile_m, tile_n, kk_next,
          sA + nxt * (64 * 16), sB + nxt * (16 * 64));
    }

    __syncthreads();
  }

  float* partial_sk = Partial + static_cast<size_t>(sk) * (SPEC_M * SPEC_N);

  wmma::store_matrix_sync(partial_sk + (m0 +  0) * SPEC_N + (n0 +  0), c[0][0], SPEC_N, wmma::mem_row_major);
  wmma::store_matrix_sync(partial_sk + (m0 +  0) * SPEC_N + (n0 + 16), c[0][1], SPEC_N, wmma::mem_row_major);
  wmma::store_matrix_sync(partial_sk + (m0 + 16) * SPEC_N + (n0 +  0), c[1][0], SPEC_N, wmma::mem_row_major);
  wmma::store_matrix_sync(partial_sk + (m0 + 16) * SPEC_N + (n0 + 16), c[1][1], SPEC_N, wmma::mem_row_major);
}

__global__ __launch_bounds__(256, 4)
void reduce_splitk_cast_vec4_kernel(
    const float* __restrict__ Partial,
    half* __restrict__ C)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = t * 4;
  if (base >= SPEC_M * SPEC_N) return;

  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

#pragma unroll
  for (int sk = 0; sk < SPLIT_K; ++sk) {
    const float* p = Partial + static_cast<size_t>(sk) * (SPEC_M * SPEC_N) + base;
    float4 v = *reinterpret_cast<const float4*>(p);
    s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
  }

  half2 h01 = __float22half2_rn(make_float2(s0, s1));
  half2 h23 = __float22half2_rn(make_float2(s2, s3));

  reinterpret_cast<half2*>(C)[base >> 1] = h01;
  reinterpret_cast<half2*>(C)[(base >> 1) + 1] = h23;
}

} // namespace

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
  (void)b;

  const half* A_ptr  = reinterpret_cast<const half*>(a.data_ptr());
  const half* Bc_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* C_ptr        = reinterpret_cast<half*>(c.data_ptr());

  const size_t partial_elems = static_cast<size_t>(SPLIT_K) * SPEC_M * SPEC_N;
  ensure_partial_buffer(partial_elems);

  cudaStream_t stream = 0;

  dim3 block(THREADS_PER_BLOCK, 1, 1);
  dim3 grid(SPEC_N / BLOCK_N, SPEC_M / BLOCK_M, SPLIT_K);

  size_t smem_bytes = (2 * (64 * 16) + 2 * (16 * 64)) * sizeof(half);

  splitk_wmma_64x64_smem_pingpong_kernel<<<grid, block, smem_bytes, stream>>>(
      A_ptr, Bc_ptr, g_partial);

  const int threads = 256;
  const int blocks = ((SPEC_M * SPEC_N) / 4 + threads - 1) / threads;
  reduce_splitk_cast_vec4_kernel<<<blocks, threads, 0, stream>>>(g_partial, C_ptr);
}