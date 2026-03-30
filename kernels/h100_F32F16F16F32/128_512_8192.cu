#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <stdexcept>

using namespace nvcuda;

namespace {

constexpr int TARGET_M = 128;
constexpr int TARGET_N = 512;
constexpr int TARGET_K = 8192;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

constexpr int WARPS_PER_BLOCK   = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int SPLIT_K_FAST = 8;
constexpr int K_TILES_TOTAL = TARGET_K / WMMA_K;
constexpr int K_TILES_PER_SPLIT = K_TILES_TOTAL / SPLIT_K_FAST;

constexpr int SMEM_PAD_K = 8;
constexpr int SMEM_STRIDE_K = WMMA_K + SMEM_PAD_K;

float*  g_split_workspace = nullptr;
size_t  g_split_workspace_elems = 0;

inline bool ensure_split_workspace(size_t elems) {
  if (g_split_workspace_elems < elems) {
    if (g_split_workspace) {
      cudaFree(g_split_workspace);
      g_split_workspace = nullptr;
      g_split_workspace_elems = 0;
    }
    cudaError_t err = cudaMalloc(&g_split_workspace, elems * sizeof(float));
    if (err != cudaSuccess) {
      g_split_workspace = nullptr;
      g_split_workspace_elems = 0;
      return false;
    }
    g_split_workspace_elems = elems;
  }
  return g_split_workspace != nullptr;
}

__device__ __forceinline__ void load_stage_vec128_padded(
    half* __restrict__ As_stage,
    half* __restrict__ Bs_stage,
    const half* __restrict__ A,
    const half* __restrict__ Bcol,
    int block_row, int block_col,
    int k0, int tid) {

  int linear_half = tid * 8;
  int r = linear_half >> 4;
  int c = linear_half & 15;

  const half* a_src = A    + (block_row + r) * TARGET_K + (k0 + c);
  const half* b_src = Bcol + (block_col + r) * TARGET_K + (k0 + c);

  half* a_dst = As_stage + r * SMEM_STRIDE_K + c;
  half* b_dst = Bs_stage + r * SMEM_STRIDE_K + c;

  int4 va = *reinterpret_cast<const int4*>(a_src);
  int4 vb = *reinterpret_cast<const int4*>(b_src);
  *reinterpret_cast<int4*>(a_dst) = va;
  *reinterpret_cast<int4*>(b_dst) = vb;
}

template<int SPLITK>
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void splitk_wmma_smem_vec128_64x64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ Bcol,
    float* __restrict__ partial) {

  __shared__ half As[2][BLOCK_M][SMEM_STRIDE_K];
  __shared__ half Bs[2][BLOCK_N][SMEM_STRIDE_K];

  const int tile_n = blockIdx.x;
  const int tile_m = blockIdx.y;
  const int split  = blockIdx.z;

  const int block_row = tile_m * BLOCK_M;
  const int block_col = tile_n * BLOCK_N;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int c_row = block_row + warp_id * WMMA_M;

  const int kt_begin = split * K_TILES_PER_SPLIT;

  {
    int k0 = kt_begin * WMMA_K;
    load_stage_vec128_padded(&As[0][0][0], &Bs[0][0][0], A, Bcol, block_row, block_col, k0, tid);
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0, acc1, acc2, acc3;
  wmma::fill_fragment(acc0, 0.0f);
  wmma::fill_fragment(acc1, 0.0f);
  wmma::fill_fragment(acc2, 0.0f);
  wmma::fill_fragment(acc3, 0.0f);

  #pragma unroll
  for (int lkt = 0; lkt < K_TILES_PER_SPLIT; ++lkt) {
    int curr = lkt & 1;
    int next = curr ^ 1;

    if (lkt + 1 < K_TILES_PER_SPLIT) {
      int k0n = (kt_begin + lkt + 1) * WMMA_K;
      load_stage_vec128_padded(&As[next][0][0], &Bs[next][0][0], A, Bcol, block_row, block_col, k0n, tid);
    }

    const half* a_ptr  = &As[curr][warp_id * WMMA_M][0];
    const half* b0_ptr = &Bs[curr][ 0][0];
    const half* b1_ptr = &Bs[curr][16][0];
    const half* b2_ptr = &Bs[curr][32][0];
    const half* b3_ptr = &Bs[curr][48][0];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b0_frag, b1_frag, b2_frag, b3_frag;

    wmma::load_matrix_sync(a_frag,  a_ptr,  SMEM_STRIDE_K);
    wmma::load_matrix_sync(b0_frag, b0_ptr, SMEM_STRIDE_K);
    wmma::load_matrix_sync(b1_frag, b1_ptr, SMEM_STRIDE_K);
    wmma::load_matrix_sync(b2_frag, b2_ptr, SMEM_STRIDE_K);
    wmma::load_matrix_sync(b3_frag, b3_ptr, SMEM_STRIDE_K);

    wmma::mma_sync(acc0, a_frag, b0_frag, acc0);
    wmma::mma_sync(acc1, a_frag, b1_frag, acc1);
    wmma::mma_sync(acc2, a_frag, b2_frag, acc2);
    wmma::mma_sync(acc3, a_frag, b3_frag, acc3);

    __syncthreads();
  }

  float* split_base = partial + (size_t)split * (size_t)TARGET_M * (size_t)TARGET_N;
  wmma::store_matrix_sync(split_base + c_row * TARGET_N + (block_col +  0), acc0, TARGET_N, wmma::mem_row_major);
  wmma::store_matrix_sync(split_base + c_row * TARGET_N + (block_col + 16), acc1, TARGET_N, wmma::mem_row_major);
  wmma::store_matrix_sync(split_base + c_row * TARGET_N + (block_col + 32), acc2, TARGET_N, wmma::mem_row_major);
  wmma::store_matrix_sync(split_base + c_row * TARGET_N + (block_col + 48), acc3, TARGET_N, wmma::mem_row_major);
}

template <int SPLITK>
__global__ void reduce_splitk_vec8_fp32_to_fp16_kernel(
    const float* __restrict__ partial,
    half* __restrict__ C,
    int MN) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * 8;
  if (base >= MN) return;

  float4 s0 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 s1 = make_float4(0.f, 0.f, 0.f, 0.f);

  #pragma unroll
  for (int s = 0; s < SPLITK; ++s) {
    const float* p = partial + (size_t)s * (size_t)MN + base;
    float4 v0 = *reinterpret_cast<const float4*>(p + 0);
    float4 v1 = *reinterpret_cast<const float4*>(p + 4);
    s0.x += v0.x; s0.y += v0.y; s0.z += v0.z; s0.w += v0.w;
    s1.x += v1.x; s1.y += v1.y; s1.z += v1.z; s1.w += v1.w;
  }

  if (base + 7 < MN) {
    C[base + 0] = __float2half_rn(s0.x);
    C[base + 1] = __float2half_rn(s0.y);
    C[base + 2] = __float2half_rn(s0.z);
    C[base + 3] = __float2half_rn(s0.w);
    C[base + 4] = __float2half_rn(s1.x);
    C[base + 5] = __float2half_rn(s1.y);
    C[base + 6] = __float2half_rn(s1.z);
    C[base + 7] = __float2half_rn(s1.w);
  }
}

__global__ void fallback_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ Bcol,
    half* __restrict__ C,
    int M, int N, int K) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += __half2float(A[row * K + k]) * __half2float(Bcol[col * K + k]);
  }
  C[row * N + col] = __float2half_rn(acc);
}

} // namespace

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)

{
  if (!a.is_cuda() || !b.is_cuda() || !b_col_major.is_cuda() || !c.is_cuda()) {
    throw std::runtime_error("All tensors must be CUDA tensors.");
  }

  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

  if (M <= 0 || N <= 0 || K <= 0) return;

  const half* A    = reinterpret_cast<const half*>(a.data_ptr());
  const half* Bcol = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* C          = reinterpret_cast<half*>(c.data_ptr());

  cudaStream_t stream = 0;
  const bool fast = (M == TARGET_M && N == TARGET_N && K == TARGET_K);

  if (fast) {
    constexpr int split_k = SPLIT_K_FAST;
    const size_t MN = (size_t)TARGET_M * (size_t)TARGET_N;

    if (ensure_split_workspace(MN * split_k)) {
      dim3 block(THREADS_PER_BLOCK);
      dim3 grid(TARGET_N / BLOCK_N, TARGET_M / BLOCK_M, split_k);

      splitk_wmma_smem_vec128_64x64_kernel<split_k><<<grid, block, 0, stream>>>(
          A, Bcol, g_split_workspace);

      int threads = 256;
      int blocks = (int)(((MN / 8) + threads - 1) / threads);
      reduce_splitk_vec8_fp32_to_fp16_kernel<split_k><<<blocks, threads, 0, stream>>>(
          g_split_workspace, C, (int)MN);
    } else {
      dim3 bdim(16, 16);
      dim3 gdim((N + bdim.x - 1) / bdim.x, (M + bdim.y - 1) / bdim.y);
      fallback_gemm_kernel<<<gdim, bdim, 0, stream>>>(A, Bcol, C, M, N, K);
    }
  } else {
    dim3 bdim(16, 16);
    dim3 gdim((N + bdim.x - 1) / bdim.x, (M + bdim.y - 1) / bdim.y);
    fallback_gemm_kernel<<<gdim, bdim, 0, stream>>>(A, Bcol, C, M, N, K);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed.");
  }
}