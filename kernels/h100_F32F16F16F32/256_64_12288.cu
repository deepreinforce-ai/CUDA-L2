#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

static constexpr int M_FIXED = 256;
static constexpr int N_FIXED = 64;
static constexpr int K_FIXED = 12288;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 64;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int WARPS = 4;
static constexpr int THREADS = WARPS * 32;

static constexpr int SPLIT_K = 32;
static constexpr int K_PER_SPLIT = K_FIXED / SPLIT_K;
static constexpr int K_TILES_PER_SPLIT = K_PER_SPLIT / BK;
static_assert((K_FIXED % SPLIT_K) == 0, "K must be divisible by SPLIT_K");
static_assert((K_PER_SPLIT % BK) == 0, "K_PER_SPLIT must be divisible by BK");

static constexpr int PAD_A = 8;
static constexpr int PAD_B = 8;
static constexpr int STRIDE_A = BK + PAD_A;
static constexpr int STRIDE_B = BK + PAD_B;

static constexpr int VEC_PER_ROW = BK / 8;
static constexpr int TILE_VEC_A = BM * VEC_PER_ROW;
static constexpr int TILE_VEC_B = BN * VEC_PER_ROW;

static float* g_workspace = nullptr;
static size_t g_workspace_elems = 0;

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  unsigned smem_u32 = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(smem_u32), "l"(gmem_ptr));
#else
  *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#endif
}

__device__ __forceinline__ void load_stage_ab(
    int stage,
    int tid,
    int block_m0,
    int k0,
    const half* __restrict__ A,
    const half* __restrict__ Bc,
    half smem_A[2][BM][STRIDE_A],
    half smem_B[2][BN][STRIDE_B]) {

  for (int idx = tid; idx < TILE_VEC_A; idx += THREADS) {
    int row = idx / VEC_PER_ROW;
    int col = (idx % VEC_PER_ROW) * 8;
    const half* gA = &A[(block_m0 + row) * K_FIXED + (k0 + col)];
    cp_async_16B(&smem_A[stage][row][col], gA);
  }

  for (int idx = tid; idx < TILE_VEC_B; idx += THREADS) {
    int row = idx / VEC_PER_ROW;
    int col = (idx % VEC_PER_ROW) * 8;
    const half* gB = &Bc[row * K_FIXED + (k0 + col)];
    cp_async_16B(&smem_B[stage][row][col], gB);
  }

  cp_async_commit();
}

__global__ void __launch_bounds__(THREADS, 4)
hgemm_splitk_wmma_cpasync_kernel(
    const half* __restrict__ A,
    const half* __restrict__ Bc,
    float* __restrict__ W
) {
  const int tile_m   = blockIdx.x;
  const int split_id = blockIdx.y;
  const int tid      = threadIdx.x;
  const int warp_id  = tid >> 5;

  const int block_m0 = tile_m * BM;
  const int k_begin  = split_id * K_PER_SPLIT;

  const int warp_row_base = (warp_id >> 1) * 32;
  const int warp_col_base = (warp_id & 1) * 32;

  __shared__ half smem_A[2][BM][STRIDE_A];
  __shared__ half smem_B[2][BN][STRIDE_B];

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  int stage = 0;
  load_stage_ab(stage, tid, block_m0, k_begin, A, Bc, smem_A, smem_B);
  cp_async_wait<0>();
  __syncthreads();

#pragma unroll
  for (int kt = 0; kt < K_TILES_PER_SPLIT; ++kt) {
    const int k0 = k_begin + kt * BK;
    const int next_stage = stage ^ 1;

    if (kt + 1 < K_TILES_PER_SPLIT) {
      load_stage_ab(next_stage, tid, block_m0, k0 + BK, A, Bc, smem_A, smem_B);
    }

#pragma unroll
    for (int kk = 0; kk < BK / WMMA_K; ++kk) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2];

#pragma unroll
      for (int i = 0; i < 2; ++i) {
        wmma::load_matrix_sync(
            a_frag[i],
            &smem_A[stage][warp_row_base + i * WMMA_M][kk * WMMA_K],
            STRIDE_A);
      }

#pragma unroll
      for (int j = 0; j < 2; ++j) {
        wmma::load_matrix_sync(
            b_frag[j],
            &smem_B[stage][warp_col_base + j * WMMA_N][kk * WMMA_K],
            STRIDE_B);
      }

#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }

    if (kt + 1 < K_TILES_PER_SPLIT) {
      cp_async_wait<0>();
      __syncthreads();
      stage = next_stage;
    }
  }

  float* plane = W + (size_t)split_id * (M_FIXED * N_FIXED);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int gr = block_m0 + warp_row_base + i * WMMA_M;
      int gc = warp_col_base + j * WMMA_N;
      wmma::store_matrix_sync(plane + gr * N_FIXED + gc, acc[i][j], N_FIXED, wmma::mem_row_major);
    }
  }
}

__global__ void reduce_splitk_and_cast_vec4_kernel(
    const float* __restrict__ W,
    half* __restrict__ C
) {
  constexpr int TOTAL = M_FIXED * N_FIXED;
  constexpr int TOTAL_VEC = TOTAL / 4;

  int vidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vidx >= TOTAL_VEC) return;

  float4 sum4 = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
  for (int s = 0; s < SPLIT_K; ++s) {
    const float4 v = reinterpret_cast<const float4*>(W + (size_t)s * TOTAL)[vidx];
    sum4.x += v.x;
    sum4.y += v.y;
    sum4.z += v.z;
    sum4.w += v.w;
  }

  int base = vidx * 4;
  half2 h01 = __halves2half2(__float2half_rn(sum4.x), __float2half_rn(sum4.y));
  half2 h23 = __halves2half2(__float2half_rn(sum4.z), __float2half_rn(sum4.w));

  reinterpret_cast<half2*>(C)[base / 2]     = h01;
  reinterpret_cast<half2*>(C)[base / 2 + 1] = h23;
}

void cuda_l2_h100_fp32(torch::Tensor a,
                                torch::Tensor b,
                                torch::Tensor b_col_major,
                                torch::Tensor c)
{
  (void)b;

  const size_t need_elems = (size_t)SPLIT_K * (size_t)M_FIXED * (size_t)N_FIXED;
  if (g_workspace == nullptr || g_workspace_elems < need_elems) {
    if (g_workspace != nullptr) cudaFree(g_workspace);
    cudaMalloc(&g_workspace, need_elems * sizeof(float));
    g_workspace_elems = need_elems;
  }

  const half* A  = reinterpret_cast<const half*>(a.data_ptr());
  const half* Bc = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* Cptr     = reinterpret_cast<half*>(c.data_ptr());

  cudaStream_t stream = 0;

  dim3 grid(M_FIXED / BM, SPLIT_K, 1);
  dim3 block(THREADS, 1, 1);
  hgemm_splitk_wmma_cpasync_kernel<<<grid, block, 0, stream>>>(A, Bc, g_workspace);

  constexpr int TOTAL_VEC = (M_FIXED * N_FIXED) / 4;
  int tpb = 256;
  int blocks = (TOTAL_VEC + tpb - 1) / tpb;
  reduce_splitk_and_cast_vec4_kernel<<<blocks, tpb, 0, stream>>>(g_workspace, Cptr);
}