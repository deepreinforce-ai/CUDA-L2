#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template <int BM, int BN, int BK, int kStage>
__global__ void hgemm_splitk_kernel(
    const half* __restrict__ Aptr,
    const half* __restrict__ Bptr,
    float* __restrict__ Workspace,
    int M, int N, int K,
    int k_split_size,
    int num_k_splits)
{
  extern __shared__ half shm_data[];

  const int tx = threadIdx.x;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;

  if (by * BM >= M || bx * BN >= N) return;

  int k_start = bz * k_split_size;
  int k_end   = min(k_start + k_split_size, K);
  int k_len   = k_end - k_start;
  if (k_len <= 0) return;

  int ntile = (k_len + BK - 1) / BK;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})));

  half* Ashm = shm_data;
  half* Bshm = shm_data + cute::cosize(SmemLayoutA{});

  Tensor gA_block = make_tensor(
      make_gmem_ptr(Aptr + (size_t)(by * BM) * K + k_start),
      make_shape(Int<BM>{}, k_len),
      make_stride(K, Int<1>{}));
  Tensor gB_block = make_tensor(
      make_gmem_ptr(Bptr + (size_t)(bx * BN) * K + k_start),
      make_shape(Int<BN>{}, k_len),
      make_stride(K, Int<1>{}));

  Tensor gOut = make_tensor(make_gmem_ptr(Workspace + (size_t)bz * M * N),
                            make_shape(M, N), make_stride(N, Int<1>{}));

  Tensor gA = local_tile(gA_block, make_tile(Int<BM>{}, Int<BK>{}),
                         make_coord(0, _));
  Tensor gB = local_tile(gB_block, make_tile(Int<BN>{}, Int<BK>{}),
                         make_coord(0, _));
  Tensor gD = local_tile(gOut, make_tile(Int<BM>{}, Int<BN>{}),
                         make_coord(by, bx));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  using mma_op     = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom   = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;
  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using TiledMMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, half>;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  using s2r_copy_op     = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, half>;

  auto s2r_tiled_copy_a = make_tiled_copy_A(s2r_copy_atom{}, tiled_mma);
  auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(tx);
  auto tAsA             = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(s2r_copy_atom{}, tiled_mma);
  auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(tx);
  auto tBsB             = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);

  int itile_to_read = 0;
  int ismem_read    = 0;
  int ismem_write   = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1 && istage < ntile; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read;
    ++ismem_write;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  if (ntile > 0) {
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));
  }

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();
        ismem_read = (ismem_read + 1) % kStage;
      }

      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  auto tCgD = thr_mma.partition_C(gD);
  cute::copy(tCrD, tCgD);
}

__global__ void hgemm_warp_collective_reduce_kernel(
    const float* __restrict__ Workspace,
    half* __restrict__ C,
    int M, int N, int num_k_splits)
{
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;
  const int total = M * N;

  const int warp_base = blockIdx.x * blockDim.x + warp_id * 32;
  const int elem_idx = warp_base + lane_id;

  if (elem_idx >= total) return;

  float sum = 0.0f;
  #pragma unroll 4
  for (int s = 0; s < num_k_splits; ++s) {
    sum += Workspace[(size_t)s * total + elem_idx];
  }

  C[elem_idx] = __float2half(sum);
}

static float* g_workspace = nullptr;
static size_t g_workspace_size = 0;

void ensure_workspace(size_t needed_bytes) {
  if (g_workspace_size < needed_bytes) {
    if (g_workspace) cudaFree(g_workspace);
    cudaMalloc(&g_workspace, needed_bytes);
    g_workspace_size = needed_bytes;
  }
}

void launch_hgemm_splitk(
    const half* a, const half* b_tn, half* c,
    int M, int N, int K, int num_k_splits)
{
  static constexpr int BM = 128;
  static constexpr int BN = 64;
  static constexpr int BK = 32;
  static constexpr int kStage = 5;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})));

  static constexpr int shm_size =
      (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(half);

  int num_blocks_m = (M + BM - 1) / BM;
  int num_blocks_n = (N + BN - 1) / BN;

  int k_split_size = (K + num_k_splits - 1) / num_k_splits;
  k_split_size = ((k_split_size + BK - 1) / BK) * BK;
  int actual_splits = (K + k_split_size - 1) / k_split_size;

  size_t ws_bytes = (size_t)actual_splits * M * N * sizeof(float);
  ensure_workspace(ws_bytes);

  dim3 grid(num_blocks_n, num_blocks_m, actual_splits);
  dim3 block(128);

  cudaFuncSetAttribute(
      hgemm_splitk_kernel<BM, BN, BK, kStage>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  hgemm_splitk_kernel<BM, BN, BK, kStage><<<grid, block, shm_size>>>(
      a, b_tn, g_workspace, M, N, K, k_split_size, actual_splits);

  int total_elements = M * N;
  int reduce_threads = 128;
  int reduce_blocks = (total_elements + reduce_threads - 1) / reduce_threads;

  hgemm_warp_collective_reduce_kernel<<<reduce_blocks, reduce_threads>>>(
      g_workspace, c, M, N, actual_splits);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  const half* a_ptr = reinterpret_cast<const half*>(a.data_ptr());
  const half* b_tn_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* c_ptr = reinterpret_cast<half*>(c.data_ptr());

  const int num_k_splits = 32;

  launch_hgemm_splitk(a_ptr, b_tn_ptr, c_ptr, M, N, K, num_k_splits);
}