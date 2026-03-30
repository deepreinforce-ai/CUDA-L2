#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, int kStage, int kSplitK,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128)
hgemm_splitk_noatomic_kernel(
    const T * __restrict__ Aptr,
    const T * __restrict__ Bptr,
    float   * __restrict__ workspace,
    int M, int N, int K)
{
  using namespace cute;

  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  const int idx = threadIdx.x;
  const int ix  = blockIdx.x;
  const int iy  = blockIdx.y;
  const int iz  = blockIdx.z;

  if (iy * BM >= M || ix * BN >= N) return;

  const int k_per_split = (K + kSplitK - 1) / kSplitK;
  const int k_start     = iz * k_per_split;
  const int k_end       = min(k_start + k_per_split, K);
  const int k_len       = k_end - k_start;
  const int ntile       = (k_len + BK - 1) / BK;

  if (k_len <= 0 || ntile == 0) return;

  Tensor A_gmem = make_tensor(make_gmem_ptr(Aptr + k_start),
                              make_shape(M, k_len),
                              make_stride(K, Int<1>{}));
  Tensor B_gmem = make_tensor(make_gmem_ptr(Bptr + k_start),
                              make_shape(N, k_len),
                              make_stride(K, Int<1>{}));
  Tensor Ws_gmem = make_tensor(make_gmem_ptr(workspace + (size_t)iz * M * N),
                               make_shape(M, N),
                               make_stride(N, Int<1>{}));

  Tensor gA  = local_tile(A_gmem,  make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB  = local_tile(B_gmem,  make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gWs = local_tile(Ws_gmem, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);

  auto tCrA  = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB  = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD  = thr_mma.partition_fragment_C(gWs);
  auto tCgWs = thr_mma.partition_C(gWs);
  clear(tCrD);

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy      = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy      = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy      = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy      = g2s_thr_copy_b.partition_D(sB);

  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA             = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB             = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);

  int itile_to_read = 0;
  int ismem_read    = 0;
  int ismem_write   = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (istage < ntile) {
      cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                 tAsA_copy(_, _, _, ismem_write));
      cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                 tBsB_copy(_, _, _, ismem_write));
      ++itile_to_read;
      ismem_write = (ismem_write + 1) % kStage;
    }
    cp_async_fence();
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  const int nk = size<2>(tCrA);

  cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
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

  CUTE_UNROLL
  for (int i = 0; i < size(tCrD); ++i) {
    tCgWs(i) = tCrD(i);
  }
}

template <int kSplitK>
__global__ void hgemm_reduce_lanes_kernel(
    const float * __restrict__ workspace,
    half        * __restrict__ output,
    int total_elems)
{
  const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx2 = tid * 2;

  if (idx2 + 1 >= total_elems) {
    if (idx2 < total_elems) {
      float acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kSplitK; ++s) {
        acc += __ldg(workspace + (size_t)s * total_elems + idx2);
      }
      output[idx2] = __float2half(acc);
    }
    return;
  }

  float acc0 = 0.0f, acc1 = 0.0f;

#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    float2 v = __ldg(reinterpret_cast<const float2*>(workspace + (size_t)s * total_elems + idx2));
    acc0 += v.x;
    acc1 += v.y;
  }

  float2 result = {acc0, acc1};
  *reinterpret_cast<half2*>(output + idx2) = __float22half2_rn(result);
}

static float *g_workspace    = nullptr;
static size_t g_workspace_sz = 0;

static float* get_workspace(size_t bytes) {
  if (g_workspace_sz < bytes) {
    if (g_workspace) cudaFree(g_workspace);
    cudaMalloc(&g_workspace, bytes);
    g_workspace_sz = bytes;
  }
  return g_workspace;
}

template <typename T, int kSplitK, int kStages>
void launch_hgemm_splitk_noatomic(
    const T *a,
    const T *b_col,
    T       *c,
    int M, int N, int K)
{
  using namespace cute;

  static constexpr int BM     = 64;
  static constexpr int BN     = 64;
  static constexpr int BK     = 64;
  static constexpr int KStage = kStages;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

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
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{},  Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op     = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA    = s2r_copy_atom;
  using S2RCopyAtomB    = s2r_copy_atom;

  const int BX = (N + BN - 1) / BN;
  const int BY = (M + BM - 1) / BM;
  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, kSplitK);

  static constexpr int kShmSize =
      (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(T);

  cudaFuncSetAttribute(
      hgemm_splitk_noatomic_kernel<T, BM, BN, BK, KStage, kSplitK, MMA,
                                   G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                                   S2RCopyAtomA, S2RCopyAtomB>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  const size_t ws_bytes = (size_t)kSplitK * M * N * sizeof(float);
  float *workspace = get_workspace(ws_bytes);

  hgemm_splitk_noatomic_kernel<T, BM, BN, BK, KStage, kSplitK, MMA,
                               G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
                               S2RCopyAtomA, S2RCopyAtomB>
      <<<grid, block, kShmSize>>>(a, b_col, workspace, M, N, K);

  const int total  = M * N;
  const int nthrd  = 256;
  const int nblks  = (total / 2 + nthrd - 1) / nthrd;
  hgemm_reduce_lanes_kernel<kSplitK><<<nblks, nthrd>>>(workspace, c, total);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    throw std::runtime_error("Tensor dtype mismatch");                         \
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

  launch_hgemm_splitk_noatomic<half, 8, 5>(
      reinterpret_cast<const half *>(a.data_ptr()),
      reinterpret_cast<const half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()),
      M, N, K);
}