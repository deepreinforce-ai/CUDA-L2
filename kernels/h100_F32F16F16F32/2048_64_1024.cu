#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, int kStage,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128, 4)
hgemm_optimized_kernel(const T *__restrict__ Aptr,
                       const T *__restrict__ Bptr,
                       T *__restrict__ Dptr,
                       int m, int n, int k) {
  using namespace cute;

  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  const int idx = threadIdx.x;
  const int ix  = blockIdx.x;
  const int iy  = blockIdx.y;

  if (iy * BM >= m || ix * BN >= n) return;

  Tensor A = make_tensor(make_gmem_ptr(Aptr),
                         make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr),
                         make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr),
                         make_shape(m, n), make_stride(n, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma  = tiled_mma.get_slice(idx);
  auto tCrA     = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB     = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD     = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  G2SCopyA g2s_copy_a;
  auto g2s_thr_a  = g2s_copy_a.get_slice(idx);
  auto tAgA_copy  = g2s_thr_a.partition_S(gA);
  auto tAsA_copy  = g2s_thr_a.partition_D(sA);

  G2SCopyB g2s_copy_b;
  auto g2s_thr_b  = g2s_copy_b.get_slice(idx);
  auto tBgB_copy  = g2s_thr_b.partition_S(gB);
  auto tBsB_copy  = g2s_thr_b.partition_D(sB);

  auto s2r_copy_a     = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_a      = s2r_copy_a.get_slice(idx);
  auto tAsA           = s2r_thr_a.partition_S(sA);
  auto tCrA_view      = s2r_thr_a.retile_D(tCrA);

  auto s2r_copy_b     = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_b      = s2r_copy_b.get_slice(idx);
  auto tBsB           = s2r_thr_b.partition_S(sB);
  auto tCrB_view      = s2r_thr_b.retile_D(tCrB);

  int itile_to_read = 0;
  int ismem_read    = 0;
  int ismem_write   = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
    cute::copy(g2s_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read;
    ++ismem_write;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  cute::copy(s2r_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
  cute::copy(s2r_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

  const int ntile = k / BK;
  const int nk    = size<2>(tCrA);

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      const int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();
        ismem_read = (ismem_read + 1) % kStage;
      }

      cute::copy(s2r_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
      cute::copy(s2r_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

      if (ik == nk - 2) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
          cp_async_fence();
        }
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  auto tCgD = thr_mma.partition_C(gD);

  auto tCrD_half = make_tensor_like<T>(tCrD);

  constexpr int num_elems = decltype(size(tCrD))::value;

#pragma unroll
  for (int i = 0; i < num_elems - 1; i += 2) {
    half2 h2 = __floats2half2_rn(tCrD(i), tCrD(i + 1));
    tCrD_half(i)     = reinterpret_cast<const T *>(&h2)[0];
    tCrD_half(i + 1) = reinterpret_cast<const T *>(&h2)[1];
  }
  if (num_elems % 2 != 0) {
    tCrD_half(num_elems - 1) = __float2half(tCrD(num_elems - 1));
  }

  cute::copy(tCrD_half, tCgD);
}

template <typename T, int Stages = 5>
void launch_hgemm_optimized(const T *a, const T *b, T *c,
                            int M, int N, int K) {
  using namespace cute;

  auto BM = Int<64>{};
  auto BN = Int<64>{};
  auto BK = Int<64>{};
  auto KStage = Int<Stages>{};

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
      make_layout(make_shape(Int<16>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op     = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA    = s2r_copy_atom;
  using S2RCopyAtomB    = s2r_copy_atom;

  const int BX = (N + BN - 1) / BN;
  const int BY = (M + BM - 1) / BM;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY);

  static constexpr int kShmSize =
      (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_optimized_kernel<T, BM, BN, BK, KStage, MMA,
                             G2SCopyA, G2SCopyB,
                             SmemLayoutA, SmemLayoutB,
                             S2RCopyAtomA, S2RCopyAtomB>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_optimized_kernel<T, BM, BN, BK, KStage, MMA,
                         G2SCopyA, G2SCopyB,
                         SmemLayoutA, SmemLayoutB,
                         S2RCopyAtomA, S2RCopyAtomB>
      <<<grid, block, kShmSize>>>(a, b, c, M, N, K);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                    \
  if ((T).options().dtype() != (th_type)) {                                     \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                  \
    throw std::runtime_error("values must be " #th_type);                       \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                     \
  if ((T).size(0) != (S0) || (T).size(1) != (S1)) {                             \
    throw std::runtime_error("Tensor size mismatch!");                           \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  const half *a_ptr   = reinterpret_cast<const half *>(a.data_ptr());
  const half *b_ptr   = reinterpret_cast<const half *>(b_col_major.data_ptr());
  half       *c_ptr   = reinterpret_cast<half       *>(c.data_ptr());

  launch_hgemm_optimized<half, 5>(a_ptr, b_ptr, c_ptr, M, N, K);
}