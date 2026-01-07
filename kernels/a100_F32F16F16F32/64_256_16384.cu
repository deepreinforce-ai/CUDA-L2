#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void cuda_l2_a100_fp32_kernel(T *Aptr, T *Bptr,
                                                  float *Dptr, int m,
                                                  int n, int k, int k_split) {
  using namespace cute;
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  int k_idx = blockIdx.z;

  if (iy * BM >= m || ix * BN >= n)
    return;

  int k_start = k_idx * k_split;
  int k_end = min(k_start + k_split, k);
  int k_tiles = (k_end - k_start + BK - 1) / BK;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}),
                         make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}),
                         make_coord(ix, _));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD = partition_fragment_C(tiled_mma, make_shape(Int<BM>{}, Int<BN>{}));
  clear(tCrD);

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  int k_tile_start = k_start / BK;
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  int prefetch_stages = min((int)kStage - 1, k_tiles);
#pragma unroll
  for (int istage = 0; istage < prefetch_stages; ++istage) {
    int k_tile_idx = k_tile_start + istage;
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, k_tile_idx),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, k_tile_idx),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read;
    ++ismem_write;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  if (k_tiles > 0) {
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));
  }

#pragma unroll 1
  for (int itile = 0; itile < k_tiles; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();
        ismem_read = (ismem_read + 1) % kStage;
      }

      if (itile * nk + ik_next < k_tiles * nk) {
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                   tCrA_view(_, _, ik_next));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                   tCrB_view(_, _, ik_next));
      }

      if (ik == 0) {
        if (itile_to_read < k_tiles) {
          int k_tile_idx = k_tile_start + itile_to_read;
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, k_tile_idx),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, k_tile_idx),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  auto tCrD_fp32 = make_tensor_like<float>(tCrD);
  cute::copy(tCrD, tCrD_fp32);

  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}),
                         make_coord(iy, ix));

  auto tCgD = thr_mma.partition_C(gD);
  
#pragma unroll
  for (int i = 0; i < size(tCrD_fp32); ++i) {
    atomicAdd(&tCgD(i), tCrD_fp32(i));
  }
}

template <typename T, const int Stages = 4>
void launch_hgemm_mma_stages(T *a, T *b, T *c, int M, int N, int K, int k_splits) {
  using namespace cute;

  auto BM = Int<32>{};
  auto BN = Int<64>{};
  auto BK = Int<64>{};
  auto KStage = Int<Stages>{};

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<BK>{}),
                                      make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  
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

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  int BZ = k_splits;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, BZ);

  static constexpr int kShmSize =
      (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(T);

  int k_split = (K + k_splits - 1) / k_splits;

  cudaMemset(c, 0, M * N * sizeof(float));

  cudaFuncSetAttribute(
      cuda_l2_a100_fp32_kernel<T, BM, BN, BK, KStage, MMA, G2SCopyA,
                                        G2SCopyB, SmemLayoutA, SmemLayoutB,
                                        S2RCopyAtomA, S2RCopyAtomB>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  cuda_l2_a100_fp32_kernel<T, BM, BN, BK, KStage, MMA, G2SCopyA,
                                    G2SCopyB, SmemLayoutA, SmemLayoutB,
                                    S2RCopyAtomA, S2RCopyAtomB>
      <<<grid, block, kShmSize>>>(a, b, reinterpret_cast<float*>(c), M, N, K, k_split);
}

void cuda_l2_a100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  
  auto c_fp32 = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(c.device()));
  
  int k_splits = 8;
  launch_hgemm_mma_stages<half, 4>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c_fp32.data_ptr()), M, N, K, k_splits);
  
  c.copy_(c_fp32.to(torch::kHalf));
}