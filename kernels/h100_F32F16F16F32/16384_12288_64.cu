#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename SmemLayoutC,
          typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(256, 2)
hgemm_fused_pipeline_kernel(T *__restrict__ Aptr, T *__restrict__ Bptr,
                            T *__restrict__ Dptr, int m, int n, int k) {
  using namespace cute;
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;

  constexpr int swizzle_factor = 4;
  int bx_raw = blockIdx.x;
  int by_raw = blockIdx.y;
  int bx = bx_raw * swizzle_factor + (by_raw % swizzle_factor);
  int by = by_raw / swizzle_factor;

  if (by * BM >= m || bx * BN >= n) return;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, 0));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, 0));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(by, bx));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);

  auto tCrA = thr_mma.partition_fragment_A(gA);
  auto tCrB = thr_mma.partition_fragment_B(gB);
  auto tCrD = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  cute::copy(g2s_tiled_copy_a, tAgA_copy, tAsA_copy);
  cute::copy(g2s_tiled_copy_b, tBgB_copy, tBsB_copy);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  using S2RCopyAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;
  using S2RCopyAtomB = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  int nk = size<2>(tCrA);

  cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0), tCrA_view(_, _, 0));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0), tCrB_view(_, _, 0));

  if (nk > 1) {
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 1), tCrA_view(_, _, 1));
  }
  cute::gemm(tiled_mma, tCrD, tCrA(_, _, 0), tCrB(_, _, 0), tCrD);
  if (nk > 1) {
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 1), tCrB_view(_, _, 1));
  }

  if (nk > 1) {
    if (nk > 2) {
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, 2), tCrA_view(_, _, 2));
    }
    cute::gemm(tiled_mma, tCrD, tCrA(_, _, 1), tCrB(_, _, 1), tCrD);
    if (nk > 2) {
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, 2), tCrB_view(_, _, 2));
    }
  }

  if (nk > 2) {
    if (nk > 3) {
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, 3), tCrA_view(_, _, 3));
    }
    cute::gemm(tiled_mma, tCrD, tCrA(_, _, 2), tCrB(_, _, 2), tCrD);
    if (nk > 3) {
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, 3), tCrB_view(_, _, 3));
    }
  }

  if (nk > 3) {
    cute::gemm(tiled_mma, tCrD, tCrA(_, _, 3), tCrB(_, _, 3), tCrD);
  }

  __syncthreads();

  auto sC = make_tensor(sA.data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

  int step = size<3>(tCsC_r2s);

  #pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    #pragma unroll
    for (int j = 0; j < step; ++j) {
      if (i + j < size<1>(tCrC_r2sx)) {
        cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
      }
    }
    __syncthreads();

    #pragma unroll
    for (int j = 0; j < step; ++j) {
      if (i + j < size<1>(tCrC_r2sx)) {
        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
      }
    }

    if (i + step < size<1>(tCrC_r2sx)) {
      __syncthreads();
    }
  }
}

template <typename T>
void launch_hgemm_fused_pipeline(T *a, T *b, T *c, int M, int N, int K) {
  using namespace cute;

  constexpr int BM_val = 128;
  constexpr int BN_val = 256;
  constexpr int BK_val = 64;

  auto BM = Int<BM_val>{};
  auto BN = Int<BN_val>{};
  auto BK = Int<BK_val>{};

  using SmemLayoutAtomA = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK_val>{}),
                  make_stride(Int<BK_val>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(Int<BM_val>{}, Int<BK_val>{})));

  using SmemLayoutAtomB = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK_val>{}),
                  make_stride(Int<BK_val>{}, Int<1>{}))));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(Int<BN_val>{}, Int<BK_val>{})));

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 4;
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
      make_layout(make_shape(Int<32>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using G2SCopyB = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<4>{})));

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  int BX = (N + BN_val - 1) / BN_val;
  int BY = (M + BM_val - 1) / BM_val;

  constexpr int swizzle_factor = 4;
  int grid_x = (BX + swizzle_factor - 1) / swizzle_factor;
  int grid_y = BY * swizzle_factor;

  dim3 block(size(MMA{}));
  dim3 grid(grid_x, grid_y);

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_fused_pipeline_kernel<T, BM_val, BN_val, BK_val, MMA,
                                  G2SCopyA, G2SCopyB,
                                  SmemLayoutA, SmemLayoutB, SmemLayoutC,
                                  R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_fused_pipeline_kernel<T, BM_val, BN_val, BK_val, MMA,
                              G2SCopyA, G2SCopyB,
                              SmemLayoutA, SmemLayoutB, SmemLayoutC,
                              R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>
      <<<grid, block, kShmSize>>>(a, b, c, M, N, K);
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

  launch_hgemm_fused_pipeline<half>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}