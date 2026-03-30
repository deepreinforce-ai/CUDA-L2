#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>

using namespace cute;

template <typename T, int BM, int BN, int BK, int kStage, int kThreads,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,
          typename S2RCopyAtomA, typename S2RCopyAtomB,
          typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void
__launch_bounds__(kThreads, 2)
hgemm_optimized_pipelined_kernel(
    const T * __restrict__ Aptr,
    const T * __restrict__ Bptr,
    T * __restrict__ Dptr,
    int m, int n, int k) {

  extern __shared__ T smem_buf[];
  T *Ashm = smem_buf;
  T *Bshm = smem_buf + cute::cosize(SmemLayoutA{});

  const int tid = threadIdx.x;
  const int ix  = blockIdx.x;
  const int iy  = blockIdx.y;

  Tensor mA = make_tensor(make_gmem_ptr(Aptr),
                          make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(Bptr),
                          make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor mD = make_tensor(make_gmem_ptr(Dptr),
                          make_shape(m, n), make_stride(n, Int<1>{}));

  Tensor gA = local_tile(mA, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB = local_tile(mB, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD = local_tile(mD, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tid);
  auto tCrA    = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB    = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD    = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  G2SCopyA g2s_copy_a;
  auto g2s_thr_a = g2s_copy_a.get_slice(tid);
  auto tAgA      = g2s_thr_a.partition_S(gA);
  auto tAsA      = g2s_thr_a.partition_D(sA);

  G2SCopyB g2s_copy_b;
  auto g2s_thr_b = g2s_copy_b.get_slice(tid);
  auto tBgB      = g2s_thr_b.partition_S(gB);
  auto tBsB      = g2s_thr_b.partition_D(sB);

  auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_a  = s2r_copy_a.get_slice(tid);
  auto sCAsA      = s2r_thr_a.partition_S(sA);
  auto tCrA_s2r   = s2r_thr_a.retile_D(tCrA);

  auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_b  = s2r_copy_b.get_slice(tid);
  auto sCBsB      = s2r_thr_b.partition_S(sB);
  auto tCrB_s2r   = s2r_thr_b.retile_D(tCrB);

  const int ntile = k / BK;
  const int nk    = size<2>(tCrA);

  int iread  = 0;
  int iwrite = 0;
  int ifetch = 0;

#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    cute::copy(g2s_copy_a, tAgA(_, _, _, s), tAsA(_, _, _, s));
    cute::copy(g2s_copy_b, tBgB(_, _, _, s), tBsB(_, _, _, s));
    cp_async_fence();
    ++ifetch;
    ++iwrite;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  cute::copy(s2r_copy_a, sCAsA(_, _, 0, iread), tCrA_s2r(_, _, 0));
  cute::copy(s2r_copy_b, sCBsB(_, _, 0, iread), tCrB_s2r(_, _, 0));
  if (nk > 1) {
    cute::copy(s2r_copy_a, sCAsA(_, _, 1, iread), tCrA_s2r(_, _, 1));
    cute::copy(s2r_copy_b, sCBsB(_, _, 1, iread), tCrB_s2r(_, _, 1));
  }

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      const int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();
        iread = (iread + 1) % kStage;
      }

      if (ik < nk - 1) {
        cute::copy(s2r_copy_a, sCAsA(_, _, ik_next, iread), tCrA_s2r(_, _, ik_next));
        cute::copy(s2r_copy_b, sCBsB(_, _, ik_next, iread), tCrB_s2r(_, _, ik_next));
      } else if (itile + 1 < ntile) {
        cute::copy(s2r_copy_a, sCAsA(_, _, 0, iread), tCrA_s2r(_, _, 0));
        cute::copy(s2r_copy_b, sCBsB(_, _, 0, iread), tCrB_s2r(_, _, 0));
        if (nk > 1) {
          cute::copy(s2r_copy_a, sCAsA(_, _, 1, iread), tCrA_s2r(_, _, 1));
          cute::copy(s2r_copy_b, sCBsB(_, _, 1, iread), tCrB_s2r(_, _, 1));
        }
      }

      if (ik == 1) {
        if (ifetch < ntile) {
          cute::copy(g2s_copy_a, tAgA(_, _, _, ifetch), tAsA(_, _, _, iwrite));
          cute::copy(g2s_copy_b, tBgB(_, _, _, ifetch), tBsB(_, _, _, iwrite));
          ++ifetch;
          iwrite = (iwrite + 1) % kStage;
        }
        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  auto tCrD_fp16 = make_tensor_like<T>(tCrD);
  cute::copy(tCrD, tCrD_fp16);

  auto sC = make_tensor(make_smem_ptr(Ashm), SmemLayoutC{});

  auto r2s_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_c  = r2s_copy_c.get_slice(tid);
  auto tCrC_r2s   = r2s_thr_c.retile_S(tCrD_fp16);
  auto tCsC_r2s   = r2s_thr_c.partition_D(sC);

  S2GCopyC s2g_copy_c;
  auto s2g_thr_c = s2g_copy_c.get_thread_slice(tid);
  auto tCsC_s2g  = s2g_thr_c.partition_S(sC);
  auto tCgC_s2g  = s2g_thr_c.partition_D(gD);

  auto tCgC_flat = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_flat = group_modes<1, 3>(tCrC_r2s);

  const int step = size<3>(tCsC_r2s);
  const int total_steps = size<1>(tCrC_flat);

#pragma unroll
  for (int j = 0; j < step; ++j) {
    if (j < total_steps) {
      cute::copy(r2s_copy_c, tCrC_flat(_, j), tCsC_r2s(_, 0, 0, j));
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = step; i < total_steps; i += step) {
#pragma unroll
    for (int j = 0; j < step; ++j) {
      const int prev_idx = i - step + j;
      if (prev_idx < total_steps) {
        cute::copy(s2g_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_flat(_, prev_idx));
      }
    }
    
    __syncthreads();
    
#pragma unroll
    for (int j = 0; j < step; ++j) {
      const int curr_idx = i + j;
      if (curr_idx < total_steps) {
        cute::copy(r2s_copy_c, tCrC_flat(_, curr_idx), tCsC_r2s(_, 0, 0, j));
      }
    }
    __syncthreads();
  }

  const int final_start = ((total_steps - 1) / step) * step;
#pragma unroll
  for (int j = 0; j < step; ++j) {
    const int idx = final_start + j;
    if (idx < total_steps) {
      cute::copy(s2g_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_flat(_, idx));
    }
  }
}


template <int BM, int BN, int BK, int Stages>
struct HgemmOptimizedPipelined {
  using T = half;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BM>{}, Int<BK>{}, Int<Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<Stages>{})));

  using mma_op     = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom   = MMA_Atom<mma_traits>;

  static constexpr int kMmaM = 2, kMmaN = 2, kMmaK = 1;
  using mma_atom_shape = typename mma_traits::Shape_MNK;
  static constexpr int kPM = 1 * kMmaM * get<0>(mma_atom_shape{});
  static constexpr int kPN = 2 * kMmaN * get<1>(mma_atom_shape{});
  static constexpr int kPK = 1 * kMmaK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaM>{}, Int<kMmaN>{}, Int<kMmaK>{})));
  using MMA_P_T = Tile<Int<kPM>, Int<kPN>, Int<kPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  static constexpr int kThreads = size(MMA{});

  using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<16>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using G2SCopyB = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<16>{}, Int<8>{}),
                  make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using s2r_copy_op     = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  static constexpr int kSmemCBatch = 4;
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kPM>{}, Int<kPN>{}),
                  make_stride(Int<kPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kPM>{}, Int<kPN>{}, Int<kSmemCBatch>{})));

  static_assert(cosize(SmemLayoutA{}) >= cosize(SmemLayoutC{}),
                "SmemC exceeds SmemA capacity");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  static constexpr int shmAB =
      (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})) * (int)sizeof(T);
  static constexpr int shmC = cosize(SmemLayoutC{}) * (int)sizeof(T);
  static constexpr int shmTotal = (shmAB > shmC) ? shmAB : shmC;

  static void launch(const T *a, const T *b, T *c, int M, int N, int K) {
    const int BX = (N + BN - 1) / BN;
    const int BY = (M + BM - 1) / BM;
    dim3 block(kThreads);
    dim3 grid(BX, BY);
    auto kfn = hgemm_optimized_pipelined_kernel<
        T, BM, BN, BK, Stages, kThreads,
        MMA, G2SCopyA, G2SCopyB,
        SmemLayoutA, SmemLayoutB, SmemLayoutC,
        S2RCopyAtomA, S2RCopyAtomB,
        R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, shmTotal);
    kfn<<<grid, block, shmTotal>>>(a, b, c, M, N, K);
  }
};


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

  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  const half *a_ptr = reinterpret_cast<const half *>(a.data_ptr());
  const half *b_ptr = reinterpret_cast<const half *>(b_col_major.data_ptr());
  half *c_ptr       = reinterpret_cast<half *>(c.data_ptr());

  HgemmOptimizedPipelined<32, 128, 64, 6>::launch(a_ptr, b_ptr, c_ptr, M, N, K);
}