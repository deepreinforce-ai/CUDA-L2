#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template <typename T, int BM, int BN, int BK>
__global__ void __launch_bounds__(128)
hgemm_optimized_kernel(T const* __restrict__ Aptr, 
                       T const* __restrict__ Bptr,
                       T* __restrict__ Dptr, 
                       int m, int n, int k) {
  
  extern __shared__ T shm_data[];
  
  T* Ashm = shm_data;
  T* Bshm = shm_data + BM * BK;
  
  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  if (by * BM >= m || bx * BN >= n) return;
  
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));
  
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, 0));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, 0));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(by, bx));
  
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                 make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BM>{}, Int<BK>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{})));
  
  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});
  
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
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
  
  MMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tid);
  
  auto tCrA = thr_mma.partition_fragment_A(gA);
  auto tCrB = thr_mma.partition_fragment_B(gB);
  auto tCrD = thr_mma.partition_fragment_C(gD);
  
  clear(tCrD);
  
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                 make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;
  
  G2SCopyA g2s_tiled_copy_a;
  G2SCopyB g2s_tiled_copy_b;
  
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
  
  auto tAgA = g2s_thr_copy_a.partition_S(gA);
  auto tAsA = g2s_thr_copy_a.partition_D(sA);
  auto tBgB = g2s_thr_copy_b.partition_S(gB);
  auto tBsB = g2s_thr_copy_b.partition_D(sB);
  
  copy(g2s_tiled_copy_a, tAgA, tAsA);
  copy(g2s_tiled_copy_b, tBgB, tBsB);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  
  auto s2r_tiled_copy_a = make_tiled_copy_A(s2r_copy_atom{}, tiled_mma);
  auto s2r_tiled_copy_b = make_tiled_copy_B(s2r_copy_atom{}, tiled_mma);
  
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  
  auto tAsA_s2r = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
  auto tBsB_s2r = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);
  
  int nk = size<2>(tCrA);
  
  #pragma unroll
  for (int ik = 0; ik < nk; ++ik) {
    copy(s2r_tiled_copy_a, tAsA_s2r(_, _, ik), tCrA_view(_, _, ik));
    copy(s2r_tiled_copy_b, tBsB_s2r(_, _, ik), tCrB_view(_, _, ik));
    gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
  }
  
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                 make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<4>{})));
  
  auto sC = make_tensor(make_smem_ptr(Ashm), SmemLayoutC{});
  
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy = r2s_tiled_copy.get_slice(tid);
  
  auto tCrC_r2s = r2s_thr_copy.retile_S(tCrD);
  auto tCsC_r2s = r2s_thr_copy.partition_D(sC);
  
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                 make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  
  S2GCopyC s2g_tiled_copy;
  auto s2g_thr_copy = s2g_tiled_copy.get_thread_slice(tid);
  
  auto tCsC_s2g = s2g_thr_copy.partition_S(sC);
  auto tCgC_s2g = s2g_thr_copy.partition_D(gD);
  
  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);
  
  int step = size<3>(tCsC_r2s);
  #pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    #pragma unroll
    for (int j = 0; j < step; ++j) {
      copy(r2s_tiled_copy, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();
    
    #pragma unroll
    for (int j = 0; j < step; ++j) {
      copy(s2g_tiled_copy, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
  }
}

template <typename T>
void launch_hgemm_optimized(T* a, T* b, T* c, int M, int N, int K) {
  constexpr int BM = 64;
  constexpr int BN = 128;
  constexpr int BK = 64;
  
  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  
  dim3 block(128);
  dim3 grid(BX, BY);
  
  int shm_size = (BM * BK + BN * BK) * sizeof(T);
  
  cudaFuncSetAttribute(
    hgemm_optimized_kernel<T, BM, BN, BK>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    shm_size
  );
  
  hgemm_optimized_kernel<T, BM, BN, BK>
    <<<grid, block, shm_size>>>(a, b, c, M, N, K);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  
  launch_hgemm_optimized<half>(
    reinterpret_cast<half*>(a.data_ptr()),
    reinterpret_cast<half*>(b_col_major.data_ptr()),
    reinterpret_cast<half*>(c.data_ptr()),
    M, N, K
  );
}