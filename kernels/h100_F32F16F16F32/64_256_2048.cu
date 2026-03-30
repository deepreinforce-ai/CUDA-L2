#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC>
__global__ void __launch_bounds__(128, 3)
cuda_l2_h100_fp32_kernel(T *Aptr, T *Bptr, T *Dptr, int m, int n, int k) {
  using namespace cute;
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  if (iy * BM >= m || ix * BN >= n)
    return;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
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

  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;
  int ntile = k / BK;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (itile_to_read < ntile) {
      cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
                 tBsB_copy(_, _, _, istage));
      cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                 tAsA_copy(_, _, _, istage));
      cp_async_fence();
      ++itile_to_read;
      ++ismem_write;
    }
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  int nk = size<2>(tCrA);

  cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

  if (nk > 1) {
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 1, ismem_read), tCrA_view(_, _, 1));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 1, ismem_read), tCrB_view(_, _, 1));
  }

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_pf0 = ik + 2;
      int ik_pf1 = ik + 3;

      if (ik == nk - 2) {
        cp_async_wait<kStage - 2>();
        __syncthreads();
        ismem_read = (ismem_read + 1) % kStage;

        if (itile + 1 < ntile) {
          cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read),
                     tCrA_view(_, _, 0));
          cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read),
                     tCrB_view(_, _, 0));
          if (nk > 1) {
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, 1, ismem_read),
                       tCrA_view(_, _, 1));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, 1, ismem_read),
                       tCrB_view(_, _, 1));
          }
        }
      } else if (ik < nk - 2) {
        if (ik_pf0 < nk) {
          cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_pf0, ismem_read),
                     tCrA_view(_, _, ik_pf0));
          cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_pf0, ismem_read),
                     tCrB_view(_, _, ik_pf0));
        }
        if (ik_pf1 < nk) {
          cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_pf1, ismem_read),
                     tCrA_view(_, _, ik_pf1));
          cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_pf1, ismem_read),
                     tCrB_view(_, _, ik_pf1));
        }
      }

      if (ik == 1 && itile_to_read < ntile) {
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                   tBsB_copy(_, _, _, ismem_write));
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                   tAsA_copy(_, _, _, ismem_write));
        cp_async_fence();
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  auto tCrD_half = make_tensor_like<T>(tCrD);
  const int nelems = size(tCrD);
  float * __restrict__ src = reinterpret_cast<float *>(tCrD.data());
  half  * __restrict__ dst = reinterpret_cast<half  *>(tCrD_half.data());

#pragma unroll 2
  for (int i = 0; i + 31 < nelems; i += 32) {
    float2 f0  = make_float2(src[i +  0], src[i +  1]);
    float2 f1  = make_float2(src[i +  2], src[i +  3]);
    float2 f2  = make_float2(src[i +  4], src[i +  5]);
    float2 f3  = make_float2(src[i +  6], src[i +  7]);
    float2 f4  = make_float2(src[i +  8], src[i +  9]);
    float2 f5  = make_float2(src[i + 10], src[i + 11]);
    float2 f6  = make_float2(src[i + 12], src[i + 13]);
    float2 f7  = make_float2(src[i + 14], src[i + 15]);
    float2 f8  = make_float2(src[i + 16], src[i + 17]);
    float2 f9  = make_float2(src[i + 18], src[i + 19]);
    float2 f10 = make_float2(src[i + 20], src[i + 21]);
    float2 f11 = make_float2(src[i + 22], src[i + 23]);
    float2 f12 = make_float2(src[i + 24], src[i + 25]);
    float2 f13 = make_float2(src[i + 26], src[i + 27]);
    float2 f14 = make_float2(src[i + 28], src[i + 29]);
    float2 f15 = make_float2(src[i + 30], src[i + 31]);

    half2 h0  = __float22half2_rn(f0);
    half2 h1  = __float22half2_rn(f1);
    half2 h2  = __float22half2_rn(f2);
    half2 h3  = __float22half2_rn(f3);
    half2 h4  = __float22half2_rn(f4);
    half2 h5  = __float22half2_rn(f5);
    half2 h6  = __float22half2_rn(f6);
    half2 h7  = __float22half2_rn(f7);
    half2 h8  = __float22half2_rn(f8);
    half2 h9  = __float22half2_rn(f9);
    half2 h10 = __float22half2_rn(f10);
    half2 h11 = __float22half2_rn(f11);
    half2 h12 = __float22half2_rn(f12);
    half2 h13 = __float22half2_rn(f13);
    half2 h14 = __float22half2_rn(f14);
    half2 h15 = __float22half2_rn(f15);

    *reinterpret_cast<half2*>(&dst[i +  0]) = h0;
    *reinterpret_cast<half2*>(&dst[i +  2]) = h1;
    *reinterpret_cast<half2*>(&dst[i +  4]) = h2;
    *reinterpret_cast<half2*>(&dst[i +  6]) = h3;
    *reinterpret_cast<half2*>(&dst[i +  8]) = h4;
    *reinterpret_cast<half2*>(&dst[i + 10]) = h5;
    *reinterpret_cast<half2*>(&dst[i + 12]) = h6;
    *reinterpret_cast<half2*>(&dst[i + 14]) = h7;
    *reinterpret_cast<half2*>(&dst[i + 16]) = h8;
    *reinterpret_cast<half2*>(&dst[i + 18]) = h9;
    *reinterpret_cast<half2*>(&dst[i + 20]) = h10;
    *reinterpret_cast<half2*>(&dst[i + 22]) = h11;
    *reinterpret_cast<half2*>(&dst[i + 24]) = h12;
    *reinterpret_cast<half2*>(&dst[i + 26]) = h13;
    *reinterpret_cast<half2*>(&dst[i + 28]) = h14;
    *reinterpret_cast<half2*>(&dst[i + 30]) = h15;
  }

  {
    int base = (nelems / 32) * 32;
#pragma unroll
    for (int i = base; i + 7 < nelems; i += 8) {
      float2 fa = make_float2(src[i + 0], src[i + 1]);
      float2 fb = make_float2(src[i + 2], src[i + 3]);
      float2 fc = make_float2(src[i + 4], src[i + 5]);
      float2 fd = make_float2(src[i + 6], src[i + 7]);
      *reinterpret_cast<half2*>(&dst[i + 0]) = __float22half2_rn(fa);
      *reinterpret_cast<half2*>(&dst[i + 2]) = __float22half2_rn(fb);
      *reinterpret_cast<half2*>(&dst[i + 4]) = __float22half2_rn(fc);
      *reinterpret_cast<half2*>(&dst[i + 6]) = __float22half2_rn(fd);
    }
    int tail = nelems & ~7;
    for (int i = tail; i + 1 < nelems; i += 2) {
      *reinterpret_cast<half2*>(&dst[i]) =
          __float22half2_rn(make_float2(src[i], src[i + 1]));
    }
    if (nelems & 1) {
      dst[nelems - 1] = __float2half_rn(src[nelems - 1]);
    }
  }

  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s  = r2s_thr_copy_c.retile_S(tCrD_half);
  auto tCsC_r2s  = r2s_thr_copy_c.partition_D(sC);

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
      cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    if (i + step < size<1>(tCrC_r2sx)) {
      __syncthreads();
    }
  }
}

template <typename T, const int Stages = 7>
void launch_hgemm_mma_stages_tn_cute(T *a, T *b, T *c, int M, int N, int K) {
  using namespace cute;

  auto BM = Int<64>{};
  auto BN = Int<64>{};
  auto BK = Int<64>{};
  auto KStage = Int<Stages>{};
  auto kSmemLayoutCBatch = Int<2>{};

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BM>{}, Int<BK>{}, KStage)));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, KStage)));

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
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op     = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA    = s2r_copy_atom;
  using S2RCopyAtomB    = s2r_copy_atom;

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                "C shared memory exceeds A's buffer");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY);

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C  = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize    =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  cudaFuncSetAttribute(
      cuda_l2_h100_fp32_kernel<
          T, BM, BN, BK, Stages, MMA,
          G2SCopyA, G2SCopyB,
          SmemLayoutA, SmemLayoutB, SmemLayoutC,
          S2RCopyAtomA, S2RCopyAtomB,
          R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  cuda_l2_h100_fp32_kernel<
      T, BM, BN, BK, Stages, MMA,
      G2SCopyA, G2SCopyB,
      SmemLayoutA, SmemLayoutB, SmemLayoutC,
      S2RCopyAtomA, S2RCopyAtomB,
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
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  launch_hgemm_mma_stages_tn_cute<half, 7>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()),
      M, N, K);
}