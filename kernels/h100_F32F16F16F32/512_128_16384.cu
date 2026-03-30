#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC>
__global__ void __launch_bounds__(128)
hgemm_dual_register_kernel(const T *Aptr, const T *Bptr, T *D_partial,
                           int m, int n, int k, int k_tiles_per_split) {
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  int iz = blockIdx.z;

  if (iy * BM >= m || ix * BN >= n)
    return;

  int k_tiles_total = k / BK;
  int tile_start = iz * k_tiles_per_split;
  int tile_end = min(tile_start + k_tiles_per_split, k_tiles_total);
  int ntile = tile_end - tile_start;

  if (ntile <= 0)
    return;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));

  T *D_slice = D_partial + (size_t)iz * (m * n);
  Tensor D = make_tensor(make_gmem_ptr(D_slice), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  Tensor gA_full = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}),
                              make_coord(iy, _));
  Tensor gB_full = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}),
                              make_coord(ix, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}),
                         make_coord(iy, ix));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);

  auto tCrA = thr_mma.partition_fragment_A(gA_full(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB_full(_, _, 0));
  auto tCrA_next = thr_mma.partition_fragment_A(gA_full(_, _, 0));
  auto tCrB_next = thr_mma.partition_fragment_B(gB_full(_, _, 0));
  auto tCrD = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  G2SCopyA g2s_copy_a;
  auto g2s_thr_a = g2s_copy_a.get_slice(idx);
  auto tAgA = g2s_thr_a.partition_S(gA_full);
  auto tAsA_dst = g2s_thr_a.partition_D(sA);

  G2SCopyB g2s_copy_b;
  auto g2s_thr_b = g2s_copy_b.get_slice(idx);
  auto tBgB = g2s_thr_b.partition_S(gB_full);
  auto tBsB_dst = g2s_thr_b.partition_D(sB);

  auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_slice(idx);
  auto tAsA_src = s2r_thr_a.partition_S(sA);
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);
  auto tCrA_next_view = s2r_thr_a.retile_D(tCrA_next);

  auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_b = s2r_copy_b.get_slice(idx);
  auto tBsB_src = s2r_thr_b.partition_S(sB);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);
  auto tCrB_next_view = s2r_thr_b.retile_D(tCrB_next);

  int itile_to_read = tile_start;
  int ismem_read = 0;
  int ismem_write = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (itile_to_read < tile_end) {
      cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read), tAsA_dst(_, _, _, istage));
      cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read), tBsB_dst(_, _, _, istage));
      ++itile_to_read;
    }
    cp_async_fence();
    ++ismem_write;
  }
  ismem_write = ismem_write % kStage;

  cp_async_wait<kStage - 2>();
  __syncthreads();

  int nk = size<2>(tCrA);

  cute::copy(s2r_copy_a, tAsA_src(_, _, 0, ismem_read), tCrA_view(_, _, 0));
  cute::copy(s2r_copy_b, tBsB_src(_, _, 0, ismem_read), tCrB_view(_, _, 0));
  
  if (nk > 1) {
    cute::copy(s2r_copy_a, tAsA_src(_, _, 1, ismem_read), tCrA_next_view(_, _, 1));
    cute::copy(s2r_copy_b, tBsB_src(_, _, 1, ismem_read), tCrB_next_view(_, _, 1));
  }

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {

    int ik = 0;
    for (; ik < nk - 1; ik += 2) {
      int ik_next0 = ik + 2;
      int ik_next1 = ik + 3;

      if (ik == 0) {
        if (itile_to_read < tile_end) {
          cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read),
                     tAsA_dst(_, _, _, ismem_write));
          cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read),
                     tBsB_dst(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }

      if (ik_next0 < nk) {
        cute::copy(s2r_copy_a, tAsA_src(_, _, ik_next0, ismem_read),
                   tCrA_view(_, _, ik_next0));
      }
      
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
      
      if (ik_next1 < nk) {
        cute::copy(s2r_copy_b, tBsB_src(_, _, ik_next1, ismem_read),
                   tCrB_next_view(_, _, ik_next1));
      }
      
      if (ik_next0 < nk) {
        cute::copy(s2r_copy_b, tBsB_src(_, _, ik_next0, ismem_read),
                   tCrB_view(_, _, ik_next0));
      }

      cute::gemm(tiled_mma, tCrD, tCrA_next(_, _, ik + 1), tCrB_next(_, _, ik + 1), tCrD);
      
      if (ik_next1 < nk) {
        cute::copy(s2r_copy_a, tAsA_src(_, _, ik_next1, ismem_read),
                   tCrA_next_view(_, _, ik_next1));
      }
    }

    if (ik < nk) {
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();
    ismem_read = (ismem_read + 1) % kStage;

    if (itile + 1 < ntile) {
      cute::copy(s2r_copy_a, tAsA_src(_, _, 0, ismem_read), tCrA_view(_, _, 0));
      cute::copy(s2r_copy_b, tBsB_src(_, _, 0, ismem_read), tCrB_view(_, _, 0));
      if (nk > 1) {
        cute::copy(s2r_copy_a, tAsA_src(_, _, 1, ismem_read), tCrA_next_view(_, _, 1));
        cute::copy(s2r_copy_b, tBsB_src(_, _, 1, ismem_read), tCrB_next_view(_, _, 1));
      }
    }
  }

  auto sC = make_tensor(sA(_, _, 0).data(), SmemLayoutC{});

  auto r2s_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_c = r2s_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_c.retile_S(tCrD);
  auto tCsC_r2s = r2s_thr_c.partition_D(sC);

  S2GCopyC s2g_copy_c;
  auto s2g_thr_c = s2g_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_c.partition_S(sC);
  auto tCgC_s2g = s2g_thr_c.partition_D(gD);

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

  int step = size<3>(tCsC_r2s);

#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(r2s_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
  }
}

__global__ void splitk_reduce_optimized(const half *__restrict__ D_partial,
                                        half *__restrict__ D_out,
                                        int m, int n, int split_k) {
  int elem_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int total_elems = m * n;
  
  if (elem_idx + 3 < total_elems) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    
#pragma unroll 8
    for (int sk = 0; sk < split_k; ++sk) {
      const half *src = D_partial + (size_t)sk * total_elems + elem_idx;
      uint2 data = *reinterpret_cast<const uint2 *>(src);
      const half *h = reinterpret_cast<const half *>(&data);
      
      acc[0] += __half2float(h[0]);
      acc[1] += __half2float(h[1]);
      acc[2] += __half2float(h[2]);
      acc[3] += __half2float(h[3]);
    }
    
    half out[4];
    out[0] = __float2half(acc[0]);
    out[1] = __float2half(acc[1]);
    out[2] = __float2half(acc[2]);
    out[3] = __float2half(acc[3]);
    
    *reinterpret_cast<uint2 *>(D_out + elem_idx) =
        *reinterpret_cast<const uint2 *>(out);
  } else if (elem_idx < total_elems) {
    for (int e = elem_idx; e < min(elem_idx + 4, total_elems); ++e) {
      float sum = 0.f;
      for (int sk = 0; sk < split_k; ++sk) {
        sum += __half2float(D_partial[(size_t)sk * total_elems + e]);
      }
      D_out[e] = __float2half(sum);
    }
  }
}

template <typename T, const int Stages = 5>
void launch_hgemm_dual_register(const T *a, const T *b, T *c,
                                int M, int N, int K,
                                T *d_partial_buf, int split_k) {
  using namespace cute;

  auto BM = Int<128>{};
  auto BN = Int<128>{};
  auto BK = Int<64>{};
  auto KStage = Int<Stages>{};
  auto kSmemLayoutCBatch = Int<4>{};

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

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

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  int k_tiles_total = K / BK;
  int k_tiles_per_split = (k_tiles_total + split_k - 1) / split_k;

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  int BZ = split_k;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, BZ);

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_dual_register_kernel<T, BM, BN, BK, Stages, MMA,
                                 G2SCopyA, G2SCopyB,
                                 SmemLayoutA, SmemLayoutB, SmemLayoutC,
                                 S2RCopyAtomA, S2RCopyAtomB,
                                 R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_dual_register_kernel<T, BM, BN, BK, Stages, MMA,
                             G2SCopyA, G2SCopyB,
                             SmemLayoutA, SmemLayoutB, SmemLayoutC,
                             S2RCopyAtomA, S2RCopyAtomB,
                             R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>
      <<<grid, block, kShmSize>>>(a, b, d_partial_buf, M, N, K,
                                  k_tiles_per_split);

  int total_elems = M * N;
  int reduce_threads = 256;
  int elems_per_thread = 4;
  int reduce_blocks = (total_elems + reduce_threads * elems_per_thread - 1) /
                      (reduce_threads * elems_per_thread);

  splitk_reduce_optimized<<<reduce_blocks, reduce_threads>>>(
      d_partial_buf, c, M, N, split_k);
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

static half *g_partial_buf = nullptr;
static size_t g_partial_buf_size = 0;

half *get_partial_buf(size_t required_bytes) {
  if (required_bytes > g_partial_buf_size) {
    if (g_partial_buf)
      cudaFree(g_partial_buf);
    cudaMalloc(&g_partial_buf, required_bytes);
    g_partial_buf_size = required_bytes;
  }
  return g_partial_buf;
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

  const int split_k = 32;

  size_t partial_bytes = (size_t)split_k * M * N * sizeof(half);
  half *partial_buf = get_partial_buf(partial_bytes);

  launch_hgemm_dual_register<half, 5>(
      reinterpret_cast<const half *>(a.data_ptr()),
      reinterpret_cast<const half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()),
      M, N, K, partial_buf, split_k);
}