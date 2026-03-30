#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template <typename T, int BM, int BN, int BK, int kStage, int K_SPLITS,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128, 2)
hgemm_splitk_optimized(
    const T * __restrict__ Aptr,
    const T * __restrict__ Bptr,
    float   * __restrict__ C_partial,
    int m, int n, int k
) {
  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int bx  = blockIdx.x;
  int by  = blockIdx.y;
  int bz  = blockIdx.z;

  int k_per_split = k / K_SPLITS;
  int k_start = bz * k_per_split;
  int k_end   = k_start + k_per_split;

  if (by * BM >= m || bx * BN >= n) return;

  Tensor A = make_tensor(make_gmem_ptr(Aptr),
                         make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr),
                         make_shape(n, k), make_stride(k, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(bx, _));

  auto gD_dummy = make_tensor(make_gmem_ptr((T*)nullptr),
                              make_shape(Int<BM>{}, Int<BN>{}),
                              make_stride(Int<BN>{}, Int<1>{}));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA    = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB    = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD    = thr_mma.partition_fragment_C(gD_dummy);
  clear(tCrD);

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a  = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy       = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy       = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b  = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy       = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy       = g2s_thr_copy_b.partition_D(sB);

  auto s2r_tiled_copy_a  = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a    = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA              = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view         = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b  = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b    = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB              = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view         = s2r_thr_copy_b.retile_D(tCrB);

  int k_tile_start = k_start / BK;
  int k_tile_end   = k_end / BK;
  int ntile        = k_tile_end - k_tile_start;

  int itile_to_read  = k_tile_start;
  int ismem_read     = 0;
  int ismem_write    = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1 && itile_to_read < k_tile_end; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read;
    ++ismem_write;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

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

      cute::copy(s2r_tiled_copy_a,
                 tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b,
                 tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < k_tile_end) {
          cute::copy(g2s_tiled_copy_a,
                     tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b,
                     tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD,
                 tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  float *out_base = C_partial + (size_t)bz * m * n;
  
  auto cD = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
  auto tCcD = thr_mma.partition_C(cD);
  
#pragma unroll
  for (int i = 0; i < size(tCrD); ++i) {
    auto coord = tCcD(i);
    int local_m = get<0>(coord);
    int local_n = get<1>(coord);
    int global_m = by * BM + local_m;
    int global_n = bx * BN + local_n;
    
    if (global_m < m && global_n < n) {
      out_base[global_m * n + global_n] = tCrD(i);
    }
  }
}

template<int K_SPLITS>
__global__ void __launch_bounds__(256, 4)
reduce_splitk_kernel(
    const float * __restrict__ partials,
    half        * __restrict__ output,
    int MN
) {
  int idx = blockIdx.x * 256 + threadIdx.x;
  
  if (idx >= MN) return;
  
  float sum = 0.0f;
  
#pragma unroll
  for (int k = 0; k < K_SPLITS; k++) {
    sum += partials[(size_t)k * MN + idx];
  }
  
  output[idx] = __float2half(sum);
}

namespace {
    float  *g_workspace      = nullptr;
    size_t  g_workspace_size = 0;

    inline float* workspace_get(size_t bytes) {
        if (bytes > g_workspace_size) {
            if (g_workspace) cudaFree(g_workspace);
            cudaMalloc(&g_workspace, bytes);
            g_workspace_size = bytes;
        }
        return g_workspace;
    }
}

template <typename T, int Stages, int K_SPLITS>
void launch_hgemm_splitk(T *a, T *b, T *c, int M, int N, int K) {
  auto BM = Int<64>{};
  auto BN = Int<64>{};
  auto BK = Int<64>{};
  auto KStage = Int<Stages>{};

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, decltype(BK){}),
                  make_stride(decltype(BK){}, Int<1>{}))));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(decltype(BM){}, decltype(BK){}, decltype(KStage){})));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(decltype(BN){}, decltype(BK){}, decltype(KStage){})));

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
  using MMA     = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T>;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op     = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA    = s2r_copy_atom;
  using S2RCopyAtomB    = s2r_copy_atom;

  size_t ws_bytes = (size_t)K_SPLITS * M * N * sizeof(float);
  float *C_partial = workspace_get(ws_bytes);

  int BX = (N + 64 - 1) / 64;
  int BY = (M + 64 - 1) / 64;
  int BZ = K_SPLITS;
  dim3 grid(BX, BY, BZ);
  dim3 block(size(MMA{}));

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int kShmSize = shm_size_AB * (int)sizeof(T);

  cudaFuncSetAttribute(
      hgemm_splitk_optimized<
          T, 64, 64, 64, Stages, K_SPLITS, MMA,
          G2SCopyA, G2SCopyB,
          SmemLayoutA, SmemLayoutB,
          S2RCopyAtomA, S2RCopyAtomB>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_splitk_optimized<
      T, 64, 64, 64, Stages, K_SPLITS, MMA,
      G2SCopyA, G2SCopyB,
      SmemLayoutA, SmemLayoutB,
      S2RCopyAtomA, S2RCopyAtomB>
      <<<grid, block, kShmSize>>>(a, b, C_partial, M, N, K);

  int MN = M * N;
  int red_blocks = (MN + 255) / 256;
  reduce_splitk_kernel<K_SPLITS>
      <<<red_blocks, 256>>>(C_partial, c, MN);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,          torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,          torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major,torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,          torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  launch_hgemm_splitk<half, 4, 16>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()),
      M, N, K);
}