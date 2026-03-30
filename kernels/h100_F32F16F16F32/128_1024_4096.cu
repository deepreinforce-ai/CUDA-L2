#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <float.h>
#include <mma.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cute/tensor.hpp>

using namespace cute;

template <typename T, int BM, int BN, int BK, int kStage, int kSplitK,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void __launch_bounds__(128)
hgemm_splitk_phase1_kernel(
    const T *__restrict__ Aptr,
    const T *__restrict__ Bptr,
    float   *__restrict__ workspace,
    int m, int n, int k)
{
  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx   = threadIdx.x;
  int bx    = blockIdx.x;
  int by    = blockIdx.y;
  int split = blockIdx.z;

  if (by * BM >= m || bx * BN >= n) return;

  int k_per_split = k / kSplitK;
  int k_start     = split * k_per_split;
  int k_tiles     = k_per_split / BK;

  Tensor A_full = make_tensor(make_gmem_ptr(Aptr + k_start),
                              make_shape(m, k_per_split),
                              make_stride(k, Int<1>{}));
  Tensor B_full = make_tensor(make_gmem_ptr(Bptr + k_start),
                              make_shape(n, k_per_split),
                              make_stride(k, Int<1>{}));

  Tensor gA = local_tile(A_full, make_tile(Int<BM>{}, Int<BK>{}),
                         make_coord(by, _));
  Tensor gB = local_tile(B_full, make_tile(Int<BN>{}, Int<BK>{}),
                         make_coord(bx, _));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA    = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB    = thr_mma.partition_fragment_B(gB(_, _, 0));

  Tensor W_full = make_tensor(make_gmem_ptr(workspace + (long)split * m * n),
                              make_shape(m, n),
                              make_stride(n, Int<1>{}));
  Tensor gW     = local_tile(W_full, make_tile(Int<BM>{}, Int<BN>{}),
                              make_coord(by, bx));

  auto tCrD = thr_mma.partition_fragment_C(gW);
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

  int ntile          = k_tiles;
  int itile_to_read  = 0;
  int ismem_read     = 0;
  int ismem_write    = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
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

  auto tCgW = thr_mma.partition_C(gW);
  cute::copy(tCrD, tCgW);
}


template <int kSplitK>
__global__ void __launch_bounds__(256)
hgemm_splitk_reduce_vectorized_kernel(
    const float *__restrict__ workspace,
    half        *__restrict__ out,
    int M, int N)
{
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  
  int global_tid = blockIdx.x * blockDim.x + tid;
  int total_elements = M * N;
  
  if (global_tid >= total_elements) return;

  int stride = M * N;
  
  float acc = 0.0f;
  
  #pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    acc += workspace[s * stride + global_tid];
  }
  
  out[global_tid] = __float2half(acc);
}


template <typename T>
void launch_splitk_hgemm(
    const T *a, const T *bT, T *c, float *workspace,
    int M, int N, int K)
{
  constexpr int BM     = 128;
  constexpr int BN     = 64;
  constexpr int BK     = 64;
  constexpr int kStage = 5;
  constexpr int kSplitK = 8;

  if (K % (BK * kSplitK) != 0) {
    printf("ERROR: K must be divisible by BK*kSplitK\n");
    return;
  }

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
  using G2SCopyA        = decltype(make_tiled_copy(
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

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;

  dim3 grid(BX, BY, kSplitK);
  dim3 block(size(MMA{}));

  static constexpr int shm_size_AB =
      (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_splitk_phase1_kernel<T, BM, BN, BK, kStage, kSplitK,
                                 MMA, G2SCopyA, G2SCopyB,
                                 SmemLayoutA, SmemLayoutB,
                                 S2RCopyAtomA, S2RCopyAtomB>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size_AB);

  hgemm_splitk_phase1_kernel<T, BM, BN, BK, kStage, kSplitK,
                             MMA, G2SCopyA, G2SCopyB,
                             SmemLayoutA, SmemLayoutB,
                             S2RCopyAtomA, S2RCopyAtomB>
      <<<grid, block, shm_size_AB>>>(a, bT, workspace, M, N, K);

  int total_elements = M * N;
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  hgemm_splitk_reduce_vectorized_kernel<kSplitK>
      <<<num_blocks, threads_per_block>>>(workspace, c, M, N);
}


#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
  if ((T).options().dtype() != (th_type)) {                        \
    std::cout << "Tensor Info:" << (T).options() << std::endl;     \
    throw std::runtime_error("values must be " #th_type);          \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                        \
  if ((T).size(0) != (S0) || (T).size(1) != (S1)) {               \
    throw std::runtime_error("Tensor size mismatch!");             \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                torch::Tensor b_col_major, torch::Tensor c)
{
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

  constexpr int kSplitK = 8;
  auto workspace = torch::empty({kSplitK, M, N},
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(a.device()));

  const half  *a_ptr  = reinterpret_cast<const half *>(a.data_ptr());
  const half  *bT_ptr = reinterpret_cast<const half *>(b_col_major.data_ptr());
  half        *c_ptr  = reinterpret_cast<half *>(c.data_ptr());
  float       *ws_ptr = reinterpret_cast<float *>(workspace.data_ptr());

  launch_splitk_hgemm<half>(a_ptr, bT_ptr, c_ptr, ws_ptr, M, N, K);
}