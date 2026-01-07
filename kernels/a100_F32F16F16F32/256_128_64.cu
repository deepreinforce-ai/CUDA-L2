#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB>
__global__ void cuda_l2_a100_fp32_kernel(T *Aptr, T *Bptr,
                                                  T *Dptr, int m,
                                                  int n, int k) {
  using namespace cute;
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + BM * BK;

  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  if (by * BM >= m || bx * BN >= n)
    return;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}),
                         make_coord(by, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}),
                         make_coord(bx, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}),
                         make_coord(by, bx));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tid);
  
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  auto tAsA_mma = thr_mma.partition_A(sA);
  auto tBsB_mma = thr_mma.partition_B(sB);

  int ntile = k / BK;
  
  #pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile), tAsA_copy);
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile), tBsB_copy);
    __syncthreads();

    int nk = size<2>(tCrA);
    #pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      cute::copy(tAsA_mma(_, _, ik), tCrA(_, _, ik));
      cute::copy(tBsB_mma(_, _, ik), tCrB(_, _, ik));
      
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
    __syncthreads();
  }

  auto tCrD_half = make_tensor_like<T>(tCrD);
  cute::copy(tCrD, tCrD_half);
  
  auto tCgD = thr_mma.partition_C(gD);
  cute::copy(tCrD_half, tCgD);
}

template <typename T>
void launch_hgemm_optimized_cute(T *a, T *b, T *c, int M, int N, int K) {
  using namespace cute;

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{})));

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  
  constexpr int kMmaEURepeatM = 2;
  constexpr int kMmaEURepeatN = 2;
  constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
  
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op = UniversalCopy<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY);

  int shm_size = (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(T);

  cuda_l2_a100_fp32_kernel<
      T, BM, BN, BK, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB
      ><<<grid, block, shm_size>>>(a, b, c, M, N, K);
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

void cuda_l2_a100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  
  launch_hgemm_optimized_cute<half>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()), 
      M, N, K);
}