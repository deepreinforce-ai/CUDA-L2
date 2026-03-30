#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

template <typename T, int BM, int BN, int BK, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB>
__global__ void __launch_bounds__(128, 2)
hgemm_optimized_kernel(T *__restrict__ Aptr, T *__restrict__ Bptr,
                       T *__restrict__ Dptr, int m, int n, int k) {
  using namespace cute;
  
  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + BM * BK;

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, 0));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, 0));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

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

  auto tCsA = thr_mma.partition_A(sA);
  auto tCsB = thr_mma.partition_B(sB);

  int nk = size<2>(tCrA);

  if (nk >= 1) {
    cute::copy(tCsA(_, _, 0), tCrA(_, _, 0));
    cute::copy(tCsB(_, _, 0), tCrB(_, _, 0));
  }
  
  if (nk >= 2) {
    cute::copy(tCsA(_, _, 1), tCrA(_, _, 1));
    cute::copy(tCsB(_, _, 1), tCrB(_, _, 1));
  }
  
  #pragma unroll
  for (int ik = 0; ik < nk; ++ik) {
    if (ik + 2 < nk) {
      cute::copy(tCsA(_, _, ik + 2), tCrA(_, _, ik + 2));
      cute::copy(tCsB(_, _, ik + 2), tCrB(_, _, ik + 2));
    }
    
    cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
  }

  auto tCrD_half = make_tensor_like<T>(tCrD);
  
  constexpr int kVecSize = 2;
  int num_pairs = size(tCrD) / kVecSize;
  
  #pragma unroll
  for (int i = 0; i < num_pairs; ++i) {
    int idx0 = i * kVecSize;
    int idx1 = idx0 + 1;
    
    float2 f2 = make_float2(float(tCrD(idx0)), float(tCrD(idx1)));
    half2 h2 = __float22half2_rn(f2);
    
    tCrD_half(idx0) = h2.x;
    tCrD_half(idx1) = h2.y;
  }
  
  if (size(tCrD) % kVecSize != 0) {
    tCrD_half(size(tCrD) - 1) = static_cast<T>(tCrD(size(tCrD) - 1));
  }

  auto cD = cute::make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
  auto thr_cD = tiled_mma.get_slice(idx).partition_C(cD);
  Tensor tDgD = thr_mma.partition_C(gD);
  
  #pragma unroll
  for (int i = 0; i < size(tCrD_half); ++i) {
    int row = get<0>(thr_cD(i)) + iy * BM;
    int col = get<1>(thr_cD(i)) + ix * BN;
    if (row < m && col < n) {
      tDgD(i) = tCrD_half(i);
    }
  }
}

template <typename T>
void launch_hgemm_optimized(T *a, T *b, T *c, int M, int N, int K) {
  using namespace cute;

  auto BM = Int<64>{};
  auto BN = Int<64>{};
  auto BK = Int<64>{};

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<64>{}),
                  make_stride(Int<64>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<64>{}, Int<64>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<64>{}, Int<64>{})));

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 2 * kMmaEURepeatM * get<0>(mma_atom_shape{});
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

  int BX = (N + 63) / 64;
  int BY = (M + 63) / 64;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY);

  static constexpr int kShmSize = (64 * 64 + 64 * 64) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_optimized_kernel<T, 64, 64, 64, MMA,
                             G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_optimized_kernel<T, 64, 64, 64, MMA,
                         G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB>
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

  half *a_ptr = reinterpret_cast<half *>(a.data_ptr());
  half *b_ptr = reinterpret_cast<half *>(b_col_major.data_ptr());
  half *c_ptr = reinterpret_cast<half *>(c.data_ptr());

  launch_hgemm_optimized<half>(a_ptr, b_ptr, c_ptr, M, N, K);
}