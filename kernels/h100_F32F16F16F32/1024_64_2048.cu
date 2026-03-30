#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace cute;

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB>
__global__ void __launch_bounds__(128, 2)
hgemm_optimized_kernel(
    const T *__restrict__ Aptr,
    const T *__restrict__ Bptr,
    T *__restrict__ Cptr,
    int m, int n, int k)
{
  extern __shared__ char smem_buf[];

  T *Ashm = reinterpret_cast<T*>(smem_buf);
  T *Bshm = Ashm + cute::cosize(SmemLayoutA{});

  const int tid = threadIdx.x;
  const int ix = blockIdx.x;
  const int iy = blockIdx.y;

  if (iy * BM >= m || ix * BN >= n) return;

  const T *A_base = Aptr + (long long)(iy * BM) * k;
  const T *B_base = Bptr + (long long)(ix * BN) * k;

  Tensor A_slice = make_tensor(make_gmem_ptr(A_base),
                               make_shape(Int<BM>{}, k),
                               make_stride(k, Int<1>{}));
  Tensor B_slice = make_tensor(make_gmem_ptr(B_base),
                               make_shape(Int<BN>{}, k),
                               make_stride(k, Int<1>{}));

  Tensor gA = local_tile(A_slice, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
  Tensor gB = local_tile(B_slice, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tid);

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  
  auto fake_gC = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(Cptr)),
                             make_shape(Int<BM>{}, Int<BN>{}),
                             make_stride(Int<BN>{}, Int<1>{}));
  auto tCrD = thr_mma.partition_fragment_C(fake_gC);
  clear(tCrD);

  G2SCopyA g2s_copy_a;
  auto g2s_thr_a = g2s_copy_a.get_slice(tid);
  auto tAgA = g2s_thr_a.partition_S(gA);
  auto tAsA = g2s_thr_a.partition_D(sA);

  G2SCopyB g2s_copy_b;
  auto g2s_thr_b = g2s_copy_b.get_slice(tid);
  auto tBgB = g2s_thr_b.partition_S(gB);
  auto tBsB = g2s_thr_b.partition_D(sB);

  using S2RCopyAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;
  using S2RCopyAtomB = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

  auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_slice(tid);
  auto tAsA_s2r = s2r_thr_a.partition_S(sA);
  auto tCrA_view = s2r_thr_a.retile_D(tCrA);

  auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_b = s2r_copy_b.get_slice(tid);
  auto tBsB_s2r = s2r_thr_b.partition_S(sB);
  auto tCrB_view = s2r_thr_b.retile_D(tCrB);

  int ntile = (k + BK - 1) / BK;
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

#pragma unroll
  for (int i = 0; i < kStage - 1 && i < ntile; ++i) {
    cute::copy(g2s_copy_a, tAgA(_, _, _, i), tAsA(_, _, _, i));
    cute::copy(g2s_copy_b, tBgB(_, _, _, i), tBsB(_, _, _, i));
    cp_async_fence();
    ++itile_to_read;
    ++ismem_write;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  if (ntile > 0) {
    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB_view(_, _, 0));
  }

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

      cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
      cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read), tAsA(_, _, _, ismem_write));
          cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read), tBsB(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  __syncthreads();
  float *smem_fp32 = reinterpret_cast<float*>(smem_buf);
  
  constexpr int total_fp32 = BM * BN;
  constexpr int vec4_init = total_fp32 / 4;
  
#pragma unroll 2
  for (int i = tid; i < vec4_init; i += 128) {
    reinterpret_cast<float4*>(smem_fp32)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (tid < (total_fp32 % 4)) {
    smem_fp32[vec4_init * 4 + tid] = 0.0f;
  }
  __syncthreads();

  auto sC_fp32 = make_tensor(make_smem_ptr(smem_fp32),
                             make_shape(Int<BM>{}, Int<BN>{}),
                             make_stride(Int<BN>{}, Int<1>{}));
  auto tCsC = thr_mma.partition_C(sC_fp32);
  cute::copy(tCrD, tCsC);
  __syncthreads();

  half *out = Cptr + (long long)(iy * BM) * n + (long long)(ix * BN);
  
  constexpr int total_elements = BM * BN;
  constexpr int num_threads = 128;
  
  for (int base_idx = tid * 8; base_idx < total_elements; base_idx += num_threads * 8) {
    int m_local = base_idx / BN;
    int n_local = base_idx % BN;
    int m_global = iy * BM + m_local;
    int n_global = ix * BN + n_local;
    
    if (m_global < m && n_global + 7 < n) {
      float4 fp32_0 = *reinterpret_cast<float4*>(&smem_fp32[base_idx]);
      float4 fp32_1 = *reinterpret_cast<float4*>(&smem_fp32[base_idx + 4]);
      
      half2 h0 = __float22half2_rn(make_float2(fp32_0.x, fp32_0.y));
      half2 h1 = __float22half2_rn(make_float2(fp32_0.z, fp32_0.w));
      half2 h2 = __float22half2_rn(make_float2(fp32_1.x, fp32_1.y));
      half2 h3 = __float22half2_rn(make_float2(fp32_1.z, fp32_1.w));
      
      half *out_ptr = &out[m_local * n + n_local];
      reinterpret_cast<half2*>(out_ptr)[0] = h0;
      reinterpret_cast<half2*>(out_ptr)[1] = h1;
      reinterpret_cast<half2*>(out_ptr)[2] = h2;
      reinterpret_cast<half2*>(out_ptr)[3] = h3;
    } else if (m_global < m) {
      for (int j = 0; j < 8 && n_global + j < n && base_idx + j < total_elements; ++j) {
        out[m_local * n + n_local + j] = __float2half(smem_fp32[base_idx + j]);
      }
    }
  }
}

template <typename T>
void launch_optimized_hgemm(T *a, T *b_col_major, T *c, int M, int N, int K) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 64;
  constexpr int KStage = 8;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

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

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY);

  static constexpr int kShmSize =
      (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_optimized_kernel<T, BM, BN, BK, KStage, MMA,
                             G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_optimized_kernel<T, BM, BN, BK, KStage, MMA,
                         G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB>
      <<<grid, block, kShmSize>>>(a, b_col_major, c, M, N, K);
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

  launch_optimized_hgemm<half>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(b_col_major.data_ptr()),
      reinterpret_cast<half *>(c.data_ptr()),
      M, N, K);
}