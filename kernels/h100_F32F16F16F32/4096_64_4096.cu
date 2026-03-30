#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

__global__ void __launch_bounds__(256)
hgemm_optimized_reduce_kernel(
    const float * __restrict__ Workspace,
    half  * __restrict__ C_out,
    int total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= total) return;

  const int stride = total;

  float sum = Workspace[0 * stride + idx] +
              Workspace[1 * stride + idx] +
              Workspace[2 * stride + idx] +
              Workspace[3 * stride + idx];

  C_out[idx] = __float2half(sum);
}

__global__ void __launch_bounds__(256)
hgemm_vec4_reduce_kernel(
    const float * __restrict__ Workspace,
    half  * __restrict__ C_out,
    int total)
{
  const int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int total4 = total / 4;
  
  if (idx4 >= total4) return;

  const int stride = total;

  float4 sum4 = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
  for (int s = 0; s < 4; s++) {
    float4 v = reinterpret_cast<const float4*>(Workspace + s * stride)[idx4];
    sum4.x += v.x;
    sum4.y += v.y;
    sum4.z += v.z;
    sum4.w += v.w;
  }

  half2 h01 = __floats2half2_rn(sum4.x, sum4.y);
  half2 h23 = __floats2half2_rn(sum4.z, sum4.w);
  reinterpret_cast<half2*>(C_out)[idx4 * 2 + 0] = h01;
  reinterpret_cast<half2*>(C_out)[idx4 * 2 + 1] = h23;
}

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,
          typename S2RCopyAtomA, typename S2RCopyAtomB,
          typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void __launch_bounds__(128, 2)
hgemm_splitk_bk64_kernel(
    const T * __restrict__ Aptr,
    const T * __restrict__ Bptr,
    float   * __restrict__ Workspace,
    int m, int n, int k, int k_split)
{
  using namespace cute;

  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  const int idx = threadIdx.x;
  const int ix  = blockIdx.x;
  const int iy  = blockIdx.y;
  const int iz  = blockIdx.z;

  if (iy * BM >= m || ix * BN >= n) return;

  const int k_start = iz * k_split;

  const T *A_ptr = Aptr + (iy * BM) * k + k_start;
  const T *B_ptr = Bptr + (ix * BN) * k + k_start;
  float *D_ptr = Workspace + iz * m * n + (iy * BM) * n + (ix * BN);

  Tensor A = make_tensor(make_gmem_ptr(A_ptr),
                         make_shape(Int<BM>{}, k_split),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(B_ptr),
                         make_shape(Int<BN>{}, k_split),
                         make_stride(k, Int<1>{}));
  Tensor gD_fp32 = make_tensor(make_gmem_ptr(D_ptr),
                               make_shape(Int<BM>{}, Int<BN>{}),
                               make_stride(n, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_fragment_C(gD_fp32);
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
  auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA             = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB             = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);

  int itile_to_read = 0;
  int ismem_read    = 0;
  int ismem_write   = 0;
  const int ntile = k_split / BK;

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
    const int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      const int ik_next = (ik + 1) % nk;

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

  auto tCgD_fp32 = thr_mma.partition_C(gD_fp32);
  cute::copy(tCrD, tCgD_fp32);
}

template <typename T, int kStage, int kSplits>
void launch_optimized_splitk(T *a, T *b_col_major, float *workspace, T *c,
                            int M, int N, int K)
{
  using namespace cute;

  static constexpr int BM = 128;
  static constexpr int BN = 64;
  static constexpr int BK = 64;
  auto kSmemLayoutCBatch = Int<4>{};

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})));

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

  using g2s_copy_op    = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom  = Copy_Atom<g2s_copy_traits, T>;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op    = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom  = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                "C smem overflow");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  const int k_split = K / kSplits;
  const int BX = (N + BN - 1) / BN;
  const int BY = (M + BM - 1) / BM;
  const int BZ = kSplits;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, BZ);

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  cudaFuncSetAttribute(
      hgemm_splitk_bk64_kernel<
          T, BM, BN, BK, kStage, MMA, G2SCopyA, G2SCopyB,
          SmemLayoutA, SmemLayoutB, SmemLayoutC,
          S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  hgemm_splitk_bk64_kernel<
      T, BM, BN, BK, kStage, MMA, G2SCopyA, G2SCopyB,
      SmemLayoutA, SmemLayoutB, SmemLayoutC,
      S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>
      <<<grid, block, kShmSize>>>(a, b_col_major, workspace, M, N, K, k_split);

  const int total = M * N;
  if (total % 4 == 0) {
    const int total4 = total / 4;
    const int nblocks = (total4 + 255) / 256;
    hgemm_vec4_reduce_kernel<<<nblocks, 256>>>(workspace, c, total);
  } else {
    const int nblocks = (total + 255) / 256;
    hgemm_optimized_reduce_kernel<<<nblocks, 256>>>(workspace, c, total);
  }
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

static float *g_workspace = nullptr;
static size_t g_workspace_size = 0;

static float* get_workspace(size_t size_bytes) {
  if (g_workspace == nullptr || g_workspace_size < size_bytes) {
    if (g_workspace != nullptr) cudaFree(g_workspace);
    cudaMalloc(&g_workspace, size_bytes);
    g_workspace_size = size_bytes;
  }
  return g_workspace;
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

  static constexpr int kSplits = 4;
  static constexpr int kStage  = 4;

  size_t ws_size = (size_t)kSplits * M * N * sizeof(float);
  float *workspace = get_workspace(ws_size);

  launch_optimized_splitk<half, kStage, kSplits>(
      reinterpret_cast<half*>(a.data_ptr()),
      reinterpret_cast<half*>(b_col_major.data_ptr()),
      workspace,
      reinterpret_cast<half*>(c.data_ptr()),
      M, N, K);
}