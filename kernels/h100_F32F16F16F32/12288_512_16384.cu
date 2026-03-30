#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/util/print_error.hpp>
#include <cutlass/util/helper_cuda.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageBreakthrough
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  
  uint64_t barrier[cute::size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(128, 1)
void gemm_breakthrough_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
    TC* C, CStride dC, TiledMma mma,
    Alpha alpha, Beta beta)
{
  using namespace cute;
  
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  auto [M, N, K] = shape_MNK;
  
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageBreakthrough<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 2>(sA), group_modes<0, 2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 2>(sB), group_modes<0, 2>(gB));

  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  constexpr int K_PIPE_MAX = 4;
  int k_tile_count = size<1>(tAgA);

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  
  using BarrierType = cutlass::arch::ClusterTransactionBarrier;
  
  if ((warp_idx == 0) && lane_predicate) {
    #pragma unroll
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
      BarrierType::init(&smem.barrier[pipe], 1);
    }
  }
  cluster_sync();

  int k_tile = 0;
  
  #pragma unroll
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if (k_tile < k_tile_count)
    {
      if ((warp_idx == 0) && lane_predicate)
      {
        BarrierType::arrive_and_expect_tx(&smem.barrier[pipe], tma_transaction_bytes);
        copy(tma_a.with(smem.barrier[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
        copy(tma_b.with(smem.barrier[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
      }
      ++k_tile;
    }
  }

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  auto read_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();

  CUTE_NO_UNROLL
  for (int tile_idx = 0; tile_idx < k_tile_count; ++tile_idx)
  {
    int read_pipe = read_state.index();
    
    BarrierType::wait(&smem.barrier[read_pipe], read_state.phase());

    warpgroup_arrive();
    gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
    warpgroup_commit_batch();

    warpgroup_wait<2>();

    ++read_state;

    if ((warp_idx == 0) && lane_predicate && (k_tile < k_tile_count))
    {
      int write_pipe = write_state.index();
      
      BarrierType::wait(&smem.barrier[write_pipe], write_state.phase());
      
      BarrierType::arrive_and_expect_tx(&smem.barrier[write_pipe], tma_transaction_bytes);
      copy(tma_a.with(smem.barrier[write_pipe]), tAgA(_, k_tile), tAsA(_, write_pipe));
      copy(tma_b.with(smem.barrier[write_pipe]), tBgB(_, k_tile), tBsB(_, write_pipe));
      
      ++k_tile;
      ++write_state;
    }
  }

  warpgroup_wait<0>();

  if (beta == Beta(0.0f)) {
    #pragma unroll
    for (int i = 0; i < size(tCrC); i += 4) {
      #pragma unroll
      for (int j = 0; j < 4 && (i + j) < size(tCrC); ++j) {
        tCgC(i + j) = static_cast<TC>(alpha * tCrC(i + j));
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < size(tCrC); i += 4) {
      #pragma unroll
      for (int j = 0; j < 4 && (i + j) < size(tCrC); ++j) {
        float acc = alpha * tCrC(i + j);
        float c_val = beta * static_cast<float>(tCgC(i + j));
        tCgC(i + j) = static_cast<TC>(acc + c_val);
      }
    }
  }
}

template <class TA, class TB, class TC>
void launch_breakthrough_hgemm(
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC* C, int ldC,
    int M, int N, int K)
{
  using namespace cute;

  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(ldC, Int<1>{});

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  
  auto bP = Int<4>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

  TiledMMA tiled_mma = make_tiled_mma(
      SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

  using SharedStorage = SharedStorageBreakthrough<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));

  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(N, bN)), dimCluster.x),
               round_up(size(ceil_div(M, bM)), dimCluster.y));
  
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  auto kernel_ptr = &gemm_breakthrough_kernel<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tmaA),
      TB, decltype(sB), decltype(tmaB),
      TC, decltype(dC), decltype(tiled_mma),
      half_t, half_t>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

  half_t alpha = half_t(1.0f);
  half_t beta = half_t(0.0f);

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*)kernel_ptr,
      prob_shape, cta_tiler,
      A, tmaA,
      B, tmaB,
      C, dC, tiled_mma,
      alpha, beta);
  
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    printf("Breakthrough kernel launch failed: status=%d\n", int(status));
  }
}

#include <torch/extension.h>
#include <torch/types.h>

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

  launch_breakthrough_hgemm<cute::half_t, cute::half_t, cute::half_t>(
      reinterpret_cast<cute::half_t const*>(a.data_ptr()),
      K,
      reinterpret_cast<cute::half_t const*>(b_col_major.data_ptr()),
      K,
      reinterpret_cast<cute::half_t*>(c.data_ptr()),
      N,
      M, N, K);
}