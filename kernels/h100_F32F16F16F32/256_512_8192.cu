#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/util/print_error.hpp>
#include <cutlass/util/helper_cuda.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageOptimized {
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  
  alignas(16) uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
  alignas(16) uint64_t mma_barrier[cute::size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(128, 2)
void gemm_kernel_optimized_async(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
    TC* C, CStride dC, TiledMma mma,
    Alpha alpha, Beta beta)
{
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  auto [M, N, K] = shape_MNK;
  
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageOptimized<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr uint32_t tma_transaction_bytes = 
      sizeof(decltype(make_tensor_like(tensor<0>(tAsA)))) +
      sizeof(decltype(make_tensor_like(tensor<0>(tBsB))));

  constexpr int K_PIPE_MAX = decltype(size<1>(tAsA))::value;
  int k_tile_total = size<1>(tAgA);

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier;
  
  if ((warp_idx == 0) && lane_predicate) {
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
      ProducerBarType::init(&producer_mbar[pipe], 1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  __syncthreads();
  cluster_sync();

  int k_tile = 0;
  if ((warp_idx == 0) && lane_predicate) {
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX && k_tile < k_tile_total; ++pipe) {
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
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

  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto read_state = cutlass::PipelineState<K_PIPE_MAX>();

  CUTE_UNROLL
  for (int i = 0; i < K_PIPE_MAX && i < k_tile_total; ++i) {
    ++write_state;
  }

  int k_iter = 0;
  CUTE_NO_UNROLL
  while (k_iter < k_tile_total) {
    int read_pipe = read_state.index();
    
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    warpgroup_arrive();
    gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
    warpgroup_commit_batch();
    
    if ((warp_idx == 0) && lane_predicate && k_tile < k_tile_total) {
      int pipe = write_state.index();
      
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
      
      ++write_state;
      ++k_tile;
    }
    
    int remaining = k_tile_total - k_iter;
    if (remaining > 5) {
      warpgroup_wait<2>();
    } else if (remaining > 3) {
      warpgroup_wait<1>();
    } else {
      warpgroup_wait<0>();
    }
    
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    
    ++read_state;
    ++k_iter;
  }

  warpgroup_wait<0>();

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC>
void launch_hgemm_optimized_async(
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC* C, int ldC,
    int M, int N, int K)
{
  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(ldC, Int<1>{});

  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<7>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);

  auto tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
  auto tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

  using SharedStorage = SharedStorageOptimized<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));

  dim3 dimBlock(128);
  dim3 dimCluster(2, 2, 1);
  dim3 dimGrid(
      round_up(size(ceil_div(N, bN)), dimCluster.x),
      round_up(size(ceil_div(M, bM)), dimCluster.y));
  
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  auto kernel = &gemm_kernel_optimized_async<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tmaA),
      TB, decltype(sB), decltype(tmaB),
      TC, decltype(dC), decltype(tiled_mma),
      half_t, half_t>;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  half_t alpha = half_t(1.0f);
  half_t beta = half_t(0.0f);

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*)kernel,
      prob_shape, cta_tiler,
      A, tmaA, B, tmaB, C, dC, tiled_mma,
      alpha, beta);
  
  if (status != cutlass::Status::kSuccess) {
    printf("Optimized async kernel launch failed with status: %d\n", static_cast<int>(status));
  }
}

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor b_col_major, torch::Tensor c)
{
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  launch_hgemm_optimized_async<half_t, half_t, half_t>(
      reinterpret_cast<half_t const*>(a.data_ptr()), K,
      reinterpret_cast<half_t const*>(b_col_major.data_ptr()), K,
      reinterpret_cast<half_t*>(c.data_ptr()), N,
      M, N, K);
}