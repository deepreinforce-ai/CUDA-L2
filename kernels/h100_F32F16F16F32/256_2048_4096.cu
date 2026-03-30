#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/util/print_error.hpp>
#include <cutlass/util/helper_cuda.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageOptimized
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  
  uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[cute::size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void gemm_microopt_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
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
  
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));

  auto [M, N, K] = shape_MNK;
  
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{});

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageOptimized<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  auto K_PIPE_MAX = size<1>(tAsA);
  int k_tile_count = size<1>(tAgA);
  int k_tile = 0;

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier;
  
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe], 1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  cluster_sync();

  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);
  
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

  int remaining_tiles = k_tile_count + K_PIPE_MAX;
  
  CUTE_NO_UNROLL
  while (remaining_tiles > 0) {
    
    bool can_dual_stage = (remaining_tiles >= 2) && (k_tile_count > -K_PIPE_MAX + 1);
    
    if (can_dual_stage) {
      int pipe0 = read_state.index();
      int pipe1 = (pipe0 + 1) % K_PIPE_MAX;
      
      ProducerBarType::wait(&producer_mbar[pipe0], read_state.phase());
      
      warpgroup_arrive();
      gemm(mma, tCrA(_, _, _, pipe0), tCrB(_, _, _, pipe0), tCrC);
      warpgroup_commit_batch();
      
      ConsumerBarType::arrive(&consumer_mbar[pipe0]);
      ++read_state;
      --remaining_tiles;
      
      ProducerBarType::wait(&producer_mbar[pipe1], read_state.phase());
      
      warpgroup_arrive();
      gemm(mma, tCrA(_, _, _, pipe1), tCrB(_, _, _, pipe1), tCrC);
      warpgroup_commit_batch();
      
      warpgroup_wait<1>();
      
      ConsumerBarType::arrive(&consumer_mbar[pipe1]);
      ++read_state;
      --remaining_tiles;
      
      if ((warp_idx == 0) && lane_predicate) {
        if (k_tile_count > -K_PIPE_MAX) {
          int wpipe0 = write_state.index();
          ConsumerBarType::wait(&consumer_mbar[wpipe0], write_state.phase());
          ProducerBarType::arrive_and_expect_tx(&producer_mbar[wpipe0], tma_transaction_bytes);
          copy(tma_a.with(producer_mbar[wpipe0]), tAgA(_, k_tile), tAsA(_, wpipe0));
          copy(tma_b.with(producer_mbar[wpipe0]), tBgB(_, k_tile), tBsB(_, wpipe0));
          ++write_state;
          --k_tile_count;
          ++k_tile;
        }
        
        if (k_tile_count > -K_PIPE_MAX) {
          int wpipe1 = write_state.index();
          ConsumerBarType::wait(&consumer_mbar[wpipe1], write_state.phase());
          ProducerBarType::arrive_and_expect_tx(&producer_mbar[wpipe1], tma_transaction_bytes);
          copy(tma_a.with(producer_mbar[wpipe1]), tAgA(_, k_tile), tAsA(_, wpipe1));
          copy(tma_b.with(producer_mbar[wpipe1]), tBgB(_, k_tile), tBsB(_, wpipe1));
          ++write_state;
          --k_tile_count;
          ++k_tile;
        }
      } else {
        int to_issue = (k_tile_count > -K_PIPE_MAX) ? 1 : 0;
        to_issue += (k_tile_count - to_issue > -K_PIPE_MAX) ? 1 : 0;
        k_tile_count -= to_issue;
        k_tile += to_issue;
      }
      
    } else {
      int pipe = read_state.index();
      
      ProducerBarType::wait(&producer_mbar[pipe], read_state.phase());
      
      warpgroup_arrive();
      gemm(mma, tCrA(_, _, _, pipe), tCrB(_, _, _, pipe), tCrC);
      warpgroup_commit_batch();
      
      warpgroup_wait<1>();
      
      ConsumerBarType::arrive(&consumer_mbar[pipe]);
      ++read_state;
      --remaining_tiles;
      
      if ((warp_idx == 0) && lane_predicate) {
        if (k_tile_count > -K_PIPE_MAX) {
          int wpipe = write_state.index();
          ConsumerBarType::wait(&consumer_mbar[wpipe], write_state.phase());
          ProducerBarType::arrive_and_expect_tx(&producer_mbar[wpipe], tma_transaction_bytes);
          copy(tma_a.with(producer_mbar[wpipe]), tAgA(_, k_tile), tAsA(_, wpipe));
          copy(tma_b.with(producer_mbar[wpipe]), tBgB(_, k_tile), tBsB(_, wpipe));
          ++write_state;
          --k_tile_count;
          ++k_tile;
        }
      } else {
        if (k_tile_count > -K_PIPE_MAX) {
          --k_tile_count;
          ++k_tile;
        }
      }
    }
  }

  warpgroup_wait<0>();

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC>
void launch_microopt_hgemm(TA const* A, int ldA, TB const* B, int ldB,
                           TC* C, int ldC, int M, int N, int K)
{
  using namespace cute;

  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(ldC, Int<1>{});

  auto bM = Int< 64>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<5>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

  using SharedStorage = SharedStorageOptimized<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));

  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2, 2, 1);
  dim3 dimGrid(round_up(size(ceil_div(N, bN)), dimCluster.x),
               round_up(size(ceil_div(M, bM)), dimCluster.y));
  
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  auto kernel_ptr = &gemm_microopt_kernel<decltype(prob_shape), decltype(cta_tiler),
                                          TA, decltype(sA), decltype(tmaA),
                                          TB, decltype(sB), decltype(tmaB),
                                          TC, decltype(dC), decltype(tiled_mma),
                                          half_t, half_t>;

  cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  half_t alpha = half_t(1.0f);
  half_t beta  = half_t(0.0f);

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
    params, (void const*) kernel_ptr,
    prob_shape, cta_tiler, A, tmaA, B, tmaB, C, dC, tiled_mma, alpha, beta);
  
  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(result));
  }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor shape mismatch"); \
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

  launch_microopt_hgemm<cute::half_t, cute::half_t, cute::half_t>(
      reinterpret_cast<cute::half_t const*>(a.data_ptr()), K,
      reinterpret_cast<cute::half_t const*>(b_col_major.data_ptr()), K,
      reinterpret_cast<cute::half_t*>(c.data_ptr()), N,
      M, N, K);
}