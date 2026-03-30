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
struct SharedStorageHyper
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  alignas(16) uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
  alignas(16) uint64_t mma_barrier[cute::size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaLoadA,
          class TB, class SmemLayoutB, class TmaLoadB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(cute::size(TiledMma{}))::value, 2)
void gemm_kernel_hyper(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A_ptr, CUTLASS_GRID_CONSTANT TmaLoadA const tma_load_a,
    TB const* B_ptr, CUTLASS_GRID_CONSTANT TmaLoadB const tma_load_b,
    TC* C_ptr, CStride dC, TiledMma tiled_mma,
    Alpha alpha, Beta beta)
{
  using namespace cute;
  
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));

  auto [M, N, K] = shape_MNK;
  
  Tensor mA = tma_load_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_load_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), dC);

  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char shared_memory[];
  using SharedStorageType = SharedStorageHyper<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorageType& smem = *reinterpret_cast<SharedStorageType*>(shared_memory);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_b.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_load_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_load_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int tma_transaction_bytes = 64 * 128 * sizeof(TA) + 128 * 128 * sizeof(TB);

  constexpr int PIPE_STAGES = size<2>(SmemLayoutA{});
  int k_tiles_total = size<1>(tAgA);
  int k_tile_idx = 0;

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  
  uint64_t* producer_barrier = smem.tma_barrier;
  uint64_t* consumer_barrier = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier;
  
  if (warp_idx == 0 && lane_predicate) {
    #pragma unroll
    for (int stage = 0; stage < PIPE_STAGES; ++stage) {
      ProducerBarType::init(&producer_barrier[stage], 1);
      ConsumerBarType::init(&consumer_barrier[stage], 128);
    }
  }
  __syncthreads();

  if (warp_idx == 0 && lane_predicate) {
    constexpr int MAX_PREFILL = PIPE_STAGES;
    #pragma unroll
    for (int stage = 0; stage < MAX_PREFILL; ++stage) {
      if (stage < k_tiles_total) {
        ProducerBarType::arrive_and_expect_tx(&producer_barrier[stage], tma_transaction_bytes);
        
        copy(tma_load_a.with(producer_barrier[stage]), tAgA(_,k_tile_idx), tAsA(_,stage));
        copy(tma_load_b.with(producer_barrier[stage]), tBgB(_,k_tile_idx), tBsB(_,stage));
        
        ++k_tile_idx;
      }
    }
  } else {
    k_tile_idx = cute::min(PIPE_STAGES, k_tiles_total);
  }

  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);
  
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  auto write_state = cutlass::PipelineState<PIPE_STAGES>();
  auto read_state  = cutlass::PipelineState<PIPE_STAGES>();

  int iterations_remaining = k_tiles_total;
  
  #pragma unroll 1
  while (iterations_remaining > 0) {
    int consume_stage = read_state.index();
    
    ProducerBarType::wait(&producer_barrier[consume_stage], read_state.phase());

    warpgroup_arrive();
    
    gemm(tiled_mma, tCrA(_,_,_,consume_stage), tCrB(_,_,_,consume_stage), tCrC);
    
    warpgroup_commit_batch();
    
    ConsumerBarType::arrive(&consumer_barrier[consume_stage]);
    
    warpgroup_wait<1>();
    
    ++read_state;
    
    int produce_stage = write_state.index();
    
    if (warp_idx == 0 && lane_predicate) {
      ConsumerBarType::wait(&consumer_barrier[produce_stage], write_state.phase());
      
      if (k_tile_idx < k_tiles_total) {
        ProducerBarType::arrive_and_expect_tx(&producer_barrier[produce_stage], tma_transaction_bytes);
        
        copy(tma_load_a.with(producer_barrier[produce_stage]), tAgA(_,k_tile_idx), tAsA(_,produce_stage));
        copy(tma_load_b.with(producer_barrier[produce_stage]), tBgB(_,k_tile_idx), tBsB(_,produce_stage));
        
        ++k_tile_idx;
      }
      
      ++write_state;
    }
    
    --iterations_remaining;
  }

  warpgroup_wait<0>();

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC>
void launch_hgemm_hyper(
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC* C, int ldC,
    int M, int N, int K)
{
  using namespace cute;

  auto problem_shape = make_shape(M, N, K);
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(ldC, Int<1>{});

  auto bM = Int<64>{};
  auto bN = Int<128>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<4>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);

  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK), Int<1>{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK), Int<1>{});

  using SharedStorageType = SharedStorageHyper<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorageType));

  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 2, 1);
  dim3 dimGrid(round_up(size(ceil_div(N, bN)), dimCluster.x),
               round_up(size(ceil_div(M, bM)), dimCluster.y));
  
  cutlass::ClusterLaunchParams params{dimGrid, dimBlock, dimCluster, smem_size};

  auto kernel = &gemm_kernel_hyper<
      decltype(problem_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tma_a),
      TB, decltype(sB), decltype(tma_b),
      TC, decltype(dC), decltype(tiled_mma),
      half_t, half_t>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  half_t alpha = half_t(1.0f);
  half_t beta  = half_t(0.0f);

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const*) kernel,
      problem_shape, cta_tiler,
      A, tma_a,
      B, tma_b,
      C, dC, tiled_mma,
      alpha, beta);
  
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    printf("Hyper kernel launch failed: %s\n", cutlassGetStatusString(status));
  }
}

#include <torch/extension.h>
#include <torch/types.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor dtype mismatch: " << (T).options() << std::endl;      \
    throw std::runtime_error("Expected tensor dtype: " #th_type);              \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor shape mismatch!");                        \
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

  launch_hgemm_hyper<cute::half_t, cute::half_t, cute::half_t>(
      reinterpret_cast<cute::half_t const*>(a.data_ptr()),
      K,
      reinterpret_cast<cute::half_t const*>(b_col_major.data_ptr()),
      K,
      reinterpret_cast<cute::half_t*>(c.data_ptr()),
      N,
      M, N, K);
}