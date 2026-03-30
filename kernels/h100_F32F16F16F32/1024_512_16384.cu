#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/util/print_error.hpp>
#include <cutlass/util/helper_cuda.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageOptimized
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  
  alignas(128) uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
  alignas(128) uint64_t mma_barrier[cute::size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(cute::size(TiledMma{}))::value, 4)
void gemm_optimized_64x64_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                 TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                 TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                                 TC* C, CStride dC, TiledMma mma,
                                 Alpha alpha, Beta beta)
{
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);
  
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));
  
  auto [M, N, K] = shape_MNK;
  
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);
  
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});
  
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
  
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  bool is_producer = (warp_idx == 0) && lane_predicate;
  
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;
  
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier;
  
  constexpr int K_PIPE_MAX = 3;
  
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if (is_producer) {
      ProducerBarType::init(&producer_mbar[pipe], 1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  __syncthreads();
  
  int k_tile_count = size<1>(tAgA);
  if (k_tile_count == 0) return;
  
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);
  
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  
  int k_tile = 0;
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX && k_tile < k_tile_count; ++pipe) {
    if (is_producer) {
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    ++k_tile;
  }
  
  int read_stage = 0;
  int write_stage = K_PIPE_MAX;
  
  CUTE_NO_UNROLL
  for (int iter = 0; iter < k_tile_count; ++iter) {
    int read_pipe = read_stage % K_PIPE_MAX;
    int read_phase = (read_stage / K_PIPE_MAX) & 1;
    
    ProducerBarType::wait(&producer_mbar[read_pipe], read_phase);
    
    warpgroup_arrive();
    
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    
    warpgroup_commit_batch();
    
    if (k_tile < k_tile_count) {
      if (is_producer) {
        int write_pipe = write_stage % K_PIPE_MAX;
        int write_phase = (write_stage / K_PIPE_MAX) & 1;
        
        ConsumerBarType::wait(&consumer_mbar[write_pipe], write_phase);
        
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[write_pipe], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[write_pipe]), tAgA(_,k_tile), tAsA(_,write_pipe));
        copy(tma_b.with(producer_mbar[write_pipe]), tBgB(_,k_tile), tBsB(_,write_pipe));
        
        ++write_stage;
      }
      ++k_tile;
    }
    
    warpgroup_wait<0>();
    
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    
    ++read_stage;
  }
  
  using ComputeType = typename decltype(tCrC)::value_type;
  ComputeType alpha_val = static_cast<ComputeType>(alpha);
  ComputeType beta_val = static_cast<ComputeType>(beta);
  
  if (beta_val == ComputeType(0)) {
    if (alpha_val == ComputeType(1)) {
      copy(tCrC, tCgC);
    } else {
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        tCrC(i) = alpha_val * tCrC(i);
      }
      copy(tCrC, tCgC);
    }
  } 
  else {
    axpby(alpha, tCrC, beta, tCgC);
  }
}

template <class TA, class TB, class TC>
void launch_optimized_64x64_hgemm(
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
  auto bP = Int<3>{};
  
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));
  
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});
  
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);
  
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
  
  using SharedStorage = SharedStorageOptimized<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));
  
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  
  auto cta_m = size(ceil_div(M, bM));
  auto cta_n = size(ceil_div(N, bN));
  dim3 dimGrid(cta_m, cta_n);
  
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  
  auto kernel_ptr = &gemm_optimized_64x64_kernel<decltype(prob_shape), decltype(cta_tiler),
                                                 TA, decltype(sA), decltype(tmaA),
                                                 TB, decltype(sB), decltype(tmaB),
                                                 TC, decltype(dC), decltype(tiled_mma),
                                                 half_t, half_t>;
  
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));
  
  half_t alpha = half_t(1.0f);
  half_t beta  = half_t(0.0f);
  
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
    params, (void const*) kernel_ptr,
    prob_shape, cta_tiler,
    A, tmaA,
    B, tmaB,
    C, dC, tiled_mma,
    alpha, beta);
  
  CUTE_CHECK_LAST();
  
  if (status != cutlass::Status::kSuccess) {
    printf("Error: Kernel launch failed\n");
  }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!"); \
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
  
  launch_optimized_64x64_hgemm<cute::half_t, cute::half_t, cute::half_t>(
      reinterpret_cast<cute::half_t const*>(a.data_ptr()),
      K,
      reinterpret_cast<cute::half_t const*>(b_col_major.data_ptr()),
      K,
      reinterpret_cast<cute::half_t*>(c.data_ptr()),
      N,
      M, N, K);
}