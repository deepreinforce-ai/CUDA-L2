#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/mma_sm90.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

#ifndef CUTLASS_GRID_CONSTANT
#define CUTLASS_GRID_CONSTANT __grid_constant__
#endif

template <int Stages>
struct MinimalPipelineState {
  int index_ = 0;
  int phase_ = 0;
  
  __device__ int index() const { return index_; }
  __device__ int phase() const { return phase_; }
  
  __device__ MinimalPipelineState& operator++() {
    ++index_;
    if (index_ == Stages) {
      index_ = 0;
      phase_ ^= 1;
    }
    return *this;
  }
};

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct MinimalSharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  alignas(64) uint64_t barrier[6];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(128)
void gemm_async_epilogue_kernel(
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
  using SharedStorage = MinimalSharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_b.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int tma_transaction_bytes = 
      sizeof(decltype(make_tensor_like(tensor<0>(tAsA)))) +
      sizeof(decltype(make_tensor_like(tensor<0>(tBsB))));

  constexpr int K_PIPE_MAX = 6;
  int k_tile_count = size<1>(tAgA);

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  
  uint64_t* barrier = smem.barrier;

  using BarrierType = cutlass::arch::ClusterTransactionBarrier;

  if ((warp_idx == 0) && lane_predicate) {
    #pragma unroll
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
      BarrierType::init(&barrier[pipe], 1);
    }
  }
  __syncthreads();

  int k_tile = 0;
  int prefetch_depth = min(K_PIPE_MAX, k_tile_count);
  
  if ((warp_idx == 0) && lane_predicate) {
    #pragma unroll
    for (int pipe = 0; pipe < prefetch_depth; ++pipe) {
      BarrierType::arrive_and_expect_tx(&barrier[pipe], tma_transaction_bytes);
      copy(tma_a.with(barrier[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
      copy(tma_b.with(barrier[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
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

  auto write_state = MinimalPipelineState<K_PIPE_MAX>();
  auto read_state = MinimalPipelineState<K_PIPE_MAX>();

  #pragma unroll
  for (int i = 0; i < prefetch_depth; ++i) {
    ++write_state;
  }

  bool is_producer = (warp_idx == 0) && lane_predicate;
  int iterations_remaining = k_tile_count;

  while (iterations_remaining >= 8) {
    #pragma unroll
    for (int iter = 0; iter < 8; ++iter) {
      int pipe_idx = read_state.index();
      
      BarrierType::wait(&barrier[pipe_idx], read_state.phase());
      
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,pipe_idx), tCrB(_,_,_,pipe_idx), tCrC);
      warpgroup_commit_batch();

      if (is_producer && k_tile < k_tile_count) {
        int write_pipe = write_state.index();
        
        BarrierType::arrive_and_expect_tx(&barrier[write_pipe], tma_transaction_bytes);
        
        copy(tma_a.with(barrier[write_pipe]), tAgA(_, k_tile), tAsA(_, write_pipe));
        copy(tma_b.with(barrier[write_pipe]), tBgB(_, k_tile), tBsB(_, write_pipe));
        
        ++write_state;
        ++k_tile;
      }

      if (iter < 7) {
        warpgroup_wait<2>();
      } else {
        warpgroup_wait<1>();
      }
      
      ++read_state;
    }
    
    iterations_remaining -= 8;
  }

  while (iterations_remaining >= 4) {
    #pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
      int pipe_idx = read_state.index();
      
      BarrierType::wait(&barrier[pipe_idx], read_state.phase());
      
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,pipe_idx), tCrB(_,_,_,pipe_idx), tCrC);
      warpgroup_commit_batch();

      if (is_producer && k_tile < k_tile_count) {
        int write_pipe = write_state.index();
        BarrierType::arrive_and_expect_tx(&barrier[write_pipe], tma_transaction_bytes);
        copy(tma_a.with(barrier[write_pipe]), tAgA(_, k_tile), tAsA(_, write_pipe));
        copy(tma_b.with(barrier[write_pipe]), tBgB(_, k_tile), tBsB(_, write_pipe));
        ++write_state;
        ++k_tile;
      }

      warpgroup_wait<1>();
      ++read_state;
    }
    
    iterations_remaining -= 4;
  }

  while (iterations_remaining > 0) {
    int pipe_idx = read_state.index();
    
    BarrierType::wait(&barrier[pipe_idx], read_state.phase());
    
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,pipe_idx), tCrB(_,_,_,pipe_idx), tCrC);
    warpgroup_commit_batch();

    if (is_producer && k_tile < k_tile_count) {
      int write_pipe = write_state.index();
      BarrierType::arrive_and_expect_tx(&barrier[write_pipe], tma_transaction_bytes);
      copy(tma_a.with(barrier[write_pipe]), tAgA(_, k_tile), tAsA(_, write_pipe));
      copy(tma_b.with(barrier[write_pipe]), tBgB(_, k_tile), tBsB(_, write_pipe));
      ++write_state;
      ++k_tile;
    }

    warpgroup_wait<0>();
    ++read_state;
    --iterations_remaining;
  }

  warpgroup_wait<0>();

  if (beta == half_t(0.0f)) {
    #pragma unroll
    for (int i = 0; i < size(tCrC); ++i) {
      tCrC(i) = alpha * tCrC(i);
    }
    
    copy(tCrC, tCgC);
  } else {
    Tensor tCrC_old = make_fragment_like(tCrC);
    copy(tCgC, tCrC_old);
    
    #pragma unroll
    for (int i = 0; i < size(tCrC); ++i) {
      tCrC(i) = alpha * tCrC(i) + beta * tCrC_old(i);
    }
    
    copy(tCrC, tCgC);
  }
}

template <class TA, class TB, class TC>
void launch_async_epilogue_hgemm(
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
  auto bP = Int<6>{};

  auto sA = tile_to_shape(
      GMMA::Layout_K_SW128_Atom<TA>{},
      make_shape(bM, bK, bP));
  auto sB = tile_to_shape(
      GMMA::Layout_K_SW128_Atom<TB>{},
      make_shape(bN, bK, bP));

  TiledMMA tiled_mma = make_tiled_mma(
      SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);

  auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
  auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK));

  using SharedStorage = MinimalSharedStorage<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));

  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2, 2, 1);
  dim3 dimGrid(
      round_up(size(ceil_div(N, bN)), dimCluster.x),
      round_up(size(ceil_div(M, bM)), dimCluster.y));

  cutlass::ClusterLaunchParams params{dimGrid, dimBlock, dimCluster, smem_size};

  auto kernel = gemm_async_epilogue_kernel<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tma_a),
      TB, decltype(sB), decltype(tma_b),
      TC, decltype(dC), decltype(tiled_mma),
      half_t, half_t>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

  half_t alpha = half_t(1.0f);
  half_t beta = half_t(0.0f);

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params,
      reinterpret_cast<void const*>(kernel),
      prob_shape, cta_tiler,
      A, tma_a,
      B, tma_b,
      C, dC, tiled_mma,
      alpha, beta);

  CUTE_CHECK_LAST();
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
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

  launch_async_epilogue_hgemm<half_t, half_t, half_t>(
      reinterpret_cast<half_t const*>(a.data_ptr()),
      K,
      reinterpret_cast<half_t const*>(b_col_major.data_ptr()),
      K,
      reinterpret_cast<half_t*>(c.data_ptr()),
      N,
      M, N, K);
}