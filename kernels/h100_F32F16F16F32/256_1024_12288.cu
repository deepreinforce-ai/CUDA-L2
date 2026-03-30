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
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  
  uint64_t tma_barrier[7];
  uint64_t mma_barrier[7];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaLoadA,
          class TB, class SmemLayoutB, class TmaLoadB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(cute::size(TiledMma{}))::value)
void gemm_device_kernel(
    ProblemShape shape_MNK,
    CtaTiler cta_tiler,
    TA const* A_ptr,
    CUTLASS_GRID_CONSTANT TmaLoadA const tma_load_a,
    TB const* B_ptr,
    CUTLASS_GRID_CONSTANT TmaLoadB const tma_load_b,
    TC* C_ptr,
    CStride dC,
    TiledMma tiled_mma,
    Alpha alpha,
    Beta beta)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  extern __shared__ char smem_[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_);

  Tensor mA = tma_load_a.get_tma_tensor(make_shape(get<0>(shape_MNK), get<2>(shape_MNK)));
  Tensor mB = tma_load_b.get_tma_tensor(make_shape(get<1>(shape_MNK), get<2>(shape_MNK)));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(get<0>(shape_MNK), get<1>(shape_MNK)), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_b.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_load_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_load_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int K_PIPE_MAX = 7;
  constexpr int tma_transaction_bytes = sizeof(decltype(make_tensor_like(tensor<0>(tAsA))))
                                       + sizeof(decltype(make_tensor_like(tensor<0>(tBsB))));

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* tma_bar = smem.tma_barrier;
  uint64_t* mma_bar = smem.mma_barrier;

  using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarrier = cutlass::arch::ClusterBarrier;

  if (warp_idx == 0 && lane_predicate) {
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      ProducerBarrier::init(&tma_bar[i], 1);
      ConsumerBarrier::init(&mma_bar[i], size(tiled_mma));
    }
  }
  __syncthreads();
  cluster_sync();

  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  int k_tile_count = size<2>(gA);
  int k_tile = 0;

  for (int pipe = 0; pipe < K_PIPE_MAX && k_tile < k_tile_count; ++pipe, ++k_tile) {
    if (warp_idx == 0 && lane_predicate) {
      ProducerBarrier::arrive_and_expect_tx(&tma_bar[pipe], tma_transaction_bytes);
      copy(tma_load_a.with(tma_bar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_load_b.with(tma_bar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
  }

  auto read_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();

  int k_remaining = k_tile_count;
  
  CUTE_NO_UNROLL
  while (k_remaining > 0) {
    int read_pipe = read_state.index();
    
    ProducerBarrier::wait(&tma_bar[read_pipe], read_state.phase());

    warpgroup_arrive();
    gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    warpgroup_commit_batch();

    warpgroup_wait<1>();
    __syncthreads();

    ConsumerBarrier::arrive(&mma_bar[read_pipe]);

    ++read_state;
    --k_remaining;

    if (k_tile < k_tile_count) {
      int write_pipe = write_state.index();
      
      if (warp_idx == 0 && lane_predicate) {
        ConsumerBarrier::wait(&mma_bar[write_pipe], write_state.phase());
        ProducerBarrier::arrive_and_expect_tx(&tma_bar[write_pipe], tma_transaction_bytes);
        copy(tma_load_a.with(tma_bar[write_pipe]), tAgA(_,k_tile), tAsA(_,write_pipe));
        copy(tma_load_b.with(tma_bar[write_pipe]), tBgB(_,k_tile), tBsB(_,write_pipe));
      }

      ++write_state;
      ++k_tile;
    }
  }

  warpgroup_wait<0>();
  __syncthreads();

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC>
void launch_gemm_wgmma(
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

  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<7>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

  auto tiled_mma = make_tiled_mma(
      SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);

  auto tma_load_a = make_tma_copy(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK), Int<1>{});
  auto tma_load_b = make_tma_copy(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK), Int<1>{});

  using SharedStorage = SharedStorage<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));

  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(4, 2, 1);
  dim3 dimGrid(
      size(ceil_div(M, bM)),
      size(ceil_div(N, bN)),
      1
  );

  void const* kernel = (void const*)
      &gemm_device_kernel<decltype(prob_shape), decltype(cta_tiler),
                          TA, decltype(sA), decltype(tma_load_a),
                          TB, decltype(sB), decltype(tma_load_b),
                          TC, decltype(dC), decltype(tiled_mma),
                          half_t, half_t>;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cutlass::ClusterLaunchParams launch_params{dimGrid, dimBlock, dimCluster, smem_size};

  half_t alpha = half_t(1.0f);
  half_t beta = half_t(0.0f);

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      launch_params, kernel,
      prob_shape, cta_tiler,
      A, tma_load_a,
      B, tma_load_b,
      C, dC, tiled_mma,
      alpha, beta);

  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  launch_gemm_wgmma<half_t, half_t, half_t>(
      reinterpret_cast<half_t const*>(a.data_ptr()),
      K,
      reinterpret_cast<half_t const*>(b_col_major.data_ptr()),
      K,
      reinterpret_cast<half_t*>(c.data_ptr()),
      N,
      M, N, K
  );
}