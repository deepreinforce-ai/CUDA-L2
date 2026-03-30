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
struct SharedStorageTMA
{
  alignas(256) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(256) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
  
  alignas(128) uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
  alignas(128) uint64_t mma_barrier[cute::size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(cute::size(TiledMma{}))::value, 1)
void gemm_wgmma_tma_ultimate_fusion_kernel(
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

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));

  auto [M, N, K] = shape_MNK;
  
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTMA<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  constexpr int tma_transaction_bytes = 
      sizeof(TA) * cute::size<0>(SmemLayoutA{}) * cute::size<1>(SmemLayoutA{}) +
      sizeof(TB) * cute::size<0>(SmemLayoutB{}) * cute::size<1>(SmemLayoutB{});

  auto K_PIPE_MAX = size<2>(sA);

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  
  uint64_t* __restrict__ producer_mbar = smem.tma_barrier;
  uint64_t* __restrict__ consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier;
  
  if ((warp_idx == 0) && lane_predicate) {
    ProducerBarType::init(&producer_mbar[0], 1);
    ConsumerBarType::init(&consumer_mbar[0], 128);
    ProducerBarType::init(&producer_mbar[1], 1);
    ConsumerBarType::init(&consumer_mbar[1], 128);
    ProducerBarType::init(&producer_mbar[2], 1);
    ConsumerBarType::init(&consumer_mbar[2], 128);
    ProducerBarType::init(&producer_mbar[3], 1);
    ConsumerBarType::init(&consumer_mbar[3], 128);
    ProducerBarType::init(&producer_mbar[4], 1);
    ConsumerBarType::init(&consumer_mbar[4], 128);
  }
  cluster_sync();

  int total_cta_m = (M + size<0>(cta_tiler) - 1) / size<0>(cta_tiler);
  int total_cta_n = (N + size<1>(cta_tiler) - 1) / size<1>(cta_tiler);
  int total_cta_tiles = total_cta_m * total_cta_n;
  
  int cta_per_grid = gridDim.x * gridDim.y;
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;

  for (int tile_idx = block_id; tile_idx < total_cta_tiles; tile_idx += cta_per_grid)
  {
    int cta_m = tile_idx / total_cta_n;
    int cta_n = tile_idx % total_cta_n;
    
    if (cta_m * size<0>(cta_tiler) >= M || cta_n * size<1>(cta_tiler) >= N) {
      continue;
    }

    auto cta_coord = make_coord(cta_m, cta_n, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sA), group_modes<0,2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sB), group_modes<0,2>(gB));

    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    int prefetch_count = min((int)K_PIPE_MAX, k_tile_count);
    
    if ((warp_idx == 0) && lane_predicate) {
      if (prefetch_count > 0) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[0], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[0]), tAgA(_,0), tAsA(_,0));
        copy(tma_b.with(producer_mbar[0]), tBgB(_,0), tBsB(_,0));
      }
      if (prefetch_count > 1) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[1], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[1]), tAgA(_,1), tAsA(_,1));
        copy(tma_b.with(producer_mbar[1]), tBgB(_,1), tBsB(_,1));
      }
      if (prefetch_count > 2) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[2], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[2]), tAgA(_,2), tAsA(_,2));
        copy(tma_b.with(producer_mbar[2]), tBgB(_,2), tBsB(_,2));
      }
      if (prefetch_count > 3) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[3], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[3]), tAgA(_,3), tAsA(_,3));
        copy(tma_b.with(producer_mbar[3]), tBgB(_,3), tBsB(_,3));
      }
      if (prefetch_count > 4) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[4], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[4]), tAgA(_,4), tAsA(_,4));
        copy(tma_b.with(producer_mbar[4]), tBgB(_,4), tBsB(_,4));
      }
    }
    k_tile = prefetch_count;

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    auto write_state = cutlass::PipelineState<decltype(K_PIPE_MAX)::value>();
    auto read_state  = cutlass::PipelineState<decltype(K_PIPE_MAX)::value>();

    int remaining_tiles = k_tile_count;

    CUTE_NO_UNROLL
    while (remaining_tiles > 0)
    {
      int read_pipe = read_state.index();
      
      ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
      warpgroup_commit_batch();
      
      int tiles_remaining_after = remaining_tiles - 1;
      if (tiles_remaining_after > 4) {
        warpgroup_wait<2>();
      } else if (tiles_remaining_after > 2) {
        warpgroup_wait<1>();
      } else {
        warpgroup_wait<0>();
      }

      ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
      ++read_state;
      --remaining_tiles;

      if ((warp_idx == 0) && lane_predicate && (k_tile < k_tile_count))
      {
        int pipe = write_state.index();
        
        ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
        
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
        
        ++k_tile;
        ++write_state;
      }
    }

    warpgroup_wait<0>();

    axpby(alpha, tCrC, beta, tCgC);
    
    __syncthreads();
  }
}

template <class TA, class TB, class TC>
void launch_hgemm_wgmma_tma_sm90(
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
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<5>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  TiledMMA tiled_mma = make_tiled_mma(
      SM90_64x128x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  using SharedStorage = SharedStorageTMA<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));

  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2, 1, 1);
  
  int grid_n = 16;
  int grid_m = 8;
  
  grid_n = round_up(grid_n, dimCluster.x);
  grid_m = round_up(grid_m, dimCluster.y);
  
  dim3 dimGrid(grid_n, grid_m);
  
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  auto kernel_ptr = &gemm_wgmma_tma_ultimate_fusion_kernel<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tmaA),
      TB, decltype(sB), decltype(tmaB),
      TC, decltype(dC), decltype(tiled_mma),
      cute::half_t, cute::half_t>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  cute::half_t alpha = cute::half_t(1.0f);
  cute::half_t beta  = cute::half_t(0.0f);

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

  launch_hgemm_wgmma_tma_sm90<cute::half_t, cute::half_t, cute::half_t>(
      reinterpret_cast<cute::half_t const*>(a.data_ptr()),
      K,
      reinterpret_cast<cute::half_t const*>(b_col_major.data_ptr()),
      K,
      reinterpret_cast<cute::half_t*>(c.data_ptr()),
      N,
      M, N, K);
}