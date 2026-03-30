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

using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage6Stage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  
  alignas(16) uint64_t barrier[6];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(128, 2)
void gemm_optimized_6stage_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                  TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                  TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                                  TC* C, CStride dC, TiledMma mma,
                                  Alpha alpha, Beta beta)
{
  using namespace cute;
  
  constexpr int kStages = 6;
  
  auto [M, N, K] = shape_MNK;
  
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);

  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage6Stage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_b.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int kTmaTransactionBytes = 
      sizeof(TA) * size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) +
      sizeof(TB) * size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{});

  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;
  
  bool is_producer_a = (warp_idx == 0) && (lane_idx == 0);
  bool is_producer_b = (warp_idx == 1) && (lane_idx == 0);

  uint64_t* barrier = smem.barrier;

  if (is_producer_a) {
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
      cutlass::arch::ClusterTransactionBarrier::init(&barrier[i], 1);
    }
  }
  __syncthreads();

  int k_tile_count = size<1>(tAgA);
  
  int initial_prefetch = min(7, k_tile_count);
  if (is_producer_a || is_producer_b) {
    #pragma unroll
    for (int prefetch = 0; prefetch < min(kStages, initial_prefetch); ++prefetch) {
      if (is_producer_a) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&barrier[prefetch], kTmaTransactionBytes);
        copy(tma_a.with(barrier[prefetch]), tAgA(_,prefetch), tAsA(_,prefetch));
      }
      if (is_producer_b) {
        copy(tma_b.with(barrier[prefetch]), tBgB(_,prefetch), tBsB(_,prefetch));
      }
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

  int smem_pipe_read = 0;
  int smem_pipe_write = kStages;
  int phase = 0;

  #pragma unroll 1
  for (int k_iter = 0; k_iter < k_tile_count; ++k_iter) {
    int read_stage = smem_pipe_read % kStages;
    int write_stage = smem_pipe_write % kStages;
    
    cutlass::arch::ClusterTransactionBarrier::wait(&barrier[read_stage], phase);
    
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), tCrC);
    warpgroup_commit_batch();
    
    int remaining = k_tile_count - k_iter;
    if (remaining > 11) {
      warpgroup_wait<5>();
    } else if (remaining > 9) {
      warpgroup_wait<4>();
    } else if (remaining > 7) {
      warpgroup_wait<3>();
    } else if (remaining > 5) {
      warpgroup_wait<2>();
    } else if (remaining > 2) {
      warpgroup_wait<1>();
    } else {
      warpgroup_wait<0>();
    }
    
    int k_tile_next = k_iter + kStages;
    if (k_tile_next < k_tile_count) {
      if (is_producer_a) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&barrier[write_stage], kTmaTransactionBytes);
        copy(tma_a.with(barrier[write_stage]), tAgA(_,k_tile_next), tAsA(_,write_stage));
      }
      if (is_producer_b) {
        copy(tma_b.with(barrier[write_stage]), tBgB(_,k_tile_next), tBsB(_,write_stage));
      }
    }
    
    ++smem_pipe_read;
    ++smem_pipe_write;
    if (smem_pipe_read >= kStages) {
      smem_pipe_read = 0;
      smem_pipe_write = kStages;
      phase ^= 1;
    }
  }

  warpgroup_wait<0>();
  
  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC>
void launch_optimized_6stage_hgemm(
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
  auto bN = Int< 64>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<6>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  auto tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  auto tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  using SharedStorage = SharedStorage6Stage<TA, TB, decltype(sA), decltype(sB)>;
  int smem_size = int(sizeof(SharedStorage));

  dim3 dimBlock(128);
  dim3 dimCluster(2, 2, 1);
  
  dim3 dimGrid(
    (N + size<1>(cta_tiler) - 1) / size<1>(cta_tiler),
    (M + size<0>(cta_tiler) - 1) / size<0>(cta_tiler)
  );
  
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  auto kernel_ptr = &gemm_optimized_6stage_kernel<
    decltype(prob_shape), decltype(cta_tiler),
    TA, decltype(sA), decltype(tmaA),
    TB, decltype(sB), decltype(tmaB),
    TC, decltype(dC), decltype(tiled_mma),
    half_t, half_t>;

  cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size
  );

  half_t alpha = half_t(1.0f);
  half_t beta  = half_t(0.0f);

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
    params, (void const*) kernel_ptr,
    prob_shape, cta_tiler,
    A, tmaA,
    B, tmaB,
    C, dC, tiled_mma,
    alpha, beta
  );

  cudaError_t result = cudaGetLastError();
  if (result != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(result));
  }
  
  if (status != cutlass::Status::kSuccess) {
    printf("Kernel launch failed\n");
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

  launch_optimized_6stage_hgemm<cute::half_t, cute::half_t, cute::half_t>(
      reinterpret_cast<cute::half_t const*>(a.data_ptr()),
      K,
      reinterpret_cast<cute::half_t const*>(b_col_major.data_ptr()),
      K,
      reinterpret_cast<cute::half_t*>(c.data_ptr()),
      N,
      M, N, K);
}