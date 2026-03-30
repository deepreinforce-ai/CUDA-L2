#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>

using namespace cute;

template<
  int kM = 128,
  int kN = 256, 
  int kK = 64,
  int kStages = 4
>
__global__ void __launch_bounds__(256)
hgemm_cute_manual_kernel(
  half const* A, int ldA,
  half const* B, int ldB,
  half* C, int ldC,
  int M, int N, int K)
{
  using namespace cute;
  
  int tile_m = blockIdx.x;
  int tile_n = blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  
  __shared__ half smem_A[kStages][kM][kK];
  __shared__ half smem_B[kStages][kN][kK];
  
  float accum[32] = {0.0f};
  
  half const* gA = A + tile_m * kM * ldA;
  half const* gB = B + tile_n * kN;
  half* gC = C + tile_m * kM * ldC + tile_n * kN;
  
  for (int stage = 0; stage < kStages - 1; ++stage) {
    int k_offset = stage * kK;
    if (k_offset < K) {
      for (int i = tid; i < kM * kK; i += 256) {
        int m_local = i / kK;
        int k_local = i % kK;
        int k_global = k_offset + k_local;
        if (k_global < K) {
          smem_A[stage][m_local][k_local] = gA[m_local * ldA + k_global];
        }
      }
      
      for (int i = tid; i < kN * kK; i += 256) {
        int n_local = i / kK;
        int k_local = i % kK;
        int k_global = k_offset + k_local;
        if (k_global < K) {
          smem_B[stage][n_local][k_local] = gB[k_global * ldB + n_local];
        }
      }
    }
  }
  __syncthreads();
  
  int num_k_tiles = (K + kK - 1) / kK;
  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    int read_stage = k_tile % kStages;
    int write_stage = (k_tile + kStages - 1) % kStages;
    
    if (k_tile + kStages - 1 < num_k_tiles) {
      int k_offset = (k_tile + kStages - 1) * kK;
      
      for (int i = tid; i < kM * kK; i += 256) {
        int m_local = i / kK;
        int k_local = i % kK;
        int k_global = k_offset + k_local;
        if (k_global < K) {
          smem_A[write_stage][m_local][k_local] = gA[m_local * ldA + k_global];
        }
      }
      
      for (int i = tid; i < kN * kK; i += 256) {
        int n_local = i / kK;
        int k_local = i % kK;
        int k_global = k_offset + k_local;
        if (k_global < K) {
          smem_B[write_stage][n_local][k_local] = gB[k_global * ldB + n_local];
        }
      }
    }
    
    int warp_m = (warp_id / 2) * 64;
    int warp_n = (warp_id % 2) * 128;
    
    for (int mk = 0; mk < kK; mk += 16) {
      for (int mm = 0; mm < 64; mm += 16) {
        for (int nn = 0; nn < 64; nn += 8) {
          int m_idx = warp_m + mm;
          int n_idx = warp_n + nn;
          
          for (int k = 0; k < 16; ++k) {
            if (m_idx + lane_id / 4 < kM && n_idx + lane_id % 4 * 2 < kN && mk + k < kK) {
              half a_val = smem_A[read_stage][m_idx + lane_id / 4][mk + k];
              half b_val = smem_B[read_stage][n_idx + lane_id % 4 * 2][mk + k];
              accum[mm / 16 * 8 + nn / 8] += __half2float(a_val) * __half2float(b_val);
            }
          }
        }
      }
    }
    
    __syncthreads();
  }
  
  int warp_m = (warp_id / 2) * 64;
  int warp_n = (warp_id % 2) * 128;
  
  for (int i = 0; i < 32; ++i) {
    int m_offset = warp_m + (i / 8) * 16 + lane_id / 4;
    int n_offset = warp_n + (i % 8) * 8 + lane_id % 4 * 2;
    
    if (tile_m * kM + m_offset < M && tile_n * kN + n_offset < N) {
      gC[m_offset * ldC + n_offset] = __float2half(accum[i]);
    }
  }
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

struct Tier4StageHgemmKernel {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute, 
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      EpilogueOp
    >::CollectiveOp;

  static constexpr int StageCount = 4;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GroupShape,
      cutlass::gemm::collective::StageCount<StageCount>,
      MainloopScheduleType
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c) {
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm = typename Tier4StageHgemmKernel::Gemm;
  using StrideA = typename Tier4StageHgemmKernel::StrideA;
  using StrideB = typename Tier4StageHgemmKernel::StrideB;
  using StrideC = typename Tier4StageHgemmKernel::StrideC;
  using StrideD = typename Tier4StageHgemmKernel::StrideD;
  using ElementA = typename Tier4StageHgemmKernel::ElementA;
  using ElementB = typename Tier4StageHgemmKernel::ElementB;
  using ElementC = typename Tier4StageHgemmKernel::ElementC;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    hw_info
  };

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM cannot implement");
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM initialization failed");
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM execution failed");
  }

  cudaDeviceSynchronize();
#else
  throw std::runtime_error("SM90 not supported");
#endif
}