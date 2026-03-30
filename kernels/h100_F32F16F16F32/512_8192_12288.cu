#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>

using namespace cute;

constexpr int kTileM = 128;
constexpr int kTileN = 256;
constexpr int kTileK = 64;
constexpr int kStages = 5;
constexpr int kGridBlockM = 2;
constexpr int kGridBlockN = 1;

constexpr int kNumWarps = 8;
constexpr int kWarpM = 2;
constexpr int kWarpN = 2;

constexpr int kSwizzle = 128;

template<int M, int K>
__device__ __forceinline__ int smem_offset_A(int m, int k, int stage) {
  int base = stage * (M * K);
  int offset = m * K + k;
  int swizzled = offset ^ ((offset / 8) & 0xF);
  return (base + swizzled);
}

template<int K, int N>
__device__ __forceinline__ int smem_offset_B(int k, int n, int stage) {
  int base = kStages * (kTileM * kTileK) + stage * (K * N);
  int offset = k * N + n;
  int swizzled = offset ^ ((offset / 8) & 0xF);
  return (base + swizzled);
}

__global__ void __launch_bounds__(256)
hgemm_kernel_hand_optimized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  extern __shared__ half smem[];
  
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  
  const int bm = (blockIdx.x / kGridBlockN) * kGridBlockM + (blockIdx.x % kGridBlockM);
  const int bn = blockIdx.y;
  
  const bool is_compute_warp = (warp_id < 4);
  
  const half* gA = A + bm * kTileM * K;
  const half* gB = B + bn * kTileN;
  half* gC = C + bm * kTileM * N + bn * kTileN;
  
  float acc[4][8] = {0.0f};
  
  if (!is_compute_warp) {
    int load_warp = warp_id - 4;
    
    #pragma unroll
    for (int stage = 0; stage < kStages - 1; ++stage) {
      int k_offset = stage * kTileK;
      
      if (load_warp < 2) {
        for (int i = 0; i < 4; ++i) {
          int m = load_warp * 64 + lane_id + i * 32;
          if (m < kTileM && k_offset < K) {
            for (int k = 0; k < kTileK; k += 8) {
              int smem_idx = smem_offset_A<kTileM, kTileK>(m, k, stage);
              float4* smem_ptr = (float4*)&smem[smem_idx];
              const float4* gmem_ptr = (const float4*)&gA[m * K + k_offset + k];
              *smem_ptr = *gmem_ptr;
            }
          }
        }
      }
      
      if (load_warp >= 2) {
        int b_warp = load_warp - 2;
        for (int i = 0; i < 4; ++i) {
          int k = b_warp * 32 + lane_id / 4 + i * 8;
          int n_base = (lane_id % 4) * 64;
          if (k < kTileK && k_offset + k < K) {
            for (int n = 0; n < 64; n += 8) {
              int smem_idx = smem_offset_B<kTileK, kTileN>(k, n_base + n, stage);
              float4* smem_ptr = (float4*)&smem[smem_idx];
              const float4* gmem_ptr = (const float4*)&gB[(k_offset + k) * N + n_base + n];
              *smem_ptr = *gmem_ptr;
            }
          }
        }
      }
    }
  }
  
  __syncthreads();
  
  int num_k_iters = (K + kTileK - 1) / kTileK;
  
  for (int k_iter = 0; k_iter < num_k_iters; ++k_iter) {
    int read_stage = k_iter % kStages;
    int write_stage = (k_iter + kStages - 1) % kStages;
    
    if (is_compute_warp) {
      half A_reg[4][4];
      half B_reg[8][4];
      
      int warp_m = (warp_id % 2) * 64;
      int warp_n = (warp_id / 2) * 128;
      
      for (int ki = 0; ki < kTileK; ki += 16) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int m = warp_m + (lane_id / 4) * 16 + i * 4;
          int k = ki + (lane_id % 4) * 4;
          int smem_idx = smem_offset_A<kTileM, kTileK>(m, k, read_stage);
          *((float2*)&A_reg[i][0]) = *((float2*)&smem[smem_idx]);
        }
        
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          int k = ki + (lane_id / 4) * 4;
          int n = warp_n + (lane_id % 4) * 32 + j * 4;
          int smem_idx = smem_offset_B<kTileK, kTileN>(k, n, read_stage);
          *((float2*)&B_reg[j][0]) = *((float2*)&smem[smem_idx]);
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          #pragma unroll
          for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
              acc[i][j] += __half2float(A_reg[i][kk]) * __half2float(B_reg[j][kk]);
            }
          }
        }
      }
    }
    
    if (!is_compute_warp && k_iter + kStages - 1 < num_k_iters) {
      int load_warp = warp_id - 4;
      int k_offset = (k_iter + kStages - 1) * kTileK;
      
      if (load_warp < 2) {
        for (int i = 0; i < 4; ++i) {
          int m = load_warp * 64 + lane_id + i * 32;
          if (m < kTileM && k_offset < K) {
            for (int k = 0; k < kTileK && k_offset + k < K; k += 8) {
              int smem_idx = smem_offset_A<kTileM, kTileK>(m, k, write_stage);
              float4* smem_ptr = (float4*)&smem[smem_idx];
              const float4* gmem_ptr = (const float4*)&gA[m * K + k_offset + k];
              *smem_ptr = *gmem_ptr;
            }
          }
        }
      }
      
      if (load_warp >= 2) {
        int b_warp = load_warp - 2;
        for (int i = 0; i < 4; ++i) {
          int k = b_warp * 32 + lane_id / 4 + i * 8;
          int n_base = (lane_id % 4) * 64;
          if (k < kTileK && k_offset + k < K) {
            for (int n = 0; n < 64; n += 8) {
              int smem_idx = smem_offset_B<kTileK, kTileN>(k, n_base + n, write_stage);
              float4* smem_ptr = (float4*)&smem[smem_idx];
              const float4* gmem_ptr = (const float4*)&gB[(k_offset + k) * N + n_base + n];
              *smem_ptr = *gmem_ptr;
            }
          }
        }
      }
    }
    
    __syncthreads();
  }
  
  if (is_compute_warp) {
    int warp_m = (warp_id % 2) * 64;
    int warp_n = (warp_id / 2) * 128;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        int m = warp_m + (lane_id / 4) * 16 + i * 4;
        int n = warp_n + (lane_id % 4) * 32 + j * 4;
        
        if (bm * kTileM + m < M && bn * kTileN + n < N) {
          gC[m * N + n] = __float2half(acc[i][j]);
        }
      }
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
#include "cutlass/util/device_memory.h"

struct HgemmFallback {
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
  using TileGroupShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      EpilogueOp
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, 
                                 torch::Tensor b_col_major, torch::Tensor c) {
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using Gemm = typename HgemmFallback::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;

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
    throw std::runtime_error("CUTLASS cannot implement");
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS init failed");
  }

  status = gemm.run();
  cudaDeviceSynchronize();
  
  if (status != cutlass::Status::kSuccess || cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("CUTLASS execution failed");
  }
#else
  throw std::runtime_error("SM90 not supported");
#endif
}