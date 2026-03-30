#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/packed_stride.hpp"

#include <torch/extension.h>
#include <torch/types.h>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct ConfigHighThroughput {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA          = cutlass::half_t;
  using ElementB          = cutlass::half_t;
  using ElementC          = cutlass::half_t;
  using ElementD          = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape  = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using GridShape  = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp
    >::CollectiveOp;

  static constexpr int PipelineStages = 3;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      cute::Int<PipelineStages>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct ConfigBalanced {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA          = cutlass::half_t;
  using ElementB          = cutlass::half_t;
  using ElementC          = cutlass::half_t;
  using ElementD          = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using TileShape  = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GridShape  = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GridShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOp
    >::CollectiveOp;

  static constexpr int PipelineStages = 6;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, GridShape,
      cute::Int<PipelineStages>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template<typename Config>
struct CachedLauncher {
  using Gemm     = typename Config::Gemm;
  using StrideA  = typename Config::StrideA;
  using StrideB  = typename Config::StrideB;
  using StrideC  = typename Config::StrideC;
  using StrideD  = typename Config::StrideD;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;

  Gemm   gemm;
  void*  workspace_ptr  = nullptr;
  size_t workspace_size = 0;

  const void* last_A = nullptr;
  const void* last_B = nullptr;
  const void* last_C = nullptr;
  int last_M = 0, last_N = 0, last_K = 0;
  bool initialized = false;

  cutlass::KernelHardwareInfo hw_info;
  bool hw_ready = false;

  ~CachedLauncher() {
    if (workspace_ptr) {
      cudaFree(workspace_ptr);
    }
  }

  void ensure_hw_info() {
    if (!hw_ready) {
      int dev = 0;
      cudaGetDevice(&dev);
      hw_info.device_id = dev;
      hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
      hw_ready = true;
    }
  }

  typename Gemm::Arguments build_args(const ElementA* pA, const ElementB* pB,
                                       const ElementC* pC, int M, int N, int K) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {const_cast<ElementA*>(pA), sA, const_cast<ElementB*>(pB), sB},
      {{1.0f, 0.0f},
       const_cast<ElementC*>(pC), sC,
       const_cast<ElementC*>(pC), sD},
      hw_info
    };
  }

  cutlass::Status launch(const ElementA* pA, const ElementB* pB,
                         const ElementC* pC, int M, int N, int K) {
    ensure_hw_info();

    bool is_hot_path = initialized &&
                       (pA == last_A) && (pB == last_B) && (pC == last_C) &&
                       (M == last_M) && (N == last_N) && (K == last_K);

    auto args = build_args(pA, pB, pC, M, N, K);

    if (!initialized) {
      size_t needed = Gemm::get_workspace_size(args);
      if (needed > 0) {
        cudaError_t err = cudaMalloc(&workspace_ptr, needed);
        if (err != cudaSuccess) {
          return cutlass::Status::kErrorInternal;
        }
        workspace_size = needed;
      }
      
      cutlass::Status s = gemm.can_implement(args);
      if (s != cutlass::Status::kSuccess) return s;
      
      s = gemm.initialize(args, workspace_ptr);
      if (s != cutlass::Status::kSuccess) return s;
      
      initialized = true;
      last_A = pA; last_B = pB; last_C = pC;
      last_M = M; last_N = N; last_K = K;
      
      return gemm.run();
    }

    if (is_hot_path) {
      cutlass::Status s = gemm.initialize(args, workspace_ptr);
      if (s != cutlass::Status::kSuccess) return s;
      return gemm.run();
    }

    cutlass::Status s = gemm.can_implement(args);
    if (s != cutlass::Status::kSuccess) return s;
    
    s = gemm.initialize(args, workspace_ptr);
    if (s != cutlass::Status::kSuccess) return s;
    
    last_A = pA; last_B = pB; last_C = pC;
    last_M = M; last_N = N; last_K = K;
    
    return gemm.run();
  }
};

static CachedLauncher<ConfigHighThroughput> g_launcher_highthroughput;
static CachedLauncher<ConfigBalanced>       g_launcher_balanced;

#endif

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
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;

  const ElementA* pA = reinterpret_cast<const ElementA*>(a.data_ptr());
  const ElementB* pB = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  const ElementC* pC = reinterpret_cast<const ElementC*>(c.data_ptr());

  cutlass::Status status = g_launcher_highthroughput.launch(pA, pB, pC, M, N, K);

  if (status != cutlass::Status::kSuccess) {
    status = g_launcher_balanced.launch(pA, pB, pC, M, N, K);
    
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM: all configurations failed");
    }
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}