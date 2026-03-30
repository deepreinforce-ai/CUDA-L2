#include <iostream>
#include <stdexcept>
#include <string>
#include <mutex>
#include <vector>
#include <limits>
#include <cstdint>
#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
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

#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

using ElementA       = cutlass::half_t;
using ElementB       = cutlass::half_t;
using ElementC       = cutlass::half_t;
using ElementD       = cutlass::half_t;
using ElementAccum   = float;
using ElementCompute = float;
using LayoutA        = cutlass::layout::RowMajor;
using LayoutB        = cutlass::layout::ColumnMajor;
using LayoutC        = cutlass::layout::RowMajor;
using LayoutD        = cutlass::layout::RowMajor;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#define DEF_V(Name, TM, TN, TK, CM, CN, CK, EpiS, MlpS, TileS)              \
struct Name {                                                                   \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;     \
  using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder<  \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      TileShape, GridShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                        \
      ElementAccum, ElementCompute,                                             \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      EpiS, EpilogueOp>::CollectiveOp;                                          \
  using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<    \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                    \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccum,                                                              \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
          static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,         \
      MlpS>::CollectiveOp;                                                     \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, MainStage, EpiStage, TileS>;                  \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEF_V(V_SK64_128x64_1x1,   128,  64, 64, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x64_1x2,   128,  64, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x64_1x4,   128,  64, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x64_1x8,   128,  64, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x128_1x1,  128, 128, 64, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x128_1x2,  128, 128, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x128_1x4,  128, 128, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x128_1x8,  128, 128, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x256_1x1,  128, 256, 64, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x256_1x2,  128, 256, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x256_1x4,  128, 256, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK64_128x256_1x8,  128, 256, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_V(V_SK32_128x64_1x1,   128,  64, 32, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x64_1x2,   128,  64, 32, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x64_1x4,   128,  64, 32, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x64_1x8,   128,  64, 32, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x128_1x1,  128, 128, 32, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x128_1x2,  128, 128, 32, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x128_1x4,  128, 128, 32, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x128_1x8,  128, 128, 32, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x256_1x2,  128, 256, 32, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x256_1x4,  128, 256, 32, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK32_128x256_1x8,  128, 256, 32, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_V(V_SK128_128x64_1x1,  128,  64, 128, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x64_1x2,  128,  64, 128, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x64_1x4,  128,  64, 128, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x64_1x8,  128,  64, 128, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x128_1x2, 128, 128, 128, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x128_1x4, 128, 128, 128, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x128_1x8, 128, 128, 128, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x256_1x2, 128, 256, 128, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x256_1x4, 128, 256, 128, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)
DEF_V(V_SK128_128x256_1x8, 128, 256, 128, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_V(V_PP64_128x64_1x1,   128,  64, 64, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x64_1x2,   128,  64, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x64_1x4,   128,  64, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x64_1x8,   128,  64, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x128_1x1,  128, 128, 64, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x128_1x2,  128, 128, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x128_1x4,  128, 128, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x128_1x8,  128, 128, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x256_1x1,  128, 256, 64, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x256_1x2,  128, 256, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x256_1x4,  128, 256, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP64_128x256_1x8,  128, 256, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)

DEF_V(V_PP128_128x64_1x1,  128,  64, 128, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x64_1x2,  128,  64, 128, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x64_1x4,  128,  64, 128, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x64_1x8,  128,  64, 128, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x128_1x1, 128, 128, 128, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x128_1x2, 128, 128, 128, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x128_1x4, 128, 128, 128, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x128_1x8, 128, 128, 128, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x256_1x1, 128, 256, 128, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x256_1x2, 128, 256, 128, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x256_1x4, 128, 256, 128, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP128_128x256_1x8, 128, 256, 128, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)

DEF_V(V_PP32_128x64_1x1,   128,  64, 32, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x64_1x2,   128,  64, 32, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x64_1x4,   128,  64, 32, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x64_1x8,   128,  64, 32, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x128_1x1,  128, 128, 32, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x128_1x2,  128, 128, 32, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x128_1x4,  128, 128, 32, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x128_1x8,  128, 128, 32, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x256_1x2,  128, 256, 32, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x256_1x4,  128, 256, 32, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_PP32_128x256_1x8,  128, 256, 32, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::gemm::PersistentScheduler)

DEF_V(V_P64_128x64_1x1,    128,  64, 64, 1, 1, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x64_1x2,    128,  64, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x64_1x4,    128,  64, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x64_1x8,    128,  64, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x128_1x2,   128, 128, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x128_1x4,   128, 128, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x128_1x8,   128, 128, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x256_1x2,   128, 256, 64, 1, 2, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x256_1x4,   128, 256, 64, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P64_128x256_1x8,   128, 256, 64, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P128_128x256_1x4,  128, 256, 128, 1, 4, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)
DEF_V(V_P128_128x256_1x8,  128, 256, 128, 1, 8, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

struct IRunner {
  virtual ~IRunner() = default;
  virtual bool init(const ElementA* A, const ElementB* B,
                    ElementC* C, ElementD* D,
                    int M, int N, int K,
                    const cutlass::KernelHardwareInfo& hw,
                    cudaStream_t stream) = 0;
  virtual bool run(const ElementA* A, const ElementB* B,
                   ElementC* C, ElementD* D,
                   int M, int N, int K,
                   const cutlass::KernelHardwareInfo& hw,
                   cudaStream_t stream) = 0;
  virtual const char* name() const = 0;

  float benchmark(const ElementA* A, const ElementB* B,
                  ElementC* C, ElementD* D,
                  int M, int N, int K,
                  const cutlass::KernelHardwareInfo& hw,
                  cudaStream_t stream,
                  int warmup = 5, int iters = 50)
  {
    for (int i = 0; i < warmup; ++i)
      if (!run(A, B, C, D, M, N, K, hw, stream)) return -1.0f;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -1.0f;

    cudaEvent_t t0, t1;
    if (cudaEventCreate(&t0) != cudaSuccess) return -1.0f;
    if (cudaEventCreate(&t1) != cudaSuccess) { cudaEventDestroy(t0); return -1.0f; }

    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; ++i) {
      if (!run(A, B, C, D, M, N, K, hw, stream)) {
        cudaEventDestroy(t0); cudaEventDestroy(t1); return -1.0f;
      }
    }
    cudaEventRecord(t1, stream);
    if (cudaEventSynchronize(t1) != cudaSuccess) {
      cudaEventDestroy(t0); cudaEventDestroy(t1); return -1.0f;
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return (ms > 0.0f) ? ms / iters : -1.0f;
  }
};

template<typename V>
struct TypedRunner : IRunner {
  using Gemm    = typename V::Gemm;
  using StrideA = typename V::StrideA;
  using StrideB = typename V::StrideB;
  using StrideC = typename V::StrideC;
  using StrideD = typename V::StrideD;

  Gemm      gemm_;
  uint8_t*  ws_    = nullptr;
  size_t    ws_sz_ = 0;
  StrideA   sa_;
  StrideB   sb_;
  StrideC   sc_;
  StrideD   sd_;
  const char* nm_;

  TypedRunner(const char* nm) : nm_(nm) {}

  ~TypedRunner() override {
    if (ws_) { cudaFree(ws_); ws_ = nullptr; }
  }

  const char* name() const override { return nm_; }

  inline typename Gemm::Arguments make_args(
      const ElementA* A, const ElementB* B,
      ElementC* C, ElementD* D,
      int M, int N, int K,
      const cutlass::KernelHardwareInfo& hw) const
  {
    return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {const_cast<ElementA*>(A), sa_, const_cast<ElementB*>(B), sb_},
      {{1.0f, 0.0f}, C, sc_, D, sd_},
      hw
    };
  }

  bool init(const ElementA* A, const ElementB* B,
            ElementC* C, ElementD* D,
            int M, int N, int K,
            const cutlass::KernelHardwareInfo& hw,
            cudaStream_t stream) override
  {
    sa_ = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    sb_ = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    sc_ = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    sd_ = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto args = make_args(A, B, C, D, M, N, K, hw);
    if (gemm_.can_implement(args) != cutlass::Status::kSuccess) return false;

    size_t needed = Gemm::get_workspace_size(args);
    size_t alloc = std::max(needed * 2 + size_t(16) * 1024 * 1024,
                            size_t(32) * 1024 * 1024);
    if (cudaMalloc(&ws_, alloc) != cudaSuccess) return false;
    ws_sz_ = alloc;

    if (needed > 0) {
      cudaMemsetAsync(ws_, 0, needed, stream);
    }

    if (gemm_.initialize(args, ws_, stream) != cutlass::Status::kSuccess) return false;
    if (gemm_.run(stream) != cutlass::Status::kSuccess) return false;
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
    return true;
  }

  bool run(const ElementA* A, const ElementB* B,
           ElementC* C, ElementD* D,
           int M, int N, int K,
           const cutlass::KernelHardwareInfo& hw,
           cudaStream_t stream) override
  {
    auto args = make_args(A, B, C, D, M, N, K, hw);
    if (gemm_.initialize(args, ws_, stream) != cutlass::Status::kSuccess) return false;
    if (gemm_.run(stream) != cutlass::Status::kSuccess) return false;
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
  }
};

static IRunner*   g_runner     = nullptr;
static bool       g_ready      = false;
static std::mutex g_mutex;

static cutlass::KernelHardwareInfo g_hw;
static bool g_hw_ready = false;

static cudaStream_t g_stream       = nullptr;
static bool         g_stream_ready = false;

static void ensure_hw() {
  if (!g_hw_ready) {
    cudaGetDevice(&g_hw.device_id);
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_hw.device_id);
    g_hw_ready = true;
  }
}

static void ensure_stream() {
  if (!g_stream_ready) {
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, hi);
    g_stream_ready = true;
  }
}

template<typename V>
static IRunner* try_build(const char* nm,
    const ElementA* A, const ElementB* B,
    ElementC* C, ElementD* D, int M, int N, int K)
{
  auto* r = new TypedRunner<V>(nm);
  if (r->init(A, B, C, D, M, N, K, g_hw, g_stream)) return r;
  delete r;
  return nullptr;
}

static void auto_tune(const ElementA* A, const ElementB* B,
                      ElementC* C, ElementD* D, int M, int N, int K)
{
  ensure_hw();
  ensure_stream();

  std::vector<IRunner*> cands;
  auto try_add = [&](const char* nm, IRunner* r) {
    if (r) cands.push_back(r);
  };

  try_add("SK64_128x64_1x1",   try_build<V_SK64_128x64_1x1>   ("SK64_128x64_1x1",   A,B,C,D,M,N,K));
  try_add("SK64_128x64_1x2",   try_build<V_SK64_128x64_1x2>   ("SK64_128x64_1x2",   A,B,C,D,M,N,K));
  try_add("SK64_128x64_1x4",   try_build<V_SK64_128x64_1x4>   ("SK64_128x64_1x4",   A,B,C,D,M,N,K));
  try_add("SK64_128x64_1x8",   try_build<V_SK64_128x64_1x8>   ("SK64_128x64_1x8",   A,B,C,D,M,N,K));
  try_add("SK64_128x128_1x1",  try_build<V_SK64_128x128_1x1>  ("SK64_128x128_1x1",  A,B,C,D,M,N,K));
  try_add("SK64_128x128_1x2",  try_build<V_SK64_128x128_1x2>  ("SK64_128x128_1x2",  A,B,C,D,M,N,K));
  try_add("SK64_128x128_1x4",  try_build<V_SK64_128x128_1x4>  ("SK64_128x128_1x4",  A,B,C,D,M,N,K));
  try_add("SK64_128x128_1x8",  try_build<V_SK64_128x128_1x8>  ("SK64_128x128_1x8",  A,B,C,D,M,N,K));
  try_add("SK64_128x256_1x1",  try_build<V_SK64_128x256_1x1>  ("SK64_128x256_1x1",  A,B,C,D,M,N,K));
  try_add("SK64_128x256_1x2",  try_build<V_SK64_128x256_1x2>  ("SK64_128x256_1x2",  A,B,C,D,M,N,K));
  try_add("SK64_128x256_1x4",  try_build<V_SK64_128x256_1x4>  ("SK64_128x256_1x4",  A,B,C,D,M,N,K));
  try_add("SK64_128x256_1x8",  try_build<V_SK64_128x256_1x8>  ("SK64_128x256_1x8",  A,B,C,D,M,N,K));

  try_add("SK32_128x64_1x1",   try_build<V_SK32_128x64_1x1>   ("SK32_128x64_1x1",   A,B,C,D,M,N,K));
  try_add("SK32_128x64_1x2",   try_build<V_SK32_128x64_1x2>   ("SK32_128x64_1x2",   A,B,C,D,M,N,K));
  try_add("SK32_128x64_1x4",   try_build<V_SK32_128x64_1x4>   ("SK32_128x64_1x4",   A,B,C,D,M,N,K));
  try_add("SK32_128x64_1x8",   try_build<V_SK32_128x64_1x8>   ("SK32_128x64_1x8",   A,B,C,D,M,N,K));
  try_add("SK32_128x128_1x1",  try_build<V_SK32_128x128_1x1>  ("SK32_128x128_1x1",  A,B,C,D,M,N,K));
  try_add("SK32_128x128_1x2",  try_build<V_SK32_128x128_1x2>  ("SK32_128x128_1x2",  A,B,C,D,M,N,K));
  try_add("SK32_128x128_1x4",  try_build<V_SK32_128x128_1x4>  ("SK32_128x128_1x4",  A,B,C,D,M,N,K));
  try_add("SK32_128x128_1x8",  try_build<V_SK32_128x128_1x8>  ("SK32_128x128_1x8",  A,B,C,D,M,N,K));
  try_add("SK32_128x256_1x2",  try_build<V_SK32_128x256_1x2>  ("SK32_128x256_1x2",  A,B,C,D,M,N,K));
  try_add("SK32_128x256_1x4",  try_build<V_SK32_128x256_1x4>  ("SK32_128x256_1x4",  A,B,C,D,M,N,K));
  try_add("SK32_128x256_1x8",  try_build<V_SK32_128x256_1x8>  ("SK32_128x256_1x8",  A,B,C,D,M,N,K));

  try_add("SK128_128x64_1x1",  try_build<V_SK128_128x64_1x1>  ("SK128_128x64_1x1",  A,B,C,D,M,N,K));
  try_add("SK128_128x64_1x2",  try_build<V_SK128_128x64_1x2>  ("SK128_128x64_1x2",  A,B,C,D,M,N,K));
  try_add("SK128_128x64_1x4",  try_build<V_SK128_128x64_1x4>  ("SK128_128x64_1x4",  A,B,C,D,M,N,K));
  try_add("SK128_128x64_1x8",  try_build<V_SK128_128x64_1x8>  ("SK128_128x64_1x8",  A,B,C,D,M,N,K));
  try_add("SK128_128x128_1x2", try_build<V_SK128_128x128_1x2> ("SK128_128x128_1x2", A,B,C,D,M,N,K));
  try_add("SK128_128x128_1x4", try_build<V_SK128_128x128_1x4> ("SK128_128x128_1x4", A,B,C,D,M,N,K));
  try_add("SK128_128x128_1x8", try_build<V_SK128_128x128_1x8> ("SK128_128x128_1x8", A,B,C,D,M,N,K));
  try_add("SK128_128x256_1x2", try_build<V_SK128_128x256_1x2> ("SK128_128x256_1x2", A,B,C,D,M,N,K));
  try_add("SK128_128x256_1x4", try_build<V_SK128_128x256_1x4> ("SK128_128x256_1x4", A,B,C,D,M,N,K));
  try_add("SK128_128x256_1x8", try_build<V_SK128_128x256_1x8> ("SK128_128x256_1x8", A,B,C,D,M,N,K));

  try_add("PP64_128x64_1x1",   try_build<V_PP64_128x64_1x1>   ("PP64_128x64_1x1",   A,B,C,D,M,N,K));
  try_add("PP64_128x64_1x2",   try_build<V_PP64_128x64_1x2>   ("PP64_128x64_1x2",   A,B,C,D,M,N,K));
  try_add("PP64_128x64_1x4",   try_build<V_PP64_128x64_1x4>   ("PP64_128x64_1x4",   A,B,C,D,M,N,K));
  try_add("PP64_128x64_1x8",   try_build<V_PP64_128x64_1x8>   ("PP64_128x64_1x8",   A,B,C,D,M,N,K));
  try_add("PP64_128x128_1x1",  try_build<V_PP64_128x128_1x1>  ("PP64_128x128_1x1",  A,B,C,D,M,N,K));
  try_add("PP64_128x128_1x2",  try_build<V_PP64_128x128_1x2>  ("PP64_128x128_1x2",  A,B,C,D,M,N,K));
  try_add("PP64_128x128_1x4",  try_build<V_PP64_128x128_1x4>  ("PP64_128x128_1x4",  A,B,C,D,M,N,K));
  try_add("PP64_128x128_1x8",  try_build<V_PP64_128x128_1x8>  ("PP64_128x128_1x8",  A,B,C,D,M,N,K));
  try_add("PP64_128x256_1x1",  try_build<V_PP64_128x256_1x1>  ("PP64_128x256_1x1",  A,B,C,D,M,N,K));
  try_add("PP64_128x256_1x2",  try_build<V_PP64_128x256_1x2>  ("PP64_128x256_1x2",  A,B,C,D,M,N,K));
  try_add("PP64_128x256_1x4",  try_build<V_PP64_128x256_1x4>  ("PP64_128x256_1x4",  A,B,C,D,M,N,K));
  try_add("PP64_128x256_1x8",  try_build<V_PP64_128x256_1x8>  ("PP64_128x256_1x8",  A,B,C,D,M,N,K));

  try_add("PP128_128x64_1x1",  try_build<V_PP128_128x64_1x1>  ("PP128_128x64_1x1",  A,B,C,D,M,N,K));
  try_add("PP128_128x64_1x2",  try_build<V_PP128_128x64_1x2>  ("PP128_128x64_1x2",  A,B,C,D,M,N,K));
  try_add("PP128_128x64_1x4",  try_build<V_PP128_128x64_1x4>  ("PP128_128x64_1x4",  A,B,C,D,M,N,K));
  try_add("PP128_128x64_1x8",  try_build<V_PP128_128x64_1x8>  ("PP128_128x64_1x8",  A,B,C,D,M,N,K));
  try_add("PP128_128x128_1x1", try_build<V_PP128_128x128_1x1> ("PP128_128x128_1x1", A,B,C,D,M,N,K));
  try_add("PP128_128x128_1x2", try_build<V_PP128_128x128_1x2> ("PP128_128x128_1x2", A,B,C,D,M,N,K));
  try_add("PP128_128x128_1x4", try_build<V_PP128_128x128_1x4> ("PP128_128x128_1x4", A,B,C,D,M,N,K));
  try_add("PP128_128x128_1x8", try_build<V_PP128_128x128_1x8> ("PP128_128x128_1x8", A,B,C,D,M,N,K));
  try_add("PP128_128x256_1x1", try_build<V_PP128_128x256_1x1> ("PP128_128x256_1x1", A,B,C,D,M,N,K));
  try_add("PP128_128x256_1x2", try_build<V_PP128_128x256_1x2> ("PP128_128x256_1x2", A,B,C,D,M,N,K));
  try_add("PP128_128x256_1x4", try_build<V_PP128_128x256_1x4> ("PP128_128x256_1x4", A,B,C,D,M,N,K));
  try_add("PP128_128x256_1x8", try_build<V_PP128_128x256_1x8> ("PP128_128x256_1x8", A,B,C,D,M,N,K));

  try_add("PP32_128x64_1x1",   try_build<V_PP32_128x64_1x1>   ("PP32_128x64_1x1",   A,B,C,D,M,N,K));
  try_add("PP32_128x64_1x2",   try_build<V_PP32_128x64_1x2>   ("PP32_128x64_1x2",   A,B,C,D,M,N,K));
  try_add("PP32_128x64_1x4",   try_build<V_PP32_128x64_1x4>   ("PP32_128x64_1x4",   A,B,C,D,M,N,K));
  try_add("PP32_128x64_1x8",   try_build<V_PP32_128x64_1x8>   ("PP32_128x64_1x8",   A,B,C,D,M,N,K));
  try_add("PP32_128x128_1x1",  try_build<V_PP32_128x128_1x1>  ("PP32_128x128_1x1",  A,B,C,D,M,N,K));
  try_add("PP32_128x128_1x2",  try_build<V_PP32_128x128_1x2>  ("PP32_128x128_1x2",  A,B,C,D,M,N,K));
  try_add("PP32_128x128_1x4",  try_build<V_PP32_128x128_1x4>  ("PP32_128x128_1x4",  A,B,C,D,M,N,K));
  try_add("PP32_128x128_1x8",  try_build<V_PP32_128x128_1x8>  ("PP32_128x128_1x8",  A,B,C,D,M,N,K));
  try_add("PP32_128x256_1x2",  try_build<V_PP32_128x256_1x2>  ("PP32_128x256_1x2",  A,B,C,D,M,N,K));
  try_add("PP32_128x256_1x4",  try_build<V_PP32_128x256_1x4>  ("PP32_128x256_1x4",  A,B,C,D,M,N,K));
  try_add("PP32_128x256_1x8",  try_build<V_PP32_128x256_1x8>  ("PP32_128x256_1x8",  A,B,C,D,M,N,K));

  try_add("P64_128x64_1x1",    try_build<V_P64_128x64_1x1>    ("P64_128x64_1x1",    A,B,C,D,M,N,K));
  try_add("P64_128x64_1x2",    try_build<V_P64_128x64_1x2>    ("P64_128x64_1x2",    A,B,C,D,M,N,K));
  try_add("P64_128x64_1x4",    try_build<V_P64_128x64_1x4>    ("P64_128x64_1x4",    A,B,C,D,M,N,K));
  try_add("P64_128x64_1x8",    try_build<V_P64_128x64_1x8>    ("P64_128x64_1x8",    A,B,C,D,M,N,K));
  try_add("P64_128x128_1x2",   try_build<V_P64_128x128_1x2>   ("P64_128x128_1x2",   A,B,C,D,M,N,K));
  try_add("P64_128x128_1x4",   try_build<V_P64_128x128_1x4>   ("P64_128x128_1x4",   A,B,C,D,M,N,K));
  try_add("P64_128x128_1x8",   try_build<V_P64_128x128_1x8>   ("P64_128x128_1x8",   A,B,C,D,M,N,K));
  try_add("P64_128x256_1x2",   try_build<V_P64_128x256_1x2>   ("P64_128x256_1x2",   A,B,C,D,M,N,K));
  try_add("P64_128x256_1x4",   try_build<V_P64_128x256_1x4>   ("P64_128x256_1x4",   A,B,C,D,M,N,K));
  try_add("P64_128x256_1x8",   try_build<V_P64_128x256_1x8>   ("P64_128x256_1x8",   A,B,C,D,M,N,K));
  try_add("P128_128x256_1x4",  try_build<V_P128_128x256_1x4>  ("P128_128x256_1x4",  A,B,C,D,M,N,K));
  try_add("P128_128x256_1x8",  try_build<V_P128_128x256_1x8>  ("P128_128x256_1x8",  A,B,C,D,M,N,K));

  if (cands.empty())
    throw std::runtime_error("No CUTLASS SM90 variant initialized successfully");

  float    best_ms  = std::numeric_limits<float>::max();
  IRunner* best     = nullptr;
  int      best_idx = -1;

  for (int i = 0; i < (int)cands.size(); ++i) {
    float ms = cands[i]->benchmark(A, B, C, D, M, N, K, g_hw, g_stream, 5, 50);
    if (ms > 0.0f && ms < best_ms) {
      best_ms  = ms;
      best     = cands[i];
      best_idx = i;
    }
  }

  if (!best)
    throw std::runtime_error("All variants failed benchmarking");

  for (int i = 0; i < (int)cands.size(); ++i)
    if (i != best_idx) delete cands[i];

  g_runner = best;
  g_ready  = true;

  g_runner->run(A, B, C, D, M, N, K, g_hw, g_stream);
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  CHECK_TORCH_TENSOR_SHAPE(a,           M, K)
  CHECK_TORCH_TENSOR_SHAPE(b,           K, N)
  CHECK_TORCH_TENSOR_SHAPE(b_col_major, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c,           M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  const auto* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
  const auto* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  auto*       ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
  auto*       ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());

  if (__builtin_expect(g_ready, 1)) {
    if (__builtin_expect(
            g_runner->run(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, g_hw, g_stream), 1)) {
      return;
    }
    g_ready = false;
    delete g_runner;
    g_runner = nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ready) {
      auto_tune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K);
      return;
    }
    if (!g_runner->run(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, g_hw, g_stream))
      throw std::runtime_error("GEMM run failed after concurrent auto-tuning");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}