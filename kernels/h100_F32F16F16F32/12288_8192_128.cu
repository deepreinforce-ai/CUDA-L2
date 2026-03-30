#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>

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
#include "cutlass/util/device_memory.h"

#include <torch/extension.h>
#include <torch/types.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace {
struct GState {
    int device_id = -1;
    int sm_count  = -1;
    uint8_t* ws   = nullptr;
    size_t ws_sz  = 0;

    void init() {
        if (sm_count >= 0) return;
        cudaGetDevice(&device_id);
        sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    }

    uint8_t* workspace(size_t need) {
        if (need > ws_sz) {
            if (ws) cudaFree(ws);
            cudaMalloc(&ws, need);
            ws_sz = need;
        }
        return ws;
    }
    ~GState() { if (ws) { cudaFree(ws); ws = nullptr; } }
};
static GState g_state;
}

#define DEF_PP(Name, TM, TN, TK, CM, CN, CK)                                   \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t;                                             \
  using ElementB = cutlass::half_t;                                             \
  using ElementC = cutlass::half_t;                                             \
  using ElementD = cutlass::half_t;                                             \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;     \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                       \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,      \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
}

#define DEF_WS(Name, TM, TN, TK, CM, CN, CK)                                   \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t;                                             \
  using ElementB = cutlass::half_t;                                             \
  using ElementC = cutlass::half_t;                                             \
  using ElementD = cutlass::half_t;                                             \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;     \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                       \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::NoSmemWarpSpecialized,                                \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;                  \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,      \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
}

#define DEF_COOP(Name, TM, TN, TK, CM, CN, CK)                                 \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t;                                             \
  using ElementB = cutlass::half_t;                                             \
  using ElementC = cutlass::half_t;                                             \
  using ElementD = cutlass::half_t;                                             \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;     \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                       \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,      \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
}

#define DEF_COOP_SK(Name, TM, TN, TK, CM, CN, CK)                              \
struct Name {                                                                    \
  using LayoutA = cutlass::layout::RowMajor;                                    \
  using LayoutB = cutlass::layout::ColumnMajor;                                 \
  using LayoutC = cutlass::layout::RowMajor;                                    \
  using LayoutD = cutlass::layout::RowMajor;                                    \
  using ElementA = cutlass::half_t;                                             \
  using ElementB = cutlass::half_t;                                             \
  using ElementC = cutlass::half_t;                                             \
  using ElementD = cutlass::half_t;                                             \
  using ElementAccumulator = float;                                             \
  using ElementCompute = float;                                                 \
  static constexpr int AlignmentA = 8;                                         \
  static constexpr int AlignmentB = 8;                                         \
  static constexpr int AlignmentC = 8;                                         \
  static constexpr int AlignmentD = 8;                                         \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;     \
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<             \
      ElementD, ElementCompute, ElementC, ElementCompute,                       \
      cutlass::FloatRoundStyle::round_to_nearest>;                              \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignmentC,                                           \
      ElementD, LayoutD, AlignmentD,                                           \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignmentA,                                           \
      ElementB, LayoutB, AlignmentB,                                           \
      ElementAccumulator,                                                       \
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue,      \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;        \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
}

DEF_PP(PP_128x128_128_4x1,  128, 128, 128, 4, 1, 1);
DEF_PP(PP_128x128_128_1x4,  128, 128, 128, 1, 4, 1);
DEF_PP(PP_128x128_128_2x2,  128, 128, 128, 2, 2, 1);
DEF_PP(PP_128x128_128_2x1,  128, 128, 128, 2, 1, 1);
DEF_PP(PP_128x128_128_1x2,  128, 128, 128, 1, 2, 1);
DEF_PP(PP_128x128_128_1x1,  128, 128, 128, 1, 1, 1);
DEF_PP(PP_128x256_128_2x1,  128, 256, 128, 2, 1, 1);
DEF_PP(PP_128x256_128_1x2,  128, 256, 128, 1, 2, 1);
DEF_PP(PP_128x256_128_2x2,  128, 256, 128, 2, 2, 1);
DEF_PP(PP_128x256_128_4x1,  128, 256, 128, 4, 1, 1);
DEF_PP(PP_128x256_128_1x1,  128, 256, 128, 1, 1, 1);
DEF_PP(PP_256x128_128_2x1,  256, 128, 128, 2, 1, 1);
DEF_PP(PP_256x128_128_1x2,  256, 128, 128, 1, 2, 1);
DEF_PP(PP_256x128_128_4x1,  256, 128, 128, 4, 1, 1);
DEF_PP(PP_256x128_128_1x1,  256, 128, 128, 1, 1, 1);
DEF_PP(PP_128x128_128_4x2,  128, 128, 128, 4, 2, 1);
DEF_PP(PP_128x128_128_2x4,  128, 128, 128, 2, 4, 1);

DEF_WS(WS_128x128_128_4x1,  128, 128, 128, 4, 1, 1);
DEF_WS(WS_128x128_128_1x4,  128, 128, 128, 1, 4, 1);
DEF_WS(WS_128x128_128_2x2,  128, 128, 128, 2, 2, 1);
DEF_WS(WS_128x128_128_2x1,  128, 128, 128, 2, 1, 1);
DEF_WS(WS_128x128_128_1x2,  128, 128, 128, 1, 2, 1);
DEF_WS(WS_128x128_128_1x1,  128, 128, 128, 1, 1, 1);
DEF_WS(WS_128x256_128_2x1,  128, 256, 128, 2, 1, 1);
DEF_WS(WS_128x256_128_1x2,  128, 256, 128, 1, 2, 1);
DEF_WS(WS_128x256_128_1x1,  128, 256, 128, 1, 1, 1);
DEF_WS(WS_256x128_128_2x1,  256, 128, 128, 2, 1, 1);
DEF_WS(WS_256x128_128_1x1,  256, 128, 128, 1, 1, 1);
DEF_WS(WS_128x128_128_4x2,  128, 128, 128, 4, 2, 1);
DEF_WS(WS_128x128_128_2x4,  128, 128, 128, 2, 4, 1);

DEF_COOP(COOP_128x128_128_4x1,  128, 128, 128, 4, 1, 1);
DEF_COOP(COOP_128x128_128_1x4,  128, 128, 128, 1, 4, 1);
DEF_COOP(COOP_128x128_128_2x2,  128, 128, 128, 2, 2, 1);
DEF_COOP(COOP_128x128_128_2x1,  128, 128, 128, 2, 1, 1);
DEF_COOP(COOP_128x128_128_1x2,  128, 128, 128, 1, 2, 1);
DEF_COOP(COOP_128x128_128_1x1,  128, 128, 128, 1, 1, 1);
DEF_COOP(COOP_128x256_128_2x1,  128, 256, 128, 2, 1, 1);
DEF_COOP(COOP_128x256_128_1x2,  128, 256, 128, 1, 2, 1);
DEF_COOP(COOP_128x256_128_1x1,  128, 256, 128, 1, 1, 1);
DEF_COOP(COOP_256x128_128_2x1,  256, 128, 128, 2, 1, 1);
DEF_COOP(COOP_256x128_128_1x1,  256, 128, 128, 1, 1, 1);
DEF_COOP(COOP_128x128_128_4x2,  128, 128, 128, 4, 2, 1);
DEF_COOP(COOP_128x128_128_2x4,  128, 128, 128, 2, 4, 1);

DEF_COOP_SK(COOP_SK_128x128_128_4x1,  128, 128, 128, 4, 1, 1);
DEF_COOP_SK(COOP_SK_128x128_128_1x4,  128, 128, 128, 1, 4, 1);
DEF_COOP_SK(COOP_SK_128x128_128_2x2,  128, 128, 128, 2, 2, 1);
DEF_COOP_SK(COOP_SK_128x128_128_2x1,  128, 128, 128, 2, 1, 1);
DEF_COOP_SK(COOP_SK_128x128_128_1x2,  128, 128, 128, 1, 2, 1);
DEF_COOP_SK(COOP_SK_128x128_128_1x1,  128, 128, 128, 1, 1, 1);
DEF_COOP_SK(COOP_SK_128x256_128_2x1,  128, 256, 128, 2, 1, 1);
DEF_COOP_SK(COOP_SK_128x256_128_1x2,  128, 256, 128, 1, 2, 1);
DEF_COOP_SK(COOP_SK_128x256_128_1x1,  128, 256, 128, 1, 1, 1);
DEF_COOP_SK(COOP_SK_256x128_128_2x1,  256, 128, 128, 2, 1, 1);
DEF_COOP_SK(COOP_SK_256x128_128_1x1,  256, 128, 128, 1, 1, 1);
DEF_COOP_SK(COOP_SK_128x128_128_4x2,  128, 128, 128, 4, 2, 1);
DEF_COOP_SK(COOP_SK_128x128_128_2x4,  128, 128, 128, 2, 4, 1);

DEF_PP(PP_128x128_64_4x1,  128, 128, 64, 4, 1, 1);
DEF_PP(PP_128x128_64_1x4,  128, 128, 64, 1, 4, 1);
DEF_PP(PP_128x128_64_2x2,  128, 128, 64, 2, 2, 1);
DEF_PP(PP_128x128_64_2x1,  128, 128, 64, 2, 1, 1);
DEF_PP(PP_128x128_64_1x2,  128, 128, 64, 1, 2, 1);
DEF_PP(PP_128x128_64_1x1,  128, 128, 64, 1, 1, 1);
DEF_PP(PP_128x256_64_2x1,  128, 256, 64, 2, 1, 1);
DEF_PP(PP_128x256_64_1x2,  128, 256, 64, 1, 2, 1);
DEF_PP(PP_128x256_64_1x1,  128, 256, 64, 1, 1, 1);
DEF_PP(PP_256x128_64_2x1,  256, 128, 64, 2, 1, 1);
DEF_PP(PP_256x128_64_1x1,  256, 128, 64, 1, 1, 1);
DEF_PP(PP_128x128_64_4x2,  128, 128, 64, 4, 2, 1);
DEF_PP(PP_128x128_64_2x4,  128, 128, 64, 2, 4, 1);

DEF_WS(WS_128x128_64_4x1,  128, 128, 64, 4, 1, 1);
DEF_WS(WS_128x128_64_1x4,  128, 128, 64, 1, 4, 1);
DEF_WS(WS_128x128_64_2x2,  128, 128, 64, 2, 2, 1);
DEF_WS(WS_128x128_64_2x1,  128, 128, 64, 2, 1, 1);
DEF_WS(WS_128x128_64_1x2,  128, 128, 64, 1, 2, 1);
DEF_WS(WS_128x128_64_1x1,  128, 128, 64, 1, 1, 1);
DEF_WS(WS_128x256_64_2x1,  128, 256, 64, 2, 1, 1);
DEF_WS(WS_128x256_64_1x2,  128, 256, 64, 1, 2, 1);
DEF_WS(WS_128x256_64_1x1,  128, 256, 64, 1, 1, 1);
DEF_WS(WS_256x128_64_2x1,  256, 128, 64, 2, 1, 1);
DEF_WS(WS_256x128_64_1x1,  256, 128, 64, 1, 1, 1);

DEF_COOP(COOP_128x128_64_4x1,  128, 128, 64, 4, 1, 1);
DEF_COOP(COOP_128x128_64_1x4,  128, 128, 64, 1, 4, 1);
DEF_COOP(COOP_128x128_64_2x2,  128, 128, 64, 2, 2, 1);
DEF_COOP(COOP_128x128_64_2x1,  128, 128, 64, 2, 1, 1);
DEF_COOP(COOP_128x128_64_1x2,  128, 128, 64, 1, 2, 1);
DEF_COOP(COOP_128x128_64_1x1,  128, 128, 64, 1, 1, 1);
DEF_COOP(COOP_128x256_64_2x1,  128, 256, 64, 2, 1, 1);
DEF_COOP(COOP_128x256_64_1x2,  128, 256, 64, 1, 2, 1);
DEF_COOP(COOP_128x256_64_1x1,  128, 256, 64, 1, 1, 1);
DEF_COOP(COOP_256x128_64_2x1,  256, 128, 64, 2, 1, 1);
DEF_COOP(COOP_256x128_64_1x1,  256, 128, 64, 1, 1, 1);

DEF_COOP_SK(COOP_SK_128x128_64_4x1,  128, 128, 64, 4, 1, 1);
DEF_COOP_SK(COOP_SK_128x128_64_1x4,  128, 128, 64, 1, 4, 1);
DEF_COOP_SK(COOP_SK_128x128_64_2x2,  128, 128, 64, 2, 2, 1);
DEF_COOP_SK(COOP_SK_128x128_64_2x1,  128, 128, 64, 2, 1, 1);
DEF_COOP_SK(COOP_SK_128x128_64_1x2,  128, 128, 64, 1, 2, 1);
DEF_COOP_SK(COOP_SK_128x128_64_1x1,  128, 128, 64, 1, 1, 1);
DEF_COOP_SK(COOP_SK_128x256_64_2x1,  128, 256, 64, 2, 1, 1);
DEF_COOP_SK(COOP_SK_128x256_64_1x2,  128, 256, 64, 1, 2, 1);
DEF_COOP_SK(COOP_SK_256x128_64_2x1,  256, 128, 64, 2, 1, 1);

struct IRunner {
    virtual ~IRunner() = default;
    virtual cutlass::Status try_init(int M, int N, int K,
                                     const void* A, const void* B, void* C) = 0;
    virtual cutlass::Status run_hot(int M, int N, int K,
                                    const void* A, const void* B, void* C) = 0;
    virtual const char* name() const = 0;
};

template <typename Cfg>
struct Runner : IRunner {
    using Gemm = typename Cfg::Gemm;

    static Gemm& get_gemm() {
        static Gemm gemm_op;
        return gemm_op;
    }

    typename Gemm::Arguments make_args(int M, int N, int K,
                                       const void* ptr_A, const void* ptr_B, void* ptr_C) {
        auto stride_A = cutlass::make_cute_packed_stride(
            typename Cfg::StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        auto stride_C = cutlass::make_cute_packed_stride(
            typename Cfg::StrideC{}, cute::make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(
            typename Cfg::StrideD{}, cute::make_shape(M, N, 1));

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = g_state.device_id;
        hw_info.sm_count  = g_state.sm_count;

        return typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {reinterpret_cast<const typename Cfg::ElementA*>(ptr_A), stride_A,
             reinterpret_cast<const typename Cfg::ElementB*>(ptr_B), stride_B},
            {{1.0f, 0.0f},
             reinterpret_cast<typename Cfg::ElementC*>(ptr_C), stride_C,
             reinterpret_cast<typename Cfg::ElementC*>(ptr_C), stride_D},
            hw_info
        };
    }

    cutlass::Status try_init(int M, int N, int K,
                             const void* A, const void* B, void* C) override {
        auto args = make_args(M, N, K, A, B, C);
        auto& g = get_gemm();
        auto status = g.can_implement(args);
        if (status != cutlass::Status::kSuccess) return status;
        size_t ws_sz = Gemm::get_workspace_size(args);
        return g.initialize(args, g_state.workspace(ws_sz));
    }

    cutlass::Status run_hot(int M, int N, int K,
                            const void* A, const void* B, void* C) override {
        auto args = make_args(M, N, K, A, B, C);
        auto& g = get_gemm();
        size_t ws_sz = Gemm::get_workspace_size(args);
        auto status = g.initialize(args, g_state.workspace(ws_sz));
        if (status != cutlass::Status::kSuccess) return status;
        return g.run();
    }

    const char* name() const override { return typeid(Cfg).name(); }
};

static IRunner* g_winner_runner = nullptr;
static bool g_initialized = false;

static std::vector<IRunner*> build_priority_runners() {
    std::vector<IRunner*> v;
    v.push_back(new Runner<PP_128x128_128_4x1>());
    v.push_back(new Runner<PP_128x128_128_4x2>());
    v.push_back(new Runner<PP_128x128_128_2x4>());
    v.push_back(new Runner<PP_128x128_128_1x4>());
    v.push_back(new Runner<PP_128x128_128_2x2>());
    v.push_back(new Runner<PP_128x128_128_2x1>());
    v.push_back(new Runner<PP_128x128_128_1x2>());
    v.push_back(new Runner<PP_128x256_128_4x1>());
    v.push_back(new Runner<PP_128x256_128_2x2>());
    v.push_back(new Runner<PP_128x256_128_2x1>());
    v.push_back(new Runner<PP_128x256_128_1x2>());
    v.push_back(new Runner<PP_256x128_128_4x1>());
    v.push_back(new Runner<PP_256x128_128_2x1>());
    v.push_back(new Runner<PP_256x128_128_1x2>());
    v.push_back(new Runner<PP_128x128_128_1x1>());
    v.push_back(new Runner<PP_128x256_128_1x1>());
    v.push_back(new Runner<PP_256x128_128_1x1>());
    return v;
}

static std::vector<IRunner*>& get_runners() {
    static std::vector<IRunner*> runners;
    if (!runners.empty()) return runners;

    runners = build_priority_runners();

    runners.push_back(new Runner<WS_128x128_128_4x1>());
    runners.push_back(new Runner<WS_128x128_128_4x2>());
    runners.push_back(new Runner<WS_128x128_128_2x4>());
    runners.push_back(new Runner<WS_128x128_128_1x4>());
    runners.push_back(new Runner<WS_128x128_128_2x2>());
    runners.push_back(new Runner<WS_128x128_128_2x1>());
    runners.push_back(new Runner<WS_128x128_128_1x2>());
    runners.push_back(new Runner<WS_128x128_128_1x1>());
    runners.push_back(new Runner<WS_128x256_128_2x1>());
    runners.push_back(new Runner<WS_128x256_128_1x2>());
    runners.push_back(new Runner<WS_128x256_128_1x1>());
    runners.push_back(new Runner<WS_256x128_128_2x1>());
    runners.push_back(new Runner<WS_256x128_128_1x1>());

    runners.push_back(new Runner<COOP_128x128_128_4x1>());
    runners.push_back(new Runner<COOP_128x128_128_4x2>());
    runners.push_back(new Runner<COOP_128x128_128_2x4>());
    runners.push_back(new Runner<COOP_128x128_128_1x4>());
    runners.push_back(new Runner<COOP_128x128_128_2x2>());
    runners.push_back(new Runner<COOP_128x128_128_2x1>());
    runners.push_back(new Runner<COOP_128x128_128_1x2>());
    runners.push_back(new Runner<COOP_128x128_128_1x1>());
    runners.push_back(new Runner<COOP_128x256_128_2x1>());
    runners.push_back(new Runner<COOP_128x256_128_1x2>());
    runners.push_back(new Runner<COOP_128x256_128_1x1>());
    runners.push_back(new Runner<COOP_256x128_128_2x1>());
    runners.push_back(new Runner<COOP_256x128_128_1x1>());

    runners.push_back(new Runner<COOP_SK_128x128_128_4x1>());
    runners.push_back(new Runner<COOP_SK_128x128_128_4x2>());
    runners.push_back(new Runner<COOP_SK_128x128_128_2x4>());
    runners.push_back(new Runner<COOP_SK_128x128_128_1x4>());
    runners.push_back(new Runner<COOP_SK_128x128_128_2x2>());
    runners.push_back(new Runner<COOP_SK_128x128_128_2x1>());
    runners.push_back(new Runner<COOP_SK_128x128_128_1x2>());
    runners.push_back(new Runner<COOP_SK_128x128_128_1x1>());
    runners.push_back(new Runner<COOP_SK_128x256_128_2x1>());
    runners.push_back(new Runner<COOP_SK_128x256_128_1x2>());
    runners.push_back(new Runner<COOP_SK_128x256_128_1x1>());
    runners.push_back(new Runner<COOP_SK_256x128_128_2x1>());
    runners.push_back(new Runner<COOP_SK_256x128_128_1x1>());

    runners.push_back(new Runner<PP_128x128_64_4x1>());
    runners.push_back(new Runner<PP_128x128_64_4x2>());
    runners.push_back(new Runner<PP_128x128_64_2x4>());
    runners.push_back(new Runner<PP_128x128_64_1x4>());
    runners.push_back(new Runner<PP_128x128_64_2x2>());
    runners.push_back(new Runner<PP_128x128_64_2x1>());
    runners.push_back(new Runner<PP_128x128_64_1x2>());
    runners.push_back(new Runner<PP_128x128_64_1x1>());
    runners.push_back(new Runner<PP_128x256_64_2x1>());
    runners.push_back(new Runner<PP_128x256_64_1x2>());
    runners.push_back(new Runner<PP_128x256_64_1x1>());
    runners.push_back(new Runner<PP_256x128_64_2x1>());
    runners.push_back(new Runner<PP_256x128_64_1x1>());

    runners.push_back(new Runner<WS_128x128_64_4x1>());
    runners.push_back(new Runner<WS_128x128_64_1x4>());
    runners.push_back(new Runner<WS_128x128_64_2x2>());
    runners.push_back(new Runner<WS_128x128_64_2x1>());
    runners.push_back(new Runner<WS_128x128_64_1x2>());
    runners.push_back(new Runner<WS_128x128_64_1x1>());
    runners.push_back(new Runner<WS_128x256_64_2x1>());
    runners.push_back(new Runner<WS_128x256_64_1x2>());
    runners.push_back(new Runner<WS_128x256_64_1x1>());
    runners.push_back(new Runner<WS_256x128_64_2x1>());
    runners.push_back(new Runner<WS_256x128_64_1x1>());

    runners.push_back(new Runner<COOP_128x128_64_4x1>());
    runners.push_back(new Runner<COOP_128x128_64_1x4>());
    runners.push_back(new Runner<COOP_128x128_64_2x2>());
    runners.push_back(new Runner<COOP_128x128_64_2x1>());
    runners.push_back(new Runner<COOP_128x128_64_1x2>());
    runners.push_back(new Runner<COOP_128x128_64_1x1>());
    runners.push_back(new Runner<COOP_128x256_64_2x1>());
    runners.push_back(new Runner<COOP_128x256_64_1x2>());
    runners.push_back(new Runner<COOP_128x256_64_1x1>());
    runners.push_back(new Runner<COOP_256x128_64_2x1>());
    runners.push_back(new Runner<COOP_256x128_64_1x1>());

    runners.push_back(new Runner<COOP_SK_128x128_64_4x1>());
    runners.push_back(new Runner<COOP_SK_128x128_64_1x4>());
    runners.push_back(new Runner<COOP_SK_128x128_64_2x2>());
    runners.push_back(new Runner<COOP_SK_128x128_64_2x1>());
    runners.push_back(new Runner<COOP_SK_128x128_64_1x2>());
    runners.push_back(new Runner<COOP_SK_128x128_64_1x1>());
    runners.push_back(new Runner<COOP_SK_128x256_64_2x1>());
    runners.push_back(new Runner<COOP_SK_128x256_64_1x2>());
    runners.push_back(new Runner<COOP_SK_256x128_64_2x1>());

    return runners;
}

static void select_winner(int M, int N, int K,
                           const void* ptr_A, const void* ptr_B, void* ptr_C) {
    auto& runners = get_runners();

    float best_ms = std::numeric_limits<float>::max();
    IRunner* best = nullptr;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    constexpr int WARMUP = 3;
    constexpr int ITERS  = 8;

    for (auto* r : runners) {
        auto status = r->try_init(M, N, K, ptr_A, ptr_B, ptr_C);
        if (status != cutlass::Status::kSuccess) continue;

        bool ok = true;
        for (int i = 0; i < WARMUP && ok; i++) {
            if (r->run_hot(M, N, K, ptr_A, ptr_B, ptr_C) != cutlass::Status::kSuccess)
                ok = false;
        }
        if (!ok) continue;

        cudaEventRecord(ev_start);
        for (int i = 0; i < ITERS && ok; i++) {
            if (r->run_hot(M, N, K, ptr_A, ptr_B, ptr_C) != cutlass::Status::kSuccess)
                ok = false;
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        if (!ok) continue;

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        ms /= ITERS;

        if (ms < best_ms) {
            best_ms = ms;
            best = r;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    if (!best)
        throw std::runtime_error("cuda_l2_h100_fp32: all configs failed during selection");

    g_winner_runner = best;
    g_winner_runner->try_init(M, N, K, ptr_A, ptr_B, ptr_C);
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    g_state.init();

    const void* ptr_A = a.data_ptr();
    const void* ptr_B = b_col_major.data_ptr();
    void*       ptr_C = c.data_ptr();

    if (!g_initialized) {
        select_winner(M, N, K, ptr_A, ptr_B, ptr_C);
        g_initialized = true;
        auto status = g_winner_runner->run_hot(M, N, K, ptr_A, ptr_B, ptr_C);
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error("cuda_l2_h100_fp32: winner run failed after selection");
        return;
    }

    auto status = g_winner_runner->run_hot(M, N, K, ptr_A, ptr_B, ptr_C);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("cuda_l2_h100_fp32: GEMM execution failed on hot path");

#else
    throw std::runtime_error("cuda_l2_h100_fp32: CUTLASS SM90 MMA not supported on this device");
#endif
}