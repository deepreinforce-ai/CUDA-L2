#include <iostream>
#include <stdexcept>
#include <cfloat>
#include <cstring>

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

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutA            = cutlass::layout::RowMajor;
using LayoutB            = cutlass::layout::ColumnMajor;
using LayoutC            = cutlass::layout::RowMajor;
using LayoutD            = cutlass::layout::RowMajor;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEF_COOP_SK(Name, TM, TN, TK, CM, CN, CK)                             \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>; \
  using WorkShape    = cute::Shape<cute::Int<CM>,cute::Int<CN>,cute::Int<CK>>; \
  using StageEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using StageMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                        \
      TileShape, WorkShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename StageEpilogue::SharedStorage))>,      \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, StageMainloop, StageEpilogue,                  \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_PP_PERS(Name, TM, TN, TK, CM, CN, CK)                             \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>; \
  using WorkShape    = cute::Shape<cute::Int<CM>,cute::Int<CN>,cute::Int<CK>>; \
  using StageEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                                \
  using StageMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                        \
      TileShape, WorkShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename StageEpilogue::SharedStorage))>,      \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, StageMainloop, StageEpilogue,                  \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_COOP_PERS(Name, TM, TN, TK, CM, CN, CK)                           \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>; \
  using WorkShape    = cute::Shape<cute::Int<CM>,cute::Int<CN>,cute::Int<CK>>; \
  using StageEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using StageMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                        \
      TileShape, WorkShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename StageEpilogue::SharedStorage))>,      \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, StageMainloop, StageEpilogue,                  \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_COOP_SK_STAGES(Name, TM, TN, TK, CM, CN, CK, NS)                  \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>; \
  using WorkShape    = cute::Shape<cute::Int<CM>,cute::Int<CN>,cute::Int<CK>>; \
  using StageEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                                \
  using StageMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                        \
      TileShape, WorkShape,                                                     \
      cutlass::gemm::collective::StageCount<NS>,                               \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, StageMainloop, StageEpilogue,                  \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_PP_PERS_STAGES(Name, TM, TN, TK, CM, CN, CK, NS)                  \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>; \
  using WorkShape    = cute::Shape<cute::Int<CM>,cute::Int<CN>,cute::Int<CK>>; \
  using StageEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, WorkShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                                \
      ElementD, LayoutD, AlignD,                                                \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                                \
  using StageMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                                \
      ElementB, LayoutB, AlignB,                                                \
      ElementAccumulator,                                                        \
      TileShape, WorkShape,                                                     \
      cutlass::gemm::collective::StageCount<NS>,                               \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, StageMainloop, StageEpilogue,                  \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEF_COOP_SK       (Cfg_CS_128x64x64_1x2,        128,  64, 64, 1, 2, 1)
DEF_PP_PERS       (Cfg_PP_128x64x64_1x2,        128,  64, 64, 1, 2, 1)
DEF_COOP_PERS     (Cfg_CP_128x64x64_1x2,        128,  64, 64, 1, 2, 1)

DEF_COOP_SK_STAGES(Cfg_CS_128x64x64_1x2_S4,     128,  64, 64, 1, 2, 1, 4)
DEF_COOP_SK_STAGES(Cfg_CS_128x64x64_1x2_S5,     128,  64, 64, 1, 2, 1, 5)
DEF_COOP_SK_STAGES(Cfg_CS_128x64x64_1x2_S6,     128,  64, 64, 1, 2, 1, 6)
DEF_PP_PERS_STAGES(Cfg_PP_128x64x64_1x2_S4,     128,  64, 64, 1, 2, 1, 4)
DEF_PP_PERS_STAGES(Cfg_PP_128x64x64_1x2_S5,     128,  64, 64, 1, 2, 1, 5)
DEF_PP_PERS_STAGES(Cfg_PP_128x64x64_1x2_S6,     128,  64, 64, 1, 2, 1, 6)

DEF_COOP_SK       (Cfg_CS_128x64x64_2x2,        128,  64, 64, 2, 2, 1)
DEF_PP_PERS       (Cfg_PP_128x64x64_2x2,        128,  64, 64, 2, 2, 1)
DEF_COOP_PERS     (Cfg_CP_128x64x64_2x2,        128,  64, 64, 2, 2, 1)
DEF_COOP_SK_STAGES(Cfg_CS_128x64x64_2x2_S5,     128,  64, 64, 2, 2, 1, 5)
DEF_PP_PERS_STAGES(Cfg_PP_128x64x64_2x2_S5,     128,  64, 64, 2, 2, 1, 5)

DEF_COOP_SK       (Cfg_CS_128x128x64_1x2,       128, 128, 64, 1, 2, 1)
DEF_PP_PERS       (Cfg_PP_128x128x64_1x2,       128, 128, 64, 1, 2, 1)
DEF_COOP_SK_STAGES(Cfg_CS_128x128x64_1x2_S4,    128, 128, 64, 1, 2, 1, 4)
DEF_COOP_SK_STAGES(Cfg_CS_128x128x64_1x2_S5,    128, 128, 64, 1, 2, 1, 5)
DEF_PP_PERS_STAGES(Cfg_PP_128x128x64_1x2_S4,    128, 128, 64, 1, 2, 1, 4)
DEF_PP_PERS_STAGES(Cfg_PP_128x128x64_1x2_S5,    128, 128, 64, 1, 2, 1, 5)

DEF_PP_PERS       (Cfg_PP_64x128x64_1x1,         64, 128, 64, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_64x128x64_2x1,         64, 128, 64, 2, 1, 1)
DEF_PP_PERS       (Cfg_PP_64x128x32_1x1,         64, 128, 32, 1, 1, 1)
DEF_PP_PERS_STAGES(Cfg_PP_64x128x64_1x1_S5,      64, 128, 64, 1, 1, 1, 5)
DEF_PP_PERS_STAGES(Cfg_PP_64x128x64_1x1_S6,      64, 128, 64, 1, 1, 1, 6)
DEF_PP_PERS_STAGES(Cfg_PP_64x128x64_2x1_S5,      64, 128, 64, 2, 1, 1, 5)

DEF_COOP_SK       (Cfg_CS_128x32x64_1x4,        128,  32, 64, 1, 4, 1)
DEF_PP_PERS       (Cfg_PP_128x32x64_1x4,        128,  32, 64, 1, 4, 1)
DEF_COOP_SK       (Cfg_CS_128x64x64_1x4,        128,  64, 64, 1, 4, 1)
DEF_PP_PERS       (Cfg_PP_128x64x64_1x4,        128,  64, 64, 1, 4, 1)

DEF_COOP_SK       (Cfg_CS_128x64x64_2x1,        128,  64, 64, 2, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x64x64_2x1,        128,  64, 64, 2, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x128x64_2x1,       128, 128, 64, 2, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x128x64_2x1,       128, 128, 64, 2, 1, 1)
DEF_COOP_PERS     (Cfg_CP_128x128x64_2x1,       128, 128, 64, 2, 1, 1)

DEF_COOP_SK       (Cfg_CS_128x64x64_4x1,        128,  64, 64, 4, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x128x64_4x1,       128, 128, 64, 4, 1, 1)

DEF_COOP_SK       (Cfg_CS_128x64x64_1x1,        128,  64, 64, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x64x64_1x1,        128,  64, 64, 1, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x128x64_1x1,       128, 128, 64, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x128x64_1x1,       128, 128, 64, 1, 1, 1)
DEF_COOP_PERS     (Cfg_CP_128x128x64_1x1,       128, 128, 64, 1, 1, 1)

DEF_COOP_SK       (Cfg_CS_128x64x32_1x2,        128,  64, 32, 1, 2, 1)
DEF_PP_PERS       (Cfg_PP_128x64x32_1x2,        128,  64, 32, 1, 2, 1)
DEF_COOP_SK       (Cfg_CS_128x64x32_2x1,        128,  64, 32, 2, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x64x32_2x1,        128,  64, 32, 2, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x64x32_1x1,        128,  64, 32, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x64x32_1x1,        128,  64, 32, 1, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x128x32_2x1,       128, 128, 32, 2, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x128x32_2x1,       128, 128, 32, 2, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x128x32_1x1,       128, 128, 32, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x128x32_1x1,       128, 128, 32, 1, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x64x32_1x4,        128,  64, 32, 1, 4, 1)
DEF_COOP_SK       (Cfg_CS_128x64x32_2x2,        128,  64, 32, 2, 2, 1)
DEF_COOP_SK       (Cfg_CS_128x128x32_1x2,       128, 128, 32, 1, 2, 1)
DEF_PP_PERS       (Cfg_PP_128x128x32_1x2,       128, 128, 32, 1, 2, 1)

DEF_COOP_SK       (Cfg_CS_256x64x64_1x2,        256,  64, 64, 1, 2, 1)
DEF_COOP_SK       (Cfg_CS_256x64x64_2x1,        256,  64, 64, 2, 1, 1)
DEF_COOP_SK       (Cfg_CS_256x64x64_1x1,        256,  64, 64, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_256x64x64_1x2,        256,  64, 64, 1, 2, 1)
DEF_PP_PERS       (Cfg_PP_256x64x64_2x1,        256,  64, 64, 2, 1, 1)
DEF_PP_PERS       (Cfg_PP_256x64x64_1x1,        256,  64, 64, 1, 1, 1)
DEF_COOP_PERS     (Cfg_CP_256x64x64_2x1,        256,  64, 64, 2, 1, 1)
DEF_COOP_SK       (Cfg_CS_256x128x64_1x1,       256, 128, 64, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_256x128x64_1x1,       256, 128, 64, 1, 1, 1)
DEF_COOP_SK       (Cfg_CS_256x128x64_2x1,       256, 128, 64, 2, 1, 1)
DEF_COOP_SK       (Cfg_CS_256x64x64_2x2,        256,  64, 64, 2, 2, 1)

DEF_COOP_SK       (Cfg_CS_128x32x64_2x1,        128,  32, 64, 2, 1, 1)
DEF_COOP_SK       (Cfg_CS_128x32x64_1x1,        128,  32, 64, 1, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x32x64_2x1,        128,  32, 64, 2, 1, 1)
DEF_PP_PERS       (Cfg_PP_128x32x64_1x1,        128,  32, 64, 1, 1, 1)

static void*  s_ws_ptr   = nullptr;
static size_t s_ws_size  = 0;
static int    s_dev_id   = -1;
static int    s_sm_cnt   = -1;
static int    s_best_cfg = 0;
static bool   s_tuned    = false;

static void ensure_workspace(size_t needed) {
  if (needed > s_ws_size) {
    if (s_ws_ptr) cudaFree(s_ws_ptr);
    cudaMalloc(&s_ws_ptr, needed);
    s_ws_size = needed;
  }
}

static void ensure_hw_info() {
  if (s_dev_id < 0) {
    cudaGetDevice(&s_dev_id);
    s_sm_cnt = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(s_dev_id);
  }
}

enum CfgID {
  ID_CS_128x64x64_1x2 = 0,
  ID_PP_128x64x64_1x2,
  ID_CP_128x64x64_1x2,
  ID_CS_128x64x64_1x2_S4,
  ID_CS_128x64x64_1x2_S5,
  ID_CS_128x64x64_1x2_S6,
  ID_PP_128x64x64_1x2_S4,
  ID_PP_128x64x64_1x2_S5,
  ID_PP_128x64x64_1x2_S6,
  ID_CS_128x64x64_2x2,
  ID_PP_128x64x64_2x2,
  ID_CP_128x64x64_2x2,
  ID_CS_128x64x64_2x2_S5,
  ID_PP_128x64x64_2x2_S5,
  ID_CS_128x128x64_1x2,
  ID_PP_128x128x64_1x2,
  ID_CS_128x128x64_1x2_S4,
  ID_CS_128x128x64_1x2_S5,
  ID_PP_128x128x64_1x2_S4,
  ID_PP_128x128x64_1x2_S5,
  ID_PP_64x128x64_1x1,
  ID_PP_64x128x64_2x1,
  ID_PP_64x128x32_1x1,
  ID_PP_64x128x64_1x1_S5,
  ID_PP_64x128x64_1x1_S6,
  ID_PP_64x128x64_2x1_S5,
  ID_CS_128x32x64_1x4,
  ID_PP_128x32x64_1x4,
  ID_CS_128x64x64_1x4,
  ID_PP_128x64x64_1x4,
  ID_CS_128x64x64_2x1,
  ID_PP_128x64x64_2x1,
  ID_CS_128x128x64_2x1,
  ID_PP_128x128x64_2x1,
  ID_CP_128x128x64_2x1,
  ID_CS_128x64x64_4x1,
  ID_CS_128x128x64_4x1,
  ID_CS_128x64x64_1x1,
  ID_PP_128x64x64_1x1,
  ID_CS_128x128x64_1x1,
  ID_PP_128x128x64_1x1,
  ID_CP_128x128x64_1x1,
  ID_CS_128x64x32_1x2,
  ID_PP_128x64x32_1x2,
  ID_CS_128x64x32_2x1,
  ID_PP_128x64x32_2x1,
  ID_CS_128x64x32_1x1,
  ID_PP_128x64x32_1x1,
  ID_CS_128x128x32_2x1,
  ID_PP_128x128x32_2x1,
  ID_CS_128x128x32_1x1,
  ID_PP_128x128x32_1x1,
  ID_CS_128x64x32_1x4,
  ID_CS_128x64x32_2x2,
  ID_CS_128x128x32_1x2,
  ID_PP_128x128x32_1x2,
  ID_CS_256x64x64_1x2,
  ID_CS_256x64x64_2x1,
  ID_CS_256x64x64_1x1,
  ID_PP_256x64x64_1x2,
  ID_PP_256x64x64_2x1,
  ID_PP_256x64x64_1x1,
  ID_CP_256x64x64_2x1,
  ID_CS_256x128x64_1x1,
  ID_PP_256x128x64_1x1,
  ID_CS_256x128x64_2x1,
  ID_CS_256x64x64_2x2,
  ID_CS_128x32x64_2x1,
  ID_CS_128x32x64_1x1,
  ID_PP_128x32x64_2x1,
  ID_PP_128x32x64_1x1,
  ID_COUNT
};

template <typename Cfg>
bool run_cfg(const half* A, const half* B, half* C, int M, int N, int K,
             bool check_impl = true) {
  using Gemm    = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;

  ensure_hw_info();

  auto sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  auto sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw;
  hw.device_id = s_dev_id;
  hw.sm_count  = s_sm_cnt;

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<const ElementA*>(A), sA,
     reinterpret_cast<const ElementB*>(B), sB},
    {{1.0f, 0.0f},
     reinterpret_cast<const ElementC*>(C), sC,
     reinterpret_cast<ElementC*>(C), sD},
    hw
  };

  Gemm gemm;
  if (check_impl && gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t ws = Gemm::get_workspace_size(args);
  ensure_workspace(ws);

  if (gemm.initialize(args, s_ws_ptr) != cutlass::Status::kSuccess) return false;
  return gemm.run() == cutlass::Status::kSuccess;
}

static bool dispatch(int id, const half* A, const half* B, half* C, int M, int N, int K,
                     bool check_impl = false) {
  switch (id) {
    case ID_CS_128x64x64_1x2:       return run_cfg<Cfg_CS_128x64x64_1x2>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_1x2:       return run_cfg<Cfg_PP_128x64x64_1x2>      (A,B,C,M,N,K,check_impl);
    case ID_CP_128x64x64_1x2:       return run_cfg<Cfg_CP_128x64x64_1x2>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_1x2_S4:    return run_cfg<Cfg_CS_128x64x64_1x2_S4>   (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_1x2_S5:    return run_cfg<Cfg_CS_128x64x64_1x2_S5>   (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_1x2_S6:    return run_cfg<Cfg_CS_128x64x64_1x2_S6>   (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_1x2_S4:    return run_cfg<Cfg_PP_128x64x64_1x2_S4>   (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_1x2_S5:    return run_cfg<Cfg_PP_128x64x64_1x2_S5>   (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_1x2_S6:    return run_cfg<Cfg_PP_128x64x64_1x2_S6>   (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_2x2:       return run_cfg<Cfg_CS_128x64x64_2x2>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_2x2:       return run_cfg<Cfg_PP_128x64x64_2x2>      (A,B,C,M,N,K,check_impl);
    case ID_CP_128x64x64_2x2:       return run_cfg<Cfg_CP_128x64x64_2x2>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_2x2_S5:    return run_cfg<Cfg_CS_128x64x64_2x2_S5>   (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_2x2_S5:    return run_cfg<Cfg_PP_128x64x64_2x2_S5>   (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x64_1x2:      return run_cfg<Cfg_CS_128x128x64_1x2>     (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x64_1x2:      return run_cfg<Cfg_PP_128x128x64_1x2>     (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x64_1x2_S4:   return run_cfg<Cfg_CS_128x128x64_1x2_S4>  (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x64_1x2_S5:   return run_cfg<Cfg_CS_128x128x64_1x2_S5>  (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x64_1x2_S4:   return run_cfg<Cfg_PP_128x128x64_1x2_S4>  (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x64_1x2_S5:   return run_cfg<Cfg_PP_128x128x64_1x2_S5>  (A,B,C,M,N,K,check_impl);
    case ID_PP_64x128x64_1x1:       return run_cfg<Cfg_PP_64x128x64_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_64x128x64_2x1:       return run_cfg<Cfg_PP_64x128x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_64x128x32_1x1:       return run_cfg<Cfg_PP_64x128x32_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_64x128x64_1x1_S5:    return run_cfg<Cfg_PP_64x128x64_1x1_S5>   (A,B,C,M,N,K,check_impl);
    case ID_PP_64x128x64_1x1_S6:    return run_cfg<Cfg_PP_64x128x64_1x1_S6>   (A,B,C,M,N,K,check_impl);
    case ID_PP_64x128x64_2x1_S5:    return run_cfg<Cfg_PP_64x128x64_2x1_S5>   (A,B,C,M,N,K,check_impl);
    case ID_CS_128x32x64_1x4:       return run_cfg<Cfg_CS_128x32x64_1x4>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x32x64_1x4:       return run_cfg<Cfg_PP_128x32x64_1x4>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_1x4:       return run_cfg<Cfg_CS_128x64x64_1x4>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_1x4:       return run_cfg<Cfg_PP_128x64x64_1x4>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_2x1:       return run_cfg<Cfg_CS_128x64x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_2x1:       return run_cfg<Cfg_PP_128x64x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x64_2x1:      return run_cfg<Cfg_CS_128x128x64_2x1>     (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x64_2x1:      return run_cfg<Cfg_PP_128x128x64_2x1>     (A,B,C,M,N,K,check_impl);
    case ID_CP_128x128x64_2x1:      return run_cfg<Cfg_CP_128x128x64_2x1>     (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_4x1:       return run_cfg<Cfg_CS_128x64x64_4x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x64_4x1:      return run_cfg<Cfg_CS_128x128x64_4x1>     (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x64_1x1:       return run_cfg<Cfg_CS_128x64x64_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x64_1x1:       return run_cfg<Cfg_PP_128x64x64_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x64_1x1:      return run_cfg<Cfg_CS_128x128x64_1x1>     (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x64_1x1:      return run_cfg<Cfg_PP_128x128x64_1x1>     (A,B,C,M,N,K,check_impl);
    case ID_CP_128x128x64_1x1:      return run_cfg<Cfg_CP_128x128x64_1x1>     (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x32_1x2:       return run_cfg<Cfg_CS_128x64x32_1x2>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x32_1x2:       return run_cfg<Cfg_PP_128x64x32_1x2>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x32_2x1:       return run_cfg<Cfg_CS_128x64x32_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x32_2x1:       return run_cfg<Cfg_PP_128x64x32_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x32_1x1:       return run_cfg<Cfg_CS_128x64x32_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x64x32_1x1:       return run_cfg<Cfg_PP_128x64x32_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x32_2x1:      return run_cfg<Cfg_CS_128x128x32_2x1>     (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x32_2x1:      return run_cfg<Cfg_PP_128x128x32_2x1>     (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x32_1x1:      return run_cfg<Cfg_CS_128x128x32_1x1>     (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x32_1x1:      return run_cfg<Cfg_PP_128x128x32_1x1>     (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x32_1x4:       return run_cfg<Cfg_CS_128x64x32_1x4>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x64x32_2x2:       return run_cfg<Cfg_CS_128x64x32_2x2>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x128x32_1x2:      return run_cfg<Cfg_CS_128x128x32_1x2>     (A,B,C,M,N,K,check_impl);
    case ID_PP_128x128x32_1x2:      return run_cfg<Cfg_PP_128x128x32_1x2>     (A,B,C,M,N,K,check_impl);
    case ID_CS_256x64x64_1x2:       return run_cfg<Cfg_CS_256x64x64_1x2>      (A,B,C,M,N,K,check_impl);
    case ID_CS_256x64x64_2x1:       return run_cfg<Cfg_CS_256x64x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_256x64x64_1x1:       return run_cfg<Cfg_CS_256x64x64_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_256x64x64_1x2:       return run_cfg<Cfg_PP_256x64x64_1x2>      (A,B,C,M,N,K,check_impl);
    case ID_PP_256x64x64_2x1:       return run_cfg<Cfg_PP_256x64x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_256x64x64_1x1:       return run_cfg<Cfg_PP_256x64x64_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_CP_256x64x64_2x1:       return run_cfg<Cfg_CP_256x64x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_256x128x64_1x1:      return run_cfg<Cfg_CS_256x128x64_1x1>     (A,B,C,M,N,K,check_impl);
    case ID_PP_256x128x64_1x1:      return run_cfg<Cfg_PP_256x128x64_1x1>     (A,B,C,M,N,K,check_impl);
    case ID_CS_256x128x64_2x1:      return run_cfg<Cfg_CS_256x128x64_2x1>     (A,B,C,M,N,K,check_impl);
    case ID_CS_256x64x64_2x2:       return run_cfg<Cfg_CS_256x64x64_2x2>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x32x64_2x1:       return run_cfg<Cfg_CS_128x32x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_CS_128x32x64_1x1:       return run_cfg<Cfg_CS_128x32x64_1x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x32x64_2x1:       return run_cfg<Cfg_PP_128x32x64_2x1>      (A,B,C,M,N,K,check_impl);
    case ID_PP_128x32x64_1x1:       return run_cfg<Cfg_PP_128x32x64_1x1>      (A,B,C,M,N,K,check_impl);
    default: return false;
  }
}

static float do_benchmark(int id, const half* A, const half* B, half* C,
                          int M, int N, int K, int warmup = 5, int reps = 20) {
  for (int i = 0; i < warmup; i++) dispatch(id, A, B, C, M, N, K, false);
  cudaDeviceSynchronize();

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  for (int i = 0; i < reps; i++) dispatch(id, A, B, C, M, N, K, false);
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev0, ev1);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  return ms / reps;
}

static const int TUNE_CANDIDATES[] = {
  ID_CS_128x64x64_1x2,
  ID_PP_128x64x64_1x2,
  ID_CP_128x64x64_1x2,
  ID_CS_128x64x64_1x2_S4,
  ID_CS_128x64x64_1x2_S5,
  ID_CS_128x64x64_1x2_S6,
  ID_PP_128x64x64_1x2_S4,
  ID_PP_128x64x64_1x2_S5,
  ID_PP_128x64x64_1x2_S6,
  ID_CS_128x64x64_2x2,
  ID_PP_128x64x64_2x2,
  ID_CP_128x64x64_2x2,
  ID_CS_128x64x64_2x2_S5,
  ID_PP_128x64x64_2x2_S5,
  ID_CS_128x128x64_1x2,
  ID_PP_128x128x64_1x2,
  ID_CS_128x128x64_1x2_S4,
  ID_CS_128x128x64_1x2_S5,
  ID_PP_128x128x64_1x2_S4,
  ID_PP_128x128x64_1x2_S5,
  ID_PP_64x128x64_1x1,
  ID_PP_64x128x64_2x1,
  ID_PP_64x128x32_1x1,
  ID_PP_64x128x64_1x1_S5,
  ID_PP_64x128x64_1x1_S6,
  ID_PP_64x128x64_2x1_S5,
  ID_CS_128x32x64_1x4,
  ID_PP_128x32x64_1x4,
  ID_CS_128x64x64_1x4,
  ID_PP_128x64x64_1x4,
  ID_CS_128x64x64_2x1,
  ID_PP_128x64x64_2x1,
  ID_CS_128x128x64_2x1,
  ID_PP_128x128x64_2x1,
  ID_CP_128x128x64_2x1,
  ID_CS_128x64x64_4x1,
  ID_CS_128x128x64_4x1,
  ID_CS_128x64x64_1x1,
  ID_PP_128x64x64_1x1,
  ID_CS_128x128x64_1x1,
  ID_PP_128x128x64_1x1,
  ID_CP_128x128x64_1x1,
  ID_CS_128x64x32_1x2,
  ID_PP_128x64x32_1x2,
  ID_CS_128x64x32_1x4,
  ID_CS_128x64x32_2x2,
  ID_CS_128x128x32_1x2,
  ID_PP_128x128x32_1x2,
  ID_CS_128x64x32_2x1,
  ID_PP_128x64x32_2x1,
  ID_CS_128x64x32_1x1,
  ID_PP_128x64x32_1x1,
  ID_CS_128x128x32_2x1,
  ID_PP_128x128x32_2x1,
  ID_CS_128x128x32_1x1,
  ID_PP_128x128x32_1x1,
  ID_CS_256x64x64_1x2,
  ID_CS_256x64x64_2x1,
  ID_PP_256x64x64_1x2,
  ID_PP_256x64x64_2x1,
  ID_CS_256x64x64_2x2,
  ID_CS_256x64x64_1x1,
  ID_PP_256x64x64_1x1,
  ID_CP_256x64x64_2x1,
  ID_CS_256x128x64_1x1,
  ID_PP_256x128x64_1x1,
  ID_CS_256x128x64_2x1,
  ID_CS_128x32x64_2x1,
  ID_PP_128x32x64_2x1,
  ID_CS_128x32x64_1x1,
  ID_PP_128x32x64_1x1,
};
static const int NUM_TUNE_CANDIDATES = (int)(sizeof(TUNE_CANDIDATES) / sizeof(int));

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
  const half* ptr_B = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half*       ptr_C = reinterpret_cast<half*>(c.data_ptr());

  ensure_hw_info();

  if (s_tuned) {
    if (dispatch(s_best_cfg, ptr_A, ptr_B, ptr_C, M, N, K, false)) return;
    s_tuned = false;
    s_best_cfg = 0;
  }

  static int s_call_count = 0;
  s_call_count++;

  if (s_call_count == 1) {
    if (dispatch(s_best_cfg, ptr_A, ptr_B, ptr_C, M, N, K, false)) {
      return;
    }
  }

  {
    bool valid[NUM_TUNE_CANDIDATES];
    memset(valid, 0, sizeof(valid));
    bool any_valid = false;

    for (int i = 0; i < NUM_TUNE_CANDIDATES; i++) {
      valid[i] = dispatch(TUNE_CANDIDATES[i], ptr_A, ptr_B, ptr_C, M, N, K, true);
      if (valid[i]) any_valid = true;
    }

    if (!any_valid) {
      throw std::runtime_error("All CUTLASS SM90 GEMM variants failed for this problem size");
    }

    float best_time = FLT_MAX;
    int   best_id   = s_best_cfg;

    for (int i = 0; i < NUM_TUNE_CANDIDATES; i++) {
      if (!valid[i]) continue;
      float t = do_benchmark(TUNE_CANDIDATES[i], ptr_A, ptr_B, ptr_C, M, N, K, 5, 20);
      if (t < best_time) {
        best_time = t;
        best_id   = TUNE_CANDIDATES[i];
      }
    }

    s_best_cfg = best_id;
    s_tuned    = true;
    dispatch(s_best_cfg, ptr_A, ptr_B, ptr_C, M, N, K, false);
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}