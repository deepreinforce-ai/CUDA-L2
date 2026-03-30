#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include <torch/extension.h>
#include <torch/types.h>

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ArchTag       = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define DEFINE_GEMM_PP_STAGE(Name, TM, TN, TK, CM, CN, CK, STAGES)           \
namespace cfg_##Name {                                                          \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TileShape, GridShape,                              \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementC, LayoutC, AlignmentC,                                             \
    cutlass::epilogue::NoSmemWarpSpecialized                                   \
>::CollectiveOp;                                                               \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass,                                                    \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator, TileShape, GridShape,                                  \
    cutlass::gemm::collective::StageCount<STAGES>,                             \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong                            \
>::CollectiveOp;                                                               \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                             \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define DEFINE_GEMM_PP_AUTO(Name, TM, TN, TK, CM, CN, CK)                     \
namespace cfg_##Name {                                                          \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TileShape, GridShape,                              \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementC, LayoutC, AlignmentC,                                             \
    cutlass::epilogue::NoSmemWarpSpecialized                                   \
>::CollectiveOp;                                                               \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass,                                                    \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator, TileShape, GridShape,                                  \
    cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,           \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong                            \
>::CollectiveOp;                                                               \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                             \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define DEFINE_GEMM_WS_STAGE(Name, TM, TN, TK, CM, CN, CK, STAGES)           \
namespace cfg_##Name {                                                          \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TileShape, GridShape,                              \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementC, LayoutC, AlignmentC,                                             \
    cutlass::epilogue::NoSmemWarpSpecialized                                   \
>::CollectiveOp;                                                               \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass,                                                    \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator, TileShape, GridShape,                                  \
    cutlass::gemm::collective::StageCount<STAGES>,                             \
    cutlass::gemm::KernelTmaWarpSpecialized                                    \
>::CollectiveOp;                                                               \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                             \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define DEFINE_GEMM_WS_AUTO(Name, TM, TN, TK, CM, CN, CK)                     \
namespace cfg_##Name {                                                          \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TileShape, GridShape,                              \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementC, LayoutC, AlignmentC,                                             \
    cutlass::epilogue::NoSmemWarpSpecialized                                   \
>::CollectiveOp;                                                               \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass,                                                    \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator, TileShape, GridShape,                                  \
    cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,           \
    cutlass::gemm::KernelTmaWarpSpecialized                                    \
>::CollectiveOp;                                                               \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                             \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define DEFINE_GEMM_COOP_STAGE(Name, TM, TN, TK, CM, CN, CK, STAGES)         \
namespace cfg_##Name {                                                          \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TileShape, GridShape,                              \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementC, LayoutC, AlignmentC,                                             \
    cutlass::epilogue::NoSmemWarpSpecialized                                   \
>::CollectiveOp;                                                               \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass,                                                    \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator, TileShape, GridShape,                                  \
    cutlass::gemm::collective::StageCount<STAGES>,                             \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative                         \
>::CollectiveOp;                                                               \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                             \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define DEFINE_GEMM_COOP_AUTO(Name, TM, TN, TK, CM, CN, CK)                   \
namespace cfg_##Name {                                                          \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TileShape, GridShape,                              \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementC, LayoutC, AlignmentC,                                             \
    cutlass::epilogue::NoSmemWarpSpecialized                                   \
>::CollectiveOp;                                                               \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass,                                                    \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator, TileShape, GridShape,                                  \
    cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,           \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative                         \
>::CollectiveOp;                                                               \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                             \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define DEFINE_GEMM_FALLBACK(Name, TM, TN, TK, CM, CN, CK)                    \
namespace cfg_##Name {                                                          \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<    \
    ArchTag, OperatorClass, TileShape, GridShape,                              \
    cutlass::epilogue::collective::EpilogueTileAuto,                           \
    ElementAccumulator, ElementAccumulator,                                    \
    ElementC, LayoutC, AlignmentC,                                             \
    ElementC, LayoutC, AlignmentC,                                             \
    cutlass::epilogue::collective::EpilogueScheduleAuto                        \
>::CollectiveOp;                                                               \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<        \
    ArchTag, OperatorClass,                                                    \
    ElementA, LayoutA, AlignmentA,                                             \
    ElementB, LayoutB, AlignmentB,                                             \
    ElementAccumulator, TileShape, GridShape,                                  \
    cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,           \
    cutlass::gemm::collective::KernelScheduleAuto                              \
>::CollectiveOp;                                                               \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                             \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x16_s3,   64,128, 64,1,16,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x16_s4,   64,128, 64,1,16,1,4)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x16_s5,   64,128, 64,1,16,1,5)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x16_s6,   64,128, 64,1,16,1,6)
DEFINE_GEMM_PP_AUTO (PP_64x128x64_c1x16_auto,  64,128, 64,1,16,1)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x16_s3,  64,128,128,1,16,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x16_s4,  64,128,128,1,16,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x128x128_c1x16_auto, 64,128,128,1,16,1)

DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x16_s3,   64,128, 64,1,16,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x16_s4,   64,128, 64,1,16,1,4)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x16_s5,   64,128, 64,1,16,1,5)
DEFINE_GEMM_WS_AUTO (WS_64x128x64_c1x16_auto,  64,128, 64,1,16,1)
DEFINE_GEMM_WS_STAGE(WS_64x128x128_c1x16_s3,  64,128,128,1,16,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x128x128_c1x16_s4,  64,128,128,1,16,1,4)
DEFINE_GEMM_WS_AUTO (WS_64x128x128_c1x16_auto, 64,128,128,1,16,1)

DEFINE_GEMM_COOP_STAGE(COOP_128x128x64_c1x16_s3,  128,128, 64,1,16,1,3)
DEFINE_GEMM_COOP_STAGE(COOP_128x128x64_c1x16_s4,  128,128, 64,1,16,1,4)
DEFINE_GEMM_COOP_AUTO (COOP_128x128x64_c1x16_auto, 128,128, 64,1,16,1)
DEFINE_GEMM_COOP_STAGE(COOP_128x256x64_c1x16_s3,  128,256, 64,1,16,1,3)
DEFINE_GEMM_COOP_STAGE(COOP_128x256x64_c1x16_s4,  128,256, 64,1,16,1,4)
DEFINE_GEMM_COOP_AUTO (COOP_128x256x64_c1x16_auto, 128,256, 64,1,16,1)
DEFINE_GEMM_COOP_STAGE(COOP_128x128x64_c1x8_s3,   128,128, 64,1,8,1,3)
DEFINE_GEMM_COOP_STAGE(COOP_128x128x64_c1x8_s4,   128,128, 64,1,8,1,4)
DEFINE_GEMM_COOP_AUTO (COOP_128x128x64_c1x8_auto,  128,128, 64,1,8,1)
DEFINE_GEMM_COOP_STAGE(COOP_128x256x64_c1x8_s3,   128,256, 64,1,8,1,3)
DEFINE_GEMM_COOP_STAGE(COOP_128x256x64_c1x8_s4,   128,256, 64,1,8,1,4)
DEFINE_GEMM_COOP_AUTO (COOP_128x256x64_c1x8_auto,  128,256, 64,1,8,1)
DEFINE_GEMM_COOP_STAGE(COOP_128x128x128_c1x8_s3,  128,128,128,1,8,1,3)
DEFINE_GEMM_COOP_STAGE(COOP_128x128x128_c1x8_s4,  128,128,128,1,8,1,4)
DEFINE_GEMM_COOP_AUTO (COOP_128x128x128_c1x8_auto, 128,128,128,1,8,1)
DEFINE_GEMM_COOP_STAGE(COOP_128x128x64_c1x4_s3,   128,128, 64,1,4,1,3)
DEFINE_GEMM_COOP_STAGE(COOP_128x128x64_c1x4_s4,   128,128, 64,1,4,1,4)
DEFINE_GEMM_COOP_AUTO (COOP_128x128x64_c1x4_auto,  128,128, 64,1,4,1)
DEFINE_GEMM_COOP_STAGE(COOP_128x256x64_c1x4_s3,   128,256, 64,1,4,1,3)
DEFINE_GEMM_COOP_STAGE(COOP_128x256x64_c1x4_s4,   128,256, 64,1,4,1,4)
DEFINE_GEMM_COOP_AUTO (COOP_128x256x64_c1x4_auto,  128,256, 64,1,4,1)

DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x8_s2,   64,128, 64,1,8,1,2)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x8_s3,   64,128, 64,1,8,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x8_s4,   64,128, 64,1,8,1,4)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x8_s5,   64,128, 64,1,8,1,5)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x8_s6,   64,128, 64,1,8,1,6)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x8_s7,   64,128, 64,1,8,1,7)
DEFINE_GEMM_PP_AUTO (PP_64x128x64_c1x8_auto,  64,128, 64,1,8,1)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x8_s3,   64,256, 64,1,8,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x8_s4,   64,256, 64,1,8,1,4)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x8_s5,   64,256, 64,1,8,1,5)
DEFINE_GEMM_PP_AUTO (PP_64x256x64_c1x8_auto,  64,256, 64,1,8,1)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x8_s3,  64,128,128,1,8,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x8_s4,  64,128,128,1,8,1,4)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x8_s5,  64,128,128,1,8,1,5)
DEFINE_GEMM_PP_AUTO (PP_64x128x128_c1x8_auto, 64,128,128,1,8,1)
DEFINE_GEMM_PP_STAGE(PP_64x256x128_c1x8_s3,  64,256,128,1,8,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x256x128_c1x8_s4,  64,256,128,1,8,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x256x128_c1x8_auto, 64,256,128,1,8,1)

DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x8_s3,   64,128, 64,1,8,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x8_s4,   64,128, 64,1,8,1,4)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x8_s5,   64,128, 64,1,8,1,5)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x8_s6,   64,128, 64,1,8,1,6)
DEFINE_GEMM_WS_AUTO (WS_64x128x64_c1x8_auto,  64,128, 64,1,8,1)
DEFINE_GEMM_WS_STAGE(WS_64x256x64_c1x8_s3,   64,256, 64,1,8,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x256x64_c1x8_s4,   64,256, 64,1,8,1,4)
DEFINE_GEMM_WS_AUTO (WS_64x256x64_c1x8_auto,  64,256, 64,1,8,1)
DEFINE_GEMM_WS_STAGE(WS_64x128x128_c1x8_s3,  64,128,128,1,8,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x128x128_c1x8_s4,  64,128,128,1,8,1,4)
DEFINE_GEMM_WS_AUTO (WS_64x128x128_c1x8_auto, 64,128,128,1,8,1)

DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x4_s3,   64,128, 64,1,4,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x4_s4,   64,128, 64,1,4,1,4)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x4_s5,   64,128, 64,1,4,1,5)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x4_s6,   64,128, 64,1,4,1,6)
DEFINE_GEMM_PP_AUTO (PP_64x128x64_c1x4_auto,  64,128, 64,1,4,1)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x4_s3,   64,256, 64,1,4,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x4_s4,   64,256, 64,1,4,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x256x64_c1x4_auto,  64,256, 64,1,4,1)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x4_s3,  64,128,128,1,4,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x4_s4,  64,128,128,1,4,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x128x128_c1x4_auto, 64,128,128,1,4,1)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x4_s3,   64,128, 64,1,4,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x4_s4,   64,128, 64,1,4,1,4)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x4_s5,   64,128, 64,1,4,1,5)
DEFINE_GEMM_WS_AUTO (WS_64x128x64_c1x4_auto,  64,128, 64,1,4,1)
DEFINE_GEMM_WS_STAGE(WS_64x256x64_c1x4_s3,   64,256, 64,1,4,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x256x64_c1x4_s4,   64,256, 64,1,4,1,4)
DEFINE_GEMM_WS_AUTO (WS_64x256x64_c1x4_auto,  64,256, 64,1,4,1)
DEFINE_GEMM_WS_STAGE(WS_64x128x128_c1x4_s3,  64,128,128,1,4,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x128x128_c1x4_s4,  64,128,128,1,4,1,4)
DEFINE_GEMM_WS_AUTO (WS_64x128x128_c1x4_auto, 64,128,128,1,4,1)

DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x2_s3,   64,128, 64,1,2,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x2_s4,   64,128, 64,1,2,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x128x64_c1x2_auto,  64,128, 64,1,2,1)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x2_s3,   64,256, 64,1,2,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x2_s4,   64,256, 64,1,2,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x256x64_c1x2_auto,  64,256, 64,1,2,1)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x2_s3,  64,128,128,1,2,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x128_c1x2_s4,  64,128,128,1,2,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x128x128_c1x2_auto, 64,128,128,1,2,1)

DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x1_s3,   64,128, 64,1,1,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x1_s4,   64,128, 64,1,1,1,4)
DEFINE_GEMM_PP_STAGE(PP_64x128x64_c1x1_s5,   64,128, 64,1,1,1,5)
DEFINE_GEMM_PP_AUTO (PP_64x128x64_c1x1_auto,  64,128, 64,1,1,1)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x1_s3,   64,256, 64,1,1,1,3)
DEFINE_GEMM_PP_STAGE(PP_64x256x64_c1x1_s4,   64,256, 64,1,1,1,4)
DEFINE_GEMM_PP_AUTO (PP_64x256x64_c1x1_auto,  64,256, 64,1,1,1)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x1_s3,   64,128, 64,1,1,1,3)
DEFINE_GEMM_WS_STAGE(WS_64x128x64_c1x1_s4,   64,128, 64,1,1,1,4)
DEFINE_GEMM_WS_AUTO (WS_64x128x64_c1x1_auto,  64,128, 64,1,1,1)

DEFINE_GEMM_FALLBACK(FB_64x128x64_c1x8,   64,128, 64,1,8,1)
DEFINE_GEMM_FALLBACK(FB_64x256x64_c1x8,   64,256, 64,1,8,1)
DEFINE_GEMM_FALLBACK(FB_64x128x128_c1x8,  64,128,128,1,8,1)
DEFINE_GEMM_FALLBACK(FB_64x128x64_c1x4,   64,128, 64,1,4,1)
DEFINE_GEMM_FALLBACK(FB_64x128x64_c1x1,   64,128, 64,1,1,1)

struct GemmRunnerBase {
    bool can_implement_flag = false;
    virtual bool check_can_implement(int M, int N, int K,
                                     int device_id, uint8_t* ws, size_t ws_size) = 0;
    virtual bool run(int M, int N, int K,
                     ElementA* A, ElementB* B, ElementC* C, ElementC* D,
                     float alpha, float beta, int device_id, uint8_t* ws,
                     cudaStream_t stream = 0) = 0;
    virtual bool run_update(ElementA* A, ElementB* B, ElementC* C, ElementC* D,
                            cudaStream_t stream = 0) = 0;
    virtual bool is_cached() const = 0;
    virtual const char* name() const = 0;
    virtual ~GemmRunnerBase() = default;
};

template<typename GemmType>
struct TypedGemmRunner : GemmRunnerBase {
    GemmType gemm;
    const char* cfg_name;
    bool cached = false;
    typename GemmType::Arguments last_args;

    TypedGemmRunner(const char* n) : cfg_name(n) {}
    const char* name() const override { return cfg_name; }
    bool is_cached() const override { return cached; }

    static typename GemmType::Arguments make_args(
        int M, int N, int K, ElementA* A, ElementB* B, ElementC* C, ElementC* D,
        float alpha, float beta, int device_id)
    {
        using StrideA_ = typename GemmType::GemmKernel::StrideA;
        using StrideC_ = typename GemmType::GemmKernel::StrideC;
        using StrideD_ = typename GemmType::GemmKernel::StrideD;

        auto stride_A = cutlass::make_cute_packed_stride(StrideA_{}, cute::make_shape(M, K, 1));
        auto stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC_{}, cute::make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD_{}, cute::make_shape(M, N, 1));

        auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<
            typename GemmType::GemmKernel>(device_id);

        return typename GemmType::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {A, stride_A, B, stride_B},
            {{alpha, beta}, C, stride_C, D, stride_D},
            hw_info
        };
    }

    bool check_can_implement(int M, int N, int K,
                              int device_id, uint8_t* ws, size_t ws_size) override
    {
        auto args = make_args(M, N, K, nullptr, nullptr, nullptr, nullptr,
                               1.0f, 0.0f, device_id);
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
            can_implement_flag = false;
            return false;
        }
        size_t needed = GemmType::get_workspace_size(args);
        if (needed > ws_size) {
            can_implement_flag = false;
            return false;
        }
        can_implement_flag = true;
        return true;
    }

    bool run(int M, int N, int K,
             ElementA* A, ElementB* B, ElementC* C, ElementC* D,
             float alpha, float beta, int device_id, uint8_t* ws,
             cudaStream_t stream = 0) override
    {
        if (!can_implement_flag) return false;
        auto args = make_args(M, N, K, A, B, C, D, alpha, beta, device_id);
        if (gemm.initialize(args, ws, stream) != cutlass::Status::kSuccess) return false;
        if (gemm.run(stream) != cutlass::Status::kSuccess) return false;
        if (cudaGetLastError() != cudaSuccess) return false;
        last_args = args;
        cached = true;
        return true;
    }

    bool run_update(ElementA* A, ElementB* B, ElementC* C, ElementC* D,
                    cudaStream_t stream = 0) override
    {
        if (!cached) return false;
        last_args.mainloop.ptr_A = A;
        last_args.mainloop.ptr_B = B;
        last_args.epilogue.ptr_C = C;
        last_args.epilogue.ptr_D = D;
        if (gemm.update(last_args) != cutlass::Status::kSuccess) {
            cached = false;
            return false;
        }
        if (gemm.run(stream) != cutlass::Status::kSuccess) return false;
        return (cudaGetLastError() == cudaSuccess);
    }
};

static int      g_best_idx  = -2;
static const size_t WS_SIZE = 256ULL * 1024 * 1024;
static uint8_t* g_workspace = nullptr;
static int g_cached_M = 0, g_cached_N = 0, g_cached_K = 0;
static cudaStream_t g_stream = nullptr;

static void ensure_workspace() {
    if (!g_workspace) {
        cudaMalloc(&g_workspace, WS_SIZE);
    }
    if (!g_stream) {
        cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
    }
}

static std::vector<GemmRunnerBase*>& get_runners() {
    static std::vector<GemmRunnerBase*> runners;
    if (!runners.empty()) return runners;

#define ADD(cfg) runners.push_back(new TypedGemmRunner<cfg::Gemm>(#cfg))

    ADD(cfg_PP_64x128x64_c1x16_s4);
    ADD(cfg_PP_64x128x64_c1x16_s3);
    ADD(cfg_PP_64x128x64_c1x16_s5);
    ADD(cfg_PP_64x128x64_c1x16_s6);
    ADD(cfg_PP_64x128x64_c1x16_auto);
    ADD(cfg_PP_64x128x128_c1x16_s3);
    ADD(cfg_PP_64x128x128_c1x16_s4);
    ADD(cfg_PP_64x128x128_c1x16_auto);

    ADD(cfg_WS_64x128x64_c1x16_s3);
    ADD(cfg_WS_64x128x64_c1x16_s4);
    ADD(cfg_WS_64x128x64_c1x16_s5);
    ADD(cfg_WS_64x128x64_c1x16_auto);
    ADD(cfg_WS_64x128x128_c1x16_s3);
    ADD(cfg_WS_64x128x128_c1x16_s4);
    ADD(cfg_WS_64x128x128_c1x16_auto);

    ADD(cfg_COOP_128x128x64_c1x16_s4);
    ADD(cfg_COOP_128x128x64_c1x16_s3);
    ADD(cfg_COOP_128x128x64_c1x16_auto);
    ADD(cfg_COOP_128x256x64_c1x16_s3);
    ADD(cfg_COOP_128x256x64_c1x16_s4);
    ADD(cfg_COOP_128x256x64_c1x16_auto);
    ADD(cfg_COOP_128x128x64_c1x8_s3);
    ADD(cfg_COOP_128x128x64_c1x8_s4);
    ADD(cfg_COOP_128x128x64_c1x8_auto);
    ADD(cfg_COOP_128x256x64_c1x8_s3);
    ADD(cfg_COOP_128x256x64_c1x8_s4);
    ADD(cfg_COOP_128x256x64_c1x8_auto);
    ADD(cfg_COOP_128x128x128_c1x8_s3);
    ADD(cfg_COOP_128x128x128_c1x8_s4);
    ADD(cfg_COOP_128x128x128_c1x8_auto);
    ADD(cfg_COOP_128x128x64_c1x4_s3);
    ADD(cfg_COOP_128x128x64_c1x4_s4);
    ADD(cfg_COOP_128x128x64_c1x4_auto);
    ADD(cfg_COOP_128x256x64_c1x4_s3);
    ADD(cfg_COOP_128x256x64_c1x4_s4);
    ADD(cfg_COOP_128x256x64_c1x4_auto);

    ADD(cfg_PP_64x128x64_c1x8_s4);
    ADD(cfg_PP_64x128x64_c1x8_s3);
    ADD(cfg_PP_64x128x64_c1x8_s5);
    ADD(cfg_PP_64x128x64_c1x8_s6);
    ADD(cfg_PP_64x128x64_c1x8_s2);
    ADD(cfg_PP_64x128x64_c1x8_s7);
    ADD(cfg_PP_64x128x64_c1x8_auto);
    ADD(cfg_PP_64x256x64_c1x8_s3);
    ADD(cfg_PP_64x256x64_c1x8_s4);
    ADD(cfg_PP_64x256x64_c1x8_s5);
    ADD(cfg_PP_64x256x64_c1x8_auto);
    ADD(cfg_PP_64x128x128_c1x8_s3);
    ADD(cfg_PP_64x128x128_c1x8_s4);
    ADD(cfg_PP_64x128x128_c1x8_s5);
    ADD(cfg_PP_64x128x128_c1x8_auto);
    ADD(cfg_PP_64x256x128_c1x8_s3);
    ADD(cfg_PP_64x256x128_c1x8_s4);
    ADD(cfg_PP_64x256x128_c1x8_auto);

    ADD(cfg_WS_64x128x64_c1x8_s3);
    ADD(cfg_WS_64x128x64_c1x8_s4);
    ADD(cfg_WS_64x128x64_c1x8_s5);
    ADD(cfg_WS_64x128x64_c1x8_s6);
    ADD(cfg_WS_64x128x64_c1x8_auto);
    ADD(cfg_WS_64x256x64_c1x8_s3);
    ADD(cfg_WS_64x256x64_c1x8_s4);
    ADD(cfg_WS_64x256x64_c1x8_auto);
    ADD(cfg_WS_64x128x128_c1x8_s3);
    ADD(cfg_WS_64x128x128_c1x8_s4);
    ADD(cfg_WS_64x128x128_c1x8_auto);

    ADD(cfg_PP_64x128x64_c1x4_s3);
    ADD(cfg_PP_64x128x64_c1x4_s4);
    ADD(cfg_PP_64x128x64_c1x4_s5);
    ADD(cfg_PP_64x128x64_c1x4_s6);
    ADD(cfg_PP_64x128x64_c1x4_auto);
    ADD(cfg_PP_64x256x64_c1x4_s3);
    ADD(cfg_PP_64x256x64_c1x4_s4);
    ADD(cfg_PP_64x256x64_c1x4_auto);
    ADD(cfg_PP_64x128x128_c1x4_s3);
    ADD(cfg_PP_64x128x128_c1x4_s4);
    ADD(cfg_PP_64x128x128_c1x4_auto);
    ADD(cfg_WS_64x128x64_c1x4_s3);
    ADD(cfg_WS_64x128x64_c1x4_s4);
    ADD(cfg_WS_64x128x64_c1x4_s5);
    ADD(cfg_WS_64x128x64_c1x4_auto);
    ADD(cfg_WS_64x256x64_c1x4_s3);
    ADD(cfg_WS_64x256x64_c1x4_s4);
    ADD(cfg_WS_64x256x64_c1x4_auto);
    ADD(cfg_WS_64x128x128_c1x4_s3);
    ADD(cfg_WS_64x128x128_c1x4_s4);
    ADD(cfg_WS_64x128x128_c1x4_auto);

    ADD(cfg_PP_64x128x64_c1x2_s3);
    ADD(cfg_PP_64x128x64_c1x2_s4);
    ADD(cfg_PP_64x128x64_c1x2_auto);
    ADD(cfg_PP_64x256x64_c1x2_s3);
    ADD(cfg_PP_64x256x64_c1x2_s4);
    ADD(cfg_PP_64x256x64_c1x2_auto);
    ADD(cfg_PP_64x128x128_c1x2_s3);
    ADD(cfg_PP_64x128x128_c1x2_s4);
    ADD(cfg_PP_64x128x128_c1x2_auto);

    ADD(cfg_PP_64x128x64_c1x1_s3);
    ADD(cfg_PP_64x128x64_c1x1_s4);
    ADD(cfg_PP_64x128x64_c1x1_s5);
    ADD(cfg_PP_64x128x64_c1x1_auto);
    ADD(cfg_PP_64x256x64_c1x1_s3);
    ADD(cfg_PP_64x256x64_c1x1_s4);
    ADD(cfg_PP_64x256x64_c1x1_auto);
    ADD(cfg_WS_64x128x64_c1x1_s3);
    ADD(cfg_WS_64x128x64_c1x1_s4);
    ADD(cfg_WS_64x128x64_c1x1_auto);

    ADD(cfg_FB_64x128x64_c1x8);
    ADD(cfg_FB_64x256x64_c1x8);
    ADD(cfg_FB_64x128x128_c1x8);
    ADD(cfg_FB_64x128x64_c1x4);
    ADD(cfg_FB_64x128x64_c1x1);

#undef ADD
    return runners;
}

static int autotune(int M, int N, int K, ElementA* A, ElementB* B,
                    ElementC* C, ElementC* D, float alpha, float beta,
                    int device_id)
{
    ensure_workspace();
    auto& runners = get_runners();
    const int n = (int)runners.size();

    for (int i = 0; i < n; i++) {
        runners[i]->check_can_implement(M, N, K, device_id, g_workspace, WS_SIZE);
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    std::vector<std::pair<float,int>> p1;
    p1.reserve(n);

    for (int i = 0; i < n; i++) {
        if (!runners[i]->can_implement_flag) continue;

        bool ok = runners[i]->run(M, N, K, A, B, C, D, alpha, beta, device_id,
                                   g_workspace, g_stream);
        if (!ok) { runners[i]->can_implement_flag = false; continue; }

        for (int w = 0; w < 8; w++)
            runners[i]->run(M, N, K, A, B, C, D, alpha, beta, device_id,
                            g_workspace, g_stream);
        cudaStreamSynchronize(g_stream);

        cudaEventRecord(ev0, g_stream);
        for (int t = 0; t < 15; t++)
            runners[i]->run(M, N, K, A, B, C, D, alpha, beta, device_id,
                            g_workspace, g_stream);
        cudaEventRecord(ev1, g_stream);
        cudaEventSynchronize(ev1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        p1.push_back({ms / 15.f, i});
    }

    if (p1.empty()) {
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        return -1;
    }
    std::sort(p1.begin(), p1.end());
    int top8 = std::min((int)p1.size(), 8);

    std::vector<std::pair<float,int>> p2;
    p2.reserve(top8);

    for (int k = 0; k < top8; k++) {
        int i = p1[k].second;
        if (!runners[i]->can_implement_flag) continue;

        for (int w = 0; w < 20; w++)
            runners[i]->run(M, N, K, A, B, C, D, alpha, beta, device_id,
                            g_workspace, g_stream);
        cudaStreamSynchronize(g_stream);

        cudaEventRecord(ev0, g_stream);
        for (int t = 0; t < 60; t++)
            runners[i]->run(M, N, K, A, B, C, D, alpha, beta, device_id,
                            g_workspace, g_stream);
        cudaEventRecord(ev1, g_stream);
        cudaEventSynchronize(ev1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        p2.push_back({ms / 60.f, i});
    }

    if (p2.empty()) {
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        return p1[0].second;
    }
    std::sort(p2.begin(), p2.end());

    if ((int)p2.size() >= 2 && (p2[1].first / p2[0].first) > 1.02f) {
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        runners[p2[0].second]->run(M, N, K, A, B, C, D, alpha, beta,
                                    device_id, g_workspace, g_stream);
        cudaStreamSynchronize(g_stream);
        return p2[0].second;
    }

    int top4 = std::min((int)p2.size(), 4);
    float best_ms  = std::numeric_limits<float>::max();
    int   best_idx = p2[0].second;

    for (int k = 0; k < top4; k++) {
        int i = p2[k].second;
        if (!runners[i]->can_implement_flag) continue;

        for (int w = 0; w < 50; w++)
            runners[i]->run(M, N, K, A, B, C, D, alpha, beta, device_id,
                            g_workspace, g_stream);
        cudaStreamSynchronize(g_stream);

        cudaEventRecord(ev0, g_stream);
        for (int t = 0; t < 300; t++)
            runners[i]->run(M, N, K, A, B, C, D, alpha, beta, device_id,
                            g_workspace, g_stream);
        cudaEventRecord(ev1, g_stream);
        cudaEventSynchronize(ev1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        float avg = ms / 300.f;

        if (avg < best_ms) {
            best_ms  = avg;
            best_idx = i;
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    runners[best_idx]->run(M, N, K, A, B, C, D, alpha, beta,
                            device_id, g_workspace, g_stream);
    cudaStreamSynchronize(g_stream);
    return best_idx;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    auto* ptr_A = reinterpret_cast<ElementA*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
    auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());
    auto* ptr_D = reinterpret_cast<ElementC*>(c.data_ptr());

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    int device_id = 0;
    cudaGetDevice(&device_id);
    ensure_workspace();

    if (g_best_idx == -2 || M != g_cached_M || N != g_cached_N || K != g_cached_K) {
        g_best_idx = autotune(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                               alpha, beta, device_id);
        g_cached_M = M; g_cached_N = N; g_cached_K = K;
    }

    if (g_best_idx < 0) {
        throw std::runtime_error(
            "All CUTLASS GEMM configurations failed for M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }

    auto& runners = get_runners();

    if (runners[g_best_idx]->is_cached()) {
        bool ok = runners[g_best_idx]->run_update(ptr_A, ptr_B, ptr_C, ptr_D, g_stream);
        if (ok) return;
    }

    bool ok = runners[g_best_idx]->run(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                                        alpha, beta, device_id, g_workspace, g_stream);
    if (!ok) {
        g_best_idx = autotune(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                               alpha, beta, device_id);
        if (g_best_idx < 0) {
            throw std::runtime_error("All CUTLASS GEMM configurations failed after retry");
        }
        runners[g_best_idx]->run(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                                  alpha, beta, device_id, g_workspace, g_stream);
    }
}