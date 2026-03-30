#include <iostream>
#include <stdexcept>

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

#include <torch/extension.h>
#include <torch/types.h>

struct GpuWorkspace {
    void* ptr = nullptr;
    explicit GpuWorkspace(size_t n) {
        if (n > 0 && cudaMalloc(&ptr, n) != cudaSuccess) ptr = nullptr;
    }
    ~GpuWorkspace() { if (ptr) cudaFree(ptr); }
    GpuWorkspace(const GpuWorkspace&) = delete;
    GpuWorkspace& operator=(const GpuWorkspace&) = delete;
};

static cutlass::KernelHardwareInfo get_hw_info() {
    cutlass::KernelHardwareInfo hw;
    int dev = 0;
    cudaGetDevice(&dev);
    hw.device_id = dev;
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
    hw.sm_count = sm;
    return hw;
}

using ElemA   = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using ElemB   = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElemC   = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;
using ElemAcc = float;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElemA>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElemB>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElemC>::value;

#define DEFINE_PINGPONG_CFG(NS, TM, TN, TK, CM, CN, CK, STAGE_POLICY)         \
namespace NS {                                                                   \
using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;       \
using GroupShape   = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;       \
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;                 \
using ESched = cutlass::epilogue::TmaWarpSpecialized;                           \
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<     \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
    TileShape, GroupShape,                                                      \
    cutlass::epilogue::collective::EpilogueTileAuto,                            \
    ElemAcc, ElemAcc,                                                           \
    ElemC, LayoutC, AlignC,                                                     \
    ElemC, LayoutC, AlignC,                                                     \
    ESched>::CollectiveOp;                                                      \
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<         \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
    ElemA, LayoutA, AlignA,                                                     \
    ElemB, LayoutB, AlignB,                                                     \
    ElemAcc, TileShape, GroupShape,                                             \
    STAGE_POLICY,                                                               \
    KSched>::CollectiveOp;                                                      \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                        \
    cute::Shape<int,int,int>, Mainloop, Epilogue>;                              \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;          \
}

#define AUTO_STAGE(NS) \
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename NS::Epilogue::SharedStorage)>

DEFINE_PINGPONG_CFG(p1_auto,  64, 128,  64,  1, 1, 1,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cute::Shape<cute::_64,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_1,cute::_1>, cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc, ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp::SharedStorage)>)

namespace p1_s8 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<8>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_s7 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<7>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_s6 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<6>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_s5 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<5>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_s4 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_s3 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<3>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n2_auto {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n2_s6 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<6>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n2_s5 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<5>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n2_s4 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n4_auto {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_4,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n4_s5 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_4,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<5>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n4_s4 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_4,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n8_auto {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_8,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace p1_n8_s4 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_8,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace dk_auto {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_128>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace dk_s5 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_128>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<5>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace dk_s4 {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_128>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace dk_n2_auto {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_128>;
using GroupShape   = cute::Shape<cute::_1,  cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace dk_n4_auto {
using TileShape    = cute::Shape<cute::_64, cute::_128, cute::_128>;
using GroupShape   = cute::Shape<cute::_1,  cute::_4,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace wn_auto {
using TileShape    = cute::Shape<cute::_64, cute::_256, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace wn_s5 {
using TileShape    = cute::Shape<cute::_64, cute::_256, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<5>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace wn_s4 {
using TileShape    = cute::Shape<cute::_64, cute::_256, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape, cutlass::gemm::collective::StageCount<4>, KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace wn_n2_auto {
using TileShape    = cute::Shape<cute::_64, cute::_256, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace wn_n4_auto {
using TileShape    = cute::Shape<cute::_64, cute::_256, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,  cute::_4,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace wn_dk_auto {
using TileShape    = cute::Shape<cute::_64, cute::_256, cute::_128>;
using GroupShape   = cute::Shape<cute::_1,  cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace wn_dk_n2_auto {
using TileShape    = cute::Shape<cute::_64, cute::_256, cute::_128>;
using GroupShape   = cute::Shape<cute::_1,  cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace coop_m2_auto {
using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
using GroupShape   = cute::Shape<cute::_2,   cute::_1,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using ESched = cutlass::epilogue::TmaWarpSpecializedCooperative;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

namespace coop_n2_auto {
using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
using GroupShape   = cute::Shape<cute::_1,   cute::_2,   cute::_1>;
using KSched = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using ESched = cutlass::epilogue::TmaWarpSpecializedCooperative;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, GroupShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElemAcc, ElemAcc,
    ElemC, LayoutC, AlignC, ElemC, LayoutC, AlignC, ESched>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElemA, LayoutA, AlignA, ElemB, LayoutB, AlignB, ElemAcc,
    TileShape, GroupShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    KSched>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

template <typename GemmType>
bool try_run_gemm(int M, int N, int K,
    cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C,
    float alpha, float beta)
{
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto hw = get_hw_info();
    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, sA, B, sB},
        {{alpha, beta}, C, sC, C, sD},
        hw
    };

    GemmType gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws_sz = GemmType::get_workspace_size(args);
    GpuWorkspace ws(ws_sz);
    if (ws_sz > 0 && !ws.ptr) return false;
    if (gemm.initialize(args, ws.ptr) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    auto* pA = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* pB = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* pC = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
    const float alpha = 1.f, beta = 0.f;

    if (try_run_gemm<p1_auto::Gemm>  (M, N, K, pA, pB, pC, alpha, beta)) return;

    if (try_run_gemm<p1_s8::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_s7::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_s6::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_s5::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_s4::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_s3::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;

    if (try_run_gemm<p1_n2_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_n2_s6::Gemm> (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_n2_s5::Gemm> (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_n2_s4::Gemm> (M, N, K, pA, pB, pC, alpha, beta)) return;

    if (try_run_gemm<p1_n4_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_n4_s5::Gemm> (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_n4_s4::Gemm> (M, N, K, pA, pB, pC, alpha, beta)) return;

    if (try_run_gemm<p1_n8_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<p1_n8_s4::Gemm> (M, N, K, pA, pB, pC, alpha, beta)) return;

    if (try_run_gemm<dk_auto::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<dk_s5::Gemm>    (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<dk_s4::Gemm>    (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<dk_n2_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<dk_n4_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;

    if (try_run_gemm<wn_auto::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<wn_s5::Gemm>    (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<wn_s4::Gemm>    (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<wn_n2_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<wn_n4_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<wn_dk_auto::Gemm>   (M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<wn_dk_n2_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;

    if (try_run_gemm<coop_m2_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;
    if (try_run_gemm<coop_n2_auto::Gemm>(M, N, K, pA, pB, pC, alpha, beta)) return;

    throw std::runtime_error("All CUTLASS GEMM configurations failed");
}