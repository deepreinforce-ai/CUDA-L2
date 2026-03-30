#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

using ElemA   = cutlass::half_t;
using ElemB   = cutlass::half_t;
using ElemC   = cutlass::half_t;
using ElemD   = cutlass::half_t;
using ElemAcc = float;
using ElemCmp = float;

using LayoutRow = cutlass::layout::RowMajor;
using LayoutCol = cutlass::layout::ColumnMajor;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElemA>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElemB>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElemC>::value;
static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElemD>::value;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElemD, ElemCmp, ElemC, ElemCmp,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEF_FWD_ROW(Name, TM, TN, TK, CM, CN, CK, MLoop, EpiP, Sched)        \
struct Name {                                                                   \
    static constexpr bool is_transposed = false;                               \
    static constexpr bool b_col_major   = false;                               \
    using TileShape  = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
    using GridShape  = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>; \
    using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                   \
        TileShape, GridShape,                                                   \
        cutlass::epilogue::collective::EpilogueTileAuto,                       \
        ElemAcc, ElemCmp, ElemC, LayoutRow, AlignC, ElemD, LayoutRow, AlignD, \
        EpiP, EpilogueOp>::CollectiveOp;                                       \
    using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<   \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                   \
        ElemA, LayoutRow, AlignA, ElemB, LayoutRow, AlignB, ElemAcc,          \
        TileShape, GridShape,                                                   \
        cutlass::gemm::collective::StageCountAutoCarveout<                     \
            static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,       \
        MLoop>::CollectiveOp;                                                  \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                   \
        cute::Shape<int,int,int>, MainStage, EpiStage, Sched>;                 \
    using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;   \
    using StrideA = typename Gemm::GemmKernel::StrideA;                        \
    using StrideB = typename Gemm::GemmKernel::StrideB;                        \
    using StrideC = typename Gemm::GemmKernel::StrideC;                        \
    using StrideD = typename Gemm::GemmKernel::StrideD;                        \
};

#define DEF_FWD(Name, TM, TN, TK, CM, CN, CK, MLoop, EpiP, Sched)            \
struct Name {                                                                   \
    static constexpr bool b_col_major = true;                                  \
    using TileShape  = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
    using GridShape  = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>; \
    using EpiStage = typename cutlass::epilogue::collective::CollectiveBuilder< \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                   \
        TileShape, GridShape,                                                   \
        cutlass::epilogue::collective::EpilogueTileAuto,                       \
        ElemAcc, ElemCmp, ElemC, LayoutRow, AlignC, ElemD, LayoutRow, AlignD, \
        EpiP, EpilogueOp>::CollectiveOp;                                       \
    using MainStage = typename cutlass::gemm::collective::CollectiveBuilder<   \
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                   \
        ElemA, LayoutRow, AlignA, ElemB, LayoutCol, AlignB, ElemAcc,          \
        TileShape, GridShape,                                                   \
        cutlass::gemm::collective::StageCountAutoCarveout<                     \
            static_cast<int>(sizeof(typename EpiStage::SharedStorage))>,       \
        MLoop>::CollectiveOp;                                                  \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                   \
        cute::Shape<int,int,int>, MainStage, EpiStage, Sched>;                 \
    using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;   \
    using StrideA = typename Gemm::GemmKernel::StrideA;                        \
    using StrideB = typename Gemm::GemmKernel::StrideB;                        \
    using StrideC = typename Gemm::GemmKernel::StrideC;                        \
    using StrideD = typename Gemm::GemmKernel::StrideD;                        \
};

DEF_FWD(PP_64x128x64_C1, 64, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x64_C2, 64, 128, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x64_C4, 64, 128, 64, 4, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x32_C1, 64, 128, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x32_C2, 64, 128, 32, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x16_C1, 64, 128, 16, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x16_C2, 64, 128, 16, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x128_C1, 64, 128, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x128x128_C2, 64, 128, 128, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_64x128x64_C1_R, 64, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_64x128x64_C2_R, 64, 128, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_64x128x32_C1_R, 64, 128, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_64x128x128_C1_R, 64, 128, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x64x64_C1, 64, 64, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x64x64_C2, 64, 64, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x64x32_C1, 64, 64, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x64x32_C2, 64, 64, 32, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_64x64x128_C1, 64, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_64x64x64_C1_R, 64, 64, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_64x64x64_C2_R, 64, 64, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_64x64x32_C1_R, 64, 64, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(SK_128x128x64_C1, 128, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_FWD(SK_128x128x64_C2, 128, 128, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_FWD(SK_128x128x32_C1, 128, 128, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_FWD(SK_128x128x32_C2, 128, 128, 32, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_FWD(SK_128x128x128_C1, 128, 128, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_FWD_ROW(SK_128x128x64_C1_R, 128, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_FWD_ROW(SK_128x128x32_C1_R, 128, 128, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

DEF_FWD(PP_128x128x64_C1, 128, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_128x128x64_C2, 128, 128, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_128x128x32_C1, 128, 128, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_128x128x32_C2, 128, 128, 32, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(PP_128x128x128_C1, 128, 128, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_128x128x64_C1_R, 128, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD_ROW(PP_128x128x32_C1_R, 128, 128, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(Coop_128x128x64_C1, 128, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(Coop_128x128x64_C2, 128, 128, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(Coop_128x128x32_C1, 128, 128, 32, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(Coop_128x128x32_C2, 128, 128, 32, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(Coop_128x128x128_C1, 128, 128, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(WS_64x128x64_C1, 64, 128, 64, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecialized,
    cutlass::epilogue::NoSmemWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

DEF_FWD(WS_128x128x64_C2, 128, 128, 64, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecialized,
    cutlass::epilogue::NoSmemWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

static uint8_t* g_workspace_ptr   = nullptr;
static size_t   g_workspace_bytes = 0;
static cudaStream_t g_stream      = nullptr;

static uint8_t* get_workspace(size_t needed) {
    if (needed > g_workspace_bytes) {
        if (g_workspace_ptr) { cudaFree(g_workspace_ptr); g_workspace_ptr = nullptr; }
        size_t alloc = (needed < (256ULL << 20)) ? (256ULL << 20) : needed;
        cudaError_t err = cudaMalloc(&g_workspace_ptr, alloc);
        g_workspace_bytes = (err == cudaSuccess) ? alloc : 0;
        if (err != cudaSuccess) g_workspace_ptr = nullptr;
    }
    return g_workspace_ptr;
}

static void ensure_stream() {
    if (!g_stream) {
        cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
    }
}

template <typename Strategy>
bool try_launch(const ElemA* pA, const ElemB* pB_col, const ElemB* pB_row,
                ElemC* pC, int M, int N, int K, int sm_count, int dev_id,
                cudaStream_t stream)
{
    using Gemm    = typename Strategy::Gemm;
    using StrideA = typename Strategy::StrideA;
    using StrideB = typename Strategy::StrideB;
    using StrideC = typename Strategy::StrideC;
    using StrideD = typename Strategy::StrideD;

    const ElemB* pB = Strategy::b_col_major ? pB_col : pB_row;

    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));

    StrideB sB;
    if constexpr (Strategy::b_col_major) {
        sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    } else {
        sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(K, N, 1));
    }

    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev_id;
    hw.sm_count  = sm_count;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {const_cast<ElemA*>(pA), sA, const_cast<ElemB*>(pB), sB},
        {{1.0f, 0.0f}, const_cast<ElemC*>(pC), sC, pC, sD},
        hw};

    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

    size_t ws_size = Gemm::get_workspace_size(args);
    uint8_t* ws = get_workspace(ws_size > 0 ? ws_size : 1);
    if (!ws && ws_size > 0) return false;

    if (gemm.initialize(args, ws, stream) != cutlass::Status::kSuccess) return false;

    cutlass::Status st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) return false;

    return true;
}

using LaunchFn = bool(*)(const ElemA*, const ElemB*, const ElemB*, ElemC*,
                          int, int, int, int, int, cudaStream_t);

static LaunchFn g_launch_fns[] = {
    try_launch<PP_64x128x64_C1>,
    try_launch<PP_64x128x64_C2>,
    try_launch<PP_64x128x64_C4>,
    try_launch<PP_64x128x128_C1>,
    try_launch<PP_64x128x128_C2>,
    try_launch<PP_64x128x32_C1>,
    try_launch<PP_64x128x32_C2>,
    try_launch<PP_64x128x16_C1>,
    try_launch<PP_64x128x16_C2>,
    try_launch<PP_64x128x64_C1_R>,
    try_launch<PP_64x128x64_C2_R>,
    try_launch<PP_64x128x128_C1_R>,
    try_launch<PP_64x128x32_C1_R>,
    try_launch<PP_64x64x64_C1>,
    try_launch<PP_64x64x64_C2>,
    try_launch<PP_64x64x128_C1>,
    try_launch<PP_64x64x32_C1>,
    try_launch<PP_64x64x32_C2>,
    try_launch<PP_64x64x64_C1_R>,
    try_launch<PP_64x64x64_C2_R>,
    try_launch<PP_64x64x32_C1_R>,
    try_launch<SK_128x128x64_C1>,
    try_launch<SK_128x128x64_C2>,
    try_launch<SK_128x128x128_C1>,
    try_launch<SK_128x128x32_C1>,
    try_launch<SK_128x128x32_C2>,
    try_launch<SK_128x128x64_C1_R>,
    try_launch<SK_128x128x32_C1_R>,
    try_launch<PP_128x128x64_C1>,
    try_launch<PP_128x128x64_C2>,
    try_launch<PP_128x128x128_C1>,
    try_launch<PP_128x128x32_C1>,
    try_launch<PP_128x128x32_C2>,
    try_launch<PP_128x128x64_C1_R>,
    try_launch<PP_128x128x32_C1_R>,
    try_launch<Coop_128x128x64_C1>,
    try_launch<Coop_128x128x64_C2>,
    try_launch<Coop_128x128x128_C1>,
    try_launch<Coop_128x128x32_C1>,
    try_launch<Coop_128x128x32_C2>,
    try_launch<WS_64x128x64_C1>,
    try_launch<WS_128x128x64_C2>,
};
static constexpr int NUM_STRATEGIES = 42;
static int g_cached_strategy = -1;

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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    const ElemA* pA     = reinterpret_cast<const ElemA*>(a.data_ptr());
    const ElemB* pB_col = reinterpret_cast<const ElemB*>(b_col_major.data_ptr());
    const ElemB* pB_row = reinterpret_cast<const ElemB*>(b.data_ptr());
    ElemC*       pC     = reinterpret_cast<ElemC*>(c.data_ptr());

    int dev_id = 0;
    cudaGetDevice(&dev_id);
    int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev_id);

    ensure_stream();
    cudaStream_t stream = g_stream;

    if (g_cached_strategy >= 0) {
        bool ok = g_launch_fns[g_cached_strategy](
            pA, pB_col, pB_row, pC, M, N, K, sm_count, dev_id, stream);
        if (ok) {
            return;
        }
        g_cached_strategy = -1;
        cudaGetLastError();
    }

    for (int i = 0; i < NUM_STRATEGIES; i++) {
        cudaGetLastError();
        if (g_launch_fns[i](pA, pB_col, pB_row, pC, M, N, K, sm_count, dev_id, stream)) {
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err == cudaSuccess) {
                err = cudaGetLastError();
            }
            if (err == cudaSuccess) {
                g_cached_strategy = i;
                return;
            }
            cudaGetLastError();
        }
    }

    throw std::runtime_error(
        std::string("All HGEMM strategies failed for M=") + std::to_string(M) +
        " N=" + std::to_string(N) + " K=" + std::to_string(K));

#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported (need SM90a / H100)");
#endif
}