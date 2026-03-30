#include <iostream>
#include <mutex>
#include <cstring>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
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
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElemD, ElemCmp, ElemC, ElemCmp,
    cutlass::FloatRoundStyle::round_to_nearest>;
static constexpr int Align = 8;

using Tile128x256x64 = cute::Shape<cute::_128, cute::_256, cute::_64>;
using Tile128x128x64 = cute::Shape<cute::_128, cute::_128, cute::_64>;
using Tile256x128x64 = cute::Shape<cute::_256, cute::_128, cute::_64>;
using Tile64x256x64  = cute::Shape<cute::_64,  cute::_256, cute::_64>;
using Tile64x128x64  = cute::Shape<cute::_64,  cute::_128, cute::_64>;
using Tile128x64x64  = cute::Shape<cute::_128, cute::_64,  cute::_64>;
using Tile256x64x64  = cute::Shape<cute::_256, cute::_64,  cute::_64>;

using GS1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using GS1x2 = cute::Shape<cute::_1, cute::_2, cute::_1>;
using GS1x4 = cute::Shape<cute::_1, cute::_4, cute::_1>;
using GS1x8 = cute::Shape<cute::_1, cute::_8, cute::_1>;
using GS2x1 = cute::Shape<cute::_2, cute::_1, cute::_1>;
using GS4x1 = cute::Shape<cute::_4, cute::_1, cute::_1>;
using GS2x2 = cute::Shape<cute::_2, cute::_2, cute::_1>;
using GS2x4 = cute::Shape<cute::_2, cute::_4, cute::_1>;

using EpiTileAuto = cutlass::epilogue::collective::EpilogueTileAuto;

#define DEF_GEMM_CP_AUTO(IDX, TILE, GRID_SZ)                                    \
using CollEpi_##IDX = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    TILE, GRID_SZ, EpiTileAuto,                                                   \
    ElemAcc, ElemCmp, ElemC, LayoutC, Align, ElemD, LayoutD, Align,             \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##IDX = typename cutlass::gemm::collective::CollectiveBuilder<   \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    ElemA, LayoutA, Align, ElemB, LayoutB, Align, ElemAcc,                      \
    TILE, GRID_SZ,                                                                \
    cutlass::gemm::collective::StageCountAutoCarveout<                           \
      static_cast<int>(sizeof(typename CollEpi_##IDX::SharedStorage))>,         \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;           \
using GemmKernel_##IDX = cutlass::gemm::kernel::GemmUniversal<                  \
    cute::Shape<int,int,int>, MainStage_##IDX, CollEpi_##IDX,                   \
    cutlass::gemm::PersistentScheduler>;                                         \
using Gemm_##IDX    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##IDX>;\
using StrideA_##IDX = typename Gemm_##IDX::GemmKernel::StrideA;                 \
using StrideB_##IDX = typename Gemm_##IDX::GemmKernel::StrideB;                 \
using StrideC_##IDX = typename Gemm_##IDX::GemmKernel::StrideC;                 \
using StrideD_##IDX = typename Gemm_##IDX::GemmKernel::StrideD;

#define DEF_GEMM_CP_STAGE(IDX, TILE, GRID_SZ, STAGES)                           \
using CollEpi_##IDX = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    TILE, GRID_SZ, EpiTileAuto,                                                   \
    ElemAcc, ElemCmp, ElemC, LayoutC, Align, ElemD, LayoutD, Align,             \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##IDX = typename cutlass::gemm::collective::CollectiveBuilder<   \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    ElemA, LayoutA, Align, ElemB, LayoutB, Align, ElemAcc,                      \
    TILE, GRID_SZ,                                                                \
    cutlass::gemm::collective::StageCount<STAGES>,                               \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;           \
using GemmKernel_##IDX = cutlass::gemm::kernel::GemmUniversal<                  \
    cute::Shape<int,int,int>, MainStage_##IDX, CollEpi_##IDX,                   \
    cutlass::gemm::PersistentScheduler>;                                         \
using Gemm_##IDX    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##IDX>;\
using StrideA_##IDX = typename Gemm_##IDX::GemmKernel::StrideA;                 \
using StrideB_##IDX = typename Gemm_##IDX::GemmKernel::StrideB;                 \
using StrideC_##IDX = typename Gemm_##IDX::GemmKernel::StrideC;                 \
using StrideD_##IDX = typename Gemm_##IDX::GemmKernel::StrideD;

#define DEF_GEMM_PP_AUTO(IDX, TILE, GRID_SZ)                                    \
using CollEpi_##IDX = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    TILE, GRID_SZ, EpiTileAuto,                                                   \
    ElemAcc, ElemCmp, ElemC, LayoutC, Align, ElemD, LayoutD, Align,             \
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;            \
using MainStage_##IDX = typename cutlass::gemm::collective::CollectiveBuilder<   \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    ElemA, LayoutA, Align, ElemB, LayoutB, Align, ElemAcc,                      \
    TILE, GRID_SZ,                                                                \
    cutlass::gemm::collective::StageCountAutoCarveout<                           \
      static_cast<int>(sizeof(typename CollEpi_##IDX::SharedStorage))>,         \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;              \
using GemmKernel_##IDX = cutlass::gemm::kernel::GemmUniversal<                  \
    cute::Shape<int,int,int>, MainStage_##IDX, CollEpi_##IDX,                   \
    cutlass::gemm::PersistentScheduler>;                                         \
using Gemm_##IDX    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##IDX>;\
using StrideA_##IDX = typename Gemm_##IDX::GemmKernel::StrideA;                 \
using StrideB_##IDX = typename Gemm_##IDX::GemmKernel::StrideB;                 \
using StrideC_##IDX = typename Gemm_##IDX::GemmKernel::StrideC;                 \
using StrideD_##IDX = typename Gemm_##IDX::GemmKernel::StrideD;

#define DEF_GEMM_PP_STAGE(IDX, TILE, GRID_SZ, STAGES)                           \
using CollEpi_##IDX = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    TILE, GRID_SZ, EpiTileAuto,                                                   \
    ElemAcc, ElemCmp, ElemC, LayoutC, Align, ElemD, LayoutD, Align,             \
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;            \
using MainStage_##IDX = typename cutlass::gemm::collective::CollectiveBuilder<   \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    ElemA, LayoutA, Align, ElemB, LayoutB, Align, ElemAcc,                      \
    TILE, GRID_SZ,                                                                \
    cutlass::gemm::collective::StageCount<STAGES>,                               \
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;              \
using GemmKernel_##IDX = cutlass::gemm::kernel::GemmUniversal<                  \
    cute::Shape<int,int,int>, MainStage_##IDX, CollEpi_##IDX,                   \
    cutlass::gemm::PersistentScheduler>;                                         \
using Gemm_##IDX    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##IDX>;\
using StrideA_##IDX = typename Gemm_##IDX::GemmKernel::StrideA;                 \
using StrideB_##IDX = typename Gemm_##IDX::GemmKernel::StrideB;                 \
using StrideC_##IDX = typename Gemm_##IDX::GemmKernel::StrideC;                 \
using StrideD_##IDX = typename Gemm_##IDX::GemmKernel::StrideD;

#define DEF_GEMM_CSK_AUTO(IDX, TILE, GRID_SZ)                                   \
using CollEpi_##IDX = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    TILE, GRID_SZ, EpiTileAuto,                                                   \
    ElemAcc, ElemCmp, ElemC, LayoutC, Align, ElemD, LayoutD, Align,             \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##IDX = typename cutlass::gemm::collective::CollectiveBuilder<   \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    ElemA, LayoutA, Align, ElemB, LayoutB, Align, ElemAcc,                      \
    TILE, GRID_SZ,                                                                \
    cutlass::gemm::collective::StageCountAutoCarveout<                           \
      static_cast<int>(sizeof(typename CollEpi_##IDX::SharedStorage))>,         \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;           \
using GemmKernel_##IDX = cutlass::gemm::kernel::GemmUniversal<                  \
    cute::Shape<int,int,int>, MainStage_##IDX, CollEpi_##IDX,                   \
    cutlass::gemm::StreamKScheduler>;                                            \
using Gemm_##IDX    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##IDX>;\
using StrideA_##IDX = typename Gemm_##IDX::GemmKernel::StrideA;                 \
using StrideB_##IDX = typename Gemm_##IDX::GemmKernel::StrideB;                 \
using StrideC_##IDX = typename Gemm_##IDX::GemmKernel::StrideC;                 \
using StrideD_##IDX = typename Gemm_##IDX::GemmKernel::StrideD;

#define DEF_GEMM_CSK_STAGE(IDX, TILE, GRID_SZ, STAGES)                          \
using CollEpi_##IDX = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    TILE, GRID_SZ, EpiTileAuto,                                                   \
    ElemAcc, ElemCmp, ElemC, LayoutC, Align, ElemD, LayoutD, Align,             \
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
using MainStage_##IDX = typename cutlass::gemm::collective::CollectiveBuilder<   \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
    ElemA, LayoutA, Align, ElemB, LayoutB, Align, ElemAcc,                      \
    TILE, GRID_SZ,                                                                \
    cutlass::gemm::collective::StageCount<STAGES>,                               \
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;           \
using GemmKernel_##IDX = cutlass::gemm::kernel::GemmUniversal<                  \
    cute::Shape<int,int,int>, MainStage_##IDX, CollEpi_##IDX,                   \
    cutlass::gemm::StreamKScheduler>;                                            \
using Gemm_##IDX    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_##IDX>;\
using StrideA_##IDX = typename Gemm_##IDX::GemmKernel::StrideA;                 \
using StrideB_##IDX = typename Gemm_##IDX::GemmKernel::StrideB;                 \
using StrideC_##IDX = typename Gemm_##IDX::GemmKernel::StrideC;                 \
using StrideD_##IDX = typename Gemm_##IDX::GemmKernel::StrideD;

DEF_GEMM_CP_AUTO (0,  Tile128x256x64, GS1x2)
DEF_GEMM_CP_STAGE(1,  Tile128x256x64, GS1x2, 2)
DEF_GEMM_CP_STAGE(2,  Tile128x256x64, GS1x2, 3)
DEF_GEMM_CP_STAGE(3,  Tile128x256x64, GS1x2, 4)
DEF_GEMM_CP_STAGE(4,  Tile128x256x64, GS1x2, 5)
DEF_GEMM_CP_STAGE(5,  Tile128x256x64, GS1x2, 6)
DEF_GEMM_CP_STAGE(6,  Tile128x256x64, GS1x2, 7)
DEF_GEMM_PP_AUTO (7,  Tile128x256x64, GS1x2)
DEF_GEMM_PP_STAGE(8,  Tile128x256x64, GS1x2, 3)
DEF_GEMM_PP_STAGE(9,  Tile128x256x64, GS1x2, 4)
DEF_GEMM_CSK_AUTO(10, Tile128x256x64, GS1x2)
DEF_GEMM_CSK_STAGE(11, Tile128x256x64, GS1x2, 3)
DEF_GEMM_CSK_STAGE(12, Tile128x256x64, GS1x2, 4)

DEF_GEMM_CP_AUTO (13, Tile128x256x64, GS1x4)
DEF_GEMM_CP_STAGE(14, Tile128x256x64, GS1x4, 3)
DEF_GEMM_CP_STAGE(15, Tile128x256x64, GS1x4, 4)
DEF_GEMM_CP_STAGE(16, Tile128x256x64, GS1x4, 5)
DEF_GEMM_PP_AUTO (17, Tile128x256x64, GS1x4)
DEF_GEMM_PP_STAGE(18, Tile128x256x64, GS1x4, 3)
DEF_GEMM_PP_STAGE(19, Tile128x256x64, GS1x4, 4)
DEF_GEMM_CSK_AUTO(20, Tile128x256x64, GS1x4)

DEF_GEMM_CP_AUTO (21, Tile128x256x64, GS2x2)
DEF_GEMM_CP_STAGE(22, Tile128x256x64, GS2x2, 3)
DEF_GEMM_CP_STAGE(23, Tile128x256x64, GS2x2, 4)
DEF_GEMM_PP_AUTO (24, Tile128x256x64, GS2x2)
DEF_GEMM_CP_AUTO (25, Tile128x256x64, GS2x4)
DEF_GEMM_CP_AUTO (26, Tile128x256x64, GS1x8)
DEF_GEMM_PP_AUTO (27, Tile128x256x64, GS1x8)

DEF_GEMM_CP_AUTO (28, Tile256x128x64, GS1x2)
DEF_GEMM_CP_AUTO (29, Tile256x128x64, GS2x1)
DEF_GEMM_CP_AUTO (30, Tile256x128x64, GS1x4)
DEF_GEMM_CP_AUTO (31, Tile256x128x64, GS2x2)
DEF_GEMM_CP_STAGE(32, Tile256x128x64, GS1x2, 3)
DEF_GEMM_CP_STAGE(33, Tile256x128x64, GS1x2, 4)
DEF_GEMM_PP_AUTO (34, Tile256x128x64, GS1x2)
DEF_GEMM_PP_AUTO (35, Tile256x128x64, GS2x1)

DEF_GEMM_CP_AUTO (36, Tile128x128x64, GS1x2)
DEF_GEMM_CP_AUTO (37, Tile128x128x64, GS1x4)
DEF_GEMM_CP_AUTO (38, Tile128x128x64, GS2x2)
DEF_GEMM_CP_STAGE(39, Tile128x128x64, GS1x4, 4)
DEF_GEMM_PP_AUTO (40, Tile128x128x64, GS1x4)

DEF_GEMM_PP_AUTO (41, Tile64x256x64, GS1x4)
DEF_GEMM_PP_AUTO (42, Tile64x256x64, GS1x8)
DEF_GEMM_PP_STAGE(43, Tile64x256x64, GS1x4, 3)
DEF_GEMM_PP_STAGE(44, Tile64x256x64, GS1x4, 4)

DEF_GEMM_CP_AUTO (45, Tile128x256x64, GS2x1)
DEF_GEMM_CP_AUTO (46, Tile128x256x64, GS4x1)

DEF_GEMM_CP_STAGE(47, Tile128x256x64, GS1x2, 8)
DEF_GEMM_CP_STAGE(48, Tile128x256x64, GS1x2, 10)
DEF_GEMM_CP_STAGE(49, Tile128x256x64, GS1x2, 12)
DEF_GEMM_PP_STAGE(50, Tile128x256x64, GS1x2, 5)
DEF_GEMM_PP_STAGE(51, Tile128x256x64, GS1x2, 6)
DEF_GEMM_PP_STAGE(52, Tile128x256x64, GS1x2, 8)

DEF_GEMM_CP_STAGE(53, Tile128x256x64, GS1x4, 6)
DEF_GEMM_CP_STAGE(54, Tile128x256x64, GS1x4, 7)
DEF_GEMM_CP_STAGE(55, Tile128x256x64, GS1x4, 8)
DEF_GEMM_PP_STAGE(56, Tile128x256x64, GS1x4, 5)
DEF_GEMM_PP_STAGE(57, Tile128x256x64, GS1x4, 6)

DEF_GEMM_CP_STAGE(58, Tile256x128x64, GS1x2, 5)
DEF_GEMM_CP_STAGE(59, Tile256x128x64, GS1x2, 6)
DEF_GEMM_CP_STAGE(60, Tile256x128x64, GS2x1, 3)
DEF_GEMM_CP_STAGE(61, Tile256x128x64, GS2x1, 4)

DEF_GEMM_CP_AUTO (62, Tile128x64x64, GS1x2)
DEF_GEMM_CP_AUTO (63, Tile128x64x64, GS1x4)
DEF_GEMM_CP_STAGE(64, Tile128x64x64, GS1x4, 4)
DEF_GEMM_CP_STAGE(65, Tile128x64x64, GS1x4, 6)
DEF_GEMM_PP_AUTO (66, Tile128x64x64, GS1x4)

DEF_GEMM_CP_AUTO (67, Tile256x64x64, GS1x2)
DEF_GEMM_CP_AUTO (68, Tile256x64x64, GS1x4)
DEF_GEMM_CP_STAGE(69, Tile256x64x64, GS1x2, 4)
DEF_GEMM_PP_AUTO (70, Tile256x64x64, GS1x2)

DEF_GEMM_CSK_AUTO(71, Tile128x256x64, GS1x4)
DEF_GEMM_CSK_STAGE(72, Tile128x256x64, GS1x4, 4)
DEF_GEMM_CSK_AUTO(73, Tile256x128x64, GS1x2)

static constexpr int NUM_CFGS = 74;

template<typename G, typename SA, typename SB, typename SC, typename SD>
struct GemmCached {
  G        gemm;
  uint8_t* workspace  = nullptr;
  size_t   ws_size    = 0;
  SA sA; SB sB; SC sC; SD sD;
  cutlass::KernelHardwareInfo hw;
  cudaStream_t stream  = nullptr;
  const half* last_pA  = nullptr;
  const half* last_pB  = nullptr;
  half*       last_pC  = nullptr;

  ~GemmCached() {
    if (workspace) { cudaFree(workspace); workspace = nullptr; }
    if (stream)    { cudaStreamDestroy(stream); stream = nullptr; }
  }

  bool try_init(const half* pA, const half* pB, half* pC, int M, int N, int K,
                const cutlass::KernelHardwareInfo& hwinfo) {
    hw = hwinfo;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;

    sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

    auto* A = reinterpret_cast<ElemA*>(const_cast<half*>(pA));
    auto* B = reinterpret_cast<ElemB*>(const_cast<half*>(pB));
    auto* C = reinterpret_cast<ElemC*>(pC);
    auto* D = reinterpret_cast<ElemD*>(pC);

    typename G::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {A, sA, B, sB},
      {{1.0f, 0.0f}, C, sC, D, sD},
      hw
    };

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    ws_size = G::get_workspace_size(args);
    if (workspace) { cudaFree(workspace); workspace = nullptr; }
    if (cudaMalloc(&workspace, std::max(ws_size, size_t(1))) != cudaSuccess) return false;
    if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
      cudaFree(workspace); workspace = nullptr; return false;
    }
    last_pA = pA; last_pB = pB; last_pC = pC;
    return true;
  }

  float benchmark(const half* pA, const half* pB, half* pC, int M, int N, int K,
                  int warmup = 10, int nruns = 40) {
    auto* A = reinterpret_cast<ElemA*>(const_cast<half*>(pA));
    auto* B = reinterpret_cast<ElemB*>(const_cast<half*>(pB));
    auto* C = reinterpret_cast<ElemC*>(pC);
    auto* D = reinterpret_cast<ElemD*>(pC);
    typename G::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {A, sA, B, sB},
      {{1.0f, 0.0f}, C, sC, D, sD},
      hw
    };
    if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return 1e9f;
    for (int w = 0; w < warmup; w++) gemm.run(stream);
    cudaStreamSynchronize(stream);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);
    for (int i = 0; i < nruns; i++) gemm.run(stream);
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    if (cudaGetLastError() != cudaSuccess) return 1e9f;
    last_pA = pA; last_pB = pB; last_pC = pC;
    return ms / nruns;
  }

  inline bool run(const half* pA, const half* pB, half* pC, int M, int N, int K) {
    if (__builtin_expect(pA != last_pA || pB != last_pB || pC != last_pC, 0)) {
      auto* A = reinterpret_cast<ElemA*>(const_cast<half*>(pA));
      auto* B = reinterpret_cast<ElemB*>(const_cast<half*>(pB));
      auto* C = reinterpret_cast<ElemC*>(pC);
      auto* D = reinterpret_cast<ElemD*>(pC);
      typename G::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A, sA, B, sB},
        {{1.0f, 0.0f}, C, sC, D, sD},
        hw
      };
      if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return false;
      last_pA = pA; last_pB = pB; last_pC = pC;
    }
    auto st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess);
  }
};

static GemmCached<Gemm_0,  StrideA_0,  StrideB_0,  StrideC_0,  StrideD_0>  g_cfg0;
static GemmCached<Gemm_1,  StrideA_1,  StrideB_1,  StrideC_1,  StrideD_1>  g_cfg1;
static GemmCached<Gemm_2,  StrideA_2,  StrideB_2,  StrideC_2,  StrideD_2>  g_cfg2;
static GemmCached<Gemm_3,  StrideA_3,  StrideB_3,  StrideC_3,  StrideD_3>  g_cfg3;
static GemmCached<Gemm_4,  StrideA_4,  StrideB_4,  StrideC_4,  StrideD_4>  g_cfg4;
static GemmCached<Gemm_5,  StrideA_5,  StrideB_5,  StrideC_5,  StrideD_5>  g_cfg5;
static GemmCached<Gemm_6,  StrideA_6,  StrideB_6,  StrideC_6,  StrideD_6>  g_cfg6;
static GemmCached<Gemm_7,  StrideA_7,  StrideB_7,  StrideC_7,  StrideD_7>  g_cfg7;
static GemmCached<Gemm_8,  StrideA_8,  StrideB_8,  StrideC_8,  StrideD_8>  g_cfg8;
static GemmCached<Gemm_9,  StrideA_9,  StrideB_9,  StrideC_9,  StrideD_9>  g_cfg9;
static GemmCached<Gemm_10, StrideA_10, StrideB_10, StrideC_10, StrideD_10> g_cfg10;
static GemmCached<Gemm_11, StrideA_11, StrideB_11, StrideC_11, StrideD_11> g_cfg11;
static GemmCached<Gemm_12, StrideA_12, StrideB_12, StrideC_12, StrideD_12> g_cfg12;
static GemmCached<Gemm_13, StrideA_13, StrideB_13, StrideC_13, StrideD_13> g_cfg13;
static GemmCached<Gemm_14, StrideA_14, StrideB_14, StrideC_14, StrideD_14> g_cfg14;
static GemmCached<Gemm_15, StrideA_15, StrideB_15, StrideC_15, StrideD_15> g_cfg15;
static GemmCached<Gemm_16, StrideA_16, StrideB_16, StrideC_16, StrideD_16> g_cfg16;
static GemmCached<Gemm_17, StrideA_17, StrideB_17, StrideC_17, StrideD_17> g_cfg17;
static GemmCached<Gemm_18, StrideA_18, StrideB_18, StrideC_18, StrideD_18> g_cfg18;
static GemmCached<Gemm_19, StrideA_19, StrideB_19, StrideC_19, StrideD_19> g_cfg19;
static GemmCached<Gemm_20, StrideA_20, StrideB_20, StrideC_20, StrideD_20> g_cfg20;
static GemmCached<Gemm_21, StrideA_21, StrideB_21, StrideC_21, StrideD_21> g_cfg21;
static GemmCached<Gemm_22, StrideA_22, StrideB_22, StrideC_22, StrideD_22> g_cfg22;
static GemmCached<Gemm_23, StrideA_23, StrideB_23, StrideC_23, StrideD_23> g_cfg23;
static GemmCached<Gemm_24, StrideA_24, StrideB_24, StrideC_24, StrideD_24> g_cfg24;
static GemmCached<Gemm_25, StrideA_25, StrideB_25, StrideC_25, StrideD_25> g_cfg25;
static GemmCached<Gemm_26, StrideA_26, StrideB_26, StrideC_26, StrideD_26> g_cfg26;
static GemmCached<Gemm_27, StrideA_27, StrideB_27, StrideC_27, StrideD_27> g_cfg27;
static GemmCached<Gemm_28, StrideA_28, StrideB_28, StrideC_28, StrideD_28> g_cfg28;
static GemmCached<Gemm_29, StrideA_29, StrideB_29, StrideC_29, StrideD_29> g_cfg29;
static GemmCached<Gemm_30, StrideA_30, StrideB_30, StrideC_30, StrideD_30> g_cfg30;
static GemmCached<Gemm_31, StrideA_31, StrideB_31, StrideC_31, StrideD_31> g_cfg31;
static GemmCached<Gemm_32, StrideA_32, StrideB_32, StrideC_32, StrideD_32> g_cfg32;
static GemmCached<Gemm_33, StrideA_33, StrideB_33, StrideC_33, StrideD_33> g_cfg33;
static GemmCached<Gemm_34, StrideA_34, StrideB_34, StrideC_34, StrideD_34> g_cfg34;
static GemmCached<Gemm_35, StrideA_35, StrideB_35, StrideC_35, StrideD_35> g_cfg35;
static GemmCached<Gemm_36, StrideA_36, StrideB_36, StrideC_36, StrideD_36> g_cfg36;
static GemmCached<Gemm_37, StrideA_37, StrideB_37, StrideC_37, StrideD_37> g_cfg37;
static GemmCached<Gemm_38, StrideA_38, StrideB_38, StrideC_38, StrideD_38> g_cfg38;
static GemmCached<Gemm_39, StrideA_39, StrideB_39, StrideC_39, StrideD_39> g_cfg39;
static GemmCached<Gemm_40, StrideA_40, StrideB_40, StrideC_40, StrideD_40> g_cfg40;
static GemmCached<Gemm_41, StrideA_41, StrideB_41, StrideC_41, StrideD_41> g_cfg41;
static GemmCached<Gemm_42, StrideA_42, StrideB_42, StrideC_42, StrideD_42> g_cfg42;
static GemmCached<Gemm_43, StrideA_43, StrideB_43, StrideC_43, StrideD_43> g_cfg43;
static GemmCached<Gemm_44, StrideA_44, StrideB_44, StrideC_44, StrideD_44> g_cfg44;
static GemmCached<Gemm_45, StrideA_45, StrideB_45, StrideC_45, StrideD_45> g_cfg45;
static GemmCached<Gemm_46, StrideA_46, StrideB_46, StrideC_46, StrideD_46> g_cfg46;
static GemmCached<Gemm_47, StrideA_47, StrideB_47, StrideC_47, StrideD_47> g_cfg47;
static GemmCached<Gemm_48, StrideA_48, StrideB_48, StrideC_48, StrideD_48> g_cfg48;
static GemmCached<Gemm_49, StrideA_49, StrideB_49, StrideC_49, StrideD_49> g_cfg49;
static GemmCached<Gemm_50, StrideA_50, StrideB_50, StrideC_50, StrideD_50> g_cfg50;
static GemmCached<Gemm_51, StrideA_51, StrideB_51, StrideC_51, StrideD_51> g_cfg51;
static GemmCached<Gemm_52, StrideA_52, StrideB_52, StrideC_52, StrideD_52> g_cfg52;
static GemmCached<Gemm_53, StrideA_53, StrideB_53, StrideC_53, StrideD_53> g_cfg53;
static GemmCached<Gemm_54, StrideA_54, StrideB_54, StrideC_54, StrideD_54> g_cfg54;
static GemmCached<Gemm_55, StrideA_55, StrideB_55, StrideC_55, StrideD_55> g_cfg55;
static GemmCached<Gemm_56, StrideA_56, StrideB_56, StrideC_56, StrideD_56> g_cfg56;
static GemmCached<Gemm_57, StrideA_57, StrideB_57, StrideC_57, StrideD_57> g_cfg57;
static GemmCached<Gemm_58, StrideA_58, StrideB_58, StrideC_58, StrideD_58> g_cfg58;
static GemmCached<Gemm_59, StrideA_59, StrideB_59, StrideC_59, StrideD_59> g_cfg59;
static GemmCached<Gemm_60, StrideA_60, StrideB_60, StrideC_60, StrideD_60> g_cfg60;
static GemmCached<Gemm_61, StrideA_61, StrideB_61, StrideC_61, StrideD_61> g_cfg61;
static GemmCached<Gemm_62, StrideA_62, StrideB_62, StrideC_62, StrideD_62> g_cfg62;
static GemmCached<Gemm_63, StrideA_63, StrideB_63, StrideC_63, StrideD_63> g_cfg63;
static GemmCached<Gemm_64, StrideA_64, StrideB_64, StrideC_64, StrideD_64> g_cfg64;
static GemmCached<Gemm_65, StrideA_65, StrideB_65, StrideC_65, StrideD_65> g_cfg65;
static GemmCached<Gemm_66, StrideA_66, StrideB_66, StrideC_66, StrideD_66> g_cfg66;
static GemmCached<Gemm_67, StrideA_67, StrideB_67, StrideC_67, StrideD_67> g_cfg67;
static GemmCached<Gemm_68, StrideA_68, StrideB_68, StrideC_68, StrideD_68> g_cfg68;
static GemmCached<Gemm_69, StrideA_69, StrideB_69, StrideC_69, StrideD_69> g_cfg69;
static GemmCached<Gemm_70, StrideA_70, StrideB_70, StrideC_70, StrideD_70> g_cfg70;
static GemmCached<Gemm_71, StrideA_71, StrideB_71, StrideC_71, StrideD_71> g_cfg71;
static GemmCached<Gemm_72, StrideA_72, StrideB_72, StrideC_72, StrideD_72> g_cfg72;
static GemmCached<Gemm_73, StrideA_73, StrideB_73, StrideC_73, StrideD_73> g_cfg73;

typedef bool (*RunFn)(const half*, const half*, half*, int, int, int);
static RunFn g_run_fn  = nullptr;
static int   g_winner  = -1;
static std::once_flag g_init_flag;

static bool run0(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg0.run(a,b,c,M,N,K);}
static bool run1(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg1.run(a,b,c,M,N,K);}
static bool run2(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg2.run(a,b,c,M,N,K);}
static bool run3(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg3.run(a,b,c,M,N,K);}
static bool run4(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg4.run(a,b,c,M,N,K);}
static bool run5(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg5.run(a,b,c,M,N,K);}
static bool run6(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg6.run(a,b,c,M,N,K);}
static bool run7(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg7.run(a,b,c,M,N,K);}
static bool run8(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg8.run(a,b,c,M,N,K);}
static bool run9(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg9.run(a,b,c,M,N,K);}
static bool run10(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg10.run(a,b,c,M,N,K);}
static bool run11(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg11.run(a,b,c,M,N,K);}
static bool run12(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg12.run(a,b,c,M,N,K);}
static bool run13(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg13.run(a,b,c,M,N,K);}
static bool run14(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg14.run(a,b,c,M,N,K);}
static bool run15(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg15.run(a,b,c,M,N,K);}
static bool run16(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg16.run(a,b,c,M,N,K);}
static bool run17(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg17.run(a,b,c,M,N,K);}
static bool run18(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg18.run(a,b,c,M,N,K);}
static bool run19(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg19.run(a,b,c,M,N,K);}
static bool run20(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg20.run(a,b,c,M,N,K);}
static bool run21(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg21.run(a,b,c,M,N,K);}
static bool run22(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg22.run(a,b,c,M,N,K);}
static bool run23(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg23.run(a,b,c,M,N,K);}
static bool run24(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg24.run(a,b,c,M,N,K);}
static bool run25(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg25.run(a,b,c,M,N,K);}
static bool run26(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg26.run(a,b,c,M,N,K);}
static bool run27(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg27.run(a,b,c,M,N,K);}
static bool run28(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg28.run(a,b,c,M,N,K);}
static bool run29(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg29.run(a,b,c,M,N,K);}
static bool run30(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg30.run(a,b,c,M,N,K);}
static bool run31(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg31.run(a,b,c,M,N,K);}
static bool run32(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg32.run(a,b,c,M,N,K);}
static bool run33(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg33.run(a,b,c,M,N,K);}
static bool run34(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg34.run(a,b,c,M,N,K);}
static bool run35(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg35.run(a,b,c,M,N,K);}
static bool run36(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg36.run(a,b,c,M,N,K);}
static bool run37(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg37.run(a,b,c,M,N,K);}
static bool run38(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg38.run(a,b,c,M,N,K);}
static bool run39(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg39.run(a,b,c,M,N,K);}
static bool run40(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg40.run(a,b,c,M,N,K);}
static bool run41(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg41.run(a,b,c,M,N,K);}
static bool run42(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg42.run(a,b,c,M,N,K);}
static bool run43(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg43.run(a,b,c,M,N,K);}
static bool run44(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg44.run(a,b,c,M,N,K);}
static bool run45(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg45.run(a,b,c,M,N,K);}
static bool run46(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg46.run(a,b,c,M,N,K);}
static bool run47(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg47.run(a,b,c,M,N,K);}
static bool run48(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg48.run(a,b,c,M,N,K);}
static bool run49(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg49.run(a,b,c,M,N,K);}
static bool run50(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg50.run(a,b,c,M,N,K);}
static bool run51(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg51.run(a,b,c,M,N,K);}
static bool run52(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg52.run(a,b,c,M,N,K);}
static bool run53(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg53.run(a,b,c,M,N,K);}
static bool run54(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg54.run(a,b,c,M,N,K);}
static bool run55(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg55.run(a,b,c,M,N,K);}
static bool run56(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg56.run(a,b,c,M,N,K);}
static bool run57(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg57.run(a,b,c,M,N,K);}
static bool run58(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg58.run(a,b,c,M,N,K);}
static bool run59(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg59.run(a,b,c,M,N,K);}
static bool run60(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg60.run(a,b,c,M,N,K);}
static bool run61(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg61.run(a,b,c,M,N,K);}
static bool run62(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg62.run(a,b,c,M,N,K);}
static bool run63(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg63.run(a,b,c,M,N,K);}
static bool run64(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg64.run(a,b,c,M,N,K);}
static bool run65(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg65.run(a,b,c,M,N,K);}
static bool run66(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg66.run(a,b,c,M,N,K);}
static bool run67(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg67.run(a,b,c,M,N,K);}
static bool run68(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg68.run(a,b,c,M,N,K);}
static bool run69(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg69.run(a,b,c,M,N,K);}
static bool run70(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg70.run(a,b,c,M,N,K);}
static bool run71(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg71.run(a,b,c,M,N,K);}
static bool run72(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg72.run(a,b,c,M,N,K);}
static bool run73(const half* a,const half* b,half* c,int M,int N,int K){return g_cfg73.run(a,b,c,M,N,K);}

static RunFn g_run_fns[NUM_CFGS] = {
  run0,run1,run2,run3,run4,run5,run6,run7,run8,run9,
  run10,run11,run12,run13,run14,run15,run16,run17,run18,run19,
  run20,run21,run22,run23,run24,run25,run26,run27,run28,run29,
  run30,run31,run32,run33,run34,run35,run36,run37,run38,run39,
  run40,run41,run42,run43,run44,run45,run46,run47,run48,run49,
  run50,run51,run52,run53,run54,run55,run56,run57,run58,run59,
  run60,run61,run62,run63,run64,run65,run66,run67,run68,run69,
  run70,run71,run72,run73
};

static void do_init(const half* pA, const half* pB, half* pC, int M, int N, int K) {
  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw;
  hw.device_id = device_id;
  hw.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  bool ok[NUM_CFGS] = {};
  ok[0]  = g_cfg0.try_init(pA,pB,pC,M,N,K,hw);
  ok[1]  = g_cfg1.try_init(pA,pB,pC,M,N,K,hw);
  ok[2]  = g_cfg2.try_init(pA,pB,pC,M,N,K,hw);
  ok[3]  = g_cfg3.try_init(pA,pB,pC,M,N,K,hw);
  ok[4]  = g_cfg4.try_init(pA,pB,pC,M,N,K,hw);
  ok[5]  = g_cfg5.try_init(pA,pB,pC,M,N,K,hw);
  ok[6]  = g_cfg6.try_init(pA,pB,pC,M,N,K,hw);
  ok[7]  = g_cfg7.try_init(pA,pB,pC,M,N,K,hw);
  ok[8]  = g_cfg8.try_init(pA,pB,pC,M,N,K,hw);
  ok[9]  = g_cfg9.try_init(pA,pB,pC,M,N,K,hw);
  ok[10] = g_cfg10.try_init(pA,pB,pC,M,N,K,hw);
  ok[11] = g_cfg11.try_init(pA,pB,pC,M,N,K,hw);
  ok[12] = g_cfg12.try_init(pA,pB,pC,M,N,K,hw);
  ok[13] = g_cfg13.try_init(pA,pB,pC,M,N,K,hw);
  ok[14] = g_cfg14.try_init(pA,pB,pC,M,N,K,hw);
  ok[15] = g_cfg15.try_init(pA,pB,pC,M,N,K,hw);
  ok[16] = g_cfg16.try_init(pA,pB,pC,M,N,K,hw);
  ok[17] = g_cfg17.try_init(pA,pB,pC,M,N,K,hw);
  ok[18] = g_cfg18.try_init(pA,pB,pC,M,N,K,hw);
  ok[19] = g_cfg19.try_init(pA,pB,pC,M,N,K,hw);
  ok[20] = g_cfg20.try_init(pA,pB,pC,M,N,K,hw);
  ok[21] = g_cfg21.try_init(pA,pB,pC,M,N,K,hw);
  ok[22] = g_cfg22.try_init(pA,pB,pC,M,N,K,hw);
  ok[23] = g_cfg23.try_init(pA,pB,pC,M,N,K,hw);
  ok[24] = g_cfg24.try_init(pA,pB,pC,M,N,K,hw);
  ok[25] = g_cfg25.try_init(pA,pB,pC,M,N,K,hw);
  ok[26] = g_cfg26.try_init(pA,pB,pC,M,N,K,hw);
  ok[27] = g_cfg27.try_init(pA,pB,pC,M,N,K,hw);
  ok[28] = g_cfg28.try_init(pA,pB,pC,M,N,K,hw);
  ok[29] = g_cfg29.try_init(pA,pB,pC,M,N,K,hw);
  ok[30] = g_cfg30.try_init(pA,pB,pC,M,N,K,hw);
  ok[31] = g_cfg31.try_init(pA,pB,pC,M,N,K,hw);
  ok[32] = g_cfg32.try_init(pA,pB,pC,M,N,K,hw);
  ok[33] = g_cfg33.try_init(pA,pB,pC,M,N,K,hw);
  ok[34] = g_cfg34.try_init(pA,pB,pC,M,N,K,hw);
  ok[35] = g_cfg35.try_init(pA,pB,pC,M,N,K,hw);
  ok[36] = g_cfg36.try_init(pA,pB,pC,M,N,K,hw);
  ok[37] = g_cfg37.try_init(pA,pB,pC,M,N,K,hw);
  ok[38] = g_cfg38.try_init(pA,pB,pC,M,N,K,hw);
  ok[39] = g_cfg39.try_init(pA,pB,pC,M,N,K,hw);
  ok[40] = g_cfg40.try_init(pA,pB,pC,M,N,K,hw);
  ok[41] = g_cfg41.try_init(pA,pB,pC,M,N,K,hw);
  ok[42] = g_cfg42.try_init(pA,pB,pC,M,N,K,hw);
  ok[43] = g_cfg43.try_init(pA,pB,pC,M,N,K,hw);
  ok[44] = g_cfg44.try_init(pA,pB,pC,M,N,K,hw);
  ok[45] = g_cfg45.try_init(pA,pB,pC,M,N,K,hw);
  ok[46] = g_cfg46.try_init(pA,pB,pC,M,N,K,hw);
  ok[47] = g_cfg47.try_init(pA,pB,pC,M,N,K,hw);
  ok[48] = g_cfg48.try_init(pA,pB,pC,M,N,K,hw);
  ok[49] = g_cfg49.try_init(pA,pB,pC,M,N,K,hw);
  ok[50] = g_cfg50.try_init(pA,pB,pC,M,N,K,hw);
  ok[51] = g_cfg51.try_init(pA,pB,pC,M,N,K,hw);
  ok[52] = g_cfg52.try_init(pA,pB,pC,M,N,K,hw);
  ok[53] = g_cfg53.try_init(pA,pB,pC,M,N,K,hw);
  ok[54] = g_cfg54.try_init(pA,pB,pC,M,N,K,hw);
  ok[55] = g_cfg55.try_init(pA,pB,pC,M,N,K,hw);
  ok[56] = g_cfg56.try_init(pA,pB,pC,M,N,K,hw);
  ok[57] = g_cfg57.try_init(pA,pB,pC,M,N,K,hw);
  ok[58] = g_cfg58.try_init(pA,pB,pC,M,N,K,hw);
  ok[59] = g_cfg59.try_init(pA,pB,pC,M,N,K,hw);
  ok[60] = g_cfg60.try_init(pA,pB,pC,M,N,K,hw);
  ok[61] = g_cfg61.try_init(pA,pB,pC,M,N,K,hw);
  ok[62] = g_cfg62.try_init(pA,pB,pC,M,N,K,hw);
  ok[63] = g_cfg63.try_init(pA,pB,pC,M,N,K,hw);
  ok[64] = g_cfg64.try_init(pA,pB,pC,M,N,K,hw);
  ok[65] = g_cfg65.try_init(pA,pB,pC,M,N,K,hw);
  ok[66] = g_cfg66.try_init(pA,pB,pC,M,N,K,hw);
  ok[67] = g_cfg67.try_init(pA,pB,pC,M,N,K,hw);
  ok[68] = g_cfg68.try_init(pA,pB,pC,M,N,K,hw);
  ok[69] = g_cfg69.try_init(pA,pB,pC,M,N,K,hw);
  ok[70] = g_cfg70.try_init(pA,pB,pC,M,N,K,hw);
  ok[71] = g_cfg71.try_init(pA,pB,pC,M,N,K,hw);
  ok[72] = g_cfg72.try_init(pA,pB,pC,M,N,K,hw);
  ok[73] = g_cfg73.try_init(pA,pB,pC,M,N,K,hw);

  float best_ms  = 1e9f;
  int   best_idx = -1;

  auto bench_if_ok = [&](int idx, auto& cfg) {
    if (!ok[idx]) return;
    float ms = cfg.benchmark(pA, pB, pC, M, N, K);
    if (ms < best_ms) { best_ms = ms; best_idx = idx; }
  };

  bench_if_ok(0,  g_cfg0);
  bench_if_ok(1,  g_cfg1);
  bench_if_ok(2,  g_cfg2);
  bench_if_ok(3,  g_cfg3);
  bench_if_ok(4,  g_cfg4);
  bench_if_ok(5,  g_cfg5);
  bench_if_ok(6,  g_cfg6);
  bench_if_ok(7,  g_cfg7);
  bench_if_ok(8,  g_cfg8);
  bench_if_ok(9,  g_cfg9);
  bench_if_ok(10, g_cfg10);
  bench_if_ok(11, g_cfg11);
  bench_if_ok(12, g_cfg12);
  bench_if_ok(13, g_cfg13);
  bench_if_ok(14, g_cfg14);
  bench_if_ok(15, g_cfg15);
  bench_if_ok(16, g_cfg16);
  bench_if_ok(17, g_cfg17);
  bench_if_ok(18, g_cfg18);
  bench_if_ok(19, g_cfg19);
  bench_if_ok(20, g_cfg20);
  bench_if_ok(21, g_cfg21);
  bench_if_ok(22, g_cfg22);
  bench_if_ok(23, g_cfg23);
  bench_if_ok(24, g_cfg24);
  bench_if_ok(25, g_cfg25);
  bench_if_ok(26, g_cfg26);
  bench_if_ok(27, g_cfg27);
  bench_if_ok(28, g_cfg28);
  bench_if_ok(29, g_cfg29);
  bench_if_ok(30, g_cfg30);
  bench_if_ok(31, g_cfg31);
  bench_if_ok(32, g_cfg32);
  bench_if_ok(33, g_cfg33);
  bench_if_ok(34, g_cfg34);
  bench_if_ok(35, g_cfg35);
  bench_if_ok(36, g_cfg36);
  bench_if_ok(37, g_cfg37);
  bench_if_ok(38, g_cfg38);
  bench_if_ok(39, g_cfg39);
  bench_if_ok(40, g_cfg40);
  bench_if_ok(41, g_cfg41);
  bench_if_ok(42, g_cfg42);
  bench_if_ok(43, g_cfg43);
  bench_if_ok(44, g_cfg44);
  bench_if_ok(45, g_cfg45);
  bench_if_ok(46, g_cfg46);
  bench_if_ok(47, g_cfg47);
  bench_if_ok(48, g_cfg48);
  bench_if_ok(49, g_cfg49);
  bench_if_ok(50, g_cfg50);
  bench_if_ok(51, g_cfg51);
  bench_if_ok(52, g_cfg52);
  bench_if_ok(53, g_cfg53);
  bench_if_ok(54, g_cfg54);
  bench_if_ok(55, g_cfg55);
  bench_if_ok(56, g_cfg56);
  bench_if_ok(57, g_cfg57);
  bench_if_ok(58, g_cfg58);
  bench_if_ok(59, g_cfg59);
  bench_if_ok(60, g_cfg60);
  bench_if_ok(61, g_cfg61);
  bench_if_ok(62, g_cfg62);
  bench_if_ok(63, g_cfg63);
  bench_if_ok(64, g_cfg64);
  bench_if_ok(65, g_cfg65);
  bench_if_ok(66, g_cfg66);
  bench_if_ok(67, g_cfg67);
  bench_if_ok(68, g_cfg68);
  bench_if_ok(69, g_cfg69);
  bench_if_ok(70, g_cfg70);
  bench_if_ok(71, g_cfg71);
  bench_if_ok(72, g_cfg72);
  bench_if_ok(73, g_cfg73);

  g_winner = best_idx;
  if (best_idx >= 0) {
    g_run_fn = g_run_fns[best_idx];
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  const half* pA = reinterpret_cast<const half*>(a.data_ptr());
  const half* pB = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half*       pC = reinterpret_cast<half*>(c.data_ptr());

  std::call_once(g_init_flag, do_init, pA, pB, pC, M, N, K);

  if (__builtin_expect(g_winner < 0, 0)) {
    throw std::runtime_error("All GEMM variants failed to initialize");
  }

  if (__builtin_expect(g_run_fn(pA, pB, pC, M, N, K), 1)) return;

  for (int i = 0; i < NUM_CFGS; i++) {
    if (i == g_winner) continue;
    if (g_run_fns[i](pA, pB, pC, M, N, K)) {
      g_run_fn = g_run_fns[i];
      g_winner = i;
      return;
    }
  }

  throw std::runtime_error("All GEMM variants failed at runtime");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}