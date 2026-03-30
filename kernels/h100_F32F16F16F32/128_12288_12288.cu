#include <iostream>
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEF_COOP_SK_STAGE(NAME, TM, TN, TK, CM, CN, CK, STAGES)              \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                    \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAccumulator,\
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCount<STAGES>,                            \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_COOP_SK_AUTO(NAME, TM, TN, TK, CM, CN, CK)                        \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                    \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAccumulator,\
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::StreamKScheduler>;                                         \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_COOP_PERS_AUTO(NAME, TM, TN, TK, CM, CN, CK)                      \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                    \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAccumulator,\
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_COOP_PERS_STAGE(NAME, TM, TN, TK, CM, CN, CK, STAGES)            \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                    \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAccumulator,\
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCount<STAGES>,                            \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;        \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEF_PING_PERS_AUTO(NAME, TM, TN, TK, CM, CN, CK)                      \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape   = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                          \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                    \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;         \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAccumulator,\
      TileShape, GroupShape,                                                    \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;           \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                      \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;     \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEF_COOP_SK_STAGE(A01_256_1x2_SK_S3,  128, 256, 64, 1, 2, 1, 3)
DEF_COOP_SK_STAGE(A02_256_1x2_SK_S4,  128, 256, 64, 1, 2, 1, 4)
DEF_COOP_SK_STAGE(A03_256_1x2_SK_S5,  128, 256, 64, 1, 2, 1, 5)
DEF_COOP_SK_STAGE(A04_256_1x2_SK_S6,  128, 256, 64, 1, 2, 1, 6)
DEF_COOP_SK_STAGE(A05_256_1x2_SK_S7,  128, 256, 64, 1, 2, 1, 7)
DEF_COOP_SK_STAGE(A06_256_1x2_SK_S8,  128, 256, 64, 1, 2, 1, 8)
DEF_COOP_SK_AUTO (A07_256_1x2_SK_Auto, 128, 256, 64, 1, 2, 1)

DEF_COOP_SK_STAGE(B01_256_1x4_SK_S3,  128, 256, 64, 1, 4, 1, 3)
DEF_COOP_SK_STAGE(B02_256_1x4_SK_S4,  128, 256, 64, 1, 4, 1, 4)
DEF_COOP_SK_STAGE(B03_256_1x4_SK_S5,  128, 256, 64, 1, 4, 1, 5)
DEF_COOP_SK_STAGE(B04_256_1x4_SK_S6,  128, 256, 64, 1, 4, 1, 6)
DEF_COOP_SK_AUTO (B05_256_1x4_SK_Auto, 128, 256, 64, 1, 4, 1)

DEF_COOP_SK_STAGE(C01_256k128_1x2_SK_S3,   128, 256, 128, 1, 2, 1, 3)
DEF_COOP_SK_STAGE(C02_256k128_1x2_SK_S4,   128, 256, 128, 1, 2, 1, 4)
DEF_COOP_SK_STAGE(C03_256k128_1x2_SK_S5,   128, 256, 128, 1, 2, 1, 5)
DEF_COOP_SK_AUTO (C04_256k128_1x2_SK_Auto,  128, 256, 128, 1, 2, 1)

DEF_COOP_SK_STAGE(D01_256k128_1x4_SK_S3,   128, 256, 128, 1, 4, 1, 3)
DEF_COOP_SK_STAGE(D02_256k128_1x4_SK_S4,   128, 256, 128, 1, 4, 1, 4)
DEF_COOP_SK_AUTO (D03_256k128_1x4_SK_Auto,  128, 256, 128, 1, 4, 1)

DEF_COOP_SK_STAGE(E01_128_1x1_SK_S4,  128, 128, 64, 1, 1, 1, 4)
DEF_COOP_SK_STAGE(E02_128_1x1_SK_S5,  128, 128, 64, 1, 1, 1, 5)
DEF_COOP_SK_STAGE(E03_128_1x1_SK_S6,  128, 128, 64, 1, 1, 1, 6)
DEF_COOP_SK_STAGE(E04_128_1x1_SK_S7,  128, 128, 64, 1, 1, 1, 7)
DEF_COOP_SK_STAGE(E05_128_1x1_SK_S8,  128, 128, 64, 1, 1, 1, 8)
DEF_COOP_SK_AUTO (E06_128_1x1_SK_Auto, 128, 128, 64, 1, 1, 1)

DEF_COOP_SK_STAGE(F01_128_1x2_SK_S4,  128, 128, 64, 1, 2, 1, 4)
DEF_COOP_SK_STAGE(F02_128_1x2_SK_S5,  128, 128, 64, 1, 2, 1, 5)
DEF_COOP_SK_STAGE(F03_128_1x2_SK_S6,  128, 128, 64, 1, 2, 1, 6)
DEF_COOP_SK_AUTO (F04_128_1x2_SK_Auto, 128, 128, 64, 1, 2, 1)

DEF_COOP_SK_STAGE(G01_128k128_1x2_SK_S3,  128, 128, 128, 1, 2, 1, 3)
DEF_COOP_SK_STAGE(G02_128k128_1x2_SK_S4,  128, 128, 128, 1, 2, 1, 4)
DEF_COOP_SK_AUTO (G03_128k128_1x2_SK_Auto, 128, 128, 128, 1, 2, 1)

DEF_COOP_SK_AUTO(H01_256_2x2_SK_Auto,  128, 256, 64, 2, 2, 1)
DEF_COOP_SK_STAGE(H02_256_2x2_SK_S4,   128, 256, 64, 2, 2, 1, 4)
DEF_COOP_SK_STAGE(H03_256_2x2_SK_S5,   128, 256, 64, 2, 2, 1, 5)

DEF_COOP_SK_STAGE(I01_256_1x8_SK_S3,  128, 256, 64, 1, 8, 1, 3)
DEF_COOP_SK_STAGE(I02_256_1x8_SK_S4,  128, 256, 64, 1, 8, 1, 4)
DEF_COOP_SK_AUTO (I03_256_1x8_SK_Auto, 128, 256, 64, 1, 8, 1)

DEF_COOP_PERS_AUTO  (J01_256_1x2_CP_Auto,  128, 256,  64, 1, 2, 1)
DEF_COOP_PERS_STAGE (J02_256_1x2_CP_S5,    128, 256,  64, 1, 2, 1, 5)
DEF_COOP_PERS_STAGE (J03_256_1x2_CP_S4,    128, 256,  64, 1, 2, 1, 4)
DEF_COOP_PERS_AUTO  (J04_256_1x4_CP_Auto,  128, 256,  64, 1, 4, 1)
DEF_COOP_PERS_AUTO  (J05_256k128_1x2_CP,   128, 256, 128, 1, 2, 1)
DEF_COOP_PERS_AUTO  (J06_128_1x1_CP_Auto,  128, 128,  64, 1, 1, 1)
DEF_COOP_PERS_AUTO  (J07_128_1x2_CP_Auto,  128, 128,  64, 1, 2, 1)

DEF_PING_PERS_AUTO(K01_256_1x2_PP_Auto,  128, 256,  64, 1, 2, 1)
DEF_PING_PERS_AUTO(K02_256_1x4_PP_Auto,  128, 256,  64, 1, 4, 1)
DEF_PING_PERS_AUTO(K03_128_1x1_PP_Auto,  128, 128,  64, 1, 1, 1)
DEF_PING_PERS_AUTO(K04_128_1x2_PP_Auto,  128, 128,  64, 1, 2, 1)

static void*  g_workspace      = nullptr;
static size_t g_workspace_size = 0;
static int    g_best_variant   = -1;

static void* get_workspace(size_t required) {
  if (required == 0) return nullptr;
  if (required <= g_workspace_size) return g_workspace;
  if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_size = 0; }
  size_t alloc = std::max(required * 2, size_t(256ULL << 20));
  if (cudaMalloc(&g_workspace, alloc) == cudaSuccess) { g_workspace_size = alloc; return g_workspace; }
  if (cudaMalloc(&g_workspace, required) == cudaSuccess) { g_workspace_size = required; return g_workspace; }
  return nullptr;
}

template <typename HgemmType>
bool try_run_impl(void* a_ptr, void* b_ptr, void* c_ptr,
                  int M, int N, int K,
                  cutlass::KernelHardwareInfo hw_info) {
  using Gemm    = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto* ptr_A = reinterpret_cast<ElementA*>(a_ptr);
  auto* ptr_B = reinterpret_cast<ElementB*>(b_ptr);
  auto* ptr_C = reinterpret_cast<ElementC*>(c_ptr);
  auto* ptr_D = reinterpret_cast<ElementD*>(c_ptr);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_D, stride_D},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
  size_t ws_size = Gemm::get_workspace_size(arguments);
  void* workspace = get_workspace(ws_size);
  if (ws_size > 0 && workspace == nullptr) return false;
  if (gemm.initialize(arguments, workspace) != cutlass::Status::kSuccess) return false;
  return gemm.run() == cutlass::Status::kSuccess;
}

static bool dispatch(int vid, void* a, void* b, void* c, int M, int N, int K,
                     cutlass::KernelHardwareInfo hw) {
  switch (vid) {
    case  1: return try_run_impl<A01_256_1x2_SK_S3>(a,b,c,M,N,K,hw);
    case  2: return try_run_impl<A02_256_1x2_SK_S4>(a,b,c,M,N,K,hw);
    case  3: return try_run_impl<A03_256_1x2_SK_S5>(a,b,c,M,N,K,hw);
    case  4: return try_run_impl<A04_256_1x2_SK_S6>(a,b,c,M,N,K,hw);
    case  5: return try_run_impl<A05_256_1x2_SK_S7>(a,b,c,M,N,K,hw);
    case  6: return try_run_impl<A06_256_1x2_SK_S8>(a,b,c,M,N,K,hw);
    case  7: return try_run_impl<A07_256_1x2_SK_Auto>(a,b,c,M,N,K,hw);
    case  8: return try_run_impl<B01_256_1x4_SK_S3>(a,b,c,M,N,K,hw);
    case  9: return try_run_impl<B02_256_1x4_SK_S4>(a,b,c,M,N,K,hw);
    case 10: return try_run_impl<B03_256_1x4_SK_S5>(a,b,c,M,N,K,hw);
    case 11: return try_run_impl<B04_256_1x4_SK_S6>(a,b,c,M,N,K,hw);
    case 12: return try_run_impl<B05_256_1x4_SK_Auto>(a,b,c,M,N,K,hw);
    case 13: return try_run_impl<C01_256k128_1x2_SK_S3>(a,b,c,M,N,K,hw);
    case 14: return try_run_impl<C02_256k128_1x2_SK_S4>(a,b,c,M,N,K,hw);
    case 15: return try_run_impl<C03_256k128_1x2_SK_S5>(a,b,c,M,N,K,hw);
    case 16: return try_run_impl<C04_256k128_1x2_SK_Auto>(a,b,c,M,N,K,hw);
    case 17: return try_run_impl<D01_256k128_1x4_SK_S3>(a,b,c,M,N,K,hw);
    case 18: return try_run_impl<D02_256k128_1x4_SK_S4>(a,b,c,M,N,K,hw);
    case 19: return try_run_impl<D03_256k128_1x4_SK_Auto>(a,b,c,M,N,K,hw);
    case 20: return try_run_impl<E01_128_1x1_SK_S4>(a,b,c,M,N,K,hw);
    case 21: return try_run_impl<E02_128_1x1_SK_S5>(a,b,c,M,N,K,hw);
    case 22: return try_run_impl<E03_128_1x1_SK_S6>(a,b,c,M,N,K,hw);
    case 23: return try_run_impl<E04_128_1x1_SK_S7>(a,b,c,M,N,K,hw);
    case 24: return try_run_impl<E05_128_1x1_SK_S8>(a,b,c,M,N,K,hw);
    case 25: return try_run_impl<E06_128_1x1_SK_Auto>(a,b,c,M,N,K,hw);
    case 26: return try_run_impl<F01_128_1x2_SK_S4>(a,b,c,M,N,K,hw);
    case 27: return try_run_impl<F02_128_1x2_SK_S5>(a,b,c,M,N,K,hw);
    case 28: return try_run_impl<F03_128_1x2_SK_S6>(a,b,c,M,N,K,hw);
    case 29: return try_run_impl<F04_128_1x2_SK_Auto>(a,b,c,M,N,K,hw);
    case 30: return try_run_impl<G01_128k128_1x2_SK_S3>(a,b,c,M,N,K,hw);
    case 31: return try_run_impl<G02_128k128_1x2_SK_S4>(a,b,c,M,N,K,hw);
    case 32: return try_run_impl<G03_128k128_1x2_SK_Auto>(a,b,c,M,N,K,hw);
    case 33: return try_run_impl<H01_256_2x2_SK_Auto>(a,b,c,M,N,K,hw);
    case 34: return try_run_impl<H02_256_2x2_SK_S4>(a,b,c,M,N,K,hw);
    case 35: return try_run_impl<H03_256_2x2_SK_S5>(a,b,c,M,N,K,hw);
    case 36: return try_run_impl<I01_256_1x8_SK_S3>(a,b,c,M,N,K,hw);
    case 37: return try_run_impl<I02_256_1x8_SK_S4>(a,b,c,M,N,K,hw);
    case 38: return try_run_impl<I03_256_1x8_SK_Auto>(a,b,c,M,N,K,hw);
    case 39: return try_run_impl<J01_256_1x2_CP_Auto>(a,b,c,M,N,K,hw);
    case 40: return try_run_impl<J02_256_1x2_CP_S5>(a,b,c,M,N,K,hw);
    case 41: return try_run_impl<J03_256_1x2_CP_S4>(a,b,c,M,N,K,hw);
    case 42: return try_run_impl<J04_256_1x4_CP_Auto>(a,b,c,M,N,K,hw);
    case 43: return try_run_impl<J05_256k128_1x2_CP>(a,b,c,M,N,K,hw);
    case 44: return try_run_impl<J06_128_1x1_CP_Auto>(a,b,c,M,N,K,hw);
    case 45: return try_run_impl<J07_128_1x2_CP_Auto>(a,b,c,M,N,K,hw);
    case 46: return try_run_impl<K01_256_1x2_PP_Auto>(a,b,c,M,N,K,hw);
    case 47: return try_run_impl<K02_256_1x4_PP_Auto>(a,b,c,M,N,K,hw);
    case 48: return try_run_impl<K03_128_1x1_PP_Auto>(a,b,c,M,N,K,hw);
    case 49: return try_run_impl<K04_128_1x2_PP_Auto>(a,b,c,M,N,K,hw);
    default: return false;
  }
}

static float time_variant(int vid, void* a, void* b, void* c, int M, int N, int K,
                           cutlass::KernelHardwareInfo hw) {
  if (!dispatch(vid, a, b, c, M, N, K, hw)) return 1e30f;
  cudaDeviceSynchronize();
  dispatch(vid, a, b, c, M, N, K, hw);
  dispatch(vid, a, b, c, M, N, K, hw);
  cudaDeviceSynchronize();
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < 5; i++) dispatch(vid, a, b, c, M, N, K, hw);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / 5.0f;
}

static void autotune(void* a, void* b, void* c, int M, int N, int K,
                     cutlass::KernelHardwareInfo hw) {
  static const int order[] = {
    4, 3, 5, 2, 6, 1, 7,
    10, 9, 11, 8, 12,
    14, 13, 15, 16,
    17, 18, 19,
    21, 22, 20, 23, 24, 25,
    27, 26, 28, 29,
    31, 30, 32,
    34, 35, 33,
    36, 37, 38,
    40, 41, 39, 42, 43, 44, 45,
    46, 47, 48, 49
  };
  static const int n = (int)(sizeof(order)/sizeof(order[0]));

  float best_t = 1e30f;
  int   best_v = -1;
  for (int i = 0; i < n; i++) {
    float t = time_variant(order[i], a, b, c, M, N, K, hw);
    if (t < best_t) { best_t = t; best_v = order[i]; }
  }
  g_best_variant = best_v;
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

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  void* a_ptr = a.data_ptr();
  void* b_ptr = b_col_major.data_ptr();
  void* c_ptr = c.data_ptr();

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  if (g_best_variant != -1) {
    if (dispatch(g_best_variant, a_ptr, b_ptr, c_ptr, M, N, K, hw_info)) return;
    g_best_variant = -1;
  }

  autotune(a_ptr, b_ptr, c_ptr, M, N, K, hw_info);

  if (g_best_variant == -1) {
    throw std::runtime_error("All CUTLASS GEMM variants failed during autotuning");
  }

  if (!dispatch(g_best_variant, a_ptr, b_ptr, c_ptr, M, N, K, hw_info)) {
    throw std::runtime_error("Best variant failed on final run");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}