#include <iostream>
#include <mutex>
#include <atomic>
#include <vector>
#include <limits>
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
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_PP(Name, TM, TN, TK, CM, CN, CK)                               \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                   \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;        \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                   \
      ElementAccumulator, TileShape, GroupShape,                               \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_PP_S(Name, TM, TN, TK, CM, CN, CK, NS)                         \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                   \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;        \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                   \
      ElementAccumulator, TileShape, GroupShape,                               \
      cutlass::gemm::collective::StageCount<NS>,                               \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_WS(Name, TM, TN, TK, CM, CN, CK)                               \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                   \
      cutlass::epilogue::NoSmemWarpSpecialized, EpilogueOp>::CollectiveOp;     \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                   \
      ElementAccumulator, TileShape, GroupShape,                               \
      cutlass::gemm::collective::StageCountAutoCarveout<                        \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,\
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;                  \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

#define DEFINE_WS_S(Name, TM, TN, TK, CM, CN, CK, NS)                         \
struct Name {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;\
  using GroupShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;\
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GroupShape,                                                    \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                   \
      cutlass::epilogue::NoSmemWarpSpecialized, EpilogueOp>::CollectiveOp;     \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                   \
      ElementAccumulator, TileShape, GroupShape,                               \
      cutlass::gemm::collective::StageCount<NS>,                               \
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;                  \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                     \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                          \
  using StrideB = typename Gemm::GemmKernel::StrideB;                          \
  using StrideC = typename Gemm::GemmKernel::StrideC;                          \
  using StrideD = typename Gemm::GemmKernel::StrideD;                          \
};

DEFINE_PP(PP_64x256x64_c1x8x1,   64, 256,  64, 1, 8, 1)
DEFINE_PP(PP_64x256x64_c1x4x1,   64, 256,  64, 1, 4, 1)
DEFINE_PP(PP_64x256x64_c1x2x1,   64, 256,  64, 1, 2, 1)
DEFINE_PP(PP_64x256x64_c1x1x1,   64, 256,  64, 1, 1, 1)
DEFINE_PP(PP_64x256x128_c1x8x1,  64, 256, 128, 1, 8, 1)
DEFINE_PP(PP_64x256x128_c1x4x1,  64, 256, 128, 1, 4, 1)
DEFINE_PP(PP_64x256x128_c1x2x1,  64, 256, 128, 1, 2, 1)
DEFINE_PP(PP_64x256x128_c1x1x1,  64, 256, 128, 1, 1, 1)
DEFINE_PP(PP_64x128x64_c1x8x1,   64, 128,  64, 1, 8, 1)
DEFINE_PP(PP_64x128x64_c1x4x1,   64, 128,  64, 1, 4, 1)
DEFINE_PP(PP_64x128x64_c1x2x1,   64, 128,  64, 1, 2, 1)
DEFINE_PP(PP_64x128x64_c1x1x1,   64, 128,  64, 1, 1, 1)
DEFINE_PP(PP_64x128x128_c1x8x1,  64, 128, 128, 1, 8, 1)
DEFINE_PP(PP_64x128x128_c1x4x1,  64, 128, 128, 1, 4, 1)
DEFINE_PP(PP_64x128x128_c1x2x1,  64, 128, 128, 1, 2, 1)
DEFINE_PP(PP_64x128x128_c1x1x1,  64, 128, 128, 1, 1, 1)
DEFINE_PP(PP_64x64x64_c1x8x1,    64,  64,  64, 1, 8, 1)
DEFINE_PP(PP_64x64x64_c1x4x1,    64,  64,  64, 1, 4, 1)
DEFINE_PP(PP_64x64x64_c1x2x1,    64,  64,  64, 1, 2, 1)
DEFINE_PP(PP_64x64x64_c1x1x1,    64,  64,  64, 1, 1, 1)

DEFINE_PP_S(PP_64x256x64_c1x8x1_s3,  64, 256,  64, 1, 8, 1, 3)
DEFINE_PP_S(PP_64x256x64_c1x8x1_s4,  64, 256,  64, 1, 8, 1, 4)
DEFINE_PP_S(PP_64x256x64_c1x8x1_s5,  64, 256,  64, 1, 8, 1, 5)
DEFINE_PP_S(PP_64x256x64_c1x4x1_s3,  64, 256,  64, 1, 4, 1, 3)
DEFINE_PP_S(PP_64x256x64_c1x4x1_s4,  64, 256,  64, 1, 4, 1, 4)
DEFINE_PP_S(PP_64x256x64_c1x4x1_s5,  64, 256,  64, 1, 4, 1, 5)
DEFINE_PP_S(PP_64x256x64_c1x2x1_s3,  64, 256,  64, 1, 2, 1, 3)
DEFINE_PP_S(PP_64x256x64_c1x2x1_s4,  64, 256,  64, 1, 2, 1, 4)
DEFINE_PP_S(PP_64x256x64_c1x2x1_s5,  64, 256,  64, 1, 2, 1, 5)
DEFINE_PP_S(PP_64x256x64_c1x1x1_s3,  64, 256,  64, 1, 1, 1, 3)
DEFINE_PP_S(PP_64x256x64_c1x1x1_s4,  64, 256,  64, 1, 1, 1, 4)
DEFINE_PP_S(PP_64x256x64_c1x1x1_s5,  64, 256,  64, 1, 1, 1, 5)
DEFINE_PP_S(PP_64x256x128_c1x4x1_s3, 64, 256, 128, 1, 4, 1, 3)
DEFINE_PP_S(PP_64x256x128_c1x4x1_s4, 64, 256, 128, 1, 4, 1, 4)
DEFINE_PP_S(PP_64x256x128_c1x8x1_s3, 64, 256, 128, 1, 8, 1, 3)
DEFINE_PP_S(PP_64x256x128_c1x8x1_s4, 64, 256, 128, 1, 8, 1, 4)
DEFINE_PP_S(PP_64x256x128_c1x2x1_s3, 64, 256, 128, 1, 2, 1, 3)
DEFINE_PP_S(PP_64x256x128_c1x2x1_s4, 64, 256, 128, 1, 2, 1, 4)
DEFINE_PP_S(PP_64x128x64_c1x8x1_s3,  64, 128,  64, 1, 8, 1, 3)
DEFINE_PP_S(PP_64x128x64_c1x8x1_s4,  64, 128,  64, 1, 8, 1, 4)
DEFINE_PP_S(PP_64x128x64_c1x8x1_s5,  64, 128,  64, 1, 8, 1, 5)
DEFINE_PP_S(PP_64x128x64_c1x4x1_s3,  64, 128,  64, 1, 4, 1, 3)
DEFINE_PP_S(PP_64x128x64_c1x4x1_s4,  64, 128,  64, 1, 4, 1, 4)
DEFINE_PP_S(PP_64x128x64_c1x4x1_s5,  64, 128,  64, 1, 4, 1, 5)
DEFINE_PP_S(PP_64x128x64_c1x2x1_s3,  64, 128,  64, 1, 2, 1, 3)
DEFINE_PP_S(PP_64x128x64_c1x2x1_s4,  64, 128,  64, 1, 2, 1, 4)
DEFINE_PP_S(PP_64x128x64_c1x2x1_s5,  64, 128,  64, 1, 2, 1, 5)
DEFINE_PP_S(PP_64x128x128_c1x4x1_s3, 64, 128, 128, 1, 4, 1, 3)
DEFINE_PP_S(PP_64x128x128_c1x4x1_s4, 64, 128, 128, 1, 4, 1, 4)
DEFINE_PP_S(PP_64x128x128_c1x8x1_s3, 64, 128, 128, 1, 8, 1, 3)
DEFINE_PP_S(PP_64x128x128_c1x2x1_s3, 64, 128, 128, 1, 2, 1, 3)

DEFINE_WS(WS_64x256x64_c1x8x1,  64, 256,  64, 1, 8, 1)
DEFINE_WS(WS_64x256x64_c1x4x1,  64, 256,  64, 1, 4, 1)
DEFINE_WS(WS_64x256x64_c1x2x1,  64, 256,  64, 1, 2, 1)
DEFINE_WS(WS_64x256x64_c1x1x1,  64, 256,  64, 1, 1, 1)
DEFINE_WS(WS_64x256x128_c1x4x1, 64, 256, 128, 1, 4, 1)
DEFINE_WS(WS_64x256x128_c1x2x1, 64, 256, 128, 1, 2, 1)
DEFINE_WS(WS_64x128x64_c1x8x1,  64, 128,  64, 1, 8, 1)
DEFINE_WS(WS_64x128x64_c1x4x1,  64, 128,  64, 1, 4, 1)
DEFINE_WS(WS_64x128x64_c1x2x1,  64, 128,  64, 1, 2, 1)
DEFINE_WS(WS_64x128x64_c1x1x1,  64, 128,  64, 1, 1, 1)
DEFINE_WS(WS_64x128x128_c1x4x1, 64, 128, 128, 1, 4, 1)
DEFINE_WS(WS_64x128x128_c1x2x1, 64, 128, 128, 1, 2, 1)
DEFINE_WS(WS_64x64x64_c1x4x1,   64,  64,  64, 1, 4, 1)
DEFINE_WS(WS_64x64x64_c1x2x1,   64,  64,  64, 1, 2, 1)

DEFINE_WS_S(WS_64x256x64_c1x8x1_s3, 64, 256,  64, 1, 8, 1, 3)
DEFINE_WS_S(WS_64x256x64_c1x8x1_s4, 64, 256,  64, 1, 8, 1, 4)
DEFINE_WS_S(WS_64x256x64_c1x4x1_s3, 64, 256,  64, 1, 4, 1, 3)
DEFINE_WS_S(WS_64x256x64_c1x4x1_s4, 64, 256,  64, 1, 4, 1, 4)
DEFINE_WS_S(WS_64x256x64_c1x2x1_s3, 64, 256,  64, 1, 2, 1, 3)
DEFINE_WS_S(WS_64x256x64_c1x2x1_s4, 64, 256,  64, 1, 2, 1, 4)
DEFINE_WS_S(WS_64x128x64_c1x8x1_s3, 64, 128,  64, 1, 8, 1, 3)
DEFINE_WS_S(WS_64x128x64_c1x8x1_s4, 64, 128,  64, 1, 8, 1, 4)
DEFINE_WS_S(WS_64x128x64_c1x4x1_s3, 64, 128,  64, 1, 4, 1, 3)
DEFINE_WS_S(WS_64x128x64_c1x4x1_s4, 64, 128,  64, 1, 4, 1, 4)
DEFINE_WS_S(WS_64x128x64_c1x2x1_s3, 64, 128,  64, 1, 2, 1, 3)
DEFINE_WS_S(WS_64x128x64_c1x2x1_s4, 64, 128,  64, 1, 2, 1, 4)

struct GemmHandleBase {
  virtual ~GemmHandleBase() = default;
  virtual bool init(void* pA, void* pB, void* pC, void* pD,
                    int M, int N, int K,
                    cutlass::KernelHardwareInfo hw_info,
                    uint8_t* ws, size_t ws_size) = 0;
  virtual bool run() = 0;
  virtual size_t get_workspace_size(void* pA, void* pB, void* pC, void* pD,
                                    int M, int N, int K,
                                    cutlass::KernelHardwareInfo hw_info) = 0;
  virtual bool validate(void* pA, void* pB, void* pC, void* pD,
                        int M, int N, int K,
                        cutlass::KernelHardwareInfo hw_info,
                        uint8_t* ws, size_t ws_size) = 0;
  virtual const char* name() = 0;
};

template <typename GemmType>
struct TypedGemmHandle : GemmHandleBase {
  using Gemm    = typename GemmType::Gemm;
  using StrideA = typename GemmType::StrideA;
  using StrideB = typename GemmType::StrideB;
  using StrideC = typename GemmType::StrideC;
  using StrideD = typename GemmType::StrideD;

  Gemm gemm_obj;
  bool initialized_ = false;
  const char* name_ = nullptr;

  TypedGemmHandle(const char* n) : name_(n) {}

  const char* name() override { return name_; }

  typename Gemm::Arguments make_args(void* pA, void* pB, void* pC, void* pD,
                                      int M, int N, int K,
                                      cutlass::KernelHardwareInfo hw_info) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
      {reinterpret_cast<ElementA*>(pA), sA, reinterpret_cast<ElementB*>(pB), sB},
      {{1.0f, 0.0f},
       reinterpret_cast<ElementC*>(pC), sC,
       reinterpret_cast<ElementD*>(pD), sD},
      hw_info};
  }

  size_t get_workspace_size(void* pA, void* pB, void* pC, void* pD,
                             int M, int N, int K,
                             cutlass::KernelHardwareInfo hw_info) override {
    auto args = make_args(pA, pB, pC, pD, M, N, K, hw_info);
    if (gemm_obj.can_implement(args) != cutlass::Status::kSuccess) return size_t(-1);
    return Gemm::get_workspace_size(args);
  }

  bool validate(void* pA, void* pB, void* pC, void* pD,
                int M, int N, int K,
                cutlass::KernelHardwareInfo hw_info,
                uint8_t* ws, size_t ws_size) override {
    auto args = make_args(pA, pB, pC, pD, M, N, K, hw_info);
    if (gemm_obj.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t needed = Gemm::get_workspace_size(args);
    if (needed > ws_size) return false;
    cudaGetLastError();
    if (gemm_obj.initialize(args, ws) != cutlass::Status::kSuccess) return false;
    if (gemm_obj.run() != cutlass::Status::kSuccess) return false;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaGetLastError(); return false; }
    if (cudaGetLastError() != cudaSuccess) return false;
    initialized_ = true;
    return true;
  }

  bool init(void* pA, void* pB, void* pC, void* pD,
            int M, int N, int K,
            cutlass::KernelHardwareInfo hw_info,
            uint8_t* ws, size_t ws_size) override {
    auto args = make_args(pA, pB, pC, pD, M, N, K, hw_info);
    if (gemm_obj.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t needed = Gemm::get_workspace_size(args);
    if (needed > ws_size) return false;
    cudaGetLastError();
    if (gemm_obj.initialize(args, ws) != cutlass::Status::kSuccess) return false;
    initialized_ = true;
    return true;
  }

  bool run() override {
    if (!initialized_) return false;
    return gemm_obj.run() == cutlass::Status::kSuccess;
  }
};

typedef GemmHandleBase* (*HandleFactory)();

template <typename GemmType>
GemmHandleBase* make_handle_impl(const char* name) {
  return new TypedGemmHandle<GemmType>(name);
}

struct VariantDesc {
  HandleFactory factory;
  const char*   name;
};

#define V(Type) { []() -> GemmHandleBase* { return new TypedGemmHandle<Type>(#Type); }, #Type }

static VariantDesc g_variants[] = {
  V(PP_64x256x64_c1x8x1),
  V(PP_64x256x64_c1x4x1),
  V(PP_64x256x64_c1x2x1),
  V(PP_64x256x64_c1x1x1),
  V(PP_64x256x128_c1x8x1),
  V(PP_64x256x128_c1x4x1),
  V(PP_64x256x128_c1x2x1),
  V(PP_64x256x128_c1x1x1),
  V(PP_64x128x64_c1x8x1),
  V(PP_64x128x64_c1x4x1),
  V(PP_64x128x64_c1x2x1),
  V(PP_64x128x64_c1x1x1),
  V(PP_64x128x128_c1x8x1),
  V(PP_64x128x128_c1x4x1),
  V(PP_64x128x128_c1x2x1),
  V(PP_64x128x128_c1x1x1),
  V(PP_64x64x64_c1x8x1),
  V(PP_64x64x64_c1x4x1),
  V(PP_64x64x64_c1x2x1),
  V(PP_64x64x64_c1x1x1),
  V(PP_64x256x64_c1x8x1_s3),
  V(PP_64x256x64_c1x8x1_s4),
  V(PP_64x256x64_c1x8x1_s5),
  V(PP_64x256x64_c1x4x1_s3),
  V(PP_64x256x64_c1x4x1_s4),
  V(PP_64x256x64_c1x4x1_s5),
  V(PP_64x256x64_c1x2x1_s3),
  V(PP_64x256x64_c1x2x1_s4),
  V(PP_64x256x64_c1x2x1_s5),
  V(PP_64x256x64_c1x1x1_s3),
  V(PP_64x256x64_c1x1x1_s4),
  V(PP_64x256x64_c1x1x1_s5),
  V(PP_64x256x128_c1x4x1_s3),
  V(PP_64x256x128_c1x4x1_s4),
  V(PP_64x256x128_c1x8x1_s3),
  V(PP_64x256x128_c1x8x1_s4),
  V(PP_64x256x128_c1x2x1_s3),
  V(PP_64x256x128_c1x2x1_s4),
  V(PP_64x128x64_c1x8x1_s3),
  V(PP_64x128x64_c1x8x1_s4),
  V(PP_64x128x64_c1x8x1_s5),
  V(PP_64x128x64_c1x4x1_s3),
  V(PP_64x128x64_c1x4x1_s4),
  V(PP_64x128x64_c1x4x1_s5),
  V(PP_64x128x64_c1x2x1_s3),
  V(PP_64x128x64_c1x2x1_s4),
  V(PP_64x128x64_c1x2x1_s5),
  V(PP_64x128x128_c1x4x1_s3),
  V(PP_64x128x128_c1x4x1_s4),
  V(PP_64x128x128_c1x8x1_s3),
  V(PP_64x128x128_c1x2x1_s3),
  V(WS_64x256x64_c1x8x1),
  V(WS_64x256x64_c1x4x1),
  V(WS_64x256x64_c1x2x1),
  V(WS_64x256x64_c1x1x1),
  V(WS_64x256x128_c1x4x1),
  V(WS_64x256x128_c1x2x1),
  V(WS_64x128x64_c1x8x1),
  V(WS_64x128x64_c1x4x1),
  V(WS_64x128x64_c1x2x1),
  V(WS_64x128x64_c1x1x1),
  V(WS_64x128x128_c1x4x1),
  V(WS_64x128x128_c1x2x1),
  V(WS_64x64x64_c1x4x1),
  V(WS_64x64x64_c1x2x1),
  V(WS_64x256x64_c1x8x1_s3),
  V(WS_64x256x64_c1x8x1_s4),
  V(WS_64x256x64_c1x4x1_s3),
  V(WS_64x256x64_c1x4x1_s4),
  V(WS_64x256x64_c1x2x1_s3),
  V(WS_64x256x64_c1x2x1_s4),
  V(WS_64x128x64_c1x8x1_s3),
  V(WS_64x128x64_c1x8x1_s4),
  V(WS_64x128x64_c1x4x1_s3),
  V(WS_64x128x64_c1x4x1_s4),
  V(WS_64x128x64_c1x2x1_s3),
  V(WS_64x128x64_c1x2x1_s4),
};
#undef V

static constexpr int NUM_VARIANTS = sizeof(g_variants) / sizeof(g_variants[0]);

struct GlobalState {
  uint8_t* workspace_ptr  = nullptr;
  size_t   workspace_size = 0;
  GemmHandleBase* best_handle     = nullptr;
  int             best_variant_idx = -1;
  cutlass::KernelHardwareInfo hw_info_cached;
  bool hw_info_valid = false;
  void* last_pA = nullptr;
  void* last_pB = nullptr;
  void* last_pC = nullptr;

  ~GlobalState() {
    if (workspace_ptr) cudaFree(workspace_ptr);
    for (int i = 0; i < NUM_VARIANTS; i++) {
      if (g_handles_arr[i]) { delete g_handles_arr[i]; g_handles_arr[i] = nullptr; }
    }
  }

  GemmHandleBase* g_handles_arr[NUM_VARIANTS] = {};

  uint8_t* ensure_workspace(size_t needed) {
    if (needed > workspace_size) {
      if (workspace_ptr) cudaFree(workspace_ptr);
      cudaMalloc(&workspace_ptr, needed);
      workspace_size = needed;
    }
    return workspace_ptr;
  }

  cutlass::KernelHardwareInfo get_hw_info() {
    if (!hw_info_valid) {
      cudaGetDevice(&hw_info_cached.device_id);
      hw_info_cached.sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
              hw_info_cached.device_id);
      hw_info_valid = true;
    }
    return hw_info_cached;
  }
};

static GlobalState g_state;
static std::mutex  g_mutex;
static std::atomic<bool> g_tuned{false};

static void autotune_and_select(void* pA, void* pB, void* pC, void* pD,
                                 int M, int N, int K) {
  auto hw_info = g_state.get_hw_info();

  for (int i = 0; i < NUM_VARIANTS; i++) {
    if (!g_state.g_handles_arr[i]) {
      g_state.g_handles_arr[i] = g_variants[i].factory();
    }
  }

  size_t max_ws = 131072;
  for (int i = 0; i < NUM_VARIANTS; i++) {
    size_t ws = g_state.g_handles_arr[i]->get_workspace_size(pA, pB, pC, pD, M, N, K, hw_info);
    if (ws != size_t(-1) && ws > max_ws) max_ws = ws;
  }
  g_state.ensure_workspace(max_ws);
  cudaDeviceSynchronize();

  std::vector<int> valid_indices;
  valid_indices.reserve(NUM_VARIANTS);
  for (int i = 0; i < NUM_VARIANTS; i++) {
    cudaGetLastError();
    bool ok = g_state.g_handles_arr[i]->validate(
        pA, pB, pC, pD, M, N, K, hw_info,
        g_state.workspace_ptr, g_state.workspace_size);
    if (ok) {
      valid_indices.push_back(i);
    } else {
      cudaGetLastError();
    }
  }

  if (valid_indices.empty()) return;

  for (int i : valid_indices) {
    g_state.g_handles_arr[i]->init(pA, pB, pC, pD, M, N, K, hw_info,
                                    g_state.workspace_ptr, g_state.workspace_size);
  }
  cudaDeviceSynchronize();

  int   best_idx  = valid_indices[0];
  float best_time = std::numeric_limits<float>::max();

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  const int WARMUP = 8;
  const int BENCH  = 40;

  for (int i : valid_indices) {
    bool ok = g_state.g_handles_arr[i]->init(pA, pB, pC, pD, M, N, K, hw_info,
                                              g_state.workspace_ptr, g_state.workspace_size);
    if (!ok) continue;

    cudaGetLastError();
    for (int w = 0; w < WARMUP; w++) g_state.g_handles_arr[i]->run();
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); continue; }

    cudaEventRecord(ev_start);
    for (int it = 0; it < BENCH; it++) g_state.g_handles_arr[i]->run();
    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);

    if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); continue; }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    float avg = ms / BENCH;
    if (avg < best_time) { best_time = avg; best_idx = i; }
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  bool ok = g_state.g_handles_arr[best_idx]->init(
      pA, pB, pC, pD, M, N, K, hw_info,
      g_state.workspace_ptr, g_state.workspace_size);
  if (ok) {
    g_state.best_handle      = g_state.g_handles_arr[best_idx];
    g_state.best_variant_idx = best_idx;
    g_state.last_pA = pA;
    g_state.last_pB = pB;
    g_state.last_pC = pC;
  }
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  void* pA = a.data_ptr();
  void* pB = b_col_major.data_ptr();
  void* pC = c.data_ptr();
  void* pD = c.data_ptr();

  if (g_state.best_handle != nullptr &&
      g_state.last_pA == pA &&
      g_state.last_pB == pB &&
      g_state.last_pC == pC) {
    g_state.best_handle->run();
    return;
  }

  if (g_state.best_handle != nullptr && g_state.best_variant_idx >= 0) {
    auto hw_info = g_state.get_hw_info();
    bool ok = g_state.best_handle->init(pA, pB, pC, pD, M, N, K, hw_info,
                                         g_state.workspace_ptr, g_state.workspace_size);
    if (ok) {
      g_state.last_pA = pA;
      g_state.last_pB = pB;
      g_state.last_pC = pC;
      g_state.best_handle->run();
      return;
    }
    g_state.best_handle      = nullptr;
    g_state.best_variant_idx = -1;
  }

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_state.best_handle == nullptr) {
      autotune_and_select(pA, pB, pC, pD, M, N, K);
    }
  }

  if (g_state.best_handle != nullptr) {
    if (g_state.last_pA != pA || g_state.last_pB != pB || g_state.last_pC != pC) {
      auto hw_info = g_state.get_hw_info();
      g_state.best_handle->init(pA, pB, pC, pD, M, N, K, hw_info,
                                 g_state.workspace_ptr, g_state.workspace_size);
      g_state.last_pA = pA;
      g_state.last_pB = pB;
      g_state.last_pC = pC;
    }
    g_state.best_handle->run();
    return;
  }

  throw std::runtime_error("All GEMM variants failed: M=" + std::to_string(M) +
                            " N=" + std::to_string(N) + " K=" + std::to_string(K));
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}