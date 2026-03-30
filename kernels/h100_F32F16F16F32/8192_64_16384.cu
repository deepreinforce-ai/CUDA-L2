#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

using namespace nvcuda;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_COOP_STREAMK(Name, TM, TN, TK, GM, GN, GK)                        \
struct Name {                                                                      \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;       \
  using GridShape    = cute::Shape<cute::_##GM, cute::_##GN, cute::_##GK>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,      \
      ElementAccumulator, ElementCompute,                                          \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                      \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                      \
      ElementAccumulator, TileShape, GridShape,                                   \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::StreamKScheduler>;                                           \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                            \
  using StrideB = typename Gemm::GemmKernel::StrideB;                            \
  using StrideC = typename Gemm::GemmKernel::StrideC;                            \
  using StrideD = typename Gemm::GemmKernel::StrideD;                            \
};

#define DEFINE_COOP_PERSISTENT(Name, TM, TN, TK, GM, GN, GK)                     \
struct Name {                                                                      \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;       \
  using GridShape    = cute::Shape<cute::_##GM, cute::_##GN, cute::_##GK>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,      \
      ElementAccumulator, ElementCompute,                                          \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                      \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;\
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                      \
      ElementAccumulator, TileShape, GridShape,                                   \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                        \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                            \
  using StrideB = typename Gemm::GemmKernel::StrideB;                            \
  using StrideC = typename Gemm::GemmKernel::StrideC;                            \
  using StrideD = typename Gemm::GemmKernel::StrideD;                            \
};

#define DEFINE_PINGPONG_PERSISTENT(Name, TM, TN, TK, GM, GN, GK)                 \
struct Name {                                                                      \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;       \
  using GridShape    = cute::Shape<cute::_##GM, cute::_##GN, cute::_##GK>;       \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      TileShape, GridShape, cutlass::epilogue::collective::EpilogueTileAuto,      \
      ElementAccumulator, ElementCompute,                                          \
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,                      \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;           \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<\
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                        \
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,                      \
      ElementAccumulator, TileShape, GridShape,                                   \
      cutlass::gemm::collective::StageCountAutoCarveout<                          \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;             \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                       \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                        \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;       \
  using StrideA = typename Gemm::GemmKernel::StrideA;                            \
  using StrideB = typename Gemm::GemmKernel::StrideB;                            \
  using StrideC = typename Gemm::GemmKernel::StrideC;                            \
  using StrideD = typename Gemm::GemmKernel::StrideD;                            \
};

DEFINE_COOP_STREAMK(SK_G_128x64x128_G8,   128, 64, 128, 8, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x128_G8,   256, 64, 128, 8, 1, 1)
DEFINE_COOP_STREAMK(SK_G_128x64x64_G8,    128, 64,  64, 8, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x64_G8,    256, 64,  64, 8, 1, 1)

DEFINE_COOP_STREAMK(SK_G_128x64x128_G4,   128, 64, 128, 4, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x128_G4,   256, 64, 128, 4, 1, 1)
DEFINE_COOP_STREAMK(SK_G_128x64x64_G4,    128, 64,  64, 4, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x64_G4,    256, 64,  64, 4, 1, 1)

DEFINE_COOP_STREAMK(SK_G_128x64x128_G2,   128, 64, 128, 2, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x128_G2,   256, 64, 128, 2, 1, 1)
DEFINE_COOP_STREAMK(SK_G_128x64x64_G2,    128, 64,  64, 2, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x64_G2,    256, 64,  64, 2, 1, 1)
DEFINE_COOP_STREAMK(SK_G_128x64x128_G1,   128, 64, 128, 1, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x128_G1,   256, 64, 128, 1, 1, 1)
DEFINE_COOP_STREAMK(SK_G_128x64x64_G1,    128, 64,  64, 1, 1, 1)
DEFINE_COOP_STREAMK(SK_G_256x64x64_G1,    256, 64,  64, 1, 1, 1)

DEFINE_COOP_PERSISTENT(P_G_128x64x128_G8,  128, 64, 128, 8, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x128_G8,  256, 64, 128, 8, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_128x64x64_G8,   128, 64,  64, 8, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x64_G8,   256, 64,  64, 8, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_128x64x128_G4,  128, 64, 128, 4, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x128_G4,  256, 64, 128, 4, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_128x64x64_G4,   128, 64,  64, 4, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x64_G4,   256, 64,  64, 4, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_128x64x128_G2,  128, 64, 128, 2, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x128_G2,  256, 64, 128, 2, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_128x64x64_G2,   128, 64,  64, 2, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x64_G2,   256, 64,  64, 2, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_128x64x128_G1,  128, 64, 128, 1, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x128_G1,  256, 64, 128, 1, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_128x64x64_G1,   128, 64,  64, 1, 1, 1)
DEFINE_COOP_PERSISTENT(P_G_256x64x64_G1,   256, 64,  64, 1, 1, 1)

DEFINE_PINGPONG_PERSISTENT(PP_128x64x128_G8, 128, 64, 128, 8, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_256x64x128_G8, 256, 64, 128, 8, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_128x64x64_G8,  128, 64,  64, 8, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_128x64x128_G4, 128, 64, 128, 4, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_256x64x128_G4, 256, 64, 128, 4, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_128x64x64_G4,  128, 64,  64, 4, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_256x64x64_G4,  256, 64,  64, 4, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_64x64x128_G4,   64, 64, 128, 4, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_128x64x128_G2, 128, 64, 128, 2, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_256x64x128_G2, 256, 64, 128, 2, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_128x64x64_G2,  128, 64,  64, 2, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_64x64x128_G2,   64, 64, 128, 2, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_256x64x64_G2,  256, 64,  64, 2, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_128x64x128_G1, 128, 64, 128, 1, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_256x64x128_G1, 256, 64, 128, 1, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_128x64x64_G1,  128, 64,  64, 1, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_256x64x64_G1,  256, 64,  64, 1, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_64x64x128_G1,   64, 64, 128, 1, 1, 1)
DEFINE_PINGPONG_PERSISTENT(PP_64x64x64_G1,    64, 64,  64, 1, 1, 1)

struct StaticWS {
    uint8_t* ptr = nullptr;
    size_t   sz  = 0;
    void ensure(size_t need) {
        if (sz < need) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, need);
            sz = need;
        }
    }
};
static StaticWS g_ws;

template <typename HgemmType>
bool try_run_gemm(const void* ptr_A, const void* ptr_B, void* ptr_C,
                  int M, int N, int K,
                  const cutlass::KernelHardwareInfo& hw_info) {
    using Gemm    = typename HgemmType::Gemm;
    using StrideA = typename HgemmType::StrideA;
    using StrideB = typename HgemmType::StrideB;
    using StrideC = typename HgemmType::StrideC;
    using StrideD = typename HgemmType::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto* pA = reinterpret_cast<const ElementA*>(ptr_A);
    auto* pB = reinterpret_cast<const ElementB*>(ptr_B);
    auto* pC = reinterpret_cast<ElementC*>(ptr_C);
    auto* pD = reinterpret_cast<ElementD*>(ptr_C);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {pA, stride_A, pB, stride_B},
        {{1.0f, 0.0f}, pC, stride_C, pD, stride_D},
        hw_info
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    g_ws.ensure(workspace_size > 0 ? workspace_size : 1);

    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
    if (gemm.initialize(arguments, g_ws.ptr) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

enum CfgID {
    CFG_SK_128x64x128_G8 = 0,
    CFG_SK_256x64x128_G8,
    CFG_SK_128x64x64_G8,
    CFG_SK_256x64x64_G8,
    CFG_SK_128x64x128_G4,
    CFG_SK_256x64x128_G4,
    CFG_SK_128x64x64_G4,
    CFG_SK_256x64x64_G4,
    CFG_SK_128x64x128_G2,
    CFG_SK_256x64x128_G2,
    CFG_SK_128x64x64_G2,
    CFG_SK_256x64x64_G2,
    CFG_SK_128x64x128_G1,
    CFG_SK_256x64x128_G1,
    CFG_SK_128x64x64_G1,
    CFG_SK_256x64x64_G1,
    CFG_P_128x64x128_G8,
    CFG_P_256x64x128_G8,
    CFG_P_128x64x64_G8,
    CFG_P_256x64x64_G8,
    CFG_P_128x64x128_G4,
    CFG_P_256x64x128_G4,
    CFG_P_128x64x64_G4,
    CFG_P_256x64x64_G4,
    CFG_P_128x64x128_G2,
    CFG_P_256x64x128_G2,
    CFG_P_128x64x64_G2,
    CFG_P_256x64x64_G2,
    CFG_P_128x64x128_G1,
    CFG_P_256x64x128_G1,
    CFG_P_128x64x64_G1,
    CFG_P_256x64x64_G1,
    CFG_PP_128x64x128_G8,
    CFG_PP_256x64x128_G8,
    CFG_PP_128x64x64_G8,
    CFG_PP_128x64x128_G4,
    CFG_PP_256x64x128_G4,
    CFG_PP_128x64x64_G4,
    CFG_PP_256x64x64_G4,
    CFG_PP_64x64x128_G4,
    CFG_PP_128x64x128_G2,
    CFG_PP_256x64x128_G2,
    CFG_PP_128x64x64_G2,
    CFG_PP_64x64x128_G2,
    CFG_PP_256x64x64_G2,
    CFG_PP_128x64x128_G1,
    CFG_PP_256x64x128_G1,
    CFG_PP_128x64x64_G1,
    CFG_PP_256x64x64_G1,
    CFG_PP_64x64x128_G1,
    CFG_PP_64x64x64_G1,
    CFG_COUNT
};

static bool dispatch_config(int cfg, const void* pA, const void* pB, void* pC,
                             int M, int N, int K,
                             const cutlass::KernelHardwareInfo& hw_info) {
    switch (cfg) {
        case CFG_SK_128x64x128_G8: return try_run_gemm<SK_G_128x64x128_G8>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x128_G8: return try_run_gemm<SK_G_256x64x128_G8>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_128x64x64_G8:  return try_run_gemm<SK_G_128x64x64_G8> (pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x64_G8:  return try_run_gemm<SK_G_256x64x64_G8> (pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_128x64x128_G4: return try_run_gemm<SK_G_128x64x128_G4>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x128_G4: return try_run_gemm<SK_G_256x64x128_G4>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_128x64x64_G4:  return try_run_gemm<SK_G_128x64x64_G4> (pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x64_G4:  return try_run_gemm<SK_G_256x64x64_G4> (pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_128x64x128_G2: return try_run_gemm<SK_G_128x64x128_G2>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x128_G2: return try_run_gemm<SK_G_256x64x128_G2>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_128x64x64_G2:  return try_run_gemm<SK_G_128x64x64_G2> (pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x64_G2:  return try_run_gemm<SK_G_256x64x64_G2> (pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_128x64x128_G1: return try_run_gemm<SK_G_128x64x128_G1>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x128_G1: return try_run_gemm<SK_G_256x64x128_G1>(pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_128x64x64_G1:  return try_run_gemm<SK_G_128x64x64_G1> (pA,pB,pC,M,N,K,hw_info);
        case CFG_SK_256x64x64_G1:  return try_run_gemm<SK_G_256x64x64_G1> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x128_G8:  return try_run_gemm<P_G_128x64x128_G8> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x128_G8:  return try_run_gemm<P_G_256x64x128_G8> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x64_G8:   return try_run_gemm<P_G_128x64x64_G8>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x64_G8:   return try_run_gemm<P_G_256x64x64_G8>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x128_G4:  return try_run_gemm<P_G_128x64x128_G4> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x128_G4:  return try_run_gemm<P_G_256x64x128_G4> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x64_G4:   return try_run_gemm<P_G_128x64x64_G4>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x64_G4:   return try_run_gemm<P_G_256x64x64_G4>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x128_G2:  return try_run_gemm<P_G_128x64x128_G2> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x128_G2:  return try_run_gemm<P_G_256x64x128_G2> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x64_G2:   return try_run_gemm<P_G_128x64x64_G2>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x64_G2:   return try_run_gemm<P_G_256x64x64_G2>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x128_G1:  return try_run_gemm<P_G_128x64x128_G1> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x128_G1:  return try_run_gemm<P_G_256x64x128_G1> (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_128x64x64_G1:   return try_run_gemm<P_G_128x64x64_G1>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_P_256x64x64_G1:   return try_run_gemm<P_G_256x64x64_G1>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x128_G8: return try_run_gemm<PP_128x64x128_G8>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_256x64x128_G8: return try_run_gemm<PP_256x64x128_G8>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x64_G8:  return try_run_gemm<PP_128x64x64_G8>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x128_G4: return try_run_gemm<PP_128x64x128_G4>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_256x64x128_G4: return try_run_gemm<PP_256x64x128_G4>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x64_G4:  return try_run_gemm<PP_128x64x64_G4>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_256x64x64_G4:  return try_run_gemm<PP_256x64x64_G4>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_64x64x128_G4:  return try_run_gemm<PP_64x64x128_G4>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x128_G2: return try_run_gemm<PP_128x64x128_G2>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_256x64x128_G2: return try_run_gemm<PP_256x64x128_G2>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x64_G2:  return try_run_gemm<PP_128x64x64_G2>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_64x64x128_G2:  return try_run_gemm<PP_64x64x128_G2>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_256x64x64_G2:  return try_run_gemm<PP_256x64x64_G2>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x128_G1: return try_run_gemm<PP_128x64x128_G1>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_256x64x128_G1: return try_run_gemm<PP_256x64x128_G1>  (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_128x64x64_G1:  return try_run_gemm<PP_128x64x64_G1>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_256x64x64_G1:  return try_run_gemm<PP_256x64x64_G1>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_64x64x128_G1:  return try_run_gemm<PP_64x64x128_G1>   (pA,pB,pC,M,N,K,hw_info);
        case CFG_PP_64x64x64_G1:   return try_run_gemm<PP_64x64x64_G1>    (pA,pB,pC,M,N,K,hw_info);
        default: return false;
    }
}

static const int CANDIDATES[] = {
    CFG_SK_128x64x128_G8,
    CFG_SK_256x64x128_G8,
    CFG_SK_128x64x64_G8,
    CFG_SK_256x64x64_G8,
    CFG_SK_128x64x128_G4,
    CFG_SK_256x64x128_G4,
    CFG_SK_128x64x64_G4,
    CFG_SK_256x64x64_G4,
    CFG_SK_128x64x128_G2,
    CFG_SK_256x64x128_G2,
    CFG_SK_128x64x64_G2,
    CFG_SK_256x64x64_G2,
    CFG_SK_128x64x128_G1,
    CFG_SK_256x64x128_G1,
    CFG_SK_128x64x64_G1,
    CFG_SK_256x64x64_G1,
    CFG_P_128x64x128_G8,
    CFG_P_256x64x128_G8,
    CFG_P_128x64x64_G8,
    CFG_P_256x64x64_G8,
    CFG_P_128x64x128_G4,
    CFG_P_256x64x128_G4,
    CFG_P_128x64x64_G4,
    CFG_P_256x64x64_G4,
    CFG_P_128x64x128_G2,
    CFG_P_256x64x128_G2,
    CFG_P_128x64x64_G2,
    CFG_P_256x64x64_G2,
    CFG_P_128x64x128_G1,
    CFG_P_256x64x128_G1,
    CFG_P_128x64x64_G1,
    CFG_P_256x64x64_G1,
    CFG_PP_128x64x128_G8,
    CFG_PP_256x64x128_G8,
    CFG_PP_128x64x64_G8,
    CFG_PP_128x64x128_G4,
    CFG_PP_256x64x128_G4,
    CFG_PP_128x64x64_G4,
    CFG_PP_256x64x64_G4,
    CFG_PP_64x64x128_G4,
    CFG_PP_128x64x128_G2,
    CFG_PP_256x64x128_G2,
    CFG_PP_128x64x64_G2,
    CFG_PP_64x64x128_G2,
    CFG_PP_256x64x64_G2,
    CFG_PP_128x64x128_G1,
    CFG_PP_256x64x128_G1,
    CFG_PP_128x64x64_G1,
    CFG_PP_256x64x64_G1,
    CFG_PP_64x64x128_G1,
    CFG_PP_64x64x64_G1,
};
static const int NUM_CANDIDATES = (int)(sizeof(CANDIDATES) / sizeof(CANDIDATES[0]));

static int g_best_cfg = -1;

static int do_autotune(const void* pA, const void* pB, void* pC,
                       int M, int N, int K,
                       const cutlass::KernelHardwareInfo& hw_info) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    float best_ms = 1e18f;
    int   best_cfg = -1;

    for (int ci = 0; ci < NUM_CANDIDATES; ci++) {
        int cfg = CANDIDATES[ci];

        if (!dispatch_config(cfg, pA, pB, pC, M, N, K, hw_info)) continue;
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { cudaGetLastError(); continue; }

        for (int w = 0; w < 3; w++)
            dispatch_config(cfg, pA, pB, pC, M, N, K, hw_info);
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        for (int r = 0; r < 8; r++)
            dispatch_config(cfg, pA, pB, pC, M, N, K, hw_info);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);

        if (ms < best_ms) {
            best_ms  = ms;
            best_cfg = cfg;
        }
    }

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    if (best_cfg < 0) {
        for (int ci = 0; ci < NUM_CANDIDATES; ci++) {
            if (dispatch_config(CANDIDATES[ci], pA, pB, pC, M, N, K, hw_info)) {
                best_cfg = CANDIDATES[ci];
                break;
            }
        }
    }
    return best_cfg;
}

#endif

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
    int device_id = 0;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    const void* pA = a.data_ptr();
    const void* pB = b_col_major.data_ptr();
    void*       pC = c.data_ptr();

    if (g_best_cfg < 0) {
        g_best_cfg = do_autotune(pA, pB, pC, M, N, K, hw_info);
    }

    if (g_best_cfg >= 0 && dispatch_config(g_best_cfg, pA, pB, pC, M, N, K, hw_info)) return;

    g_best_cfg = -1;
    for (int ci = 0; ci < NUM_CANDIDATES; ci++) {
        if (dispatch_config(CANDIDATES[ci], pA, pB, pC, M, N, K, hw_info)) {
            g_best_cfg = CANDIDATES[ci];
            return;
        }
    }

    throw std::runtime_error("All GEMM configurations failed");

#else
    throw std::runtime_error("SM90 not supported — requires CUTLASS_ARCH_MMA_SM90_SUPPORTED");
#endif
}