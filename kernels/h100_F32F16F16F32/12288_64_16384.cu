#include <iostream>
#include <stdexcept>
#include <string>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

using ElementA           = cutlass::half_t;
using LayoutA            = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB           = cutlass::half_t;
using LayoutB            = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC           = cutlass::half_t;
using LayoutC            = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

#define MAKE_GEMM_PP(NS, TM, TN, TK, CM, CN, CK)                                \
namespace NS {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OperatorClass, TileShape, GroupShape,                           \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementAccumulator,                                    \
      ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,             \
      cutlass::epilogue::TmaWarpSpecialized                                      \
  >::CollectiveOp;                                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OperatorClass,                                                     \
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,             \
      ElementAccumulator, TileShape, GroupShape,                                \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong                            \
  >::CollectiveOp;                                                                \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;         \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define MAKE_GEMM_WS(NS, TM, TN, TK, CM, CN, CK)                                \
namespace NS {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OperatorClass, TileShape, GroupShape,                           \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementAccumulator,                                    \
      ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,             \
      cutlass::epilogue::TmaWarpSpecialized                                      \
  >::CollectiveOp;                                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OperatorClass,                                                     \
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,             \
      ElementAccumulator, TileShape, GroupShape,                                \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecialized                                    \
  >::CollectiveOp;                                                                \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;         \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define MAKE_GEMM_COOP(NS, TM, TN, TK, CM, CN, CK)                              \
namespace NS {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OperatorClass, TileShape, GroupShape,                           \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementAccumulator,                                    \
      ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,             \
      cutlass::epilogue::TmaWarpSpecializedCooperative                           \
  >::CollectiveOp;                                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OperatorClass,                                                     \
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,             \
      ElementAccumulator, TileShape, GroupShape,                                \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative                         \
  >::CollectiveOp;                                                                \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;         \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

#define MAKE_GEMM_AUTO(NS, TM, TN, TK, CM, CN, CK)                              \
namespace NS {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;      \
  using GroupShape = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;      \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OperatorClass, TileShape, GroupShape,                           \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementAccumulator,                                    \
      ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,             \
      cutlass::epilogue::collective::EpilogueScheduleAuto                        \
  >::CollectiveOp;                                                                \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OperatorClass,                                                     \
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,             \
      ElementAccumulator, TileShape, GroupShape,                                \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::collective::KernelScheduleAuto                              \
  >::CollectiveOp;                                                                \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;         \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
}

MAKE_GEMM_PP  (pp_64_64_256_c8,      64,  64, 256,  8, 1, 1)
MAKE_GEMM_PP  (pp_128_64_256_c8,    128,  64, 256,  8, 1, 1)
MAKE_GEMM_WS  (ws_64_64_256_c8,      64,  64, 256,  8, 1, 1)
MAKE_GEMM_WS  (ws_128_64_256_c8,    128,  64, 256,  8, 1, 1)
MAKE_GEMM_COOP(coop_128_64_256_c8,  128,  64, 256,  8, 1, 1)
MAKE_GEMM_AUTO(auto_64_64_256_c8,    64,  64, 256,  8, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_256_c8,  128,  64, 256,  8, 1, 1)

MAKE_GEMM_PP  (pp_64_64_128_c8,      64,  64, 128,  8, 1, 1)
MAKE_GEMM_PP  (pp_128_64_128_c8,    128,  64, 128,  8, 1, 1)
MAKE_GEMM_WS  (ws_64_64_128_c8,      64,  64, 128,  8, 1, 1)
MAKE_GEMM_WS  (ws_128_64_128_c8,    128,  64, 128,  8, 1, 1)
MAKE_GEMM_COOP(coop_128_64_128_c8,  128,  64, 128,  8, 1, 1)
MAKE_GEMM_AUTO(auto_64_64_128_c8,    64,  64, 128,  8, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_128_c8,  128,  64, 128,  8, 1, 1)

MAKE_GEMM_PP  (pp_64_64_64_c8,       64,  64,  64,  8, 1, 1)
MAKE_GEMM_PP  (pp_128_64_64_c8,     128,  64,  64,  8, 1, 1)
MAKE_GEMM_WS  (ws_64_64_64_c8,       64,  64,  64,  8, 1, 1)
MAKE_GEMM_AUTO(auto_64_64_64_c8,     64,  64,  64,  8, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_64_c8,   128,  64,  64,  8, 1, 1)

MAKE_GEMM_PP  (pp_64_64_256_c4x2,    64,  64, 256,  4, 2, 1)
MAKE_GEMM_PP  (pp_128_64_256_c4x2,  128,  64, 256,  4, 2, 1)
MAKE_GEMM_COOP(coop_128_64_256_c4x2,128,  64, 256,  4, 2, 1)
MAKE_GEMM_AUTO(auto_64_64_256_c4x2,  64,  64, 256,  4, 2, 1)
MAKE_GEMM_AUTO(auto_128_64_256_c4x2,128,  64, 256,  4, 2, 1)

MAKE_GEMM_PP  (pp_64_64_128_c4x2,    64,  64, 128,  4, 2, 1)
MAKE_GEMM_PP  (pp_128_64_128_c4x2,  128,  64, 128,  4, 2, 1)
MAKE_GEMM_WS  (ws_64_64_128_c4x2,    64,  64, 128,  4, 2, 1)
MAKE_GEMM_WS  (ws_128_64_128_c4x2,  128,  64, 128,  4, 2, 1)
MAKE_GEMM_COOP(coop_128_64_128_c4x2,128,  64, 128,  4, 2, 1)
MAKE_GEMM_AUTO(auto_64_64_128_c4x2,  64,  64, 128,  4, 2, 1)
MAKE_GEMM_AUTO(auto_128_64_128_c4x2,128,  64, 128,  4, 2, 1)

MAKE_GEMM_PP  (pp_64_64_256_c2x2,    64,  64, 256,  2, 2, 1)
MAKE_GEMM_PP  (pp_128_64_256_c2x2,  128,  64, 256,  2, 2, 1)
MAKE_GEMM_PP  (pp_64_64_128_c2x2,    64,  64, 128,  2, 2, 1)
MAKE_GEMM_PP  (pp_128_64_128_c2x2,  128,  64, 128,  2, 2, 1)
MAKE_GEMM_WS  (ws_128_64_128_c2x2,  128,  64, 128,  2, 2, 1)
MAKE_GEMM_AUTO(auto_64_64_128_c2x2,  64,  64, 128,  2, 2, 1)
MAKE_GEMM_AUTO(auto_128_64_128_c2x2,128,  64, 128,  2, 2, 1)

MAKE_GEMM_PP  (pp_64_64_256_c4,      64,  64, 256,  4, 1, 1)
MAKE_GEMM_PP  (pp_128_64_256_c4,    128,  64, 256,  4, 1, 1)
MAKE_GEMM_PP  (pp_64_64_128_c4,      64,  64, 128,  4, 1, 1)
MAKE_GEMM_PP  (pp_128_64_128_c4,    128,  64, 128,  4, 1, 1)
MAKE_GEMM_PP  (pp_256_64_128_c4,    256,  64, 128,  4, 1, 1)
MAKE_GEMM_WS  (ws_128_64_128_c4,    128,  64, 128,  4, 1, 1)
MAKE_GEMM_WS  (ws_128_64_256_c4,    128,  64, 256,  4, 1, 1)
MAKE_GEMM_WS  (ws_256_64_128_c4,    256,  64, 128,  4, 1, 1)
MAKE_GEMM_COOP(coop_128_64_128_c4,  128,  64, 128,  4, 1, 1)
MAKE_GEMM_COOP(coop_128_64_256_c4,  128,  64, 256,  4, 1, 1)
MAKE_GEMM_COOP(coop_256_64_128_c4,  256,  64, 128,  4, 1, 1)
MAKE_GEMM_AUTO(auto_64_64_128_c4,    64,  64, 128,  4, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_128_c4,  128,  64, 128,  4, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_256_c4,  128,  64, 256,  4, 1, 1)
MAKE_GEMM_AUTO(auto_256_64_128_c4,  256,  64, 128,  4, 1, 1)

MAKE_GEMM_PP  (pp_64_64_256_c2,      64,  64, 256,  2, 1, 1)
MAKE_GEMM_PP  (pp_128_64_256_c2,    128,  64, 256,  2, 1, 1)
MAKE_GEMM_PP  (pp_128_64_128_c2,    128,  64, 128,  2, 1, 1)
MAKE_GEMM_PP  (pp_256_64_128_c2,    256,  64, 128,  2, 1, 1)
MAKE_GEMM_WS  (ws_128_64_128_c2,    128,  64, 128,  2, 1, 1)
MAKE_GEMM_WS  (ws_128_64_256_c2,    128,  64, 256,  2, 1, 1)
MAKE_GEMM_WS  (ws_256_64_128_c2,    256,  64, 128,  2, 1, 1)
MAKE_GEMM_COOP(coop_128_64_128_c2,  128,  64, 128,  2, 1, 1)
MAKE_GEMM_COOP(coop_128_64_256_c2,  128,  64, 256,  2, 1, 1)
MAKE_GEMM_COOP(coop_256_64_128_c2,  256,  64, 128,  2, 1, 1)
MAKE_GEMM_AUTO(auto_64_64_128_c2,    64,  64, 128,  2, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_128_c2,  128,  64, 128,  2, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_256_c2,  128,  64, 256,  2, 1, 1)
MAKE_GEMM_AUTO(auto_256_64_128_c2,  256,  64, 128,  2, 1, 1)

MAKE_GEMM_PP  (pp_128_64_128_c1,    128,  64, 128,  1, 1, 1)
MAKE_GEMM_PP  (pp_64_64_128_c1,      64,  64, 128,  1, 1, 1)
MAKE_GEMM_WS  (ws_128_64_128_c1,    128,  64, 128,  1, 1, 1)
MAKE_GEMM_COOP(coop_128_64_128_c1,  128,  64, 128,  1, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_128_c1,  128,  64, 128,  1, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_64_c4,   128,  64,  64,  4, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_64_c2,   128,  64,  64,  2, 1, 1)
MAKE_GEMM_AUTO(auto_128_64_64_c1,   128,  64,  64,  1, 1, 1)

struct PersistentRunner {
    virtual ~PersistentRunner() = default;
    virtual bool run(int M, int N, int K,
                     cutlass::half_t* A, cutlass::half_t* B,
                     cutlass::half_t* C, cutlass::half_t* D,
                     float alpha, float beta) = 0;
};

template<typename GemmType>
struct TypedRunner : public PersistentRunner {
    GemmType gemm;
    cutlass::device_memory::allocation<uint8_t> workspace;
    int device_id{0};
    int sm_count{0};

    bool initialize(int M, int N, int K,
                    cutlass::half_t* A, cutlass::half_t* B,
                    cutlass::half_t* C, cutlass::half_t* D,
                    float alpha, float beta, int dev, int smc)
    {
        using StrideA = typename GemmType::GemmKernel::StrideA;
        using StrideB = typename GemmType::GemmKernel::StrideB;
        using StrideC = typename GemmType::GemmKernel::StrideC;
        using StrideD = typename GemmType::GemmKernel::StrideD;

        StrideA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
        StrideD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = dev;
        hw_info.sm_count  = smc;

        typename GemmType::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {A, sA, B, sB},
            {{alpha, beta}, C, sC, D, sD},
            hw_info
        };

        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

        size_t ws_size = GemmType::get_workspace_size(args);
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);

        if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;

        device_id = dev; sm_count = smc;
        return cudaGetLastError() == cudaSuccess;
    }

    bool run(int M, int N, int K,
             cutlass::half_t* A, cutlass::half_t* B,
             cutlass::half_t* C, cutlass::half_t* D,
             float alpha, float beta) override
    {
        using StrideA = typename GemmType::GemmKernel::StrideA;
        using StrideB = typename GemmType::GemmKernel::StrideB;
        using StrideC = typename GemmType::GemmKernel::StrideC;
        using StrideD = typename GemmType::GemmKernel::StrideD;

        StrideA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
        StrideC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
        StrideD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = device_id;
        hw_info.sm_count  = sm_count;

        typename GemmType::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            {A, sA, B, sB},
            {{alpha, beta}, C, sC, D, sD},
            hw_info
        };

        cutlass::Status st = gemm.update(args, workspace.get());
        if (st != cutlass::Status::kSuccess) {
            st = gemm.initialize(args, workspace.get());
            if (st != cutlass::Status::kSuccess) return false;
        }
        if (gemm.run() != cutlass::Status::kSuccess) return false;
        return cudaGetLastError() == cudaSuccess;
    }
};

template<typename GemmType>
static bool run_gemm_impl(
    int M, int N, int K,
    cutlass::half_t* ptr_A, cutlass::half_t* ptr_B,
    cutlass::half_t* ptr_C, cutlass::half_t* ptr_D,
    float alpha, float beta, int device_id, int sm_count)
{
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    StrideA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    StrideD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = sm_count;

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, sA, ptr_B, sB},
        {{alpha, beta}, ptr_C, sC, ptr_D, sD},
        hw_info
    };

    GemmType gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws_size = GemmType::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> ws(ws_size);
    if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

template<typename GemmType>
static float time_gemm_impl(
    int M, int N, int K,
    cutlass::half_t* ptr_A, cutlass::half_t* ptr_B,
    cutlass::half_t* ptr_C, cutlass::half_t* ptr_D,
    float alpha, float beta, int device_id, int sm_count,
    int warmup = 3, int iters = 10)
{
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    StrideA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    StrideD sD = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = sm_count;

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, sA, ptr_B, sB},
        {{alpha, beta}, ptr_C, sC, ptr_D, sD},
        hw_info
    };

    GemmType gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1.f;
    size_t ws_size = GemmType::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> ws(ws_size);
    if (gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) return -1.f;

    for (int i = 0; i < warmup; i++) {
        if (gemm.run() != cutlass::Status::kSuccess) return -1.f;
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return -1.f;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; i++) gemm.run();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (cudaGetLastError() != cudaSuccess) return -1.f;
    return ms / iters;
}

static bool dispatch_run(int idx,
    int M, int N, int K,
    cutlass::half_t* A, cutlass::half_t* B,
    cutlass::half_t* C, cutlass::half_t* D,
    float alpha, float beta, int dev, int sm)
{
#define RUN(ns) return run_gemm_impl<ns::Gemm>(M,N,K,A,B,C,D,alpha,beta,dev,sm)
    switch (idx) {
        case  1: RUN(pp_64_64_256_c8);
        case  2: RUN(pp_128_64_256_c8);
        case  3: RUN(ws_64_64_256_c8);
        case  4: RUN(ws_128_64_256_c8);
        case  5: RUN(coop_128_64_256_c8);
        case  6: RUN(auto_64_64_256_c8);
        case  7: RUN(auto_128_64_256_c8);
        case  8: RUN(pp_64_64_128_c8);
        case  9: RUN(pp_128_64_128_c8);
        case 10: RUN(ws_64_64_128_c8);
        case 11: RUN(ws_128_64_128_c8);
        case 12: RUN(coop_128_64_128_c8);
        case 13: RUN(auto_64_64_128_c8);
        case 14: RUN(auto_128_64_128_c8);
        case 15: RUN(pp_64_64_64_c8);
        case 16: RUN(pp_128_64_64_c8);
        case 17: RUN(ws_64_64_64_c8);
        case 18: RUN(auto_64_64_64_c8);
        case 19: RUN(auto_128_64_64_c8);
        case 20: RUN(pp_64_64_256_c4x2);
        case 21: RUN(pp_128_64_256_c4x2);
        case 22: RUN(coop_128_64_256_c4x2);
        case 23: RUN(auto_64_64_256_c4x2);
        case 24: RUN(auto_128_64_256_c4x2);
        case 25: RUN(pp_64_64_128_c4x2);
        case 26: RUN(pp_128_64_128_c4x2);
        case 27: RUN(ws_64_64_128_c4x2);
        case 28: RUN(ws_128_64_128_c4x2);
        case 29: RUN(coop_128_64_128_c4x2);
        case 30: RUN(auto_64_64_128_c4x2);
        case 31: RUN(auto_128_64_128_c4x2);
        case 32: RUN(pp_64_64_256_c2x2);
        case 33: RUN(pp_128_64_256_c2x2);
        case 34: RUN(pp_64_64_128_c2x2);
        case 35: RUN(pp_128_64_128_c2x2);
        case 36: RUN(ws_128_64_128_c2x2);
        case 37: RUN(auto_64_64_128_c2x2);
        case 38: RUN(auto_128_64_128_c2x2);
        case 39: RUN(pp_64_64_256_c4);
        case 40: RUN(pp_128_64_256_c4);
        case 41: RUN(pp_64_64_128_c4);
        case 42: RUN(pp_128_64_128_c4);
        case 43: RUN(pp_256_64_128_c4);
        case 44: RUN(ws_128_64_128_c4);
        case 45: RUN(ws_128_64_256_c4);
        case 46: RUN(ws_256_64_128_c4);
        case 47: RUN(coop_128_64_128_c4);
        case 48: RUN(coop_128_64_256_c4);
        case 49: RUN(coop_256_64_128_c4);
        case 50: RUN(auto_64_64_128_c4);
        case 51: RUN(auto_128_64_128_c4);
        case 52: RUN(auto_128_64_256_c4);
        case 53: RUN(auto_256_64_128_c4);
        case 54: RUN(pp_64_64_256_c2);
        case 55: RUN(pp_128_64_256_c2);
        case 56: RUN(pp_128_64_128_c2);
        case 57: RUN(pp_256_64_128_c2);
        case 58: RUN(ws_128_64_128_c2);
        case 59: RUN(ws_128_64_256_c2);
        case 60: RUN(ws_256_64_128_c2);
        case 61: RUN(coop_128_64_128_c2);
        case 62: RUN(coop_128_64_256_c2);
        case 63: RUN(coop_256_64_128_c2);
        case 64: RUN(auto_64_64_128_c2);
        case 65: RUN(auto_128_64_128_c2);
        case 66: RUN(auto_128_64_256_c2);
        case 67: RUN(auto_256_64_128_c2);
        case 68: RUN(pp_128_64_128_c1);
        case 69: RUN(pp_64_64_128_c1);
        case 70: RUN(ws_128_64_128_c1);
        case 71: RUN(coop_128_64_128_c1);
        case 72: RUN(auto_128_64_128_c1);
        case 73: RUN(auto_128_64_64_c4);
        case 74: RUN(auto_128_64_64_c2);
        case 75: RUN(auto_128_64_64_c1);
        default: return false;
    }
#undef RUN
}

static float dispatch_time(int idx,
    int M, int N, int K,
    cutlass::half_t* A, cutlass::half_t* B,
    cutlass::half_t* C, cutlass::half_t* D,
    float alpha, float beta, int dev, int sm,
    int warmup = 3, int iters = 10)
{
#define TIME(ns) return time_gemm_impl<ns::Gemm>(M,N,K,A,B,C,D,alpha,beta,dev,sm,warmup,iters)
    switch (idx) {
        case  1: TIME(pp_64_64_256_c8);
        case  2: TIME(pp_128_64_256_c8);
        case  3: TIME(ws_64_64_256_c8);
        case  4: TIME(ws_128_64_256_c8);
        case  5: TIME(coop_128_64_256_c8);
        case  6: TIME(auto_64_64_256_c8);
        case  7: TIME(auto_128_64_256_c8);
        case  8: TIME(pp_64_64_128_c8);
        case  9: TIME(pp_128_64_128_c8);
        case 10: TIME(ws_64_64_128_c8);
        case 11: TIME(ws_128_64_128_c8);
        case 12: TIME(coop_128_64_128_c8);
        case 13: TIME(auto_64_64_128_c8);
        case 14: TIME(auto_128_64_128_c8);
        case 15: TIME(pp_64_64_64_c8);
        case 16: TIME(pp_128_64_64_c8);
        case 17: TIME(ws_64_64_64_c8);
        case 18: TIME(auto_64_64_64_c8);
        case 19: TIME(auto_128_64_64_c8);
        case 20: TIME(pp_64_64_256_c4x2);
        case 21: TIME(pp_128_64_256_c4x2);
        case 22: TIME(coop_128_64_256_c4x2);
        case 23: TIME(auto_64_64_256_c4x2);
        case 24: TIME(auto_128_64_256_c4x2);
        case 25: TIME(pp_64_64_128_c4x2);
        case 26: TIME(pp_128_64_128_c4x2);
        case 27: TIME(ws_64_64_128_c4x2);
        case 28: TIME(ws_128_64_128_c4x2);
        case 29: TIME(coop_128_64_128_c4x2);
        case 30: TIME(auto_64_64_128_c4x2);
        case 31: TIME(auto_128_64_128_c4x2);
        case 32: TIME(pp_64_64_256_c2x2);
        case 33: TIME(pp_128_64_256_c2x2);
        case 34: TIME(pp_64_64_128_c2x2);
        case 35: TIME(pp_128_64_128_c2x2);
        case 36: TIME(ws_128_64_128_c2x2);
        case 37: TIME(auto_64_64_128_c2x2);
        case 38: TIME(auto_128_64_128_c2x2);
        case 39: TIME(pp_64_64_256_c4);
        case 40: TIME(pp_128_64_256_c4);
        case 41: TIME(pp_64_64_128_c4);
        case 42: TIME(pp_128_64_128_c4);
        case 43: TIME(pp_256_64_128_c4);
        case 44: TIME(ws_128_64_128_c4);
        case 45: TIME(ws_128_64_256_c4);
        case 46: TIME(ws_256_64_128_c4);
        case 47: TIME(coop_128_64_128_c4);
        case 48: TIME(coop_128_64_256_c4);
        case 49: TIME(coop_256_64_128_c4);
        case 50: TIME(auto_64_64_128_c4);
        case 51: TIME(auto_128_64_128_c4);
        case 52: TIME(auto_128_64_256_c4);
        case 53: TIME(auto_256_64_128_c4);
        case 54: TIME(pp_64_64_256_c2);
        case 55: TIME(pp_128_64_256_c2);
        case 56: TIME(pp_128_64_128_c2);
        case 57: TIME(pp_256_64_128_c2);
        case 58: TIME(ws_128_64_128_c2);
        case 59: TIME(ws_128_64_256_c2);
        case 60: TIME(ws_256_64_128_c2);
        case 61: TIME(coop_128_64_128_c2);
        case 62: TIME(coop_128_64_256_c2);
        case 63: TIME(coop_256_64_128_c2);
        case 64: TIME(auto_64_64_128_c2);
        case 65: TIME(auto_128_64_128_c2);
        case 66: TIME(auto_128_64_256_c2);
        case 67: TIME(auto_256_64_128_c2);
        default: return -1.f;
    }
#undef TIME
}

static PersistentRunner* make_persistent_runner(int idx,
    int M, int N, int K,
    cutlass::half_t* A, cutlass::half_t* B,
    cutlass::half_t* C, cutlass::half_t* D,
    float alpha, float beta, int dev, int sm)
{
#define MAKE(ns) do { \
    auto* r = new TypedRunner<ns::Gemm>(); \
    if (r->initialize(M,N,K,A,B,C,D,alpha,beta,dev,sm)) return r; \
    delete r; return nullptr; \
} while(0)
    switch (idx) {
        case  1: MAKE(pp_64_64_256_c8);
        case  2: MAKE(pp_128_64_256_c8);
        case  3: MAKE(ws_64_64_256_c8);
        case  4: MAKE(ws_128_64_256_c8);
        case  5: MAKE(coop_128_64_256_c8);
        case  6: MAKE(auto_64_64_256_c8);
        case  7: MAKE(auto_128_64_256_c8);
        case  8: MAKE(pp_64_64_128_c8);
        case  9: MAKE(pp_128_64_128_c8);
        case 10: MAKE(ws_64_64_128_c8);
        case 11: MAKE(ws_128_64_128_c8);
        case 12: MAKE(coop_128_64_128_c8);
        case 13: MAKE(auto_64_64_128_c8);
        case 14: MAKE(auto_128_64_128_c8);
        case 15: MAKE(pp_64_64_64_c8);
        case 16: MAKE(pp_128_64_64_c8);
        case 17: MAKE(ws_64_64_64_c8);
        case 18: MAKE(auto_64_64_64_c8);
        case 19: MAKE(auto_128_64_64_c8);
        case 20: MAKE(pp_64_64_256_c4x2);
        case 21: MAKE(pp_128_64_256_c4x2);
        case 22: MAKE(coop_128_64_256_c4x2);
        case 23: MAKE(auto_64_64_256_c4x2);
        case 24: MAKE(auto_128_64_256_c4x2);
        case 25: MAKE(pp_64_64_128_c4x2);
        case 26: MAKE(pp_128_64_128_c4x2);
        case 27: MAKE(ws_64_64_128_c4x2);
        case 28: MAKE(ws_128_64_128_c4x2);
        case 29: MAKE(coop_128_64_128_c4x2);
        case 30: MAKE(auto_64_64_128_c4x2);
        case 31: MAKE(auto_128_64_128_c4x2);
        case 32: MAKE(pp_64_64_256_c2x2);
        case 33: MAKE(pp_128_64_256_c2x2);
        case 34: MAKE(pp_64_64_128_c2x2);
        case 35: MAKE(pp_128_64_128_c2x2);
        case 36: MAKE(ws_128_64_128_c2x2);
        case 37: MAKE(auto_64_64_128_c2x2);
        case 38: MAKE(auto_128_64_128_c2x2);
        case 39: MAKE(pp_64_64_256_c4);
        case 40: MAKE(pp_128_64_256_c4);
        case 41: MAKE(pp_64_64_128_c4);
        case 42: MAKE(pp_128_64_128_c4);
        case 43: MAKE(pp_256_64_128_c4);
        case 44: MAKE(ws_128_64_128_c4);
        case 45: MAKE(ws_128_64_256_c4);
        case 46: MAKE(ws_256_64_128_c4);
        case 47: MAKE(coop_128_64_128_c4);
        case 48: MAKE(coop_128_64_256_c4);
        case 49: MAKE(coop_256_64_128_c4);
        case 50: MAKE(auto_64_64_128_c4);
        case 51: MAKE(auto_128_64_128_c4);
        case 52: MAKE(auto_128_64_256_c4);
        case 53: MAKE(auto_256_64_128_c4);
        case 54: MAKE(pp_64_64_256_c2);
        case 55: MAKE(pp_128_64_256_c2);
        case 56: MAKE(pp_128_64_128_c2);
        case 57: MAKE(pp_256_64_128_c2);
        case 58: MAKE(ws_128_64_128_c2);
        case 59: MAKE(ws_128_64_256_c2);
        case 60: MAKE(ws_256_64_128_c2);
        case 61: MAKE(coop_128_64_128_c2);
        case 62: MAKE(coop_128_64_256_c2);
        case 63: MAKE(coop_256_64_128_c2);
        case 64: MAKE(auto_64_64_128_c2);
        case 65: MAKE(auto_128_64_128_c2);
        case 66: MAKE(auto_128_64_256_c2);
        case 67: MAKE(auto_256_64_128_c2);
        case 68: MAKE(pp_128_64_128_c1);
        case 69: MAKE(pp_64_64_128_c1);
        case 70: MAKE(ws_128_64_128_c1);
        case 71: MAKE(coop_128_64_128_c1);
        case 72: MAKE(auto_128_64_128_c1);
        case 73: MAKE(auto_128_64_64_c4);
        case 74: MAKE(auto_128_64_64_c2);
        case 75: MAKE(auto_128_64_64_c1);
        default: return nullptr;
    }
#undef MAKE
}

static std::atomic<PersistentRunner*> g_runner{nullptr};
static std::mutex g_tune_mutex;

static const int PRIORITY_CONFIGS[] = {1, 2, 5, 3, 4, 6, 7};
static const int NUM_PRIORITY = 7;

static const int SM_VARIANTS[]  = {128, 120, 136, 132};
static const int NUM_SM_VARIANTS = 4;

static const int NUM_TIMED = 67;
static const int NUM_TOTAL = 75;

struct TuneResult { int best_idx; int best_sm; };

static TuneResult run_autotune(
    int M, int N, int K,
    cutlass::half_t* A, cutlass::half_t* B,
    cutlass::half_t* C, cutlass::half_t* D,
    float alpha, float beta, int dev, int raw_sm)
{
    int   best_idx = -1;
    int   best_sm  = raw_sm;
    float best_ms  = 1e30f;

    for (int pi = 0; pi < NUM_PRIORITY; pi++) {
        int idx = PRIORITY_CONFIGS[pi];
        for (int si = 0; si < NUM_SM_VARIANTS; si++) {
            int sm = SM_VARIANTS[si];
            if (sm > raw_sm + 8) continue;
            float ms = dispatch_time(idx, M, N, K, A, B, C, D, alpha, beta, dev, sm, 3, 10);
            if (ms > 0.f && ms < best_ms) {
                best_ms  = ms;
                best_idx = idx;
                best_sm  = sm;
            }
        }
    }

    for (int idx = 1; idx <= NUM_TIMED; idx++) {
        bool already = false;
        for (int pi = 0; pi < NUM_PRIORITY; pi++) {
            if (PRIORITY_CONFIGS[pi] == idx) { already = true; break; }
        }
        if (already) continue;
        float ms = dispatch_time(idx, M, N, K, A, B, C, D, alpha, beta, dev, best_sm, 2, 5);
        if (ms > 0.f && ms < best_ms) {
            best_ms  = ms;
            best_idx = idx;
        }
    }

    if (best_idx >= 1 && best_idx != PRIORITY_CONFIGS[0]) {
        float ms_w = dispatch_time(best_idx,            M,N,K,A,B,C,D,alpha,beta,dev,best_sm,5,20);
        float ms_p = dispatch_time(PRIORITY_CONFIGS[0], M,N,K,A,B,C,D,alpha,beta,dev,best_sm,5,20);
        if (ms_p > 0.f && (ms_w < 0.f || ms_p <= ms_w * 1.01f)) {
            best_idx = PRIORITY_CONFIGS[0];
        }
    }

    if (best_idx >= 1) return {best_idx, best_sm};

    for (int idx = NUM_TIMED + 1; idx <= NUM_TOTAL; idx++) {
        if (dispatch_run(idx, M, N, K, A, B, C, D, alpha, beta, dev, raw_sm))
            return {idx, raw_sm};
    }
    return {-1, raw_sm};
}

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

    auto* ptr_A = reinterpret_cast<cutlass::half_t*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<cutlass::half_t*>(b_col_major.data_ptr());
    auto* ptr_C = reinterpret_cast<cutlass::half_t*>(c.data_ptr());
    auto* ptr_D = ptr_C;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    int dev = 0;
    cudaGetDevice(&dev);

    PersistentRunner* runner = g_runner.load(std::memory_order_acquire);
    if (__builtin_expect(runner != nullptr, 1)) {
        bool ok = runner->run(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D, alpha, beta);
        if (__builtin_expect(ok, 1)) return;
        g_runner.store(nullptr, std::memory_order_release);
        delete runner;
        runner = nullptr;
    }

    std::lock_guard<std::mutex> lock(g_tune_mutex);

    runner = g_runner.load(std::memory_order_relaxed);
    if (runner != nullptr) {
        runner->run(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D, alpha, beta);
        return;
    }

    int raw_sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    int init_sm = (raw_sm >= 128) ? 128 : ((raw_sm / 8) * 8);
    if (init_sm <= 0) init_sm = raw_sm;

    dispatch_run(PRIORITY_CONFIGS[0], M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                 alpha, beta, dev, init_sm);

    TuneResult result = run_autotune(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                                     alpha, beta, dev, raw_sm);

    if (result.best_idx < 0) {
        for (int sm_try : {128, 120, 132, raw_sm}) {
            if (sm_try > raw_sm + 8) continue;
            PersistentRunner* r = make_persistent_runner(
                PRIORITY_CONFIGS[0], M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                alpha, beta, dev, sm_try);
            if (r) {
                g_runner.store(r, std::memory_order_release);
                return;
            }
        }
        throw std::runtime_error(
            "All CUTLASS GEMM configurations failed for M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }

    PersistentRunner* new_runner = make_persistent_runner(
        result.best_idx, M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
        alpha, beta, dev, result.best_sm);

    if (new_runner == nullptr) {
        dispatch_run(result.best_idx, M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                     alpha, beta, dev, result.best_sm);
        new_runner = make_persistent_runner(
            PRIORITY_CONFIGS[0], M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
            alpha, beta, dev, result.best_sm);
    }

    if (new_runner) {
        g_runner.store(new_runner, std::memory_order_release);
        new_runner->run(M, N, K, ptr_A, ptr_B, ptr_C, ptr_D, alpha, beta);
    } else {
        dispatch_run(result.best_idx, M, N, K, ptr_A, ptr_B, ptr_C, ptr_D,
                     alpha, beta, dev, result.best_sm);
    }
}