#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

#define DEF_PP(NAME, TM, TN, TK, CM, CN, CK)                                  \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEF_PP_S(NAME, TM, TN, TK, CM, CN, CK, STAGES)                        \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecialized,                                   \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCount<STAGES>,                           \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;          \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEF_COOP(NAME, TM, TN, TK, CM, CN, CK)                                \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::PersistentScheduler>;                                     \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

#define DEF_SK(NAME, TM, TN, TK, CM, CN, CK)                                  \
struct NAME {                                                                   \
  using TileShape    = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>; \
  using GridShape    = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      TileShape, GridShape,                                                     \
      cutlass::epilogue::collective::EpilogueTileAuto,                         \
      ElementAccumulator, ElementCompute,                                       \
      ElementC, LayoutC, AlignC,                                               \
      ElementD, LayoutD, AlignD,                                               \
      cutlass::epilogue::TmaWarpSpecializedCooperative,                        \
      EpilogueOp>::CollectiveOp;                                               \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                     \
      ElementA, LayoutA, AlignA,                                               \
      ElementB, LayoutB, AlignB,                                               \
      ElementAccumulator,                                                       \
      TileShape, GridShape,                                                     \
      cutlass::gemm::collective::StageCountAutoCarveout<                       \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;       \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                    \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,        \
      cutlass::gemm::StreamKScheduler>;                                        \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;    \
  using StrideA = typename Gemm::GemmKernel::StrideA;                         \
  using StrideB = typename Gemm::GemmKernel::StrideB;                         \
  using StrideC = typename Gemm::GemmKernel::StrideC;                         \
  using StrideD = typename Gemm::GemmKernel::StrideD;                         \
};

DEF_PP(PP_64x32x128,           64,  32, 128, 1, 1, 1)
DEF_PP(PP_64x32x64,            64,  32,  64, 1, 1, 1)
DEF_PP(PP_64x32x256,           64,  32, 256, 1, 1, 1)
DEF_PP(PP_64x32x128_C211,      64,  32, 128, 2, 1, 1)
DEF_PP(PP_64x32x128_C121,      64,  32, 128, 1, 2, 1)
DEF_PP_S(PP_64x32x128_S4,      64,  32, 128, 1, 1, 1, 4)
DEF_PP_S(PP_64x32x128_S5,      64,  32, 128, 1, 1, 1, 5)

DEF_PP(PP_64x64x128,           64,  64, 128, 1, 1, 1)
DEF_PP(PP_64x64x64,            64,  64,  64, 1, 1, 1)
DEF_PP(PP_64x64x256,           64,  64, 256, 1, 1, 1)
DEF_PP_S(PP_64x64x128_S4,      64,  64, 128, 1, 1, 1, 4)
DEF_PP_S(PP_64x64x128_S5,      64,  64, 128, 1, 1, 1, 5)
DEF_PP_S(PP_64x64x64_S4,       64,  64,  64, 1, 1, 1, 4)
DEF_PP_S(PP_64x64x64_S5,       64,  64,  64, 1, 1, 1, 5)
DEF_PP(PP_64x64x128_C211,      64,  64, 128, 2, 1, 1)
DEF_PP(PP_64x64x64_C211,       64,  64,  64, 2, 1, 1)
DEF_PP(PP_64x64x256_C211,      64,  64, 256, 2, 1, 1)
DEF_PP(PP_64x64x128_C121,      64,  64, 128, 1, 2, 1)
DEF_PP(PP_64x64x128_C411,      64,  64, 128, 4, 1, 1)
DEF_PP(PP_64x64x64_C411,       64,  64,  64, 4, 1, 1)
DEF_PP_S(PP_64x64x128_C211_S5, 64,  64, 128, 2, 1, 1, 5)

DEF_PP(PP_64x128x64,           64, 128,  64, 1, 1, 1)
DEF_PP(PP_64x128x128,          64, 128, 128, 1, 1, 1)
DEF_PP(PP_64x128x256,          64, 128, 256, 1, 1, 1)
DEF_PP(PP_64x128x128_C211,     64, 128, 128, 2, 1, 1)
DEF_PP_S(PP_64x128x128_S5,     64, 128, 128, 1, 1, 1, 5)

DEF_PP(PP_128x64x64,          128,  64,  64, 1, 1, 1)
DEF_PP(PP_128x64x128,         128,  64, 128, 1, 1, 1)
DEF_PP_S(PP_128x64x128_S5,    128,  64, 128, 1, 1, 1, 5)
DEF_PP(PP_128x64x128_C211,    128,  64, 128, 2, 1, 1)
DEF_COOP(CO_128x64x64,        128,  64,  64, 1, 1, 1)
DEF_COOP(CO_128x64x128,       128,  64, 128, 1, 1, 1)
DEF_COOP(CO_128x64x128_C211,  128,  64, 128, 2, 1, 1)

DEF_SK(SK_128x64x128,         128,  64, 128, 1, 1, 1)
DEF_SK(SK_128x64x64,          128,  64,  64, 1, 1, 1)
DEF_SK(SK_128x128x64,         128, 128,  64, 1, 1, 1)
DEF_SK(SK_128x128x128,        128, 128, 128, 1, 1, 1)
DEF_SK(SK_128x64x128_C211,    128,  64, 128, 2, 1, 1)

DEF_PP(PP_128x128x64,         128, 128,  64, 1, 1, 1)
DEF_PP(PP_128x128x128,        128, 128, 128, 1, 1, 1)
DEF_COOP(CO_128x128x64,       128, 128,  64, 1, 1, 1)
DEF_COOP(CO_128x128x128,      128, 128, 128, 1, 1, 1)
DEF_COOP(CO_128x128x64_C211,  128, 128,  64, 2, 1, 1)

#undef DEF_PP
#undef DEF_PP_S
#undef DEF_COOP
#undef DEF_SK

template <typename HgemmType>
cutlass::Status run_gemm(
    void* ptr_A, void* ptr_B_col, void* ptr_C, void* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo hw_info,
    int split_k = 1)
{
  using Gemm    = typename HgemmType::Gemm;
  using StrideA = typename HgemmType::StrideA;
  using StrideB = typename HgemmType::StrideB;
  using StrideC = typename HgemmType::StrideC;
  using StrideD = typename HgemmType::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {reinterpret_cast<ElementA*>(ptr_A),   stride_A,
     reinterpret_cast<ElementB*>(ptr_B_col), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<ElementC*>(ptr_C), stride_C,
     reinterpret_cast<ElementD*>(ptr_D), stride_D},
    hw_info,
    split_k
  };

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return status;

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) return status;

  if (cudaGetLastError() != cudaSuccess) return cutlass::Status::kErrorInternal;
  return cutlass::Status::kSuccess;
}

template <typename HgemmType>
bool attempt(void* ptr_A, void* ptr_B, void* ptr_C, void* ptr_D,
             int M, int N, int K, cutlass::KernelHardwareInfo hw_info,
             int split_k = 1) {
  return run_gemm<HgemmType>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info, split_k)
         == cutlass::Status::kSuccess;
}

static int g_best_candidate = -1;

struct Candidate {
  const char* name;
  int split_k;
  std::function<bool(void*, void*, void*, void*, int, int, int, cutlass::KernelHardwareInfo, int)> fn;
};

static double time_candidate(
    const Candidate& c,
    void* A, void* B, void* C, void* D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo hw,
    int warmup = 3, int iters = 8)
{
  for (int i = 0; i < warmup; i++)
    if (!c.fn(A, B, C, D, M, N, K, hw, c.split_k)) return -1.0;
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < iters; i++)
    c.fn(A, B, C, D, M, N, K, hw, c.split_k);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return (double)(ms / iters * 1000.0);
}

static std::vector<Candidate> make_candidates() {
  std::vector<Candidate> v;

  v.push_back({"PP_64x64x128_sk4",        4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_S5_sk4",     4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128_S5>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_S4_sk4",     4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128_S4>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_C211_sk4",   4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128_C211>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_C211S5_sk4", 4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128_C211_S5>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x256_sk2",        2, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x256>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x256_C211_sk2",   2, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x256_C211>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x64_S5_sk4",      4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x64_S5>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_C411_sk4",   4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128_C411>(A,B,C,D,M,N,K,hw,sk); }});

  v.push_back({"PP_64x32x128_sk1",        1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x128>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x32x128_S5_sk1",     1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x128_S5>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x32x128_S4_sk1",     1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x128_S4>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x32x64_sk1",         1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x64>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x32x256_sk1",        1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x256>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x32x128_C211_sk1",   1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x128_C211>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x32x128_C121_sk1",   1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x128_C121>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x32x128_sk2",        2, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x32x128>(A,B,C,D,M,N,K,hw,sk); }});

  v.push_back({"PP_64x64x128_sk3",        3, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_sk8",        8, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_sk2",        2, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x64x128_sk1",        1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x64x128>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x128x256_sk2",       2, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x128x256>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_64x128x128_S5_sk4",    4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_64x128x128_S5>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"PP_128x64x128_S5_sk4",    4, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<PP_128x64x128_S5>(A,B,C,D,M,N,K,hw,sk); }});
  v.push_back({"SK_128x64x128",           1, [](void* A,void* B,void* C,void* D,int M,int N,int K,cutlass::KernelHardwareInfo hw,int sk){ return attempt<SK_128x64x128>(A,B,C,D,M,N,K,hw,sk); }});

  return v;
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

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  void* ptr_A  = a.data_ptr();
  void* ptr_Bc = b_col_major.data_ptr();
  void* ptr_C  = c.data_ptr();
  void* ptr_D  = c.data_ptr();

  if (g_best_candidate < 0) {
    auto candidates = make_candidates();
    int best_idx = 0;
    double best_time = 1e18;

    for (int i = 0; i < (int)candidates.size(); i++) {
      double t = time_candidate(candidates[i], ptr_A, ptr_Bc, ptr_C, ptr_D,
                                M, N, K, hw_info, 3, 8);
      if (t > 0.0 && t < best_time) {
        best_time = t;
        best_idx = i;
      }
    }
    g_best_candidate = best_idx;
    candidates[best_idx].fn(ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info,
                             candidates[best_idx].split_k);
    return;
  }

  switch (g_best_candidate) {
    case 0:  attempt<PP_64x64x128>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 1:  attempt<PP_64x64x128_S5>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 2:  attempt<PP_64x64x128_S4>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 3:  attempt<PP_64x64x128_C211>   (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 4:  attempt<PP_64x64x128_C211_S5>(ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 5:  attempt<PP_64x64x256>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 2); return;
    case 6:  attempt<PP_64x64x256_C211>   (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 2); return;
    case 7:  attempt<PP_64x64x64_S5>      (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 8:  attempt<PP_64x64x128_C411>   (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 9:  attempt<PP_64x32x128>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 10: attempt<PP_64x32x128_S5>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 11: attempt<PP_64x32x128_S4>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 12: attempt<PP_64x32x64>         (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 13: attempt<PP_64x32x256>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 14: attempt<PP_64x32x128_C211>   (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 15: attempt<PP_64x32x128_C121>   (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 16: attempt<PP_64x32x128>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 2); return;
    case 17: attempt<PP_64x64x128>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 3); return;
    case 18: attempt<PP_64x64x128>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 8); return;
    case 19: attempt<PP_64x64x128>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 2); return;
    case 20: attempt<PP_64x64x128>        (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    case 21: attempt<PP_64x128x256>       (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 2); return;
    case 22: attempt<PP_64x128x128_S5>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 23: attempt<PP_128x64x128_S5>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4); return;
    case 24: attempt<SK_128x64x128>       (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1); return;
    default:
      if (attempt<PP_64x64x128>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4)) return;
      if (attempt<PP_64x64x128_S5>  (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4)) return;
      if (attempt<PP_64x64x128>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 2)) return;
      if (attempt<PP_64x32x128>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1)) return;
      if (attempt<PP_64x64x128>     (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1)) return;
      if (attempt<PP_64x128x128>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4)) return;
      if (attempt<PP_128x64x128>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4)) return;
      if (attempt<CO_128x64x128>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 4)) return;
      if (attempt<SK_128x64x128>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1)) return;
      if (attempt<CO_128x128x64>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1)) return;
      if (attempt<PP_128x128x64>    (ptr_A, ptr_Bc, ptr_C, ptr_D, M, N, K, hw_info, 1)) return;
      throw std::runtime_error("All CUTLASS GEMM variants failed");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported");
#endif
}