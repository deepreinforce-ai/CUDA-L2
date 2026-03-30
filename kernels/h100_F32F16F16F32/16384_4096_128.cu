#include <iostream>
#include <cfloat>
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

#define DEF_GEMM_P(Name, TM, TN, TK, CM, CN, CK, MainSched, EpiSched)         \
struct Name {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;     \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridShape,                                                      \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignC,                                                 \
      ElementD, LayoutD, AlignD,                                                 \
      cutlass::epilogue::EpiSched,                                               \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignA,                                                 \
      ElementB, LayoutB, AlignB,                                                 \
      ElementAccumulator,                                                        \
      TileShape, GridShape,                                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::MainSched>::CollectiveOp;                                   \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::PersistentScheduler>;                                       \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;      \
  using StrideA = typename Gemm::GemmKernel::StrideA;                           \
  using StrideB = typename Gemm::GemmKernel::StrideB;                           \
  using StrideC = typename Gemm::GemmKernel::StrideC;                           \
  using StrideD = typename Gemm::GemmKernel::StrideD;                           \
};

#define DEF_GEMM_SK(Name, TM, TN, TK, CM, CN, CK, MainSched, EpiSched)        \
struct Name {                                                                    \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>;     \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>;     \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      TileShape, GridShape,                                                      \
      cutlass::epilogue::collective::EpilogueTileAuto,                           \
      ElementAccumulator, ElementCompute,                                        \
      ElementC, LayoutC, AlignC,                                                 \
      ElementD, LayoutD, AlignD,                                                 \
      cutlass::epilogue::EpiSched,                                               \
      EpilogueOp>::CollectiveOp;                                                 \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                      \
      ElementA, LayoutA, AlignA,                                                 \
      ElementB, LayoutB, AlignB,                                                 \
      ElementAccumulator,                                                        \
      TileShape, GridShape,                                                      \
      cutlass::gemm::collective::StageCountAutoCarveout<                         \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   \
      cutlass::gemm::MainSched>::CollectiveOp;                                   \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                      \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,          \
      cutlass::gemm::StreamKScheduler>;                                          \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;      \
  using StrideA = typename Gemm::GemmKernel::StrideA;                           \
  using StrideB = typename Gemm::GemmKernel::StrideB;                           \
  using StrideC = typename Gemm::GemmKernel::StrideC;                           \
  using StrideD = typename Gemm::GemmKernel::StrideD;                           \
};

DEF_GEMM_P(P_Coop256x128_K128_C2x1, 256,128,128, 2,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop256x128_K128_C1x2, 256,128,128, 1,2,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop256x128_K128_C4x1, 256,128,128, 4,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop256x128_K128_C1x1, 256,128,128, 1,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop256x128_K64_C2x1,  256,128,64,  2,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop256x128_K64_C1x2,  256,128,64,  1,2,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop256x128_K64_C4x1,  256,128,64,  4,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop256x128_K64_C1x1,  256,128,64,  1,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)

DEF_GEMM_P(P_Ping128x256_K128_C2x1, 128,256,128, 2,1,1, KernelTmaWarpSpecializedPingpong, TmaWarpSpecialized)
DEF_GEMM_P(P_Ping128x256_K128_C1x2, 128,256,128, 1,2,1, KernelTmaWarpSpecializedPingpong, TmaWarpSpecialized)
DEF_GEMM_P(P_Ping128x256_K128_C1x1, 128,256,128, 1,1,1, KernelTmaWarpSpecializedPingpong, TmaWarpSpecialized)
DEF_GEMM_P(P_Coop128x256_K128_C2x1, 128,256,128, 2,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop128x256_K128_C1x2, 128,256,128, 1,2,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop128x256_K128_C1x1, 128,256,128, 1,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Ping128x256_K64_C2x1,  128,256,64,  2,1,1, KernelTmaWarpSpecializedPingpong, TmaWarpSpecialized)
DEF_GEMM_P(P_Ping128x256_K64_C1x2,  128,256,64,  1,2,1, KernelTmaWarpSpecializedPingpong, TmaWarpSpecialized)
DEF_GEMM_P(P_Coop128x256_K64_C2x1,  128,256,64,  2,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop128x256_K64_C1x2,  128,256,64,  1,2,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)

DEF_GEMM_P(P_Ping128x128_K128_C2x2, 128,128,128, 2,2,1, KernelTmaWarpSpecializedPingpong, TmaWarpSpecialized)
DEF_GEMM_P(P_Ping128x128_K128_C2x1, 128,128,128, 2,1,1, KernelTmaWarpSpecializedPingpong, TmaWarpSpecialized)
DEF_GEMM_P(P_Coop128x128_K128_C2x2, 128,128,128, 2,2,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_P(P_Coop128x128_K128_C2x1, 128,128,128, 2,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)

DEF_GEMM_SK(SK_Coop256x128_K128_C2x1, 256,128,128, 2,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_SK(SK_Coop256x128_K128_C1x2, 256,128,128, 1,2,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_SK(SK_Coop256x128_K128_C1x1, 256,128,128, 1,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_SK(SK_Coop128x256_K128_C2x1, 128,256,128, 2,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)
DEF_GEMM_SK(SK_Coop128x256_K128_C1x1, 128,256,128, 1,1,1, KernelTmaWarpSpecializedCooperative, TmaWarpSpecializedCooperative)

static void*  g_workspace      = nullptr;
static size_t g_workspace_size = 0;

static void* get_workspace(size_t needed) {
  if (needed <= g_workspace_size) return g_workspace;
  if (g_workspace) cudaFree(g_workspace);
  cudaMalloc(&g_workspace, needed);
  g_workspace_size = needed;
  return g_workspace;
}

template <typename GV>
bool try_gemm(const ElementA* ptr_A, const ElementB* ptr_B,
              const ElementC* ptr_C, ElementD* ptr_D,
              int M, int N, int K,
              const cutlass::KernelHardwareInfo& hw_info) {
  using Gemm    = typename GV::Gemm;
  using StrideA = typename GV::StrideA;
  using StrideB = typename GV::StrideB;
  using StrideC = typename GV::StrideC;
  using StrideD = typename GV::StrideD;

  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sb = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {const_cast<ElementA*>(ptr_A), sa, const_cast<ElementB*>(ptr_B), sb},
    {{1.0f, 0.0f}, const_cast<ElementC*>(ptr_C), sc, ptr_D, sd},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws = get_workspace(ws_size);
  if (!ws && ws_size > 0) return false;

  if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) return false;
  if (gemm.run()                != cutlass::Status::kSuccess) return false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { cudaGetLastError(); return false; }
  return true;
}

using RunnerFn = bool(*)(const ElementA*, const ElementB*, const ElementC*, ElementD*,
                          int, int, int, const cutlass::KernelHardwareInfo&);

struct Candidate { RunnerFn fn; const char* name; };

static const Candidate kCandidates[] = {
  { try_gemm<P_Coop256x128_K128_C2x1>,  "P_Coop256x128_K128_C2x1"  },
  { try_gemm<P_Coop256x128_K128_C1x2>,  "P_Coop256x128_K128_C1x2"  },
  { try_gemm<P_Coop256x128_K128_C4x1>,  "P_Coop256x128_K128_C4x1"  },
  { try_gemm<P_Coop256x128_K128_C1x1>,  "P_Coop256x128_K128_C1x1"  },
  { try_gemm<P_Coop256x128_K64_C2x1>,   "P_Coop256x128_K64_C2x1"   },
  { try_gemm<P_Coop256x128_K64_C1x2>,   "P_Coop256x128_K64_C1x2"   },
  { try_gemm<P_Coop256x128_K64_C4x1>,   "P_Coop256x128_K64_C4x1"   },
  { try_gemm<P_Coop256x128_K64_C1x1>,   "P_Coop256x128_K64_C1x1"   },
  { try_gemm<P_Ping128x256_K128_C2x1>,  "P_Ping128x256_K128_C2x1"  },
  { try_gemm<P_Ping128x256_K128_C1x2>,  "P_Ping128x256_K128_C1x2"  },
  { try_gemm<P_Ping128x256_K128_C1x1>,  "P_Ping128x256_K128_C1x1"  },
  { try_gemm<P_Coop128x256_K128_C2x1>,  "P_Coop128x256_K128_C2x1"  },
  { try_gemm<P_Coop128x256_K128_C1x2>,  "P_Coop128x256_K128_C1x2"  },
  { try_gemm<P_Coop128x256_K128_C1x1>,  "P_Coop128x256_K128_C1x1"  },
  { try_gemm<SK_Coop256x128_K128_C2x1>, "SK_Coop256x128_K128_C2x1" },
  { try_gemm<SK_Coop256x128_K128_C1x2>, "SK_Coop256x128_K128_C1x2" },
  { try_gemm<SK_Coop256x128_K128_C1x1>, "SK_Coop256x128_K128_C1x1" },
  { try_gemm<SK_Coop128x256_K128_C2x1>, "SK_Coop128x256_K128_C2x1" },
  { try_gemm<SK_Coop128x256_K128_C1x1>, "SK_Coop128x256_K128_C1x1" },
  { try_gemm<P_Ping128x256_K64_C2x1>,   "P_Ping128x256_K64_C2x1"   },
  { try_gemm<P_Ping128x256_K64_C1x2>,   "P_Ping128x256_K64_C1x2"   },
  { try_gemm<P_Coop128x256_K64_C2x1>,   "P_Coop128x256_K64_C2x1"   },
  { try_gemm<P_Coop128x256_K64_C1x2>,   "P_Coop128x256_K64_C1x2"   },
  { try_gemm<P_Ping128x128_K128_C2x2>,  "P_Ping128x128_K128_C2x2"  },
  { try_gemm<P_Ping128x128_K128_C2x1>,  "P_Ping128x128_K128_C2x1"  },
  { try_gemm<P_Coop128x128_K128_C2x2>,  "P_Coop128x128_K128_C2x2"  },
  { try_gemm<P_Coop128x128_K128_C2x1>,  "P_Coop128x128_K128_C2x1"  },
};

static constexpr int kNumCandidates = static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));

struct CacheKey { int M, N, K; };
static CacheKey  g_cached_key  = {-1, -1, -1};
static RunnerFn  g_best_fn     = nullptr;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  auto* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<const ElementC*>(c.data_ptr());
  auto* ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  if (g_best_fn != nullptr &&
      g_cached_key.M == M && g_cached_key.N == N && g_cached_key.K == K) {
    if (g_best_fn(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info)) return;
    g_best_fn = nullptr;
  }

  cudaDeviceSynchronize();

  int   best_idx = -1;
  float best_ms  = FLT_MAX;

  static constexpr int kWarmup = 3;
  static constexpr int kTiming = 8;

  for (int i = 0; i < kNumCandidates; ++i) {
    bool ok = kCandidates[i].fn(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    if (!ok) continue;
    cudaDeviceSynchronize();

    for (int w = 1; w < kWarmup; ++w) {
      kCandidates[i].fn(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < kTiming; ++r) {
      kCandidates[i].fn(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    if (ms < best_ms) {
      best_ms  = ms;
      best_idx = i;
    }
  }

  if (best_idx < 0) {
    throw std::runtime_error("All GEMM variants failed to execute");
  }

  g_best_fn    = kCandidates[best_idx].fn;
  g_cached_key = {M, N, K};

  g_best_fn(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}