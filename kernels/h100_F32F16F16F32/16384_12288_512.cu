#include <iostream>
#include <cuda_runtime.h>

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

using ElementA       = cutlass::half_t;
using ElementB       = cutlass::half_t;
using ElementC       = cutlass::half_t;
using ElementD       = cutlass::half_t;
using ElementAcc     = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 16 / sizeof(ElementA);
static constexpr int AlignB = 16 / sizeof(ElementB);
static constexpr int AlignC = 16 / sizeof(ElementC);
static constexpr int AlignD = 16 / sizeof(ElementD);

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_CFG(Name, TileM, TileN, TileK, GrM, GrN, Stages, MainSched, EpiSched) \
struct Name { \
  using TileShape   = cute::Shape<cute::_##TileM, cute::_##TileN, cute::_##TileK>; \
  using GridShape = cute::Shape<cute::_##GrM, cute::_##GrN, cute::_##1>; \
  using StageCount = cutlass::gemm::collective::StageCount<Stages>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GridShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElementAcc, ElementCompute, \
      ElementC, LayoutC, AlignC, \
      ElementD, LayoutD, AlignD, \
      EpiSched, EpilogueOp \
    >::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElementA, LayoutA, AlignA, \
      ElementB, LayoutB, AlignB, \
      ElementAcc, \
      TileShape, GridShape, StageCount, MainSched \
    >::CollectiveOp; \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
      cutlass::gemm::PersistentScheduler>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
};

DEFINE_CFG(Cfg00, 256, 256, 128, 2, 2, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg01, 256, 256, 128, 1, 2, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg02, 256, 256, 128, 2, 1, 7, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg03, 256, 256, 128, 1, 1, 6, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg04, 256, 256, 128, 2, 2, 5, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)

DEFINE_CFG(Cfg05, 128, 256, 128, 1, 2, 6, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg06, 128, 256, 128, 2, 1, 6, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg07, 128, 256, 128, 1, 4, 5, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg08, 128, 256, 128, 1, 2, 5, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)

DEFINE_CFG(Cfg09, 128, 256, 96, 2, 1, 5, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg10, 128, 256, 80, 1, 2, 5, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg11, 128, 192, 128, 2, 2, 4, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_CFG(Cfg12, 128, 256, 64, 2, 1, 4, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg13, 128, 128, 128, 2, 1, 4, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_CFG(Cfg14, 128, 128, 64, 2, 1, 4, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)

#undef DEFINE_CFG

template <typename CfgType>
static cutlass::Status run_gemm_impl(
    const ElementA* ptr_A, const ElementB* ptr_B, ElementC* ptr_C,
    int M, int N, int K, cutlass::KernelHardwareInfo& hw_info, void* workspace, size_t workspace_size)
{
  using Gemm = typename CfgType::Gemm;
  using StrideA = typename CfgType::StrideA;
  using StrideB = typename CfgType::StrideB;
  using StrideC = typename CfgType::StrideC;
  using StrideD = typename CfgType::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D},
    hw_info
  };

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return status;

  size_t required_ws = Gemm::get_workspace_size(arguments);
  if (required_ws > workspace_size) return cutlass::Status::kErrorInvalidProblem;

  status = gemm.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) return status;

  return gemm.run();
}

enum CfgId {
  CFG00, CFG01, CFG02, CFG03, CFG04, CFG05, CFG06, CFG07, CFG08, CFG09, CFG10, CFG11, CFG12, CFG13, CFG14, CFG_COUNT
};

template<typename CfgType>
static size_t get_ws_for_config(int M, int N, int K, cutlass::KernelHardwareInfo& hw_info) {
  using StrideA = typename CfgType::StrideA;
  using StrideB = typename CfgType::StrideB;
  using StrideC = typename CfgType::StrideC;
  using StrideD = typename CfgType::StrideD;
  
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  
  typename CfgType::Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
    {nullptr, stride_A, nullptr, stride_B},
    {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}, hw_info
  };
  return CfgType::Gemm::get_workspace_size(args);
}

static size_t get_workspace_size(int cfg_id, int M, int N, int K, cutlass::KernelHardwareInfo& hw_info) {
  switch (cfg_id) {
    case CFG00: return get_ws_for_config<Cfg00>(M,N,K,hw_info);
    case CFG01: return get_ws_for_config<Cfg01>(M,N,K,hw_info);
    case CFG02: return get_ws_for_config<Cfg02>(M,N,K,hw_info);
    case CFG03: return get_ws_for_config<Cfg03>(M,N,K,hw_info);
    case CFG04: return get_ws_for_config<Cfg04>(M,N,K,hw_info);
    case CFG05: return get_ws_for_config<Cfg05>(M,N,K,hw_info);
    case CFG06: return get_ws_for_config<Cfg06>(M,N,K,hw_info);
    case CFG07: return get_ws_for_config<Cfg07>(M,N,K,hw_info);
    case CFG08: return get_ws_for_config<Cfg08>(M,N,K,hw_info);
    case CFG09: return get_ws_for_config<Cfg09>(M,N,K,hw_info);
    case CFG10: return get_ws_for_config<Cfg10>(M,N,K,hw_info);
    case CFG11: return get_ws_for_config<Cfg11>(M,N,K,hw_info);
    case CFG12: return get_ws_for_config<Cfg12>(M,N,K,hw_info);
    case CFG13: return get_ws_for_config<Cfg13>(M,N,K,hw_info);
    case CFG14: return get_ws_for_config<Cfg14>(M,N,K,hw_info);
    default: return 0;
  }
}

static cutlass::Status dispatch_gemm(int cfg_id, const ElementA* ptr_A, const ElementB* ptr_B, ElementC* ptr_C,
    int M, int N, int K, cutlass::KernelHardwareInfo& hw_info, void* workspace, size_t workspace_size)
{
  switch (cfg_id) {
    case CFG00: return run_gemm_impl<Cfg00>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG01: return run_gemm_impl<Cfg01>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG02: return run_gemm_impl<Cfg02>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG03: return run_gemm_impl<Cfg03>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG04: return run_gemm_impl<Cfg04>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG05: return run_gemm_impl<Cfg05>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG06: return run_gemm_impl<Cfg06>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG07: return run_gemm_impl<Cfg07>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG08: return run_gemm_impl<Cfg08>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG09: return run_gemm_impl<Cfg09>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG10: return run_gemm_impl<Cfg10>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG11: return run_gemm_impl<Cfg11>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG12: return run_gemm_impl<Cfg12>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG13: return run_gemm_impl<Cfg13>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    case CFG14: return run_gemm_impl<Cfg14>(ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
    default: return cutlass::Status::kErrorInvalidProblem;
  }
}

struct AutoTuner {
  int best_cfg = -1;
  size_t workspace_size = 0;
  void* workspace = nullptr;
  ~AutoTuner() { if (workspace) cudaFree(workspace); }

  void tune(const ElementA* ptr_A, const ElementB* ptr_B, ElementC* ptr_C, int M, int N, int K, cutlass::KernelHardwareInfo& hw_info) {
    workspace_size = 0;
    for (int i = 0; i < CFG_COUNT; ++i) {
      size_t ws = get_workspace_size(i, M, N, K, hw_info);
      if (ws > workspace_size) workspace_size = ws;
    }
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

    double best_time = 1e18;
    best_cfg = CFG14;

    int priority[] = {CFG00,CFG01,CFG02,CFG05,CFG06,CFG03,CFG07,CFG04,CFG09,CFG08,CFG10,CFG11,CFG12,CFG13,CFG14};
    
    for (int idx = 0; idx < CFG_COUNT; ++idx) {
      int cfg = priority[idx];
      bool valid = true;
      for (int i = 0; i < 3; ++i) {
        if (dispatch_gemm(cfg,ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size) != cutlass::Status::kSuccess) {
          valid = false; break;
        }
      }
      if (!valid) continue;
      cudaDeviceSynchronize();

      cudaEvent_t start, end;
      cudaEventCreate(&start); cudaEventCreate(&end);
      double total_ms = 0.0;
      for (int i = 0; i < 10; ++i) {
        cudaEventRecord(start);
        dispatch_gemm(cfg,ptr_A,ptr_B,ptr_C,M,N,K,hw_info,workspace,workspace_size);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float ms; cudaEventElapsedTime(&ms, start, end);
        total_ms += ms;
      }
      cudaEventDestroy(start); cudaEventDestroy(end);
      double avg_ms = total_ms / 10;
      if (avg_ms < best_time) { best_time = avg_ms; best_cfg = cfg; }
    }
  }
};

static AutoTuner g_tuner;
static bool g_tuned = false;

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  auto* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
  auto* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
  auto* ptr_C = reinterpret_cast<ElementC*>(c.data_ptr());

  int device_id; cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  if (!g_tuned) { g_tuner.tune(ptr_A, ptr_B, ptr_C, M, N, K, hw_info); g_tuned = true; }
  if (!g_tuner.workspace && g_tuner.workspace_size > 0) cudaMalloc(&g_tuner.workspace, g_tuner.workspace_size);

  cutlass::Status status = dispatch_gemm(g_tuner.best_cfg, ptr_A, ptr_B, ptr_C, M, N, K, hw_info, g_tuner.workspace, g_tuner.workspace_size);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM failed; status: " + std::to_string(static_cast<int>(status)));
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
#else
  throw std::runtime_error("CUTLASS SM90 not supported");
#endif
}