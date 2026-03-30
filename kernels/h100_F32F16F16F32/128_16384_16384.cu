#include <iostream>
#include <stdexcept>
#include <string>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElemA   = cutlass::half_t;
using ElemB   = cutlass::half_t;
using ElemC   = cutlass::half_t;
using ElemD   = cutlass::half_t;
using ElemAcc = float;
using ElemCmp = float;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElemD, ElemCmp, ElemC, ElemCmp,
    cutlass::FloatRoundStyle::round_to_nearest>;

using LayoutColMaj = cutlass::layout::ColumnMajor;

#define DEF_PP_AUTO(Name, TM, TN, TK, CM, CN, CK) \
struct Name { \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GridShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElemAcc, ElemCmp, ElemC, LayoutColMaj, 8, ElemD, LayoutColMaj, 8, \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElemA, LayoutColMaj, 8, ElemB, LayoutColMaj, 8, ElemAcc, \
      TileShape, GridShape, \
      cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
      cutlass::gemm::PersistentScheduler>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
};

#define DEF_PP_STAGE(Name, TM, TN, TK, CM, CN, CK, SC) \
struct Name { \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GridShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElemAcc, ElemCmp, ElemC, LayoutColMaj, 8, ElemD, LayoutColMaj, 8, \
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElemA, LayoutColMaj, 8, ElemB, LayoutColMaj, 8, ElemAcc, \
      TileShape, GridShape, \
      cutlass::gemm::collective::StageCount<SC>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
      cutlass::gemm::PersistentScheduler>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
};

#define DEF_COOP_PERSISTENT(Name, TM, TN, TK, CM, CN, CK) \
struct Name { \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GridShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElemAcc, ElemCmp, ElemC, LayoutColMaj, 8, ElemD, LayoutColMaj, 8, \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElemA, LayoutColMaj, 8, ElemB, LayoutColMaj, 8, ElemAcc, \
      TileShape, GridShape, \
      cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
      cutlass::gemm::PersistentScheduler>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
};

#define DEF_COOP_STREAMK(Name, TM, TN, TK, CM, CN, CK) \
struct Name { \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GridShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElemAcc, ElemCmp, ElemC, LayoutColMaj, 8, ElemD, LayoutColMaj, 8, \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElemA, LayoutColMaj, 8, ElemB, LayoutColMaj, 8, ElemAcc, \
      TileShape, GridShape, \
      cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
      cutlass::gemm::StreamKScheduler>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
};

#define DEF_COOP_STREAMK_STAGE(Name, TM, TN, TK, CM, CN, CK, SC) \
struct Name { \
  using TileShape    = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using GridShape    = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GridShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElemAcc, ElemCmp, ElemC, LayoutColMaj, 8, ElemD, LayoutColMaj, 8, \
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElemA, LayoutColMaj, 8, ElemB, LayoutColMaj, 8, ElemAcc, \
      TileShape, GridShape, \
      cutlass::gemm::collective::StageCount<SC>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, \
      cutlass::gemm::StreamKScheduler>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
};

DEF_PP_AUTO  (Cfg_Anchor,          128, 128,  64, 1, 1, 1)
DEF_PP_STAGE (Cfg_S5,              128, 128,  64, 1, 1, 1, 5)
DEF_PP_STAGE (Cfg_S6,              128, 128,  64, 1, 1, 1, 6)
DEF_PP_STAGE (Cfg_S4,              128, 128,  64, 1, 1, 1, 4)
DEF_PP_STAGE (Cfg_S7,              128, 128,  64, 1, 1, 1, 7)
DEF_PP_STAGE (Cfg_S8,              128, 128,  64, 1, 1, 1, 8)
DEF_PP_STAGE (Cfg_S3,              128, 128,  64, 1, 1, 1, 3)

DEF_PP_AUTO  (Cfg_C2x1,            128, 128,  64, 2, 1, 1)
DEF_PP_STAGE (Cfg_C2x1_S5,         128, 128,  64, 2, 1, 1, 5)
DEF_PP_STAGE (Cfg_C2x1_S4,         128, 128,  64, 2, 1, 1, 4)
DEF_PP_STAGE (Cfg_C2x1_S6,         128, 128,  64, 2, 1, 1, 6)
DEF_PP_AUTO  (Cfg_C4x1,            128, 128,  64, 4, 1, 1)
DEF_PP_STAGE (Cfg_C4x1_S5,         128, 128,  64, 4, 1, 1, 5)
DEF_PP_STAGE (Cfg_C4x1_S4,         128, 128,  64, 4, 1, 1, 4)
DEF_PP_STAGE (Cfg_C4x1_S3,         128, 128,  64, 4, 1, 1, 3)
DEF_PP_AUTO  (Cfg_C8x1,            128, 128,  64, 8, 1, 1)
DEF_PP_STAGE (Cfg_C8x1_S4,         128, 128,  64, 8, 1, 1, 4)
DEF_PP_STAGE (Cfg_C8x1_S3,         128, 128,  64, 8, 1, 1, 3)

DEF_PP_AUTO  (Cfg_1x2,             128, 128,  64, 1, 2, 1)
DEF_PP_STAGE (Cfg_1x2_S5,          128, 128,  64, 1, 2, 1, 5)
DEF_PP_STAGE (Cfg_1x2_S4,          128, 128,  64, 1, 2, 1, 4)
DEF_PP_STAGE (Cfg_1x2_S6,          128, 128,  64, 1, 2, 1, 6)
DEF_PP_STAGE (Cfg_1x2_S3,          128, 128,  64, 1, 2, 1, 3)
DEF_PP_AUTO  (Cfg_1x4,             128, 128,  64, 1, 4, 1)
DEF_PP_STAGE (Cfg_1x4_S5,          128, 128,  64, 1, 4, 1, 5)
DEF_PP_STAGE (Cfg_1x4_S4,          128, 128,  64, 1, 4, 1, 4)
DEF_PP_STAGE (Cfg_1x4_S3,          128, 128,  64, 1, 4, 1, 3)
DEF_PP_AUTO  (Cfg_1x8,             128, 128,  64, 1, 8, 1)
DEF_PP_STAGE (Cfg_1x8_S4,          128, 128,  64, 1, 8, 1, 4)
DEF_PP_STAGE (Cfg_1x8_S3,          128, 128,  64, 1, 8, 1, 3)

DEF_PP_AUTO  (Cfg_M256_1x1,        256, 128,  64, 1, 1, 1)
DEF_PP_STAGE (Cfg_M256_S5,         256, 128,  64, 1, 1, 1, 5)
DEF_PP_STAGE (Cfg_M256_S4,         256, 128,  64, 1, 1, 1, 4)
DEF_PP_STAGE (Cfg_M256_S3,         256, 128,  64, 1, 1, 1, 3)
DEF_PP_AUTO  (Cfg_M256_2x1,        256, 128,  64, 2, 1, 1)
DEF_PP_STAGE (Cfg_M256_2x1_S4,     256, 128,  64, 2, 1, 1, 4)
DEF_PP_STAGE (Cfg_M256_2x1_S3,     256, 128,  64, 2, 1, 1, 3)
DEF_PP_AUTO  (Cfg_M256_4x1,        256, 128,  64, 4, 1, 1)
DEF_PP_STAGE (Cfg_M256_4x1_S4,     256, 128,  64, 4, 1, 1, 4)
DEF_PP_STAGE (Cfg_M256_4x1_S3,     256, 128,  64, 4, 1, 1, 3)
DEF_PP_AUTO  (Cfg_M256_1x2,        256, 128,  64, 1, 2, 1)
DEF_PP_STAGE (Cfg_M256_1x2_S4,     256, 128,  64, 1, 2, 1, 4)
DEF_PP_STAGE (Cfg_M256_1x2_S3,     256, 128,  64, 1, 2, 1, 3)

DEF_PP_AUTO  (Cfg_M64_1x1,          64, 128,  64, 1, 1, 1)
DEF_PP_STAGE (Cfg_M64_S6,           64, 128,  64, 1, 1, 1, 6)
DEF_PP_STAGE (Cfg_M64_S5,           64, 128,  64, 1, 1, 1, 5)
DEF_PP_STAGE (Cfg_M64_S4,           64, 128,  64, 1, 1, 1, 4)
DEF_PP_AUTO  (Cfg_M64_2x1,          64, 128,  64, 2, 1, 1)
DEF_PP_AUTO  (Cfg_M64_4x1,          64, 128,  64, 4, 1, 1)
DEF_PP_STAGE (Cfg_M64_4x1_S4,       64, 128,  64, 4, 1, 1, 4)
DEF_PP_AUTO  (Cfg_M64_8x1,          64, 128,  64, 8, 1, 1)
DEF_PP_STAGE (Cfg_M64_8x1_S4,       64, 128,  64, 8, 1, 1, 4)

DEF_PP_AUTO  (Cfg_K128_1x1,        128, 128, 128, 1, 1, 1)
DEF_PP_STAGE (Cfg_K128_S4,         128, 128, 128, 1, 1, 1, 4)
DEF_PP_STAGE (Cfg_K128_S3,         128, 128, 128, 1, 1, 1, 3)
DEF_PP_AUTO  (Cfg_K128_2x1,        128, 128, 128, 2, 1, 1)
DEF_PP_AUTO  (Cfg_K128_4x1,        128, 128, 128, 4, 1, 1)
DEF_PP_AUTO  (Cfg_K128_1x2,        128, 128, 128, 1, 2, 1)
DEF_PP_STAGE (Cfg_K128_1x2_S4,     128, 128, 128, 1, 2, 1, 4)

DEF_COOP_STREAMK      (Cfg_SK_256x128,      256, 128,  64, 1, 1, 1)
DEF_COOP_STREAMK_STAGE(Cfg_SK_256x128_S4,   256, 128,  64, 1, 1, 1, 4)
DEF_COOP_STREAMK_STAGE(Cfg_SK_256x128_S3,   256, 128,  64, 1, 1, 1, 3)
DEF_COOP_STREAMK      (Cfg_SK_128x256,      128, 256,  64, 1, 1, 1)
DEF_COOP_STREAMK      (Cfg_SK_128x128,      128, 128,  64, 1, 1, 1)
DEF_COOP_STREAMK_STAGE(Cfg_SK_128x128_S4,   128, 128,  64, 1, 1, 1, 4)
DEF_COOP_STREAMK_STAGE(Cfg_SK_128x128_S3,   128, 128,  64, 1, 1, 1, 3)
DEF_COOP_STREAMK      (Cfg_SK_128x128x128,  128, 128, 128, 1, 1, 1)
DEF_COOP_STREAMK_STAGE(Cfg_SK_128x128x128_S4,128,128, 128, 1, 1, 1, 4)
DEF_COOP_STREAMK      (Cfg_SK_256x128_1x2,  256, 128,  64, 1, 2, 1)
DEF_COOP_STREAMK      (Cfg_SK_128x128_1x2,  128, 128,  64, 1, 2, 1)

DEF_COOP_PERSISTENT(Cfg_CP_256x128,    256, 128,  64, 1, 1, 1)
DEF_COOP_PERSISTENT(Cfg_CP_128x256,    128, 256,  64, 1, 1, 1)
DEF_COOP_PERSISTENT(Cfg_CP_128x128,    128, 128,  64, 1, 1, 1)
DEF_COOP_PERSISTENT(Cfg_CP_128x128x128,128, 128, 128, 1, 1, 1)
DEF_COOP_PERSISTENT(Cfg_CP_256x128_1x2,256, 128,  64, 1, 2, 1)
DEF_COOP_PERSISTENT(Cfg_CP_128x128_1x2,128, 128,  64, 1, 2, 1)

template <typename Cfg>
bool try_run_transposed(void* ptr_b_orig, void* ptr_a_orig, void* ptr_c,
                        int M_orig, int N_orig, int K_orig) {
  using Gemm = typename Cfg::Gemm;

  const int Mp = N_orig;
  const int Np = M_orig;
  const int Kp = K_orig;

  typename Cfg::StrideA stride_A = cute::make_stride(
      cute::Int<1>{}, int64_t(N_orig), int64_t(0));
  typename Cfg::StrideB stride_B = cute::make_stride(
      int64_t(K_orig), cute::Int<1>{}, int64_t(0));
  typename Cfg::StrideC stride_C = cute::make_stride(
      cute::Int<1>{}, int64_t(N_orig), int64_t(0));
  typename Cfg::StrideD stride_D = cute::make_stride(
      cute::Int<1>{}, int64_t(N_orig), int64_t(0));

  int device_id = 0;
  cudaGetDevice(&device_id);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {Mp, Np, Kp},
    {reinterpret_cast<ElemA*>(ptr_b_orig), stride_A,
     reinterpret_cast<ElemB*>(ptr_a_orig), stride_B},
    {{1.0f, 0.0f},
     reinterpret_cast<ElemC*>(ptr_c), stride_C,
     reinterpret_cast<ElemC*>(ptr_c), stride_D},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) return false;
  return gemm.run() == cutlass::Status::kSuccess;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  void* ptr_a = a.data_ptr();
  void* ptr_b = b.data_ptr();
  void* ptr_c = c.data_ptr();

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (try_run_transposed<Cfg_Anchor>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_S5>            (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_S6>            (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_S4>            (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_S7>            (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_S8>            (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_S3>            (ptr_b, ptr_a, ptr_c, M, N, K)) return;

  if (try_run_transposed<Cfg_C4x1>          (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C4x1_S5>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C4x1_S4>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C4x1_S3>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C8x1>          (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C8x1_S4>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C8x1_S3>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C2x1>          (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C2x1_S5>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C2x1_S4>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_C2x1_S6>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;

  if (try_run_transposed<Cfg_1x4>           (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x4_S5>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x4_S4>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x4_S3>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x8>           (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x8_S4>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x8_S3>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x2>           (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x2_S5>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x2_S4>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x2_S6>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_1x2_S3>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;

  if (try_run_transposed<Cfg_M256_S5>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_S4>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_S3>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_1x1>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_2x1>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_2x1_S4>   (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_2x1_S3>   (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_4x1>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_4x1_S4>   (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_4x1_S3>   (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_1x2>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_1x2_S4>   (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M256_1x2_S3>   (ptr_b, ptr_a, ptr_c, M, N, K)) return;

  if (try_run_transposed<Cfg_M64_S6>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_S5>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_S4>        (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_1x1>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_2x1>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_4x1>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_4x1_S4>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_8x1>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_M64_8x1_S4>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;

  if (try_run_transposed<Cfg_K128_S4>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_K128_S3>       (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_K128_1x1>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_K128_2x1>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_K128_4x1>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_K128_1x2>      (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_K128_1x2_S4>   (ptr_b, ptr_a, ptr_c, M, N, K)) return;

  if (try_run_transposed<Cfg_SK_256x128>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_256x128_S4> (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_256x128_S3> (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_256x128_1x2>(ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_128x256>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_128x128>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_128x128_S4> (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_128x128_S3> (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_128x128_1x2>(ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_128x128x128>(ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_SK_128x128x128_S4>(ptr_b,ptr_a,ptr_c,M, N, K)) return;

  if (try_run_transposed<Cfg_CP_256x128>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_CP_128x256>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_CP_128x128>    (ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_CP_256x128_1x2>(ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_CP_128x128_1x2>(ptr_b, ptr_a, ptr_c, M, N, K)) return;
  if (try_run_transposed<Cfg_CP_128x128x128>(ptr_b, ptr_a, ptr_c, M, N, K)) return;

  throw std::runtime_error("All HGEMM configurations failed");
#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}