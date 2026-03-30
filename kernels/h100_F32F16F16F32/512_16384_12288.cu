#include <iostream>
#include <cstring>
#include <algorithm>

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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA       = cutlass::half_t;
using ElementB       = cutlass::half_t;
using ElementC       = cutlass::half_t;
using ElementD       = cutlass::half_t;
using ElementAcc     = float;
using ElementCompute = float;
using LayoutA        = cutlass::layout::RowMajor;
using LayoutB        = cutlass::layout::ColumnMajor;
using LayoutC        = cutlass::layout::RowMajor;
using LayoutD        = cutlass::layout::RowMajor;

static constexpr int AlignA = 16 / sizeof(ElementA);
static constexpr int AlignB = 16 / sizeof(ElementB);
static constexpr int AlignC = 16 / sizeof(ElementC);
static constexpr int AlignD = 16 / sizeof(ElementD);

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEFINE_HGEMM_KERNEL(Name, TileM, TileN, TileK, GrM, GrN, GrK, Stages, EpiPolicy) \
struct Name { \
  using TileShape  = cute::Shape<cute::_##TileM, cute::_##TileN, cute::_##TileK>; \
  using GroupShape = cute::Shape<cute::_##GrM, cute::_##GrN, cute::_##GrK>; \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GroupShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElementAcc, ElementCompute, \
      ElementC, LayoutC, AlignC, \
      ElementD, LayoutD, AlignD, \
      cutlass::epilogue::EpiPolicy, \
      EpilogueOp \
    >::CollectiveOp; \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElementA, LayoutA, AlignA, \
      ElementB, LayoutB, AlignB, \
      ElementAcc, \
      TileShape, GroupShape, \
      cutlass::gemm::collective::StageCount<Stages>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative \
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

DEFINE_HGEMM_KERNEL(Cfg_K128_TmaCoop, 128, 256, 128, 2, 1, 1, 4, TmaWarpSpecializedCooperative)
DEFINE_HGEMM_KERNEL(Cfg_K128_NoSmem, 128, 256, 128, 2, 1, 1, 4, NoSmemWarpSpecialized)
DEFINE_HGEMM_KERNEL(Cfg_K128_3S_TmaCoop, 128, 256, 128, 2, 1, 1, 3, TmaWarpSpecializedCooperative)
DEFINE_HGEMM_KERNEL(Cfg_K128_5S_NoSmem, 128, 256, 128, 2, 1, 1, 5, NoSmemWarpSpecialized)

DEFINE_HGEMM_KERNEL(Cfg_K64_TmaCoop, 128, 256, 64, 2, 1, 1, 4, TmaWarpSpecializedCooperative)
DEFINE_HGEMM_KERNEL(Cfg_K64_NoSmem, 128, 256, 64, 2, 1, 1, 4, NoSmemWarpSpecialized)
DEFINE_HGEMM_KERNEL(Cfg_K64_3S_TmaCoop, 128, 256, 64, 2, 1, 1, 3, TmaWarpSpecializedCooperative)
DEFINE_HGEMM_KERNEL(Cfg_K64_5S_NoSmem, 128, 256, 64, 2, 1, 1, 5, NoSmemWarpSpecialized)

enum class BestKernel : int {
    None = -1,
    K128_TmaCoop_SK2 = 0,
    K128_NoSmem_SK2  = 1,
    K128_TmaCoop_SK3 = 2,
    K128_NoSmem_SK3  = 3,
    K128_TmaCoop_SK4 = 4,
    K128_NoSmem_SK4  = 5,
    K128_TmaCoop_SK6 = 6,
    K128_NoSmem_SK6  = 7,
    K64_TmaCoop_SK2 = 10,
    K64_NoSmem_SK2  = 11,
    K64_TmaCoop_SK3 = 12,
    K64_NoSmem_SK3  = 13,
    K64_TmaCoop_SK4 = 14,
    K64_NoSmem_SK4  = 15,
    K64_TmaCoop_SK6 = 16,
    K64_NoSmem_SK6  = 17,
    K128_TmaCoop_Std = 20,
    K128_NoSmem_Std  = 21,
    K128_3S_TmaCoop_Std = 22,
    K128_5S_NoSmem_Std = 23,
    K64_TmaCoop_Std = 24,
    K64_NoSmem_Std  = 25,
    K64_3S_TmaCoop_Std = 26,
    K64_5S_NoSmem_Std = 27
};

static BestKernel g_best_kernel = BestKernel::None;

template <typename KernelCfg>
double run_gemm_timed(
    const ElementA* ptr_A, const ElementB* ptr_B,
    const ElementC* ptr_C, ElementD* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo& hw_info,
    int split_k_slices = 1,
    int warmup = 2, int measure = 5)
{
    using Gemm    = typename KernelCfg::Gemm;
    using StrideA = typename KernelCfg::StrideA;
    using StrideB = typename KernelCfg::StrideB;
    using StrideC = typename KernelCfg::StrideC;
    using StrideD = typename KernelCfg::StrideD;

    if (K % split_k_slices != 0) return -1.0;

    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, sA, ptr_B, sB},
        {{1.0f, 0.0f}, ptr_C, sC, ptr_D, sD},
        hw_info,
        split_k_slices
    };

    Gemm gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1.0;

    size_t ws_size = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

    status = gemm.initialize(args, workspace.get());
    if (status != cutlass::Status::kSuccess) return -1.0;

    for (int i = 0; i < warmup; ++i) {
        status = gemm.run();
        if (status != cutlass::Status::kSuccess) return -1.0;
    }
    cudaDeviceSynchronize();

    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start);
    for (int i = 0; i < measure; ++i) {
        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_end);
            return -1.0;
        }
    }
    cudaEventRecord(ev_end);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    return (double)ms * 1000.0 / measure;
}

template <typename KernelCfg>
bool launch_gemm_fast(
    const ElementA* ptr_A, const ElementB* ptr_B,
    const ElementC* ptr_C, ElementD* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo& hw_info,
    int split_k_slices = 1)
{
    using Gemm    = typename KernelCfg::Gemm;
    using StrideA = typename KernelCfg::StrideA;
    using StrideB = typename KernelCfg::StrideB;
    using StrideC = typename KernelCfg::StrideC;
    using StrideD = typename KernelCfg::StrideD;

    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {ptr_A, sA, ptr_B, sB},
        {{1.0f, 0.0f}, ptr_C, sC, ptr_D, sD},
        hw_info,
        split_k_slices
    };

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

    cutlass::Status status = gemm.initialize(args, workspace.get());
    if (status != cutlass::Status::kSuccess) return false;

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) return false;

    cudaDeviceSynchronize();
    return true;
}

static bool dispatch_best(
    const ElementA* ptr_A, const ElementB* ptr_B,
    const ElementC* ptr_C, ElementD* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo& hw_info)
{
    switch (g_best_kernel) {
        case BestKernel::K128_TmaCoop_SK2:  return launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,2);
        case BestKernel::K128_NoSmem_SK2:   return launch_gemm_fast<Cfg_K128_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,2);
        case BestKernel::K128_TmaCoop_SK3:  return launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,3);
        case BestKernel::K128_NoSmem_SK3:   return launch_gemm_fast<Cfg_K128_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,3);
        case BestKernel::K128_TmaCoop_SK4:  return launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,4);
        case BestKernel::K128_NoSmem_SK4:   return launch_gemm_fast<Cfg_K128_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,4);
        case BestKernel::K128_TmaCoop_SK6:  return launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,6);
        case BestKernel::K128_NoSmem_SK6:   return launch_gemm_fast<Cfg_K128_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,6);
        case BestKernel::K64_TmaCoop_SK2:   return launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,2);
        case BestKernel::K64_NoSmem_SK2:    return launch_gemm_fast<Cfg_K64_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,2);
        case BestKernel::K64_TmaCoop_SK3:   return launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,3);
        case BestKernel::K64_NoSmem_SK3:    return launch_gemm_fast<Cfg_K64_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,3);
        case BestKernel::K64_TmaCoop_SK4:   return launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,4);
        case BestKernel::K64_NoSmem_SK4:    return launch_gemm_fast<Cfg_K64_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,4);
        case BestKernel::K64_TmaCoop_SK6:   return launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,6);
        case BestKernel::K64_NoSmem_SK6:    return launch_gemm_fast<Cfg_K64_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,6);
        case BestKernel::K128_TmaCoop_Std:  return launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        case BestKernel::K128_NoSmem_Std:   return launch_gemm_fast<Cfg_K128_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        case BestKernel::K128_3S_TmaCoop_Std: return launch_gemm_fast<Cfg_K128_3S_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        case BestKernel::K128_5S_NoSmem_Std: return launch_gemm_fast<Cfg_K128_5S_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        case BestKernel::K64_TmaCoop_Std:   return launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        case BestKernel::K64_NoSmem_Std:    return launch_gemm_fast<Cfg_K64_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        case BestKernel::K64_3S_TmaCoop_Std: return launch_gemm_fast<Cfg_K64_3S_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        case BestKernel::K64_5S_NoSmem_Std: return launch_gemm_fast<Cfg_K64_5S_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1);
        default: return false;
    }
}

static void run_autotune(
    const ElementA* ptr_A, const ElementB* ptr_B,
    const ElementC* ptr_C, ElementD* ptr_D,
    int M, int N, int K,
    cutlass::KernelHardwareInfo& hw_info)
{
    double best_us = 1e18;
    BestKernel best_id = BestKernel::None;

    #define BENCH(CFG, SK, ID) do { \
        double us = run_gemm_timed<CFG>(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info, SK, 2, 5); \
        if (us > 0 && us < best_us) { \
            best_us = us; \
            best_id = BestKernel::ID; \
        } \
    } while(0)

    BENCH(Cfg_K128_TmaCoop, 2, K128_TmaCoop_SK2);
    BENCH(Cfg_K128_NoSmem,  2, K128_NoSmem_SK2);
    BENCH(Cfg_K128_TmaCoop, 3, K128_TmaCoop_SK3);
    BENCH(Cfg_K128_NoSmem,  3, K128_NoSmem_SK3);
    BENCH(Cfg_K128_TmaCoop, 4, K128_TmaCoop_SK4);
    BENCH(Cfg_K128_NoSmem,  4, K128_NoSmem_SK4);
    BENCH(Cfg_K128_TmaCoop, 6, K128_TmaCoop_SK6);
    BENCH(Cfg_K128_NoSmem,  6, K128_NoSmem_SK6);
    BENCH(Cfg_K64_TmaCoop, 2, K64_TmaCoop_SK2);
    BENCH(Cfg_K64_NoSmem,  2, K64_NoSmem_SK2);
    BENCH(Cfg_K64_TmaCoop, 3, K64_TmaCoop_SK3);
    BENCH(Cfg_K64_NoSmem,  3, K64_NoSmem_SK3);
    BENCH(Cfg_K64_TmaCoop, 4, K64_TmaCoop_SK4);
    BENCH(Cfg_K64_NoSmem,  4, K64_NoSmem_SK4);
    BENCH(Cfg_K64_TmaCoop, 6, K64_TmaCoop_SK6);
    BENCH(Cfg_K64_NoSmem,  6, K64_NoSmem_SK6);
    BENCH(Cfg_K128_TmaCoop, 1, K128_TmaCoop_Std);
    BENCH(Cfg_K128_NoSmem,  1, K128_NoSmem_Std);
    BENCH(Cfg_K128_3S_TmaCoop, 1, K128_3S_TmaCoop_Std);
    BENCH(Cfg_K128_5S_NoSmem, 1, K128_5S_NoSmem_Std);
    BENCH(Cfg_K64_TmaCoop, 1, K64_TmaCoop_Std);
    BENCH(Cfg_K64_NoSmem,  1, K64_NoSmem_Std);
    BENCH(Cfg_K64_3S_TmaCoop, 1, K64_3S_TmaCoop_Std);
    BENCH(Cfg_K64_5S_NoSmem, 1, K64_5S_NoSmem_Std);

    #undef BENCH

    if (best_id != BestKernel::None) {
        g_best_kernel = best_id;
    } else {
        g_best_kernel = BestKernel::K64_TmaCoop_Std;
    }
}

#endif

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!"); \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
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

    auto* ptr_A = reinterpret_cast<const ElementA*>(a.data_ptr());
    auto* ptr_B = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
    auto* ptr_C = reinterpret_cast<const ElementC*>(c.data_ptr());
    auto* ptr_D = reinterpret_cast<ElementD*>(c.data_ptr());

    int device_id = 0;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    if (g_best_kernel != BestKernel::None) {
        if (dispatch_best(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info)) {
            return;
        }
        g_best_kernel = BestKernel::None;
    }

    run_autotune(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info);

    if (g_best_kernel != BestKernel::None) {
        if (dispatch_best(ptr_A, ptr_B, ptr_C, ptr_D, M, N, K, hw_info)) {
            return;
        }
    }

    if (launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,4)) return;
    if (launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,3)) return;
    if (launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,2)) return;
    if (launch_gemm_fast<Cfg_K64_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1)) return;
    if (launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,4)) return;
    if (launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,3)) return;
    if (launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,2)) return;
    if (launch_gemm_fast<Cfg_K128_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1)) return;
    if (launch_gemm_fast<Cfg_K64_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1)) return;
    if (launch_gemm_fast<Cfg_K128_NoSmem>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1)) return;
    if (launch_gemm_fast<Cfg_K64_3S_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1)) return;
    if (launch_gemm_fast<Cfg_K128_3S_TmaCoop>(ptr_A,ptr_B,ptr_C,ptr_D,M,N,K,hw_info,1)) return;

    throw std::runtime_error("All GEMM kernel variants failed to execute");

#else
    throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}