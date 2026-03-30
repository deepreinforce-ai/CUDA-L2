#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>
#include <atomic>
#include <mutex>

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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if ((T).options().dtype() != (th_type)) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

#define DEFINE_GEMM_CONFIG(NS, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, EPI) \
namespace NS { \
using ElementA = cutlass::half_t; \
using ElementB = cutlass::half_t; \
using ElementC = cutlass::half_t; \
using ElementAccumulator = float; \
using LayoutA = cutlass::layout::RowMajor; \
using LayoutB = cutlass::layout::ColumnMajor; \
using LayoutC = cutlass::layout::RowMajor; \
constexpr int AlignA = 8; \
constexpr int AlignB = 8; \
constexpr int AlignC = 8; \
using TileShape  = cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_N>, cute::Int<TILE_K>>; \
using GroupShape = cute::Shape<cute::Int<CL_M>,   cute::Int<CL_N>,  cute::Int<CL_K>>; \
using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    TileShape, GroupShape, \
    cutlass::epilogue::collective::EpilogueTileAuto, \
    ElementAccumulator, ElementAccumulator, \
    ElementC, LayoutC, AlignC, \
    ElementC, LayoutC, AlignC, \
    EPI \
>::CollectiveOp; \
using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder< \
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
    ElementA, LayoutA, AlignA, \
    ElementB, LayoutB, AlignB, \
    ElementAccumulator, \
    TileShape, GroupShape, \
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename EpilogueOp::SharedStorage))>, \
    SCHED \
>::CollectiveOp; \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, MainloopOp, EpilogueOp>; \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
}

DEFINE_GEMM_CONFIG(CO_64_64,   128, 64,  64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_64_64_C2, 128, 64, 64,  1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_64_128,  128, 64,  128, 1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_64_128_C2, 128, 64, 128, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_128_64,  128, 128, 64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_128_64_C2, 128, 128, 64, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_128_128, 128, 128, 128, 1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_128_128_C2, 128, 128, 128, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_32_64,   128, 32,  64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)
DEFINE_GEMM_CONFIG(CO_32_128,  128, 32,  128, 1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative)

DEFINE_GEMM_CONFIG(NS_64_64,   128, 64,  64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(NS_64_64_C2, 128, 64, 64,  1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(NS_64_128,  128, 64,  128, 1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(NS_64_128_C2, 128, 64, 128, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(NS_128_64,  128, 128, 64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(NS_128_64_C2, 128, 128, 64, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(NS_32_64,   128, 32,  64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::NoSmemWarpSpecialized)

DEFINE_GEMM_CONFIG(PP_64_64,   128, 64,  64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_64_64_C2, 128, 64, 64,  1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_64_128,  128, 64,  128, 1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_64_128_C2, 128, 64, 128, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_128_64,  128, 128, 64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_128_64_C2, 128, 128, 64, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_32_64,   128, 32,  64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_32_128,  128, 32,  128, 1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecialized)

DEFINE_GEMM_CONFIG(PP_NS_64_64, 128, 64, 64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_NS_64_64_C2, 128, 64, 64, 1, 2, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::NoSmemWarpSpecialized)
DEFINE_GEMM_CONFIG(PP_NS_32_64, 128, 32, 64,  1, 1, 1, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::NoSmemWarpSpecialized)

template<typename GemmType>
static bool run_gemm(
    int M, int N, int K,
    cutlass::gemm::GemmUniversalMode mode,
    int split_k,
    void* ptr_A_raw, void* ptr_B_raw, void* ptr_C_raw,
    int64_t lda, int64_t ldb, int64_t ldc,
    cutlass::KernelHardwareInfo hw_info,
    uint8_t* workspace_ptr, size_t workspace_size)
{
    using ElementA = typename GemmType::ElementA;
    using ElementB = typename GemmType::ElementB;
    using ElementC = typename GemmType::ElementC;
    using StrideA  = typename GemmType::GemmKernel::StrideA;
    using StrideB  = typename GemmType::GemmKernel::StrideB;
    using StrideC  = typename GemmType::GemmKernel::StrideC;
    using StrideD  = typename GemmType::GemmKernel::StrideD;

    auto* ptr_A = reinterpret_cast<ElementA*>(ptr_A_raw);
    auto* ptr_B = reinterpret_cast<ElementB*>(ptr_B_raw);
    auto* ptr_C = reinterpret_cast<ElementC*>(ptr_C_raw);

    StrideA stride_A = cute::make_stride(lda, cute::Int<1>{}, int64_t(0));
    StrideB stride_B = cute::make_stride(ldb, cute::Int<1>{}, int64_t(0));
    StrideC stride_C = cute::make_stride(ldc, cute::Int<1>{}, int64_t(0));
    StrideD stride_D = cute::make_stride(ldc, cute::Int<1>{}, int64_t(0));

    float alpha = 1.0f, beta = 0.0f;
    typename GemmType::Arguments arguments{
        mode,
        {M, N, K},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{alpha, beta}, ptr_C, stride_C, ptr_C, stride_D},
        hw_info,
        split_k
    };
    GemmType gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
    size_t ws_needed = GemmType::get_workspace_size(arguments);
    if (ws_needed > workspace_size) return false;
    if (gemm.initialize(arguments, workspace_ptr) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

struct GemmConfig {
    int config_id;
    int split_k;
    float best_time_ms;
};

static GemmConfig g_best_config = {-1, 1, std::numeric_limits<float>::max()};
static std::mutex g_config_mutex;
static int g_tune_calls = 0;
static const int TUNE_CALLS = 5;

static constexpr size_t WORKSPACE_SIZE = 256ULL * 1024 * 1024;
static cutlass::device_memory::allocation<uint8_t>* g_workspace = nullptr;
static std::once_flag g_workspace_flag;

static void init_workspace() {
    g_workspace = new cutlass::device_memory::allocation<uint8_t>(WORKSPACE_SIZE);
}

#define NUM_CONFIGS 90

struct ConfigSpec {
    int type;
    int split_k;
};

static const ConfigSpec CONFIG_LIST[] = {
    {0,  1}, {0,  2}, {0,  3}, {0,  4},
    {1,  1}, {1,  2}, {1,  3}, {1,  4},
    {2,  1}, {2,  2}, {2,  3},
    {3,  1}, {3,  2}, {3,  3},
    {4,  2}, {4,  4}, {4,  8},
    {5,  2}, {5,  4}, {5,  8},
    {6,  4}, {6,  8},
    {7,  4}, {7,  8},
    {8,  1}, {8,  2},
    {9,  1}, {9,  2},
    {10, 1}, {10, 2}, {10, 3},
    {11, 1}, {11, 2}, {11, 3},
    {12, 1}, {12, 2},
    {13, 1}, {13, 2},
    {14, 2}, {14, 4},
    {15, 2}, {15, 4},
    {16, 1}, {16, 2},
    {17, 1}, {17, 2}, {17, 3},
    {18, 1}, {18, 2}, {18, 3},
    {19, 1}, {19, 2},
    {20, 1}, {20, 2},
    {21, 2}, {21, 4},
    {22, 2}, {22, 4},
    {23, 1}, {23, 2},
    {24, 1},
    {25, 1}, {25, 2},
    {26, 1}, {26, 2},
    {27, 1},
};

static const int N_CONFIGS = sizeof(CONFIG_LIST) / sizeof(CONFIG_LIST[0]);

static bool dispatch_config(
    int type, int split_k,
    int M, int N, int K,
    void* ptr_A, void* ptr_B, void* ptr_C,
    int64_t lda, int64_t ldb, int64_t ldc,
    cutlass::KernelHardwareInfo hw_info,
    uint8_t* ws, size_t ws_size)
{
    auto mode = (split_k > 1)
        ? cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel
        : cutlass::gemm::GemmUniversalMode::kGemm;

    switch(type) {
        case 0:  return run_gemm<CO_64_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 1:  return run_gemm<CO_64_64_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 2:  return run_gemm<CO_64_128::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 3:  return run_gemm<CO_64_128_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 4:  return run_gemm<CO_128_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 5:  return run_gemm<CO_128_64_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 6:  return run_gemm<CO_128_128::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 7:  return run_gemm<CO_128_128_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 8:  return run_gemm<CO_32_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 9:  return run_gemm<CO_32_128::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 10: return run_gemm<NS_64_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 11: return run_gemm<NS_64_64_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 12: return run_gemm<NS_64_128::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 13: return run_gemm<NS_64_128_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 14: return run_gemm<NS_128_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 15: return run_gemm<NS_128_64_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 16: return run_gemm<NS_32_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 17: return run_gemm<PP_64_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 18: return run_gemm<PP_64_64_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 19: return run_gemm<PP_64_128::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 20: return run_gemm<PP_64_128_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 21: return run_gemm<PP_128_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 22: return run_gemm<PP_128_64_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 23: return run_gemm<PP_32_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 24: return run_gemm<PP_32_128::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 25: return run_gemm<PP_NS_64_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 26: return run_gemm<PP_NS_64_64_C2::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        case 27: return run_gemm<PP_NS_32_64::Gemm>(M, N, K, mode, split_k, ptr_A, ptr_B, ptr_C, lda, ldb, ldc, hw_info, ws, ws_size);
        default: return false;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    int64_t lda = int64_t(K);
    int64_t ldb = int64_t(K);
    int64_t ldc = int64_t(N);

    void* ptr_A = a.data_ptr();
    void* ptr_B = b_col_major.data_ptr();
    void* ptr_C = c.data_ptr();

    int device_id = 0;
    cudaGetDevice(&device_id);

    std::call_once(g_workspace_flag, init_workspace);
    uint8_t* ws_ptr = g_workspace->get();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    std::lock_guard<std::mutex> lock(g_config_mutex);

    if (g_tune_calls < TUNE_CALLS) {
        g_tune_calls++;

        cudaStream_t stream = 0;
        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        int best_idx = -1;
        float best_time = std::numeric_limits<float>::max();

        bool warmed = false;
        for (int i = 0; i < N_CONFIGS && !warmed; i++) {
            warmed = dispatch_config(CONFIG_LIST[i].type, CONFIG_LIST[i].split_k,
                                     M, N, K, ptr_A, ptr_B, ptr_C, lda, ldb, ldc,
                                     hw_info, ws_ptr, WORKSPACE_SIZE);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < N_CONFIGS; i++) {
            cudaEventRecord(start_ev, stream);
            bool ok = dispatch_config(CONFIG_LIST[i].type, CONFIG_LIST[i].split_k,
                                      M, N, K, ptr_A, ptr_B, ptr_C, lda, ldb, ldc,
                                      hw_info, ws_ptr, WORKSPACE_SIZE);
            cudaEventRecord(stop_ev, stream);
            cudaEventSynchronize(stop_ev);

            if (!ok) continue;

            float ms = 0.f;
            cudaEventElapsedTime(&ms, start_ev, stop_ev);

            if (ms < best_time) {
                best_time = ms;
                best_idx = i;
            }
        }

        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);

        if (best_idx >= 0 && best_time < g_best_config.best_time_ms) {
            g_best_config.config_id = best_idx;
            g_best_config.split_k = CONFIG_LIST[best_idx].split_k;
            g_best_config.best_time_ms = best_time;
        }

        if (g_best_config.config_id >= 0) {
            int idx = g_best_config.config_id;
            dispatch_config(CONFIG_LIST[idx].type, CONFIG_LIST[idx].split_k,
                           M, N, K, ptr_A, ptr_B, ptr_C, lda, ldb, ldc,
                           hw_info, ws_ptr, WORKSPACE_SIZE);
        } else {
            dispatch_config(0, 2, M, N, K, ptr_A, ptr_B, ptr_C, lda, ldb, ldc,
                           hw_info, ws_ptr, WORKSPACE_SIZE);
        }
    } else {
        int idx = g_best_config.config_id;
        if (idx < 0) idx = 0;
        bool ok = dispatch_config(CONFIG_LIST[idx].type, CONFIG_LIST[idx].split_k,
                                  M, N, K, ptr_A, ptr_B, ptr_C, lda, ldb, ldc,
                                  hw_info, ws_ptr, WORKSPACE_SIZE);
        if (!ok) {
            for (int i = 0; i < N_CONFIGS; i++) {
                if (dispatch_config(CONFIG_LIST[i].type, CONFIG_LIST[i].split_k,
                                   M, N, K, ptr_A, ptr_B, ptr_C, lda, ldb, ldc,
                                   hw_info, ws_ptr, WORKSPACE_SIZE)) break;
            }
        }
    }
}