#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef CUTLASS_DISABLE_CHECKS
#define CUTLASS_DISABLE_CHECKS
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <stdexcept>
#include <string>
#include <limits>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

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

using KernelScheduleWS     = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueScheduleWS   = cutlass::epilogue::TmaWarpSpecialized;

using KernelScheduleCoop   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueScheduleCoop = cutlass::epilogue::TmaWarpSpecializedCooperative;

using KernelScheduleAuto   = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueScheduleAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;

static void*  g_workspace_ptr  = nullptr;
static size_t g_workspace_size = 0;

static inline void* get_workspace(size_t needed_bytes) {
  if (needed_bytes == 0) return nullptr;
  if (needed_bytes <= g_workspace_size) return g_workspace_ptr;

  if (g_workspace_ptr) {
    cudaFree(g_workspace_ptr);
    g_workspace_ptr = nullptr;
    g_workspace_size = 0;
  }

  size_t alloc_bytes = needed_bytes + (1 << 20);
  if (cudaMalloc(&g_workspace_ptr, alloc_bytes) != cudaSuccess) {
    g_workspace_ptr = nullptr;
    g_workspace_size = 0;
    return nullptr;
  }
  g_workspace_size = alloc_bytes;
  return g_workspace_ptr;
}

static cutlass::KernelHardwareInfo g_hw_info;
static bool g_hw_info_ready = false;

static inline cutlass::KernelHardwareInfo const& get_hw_info() {
  if (!g_hw_info_ready) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    g_hw_info.device_id = device_id;
    g_hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    g_hw_info_ready = true;
  }
  return g_hw_info;
}

template<int TM, int TN, int TK, int CM, typename KSched, typename ESched>
struct GemmCfg {
  using TileShape     = cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>;
  using TileGroupShape = cute::Shape<cute::Int<CM>, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, TileGroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      ESched>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, TileGroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KSched>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Cfg_WS_64x64x128_C1    = GemmCfg<64,  64, 128, 1, KernelScheduleWS,   EpilogueScheduleWS>;
using Cfg_WS_128x64x128_C1   = GemmCfg<128, 64, 128, 1, KernelScheduleWS,   EpilogueScheduleWS>;
using Cfg_WS_64x64x64_C1     = GemmCfg<64,  64,  64, 1, KernelScheduleWS,   EpilogueScheduleWS>;
using Cfg_WS_128x64x64_C1    = GemmCfg<128, 64,  64, 1, KernelScheduleWS,   EpilogueScheduleWS>;
using Cfg_Coop_128x64x128_C1 = GemmCfg<128, 64, 128, 1, KernelScheduleCoop, EpilogueScheduleCoop>;
using Cfg_Auto_128x64x64_C8  = GemmCfg<128, 64,  64, 8, KernelScheduleAuto, EpilogueScheduleAuto>;

template<typename GemmType>
static bool run_cutlass_gemm(
    int M, int N, int K,
    const ElementA* A,
    const ElementB* Bc,
    ElementC* C,
    float alpha, float beta,
    cudaStream_t stream) {

  using StrideA = typename GemmType::GemmKernel::StrideA;
  using StrideB = typename GemmType::GemmKernel::StrideB;
  using StrideC = typename GemmType::GemmKernel::StrideC;
  using StrideD = typename GemmType::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename GemmType::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {A, stride_A, Bc, stride_B},
      {{alpha, beta}, C, stride_C, C, stride_D},
      get_hw_info()
  };

  GemmType gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

  size_t ws = GemmType::get_workspace_size(args);
  void* workspace = get_workspace(ws);
  if (ws > 0 && workspace == nullptr) return false;

  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return false;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return false;
  return (cudaGetLastError() == cudaSuccess);
}

enum KernelId : int {
  K_WS_64x64x128_C1 = 0,
  K_WS_128x64x128_C1,
  K_WS_64x64x64_C1,
  K_WS_128x64x64_C1,
  K_COOP_128x64x128_C1,
  K_AUTO_128x64x64_C8,
  K_COUNT
};

static bool dispatch_kernel(
    int kid, int M, int N, int K,
    const ElementA* A, const ElementB* Bc, ElementC* C,
    float alpha, float beta, cudaStream_t stream) {
  switch (kid) {
    case K_WS_64x64x128_C1:
      return run_cutlass_gemm<Cfg_WS_64x64x128_C1::Gemm>(M,N,K,A,Bc,C,alpha,beta,stream);
    case K_WS_128x64x128_C1:
      return run_cutlass_gemm<Cfg_WS_128x64x128_C1::Gemm>(M,N,K,A,Bc,C,alpha,beta,stream);
    case K_WS_64x64x64_C1:
      return run_cutlass_gemm<Cfg_WS_64x64x64_C1::Gemm>(M,N,K,A,Bc,C,alpha,beta,stream);
    case K_WS_128x64x64_C1:
      return run_cutlass_gemm<Cfg_WS_128x64x64_C1::Gemm>(M,N,K,A,Bc,C,alpha,beta,stream);
    case K_COOP_128x64x128_C1:
      return run_cutlass_gemm<Cfg_Coop_128x64x128_C1::Gemm>(M,N,K,A,Bc,C,alpha,beta,stream);
    case K_AUTO_128x64x64_C8:
      return run_cutlass_gemm<Cfg_Auto_128x64x64_C8::Gemm>(M,N,K,A,Bc,C,alpha,beta,stream);
    default:
      return false;
  }
}

static bool g_tuned = false;
static int  g_best_kernel = K_WS_64x64x128_C1;

static void autotune_once(
    int M, int N, int K,
    const ElementA* A, const ElementB* Bc, ElementC* C,
    float alpha, float beta, cudaStream_t stream) {
  if (g_tuned) return;

  if (M == 8192 && N == 64 && K == 8192) {
    if (dispatch_kernel(K_WS_64x64x128_C1, M, N, K, A, Bc, C, alpha, beta, stream)) {
      g_best_kernel = K_WS_64x64x128_C1;
      g_tuned = true;
      return;
    }
  }

  const int candidates[] = {
      K_WS_64x64x128_C1,
      K_WS_128x64x128_C1,
      K_COOP_128x64x128_C1,
      K_WS_128x64x64_C1,
      K_AUTO_128x64x64_C8
  };

  float best_ms = std::numeric_limits<float>::max();
  int best_id = K_AUTO_128x64x64_C8;

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  for (int kid : candidates) {
    bool ok = dispatch_kernel(kid, M, N, K, A, Bc, C, alpha, beta, stream);
    if (!ok || cudaStreamSynchronize(stream) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    cudaEventRecord(ev0, stream);
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
      ok = dispatch_kernel(kid, M, N, K, A, Bc, C, alpha, beta, stream);
      if (!ok) break;
    }
    cudaEventRecord(ev1, stream);
    if (!ok || cudaEventSynchronize(ev1) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms /= 3.0f;

    if (ms < best_ms) {
      best_ms = ms;
      best_id = kid;
    }
  }

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  g_best_kernel = best_id;
  g_tuned = true;
}

__global__ void __launch_bounds__(128, 2) wmma_fallback_kernel(
    const half* __restrict__ A,
    const half* __restrict__ Bcol,
    half* __restrict__ C,
    int M, int N, int K) {

  const int BM=64, BN=64, BK=32;
  int block_m = blockIdx.x;
  int m0 = block_m * BM;
  if (m0 >= M) return;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int warp_m = warp >> 1;
  int warp_n = warp & 1;

  extern __shared__ half smem[];
  half* sA = smem;
  half* sB = smem + BM * BK;

  using namespace nvcuda::wmma;
  float acc[2][2][8] = {};

  for (int kk = 0; kk < K; kk += BK) {
    int arow = tid >> 1;
    int acol = (tid & 1) * 16;
    if (arow < BM && (m0 + arow) < M) {
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        int kx = acol + i;
        sA[arow * BK + kx] = (kk + kx < K) ? A[(m0 + arow) * K + (kk + kx)] : __float2half(0.f);
      }
    }

    int flat0 = tid * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      int flat = flat0 + i;
      int bk = flat / BN;
      int bn = flat % BN;
      sB[bk * BN + bn] = (kk + bk < K && bn < N) ? Bcol[bn * K + (kk + bk)] : __float2half(0.f);
    }
    __syncthreads();

    int cm = warp_m * 32;
    int cn = warp_n * 32;

    fragment<matrix_a,16,16,16,half,row_major> fa[2][2];
    fragment<matrix_b,16,16,16,half,row_major> fb[2][2];
    fragment<accumulator,16,16,16,float> fc[2][2];

    #pragma unroll
    for (int mt=0; mt<2; ++mt)
      for (int kt=0; kt<2; ++kt)
        load_matrix_sync(fa[mt][kt], sA + (cm + mt*16) * BK + kt*16, BK);

    #pragma unroll
    for (int kt=0; kt<2; ++kt)
      for (int nt=0; nt<2; ++nt)
        load_matrix_sync(fb[kt][nt], sB + kt*16*BN + (cn + nt*16), BN);

    #pragma unroll
    for (int mt=0; mt<2; ++mt) {
      for (int nt=0; nt<2; ++nt) {
        fill_fragment(fc[mt][nt], 0.f);
        for (int kt=0; kt<2; ++kt) mma_sync(fc[mt][nt], fa[mt][kt], fb[kt][nt], fc[mt][nt]);
        for (int i=0; i<8; ++i) acc[mt][nt][i] += fc[mt][nt].x[i];
      }
    }
    __syncthreads();
  }

  int cm = warp_m * 32;
  int cn = warp_n * 32;
  #pragma unroll
  for (int mt=0; mt<2; ++mt) {
    for (int nt=0; nt<2; ++nt) {
      int row = m0 + cm + mt*16;
      int col = cn + nt*16;
      if (row < M && col + 15 < N) {
        fragment<accumulator,16,16,16,half> out;
        for (int i=0; i<8; ++i) out.x[i] = __float2half(acc[mt][nt][i]);
        store_matrix_sync(C + row * N + col, out, N, mem_row_major);
      }
    }
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  auto* A  = reinterpret_cast<const ElementA*>(a.data_ptr<at::Half>());
  auto* Bc = reinterpret_cast<const ElementB*>(b_col_major.data_ptr<at::Half>());
  auto* C  = reinterpret_cast<ElementC*>(c.data_ptr<at::Half>());

  const float alpha = 1.0f, beta = 0.0f;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (!g_tuned) {
    autotune_once(M, N, K, A, Bc, C, alpha, beta, stream);
  }

  if (g_best_kernel >= 0 &&
      dispatch_kernel(g_best_kernel, M, N, K, A, Bc, C, alpha, beta, stream)) {
    return;
  }

  const int fallback_order[] = {
      K_WS_64x64x128_C1,
      K_WS_128x64x128_C1,
      K_COOP_128x64x128_C1,
      K_WS_64x64x64_C1,
      K_WS_128x64x64_C1,
      K_AUTO_128x64x64_C8
  };
  for (int kid : fallback_order) {
    if (kid == g_best_kernel) continue;
    if (dispatch_kernel(kid, M, N, K, A, Bc, C, alpha, beta, stream)) return;
  }

  int smem = (64 * 32 + 32 * 64) * int(sizeof(half));
  cudaFuncSetAttribute(wmma_fallback_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  wmma_fallback_kernel<<<(M + 63) / 64, 128, smem, stream>>>(
      reinterpret_cast<const half*>(A),
      reinterpret_cast<const half*>(Bc),
      reinterpret_cast<half*>(C), M, N, K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda_l2_h100_fp32 failed: ") + cudaGetErrorString(err));
  }
}