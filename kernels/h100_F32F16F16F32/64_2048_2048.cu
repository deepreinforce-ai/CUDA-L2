#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <torch/types.h>

#include <stdexcept>
#include <string>
#include <mutex>
#include <limits>
#include <type_traits>

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

static inline void check_tensor_2d_half_cuda_contig(const torch::Tensor& t, const char* name) {
  if (!t.is_cuda()) throw std::runtime_error(std::string(name) + " must be CUDA tensor");
  if (t.scalar_type() != torch::kHalf) throw std::runtime_error(std::string(name) + " must be torch.float16");
  if (t.dim() != 2) throw std::runtime_error(std::string(name) + " must be 2D");
  if (!t.is_contiguous()) throw std::runtime_error(std::string(name) + " must be contiguous");
}

static inline void check_cuda(const char* tag) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(tag) + " failed: " + cudaGetErrorString(err));
  }
}

__global__ void hgemm_fallback_kernel(const half* __restrict__ A,
                                      const half* __restrict__ Bcol,
                                      half* __restrict__ C,
                                      int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
      acc += __half2float(A[row * K + k]) * __half2float(Bcol[col * K + k]);
    }
    C[row * N + col] = __float2half_rn(acc);
  }
}

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutBCol = cutlass::layout::ColumnMajor;
using LayoutBRow = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 16;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using TileShapeA    = cute::Shape<cute::_64, cute::_32,  cute::_64>;
using WorkShapeA    = cute::Shape<cute::_1,  cute::_2,   cute::_1>;

using TileShapeF    = cute::Shape<cute::_64, cute::_32,  cute::_128>;
using WorkShapeF    = cute::Shape<cute::_1,  cute::_2,   cute::_1>;

using TileShapeC    = cute::Shape<cute::_64, cute::_16,  cute::_128>;
using WorkShapeC    = cute::Shape<cute::_1,  cute::_1,   cute::_1>;

using TileShapeD    = cute::Shape<cute::_64, cute::_16,  cute::_64>;
using WorkShapeD    = cute::Shape<cute::_1,  cute::_2,   cute::_1>;

using TileShapeE    = cute::Shape<cute::_64, cute::_8,   cute::_128>;
using WorkShapeE    = cute::Shape<cute::_1,  cute::_1,   cute::_1>;

using TileShapeM    = cute::Shape<cute::_64, cute::_8,   cute::_64>;
using WorkShapeM    = cute::Shape<cute::_1,  cute::_1,   cute::_1>;

template <class LayoutB, class TileShape, class WorkShape, int Stages>
struct GemmVariant {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, WorkShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, WorkShape,
      cutlass::gemm::collective::StageCount<Stages>,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using GemmColA = typename GemmVariant<LayoutBCol, TileShapeA, WorkShapeA, 5>::DeviceGemm;
using GemmColF = typename GemmVariant<LayoutBCol, TileShapeF, WorkShapeF, 5>::DeviceGemm;
using GemmColC = typename GemmVariant<LayoutBCol, TileShapeC, WorkShapeC, 6>::DeviceGemm;
using GemmColD = typename GemmVariant<LayoutBCol, TileShapeD, WorkShapeD, 5>::DeviceGemm;
using GemmColE = typename GemmVariant<LayoutBCol, TileShapeE, WorkShapeE, 5>::DeviceGemm;
using GemmColM = typename GemmVariant<LayoutBCol, TileShapeM, WorkShapeM, 4>::DeviceGemm;

using GemmRowC = typename GemmVariant<LayoutBRow, TileShapeC, WorkShapeC, 6>::DeviceGemm;
using GemmRowE = typename GemmVariant<LayoutBRow, TileShapeE, WorkShapeE, 5>::DeviceGemm;

template <typename GemmT>
static typename GemmT::Arguments make_args(
    const ElementA* ptr_A,
    const ElementB* ptr_B,
    ElementC* ptr_C,
    const cutlass::KernelHardwareInfo& hw_info) {

  using StrideA = typename GemmT::GemmKernel::StrideA;
  using StrideB = typename GemmT::GemmKernel::StrideB;
  using StrideC = typename GemmT::GemmKernel::StrideC;
  using StrideD = typename GemmT::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(64, 2048, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(2048, 2048, 1));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(64, 2048, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(64, 2048, 1));

  float alpha = 1.0f;
  float beta  = 0.0f;

  return typename GemmT::Arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {64, 2048, 2048},
    {ptr_A, stride_A, ptr_B, stride_B},
    {{alpha, beta}, ptr_C, stride_C, ptr_C, stride_D},
    hw_info
  };
}

template <typename GemmT>
static size_t get_ws_size(const ElementA* ptr_A, const ElementB* ptr_B, ElementC* ptr_C,
                          const cutlass::KernelHardwareInfo& hw_info) {
  auto args = make_args<GemmT>(ptr_A, ptr_B, ptr_C, hw_info);
  return GemmT::get_workspace_size(args);
}

template <typename GemmT>
struct PersistentRunner {
  GemmT gemm;
  cutlass::KernelHardwareInfo hw{};
  void* workspace = nullptr;
  size_t workspace_bytes = 0;

  bool initialized = false;
  const ElementA* last_A = nullptr;
  const ElementB* last_B = nullptr;
  ElementC* last_C = nullptr;

  cutlass::Status run(const ElementA* A, const ElementB* B, ElementC* C, cudaStream_t stream) {
    if (!initialized || A != last_A || B != last_B || C != last_C) {
      auto args = make_args<GemmT>(A, B, C, hw);
      cutlass::Status st = gemm.initialize(args, workspace, stream);
      if (st != cutlass::Status::kSuccess) return st;
      initialized = true;
      last_A = A; last_B = B; last_C = C;
    }
    return gemm.run(stream);
  }

  cutlass::Status run_reinit(const ElementA* A, const ElementB* B, ElementC* C, cudaStream_t stream) {
    auto args = make_args<GemmT>(A, B, C, hw);
    cutlass::Status st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) return st;
    initialized = true;
    last_A = A; last_B = B; last_C = C;
    return gemm.run(stream);
  }
};

enum FastKernelChoice : int {
  CH_COL_A = 0,
  CH_COL_F = 1,
  CH_COL_C = 2,
  CH_COL_D = 3,
  CH_COL_E = 4,
  CH_COL_M = 5,
  CH_ROW_C = 6,
  CH_ROW_E = 7
};

struct FastRuntimeState {
  PersistentRunner<GemmColA> colA;
  PersistentRunner<GemmColF> colF;
  PersistentRunner<GemmColC> colC;
  PersistentRunner<GemmColD> colD;
  PersistentRunner<GemmColE> colE;
  PersistentRunner<GemmColM> colM;
  PersistentRunner<GemmRowC> rowC;
  PersistentRunner<GemmRowE> rowE;
  int best = CH_COL_C;
};

static FastRuntimeState g_state;
static std::once_flag g_init_once;
static std::once_flag g_tune_once;

static void alloc_ws(void** ptr, size_t bytes) {
  if (bytes == 0) { *ptr = nullptr; return; }
  cudaError_t e = cudaMalloc(ptr, bytes);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc workspace failed: ") + cudaGetErrorString(e));
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)

{
  bool fast_shape = (a.is_cuda() && b.is_cuda() && b_col_major.is_cuda() && c.is_cuda() &&
                     a.scalar_type() == torch::kHalf &&
                     b.scalar_type() == torch::kHalf &&
                     b_col_major.scalar_type() == torch::kHalf &&
                     c.scalar_type() == torch::kHalf &&
                     a.dim() == 2 && b.dim() == 2 && b_col_major.dim() == 2 && c.dim() == 2 &&
                     a.is_contiguous() && b.is_contiguous() && b_col_major.is_contiguous() && c.is_contiguous() &&
                     a.size(0) == 64 && a.size(1) == 2048 &&
                     b.size(0) == 2048 && b.size(1) == 2048 &&
                     b_col_major.size(0) == 2048 && b_col_major.size(1) == 2048 &&
                     c.size(0) == 64 && c.size(1) == 2048);

  cudaStream_t stream = 0;

  if (fast_shape) {
    const ElementA* ptr_A    = reinterpret_cast<const ElementA*>(a.data_ptr());
    const ElementB* ptr_Bcol = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
    const ElementB* ptr_Brow = reinterpret_cast<const ElementB*>(b.data_ptr());
    ElementC* ptr_C          = reinterpret_cast<ElementC*>(c.data_ptr());

    std::call_once(g_init_once, [&]() {
      int device_id = 0;
      cudaGetDevice(&device_id);

      g_state.colA.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmColA::GemmKernel>(device_id);
      g_state.colF.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmColF::GemmKernel>(device_id);
      g_state.colC.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmColC::GemmKernel>(device_id);
      g_state.colD.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmColD::GemmKernel>(device_id);
      g_state.colE.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmColE::GemmKernel>(device_id);
      g_state.colM.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmColM::GemmKernel>(device_id);
      g_state.rowC.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmRowC::GemmKernel>(device_id);
      g_state.rowE.hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmRowE::GemmKernel>(device_id);

      g_state.colA.workspace_bytes = get_ws_size<GemmColA>(ptr_A, ptr_Bcol, ptr_C, g_state.colA.hw);
      g_state.colF.workspace_bytes = get_ws_size<GemmColF>(ptr_A, ptr_Bcol, ptr_C, g_state.colF.hw);
      g_state.colC.workspace_bytes = get_ws_size<GemmColC>(ptr_A, ptr_Bcol, ptr_C, g_state.colC.hw);
      g_state.colD.workspace_bytes = get_ws_size<GemmColD>(ptr_A, ptr_Bcol, ptr_C, g_state.colD.hw);
      g_state.colE.workspace_bytes = get_ws_size<GemmColE>(ptr_A, ptr_Bcol, ptr_C, g_state.colE.hw);
      g_state.colM.workspace_bytes = get_ws_size<GemmColM>(ptr_A, ptr_Bcol, ptr_C, g_state.colM.hw);
      g_state.rowC.workspace_bytes = get_ws_size<GemmRowC>(ptr_A, ptr_Brow, ptr_C, g_state.rowC.hw);
      g_state.rowE.workspace_bytes = get_ws_size<GemmRowE>(ptr_A, ptr_Brow, ptr_C, g_state.rowE.hw);

      alloc_ws(&g_state.colA.workspace, g_state.colA.workspace_bytes);
      alloc_ws(&g_state.colF.workspace, g_state.colF.workspace_bytes);
      alloc_ws(&g_state.colC.workspace, g_state.colC.workspace_bytes);
      alloc_ws(&g_state.colD.workspace, g_state.colD.workspace_bytes);
      alloc_ws(&g_state.colE.workspace, g_state.colE.workspace_bytes);
      alloc_ws(&g_state.colM.workspace, g_state.colM.workspace_bytes);
      alloc_ws(&g_state.rowC.workspace, g_state.rowC.workspace_bytes);
      alloc_ws(&g_state.rowE.workspace, g_state.rowE.workspace_bytes);
    });

    std::call_once(g_tune_once, [&]() {
      cudaEvent_t s, e;
      cudaEventCreate(&s);
      cudaEventCreate(&e);

      (void)g_state.colA.run_reinit(ptr_A, ptr_Bcol, ptr_C, stream);
      (void)g_state.colF.run_reinit(ptr_A, ptr_Bcol, ptr_C, stream);
      (void)g_state.colC.run_reinit(ptr_A, ptr_Bcol, ptr_C, stream);
      (void)g_state.colD.run_reinit(ptr_A, ptr_Bcol, ptr_C, stream);
      (void)g_state.colE.run_reinit(ptr_A, ptr_Bcol, ptr_C, stream);
      (void)g_state.colM.run_reinit(ptr_A, ptr_Bcol, ptr_C, stream);
      (void)g_state.rowC.run_reinit(ptr_A, ptr_Brow, ptr_C, stream);
      (void)g_state.rowE.run_reinit(ptr_A, ptr_Brow, ptr_C, stream);
      cudaDeviceSynchronize();

      auto bench_hot = [&](auto& runner, const ElementB* Bptr) -> float {
        constexpr int ITERS = 12;
        cudaEventRecord(s, stream);
        #pragma unroll
        for (int i = 0; i < ITERS; ++i) {
          (void)runner.run(ptr_A, Bptr, ptr_C, stream);
        }
        cudaEventRecord(e, stream);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        return ms / ITERS;
      };

      float t_colA = bench_hot(g_state.colA, ptr_Bcol);
      float t_colF = bench_hot(g_state.colF, ptr_Bcol);
      float t_colC = bench_hot(g_state.colC, ptr_Bcol);
      float t_colD = bench_hot(g_state.colD, ptr_Bcol);
      float t_colE = bench_hot(g_state.colE, ptr_Bcol);
      float t_colM = bench_hot(g_state.colM, ptr_Bcol);
      float t_rowC = bench_hot(g_state.rowC, ptr_Brow);
      float t_rowE = bench_hot(g_state.rowE, ptr_Brow);

      float best_t = t_colC;
      g_state.best = CH_COL_C;
      if (t_colE < best_t) { best_t = t_colE; g_state.best = CH_COL_E; }
      if (t_colM < best_t) { best_t = t_colM; g_state.best = CH_COL_M; }
      if (t_colD < best_t) { best_t = t_colD; g_state.best = CH_COL_D; }
      if (t_colF < best_t) { best_t = t_colF; g_state.best = CH_COL_F; }
      if (t_colA < best_t) { best_t = t_colA; g_state.best = CH_COL_A; }
      if (t_rowE < best_t) { best_t = t_rowE; g_state.best = CH_ROW_E; }
      if (t_rowC < best_t) { best_t = t_rowC; g_state.best = CH_ROW_C; }

      cudaEventDestroy(s);
      cudaEventDestroy(e);
    });

    cutlass::Status st = cutlass::Status::kInvalid;
    switch (g_state.best) {
      case CH_COL_E: st = g_state.colE.run(ptr_A, ptr_Bcol, ptr_C, stream); break;
      case CH_COL_M: st = g_state.colM.run(ptr_A, ptr_Bcol, ptr_C, stream); break;
      case CH_COL_D: st = g_state.colD.run(ptr_A, ptr_Bcol, ptr_C, stream); break;
      case CH_COL_F: st = g_state.colF.run(ptr_A, ptr_Bcol, ptr_C, stream); break;
      case CH_COL_A: st = g_state.colA.run(ptr_A, ptr_Bcol, ptr_C, stream); break;
      case CH_ROW_C: st = g_state.rowC.run(ptr_A, ptr_Brow, ptr_C, stream); break;
      case CH_ROW_E: st = g_state.rowE.run(ptr_A, ptr_Brow, ptr_C, stream); break;
      default:       st = g_state.colC.run(ptr_A, ptr_Bcol, ptr_C, stream); break;
    }

    if (st != cutlass::Status::kSuccess) st = g_state.colC.run(ptr_A, ptr_Bcol, ptr_C, stream);
    if (st != cutlass::Status::kSuccess) st = g_state.colE.run(ptr_A, ptr_Bcol, ptr_C, stream);
    if (st != cutlass::Status::kSuccess) st = g_state.colM.run(ptr_A, ptr_Bcol, ptr_C, stream);
    if (st != cutlass::Status::kSuccess) st = g_state.colD.run(ptr_A, ptr_Bcol, ptr_C, stream);
    if (st != cutlass::Status::kSuccess) st = g_state.colF.run(ptr_A, ptr_Bcol, ptr_C, stream);
    if (st != cutlass::Status::kSuccess) st = g_state.colA.run(ptr_A, ptr_Bcol, ptr_C, stream);
    if (st != cutlass::Status::kSuccess) st = g_state.rowE.run(ptr_A, ptr_Brow, ptr_C, stream);
    if (st != cutlass::Status::kSuccess) st = g_state.rowC.run(ptr_A, ptr_Brow, ptr_C, stream);

    if (st != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS run failed for all fast variants");
    }

    check_cuda("cuda_l2_h100_fp32 fast path");
    return;
  }

  check_tensor_2d_half_cuda_contig(a, "a");
  check_tensor_2d_half_cuda_contig(b, "b");
  check_tensor_2d_half_cuda_contig(b_col_major, "b_col_major");
  check_tensor_2d_half_cuda_contig(c, "c");

  const int64_t M  = a.size(0);
  const int64_t K  = a.size(1);
  const int64_t Kb = b.size(0);
  const int64_t N  = b.size(1);

  if (Kb != K) throw std::runtime_error("Shape mismatch: b.size(0) must equal a.size(1)");
  if (c.size(0) != M || c.size(1) != N) throw std::runtime_error("Shape mismatch: c must be (M,N)");
  if (b_col_major.size(0) != K || b_col_major.size(1) != N) throw std::runtime_error("Shape mismatch: b_col_major must be (K,N)");

  hgemm_fallback_kernel<<<dim3((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16)),
                          dim3(16, 16), 0, stream>>>(
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b_col_major.data_ptr()),
      reinterpret_cast<half*>(c.data_ptr()),
      (int)M, (int)N, (int)K);

  check_cuda("hgemm_fallback_kernel");
}