#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>

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

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
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

static constexpr int AlignA = 8;
static constexpr int AlignB = 8;
static constexpr int AlignC = 8;
static constexpr int AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

#define DEF_KERNEL(Name, TM, TN, TK, CM, CN, CK, EpiSched, MainSched, GemmSched) \
struct Name { \
  using TileShape  = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using GridShape  = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      TileShape, GridShape, \
      cutlass::epilogue::collective::EpilogueTileAuto, \
      ElementAccumulator, ElementCompute, \
      ElementC, LayoutC, AlignC, \
      ElementD, LayoutD, AlignD, \
      cutlass::epilogue::EpiSched, \
      EpilogueOp>::CollectiveOp; \
  using MainCollOp = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
      ElementA, LayoutA, AlignA, \
      ElementB, LayoutB, AlignB, \
      ElementAccumulator, \
      TileShape, GridShape, \
      cutlass::gemm::collective::StageCountAutoCarveout< \
        static_cast<int>(sizeof(typename CollEpi::SharedStorage))>, \
      cutlass::gemm::MainSched>::CollectiveOp; \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal< \
      cute::Shape<int,int,int>, MainCollOp, CollEpi, cutlass::gemm::GemmSched>; \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; \
  using StrideA = typename Gemm::GemmKernel::StrideA; \
  using StrideB = typename Gemm::GemmKernel::StrideB; \
  using StrideC = typename Gemm::GemmKernel::StrideC; \
  using StrideD = typename Gemm::GemmKernel::StrideD; \
};

DEF_KERNEL(K00, 128, 256, 128, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K01, 128, 256, 128, 1, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K02, 128, 256, 128, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K03, 128, 256, 128, 2, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K04, 128, 256, 128, 1, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K05, 128, 256, 128, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K06, 128, 256, 128, 2, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K07, 128, 256, 128, 1, 4, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K08, 128, 256, 128, 1, 2, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)
DEF_KERNEL(K09, 128, 256, 128, 2, 1, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)
DEF_KERNEL(K10, 256, 128, 128, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K11, 256, 128, 128, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K12, 256, 128, 128, 1, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K13, 256, 128, 128, 2, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K14, 256, 128, 128, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K15, 256, 128, 128, 1, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K16, 128, 128, 128, 2, 2, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)
DEF_KERNEL(K17, 128, 128, 128, 1, 2, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)
DEF_KERNEL(K18, 128, 128, 128, 2, 1, 1, TmaWarpSpecialized, KernelTmaWarpSpecializedPingpong, PersistentScheduler)
DEF_KERNEL(K19, 128, 256, 64, 1, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K20, 128, 256, 64, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K21, 128, 256, 64, 2, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K22, 128, 256, 64, 1, 4, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K23, 128, 256, 64, 1, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K24, 128, 256, 64, 2, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, PersistentScheduler)
DEF_KERNEL(K25, 256, 128, 64, 1, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K26, 256, 128, 64, 2, 1, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)
DEF_KERNEL(K27, 256, 128, 64, 1, 2, 1, TmaWarpSpecializedCooperative, KernelTmaWarpSpecializedCooperative, StreamKScheduler)

static constexpr int NUM_KERNELS = 28;

struct GemmState {
  int    active  = -1;
  bool   sm_init = false;
  int    sm_cnt  = 0;
  void*  ws      = nullptr;
  size_t ws_sz   = 0;

  K00::Gemm g00; K00::Gemm::Arguments a00;
  K01::Gemm g01; K01::Gemm::Arguments a01;
  K02::Gemm g02; K02::Gemm::Arguments a02;
  K03::Gemm g03; K03::Gemm::Arguments a03;
  K04::Gemm g04; K04::Gemm::Arguments a04;
  K05::Gemm g05; K05::Gemm::Arguments a05;
  K06::Gemm g06; K06::Gemm::Arguments a06;
  K07::Gemm g07; K07::Gemm::Arguments a07;
  K08::Gemm g08; K08::Gemm::Arguments a08;
  K09::Gemm g09; K09::Gemm::Arguments a09;
  K10::Gemm g10; K10::Gemm::Arguments a10;
  K11::Gemm g11; K11::Gemm::Arguments a11;
  K12::Gemm g12; K12::Gemm::Arguments a12;
  K13::Gemm g13; K13::Gemm::Arguments a13;
  K14::Gemm g14; K14::Gemm::Arguments a14;
  K15::Gemm g15; K15::Gemm::Arguments a15;
  K16::Gemm g16; K16::Gemm::Arguments a16;
  K17::Gemm g17; K17::Gemm::Arguments a17;
  K18::Gemm g18; K18::Gemm::Arguments a18;
  K19::Gemm g19; K19::Gemm::Arguments a19;
  K20::Gemm g20; K20::Gemm::Arguments a20;
  K21::Gemm g21; K21::Gemm::Arguments a21;
  K22::Gemm g22; K22::Gemm::Arguments a22;
  K23::Gemm g23; K23::Gemm::Arguments a23;
  K24::Gemm g24; K24::Gemm::Arguments a24;
  K25::Gemm g25; K25::Gemm::Arguments a25;
  K26::Gemm g26; K26::Gemm::Arguments a26;
  K27::Gemm g27; K27::Gemm::Arguments a27;
};

static GemmState g_st;

static void* ensure_ws(size_t need) {
  need = (need > 0) ? need : 1;
  if (need > g_st.ws_sz) {
    if (g_st.ws) { cudaFree(g_st.ws); g_st.ws = nullptr; }
    cudaMalloc(&g_st.ws, need);
    g_st.ws_sz = need;
  }
  return g_st.ws;
}

template<typename KT>
typename KT::Gemm::Arguments build_args(
    ElementA* pA, ElementB* pB, ElementC* pC, int M, int N, int K)
{
  typename KT::StrideA sA = cutlass::make_cute_packed_stride(
      typename KT::StrideA{}, cute::make_shape(M, K, 1));
  typename KT::StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  typename KT::StrideC sC = cutlass::make_cute_packed_stride(
      typename KT::StrideC{}, cute::make_shape(M, N, 1));
  typename KT::StrideD sD = cutlass::make_cute_packed_stride(
      typename KT::StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw;
  hw.device_id = 0;
  hw.sm_count  = g_st.sm_cnt;

  return typename KT::Gemm::Arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {pA, sA, pB, sB},
    {{1.0f, 0.0f}, pC, sC, pC, sD},
    hw
  };
}

template<typename KT>
float bench_kernel(
    typename KT::Gemm& gemm, typename KT::Gemm::Arguments& saved,
    ElementA* pA, ElementB* pB, ElementC* pC,
    int M, int N, int K, cudaStream_t stream,
    int warmup = 10, int iters = 30)
{
  auto args = build_args<KT>(pA, pB, pC, M, N, K);
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1.f;

  size_t need = KT::Gemm::get_workspace_size(args);
  void* ws = ensure_ws(need);

  if (gemm.initialize(args, ws, stream) != cutlass::Status::kSuccess) return -1.f;

  for (int i = 0; i < warmup; i++) {
    if (gemm.run(stream) != cutlass::Status::kSuccess) return -1.f;
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0, stream);
  for (int i = 0; i < iters; i++) {
    gemm.run(stream);
  }
  cudaEventRecord(ev1, stream);
  cudaEventSynchronize(ev1);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, ev0, ev1);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  saved = args;
  return ms / iters;
}

template<typename KT>
inline bool hot_run(
    typename KT::Gemm& gemm, typename KT::Gemm::Arguments& args,
    ElementA* pA, ElementB* pB, ElementC* pC, cudaStream_t stream)
{
  args.mainloop.ptr_A = pA;
  args.mainloop.ptr_B = pB;
  args.epilogue.ptr_C = pC;
  args.epilogue.ptr_D = pC;
  if (gemm.initialize(args, g_st.ws, stream) != cutlass::Status::kSuccess) return false;
  return gemm.run(stream) == cutlass::Status::kSuccess;
}

static bool do_hot_run(int id, ElementA* pA, ElementB* pB, ElementC* pC, cudaStream_t stream) {
  switch (id) {
    case  0: return hot_run<K00>(g_st.g00, g_st.a00, pA, pB, pC, stream);
    case  1: return hot_run<K01>(g_st.g01, g_st.a01, pA, pB, pC, stream);
    case  2: return hot_run<K02>(g_st.g02, g_st.a02, pA, pB, pC, stream);
    case  3: return hot_run<K03>(g_st.g03, g_st.a03, pA, pB, pC, stream);
    case  4: return hot_run<K04>(g_st.g04, g_st.a04, pA, pB, pC, stream);
    case  5: return hot_run<K05>(g_st.g05, g_st.a05, pA, pB, pC, stream);
    case  6: return hot_run<K06>(g_st.g06, g_st.a06, pA, pB, pC, stream);
    case  7: return hot_run<K07>(g_st.g07, g_st.a07, pA, pB, pC, stream);
    case  8: return hot_run<K08>(g_st.g08, g_st.a08, pA, pB, pC, stream);
    case  9: return hot_run<K09>(g_st.g09, g_st.a09, pA, pB, pC, stream);
    case 10: return hot_run<K10>(g_st.g10, g_st.a10, pA, pB, pC, stream);
    case 11: return hot_run<K11>(g_st.g11, g_st.a11, pA, pB, pC, stream);
    case 12: return hot_run<K12>(g_st.g12, g_st.a12, pA, pB, pC, stream);
    case 13: return hot_run<K13>(g_st.g13, g_st.a13, pA, pB, pC, stream);
    case 14: return hot_run<K14>(g_st.g14, g_st.a14, pA, pB, pC, stream);
    case 15: return hot_run<K15>(g_st.g15, g_st.a15, pA, pB, pC, stream);
    case 16: return hot_run<K16>(g_st.g16, g_st.a16, pA, pB, pC, stream);
    case 17: return hot_run<K17>(g_st.g17, g_st.a17, pA, pB, pC, stream);
    case 18: return hot_run<K18>(g_st.g18, g_st.a18, pA, pB, pC, stream);
    case 19: return hot_run<K19>(g_st.g19, g_st.a19, pA, pB, pC, stream);
    case 20: return hot_run<K20>(g_st.g20, g_st.a20, pA, pB, pC, stream);
    case 21: return hot_run<K21>(g_st.g21, g_st.a21, pA, pB, pC, stream);
    case 22: return hot_run<K22>(g_st.g22, g_st.a22, pA, pB, pC, stream);
    case 23: return hot_run<K23>(g_st.g23, g_st.a23, pA, pB, pC, stream);
    case 24: return hot_run<K24>(g_st.g24, g_st.a24, pA, pB, pC, stream);
    case 25: return hot_run<K25>(g_st.g25, g_st.a25, pA, pB, pC, stream);
    case 26: return hot_run<K26>(g_st.g26, g_st.a26, pA, pB, pC, stream);
    case 27: return hot_run<K27>(g_st.g27, g_st.a27, pA, pB, pC, stream);
    default: return false;
  }
}

static int do_benchmark(ElementA* pA, ElementB* pB, ElementC* pC,
                        int M, int N, int K, cudaStream_t stream)
{
  float best_t = 1e18f;
  int   best_id = -1;

  auto check = [&](float t, int id) {
    if (t > 0.f && t < best_t) { best_t = t; best_id = id; }
  };

  check(bench_kernel<K00>(g_st.g00, g_st.a00, pA, pB, pC, M, N, K, stream),  0);
  check(bench_kernel<K01>(g_st.g01, g_st.a01, pA, pB, pC, M, N, K, stream),  1);
  check(bench_kernel<K02>(g_st.g02, g_st.a02, pA, pB, pC, M, N, K, stream),  2);
  check(bench_kernel<K03>(g_st.g03, g_st.a03, pA, pB, pC, M, N, K, stream),  3);
  check(bench_kernel<K04>(g_st.g04, g_st.a04, pA, pB, pC, M, N, K, stream),  4);
  check(bench_kernel<K05>(g_st.g05, g_st.a05, pA, pB, pC, M, N, K, stream),  5);
  check(bench_kernel<K06>(g_st.g06, g_st.a06, pA, pB, pC, M, N, K, stream),  6);
  check(bench_kernel<K07>(g_st.g07, g_st.a07, pA, pB, pC, M, N, K, stream),  7);
  check(bench_kernel<K08>(g_st.g08, g_st.a08, pA, pB, pC, M, N, K, stream),  8);
  check(bench_kernel<K09>(g_st.g09, g_st.a09, pA, pB, pC, M, N, K, stream),  9);
  check(bench_kernel<K10>(g_st.g10, g_st.a10, pA, pB, pC, M, N, K, stream), 10);
  check(bench_kernel<K11>(g_st.g11, g_st.a11, pA, pB, pC, M, N, K, stream), 11);
  check(bench_kernel<K12>(g_st.g12, g_st.a12, pA, pB, pC, M, N, K, stream), 12);
  check(bench_kernel<K13>(g_st.g13, g_st.a13, pA, pB, pC, M, N, K, stream), 13);
  check(bench_kernel<K14>(g_st.g14, g_st.a14, pA, pB, pC, M, N, K, stream), 14);
  check(bench_kernel<K15>(g_st.g15, g_st.a15, pA, pB, pC, M, N, K, stream), 15);
  check(bench_kernel<K16>(g_st.g16, g_st.a16, pA, pB, pC, M, N, K, stream), 16);
  check(bench_kernel<K17>(g_st.g17, g_st.a17, pA, pB, pC, M, N, K, stream), 17);
  check(bench_kernel<K18>(g_st.g18, g_st.a18, pA, pB, pC, M, N, K, stream), 18);
  check(bench_kernel<K19>(g_st.g19, g_st.a19, pA, pB, pC, M, N, K, stream), 19);
  check(bench_kernel<K20>(g_st.g20, g_st.a20, pA, pB, pC, M, N, K, stream), 20);
  check(bench_kernel<K21>(g_st.g21, g_st.a21, pA, pB, pC, M, N, K, stream), 21);
  check(bench_kernel<K22>(g_st.g22, g_st.a22, pA, pB, pC, M, N, K, stream), 22);
  check(bench_kernel<K23>(g_st.g23, g_st.a23, pA, pB, pC, M, N, K, stream), 23);
  check(bench_kernel<K24>(g_st.g24, g_st.a24, pA, pB, pC, M, N, K, stream), 24);
  check(bench_kernel<K25>(g_st.g25, g_st.a25, pA, pB, pC, M, N, K, stream), 25);
  check(bench_kernel<K26>(g_st.g26, g_st.a26, pA, pB, pC, M, N, K, stream), 26);
  check(bench_kernel<K27>(g_st.g27, g_st.a27, pA, pB, pC, M, N, K, stream), 27);

  return best_id;
}

#endif

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
  CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

  auto* pA = reinterpret_cast<ElementA*>(a.data_ptr());
  auto* pB = reinterpret_cast<ElementB*>(b_col_major.data_ptr());
  auto* pC = reinterpret_cast<ElementC*>(c.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (g_st.active >= 0) {
    bool ok = do_hot_run(g_st.active, pA, pB, pC, stream);
    if (ok) return;
    g_st.active = -1;
  }

  if (!g_st.sm_init) {
    int dev = 0;
    cudaGetDevice(&dev);
    g_st.sm_cnt = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    g_st.sm_init = true;
  }

  int best_id = do_benchmark(pA, pB, pC, M, N, K, stream);
  if (best_id < 0) {
    throw std::runtime_error("All CUTLASS SM90 GEMM kernels failed");
  }
  g_st.active = best_id;

  bool ok = do_hot_run(g_st.active, pA, pB, pC, stream);
  if (!ok) {
    g_st.active = -1;
    throw std::runtime_error("CUTLASS GEMM run failed after benchmark selection");
  }

#else
  throw std::runtime_error("CUTLASS SM90 MMA not supported on this device");
#endif
}