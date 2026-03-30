#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include <torch/extension.h>
#include <torch/types.h>

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ArchTag = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;

#define DEF_PP_S(NAME, TM, TN, TK, CM, CN, CK, STAGES) \
struct NAME { \
  using TS = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using CS = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CE = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OpClass, TS, CS, cutlass::epilogue::collective::EpilogueTileAuto, \
      ElementAccumulator, ElementAccumulator, \
      ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC, \
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp; \
  using CM_ = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OpClass, ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, \
      ElementAccumulator, TS, CS, cutlass::gemm::collective::StageCount<STAGES>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
  using GK   = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM_, CE>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>; \
  using SA = typename GK::StrideA; using SB = typename GK::StrideB; \
  using SC = typename GK::StrideC; using SD = typename GK::StrideD; \
};

#define DEF_COOP_S(NAME, TM, TN, TK, CM, CN, CK, STAGES) \
struct NAME { \
  using TS = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using CS = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CE = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OpClass, TS, CS, cutlass::epilogue::collective::EpilogueTileAuto, \
      ElementAccumulator, ElementAccumulator, \
      ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC, \
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp; \
  using CM_ = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OpClass, ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, \
      ElementAccumulator, TS, CS, cutlass::gemm::collective::StageCount<STAGES>, \
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp; \
  using GK   = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM_, CE>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>; \
  using SA = typename GK::StrideA; using SB = typename GK::StrideB; \
  using SC = typename GK::StrideC; using SD = typename GK::StrideD; \
};

#define DEF_PP_AUTO(NAME, TM, TN, TK, CM, CN, CK) \
struct NAME { \
  using TS = cute::Shape<cute::_##TM, cute::_##TN, cute::_##TK>; \
  using CS = cute::Shape<cute::_##CM, cute::_##CN, cute::_##CK>; \
  using CE = typename cutlass::epilogue::collective::CollectiveBuilder< \
      ArchTag, OpClass, TS, CS, cutlass::epilogue::collective::EpilogueTileAuto, \
      ElementAccumulator, ElementAccumulator, \
      ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC, \
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp; \
  using CM_ = typename cutlass::gemm::collective::CollectiveBuilder< \
      ArchTag, OpClass, ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, \
      ElementAccumulator, TS, CS, \
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CE::SharedStorage))>, \
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp; \
  using GK   = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM_, CE>; \
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>; \
  using SA = typename GK::StrideA; using SB = typename GK::StrideB; \
  using SC = typename GK::StrideC; using SD = typename GK::StrideD; \
};

DEF_PP_S   (V00, 128, 256, 64,  1, 8, 1, 4)
DEF_PP_S   (V01, 128, 256, 64,  1, 8, 1, 5)
DEF_COOP_S (V02, 128, 256, 64,  1, 8, 1, 4)
DEF_COOP_S (V03, 128, 256, 64,  1, 8, 1, 5)
DEF_PP_S   (V04, 128, 256, 64,  1, 4, 1, 4)
DEF_PP_S   (V05, 128, 256, 64,  1, 4, 1, 5)
DEF_COOP_S (V06, 128, 256, 64,  1, 4, 1, 4)
DEF_COOP_S (V07, 128, 256, 64,  1, 4, 1, 5)
DEF_PP_S   (V08, 128, 256, 128, 1, 4, 1, 3)
DEF_PP_S   (V09, 128, 256, 128, 1, 2, 1, 3)
DEF_COOP_S (V10, 128, 256, 128, 1, 4, 1, 3)
DEF_COOP_S (V11, 128, 256, 128, 1, 2, 1, 3)
DEF_PP_S   (V12, 128, 128, 128, 1, 8, 1, 3)
DEF_PP_S   (V13, 128, 128, 64,  1, 8, 1, 4)
DEF_COOP_S (V14, 128, 128, 64,  1, 8, 1, 4)
DEF_PP_AUTO(V15, 128, 256, 64,  1, 8, 1)
DEF_PP_AUTO(V16, 128, 256, 128, 1, 4, 1)

#define WF_BM 128
#define WF_BN 256
#define WF_BK 64
#define WF_WARP_N 4
#define WF_WMT 4
#define WF_WNT 4
#define WF_WKS 4
#define WF_PAD_A 8
#define WF_PAD_B 8
#define WF_SA_STRIDE (WF_BK + WF_PAD_A)
#define WF_SB_STRIDE (WF_BN + WF_PAD_B)

static constexpr size_t WF_SMEM_SIZE =
    sizeof(__half) * (2 * WF_BM * WF_SA_STRIDE + 2 * WF_BK * WF_SB_STRIDE);

__global__ void __launch_bounds__(256, 1)
hgemm_wmma_wide(const __half* __restrict__ A, const __half* __restrict__ B_col,
                __half* __restrict__ C, int M, int N, int K) {
  extern __shared__ __half dyn_smem[];
  __half (*smA)[WF_BM][WF_SA_STRIDE] = reinterpret_cast<__half(*)[WF_BM][WF_SA_STRIDE]>(dyn_smem);
  __half (*smB)[WF_BK][WF_SB_STRIDE] = reinterpret_cast<__half(*)[WF_BK][WF_SB_STRIDE]>(
      dyn_smem + 2 * WF_BM * WF_SA_STRIDE);

  int cta_n = blockIdx.x * WF_BN;
  if (cta_n >= N) return;
  int tid = threadIdx.x, warp_id = tid >> 5;
  int warp_row = warp_id / WF_WARP_N, warp_col = warp_id % WF_WARP_N;
  int wm_base = warp_row * (WF_WMT * 16), wn_base = warp_col * (WF_WNT * 16);

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> acc[WF_WMT][WF_WNT];
  #pragma unroll
  for (int i=0;i<WF_WMT;i++) for (int j=0;j<WF_WNT;j++) nvcuda::wmma::fill_fragment(acc[i][j], 0.0f);

  constexpr int A_PASSES = (WF_BM * WF_BK) / (256 * 8);
  constexpr int B_PASSES = (WF_BK * WF_BN) / (256 * 8);

  #pragma unroll
  for (int p=0;p<A_PASSES;p++){
    int idx = p*256 + tid; int row = idx / (WF_BK/8), c8 = idx % (WF_BK/8);
    float4 v={0,0,0,0}; if (row<M && c8*8<K) v=*reinterpret_cast<const float4*>(&A[row*K + c8*8]);
    *reinterpret_cast<float4*>(&smA[0][row][c8*8]) = v;
  }
  #pragma unroll
  for (int p=0;p<B_PASSES;p++){
    int idx = p*256 + tid; int ln = idx / (WF_BK/8), lk8 = idx % (WF_BK/8), gn = cta_n + ln;
    float4 v={0,0,0,0}; if (gn<N && lk8*8<K) v=*reinterpret_cast<const float4*>(&B_col[gn*K + lk8*8]);
    *reinterpret_cast<float4*>(&smB[0][lk8*8][ln]) = v;
  }
  __syncthreads();

  int nkt = (K + WF_BK - 1) / WF_BK, cur = 0;
  for (int kt=0; kt<nkt; kt++) {
    int nxt = 1-cur, nk = (kt+1)*WF_BK;
    if (nk < K) {
      #pragma unroll
      for (int p=0;p<A_PASSES;p++){
        int idx = p*256 + tid; int row = idx / (WF_BK/8), c8 = idx % (WF_BK/8);
        float4 v={0,0,0,0}; if (row<M && nk + c8*8 < K) v=*reinterpret_cast<const float4*>(&A[row*K + nk + c8*8]);
        *reinterpret_cast<float4*>(&smA[nxt][row][c8*8]) = v;
      }
      #pragma unroll
      for (int p=0;p<B_PASSES;p++){
        int idx = p*256 + tid; int ln = idx / (WF_BK/8), lk8 = idx % (WF_BK/8), gn = cta_n + ln;
        float4 v={0,0,0,0}; if (gn<N && nk + lk8*8 < K) v=*reinterpret_cast<const float4*>(&B_col[gn*K + nk + lk8*8]);
        *reinterpret_cast<float4*>(&smB[nxt][lk8*8][ln]) = v;
      }
    }

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__half,nvcuda::wmma::row_major> af[WF_WMT];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__half,nvcuda::wmma::row_major> bf[WF_WNT];
    #pragma unroll
    for (int ki=0; ki<WF_WKS; ki++) {
      #pragma unroll
      for (int i=0;i<WF_WMT;i++) nvcuda::wmma::load_matrix_sync(af[i], &smA[cur][wm_base + i*16][ki*16], WF_SA_STRIDE);
      #pragma unroll
      for (int j=0;j<WF_WNT;j++) nvcuda::wmma::load_matrix_sync(bf[j], &smB[cur][ki*16][wn_base + j*16], WF_SB_STRIDE);
      #pragma unroll
      for (int i=0;i<WF_WMT;i++) for (int j=0;j<WF_WNT;j++) nvcuda::wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
    cur = nxt;
  }

  #pragma unroll
  for (int i=0;i<WF_WMT;i++) for (int j=0;j<WF_WNT;j++) {
    int r = wm_base + i*16, c = cta_n + wn_base + j*16;
    if (r < M && c + 15 < N) {
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,__half> of;
      #pragma unroll
      for (int e=0;e<(int)acc[i][j].num_elements;e++) of.x[e] = __float2half(acc[i][j].x[e]);
      nvcuda::wmma::store_matrix_sync(C + r*N + c, of, N, nvcuda::wmma::mem_row_major);
    }
  }
}

static int          g_variant        = -1;
static uint8_t*     g_workspace      = nullptr;
static size_t       g_workspace_sz   = 0;
static cudaStream_t g_stream         = nullptr;
static bool         g_stream_created = false;
static bool         g_wmma_cfg       = false;

static cudaStream_t get_stream() {
  if (!g_stream_created) {
    cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
    g_stream_created = true;
  }
  return g_stream;
}
static bool ensure_workspace(size_t needed) {
  if (needed == 0 || needed <= g_workspace_sz) return true;
  if (g_workspace) { cudaFree(g_workspace); g_workspace = nullptr; g_workspace_sz = 0; }
  if (cudaMalloc(&g_workspace, needed) != cudaSuccess) return false;
  g_workspace_sz = needed;
  return true;
}

template <typename VS>
static typename VS::Gemm::Arguments make_args(int M,int N,int K,void* pA,void* pB,void* pC){
  typename VS::SA sa = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(M*K));
  typename VS::SB sb = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
  typename VS::SC sc = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(M*N));
  typename VS::SD sd = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(M*N));

  int dev=0; cudaGetDevice(&dev);
  auto hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename VS::GK>(dev);

  return typename VS::Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K},
      {(ElementA*)pA, sa, (ElementB*)pB, sb},
      {{1.f,0.f}, (ElementC*)pC, sc, (ElementC*)pC, sd}, hw
  };
}

template <typename VS>
static bool run_v(int M,int N,int K,void* A,void* B,void* C){
  auto args = make_args<VS>(M,N,K,A,B,C);
  typename VS::Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
  size_t ws = VS::Gemm::get_workspace_size(args);
  if (!ensure_workspace(ws)) return false;
  if (gemm.initialize(args, g_workspace, get_stream()) != cutlass::Status::kSuccess) return false;
  if (gemm.run(get_stream()) != cutlass::Status::kSuccess) return false;
  return cudaGetLastError() == cudaSuccess;
}

template <typename VS>
static float time_v(int M,int N,int K,void* A,void* B,void* C){
  auto args = make_args<VS>(M,N,K,A,B,C);
  typename VS::Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return FLT_MAX;
  size_t ws = VS::Gemm::get_workspace_size(args);
  if (!ensure_workspace(ws)) return FLT_MAX;

  for (int i=0;i<2;i++) {
    if (gemm.initialize(args, g_workspace, get_stream()) != cutlass::Status::kSuccess) return FLT_MAX;
    if (gemm.run(get_stream()) != cutlass::Status::kSuccess) return FLT_MAX;
  }
  if (cudaStreamSynchronize(get_stream()) != cudaSuccess) return FLT_MAX;

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0, get_stream());
  constexpr int REPS=8;
  for (int i=0;i<REPS;i++){
    gemm.initialize(args, g_workspace, get_stream());
    gemm.run(get_stream());
  }
  cudaEventRecord(t1, get_stream());
  cudaEventSynchronize(t1);
  float ms=FLT_MAX;
  cudaEventElapsedTime(&ms,t0,t1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  return ms / REPS;
}

static bool run_by_id(int id, int M,int N,int K, void* A,void* B,void* C) {
  switch(id){
    case 0: return run_v<V00>(M,N,K,A,B,C);
    case 1: return run_v<V01>(M,N,K,A,B,C);
    case 2: return run_v<V02>(M,N,K,A,B,C);
    case 3: return run_v<V03>(M,N,K,A,B,C);
    case 4: return run_v<V04>(M,N,K,A,B,C);
    case 5: return run_v<V05>(M,N,K,A,B,C);
    case 6: return run_v<V06>(M,N,K,A,B,C);
    case 7: return run_v<V07>(M,N,K,A,B,C);
    case 8: return run_v<V08>(M,N,K,A,B,C);
    case 9: return run_v<V09>(M,N,K,A,B,C);
    case 10:return run_v<V10>(M,N,K,A,B,C);
    case 11:return run_v<V11>(M,N,K,A,B,C);
    case 12:return run_v<V12>(M,N,K,A,B,C);
    case 13:return run_v<V13>(M,N,K,A,B,C);
    case 14:return run_v<V14>(M,N,K,A,B,C);
    case 15:return run_v<V15>(M,N,K,A,B,C);
    case 16:return run_v<V16>(M,N,K,A,B,C);
    default: return false;
  }
}

static float time_by_id(int id, int M,int N,int K, void* A,void* B,void* C) {
  switch(id){
    case 0: return time_v<V00>(M,N,K,A,B,C);
    case 1: return time_v<V01>(M,N,K,A,B,C);
    case 2: return time_v<V02>(M,N,K,A,B,C);
    case 3: return time_v<V03>(M,N,K,A,B,C);
    case 4: return time_v<V04>(M,N,K,A,B,C);
    case 5: return time_v<V05>(M,N,K,A,B,C);
    case 6: return time_v<V06>(M,N,K,A,B,C);
    case 7: return time_v<V07>(M,N,K,A,B,C);
    case 8: return time_v<V08>(M,N,K,A,B,C);
    case 9: return time_v<V09>(M,N,K,A,B,C);
    case 10:return time_v<V10>(M,N,K,A,B,C);
    case 11:return time_v<V11>(M,N,K,A,B,C);
    case 12:return time_v<V12>(M,N,K,A,B,C);
    case 13:return time_v<V13>(M,N,K,A,B,C);
    case 14:return time_v<V14>(M,N,K,A,B,C);
    case 15:return time_v<V15>(M,N,K,A,B,C);
    case 16:return time_v<V16>(M,N,K,A,B,C);
    default: return FLT_MAX;
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
  const int M = (int)a.size(0);
  const int K = (int)a.size(1);
  const int N = (int)b.size(1);

  void* pA = a.data_ptr();
  void* pB = b_col_major.data_ptr();
  void* pC = c.data_ptr();

  if (g_variant >= 0) {
    if (run_by_id(g_variant, M,N,K,pA,pB,pC)) return;
    g_variant = -1;
    cudaGetLastError();
  }

  if (M == 128 && K == 1024 && (N % 256 == 0)) {
    const int fast_ids[] = {0,1,2,3,15,8,10,4,5,6,7,9,11,16,13,14,12};
    for (int i = 0; i < (int)(sizeof(fast_ids)/sizeof(fast_ids[0])); ++i) {
      int id = fast_ids[i];
      if (run_by_id(id, M,N,K,pA,pB,pC)) { g_variant = id; return; }
    }
  }

  float best = FLT_MAX; int best_id = -1;
  for (int id = 0; id <= 16; ++id) {
    float t = time_by_id(id, M,N,K,pA,pB,pC);
    if (t < best) { best = t; best_id = id; }
  }
  if (best_id >= 0 && run_by_id(best_id, M,N,K,pA,pB,pC)) {
    g_variant = best_id;
    return;
  }

  if (!g_wmma_cfg) {
    cudaFuncSetAttribute(hgemm_wmma_wide, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)WF_SMEM_SIZE);
    g_wmma_cfg = true;
  }
  dim3 grid((N + WF_BN - 1) / WF_BN, 1);
  hgemm_wmma_wide<<<grid, 256, WF_SMEM_SIZE, get_stream()>>>(
      reinterpret_cast<const __half*>(pA),
      reinterpret_cast<const __half*>(pB),
      reinterpret_cast<__half*>(pC),
      M, N, K);
  cudaGetLastError();
}