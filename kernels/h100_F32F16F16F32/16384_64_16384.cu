#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <float.h>
#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
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

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using LayoutA_t     = cutlass::layout::RowMajor;
using LayoutB_t     = cutlass::layout::ColumnMajor;
using LayoutC_t     = cutlass::layout::RowMajor;
using LayoutD_t     = cutlass::layout::RowMajor;
using ElementA_t    = cutlass::half_t;
using ElementB_t    = cutlass::half_t;
using ElementC_t    = cutlass::half_t;
using ElementD_t    = cutlass::half_t;
using ElementAcc_t  = float;
using ElementComp_t = float;
using EpilogueOp_t  = cutlass::epilogue::fusion::LinearCombination<
    ElementD_t, ElementComp_t, ElementC_t, ElementComp_t,
    cutlass::FloatRoundStyle::round_to_nearest>;

static constexpr int AlignABCD = 8;
static constexpr int H100_SM_COUNT = 132;

#define MAKE_GEMM(NAME, TM, TN, TK, CM, CN, CK, MS, ES, TS)                      \
struct NAME {                                                                      \
  using TileShape    = cute::Shape<cute::Int<TM>,cute::Int<TN>,cute::Int<TK>>;    \
  using GroupShape   = cute::Shape<cute::Int<CM>,cute::Int<CN>,cute::Int<CK>>;    \
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
      TileShape, GroupShape,                                                        \
      cutlass::epilogue::collective::EpilogueTileAuto,                             \
      ElementAcc_t, ElementComp_t,                                                 \
      ElementC_t, LayoutC_t, AlignABCD,                                            \
      ElementD_t, LayoutD_t, AlignABCD,                                            \
      ES, EpilogueOp_t>::CollectiveOp;                                             \
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder< \
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,                         \
      ElementA_t, LayoutA_t, AlignABCD,                                            \
      ElementB_t, LayoutB_t, AlignABCD,                                            \
      ElementAcc_t, TileShape, GroupShape,                                          \
      cutlass::gemm::collective::StageCountAutoCarveout<                            \
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,     \
      MS>::CollectiveOp;                                                            \
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                         \
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue, TS>;       \
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;         \
  using StrideA = typename Gemm::GemmKernel::StrideA;                              \
  using StrideB = typename Gemm::GemmKernel::StrideB;                              \
  using StrideC = typename Gemm::GemmKernel::StrideC;                              \
  using StrideD = typename Gemm::GemmKernel::StrideD;                              \
};

MAKE_GEMM(V0_PP_128x64x128_C1,
    128, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

MAKE_GEMM(V1_PP_128x64x256_C1,
    128, 64, 256, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

MAKE_GEMM(V2_Coop_128x64x128_C1_SK,
    128, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

MAKE_GEMM(V3_PP_128x64x128_C2,
    128, 64, 128, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

MAKE_GEMM(V4_Coop_128x64x256_C1_SK,
    128, 64, 256, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::StreamKScheduler)

MAKE_GEMM(V5_PP_256x64x128_C1,
    256, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

MAKE_GEMM(V6_Coop_128x64x128_C1_Pers,
    128, 64, 128, 1, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::PersistentScheduler)

MAKE_GEMM(V7_PP_128x64x256_C2,
    128, 64, 256, 2, 1, 1,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::gemm::PersistentScheduler)

static constexpr int NUM_VARIANTS = 8;

static constexpr size_t WS_BYTES = 128ULL * 1024 * 1024;
static uint8_t* g_workspace  = nullptr;
static int      g_best_var   = 0;
static bool     g_initialized = false;
static int      g_device_id  = 0;

static bool g_args_cached = false;
static int  g_cached_M = 0, g_cached_N = 0, g_cached_K = 0;

static void ensure_workspace() {
    if (!g_workspace) {
        cudaGetDevice(&g_device_id);
        cudaMalloc(&g_workspace, WS_BYTES);
        g_initialized = true;
    }
}

template <typename HT>
static cutlass::Status run_gemm(
    const void* pA, const void* pB, void* pC,
    int M, int N, int K)
{
    using Gemm    = typename HT::Gemm;
    using StrideA = typename HT::StrideA;
    using StrideB = typename HT::StrideB;
    using StrideC = typename HT::StrideC;
    using StrideD = typename HT::StrideD;

    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = g_device_id;
    hw_info.sm_count  = H100_SM_COUNT;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const ElementA_t*>(pA), sA,
         reinterpret_cast<const ElementB_t*>(pB), sB},
        {{1.0f, 0.0f},
         reinterpret_cast<ElementC_t*>(pC), sC,
         reinterpret_cast<ElementD_t*>(pC), sD},
        hw_info
    };

    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;

    size_t needed = Gemm::get_workspace_size(args);
    if (needed > WS_BYTES) return cutlass::Status::kErrorNotSupported;

    status = gemm.initialize(args, g_workspace);
    if (status != cutlass::Status::kSuccess) return status;

    return gemm.run();
}

static cutlass::Status dispatch(int idx, const void* A, const void* B, void* C, int M, int N, int K) {
    switch (idx) {
        case 0: return run_gemm<V0_PP_128x64x128_C1>      (A,B,C,M,N,K);
        case 1: return run_gemm<V1_PP_128x64x256_C1>      (A,B,C,M,N,K);
        case 2: return run_gemm<V2_Coop_128x64x128_C1_SK> (A,B,C,M,N,K);
        case 3: return run_gemm<V3_PP_128x64x128_C2>      (A,B,C,M,N,K);
        case 4: return run_gemm<V4_Coop_128x64x256_C1_SK> (A,B,C,M,N,K);
        case 5: return run_gemm<V5_PP_256x64x128_C1>      (A,B,C,M,N,K);
        case 6: return run_gemm<V6_Coop_128x64x128_C1_Pers>(A,B,C,M,N,K);
        case 7: return run_gemm<V7_PP_128x64x256_C2>      (A,B,C,M,N,K);
        default: return cutlass::Status::kErrorNotSupported;
    }
}

static float time_variant(int idx, const void* A, const void* B, void* C, int M, int N, int K) {
    for (int w = 0; w < 2; w++) {
        auto s = dispatch(idx, A, B, C, M, N, K);
        if (s != cutlass::Status::kSuccess) return FLT_MAX;
    }
    cudaDeviceSynchronize();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    const int REPS = 8;
    cudaEventRecord(ev0);
    for (int r = 0; r < REPS; r++) {
        auto s = dispatch(idx, A, B, C, M, N, K);
        if (s != cutlass::Status::kSuccess) {
            cudaEventDestroy(ev0); cudaEventDestroy(ev1);
            return FLT_MAX;
        }
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return ms / REPS;
}

#endif

static constexpr int W_BM        = 256;
static constexpr int W_BN        = 64;
static constexpr int W_BK        = 64;
static constexpr int W_WARPS     = 8;
static constexpr int W_THREADS   = W_WARPS * 32;
static constexpr int W_WMMA_M    = 16;
static constexpr int W_WMMA_N    = 16;
static constexpr int W_WMMA_K    = 16;
static constexpr int W_TILE_M    = 2;
static constexpr int W_TILE_N    = 4;
static constexpr int W_STAGES    = 4;
static constexpr int W_SA_STRIDE = W_BK + 16;
static constexpr int W_SB_STRIDE = W_BN + 8;
static constexpr int W_SA_SIZE   = W_BM * W_SA_STRIDE;
static constexpr int W_SB_SIZE   = W_BK * W_SB_STRIDE;
static constexpr int W_SMEM_SIZE = W_STAGES * (W_SA_SIZE + W_SB_SIZE) * (int)sizeof(half);

__device__ __forceinline__
void cp_async16(half* dst, const half* src, bool valid) {
    unsigned dst_addr = __cvta_generic_to_shared(dst);
    if (valid) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst_addr), "l"(src));
    } else {
        *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

__global__ __launch_bounds__(W_THREADS, 1)
void hgemm_wmma_fallback_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int M, int N, int K)
{
    const int block_m      = blockIdx.x;
    const int m_start      = block_m * W_BM;
    const int tid          = threadIdx.x;
    const int warp_id      = tid / 32;
    const int warp_m_start = warp_id * (W_TILE_M * W_WMMA_M);

    extern __shared__ half smem[];

    auto sA = [&](int s) -> half* { return smem + s * (W_SA_SIZE + W_SB_SIZE); };
    auto sB = [&](int s) -> half* { return smem + s * (W_SA_SIZE + W_SB_SIZE) + W_SA_SIZE; };

    wmma::fragment<wmma::accumulator, W_WMMA_M, W_WMMA_N, W_WMMA_K, float>
        acc[W_TILE_M][W_TILE_N];
    #pragma unroll
    for (int mi = 0; mi < W_TILE_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < W_TILE_N; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    const int K_tiles = (K + W_BK - 1) / W_BK;
    static constexpr int A_LOADS = 4;
    static constexpr int B_LOADS = 1;

    auto issue_loads = [&](int k_tile, int stage) {
        if (k_tile >= K_tiles) return;
        const int k_off = k_tile * W_BK;
        half* smA_s = sA(stage);
        half* smB_s = sB(stage);
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {
            int chunk = tid + i * W_THREADS;
            int row   = chunk / (W_BK / 8);
            int cf4   = chunk % (W_BK / 8);
            int gm    = m_start + row;
            int gk    = k_off + cf4 * 8;
            cp_async16(smA_s + row * W_SA_STRIDE + cf4 * 8,
                       A + gm * K + gk, (gm < M) && (gk + 7 < K));
        }
        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            int chunk = tid + i * W_THREADS;
            int row   = chunk / (W_BN / 8);
            int cf4   = chunk % (W_BN / 8);
            int gk    = k_off + row;
            int gn    = cf4 * 8;
            cp_async16(smB_s + row * W_SB_STRIDE + cf4 * 8,
                       B + gk * N + gn, (gk < K) && (gn + 7 < N));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const int prefetch = (K_tiles < W_STAGES) ? K_tiles : W_STAGES;
    for (int s = 0; s < prefetch; s++) issue_loads(s, s);

    wmma::fragment<wmma::matrix_a, W_WMMA_M, W_WMMA_N, W_WMMA_K, half, wmma::row_major> a_frag[W_TILE_M];
    wmma::fragment<wmma::matrix_b, W_WMMA_M, W_WMMA_N, W_WMMA_K, half, wmma::row_major> b_frag[W_TILE_N];
    const int K_WMMA = W_BK / W_WMMA_K;

    for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
        int cur_stage = k_tile % W_STAGES;
        asm volatile("cp.async.wait_group %0;\n" :: "n"(W_STAGES - 1));
        __syncthreads();
        issue_loads(k_tile + W_STAGES, (k_tile + W_STAGES) % W_STAGES);

        half* smA_c = sA(cur_stage);
        half* smB_c = sB(cur_stage);
        #pragma unroll
        for (int kk = 0; kk < K_WMMA; kk++) {
            #pragma unroll
            for (int ni = 0; ni < W_TILE_N; ni++)
                wmma::load_matrix_sync(b_frag[ni],
                    smB_c + kk * W_WMMA_K * W_SB_STRIDE + ni * W_WMMA_N, W_SB_STRIDE);
            #pragma unroll
            for (int mi = 0; mi < W_TILE_M; mi++) {
                wmma::load_matrix_sync(a_frag[mi],
                    smA_c + (warp_m_start + mi * W_WMMA_M) * W_SA_STRIDE + kk * W_WMMA_K, W_SA_STRIDE);
                #pragma unroll
                for (int ni = 0; ni < W_TILE_N; ni++)
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < W_TILE_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < W_TILE_N; ni++) {
            int c_m = m_start + warp_m_start + mi * W_WMMA_M;
            int c_n = ni * W_WMMA_N;
            if (c_m < M && c_n < N) {
                wmma::fragment<wmma::accumulator, W_WMMA_M, W_WMMA_N, W_WMMA_K, half> out_f;
                #pragma unroll
                for (int x = 0; x < out_f.num_elements; x++)
                    out_f.x[x] = __float2half(acc[mi][ni].x[x]);
                wmma::store_matrix_sync(C + c_m * N + c_n, out_f, N, wmma::mem_row_major);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const void* pA = a.data_ptr();
    const void* pB = b_col_major.data_ptr();
    void*       pC = c.data_ptr();

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    ensure_workspace();

    if (g_args_cached && g_cached_M == M && g_cached_N == N && g_cached_K == K) {
        cutlass::Status status = dispatch(g_best_var, pA, pB, pC, M, N, K);
        if (status == cutlass::Status::kSuccess) {
            return;
        }
    }

    {
        cutlass::Status s0 = dispatch(0, pA, pB, pC, M, N, K);
        if (s0 == cutlass::Status::kSuccess) {
            for (int w = 0; w < 2; w++) dispatch(0, pA, pB, pC, M, N, K);
            cudaDeviceSynchronize();

            cudaEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);

            cudaEventRecord(ev0);
            for (int r = 0; r < 5; r++) dispatch(0, pA, pB, pC, M, N, K);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms0 = 0.f;
            cudaEventElapsedTime(&ms0, ev0, ev1);
            ms0 /= 5.f;
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);

            float best_ms = ms0;
            int   best_idx = 0;

            const int candidates[] = {1, 2, 3, 4, 5, 6, 7};
            for (int ci = 0; ci < 7; ci++) {
                int v = candidates[ci];
                float ms = time_variant(v, pA, pB, pC, M, N, K);
                if (ms < best_ms) {
                    best_ms  = ms;
                    best_idx = v;
                }
            }

            g_best_var   = best_idx;
            g_cached_M   = M;
            g_cached_N   = N;
            g_cached_K   = K;
            g_args_cached = true;

            dispatch(g_best_var, pA, pB, pC, M, N, K);
            return;
        }
    }

    for (int v = 0; v < NUM_VARIANTS; v++) {
        cutlass::Status s = dispatch(v, pA, pB, pC, M, N, K);
        if (s == cutlass::Status::kSuccess) {
            g_best_var    = v;
            g_cached_M    = M; g_cached_N = N; g_cached_K = K;
            g_args_cached = true;
            return;
        }
    }

#endif

    const half* pA_raw = reinterpret_cast<const half*>(a.data_ptr());
    const half* pB_raw = reinterpret_cast<const half*>(b.data_ptr());
    half*       pC_raw = reinterpret_cast<half*>(c.data_ptr());

    cudaFuncSetAttribute(hgemm_wmma_fallback_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, W_SMEM_SIZE);

    dim3 grid((M + W_BM - 1) / W_BM);
    dim3 block(W_THREADS);
    hgemm_wmma_fallback_kernel<<<grid, block, W_SMEM_SIZE>>>(pA_raw, pB_raw, pC_raw, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}