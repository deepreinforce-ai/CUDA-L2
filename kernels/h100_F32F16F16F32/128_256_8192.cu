#include <iostream>
#include <stdexcept>
#include <string>

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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>
#include <torch/types.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementD           = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

struct GemmState {
    cutlass::device_memory::allocation<uint8_t> ws;
    size_t ws_sz = 0;
    bool inited = false;
};

using TS0 = cute::Shape<cute::_128, cute::_256, cute::_64>;
using CS0 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CE0 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS0, CS0,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
using CM0 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAccumulator, TS0, CS0,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE0::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using GK0 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM0, CE0, cutlass::gemm::PersistentScheduler>;
using Gemm0 = cutlass::gemm::device::GemmUniversalAdapter<GK0>;

using TS1 = cute::Shape<cute::_128, cute::_128, cute::_64>;
using CS1 = cute::Shape<cute::_1, cute::_2, cute::_1>;
using CE1 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS1, CS1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
using CM1 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAccumulator, TS1, CS1,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE1::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using GK1 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM1, CE1, cutlass::gemm::PersistentScheduler>;
using Gemm1 = cutlass::gemm::device::GemmUniversalAdapter<GK1>;

using CE2 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS0, CS0,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CM2 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAccumulator, TS0, CS0,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE2::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GK2 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM2, CE2, cutlass::gemm::PersistentScheduler>;
using Gemm2 = cutlass::gemm::device::GemmUniversalAdapter<GK2>;

using CE3 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS1, CS1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
using CM3 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAccumulator, TS1, CS1,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE3::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GK3 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM3, CE3, cutlass::gemm::PersistentScheduler>;
using Gemm3 = cutlass::gemm::device::GemmUniversalAdapter<GK3>;

using TS4 = cute::Shape<cute::_128, cute::_256, cute::_128>;
using CS4 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CE4 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS4, CS4,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
using CM4 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAccumulator, TS4, CS4,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE4::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using GK4 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM4, CE4, cutlass::gemm::PersistentScheduler>;
using Gemm4 = cutlass::gemm::device::GemmUniversalAdapter<GK4>;

using TS5 = cute::Shape<cute::_64, cute::_256, cute::_64>;
using CS5 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CE5 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS5, CS5,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
using CM5 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAccumulator, TS5, CS5,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE5::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using GK5 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM5, CE5, cutlass::gemm::PersistentScheduler>;
using Gemm5 = cutlass::gemm::device::GemmUniversalAdapter<GK5>;

using TS6 = cute::Shape<cute::_64, cute::_128, cute::_64>;
using CS6 = cute::Shape<cute::_1, cute::_2, cute::_1>;
using CE6 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS6, CS6,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
using CM6 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
    ElementAccumulator, TS6, CS6,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE6::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using GK6 = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>, CM6, CE6, cutlass::gemm::PersistentScheduler>;
using Gemm6 = cutlass::gemm::device::GemmUniversalAdapter<GK6>;

static constexpr int NUM_CFGS = 7;

static Gemm0 g_gemm0; static GemmState g_st0;
static Gemm1 g_gemm1; static GemmState g_st1;
static Gemm2 g_gemm2; static GemmState g_st2;
static Gemm3 g_gemm3; static GemmState g_st3;
static Gemm4 g_gemm4; static GemmState g_st4;
static Gemm5 g_gemm5; static GemmState g_st5;
static Gemm6 g_gemm6; static GemmState g_st6;

static int g_best_cfg = -1;

template<typename GemmT>
static typename GemmT::Arguments make_args(
    void* pA, void* pB, void* pC, void* pD,
    int M, int N, int K, cutlass::KernelHardwareInfo hw_info)
{
    using SA = typename GemmT::GemmKernel::StrideA;
    using SB = typename GemmT::GemmKernel::StrideB;
    using SC = typename GemmT::GemmKernel::StrideC;
    using SD = typename GemmT::GemmKernel::StrideD;
    SA sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    SD sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));
    return typename GemmT::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<ElementA*>(pA), sA, reinterpret_cast<ElementB*>(pB), sB},
        {{ElementCompute(1.f), ElementCompute(0.f)},
         reinterpret_cast<ElementC*>(pC), sC,
         reinterpret_cast<ElementD*>(pD), sD},
        hw_info
    };
}

template<typename GemmT>
static float bench_cfg(GemmT& gemm, GemmState& state,
    void* pA, void* pB, void* pC, void* pD,
    int M, int N, int K, cutlass::KernelHardwareInfo hw_info)
{
    try {
        auto args = make_args<GemmT>(pA, pB, pC, pD, M, N, K, hw_info);
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1e9f;
        size_t needed = GemmT::get_workspace_size(args);
        if (needed > state.ws_sz) {
            state.ws = cutlass::device_memory::allocation<uint8_t>(needed + 4096);
            state.ws_sz = needed + 4096;
        }
        if (gemm.initialize(args, state.ws.get()) != cutlass::Status::kSuccess) return 1e9f;
        state.inited = true;
        for (int i = 0; i < 5; i++) gemm.run();
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < 30; i++) gemm.run();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms = 0.f; cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return ms / 30.f;
    } catch (...) { return 1e9f; }
}

template<typename GemmT>
static void run_cfg(GemmT& gemm, GemmState& state,
    void* pA, void* pB, void* pC, void* pD,
    int M, int N, int K, cutlass::KernelHardwareInfo hw_info)
{
    auto args = make_args<GemmT>(pA, pB, pC, pD, M, N, K, hw_info);
    if (!state.inited) {
        if (gemm.can_implement(args) != cutlass::Status::kSuccess)
            throw std::runtime_error("CUTLASS cannot implement");
        size_t needed = GemmT::get_workspace_size(args);
        if (needed > state.ws_sz) {
            state.ws = cutlass::device_memory::allocation<uint8_t>(needed + 4096);
            state.ws_sz = needed + 4096;
        }
        if (gemm.initialize(args, state.ws.get()) != cutlass::Status::kSuccess)
            throw std::runtime_error("CUTLASS init failed");
        state.inited = true;
    } else {
        auto st = gemm.update(args, state.ws.get());
        if (st != cutlass::Status::kSuccess) {
            size_t needed = GemmT::get_workspace_size(args);
            if (needed > state.ws_sz) {
                state.ws = cutlass::device_memory::allocation<uint8_t>(needed + 4096);
                state.ws_sz = needed + 4096;
            }
            if (gemm.initialize(args, state.ws.get()) != cutlass::Status::kSuccess)
                throw std::runtime_error("CUTLASS re-init failed");
        }
    }
    if (gemm.run() != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS run failed");
}

#endif

using namespace nvcuda;

static constexpr int FB_TILE_M = 64;
static constexpr int FB_TILE_N = 64;
static constexpr int FB_TILE_K = 64;
static constexpr int FB_WMMA_M = 16;
static constexpr int FB_WMMA_N = 16;
static constexpr int FB_WMMA_K = 16;
static constexpr int FB_WARPS_M = 4;
static constexpr int FB_WARPS_N = 4;
static constexpr int FB_THREADS = FB_WARPS_M * FB_WARPS_N * 32;
static constexpr int FB_SA_STRIDE = FB_TILE_K + 8;
static constexpr int FB_SB_STRIDE = FB_TILE_N + 8;

__global__ __launch_bounds__(FB_THREADS, 2)
void fb_splitk_gemm(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ acc,
    int M, int N, int K, int k_splits)
{
    const int tile_n  = blockIdx.x;
    const int tile_m  = blockIdx.y;
    const int k_split = blockIdx.z;

    const int m_start = tile_m * FB_TILE_M;
    const int n_start = tile_n * FB_TILE_N;
    const int k_per   = K / k_splits;
    const int k_start = k_split * k_per;
    const int niters  = k_per / FB_TILE_K;

    const int warp_id = threadIdx.x / 32;
    const int warp_m  = warp_id / FB_WARPS_N;
    const int warp_n  = warp_id % FB_WARPS_N;
    const int tx      = threadIdx.x;

    wmma::fragment<wmma::accumulator, FB_WMMA_M, FB_WMMA_N, FB_WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    __shared__ __half smem_a[2][FB_TILE_M][FB_SA_STRIDE];
    __shared__ __half smem_b[2][FB_TILE_K][FB_SB_STRIDE];

    const int a_f4 = (FB_TILE_M * FB_TILE_K) / 8;
    const int b_f4 = (FB_TILE_K * FB_TILE_N) / 8;

    for (int idx = tx; idx < a_f4; idx += FB_THREADS) {
        int lin = idx*8, r = lin/FB_TILE_K, c = lin%FB_TILE_K;
        *reinterpret_cast<float4*>(&smem_a[0][r][c]) =
            *reinterpret_cast<const float4*>(&A[(m_start+r)*K + k_start+c]);
    }
    for (int idx = tx; idx < b_f4; idx += FB_THREADS) {
        int lin = idx*8, r = lin/FB_TILE_N, c = lin%FB_TILE_N;
        *reinterpret_cast<float4*>(&smem_b[0][r][c]) =
            *reinterpret_cast<const float4*>(&B[(k_start+r)*N + n_start+c]);
    }
    __syncthreads();

    for (int iter = 0; iter < niters; iter++) {
        int cur = iter & 1, nxt = 1 - cur;
        int k_nxt = k_start + (iter+1)*FB_TILE_K;

        if (iter+1 < niters) {
            for (int idx = tx; idx < a_f4; idx += FB_THREADS) {
                int lin = idx*8, r = lin/FB_TILE_K, c = lin%FB_TILE_K;
                __pipeline_memcpy_async(&smem_a[nxt][r][c], &A[(m_start+r)*K+k_nxt+c], 16);
            }
            for (int idx = tx; idx < b_f4; idx += FB_THREADS) {
                int lin = idx*8, r = lin/FB_TILE_N, c = lin%FB_TILE_N;
                __pipeline_memcpy_async(&smem_b[nxt][r][c], &B[(k_nxt+r)*N+n_start+c], 16);
            }
            __pipeline_commit();
        }

        #pragma unroll
        for (int ki = 0; ki < FB_TILE_K; ki += FB_WMMA_K) {
            wmma::fragment<wmma::matrix_a, FB_WMMA_M, FB_WMMA_N, FB_WMMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, FB_WMMA_M, FB_WMMA_N, FB_WMMA_K, __half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, &smem_a[cur][warp_m*FB_WMMA_M][ki], FB_SA_STRIDE);
            wmma::load_matrix_sync(b_frag, &smem_b[cur][ki][warp_n*FB_WMMA_N], FB_SB_STRIDE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        if (iter+1 < niters) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

    int out_m = m_start + warp_m*FB_WMMA_M;
    int out_n = n_start + warp_n*FB_WMMA_N;
    if (out_m < M && out_n < N)
        wmma::store_matrix_sync(acc + (size_t)k_split*M*N + out_m*N + out_n, c_frag, N, wmma::mem_row_major);
}

template<int SPLITS>
__global__ __launch_bounds__(256)
void reduce_fixed4(const float* __restrict__ acc, __half* __restrict__ c, int MN)
{
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < MN) {
        float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
        #pragma unroll
        for (int k = 0; k < SPLITS; k++) {
            const float* p = acc + (size_t)k*MN + base;
            float4 v = *reinterpret_cast<const float4*>(p);
            s0+=v.x; s1+=v.y; s2+=v.z; s3+=v.w;
        }
        *reinterpret_cast<__half2*>(&c[base])   = __floats2half2_rn(s0, s1);
        *reinterpret_cast<__half2*>(&c[base+2]) = __floats2half2_rn(s2, s3);
    } else {
        for (int i = 0; i < 4 && base+i < MN; i++) {
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < SPLITS; k++) s += acc[(size_t)k*MN+base+i];
            c[base+i] = __float2half(s);
        }
    }
}

static void launch_reduce(int splits, const float* acc, __half* c, int MN) {
    int rb = (MN/4 + 255) / 256;
    switch (splits) {
        case  8:  reduce_fixed4< 8><<<rb,256>>>(acc,c,MN); break;
        case 16:  reduce_fixed4<16><<<rb,256>>>(acc,c,MN); break;
        case 32:  reduce_fixed4<32><<<rb,256>>>(acc,c,MN); break;
        case 64:  reduce_fixed4<64><<<rb,256>>>(acc,c,MN); break;
        case 128: reduce_fixed4<128><<<rb,256>>>(acc,c,MN); break;
        default:  reduce_fixed4<16><<<rb,256>>>(acc,c,MN); break;
    }
}

static float* g_acc    = nullptr;
static size_t g_acc_sz = 0;
static int g_fb_splits = -1;

static void ensure_ws(size_t needed) {
    if (needed > g_acc_sz) {
        if (g_acc) cudaFree(g_acc);
        cudaMalloc(&g_acc, needed);
        g_acc_sz = needed;
    }
}

static void fb_launch(int splits, const __half* pA, const __half* pB,
    float* acc, __half* pC, int M, int N, int K)
{
    int ntm = M/FB_TILE_M, ntn = N/FB_TILE_N;
    dim3 grid(ntn, ntm, splits);
    dim3 block(FB_THREADS);
    fb_splitk_gemm<<<grid,block>>>(pA,pB,acc,M,N,K,splits);
    launch_reduce(splits, acc, pC, M*N);
}

static float fb_bench(int splits, const __half* pA, const __half* pB,
    float* acc, __half* pC, int M, int N, int K)
{
    if (K % (splits * FB_TILE_K) != 0) return 1e9f;
    for (int i = 0; i < 3; i++) fb_launch(splits,pA,pB,acc,pC,M,N,K);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 30; i++) fb_launch(splits,pA,pB,acc,pC,M,N,K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0.f; cudaEventElapsedTime(&ms,t0,t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms/30.f;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* pA = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* pB = reinterpret_cast<const __half*>(b.data_ptr());
    __half* pC       = reinterpret_cast<__half*>(c.data_ptr());

    ensure_ws((size_t)128 * M * N * sizeof(float));

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    int device_id = 0;
    cudaGetDevice(&device_id);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count  = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    void* pBcol = b_col_major.data_ptr();

    if (g_best_cfg < 0) {
        float times[NUM_CFGS];
        times[0] = bench_cfg(g_gemm0, g_st0, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info);
        times[1] = bench_cfg(g_gemm1, g_st1, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info);
        times[2] = bench_cfg(g_gemm2, g_st2, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info);
        times[3] = bench_cfg(g_gemm3, g_st3, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info);
        times[4] = bench_cfg(g_gemm4, g_st4, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info);
        times[5] = bench_cfg(g_gemm5, g_st5, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info);
        times[6] = bench_cfg(g_gemm6, g_st6, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info);

        float best_wmma = 1e9f;
        int best_fb_splits = 16;
        int split_list[] = {8, 16, 32, 64};
        for (int s : split_list) {
            float t = fb_bench(s, pA, pB, g_acc, pC, M, N, K);
            if (t < best_wmma) { best_wmma = t; best_fb_splits = s; }
        }
        g_fb_splits = best_fb_splits;

        int best = NUM_CFGS;
        float best_t = best_wmma;
        for (int i = 0; i < NUM_CFGS; i++) {
            if (times[i] < best_t) { best_t = times[i]; best = i; }
        }
        g_best_cfg = best;

        g_st0.inited = false; g_st1.inited = false;
        g_st2.inited = false; g_st3.inited = false;
        g_st4.inited = false; g_st5.inited = false;
        g_st6.inited = false;
    }

    if (g_best_cfg < NUM_CFGS) {
        switch (g_best_cfg) {
            case 0: run_cfg(g_gemm0, g_st0, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
            case 1: run_cfg(g_gemm1, g_st1, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
            case 2: run_cfg(g_gemm2, g_st2, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
            case 3: run_cfg(g_gemm3, g_st3, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
            case 4: run_cfg(g_gemm4, g_st4, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
            case 5: run_cfg(g_gemm5, g_st5, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
            case 6: run_cfg(g_gemm6, g_st6, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
            default: run_cfg(g_gemm0, g_st0, (void*)pA, pBcol, (void*)pC, (void*)pC, M, N, K, hw_info); break;
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
        return;
    }

#else
    if (g_fb_splits < 0) {
        float best_t = 1e9f;
        int best_s = 16;
        int split_list[] = {8, 16, 32, 64};
        for (int s : split_list) {
            float t = fb_bench(s, pA, pB, g_acc, pC, M, N, K);
            if (t < best_t) { best_t = t; best_s = s; }
        }
        g_fb_splits = best_s;
        g_best_cfg = NUM_CFGS;
    }
#endif

    fb_launch(g_fb_splits, pA, pB, g_acc, pC, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}