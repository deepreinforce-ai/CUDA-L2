#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <cute/tensor.hpp>
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
#include <iostream>
#include <stdexcept>

__device__ __forceinline__ uint32_t pack_halves(half lo, half hi) {
    uint32_t r;
    asm("mov.b32 %0, {%1,%2};\n" : "=r"(r) : "h"(__half_as_ushort(lo)), "h"(__half_as_ushort(hi)));
    return r;
}

__device__ __forceinline__ void mma_m16n8k16(
    float* acc,
    uint32_t* a_frag,
    uint32_t* b_frag
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
        : "r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
          "r"(b_frag[0]),"r"(b_frag[1])
    );
}

__global__ __launch_bounds__(128, 2)
void hgemm_kernel_v1(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int STAGES = 4;
    constexpr int AS = BK + 8;
    constexpr int BS = BK + 8;

    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;

    int m_base = bx * BM;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + STAGES * BM * AS;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

    int num_tiles = K / BK;

    int a_r0 = lid / 4;
    int a_r8 = a_r0 + 8;
    int a_c0 = (lid % 4) * 2;
    int a_c8 = a_c0 + 8;

    int b_n0 = (lid % 4) * 2;
    int b_n1 = b_n0 + 1;

    int wr = 0, rd = 0;
    {
        int pre = (STAGES - 1 < num_tiles) ? STAGES - 1 : num_tiles;
        for (int s = 0; s < pre; s++) {
            {
                int base = tid * 32;
                half* dst = sA + s * BM * AS;
                int k0 = s * BK;
                #pragma unroll
                for (int e = 0; e < 32; e += 8) {
                    int idx = base + e;
                    int row = idx / BK, col = idx % BK;
                    int gr = m_base + row, gk = k0 + col;
                    half* d = dst + row * AS + col;
                    if (gr < M) {
                        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                            ::"r"((uint32_t)__cvta_generic_to_shared(d)),
                              "l"((uint64_t)(A + gr * K + gk)));
                    }
                }
            }
            {
                int base = tid * 32;
                half* dst = sB + s * BN * BS;
                int k0 = s * BK;
                #pragma unroll
                for (int e = 0; e < 32; e += 8) {
                    int idx = base + e;
                    int n = idx / BK, k = idx % BK;
                    int gk = k0 + k;
                    half* d = dst + n * BS + k;
                    if (n < N && gk < K) {
                        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                            ::"r"((uint32_t)__cvta_generic_to_shared(d)),
                              "l"((uint64_t)(B_col + (uint64_t)n * K + gk)));
                    }
                }
            }
            asm volatile("cp.async.commit_group;\n"::);
        }
        wr = pre % STAGES;
    }

    for (int t = 0; t < num_tiles; t++) {
        asm volatile("cp.async.wait_group 2;\n"::);
        __syncthreads();

        int nt = t + (STAGES - 1);
        if (nt < num_tiles) {
            {
                int base = tid * 32;
                half* dst = sA + wr * BM * AS;
                int k0 = nt * BK;
                #pragma unroll
                for (int e = 0; e < 32; e += 8) {
                    int idx = base + e;
                    int row = idx / BK, col = idx % BK;
                    int gr = m_base + row, gk = k0 + col;
                    half* d = dst + row * AS + col;
                    if (gr < M) {
                        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                            ::"r"((uint32_t)__cvta_generic_to_shared(d)),
                              "l"((uint64_t)(A + gr * K + gk)));
                    }
                }
            }
            {
                int base = tid * 32;
                half* dst = sB + wr * BN * BS;
                int k0 = nt * BK;
                #pragma unroll
                for (int e = 0; e < 32; e += 8) {
                    int idx = base + e;
                    int n = idx / BK, k = idx % BK;
                    int gk = k0 + k;
                    half* d = dst + n * BS + k;
                    if (n < N && gk < K) {
                        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                            ::"r"((uint32_t)__cvta_generic_to_shared(d)),
                              "l"((uint64_t)(B_col + (uint64_t)n * K + gk)));
                    }
                }
            }
            asm volatile("cp.async.commit_group;\n"::);
            wr = (wr + 1) % STAGES;
        }

        half* cA = sA + rd * BM * AS + wid * 16 * AS;
        half* cB = sB + rd * BN * BS;

        #pragma unroll
        for (int ki = 0; ki < BK; ki += 16) {
            uint32_t af[4];
            af[0] = pack_halves(cA[a_r0*AS + ki+a_c0], cA[a_r0*AS + ki+a_c0+1]);
            af[1] = pack_halves(cA[a_r0*AS + ki+a_c8], cA[a_r0*AS + ki+a_c8+1]);
            af[2] = pack_halves(cA[a_r8*AS + ki+a_c0], cA[a_r8*AS + ki+a_c0+1]);
            af[3] = pack_halves(cA[a_r8*AS + ki+a_c8], cA[a_r8*AS + ki+a_c8+1]);

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n_off = ni * 8;
                uint32_t bf[2];
                bf[0] = pack_halves(cB[(n_off+b_n0)*BS + ki+0],
                                    cB[(n_off+b_n1)*BS + ki+0]);
                bf[1] = pack_halves(cB[(n_off+b_n0)*BS + ki+8],
                                    cB[(n_off+b_n1)*BS + ki+8]);
                mma_m16n8k16(acc[ni], af, bf);
            }
        }

        rd = (rd + 1) % STAGES;
        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n"::);

    int row0 = m_base + wid * 16 + lid / 4;
    int row8 = row0 + 8;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        int col0 = ni * 8 + (lid % 4) * 2;
        int col1 = col0 + 1;
        if (row0 < M && col0 < N) C[row0*N+col0] = __float2half(acc[ni][0]);
        if (row0 < M && col1 < N) C[row0*N+col1] = __float2half(acc[ni][1]);
        if (row8 < M && col0 < N) C[row8*N+col0] = __float2half(acc[ni][2]);
        if (row8 < M && col1 < N) C[row8*N+col1] = __float2half(acc[ni][3]);
    }
}

__global__ __launch_bounds__(128, 2)
void hgemm_kernel_v2(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 128;
    constexpr int STAGES = 3;
    constexpr int AS = BK + 8;
    constexpr int BS = BK + 8;

    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;

    int m_base = bx * BM;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + STAGES * BM * AS;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

    int num_tiles = K / BK;

    int a_r0 = lid / 4;
    int a_r8 = a_r0 + 8;
    int a_c0 = (lid % 4) * 2;
    int a_c8 = a_c0 + 8;
    int b_n0 = (lid % 4) * 2;
    int b_n1 = b_n0 + 1;

    auto load_A = [&](int s, int t) {
        half* dst = sA + s * BM * AS;
        int k0 = t * BK;
        int base = tid * 64;
        #pragma unroll
        for (int e = 0; e < 64; e += 8) {
            int idx = base + e;
            int row = idx / BK, col = idx % BK;
            int gr = m_base + row, gk = k0 + col;
            half* d = dst + row * AS + col;
            if (gr < M)
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                    ::"r"((uint32_t)__cvta_generic_to_shared(d)),
                      "l"((uint64_t)(A + gr * K + gk)));
        }
    };
    auto load_B = [&](int s, int t) {
        half* dst = sB + s * BN * BS;
        int k0 = t * BK;
        int base = tid * 64;
        #pragma unroll
        for (int e = 0; e < 64; e += 8) {
            int idx = base + e;
            int n = idx / BK, k = idx % BK;
            int gk = k0 + k;
            half* d = dst + n * BS + k;
            if (n < N && gk < K)
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                    ::"r"((uint32_t)__cvta_generic_to_shared(d)),
                      "l"((uint64_t)(B_col + (uint64_t)n * K + gk)));
        }
    };

    int pre = (STAGES-1 < num_tiles) ? STAGES-1 : num_tiles;
    for (int s = 0; s < pre; s++) {
        load_A(s, s);
        load_B(s, s);
        asm volatile("cp.async.commit_group;\n"::);
    }
    int wr = pre % STAGES, rd = 0;

    for (int t = 0; t < num_tiles; t++) {
        asm volatile("cp.async.wait_group 1;\n"::);
        __syncthreads();

        int nt = t + (STAGES - 1);
        if (nt < num_tiles) {
            load_A(wr, nt);
            load_B(wr, nt);
            asm volatile("cp.async.commit_group;\n"::);
            wr = (wr + 1) % STAGES;
        }

        half* cA = sA + rd * BM * AS + wid * 16 * AS;
        half* cB = sB + rd * BN * BS;

        #pragma unroll
        for (int ki = 0; ki < BK; ki += 16) {
            uint32_t af[4];
            af[0] = pack_halves(cA[a_r0*AS+ki+a_c0], cA[a_r0*AS+ki+a_c0+1]);
            af[1] = pack_halves(cA[a_r0*AS+ki+a_c8], cA[a_r0*AS+ki+a_c8+1]);
            af[2] = pack_halves(cA[a_r8*AS+ki+a_c0], cA[a_r8*AS+ki+a_c0+1]);
            af[3] = pack_halves(cA[a_r8*AS+ki+a_c8], cA[a_r8*AS+ki+a_c8+1]);

            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                int n_off = ni * 8;
                uint32_t bf[2];
                bf[0] = pack_halves(cB[(n_off+b_n0)*BS+ki+0], cB[(n_off+b_n1)*BS+ki+0]);
                bf[1] = pack_halves(cB[(n_off+b_n0)*BS+ki+8], cB[(n_off+b_n1)*BS+ki+8]);
                mma_m16n8k16(acc[ni], af, bf);
            }
        }

        rd = (rd + 1) % STAGES;
        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n"::);

    int row0 = m_base + wid * 16 + lid / 4;
    int row8 = row0 + 8;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        int col0 = ni * 8 + (lid % 4) * 2;
        int col1 = col0 + 1;
        if (row0 < M && col0 < N) C[row0*N+col0] = __float2half(acc[ni][0]);
        if (row0 < M && col1 < N) C[row0*N+col1] = __float2half(acc[ni][1]);
        if (row8 < M && col0 < N) C[row8*N+col0] = __float2half(acc[ni][2]);
        if (row8 < M && col1 < N) C[row8*N+col1] = __float2half(acc[ni][3]);
    }
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct CutlassKernelPingpong {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;
    using TileShape    = cute::Shape<cute::_64, cute::_8, cute::_256>;
    using GridShape    = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, GridShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::NoSmemWarpSpecialized, EpilogueOp>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAccumulator,
        TileShape, GridShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct CutlassKernelWarpSpec {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;
    using TileShape    = cute::Shape<cute::_64, cute::_8, cute::_256>;
    using GridShape    = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, GridShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC,
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::NoSmemWarpSpecialized, EpilogueOp>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA,
        ElementB, LayoutB, AlignB,
        ElementAccumulator,
        TileShape, GridShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

template<typename KS>
static bool try_cutlass(int M, int N, int K, void* pA, void* pB, void* pC) {
    using Gemm = typename KS::Gemm;
    using EA = typename KS::ElementA;
    using EB = typename KS::ElementB;
    using EC = typename KS::ElementC;
    using SA = typename KS::StrideA;
    using SB = typename KS::StrideB;
    using SC = typename KS::StrideC;
    using SD = typename KS::StrideD;

    SA sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
    SB sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    SC sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
    SD sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));

    int dev; cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {(EA*)pA, sA, (EB*)pB, sB},
        {{1.f, 0.f}, (EC*)pC, sC, (EC*)pC, sD},
        hw
    };
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    size_t ws = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return false;
    if (gemm.run() != cutlass::Status::kSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}
#endif

#define CHECK_DTYPE(T, dt) if ((T).options().dtype()!=(dt)) throw std::runtime_error("dtype mismatch")
#define CHECK_SHAPE(T, r, c) if ((T).size(0)!=(r)||(T).size(1)!=(c)) throw std::runtime_error("shape mismatch")

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_DTYPE(a, torch::kHalf);
    CHECK_DTYPE(b, torch::kHalf);
    CHECK_DTYPE(b_col_major, torch::kHalf);
    CHECK_DTYPE(c, torch::kHalf);

    int M = a.size(0), K = a.size(1), N = b.size(1);
    CHECK_SHAPE(a, M, K);
    CHECK_SHAPE(b, K, N);
    CHECK_SHAPE(c, M, N);

    auto* pA = reinterpret_cast<half*>(a.data_ptr());
    auto* pB = reinterpret_cast<half*>(b_col_major.data_ptr());
    auto* pC = reinterpret_cast<half*>(c.data_ptr());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    if (try_cutlass<CutlassKernelPingpong>(M, N, K, a.data_ptr(), b_col_major.data_ptr(), c.data_ptr()))
        return;
    if (try_cutlass<CutlassKernelWarpSpec>(M, N, K, a.data_ptr(), b_col_major.data_ptr(), c.data_ptr()))
        return;
#endif

    if (M % 64 == 0 && N == 64 && K % 128 == 0) {
        constexpr int BM=64, BK=128, STAGES=3, AS=BK+8, BS=BK+8;
        int smem = STAGES * (BM * AS + 64 * BS) * (int)sizeof(half);
        cudaFuncSetAttribute(hgemm_kernel_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        hgemm_kernel_v2<<<dim3(M/64,1), 128, smem>>>(pA, pB, pC, M, N, K);
        cudaDeviceSynchronize();
        if (cudaGetLastError() == cudaSuccess) return;
    }

    if (M % 64 == 0 && N == 64 && K % 64 == 0) {
        constexpr int BM=64, BK=64, STAGES=4, AS=BK+8, BS=BK+8;
        int smem = STAGES * (BM * AS + 64 * BS) * (int)sizeof(half);
        cudaFuncSetAttribute(hgemm_kernel_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        hgemm_kernel_v1<<<dim3(M/64,1), 128, smem>>>(pA, pB, pC, M, N, K);
        cudaDeviceSynchronize();
        if (cudaGetLastError() == cudaSuccess) return;
    }

    throw std::runtime_error("All GEMM variants failed");
}