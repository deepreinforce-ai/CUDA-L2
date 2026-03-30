#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>

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

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

static constexpr int SMEM_B_STRIDE = 264;
static constexpr int SMEM_A_STRIDE = 72;

__global__ void __launch_bounds__(128, 4)
hgemm_wmma_32x256_4warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    const int warp_id   = threadIdx.x >> 5;
    const int lane_id   = threadIdx.x & 31;
    const int block_row = blockIdx.x * 32;
    const int warp_m    = warp_id >> 1;
    const int warp_n    = warp_id & 1;
    const int row16     = block_row + warp_m * 16;
    const int col128    = warp_n * 128;

    __shared__ __align__(128) half  smem_B[64 * SMEM_B_STRIDE];
    __shared__ __align__(128) half  smem_A[32 * SMEM_A_STRIDE];
    __shared__ __align__(16)  float smem_scratch[4][256];

    const int tid = threadIdx.x;

    {
        const float4* Bf4 = reinterpret_cast<const float4*>(B);
        #pragma unroll 4
        for (int i = tid; i < 64 * 32; i += 128) {
            int row  = i >> 5;
            int col8 = i & 31;
            cp_async16(smem_B + row * SMEM_B_STRIDE + col8 * 8, Bf4 + row * 32 + col8);
        }
    }

    if (block_row < M) {
        const float4* Af4 = reinterpret_cast<const float4*>(A + block_row * 64);
        #pragma unroll 2
        for (int i = tid; i < 32 * 8; i += 128) {
            int row  = i >> 3;
            int col8 = i & 7;
            if (block_row + row < M)
                cp_async16(smem_A + row * SMEM_A_STRIDE + col8 * 8, Af4 + row * 8 + col8);
        }
    }

    cp_async_wait_all();
    __syncthreads();

    if (row16 >= M) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(acc[i], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    const half* sA_warp = smem_A + warp_m * 16 * SMEM_A_STRIDE;

    #pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        wmma::load_matrix_sync(a_frag, sA_warp + ks * 16, SMEM_A_STRIDE);
        const half* sB_k = smem_B + ks * 16 * SMEM_B_STRIDE + col128;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            wmma::load_matrix_sync(b_frag, sB_k + nt * 16, SMEM_B_STRIDE);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }

    half* C_warp = C + row16 * 256 + col128;
    float* scratch = smem_scratch[warp_id];

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        wmma::store_matrix_sync(scratch, acc[nt], 16, wmma::mem_row_major);
        __syncwarp();
        half* C_nt = C_warp + nt * 16;
        #pragma unroll
        for (int i = lane_id; i < 128; i += 32) {
            int row = i >> 3;
            int col = (i & 7) * 2;
            if (row16 + row < M) {
                float f0 = scratch[row * 16 + col];
                float f1 = scratch[row * 16 + col + 1];
                half2 h2 = __floats2half2_rn(f0, f1);
                *reinterpret_cast<half2*>(C_nt + row * 256 + col) = h2;
            }
        }
    }
}

__global__ void __launch_bounds__(64, 8)
hgemm_wmma_16x256_2warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    const int warp_id   = threadIdx.x >> 5;
    const int lane_id   = threadIdx.x & 31;
    const int block_row = blockIdx.x * 16;
    const int warp_col  = warp_id * 128;

    if (block_row >= M) return;

    __shared__ __align__(128) half  smem_B[64 * SMEM_B_STRIDE];
    __shared__ __align__(128) half  smem_A[16 * SMEM_A_STRIDE];
    __shared__ __align__(16)  float smem_scratch[2][256];

    const int tid = threadIdx.x;

    {
        const float4* Bf4 = reinterpret_cast<const float4*>(B);
        #pragma unroll 8
        for (int i = tid; i < 64 * 32; i += 64) {
            int row  = i >> 5;
            int col8 = i & 31;
            cp_async16(smem_B + row * SMEM_B_STRIDE + col8 * 8, Bf4 + row * 32 + col8);
        }
    }

    {
        const float4* Af4 = reinterpret_cast<const float4*>(A + block_row * 64);
        #pragma unroll 2
        for (int i = tid; i < 16 * 8; i += 64) {
            int row  = i >> 3;
            int col8 = i & 7;
            cp_async16(smem_A + row * SMEM_A_STRIDE + col8 * 8, Af4 + row * 8 + col8);
        }
    }

    cp_async_wait_all();
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(acc[i], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    #pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        wmma::load_matrix_sync(a_frag, smem_A + ks * 16, SMEM_A_STRIDE);
        const half* sB_k = smem_B + ks * 16 * SMEM_B_STRIDE + warp_col;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            wmma::load_matrix_sync(b_frag, sB_k + nt * 16, SMEM_B_STRIDE);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }

    half* C_warp = C + block_row * 256 + warp_col;
    float* scratch = smem_scratch[warp_id];

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        wmma::store_matrix_sync(scratch, acc[nt], 16, wmma::mem_row_major);
        __syncwarp();
        half* C_nt = C_warp + nt * 16;
        #pragma unroll
        for (int i = lane_id; i < 128; i += 32) {
            int row = i >> 3;
            int col = (i & 7) * 2;
            float f0 = scratch[row * 16 + col];
            float f1 = scratch[row * 16 + col + 1];
            half2 h2 = __floats2half2_rn(f0, f1);
            *reinterpret_cast<half2*>(C_nt + row * 256 + col) = h2;
        }
    }
}

__global__ void __launch_bounds__(192, 2)
hgemm_wmma_48x256_6warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    const int warp_id   = threadIdx.x >> 5;
    const int lane_id   = threadIdx.x & 31;
    const int block_row = blockIdx.x * 48;
    const int warp_m    = warp_id / 2;
    const int warp_n    = warp_id & 1;
    const int row16     = block_row + warp_m * 16;
    const int col128    = warp_n * 128;

    __shared__ __align__(128) half  smem_B[64 * SMEM_B_STRIDE];
    __shared__ __align__(128) half  smem_A[48 * SMEM_A_STRIDE];
    __shared__ __align__(16)  float smem_scratch[6][256];

    const int tid = threadIdx.x;

    {
        const float4* Bf4 = reinterpret_cast<const float4*>(B);
        for (int i = tid; i < 64 * 32; i += 192) {
            int row  = i >> 5;
            int col8 = i & 31;
            cp_async16(smem_B + row * SMEM_B_STRIDE + col8 * 8, Bf4 + row * 32 + col8);
        }
    }

    if (block_row < M) {
        const float4* Af4 = reinterpret_cast<const float4*>(A + block_row * 64);
        int rows_avail = min(48, M - block_row);
        for (int i = tid; i < rows_avail * 8; i += 192) {
            int row  = i >> 3;
            int col8 = i & 7;
            cp_async16(smem_A + row * SMEM_A_STRIDE + col8 * 8, Af4 + row * 8 + col8);
        }
    }

    cp_async_wait_all();
    __syncthreads();

    if (row16 >= M) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(acc[i], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    const half* sA_warp = smem_A + warp_m * 16 * SMEM_A_STRIDE;

    #pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        wmma::load_matrix_sync(a_frag, sA_warp + ks * 16, SMEM_A_STRIDE);
        const half* sB_k = smem_B + ks * 16 * SMEM_B_STRIDE + col128;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            wmma::load_matrix_sync(b_frag, sB_k + nt * 16, SMEM_B_STRIDE);
            wmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
        }
    }

    half* C_warp = C + row16 * 256 + col128;
    float* scratch = smem_scratch[warp_id];

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        wmma::store_matrix_sync(scratch, acc[nt], 16, wmma::mem_row_major);
        __syncwarp();
        half* C_nt = C_warp + nt * 16;
        #pragma unroll
        for (int i = lane_id; i < 128; i += 32) {
            int row = i >> 3;
            int col = (i & 7) * 2;
            if (row16 + row < M) {
                float f0 = scratch[row * 16 + col];
                float f1 = scratch[row * 16 + col + 1];
                half2 h2 = __floats2half2_rn(f0, f1);
                *reinterpret_cast<half2*>(C_nt + row * 256 + col) = h2;
            }
        }
    }
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct Cfg_128x256_Coop_C2_SK {
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
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_2,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_128x256_Coop_C1_SK {
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
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_128x256_Coop_C2_Pers {
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
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_2,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_128x256_Coop_C1_Pers {
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
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_128x256_Ping_C2_Pers {
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
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_2,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_128x256_Ping_C1_Pers {
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
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_128x128_Coop_C1x2_Pers {
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
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_2,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct Cfg_128x128_Coop_C1_SK {
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
  using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using GroupShape   = cute::Shape<cute::_1,   cute::_1,   cute::_1>;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, GroupShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
      ElementAccumulator, TileShape, GroupShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm    = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <typename Cfg>
struct StaticGemmRunner {
    using Gemm    = typename Cfg::Gemm;
    using StrideA = typename Cfg::StrideA;
    using StrideB = typename Cfg::StrideB;
    using StrideC = typename Cfg::StrideC;
    using StrideD = typename Cfg::StrideD;

    static int& state() { static int s = 0; return s; }
    static Gemm& gemm_obj() { static Gemm g; return g; }
    static cutlass::device_memory::allocation<uint8_t>& workspace() {
        static cutlass::device_memory::allocation<uint8_t> ws;
        return ws;
    }

    static bool run(void* pA, void* pB_col, void* pC, int M, int N, int K,
                    const cutlass::KernelHardwareInfo& hw_info) {
        if (state() == -1) return false;

        StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
            {reinterpret_cast<typename Cfg::ElementA*>(pA), sA,
             reinterpret_cast<typename Cfg::ElementB*>(pB_col), sB},
            {{1.0f, 0.0f},
             reinterpret_cast<typename Cfg::ElementC*>(pC), sC,
             reinterpret_cast<typename Cfg::ElementC*>(pC), sD},
            hw_info};

        if (state() == 0) {
            if (gemm_obj().can_implement(args) != cutlass::Status::kSuccess) {
                state() = -1; return false;
            }
            size_t ws_size = Gemm::get_workspace_size(args);
            workspace() = cutlass::device_memory::allocation<uint8_t>(ws_size);
            if (gemm_obj().initialize(args, workspace().get()) != cutlass::Status::kSuccess) {
                state() = -1; return false;
            }
            state() = 1;
        } else {
            if (gemm_obj().initialize(args, workspace().get()) != cutlass::Status::kSuccess) {
                return false;
            }
        }
        if (gemm_obj().run() != cutlass::Status::kSuccess) return false;
        return cudaGetLastError() == cudaSuccess;
    }
};

template <typename Cfg>
bool try_gemm_static(void* pA, void* pB_col, void* pC, int M, int N, int K,
                     const cutlass::KernelHardwareInfo& hw_info) {
    return StaticGemmRunner<Cfg>::run(pA, pB_col, pC, M, N, K, hw_info);
}

#endif


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

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr());

    if (K == 64 && N == 256) {
        if ((M % 48) == 0) {
            int num_blocks = M / 48;
            hgemm_wmma_48x256_6warp<<<num_blocks, 192>>>(ptr_A, ptr_B, ptr_C, M);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaDeviceSynchronize(); cudaGetLastError();
        }

        if ((M % 32) == 0) {
            int num_blocks = M / 32;
            hgemm_wmma_32x256_4warp<<<num_blocks, 128>>>(ptr_A, ptr_B, ptr_C, M);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaDeviceSynchronize(); cudaGetLastError();
        }

        if ((M % 16) == 0) {
            int num_blocks = M / 16;
            hgemm_wmma_16x256_2warp<<<num_blocks, 64>>>(ptr_A, ptr_B, ptr_C, M);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaDeviceSynchronize(); cudaGetLastError();
        }

        {
            int num_blocks = (M + 47) / 48;
            hgemm_wmma_48x256_6warp<<<num_blocks, 192>>>(ptr_A, ptr_B, ptr_C, M);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) return;
            cudaDeviceSynchronize(); cudaGetLastError();
        }
    }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    {
        cutlass::KernelHardwareInfo hw_info;
        cudaGetDevice(&hw_info.device_id);
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

        void* pA_v   = a.data_ptr();
        void* pB_col = b_col_major.data_ptr();
        void* pC_v   = c.data_ptr();

        if (try_gemm_static<Cfg_128x256_Coop_C2_SK>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;
        if (try_gemm_static<Cfg_128x256_Coop_C1_SK>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;
        if (try_gemm_static<Cfg_128x256_Coop_C2_Pers>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;
        if (try_gemm_static<Cfg_128x256_Coop_C1_Pers>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;
        if (try_gemm_static<Cfg_128x256_Ping_C2_Pers>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;
        if (try_gemm_static<Cfg_128x256_Ping_C1_Pers>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;
        if (try_gemm_static<Cfg_128x128_Coop_C1x2_Pers>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;
        if (try_gemm_static<Cfg_128x128_Coop_C1_SK>(pA_v, pB_col, pC_v, M, N, K, hw_info)) return;

        throw std::runtime_error("cuda_l2_h100_fp32: all configs failed");
    }
#else
    throw std::runtime_error("cuda_l2_h100_fp32: no supported implementation");
#endif
}