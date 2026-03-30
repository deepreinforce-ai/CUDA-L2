#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/util/packed_stride.hpp"

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if ((T).options().dtype() != (th_type)) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

static constexpr int M_GLOBAL = 128;
static constexpr int N_GLOBAL = 128;
static constexpr int K_GLOBAL = 2048;
static constexpr int KBLOCK   = 64;
static constexpr int NSTAGES  = 5;

static constexpr int A_STRIDE  = KBLOCK + 8;
static constexpr int B_STRIDE  = N_GLOBAL + 8;

static constexpr int A_HALFS     = M_GLOBAL * A_STRIDE;
static constexpr int B_HALFS     = KBLOCK * B_STRIDE;
static constexpr int STAGE_HALFS = A_HALFS + B_HALFS;
static constexpr int TOTAL_HALFS = NSTAGES * STAGE_HALFS;
static constexpr int TOTAL_SMEM  = TOTAL_HALFS * 2;

__global__ __launch_bounds__(256, 1)
void hgemm_v8(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
)
{
    extern __shared__ half smem_pool[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int wg         = warp_id >> 2;
    const int warp_in_wg = warp_id &  3;

    const int wg_row_base   = wg * 64;
    const int warp_col_base = warp_in_wg * 32;

    static constexpr int WM = 4;
    static constexpr int WN = 4;

    float acc[WM][WN][4];
    #pragma unroll
    for (int mi = 0; mi < WM; mi++)
        #pragma unroll
        for (int ni = 0; ni < WN; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    #define SMEM_A(s) (smem_pool + (s)*STAGE_HALFS)
    #define SMEM_B(s) (smem_pool + (s)*STAGE_HALFS + A_HALFS)

    #define CP16(dst, src) \
        asm volatile("cp.async.cg.shared.global.L2::128B [%0],[%1],16;\n" \
            ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"((uint64_t)(src)))
    #define COMMIT() asm volatile("cp.async.commit_group;\n"::)
    #define WAIT(n)  asm volatile("cp.async.wait_group " #n ";\n"::)

    auto issue_stage = [&](int s, int kt) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int chunk = tid + i * 256;
            int row   = chunk >> 3;
            int col8  = chunk &  7;
            CP16(SMEM_A(s) + row * A_STRIDE + col8*8,
                 A + (long)row * K_GLOBAL + kt * KBLOCK + col8*8);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int chunk = tid + i * 256;
            int row   = chunk >> 4;
            int col8  = chunk & 15;
            CP16(SMEM_B(s) + row * B_STRIDE + col8*8,
                 B + (long)(kt * KBLOCK + row) * N_GLOBAL + col8*8);
        }
        COMMIT();
    };

    const int num_kt = K_GLOBAL / KBLOCK;

    #pragma unroll
    for (int s = 0; s < NSTAGES - 1 && s < num_kt; s++) {
        issue_stage(s, s);
    }

    #pragma unroll 1
    for (int kt = 0; kt < num_kt; kt++) {
        int cur    = kt % NSTAGES;
        int pre_kt = kt + NSTAGES - 1;
        if (pre_kt < num_kt) {
            issue_stage(pre_kt % NSTAGES, pre_kt);
        }

        WAIT(3);
        __syncthreads();

        const half* As = SMEM_A(cur);
        const half* Bs = SMEM_B(cur);

        uint32_t a_frag[WM][4];
        uint32_t b_frag[WN][2];

        #pragma unroll
        for (int mi = 0; mi < WM; mi++) {
            int base_row = wg_row_base + mi * 16;
            int row = base_row + (lane & 15);
            int col = (lane >> 4) * 8;
            uint32_t addr = __cvta_generic_to_shared(As + row * A_STRIDE + col);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                  "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            int n_base = warp_col_base + ni * 8;
            int row    = (lane & 15);
            int col    = n_base + ((lane >> 4) * 8);
            uint32_t addr = __cvta_generic_to_shared(Bs + row * B_STRIDE + col);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                : "r"(addr)
            );
        }

        #pragma unroll
        for (int ki = 0; ki < KBLOCK / 16; ki++) {
            uint32_t a_next[WM][4];
            uint32_t b_next[WN][2];

            if (ki + 1 < KBLOCK / 16) {
                int nk = (ki + 1) * 16;
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    int base_row = wg_row_base + mi * 16;
                    int row = base_row + (lane & 15);
                    int col = nk + ((lane >> 4) * 8);
                    uint32_t addr = __cvta_generic_to_shared(As + row * A_STRIDE + col);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                        : "=r"(a_next[mi][0]), "=r"(a_next[mi][1]),
                          "=r"(a_next[mi][2]), "=r"(a_next[mi][3])
                        : "r"(addr)
                    );
                }
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    int n_base = warp_col_base + ni * 8;
                    int row    = nk + (lane & 15);
                    int col    = n_base + ((lane >> 4) * 8);
                    uint32_t addr = __cvta_generic_to_shared(Bs + row * B_STRIDE + col);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                        : "=r"(b_next[ni][0]), "=r"(b_next[ni][1])
                        : "r"(addr)
                    );
                }
            }

            #pragma unroll
            for (int mi = 0; mi < WM; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(acc[mi][ni][0]), "=f"(acc[mi][ni][1]),
                          "=f"(acc[mi][ni][2]), "=f"(acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                          "f"(acc[mi][ni][0]), "f"(acc[mi][ni][1]),
                          "f"(acc[mi][ni][2]), "f"(acc[mi][ni][3])
                    );
                }
            }

            if (ki + 1 < KBLOCK / 16) {
                #pragma unroll
                for (int mi = 0; mi < WM; mi++) {
                    a_frag[mi][0] = a_next[mi][0]; a_frag[mi][1] = a_next[mi][1];
                    a_frag[mi][2] = a_next[mi][2]; a_frag[mi][3] = a_next[mi][3];
                }
                #pragma unroll
                for (int ni = 0; ni < WN; ni++) {
                    b_frag[ni][0] = b_next[ni][0];
                    b_frag[ni][1] = b_next[ni][1];
                }
            }
        }

        __syncthreads();
    }

    WAIT(0);

    const int epi_row_lo = lane >> 2;
    const int epi_col_lo = (lane & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < WM; mi++) {
        int r0 = wg_row_base + mi * 16 + epi_row_lo;
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WN; ni++) {
            int col = warp_col_base + ni * 8 + epi_col_lo;
            half2 v01 = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            half2 v23 = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            *reinterpret_cast<half2*>(C + (long)r0 * N_GLOBAL + col) = v01;
            *reinterpret_cast<half2*>(C + (long)r1 * N_GLOBAL + col) = v23;
        }
    }

    #undef SMEM_A
    #undef SMEM_B
    #undef CP16
    #undef COMMIT
    #undef WAIT
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
static constexpr int AlignA = 8, AlignB = 8, AlignC = 8, AlignD = 8;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

struct HgemmPingpong {
    using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using TilingShape  = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TilingShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
        cutlass::epilogue::TmaWarpSpecialized, EpilogueOp>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
        ElementAccumulator, TileShape, TilingShape,
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

struct HgemmCooperative {
    using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_64>;
    using TilingShape  = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, TilingShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
        cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueOp>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB,
        ElementAccumulator, TileShape, TilingShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int>, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

struct GlobalState {
    int active   = 0;
    bool smem_set = false;

    HgemmPingpong::Gemm pp_gemm;
    typename HgemmPingpong::Gemm::Arguments pp_args;
    HgemmPingpong::StrideA pp_sA; HgemmPingpong::StrideB pp_sB;
    HgemmPingpong::StrideC pp_sC; HgemmPingpong::StrideD pp_sD;
    uint8_t* pp_ws = nullptr; size_t pp_ws_size = 0;

    HgemmCooperative::Gemm coop_gemm;
    typename HgemmCooperative::Gemm::Arguments coop_args;
    HgemmCooperative::StrideA coop_sA; HgemmCooperative::StrideB coop_sB;
    HgemmCooperative::StrideC coop_sC; HgemmCooperative::StrideD coop_sD;
    uint8_t* coop_ws = nullptr; size_t coop_ws_size = 0;

    cudaStream_t stream = nullptr;
    cutlass::KernelHardwareInfo hw;
    bool hw_ok = false;

    ~GlobalState() {
        if (pp_ws)   cudaFree(pp_ws);
        if (coop_ws) cudaFree(coop_ws);
        if (stream)  cudaStreamDestroy(stream);
    }
};
static GlobalState gs;

static void ensure_infra() {
    if (!gs.hw_ok) {
        cudaGetDevice(&gs.hw.device_id);
        gs.hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(gs.hw.device_id);
        gs.hw_ok = true;
    }
    if (!gs.stream)
        cudaStreamCreateWithFlags(&gs.stream, cudaStreamNonBlocking);
}

static bool init_pp(const ElementA* pA, const ElementB* pB,
                    const ElementC* pC, ElementD* pD, int M, int N, int K) {
    gs.pp_sA = cutlass::make_cute_packed_stride(HgemmPingpong::StrideA{}, cute::make_shape(M, K, 1));
    gs.pp_sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    gs.pp_sC = cutlass::make_cute_packed_stride(HgemmPingpong::StrideC{}, cute::make_shape(M, N, 1));
    gs.pp_sD = cutlass::make_cute_packed_stride(HgemmPingpong::StrideD{}, cute::make_shape(M, N, 1));
    gs.pp_args = {
        cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K},
        {pA, gs.pp_sA, pB, gs.pp_sB}, {{1.f,0.f}, pC, gs.pp_sC, pD, gs.pp_sD}, gs.hw
    };
    if (gs.pp_gemm.can_implement(gs.pp_args) != cutlass::Status::kSuccess) return false;
    size_t wsz = HgemmPingpong::Gemm::get_workspace_size(gs.pp_args);
    if (!wsz) wsz = 256;
    if (wsz > gs.pp_ws_size) {
        if (gs.pp_ws) cudaFree(gs.pp_ws);
        if (cudaMalloc(&gs.pp_ws, wsz) != cudaSuccess) return false;
        gs.pp_ws_size = wsz;
    }
    return gs.pp_gemm.initialize(gs.pp_args, gs.pp_ws) == cutlass::Status::kSuccess;
}

static bool init_coop(const ElementA* pA, const ElementB* pB,
                      const ElementC* pC, ElementD* pD, int M, int N, int K) {
    gs.coop_sA = cutlass::make_cute_packed_stride(HgemmCooperative::StrideA{}, cute::make_shape(M, K, 1));
    gs.coop_sB = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    gs.coop_sC = cutlass::make_cute_packed_stride(HgemmCooperative::StrideC{}, cute::make_shape(M, N, 1));
    gs.coop_sD = cutlass::make_cute_packed_stride(HgemmCooperative::StrideD{}, cute::make_shape(M, N, 1));
    gs.coop_args = {
        cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K},
        {pA, gs.coop_sA, pB, gs.coop_sB}, {{1.f,0.f}, pC, gs.coop_sC, pD, gs.coop_sD}, gs.hw
    };
    if (gs.coop_gemm.can_implement(gs.coop_args) != cutlass::Status::kSuccess) return false;
    size_t wsz = HgemmCooperative::Gemm::get_workspace_size(gs.coop_args);
    if (!wsz) wsz = 256;
    if (wsz > gs.coop_ws_size) {
        if (gs.coop_ws) cudaFree(gs.coop_ws);
        if (cudaMalloc(&gs.coop_ws, wsz) != cudaSuccess) return false;
        gs.coop_ws_size = wsz;
    }
    return gs.coop_gemm.initialize(gs.coop_args, gs.coop_ws) == cutlass::Status::kSuccess;
}

#endif

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

    const half* pA_raw = reinterpret_cast<const half*>(a.data_ptr());
    const half* pB_raw = reinterpret_cast<const half*>(b.data_ptr());
    half*       pC_raw = reinterpret_cast<half*>(c.data_ptr());

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    ensure_infra();

    auto* pA     = reinterpret_cast<const ElementA*>(a.data_ptr());
    auto* pB_col = reinterpret_cast<const ElementB*>(b_col_major.data_ptr());
    auto* pC     = reinterpret_cast<const ElementC*>(c.data_ptr());
    auto* pD     = reinterpret_cast<ElementD*>(c.data_ptr());

    if (gs.active == 1) {
        hgemm_v8<<<1, 256, TOTAL_SMEM, gs.stream>>>(pA_raw, pB_raw, pC_raw);
        return;
    }

    if (gs.active == 2) {
        gs.pp_args.mainloop.ptr_A = pA;
        gs.pp_args.mainloop.ptr_B = pB_col;
        gs.pp_args.epilogue.ptr_C = pC;
        gs.pp_args.epilogue.ptr_D = pD;
        if (gs.pp_gemm.initialize(gs.pp_args, gs.pp_ws) == cutlass::Status::kSuccess &&
            gs.pp_gemm.run(gs.stream) == cutlass::Status::kSuccess) {
            return;
        }
        gs.active = 0;
    }

    if (gs.active == 3) {
        gs.coop_args.mainloop.ptr_A = pA;
        gs.coop_args.mainloop.ptr_B = pB_col;
        gs.coop_args.epilogue.ptr_C = pC;
        gs.coop_args.epilogue.ptr_D = pD;
        if (gs.coop_gemm.initialize(gs.coop_args, gs.coop_ws) == cutlass::Status::kSuccess &&
            gs.coop_gemm.run(gs.stream) == cutlass::Status::kSuccess) {
            return;
        }
        gs.active = 0;
    }

    if (gs.active == 0) {
        if (!gs.smem_set) {
            cudaFuncSetAttribute(hgemm_v8,
                cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM);
            gs.smem_set = true;
        }

        hgemm_v8<<<1, 256, TOTAL_SMEM, gs.stream>>>(pA_raw, pB_raw, pC_raw);
        cudaStreamSynchronize(gs.stream);
        if (cudaGetLastError() == cudaSuccess) {
            gs.active = 1;
            return;
        }

        if (init_pp(pA, pB_col, pC, pD, M, N, K)) {
            if (gs.pp_gemm.run(gs.stream) == cutlass::Status::kSuccess) {
                cudaStreamSynchronize(gs.stream);
                if (cudaGetLastError() == cudaSuccess) {
                    gs.active = 2;
                    return;
                }
            }
        }

        if (init_coop(pA, pB_col, pC, pD, M, N, K)) {
            if (gs.coop_gemm.run(gs.stream) == cutlass::Status::kSuccess) {
                cudaStreamSynchronize(gs.stream);
                if (cudaGetLastError() == cudaSuccess) {
                    gs.active = 3;
                    return;
                }
            }
        }

        gs.active = 1;
        hgemm_v8<<<1, 256, TOTAL_SMEM, gs.stream>>>(pA_raw, pB_raw, pC_raw);
    }

#else
    if (!gs.stream)
        cudaStreamCreateWithFlags(&gs.stream, cudaStreamNonBlocking);
    if (!gs.smem_set) {
        cudaFuncSetAttribute(hgemm_v8,
            cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM);
        gs.smem_set = true;
    }
    hgemm_v8<<<1, 256, TOTAL_SMEM, gs.stream>>>(pA_raw, pB_raw, pC_raw);
#endif
}