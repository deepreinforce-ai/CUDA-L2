#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace cute;

using mma_op_t     = SM80_16x8x16_F32F16F16F32_TN;
using mma_traits_t = MMA_Traits<mma_op_t>;
using mma_atom_t   = MMA_Atom<mma_traits_t>;

static constexpr int kMmaEURepeatM = 2;
static constexpr int kMmaEURepeatN = 2;
static constexpr int kMmaEURepeatK = 1;

static constexpr int kMmaPM = kMmaEURepeatM * 16;
static constexpr int kMmaPN = 2 * kMmaEURepeatN * 8;
static constexpr int kMmaPK = kMmaEURepeatK * 16;

using MMA_EU_Repeat = decltype(make_layout(make_shape(
    Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
using MMA_P = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
using TiledMMA_t = decltype(make_tiled_mma(mma_atom_t{}, MMA_EU_Repeat{}, MMA_P{}));

using G2SCopyAtom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, half>;
using G2SCopyTiled = decltype(make_tiled_copy(
    G2SCopyAtom{},
    make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
    make_layout(make_shape(Int<1>{}, Int<8>{}))
));

using S2RCopyAtom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;

using SmemSwizzle  = Swizzle<3, 3, 3>;
using SmemAtomBK64 = decltype(composition(
    SmemSwizzle{},
    make_layout(make_shape(Int<8>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}))
));

template <int BM, int BN, int BK, int kStage, int K_SPLITS>
__global__ __launch_bounds__(128, 3)
void hgemm_splitk_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ W,
    int M, int N, int K)
{
    using SmemLayoutA = decltype(tile_to_shape(
        SmemAtomBK64{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemAtomBK64{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})
    ));

    extern __shared__ half shm[];
    half* Ashm = shm;
    half* Bshm = shm + cute::cosize(SmemLayoutA{});

    const int tid = threadIdx.x;
    const int ix  = blockIdx.x;
    const int iy  = blockIdx.y;
    const int iz  = blockIdx.z;

    const int k_per_split = K / K_SPLITS;
    const int k_start     = iz * k_per_split;
    const int ntile       = k_per_split / BK;

    const half* A_base = A + (long long)(iy * BM) * K + k_start;
    const half* B_base = B + (long long)(ix * BN) * K + k_start;

    Tensor gA_slice = make_tensor(
        make_gmem_ptr(A_base),
        make_shape(Int<BM>{}, k_per_split),
        make_stride(K, Int<1>{}));
    Tensor gB_slice = make_tensor(
        make_gmem_ptr(B_base),
        make_shape(Int<BN>{}, k_per_split),
        make_stride(K, Int<1>{}));

    Tensor gA = local_tile(gA_slice, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
    Tensor gB = local_tile(gB_slice, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));

    float* W_tile = W + (long long)(iz) * M * N
                      + (long long)(iy * BM) * N
                      + (ix * BN);
    Tensor gW = make_tensor(make_gmem_ptr(W_tile),
                            make_shape(Int<BM>{}, Int<BN>{}),
                            make_stride(N, Int<1>{}));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gW);
    clear(tCrD);

    G2SCopyTiled g2s_a, g2s_b;
    auto g2s_thr_a = g2s_a.get_slice(tid);
    auto tAgA_s    = g2s_thr_a.partition_S(gA);
    auto tAsA_d    = g2s_thr_a.partition_D(sA);
    auto g2s_thr_b = g2s_b.get_slice(tid);
    auto tBgB_s    = g2s_thr_b.partition_S(gB);
    auto tBsB_d    = g2s_thr_b.partition_D(sB);

    auto s2r_a     = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_a = s2r_a.get_slice(tid);
    auto tAsA      = s2r_thr_a.partition_S(sA);
    auto tCrA_view = s2r_thr_a.retile_D(tCrA);

    auto s2r_b     = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
    auto s2r_thr_b = s2r_b.get_slice(tid);
    auto tBsB      = s2r_thr_b.partition_S(sB);
    auto tCrB_view = s2r_thr_b.retile_D(tCrB);

    int itile_rd = 0, ismem_rd = 0, ismem_wr = 0;

    #pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        cute::copy(g2s_a, tAgA_s(_, _, _, s), tAsA_d(_, _, _, s));
        cute::copy(g2s_b, tBgB_s(_, _, _, s), tBsB_d(_, _, _, s));
        cp_async_fence();
        ++itile_rd;
        ++ismem_wr;
    }
    cp_async_wait<kStage - 2>();
    __syncthreads();

    const int nk = size<2>(tCrA);
    cute::copy(s2r_a, tAsA(_, _, 0, ismem_rd), tCrA_view(_, _, 0));
    cute::copy(s2r_b, tBsB(_, _, 0, ismem_rd), tCrB_view(_, _, 0));

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_rd = (ismem_rd + 1) % kStage;
            }

            cute::copy(s2r_a, tAsA(_, _, ik_next, ismem_rd), tCrA_view(_, _, ik_next));
            cute::copy(s2r_b, tBsB(_, _, ik_next, ismem_rd), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_rd < ntile) {
                    cute::copy(g2s_a, tAgA_s(_, _, _, itile_rd),
                                      tAsA_d(_, _, _, ismem_wr));
                    cute::copy(g2s_b, tBgB_s(_, _, _, itile_rd),
                                      tBsB_d(_, _, _, ismem_wr));
                    ismem_wr = (ismem_wr + 1) % kStage;
                }
                ++itile_rd;
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto tCgW = thr_mma.partition_C(gW);
    #pragma unroll
    for (int i = 0; i < size(tCrD); ++i) {
        tCgW(i) = tCrD(i);
    }
}

template <int K_SPLITS>
__global__ __launch_bounds__(256, 4)
void enhanced_reduce_kernel(
    const float* __restrict__ workspace,
    half*        __restrict__ output,
    int          total_elements)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int base_idx = tid * 8;
    
    if (base_idx + 7 < total_elements) {
        float4 sum1 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 sum2 = make_float4(0.f, 0.f, 0.f, 0.f);
        
        #pragma unroll
        for (int k = 0; k < K_SPLITS; ++k) {
            const float* src = workspace + (long long)k * total_elements + base_idx;
            float4 v1 = reinterpret_cast<const float4*>(src)[0];
            float4 v2 = reinterpret_cast<const float4*>(src)[1];
            
            sum1.x += v1.x; sum1.y += v1.y; sum1.z += v1.z; sum1.w += v1.w;
            sum2.x += v2.x; sum2.y += v2.y; sum2.z += v2.z; sum2.w += v2.w;
        }
        
        half* out = output + base_idx;
        reinterpret_cast<half2*>(out)[0] = __float22half2_rn(make_float2(sum1.x, sum1.y));
        reinterpret_cast<half2*>(out)[1] = __float22half2_rn(make_float2(sum1.z, sum1.w));
        reinterpret_cast<half2*>(out)[2] = __float22half2_rn(make_float2(sum2.x, sum2.y));
        reinterpret_cast<half2*>(out)[3] = __float22half2_rn(make_float2(sum2.z, sum2.w));
        
    } else if (base_idx < total_elements) {
        for (int i = base_idx; i < total_elements && i < base_idx + 8; ++i) {
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < K_SPLITS; ++k) {
                sum += workspace[(long long)k * total_elements + i];
            }
            output[i] = __float2half(sum);
        }
    }
}

template <int BM, int BN, int BK, int kStage>
size_t get_smem_size() {
    using SmemLayoutA = decltype(tile_to_shape(
        SmemAtomBK64{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemAtomBK64{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})
    ));
    return (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(half);
}

namespace {
    float* g_workspace      = nullptr;
    size_t g_workspace_size = 0;
}

static float* ensure_workspace(size_t bytes) {
    if (g_workspace_size < bytes) {
        if (g_workspace) {
            cudaFree(g_workspace);
            g_workspace = nullptr;
        }
        cudaMalloc(&g_workspace, bytes);
        g_workspace_size = bytes;
    }
    return g_workspace;
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                \
    do {                                                                    \
        if ((T).options().dtype() != (th_type)) {                           \
            std::cout << "Dtype mismatch: " << (T).options() << "\n";      \
            throw std::runtime_error("values must be " #th_type);          \
        }                                                                    \
    } while (0)

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                 \
    do {                                                                    \
        if ((T).size(0) != (S0) || (T).size(1) != (S1)) {                  \
            throw std::runtime_error("Tensor shape mismatch");              \
        }                                                                    \
    } while (0)

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf);
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf);

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    CHECK_TORCH_TENSOR_SHAPE(a, M, K);
    CHECK_TORCH_TENSOR_SHAPE(b, K, N);
    CHECK_TORCH_TENSOR_SHAPE(c, M, N);

    const half* Aptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* Bptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       Cptr = reinterpret_cast<half*>(c.data_ptr());

    constexpr int BM     = 64;
    constexpr int BN     = 64;
    constexpr int BK     = 64;
    constexpr int SPLITS = 16;
    constexpr int STAGE  = 4;

    static_assert(8192 % (SPLITS * BK) == 0,
                  "K=8192 must be divisible by SPLITS*BK");

    const int total_mn  = M * N;
    const size_t ws_bytes = (size_t)SPLITS * total_mn * sizeof(float);
    float* workspace = ensure_workspace(ws_bytes);

    const size_t smem = get_smem_size<BM, BN, BK, STAGE>();

    cudaFuncSetAttribute(
        hgemm_splitk_kernel<BM, BN, BK, STAGE, SPLITS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));

    dim3 grid_compute((N + BN - 1) / BN,
                      (M + BM - 1) / BM,
                      SPLITS);

    hgemm_splitk_kernel<BM, BN, BK, STAGE, SPLITS>
        <<<grid_compute, 128, smem>>>(
            Aptr, Bptr, workspace, M, N, K);

    constexpr int REDUCE_THREADS = 256;
    const int elements_per_thread = 8;
    const int reduce_blocks = (total_mn + (REDUCE_THREADS * elements_per_thread) - 1) / 
                              (REDUCE_THREADS * elements_per_thread);

    enhanced_reduce_kernel<SPLITS>
        <<<reduce_blocks, REDUCE_THREADS>>>(
            workspace, Cptr, total_mn);
}