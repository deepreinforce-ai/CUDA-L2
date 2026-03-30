#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace cute;

template <int BM, int BN, int BK, int KSTAGE, int NUM_SPLITS>
__global__ void __launch_bounds__(128, 2)
hgemm_splitk_phase1(
    const half*  __restrict__ A,
    const half*  __restrict__ B_cm,
    float*       __restrict__ C_partial,
    int M, int N, int K,
    int tiles_per_split,
    int ntile_k
) {
    const int split_id = blockIdx.z;
    const int k_start  = split_id * tiles_per_split;
    if (k_start >= ntile_k) return;
    const int k_count  = min(tiles_per_split, ntile_k - k_start);
    const int k_end    = k_start + k_count;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SLayoutA = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<KSTAGE>{})));
    using SLayoutB = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<KSTAGE>{})));

    extern __shared__ half smem[];
    auto sA = make_tensor(make_smem_ptr(smem), SLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem + cosize(SLayoutA{})), SLayoutB{});

    using MmaOp     = SM80_16x8x16_F32F16F16F32_TN;
    using MmaTraits = MMA_Traits<MmaOp>;
    using MmaAtom_t = MMA_Atom<MmaTraits>;
    using MmaShape  = MmaTraits::Shape_MNK;
    static constexpr int PM = 2 * 1 * get<0>(MmaShape{});
    static constexpr int PN = 2 * 2 * get<1>(MmaShape{});
    static constexpr int PK = 1 * 1 * get<2>(MmaShape{});
    using MmaRepeat = decltype(make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})));
    using MmaTile   = Tile<Int<PM>, Int<PN>, Int<PK>>;
    using TiledMMA  = decltype(make_tiled_mma(MmaAtom_t{}, MmaRepeat{}, MmaTile{}));

    using G2SAtom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, half>;
    using G2SCopy = decltype(make_tiled_copy(G2SAtom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));

    using S2RAtom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    Tensor gA_global = make_tensor(make_gmem_ptr(A),
                                   make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor gB_global = make_tensor(make_gmem_ptr(B_cm),
                                   make_shape(N, K), make_stride(K, Int<1>{}));

    Tensor tgA = local_tile(gA_global, make_tile(Int<BM>{}, Int<BK>{}), make_coord(0, _));
    Tensor tgB = local_tile(gB_global, make_tile(Int<BN>{}, Int<BK>{}), make_coord(0, _));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    auto rA = thr_mma.partition_fragment_A(tgA(_, _, 0));
    auto rB = thr_mma.partition_fragment_B(tgB(_, _, 0));
    
    Tensor gCdummy = make_tensor(make_gmem_ptr(C_partial),
                                 make_shape(M, N), make_stride(N, Int<1>{}));
    Tensor gCtile  = local_tile(gCdummy, make_tile(Int<BM>{}, Int<BN>{}), make_coord(0, 0));
    auto rC = thr_mma.partition_fragment_C(gCtile);
    auto rD = thr_mma.make_fragment_C(rC);
    clear(rD);

    G2SCopy g2s_a, g2s_b;
    auto g2s_a_thr = g2s_a.get_slice(tid);
    auto g2s_b_thr = g2s_b.get_slice(tid);
    auto tAgA = g2s_a_thr.partition_S(tgA);
    auto tAsA = g2s_a_thr.partition_D(sA);
    auto tBgB = g2s_b_thr.partition_S(tgB);
    auto tBsB = g2s_b_thr.partition_D(sB);

    auto s2r_a     = make_tiled_copy_A(S2RAtom{}, tiled_mma);
    auto s2r_b     = make_tiled_copy_B(S2RAtom{}, tiled_mma);
    auto s2r_a_thr = s2r_a.get_slice(tid);
    auto s2r_b_thr = s2r_b.get_slice(tid);
    auto sArA = s2r_a_thr.partition_S(sA);
    auto rAv  = s2r_a_thr.retile_D(rA);
    auto sBrB = s2r_b_thr.partition_S(sB);
    auto rBv  = s2r_b_thr.retile_D(rB);

    int rd = 0, wr = 0, nxt = k_start;

    #pragma unroll
    for (int s = 0; s < KSTAGE - 1; ++s) {
        if (nxt < k_end) {
            copy(g2s_a, tAgA(_, _, _, nxt), tAsA(_, _, _, wr));
            copy(g2s_b, tBgB(_, _, _, nxt), tBsB(_, _, _, wr));
            ++nxt;
            wr = (wr + 1) % KSTAGE;
        }
        cp_async_fence();
    }

    cp_async_wait<KSTAGE - 2>();
    __syncthreads();

    copy(s2r_a, sArA(_, _, 0, rd), rAv(_, _, 0));
    copy(s2r_b, sBrB(_, _, 0, rd), rBv(_, _, 0));

    #pragma unroll 1
    for (int itile = 0; itile < k_count; ++itile) {
        const int nk = size<2>(rA);

        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<KSTAGE - 2>();
                __syncthreads();
                rd = (rd + 1) % KSTAGE;
            }

            copy(s2r_a, sArA(_, _, ik_next, rd), rAv(_, _, ik_next));
            copy(s2r_b, sBrB(_, _, ik_next, rd), rBv(_, _, ik_next));

            if (ik == 0 && warp_id < 2) {
                if (nxt < k_end) {
                    copy(g2s_a, tAgA(_, _, _, nxt), tAsA(_, _, _, wr));
                    copy(g2s_b, tBgB(_, _, _, nxt), tBsB(_, _, _, wr));
                    if (warp_id == 0) {
                        ++nxt;
                        wr = (wr + 1) % KSTAGE;
                    }
                }
                cp_async_fence();
            }

            gemm(tiled_mma, rD, rA(_, _, ik), rB(_, _, ik), rD);
        }
    }

    float* Cslice = C_partial + (long long)split_id * M * N;
    Tensor gCout  = make_tensor(make_gmem_ptr(Cslice),
                                make_shape(M, N), make_stride(N, Int<1>{}));
    Tensor gCtout = local_tile(gCout, make_tile(Int<BM>{}, Int<BN>{}), make_coord(0, 0));
    auto tCgC = thr_mma.partition_C(gCtout);

    copy(rD, tCgC);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int NUM_SPLITS>
__global__ void __launch_bounds__(512)
hgemm_warp_shuffle_reduce(
    const float* __restrict__ Cpart,
    half*        __restrict__ Cout,
    int MN
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = threadIdx.x % 32;
    
    const int base = tid * 2;
    if (base >= MN) return;
    
    float acc0 = 0.f, acc1 = 0.f;
    
    #pragma unroll
    for (int s = 0; s < NUM_SPLITS; ++s) {
        const float* src = Cpart + (long long)s * MN + base;
        if (base < MN) acc0 += src[0];
        if (base + 1 < MN) acc1 += src[1];
    }
    
    if (base < MN) {
        Cout[base] = __float2half_rn(acc0);
    }
    if (base + 1 < MN) {
        Cout[base + 1] = __float2half_rn(acc1);
    }
}

static float* s_scratch   = nullptr;
static size_t s_scratch_n = 0;

static float* get_scratch(int nsplits, int M, int N) {
    const size_t need = (size_t)nsplits * M * N;
    if (need > s_scratch_n) {
        if (s_scratch) cudaFree(s_scratch);
        cudaMalloc(&s_scratch, need * sizeof(float));
        s_scratch_n = need;
    }
    return s_scratch;
}

template <int NUM_SPLITS>
static void run_optimized_splitk_hgemm(half* A, half* B_cm, half* C, int M, int N, int K) {
    static constexpr int BM = 128, BN = 64, BK = 64, KSTAGE = 5;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SLayoutA = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<KSTAGE>{})));
    using SLayoutB = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<KSTAGE>{})));
    static constexpr int kSmem =
        (cosize(SLayoutA{}) + cosize(SLayoutB{})) * sizeof(half);

    float* partial = get_scratch(NUM_SPLITS, M, N);
    const int ntile_k = K / BK;
    const int tps     = (ntile_k + NUM_SPLITS - 1) / NUM_SPLITS;

    cudaFuncSetAttribute(
        hgemm_splitk_phase1<BM, BN, BK, KSTAGE, NUM_SPLITS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);

    hgemm_splitk_phase1<BM, BN, BK, KSTAGE, NUM_SPLITS>
        <<<dim3(1, 1, NUM_SPLITS), 128, kSmem>>>(
            A, B_cm, partial, M, N, K, tps, ntile_k);

    const int MN = M * N;
    const int num_threads = 512;
    const int elements_per_thread = 2;
    const int nb = (MN + num_threads * elements_per_thread - 1) / (num_threads * elements_per_thread);
    
    hgemm_warp_shuffle_reduce<NUM_SPLITS><<<nb, num_threads>>>(partial, C, MN);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    half* A_ptr = reinterpret_cast<half*>(a.data_ptr());
    half* B_ptr = reinterpret_cast<half*>(b_col_major.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr());

    run_optimized_splitk_hgemm<16>(A_ptr, B_ptr, C_ptr, M, N, K);
}