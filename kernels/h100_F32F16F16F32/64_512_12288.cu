#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <algorithm>
#include <stdio.h>

template <int SPLIT_K>
__global__ __launch_bounds__(256, 4)
void split_k_reduce_warp_tree(
    const float* __restrict__ workspace,
    half*         __restrict__ output,
    int MN)
{
    __shared__ float smem[256 * 2];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_tid = bid * blockDim.x + tid;
    const int base = global_tid * 2;
    
    if (base >= MN) return;
    
    float acc0 = 0.f, acc1 = 0.f;
    
    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        const float* slice = workspace + (long long)s * MN;
        if (base < MN) {
            acc0 += slice[base];
            if (base + 1 < MN) {
                acc1 += slice[base + 1];
            }
        }
    }
    
    smem[tid * 2]     = acc0;
    smem[tid * 2 + 1] = acc1;
    __syncthreads();
    
    if (base < MN) {
        if (base + 1 < MN) {
            half2 result = __float22half2_rn(make_float2(smem[tid * 2], smem[tid * 2 + 1]));
            *reinterpret_cast<half2*>(output + base) = result;
        } else {
            output[base] = __float2half(smem[tid * 2]);
        }
    }
}

template <typename T, int BM, int BN, int BK, int kStage,
          typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB>
__global__ __launch_bounds__(128, 5)
void hgemm_splitk_tiled(
    const T*     __restrict__ Aptr,
    const T*     __restrict__ Bptr,
    float*       __restrict__ workspace,
    int m, int n, int k,
    int k_per_split)
{
    using namespace cute;

    extern __shared__ T shm_data[];
    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx       = threadIdx.x;
    const int ix        = blockIdx.x;
    const int iy        = blockIdx.y;
    const int split_idx = blockIdx.z;

    if (iy * BM >= m || ix * BN >= n) return;

    int k_start = split_idx * k_per_split;
    int k_count = k_per_split;

    Tensor A = make_tensor(make_gmem_ptr(Aptr + k_start),
                           make_shape(m, k_count),
                           make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr + k_start),
                           make_shape(n, k_count),
                           make_stride(k, Int<1>{}));
    Tensor D_ws = make_tensor(
        make_gmem_ptr(workspace + (long long)split_idx * m * n),
        make_shape(m, n),
        make_stride(n, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    Tensor gD = local_tile(D_ws, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));

    G2SCopyA g2s_copy_a;
    auto g2s_a_thr = g2s_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_a_thr.partition_S(gA);
    auto tAsA_copy = g2s_a_thr.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_b_thr = g2s_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_b_thr.partition_S(gB);
    auto tBsB_copy = g2s_b_thr.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(
        Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>{}, tiled_mma);
    auto s2r_a_thr = s2r_copy_a.get_slice(idx);
    auto tAsA_s    = s2r_a_thr.partition_S(sA);
    auto tCrA_view = s2r_a_thr.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(
        Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>{}, tiled_mma);
    auto s2r_b_thr = s2r_copy_b.get_slice(idx);
    auto tBsB_s    = s2r_b_thr.partition_S(sB);
    auto tCrB_view = s2r_b_thr.retile_D(tCrB);

    int itile_to_read = 0, ismem_read = 0, ismem_write = 0;
    const int ntile = k_count / BK;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();
        ++itile_to_read;
        ++ismem_write;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_copy_a, tAsA_s(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        const int nk = size<2>(tCrA);

        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_copy_a, tAsA_s(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_copy_a, tAgA_copy(_, _, _, itile_to_read),
                               tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b, tBgB_copy(_, _, _, itile_to_read),
                               tBsB_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto tCgD = thr_mma.partition_C(gD);
    
    CUTE_UNROLL
    for (int i = 0; i < size(tCrD); ++i) {
        tCgD(i) = tCrD(i);
    }
}

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC>
__global__ __launch_bounds__(128, 2)
void hgemm_standard_kernel(
    const T* __restrict__ Aptr, const T* __restrict__ Bptr, T* __restrict__ Dptr,
    int m, int n, int k)
{
    using namespace cute;
    extern __shared__ T shm_data[];
    T* Ashm = shm_data;
    T* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x, ix = blockIdx.x, iy = blockIdx.y;
    if (iy * BM >= m || ix * BN >= n) return;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_a; auto g2s_a_thr = g2s_a.get_slice(idx);
    auto tAgA = g2s_a_thr.partition_S(gA); auto tAsA_c = g2s_a_thr.partition_D(sA);
    G2SCopyB g2s_b; auto g2s_b_thr = g2s_b.get_slice(idx);
    auto tBgB = g2s_b_thr.partition_S(gB); auto tBsB_c = g2s_b_thr.partition_D(sB);

    auto s2r_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_a_thr = s2r_a.get_slice(idx);
    auto tAsA_s = s2r_a_thr.partition_S(sA); auto tCrA_v = s2r_a_thr.retile_D(tCrA);
    auto s2r_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_b_thr = s2r_b.get_slice(idx);
    auto tBsB_s = s2r_b_thr.partition_S(sB); auto tCrB_v = s2r_b_thr.retile_D(tCrB);

    int itile_to_read = 0, ismem_read = 0, ismem_write = 0;
    int ntile = k / BK;
    #pragma unroll
    for (int is = 0; is < kStage - 1; ++is) {
        cute::copy(g2s_a, tAgA(_, _, _, is), tAsA_c(_, _, _, is));
        cute::copy(g2s_b, tBgB(_, _, _, is), tBsB_c(_, _, _, is));
        cp_async_fence(); ++itile_to_read; ++ismem_write;
    }
    cp_async_wait<kStage - 2>(); __syncthreads();
    cute::copy(s2r_a, tAsA_s(_, _, 0, ismem_read), tCrA_v(_, _, 0));
    cute::copy(s2r_b, tBsB_s(_, _, 0, ismem_read), tCrB_v(_, _, 0));

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);
        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;
            if (ik == nk - 1) { cp_async_wait<kStage - 2>(); __syncthreads(); ismem_read = (ismem_read + 1) % kStage; }
            cute::copy(s2r_a, tAsA_s(_, _, ik_next, ismem_read), tCrA_v(_, _, ik_next));
            cute::copy(s2r_b, tBsB_s(_, _, ik_next, ismem_read), tCrB_v(_, _, ik_next));
            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_a, tAgA(_, _, _, itile_to_read), tAsA_c(_, _, _, ismem_write));
                    cute::copy(g2s_b, tBgB(_, _, _, itile_to_read), tBsB_c(_, _, _, ismem_write));
                    ++itile_to_read; ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    auto tCrD_half = make_tensor_like<T>(tCrD);
    cute::copy(tCrD, tCrD_half);
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
    auto r2s = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr = r2s.get_slice(idx);
    auto tCrC_r2s = r2s_thr.retile_S(tCrD_half);
    auto tCsC_r2s = r2s_thr.partition_D(sC);
    S2GCopyC s2g; auto s2g_thr = s2g.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr.partition_S(sC);
    auto tCgC_s2g = s2g_thr.partition_D(gD);
    auto tCgC_x = group_modes<1,3>(tCgC_s2g);
    auto tCrC_x = group_modes<1,3>(tCrC_r2s);
    int step = size<3>(tCsC_r2s);
    #pragma unroll
    for (int i = 0; i < size<1>(tCrC_x); i += step) {
        #pragma unroll
        for (int j = 0; j < step; ++j) {
            auto t = make_tensor_like<T>(tCrC_x(_, i+j));
            cute::copy(tCrC_x(_, i+j), t);
            cute::copy(r2s, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < step; ++j)
            cute::copy(s2g, tCsC_s2g(_, 0, 0, j), tCgC_x(_, i+j));
        __syncthreads();
    }
}

template <int SPLIT_K>
void run_splitk_tiled(
    const half* a, const half* b_tn, half* c,
    int M, int N, int K, float* workspace)
{
    using namespace cute;
    using T = half;

    static constexpr int BM = 64, BN = 64, BK = 64, kStage = 4;

    using SmemAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemAtom{},
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})));

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using atom_sh = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1*2*get<0>(atom_sh{});
    static constexpr int kMmaPN = 2*2*get<1>(atom_sh{});
    static constexpr int kMmaPK = 1*1*get<2>(atom_sh{});
    using RepT = decltype(make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})));
    using PT   = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA  = decltype(make_tiled_mma(mma_atom{}, RepT{}, PT{}));

    using g2s_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>, T>;
    using G2SCopyA = decltype(make_tiled_copy(g2s_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    const int BX = (N + BN - 1) / BN;
    const int BY = (M + BM - 1) / BM;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY, SPLIT_K);

    static constexpr int shm_size =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * (int)sizeof(T);

    using KernelT = decltype(&hgemm_splitk_tiled<
        T, BM, BN, BK, kStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB>);
    KernelT kfn = hgemm_splitk_tiled<
        T, BM, BN, BK, kStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB>;

    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(kfn, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    const int k_per_split = K / SPLIT_K;
    kfn<<<grid, block, shm_size>>>(a, b_tn, workspace, M, N, K, k_per_split);

    const int MN = M * N;
    const int red_threads = 256;
    const int red_blocks = (MN / 2 + red_threads - 1) / red_threads;
    split_k_reduce_warp_tree<SPLIT_K><<<red_blocks, red_threads>>>(workspace, c, MN);
}

template <typename T, int Stages = 3>
void run_standard_hgemm(const T* a, const T* b_tn, T* c, int M, int N, int K) {
    using namespace cute;
    static constexpr int BM = 128, BN = 128, BK = 32;
    auto kKStage = Int<Stages>{}; auto kSCBatch = Int<4>{};

    using SmemAtom = decltype(composition(Swizzle<3,3,3>{},
        make_layout(make_shape(Int<8>{},Int<BK>{}),make_stride(Int<BK>{},Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemAtom{}, make_shape(Int<BM>{},Int<BK>{},kKStage)));
    using SmemLayoutB = decltype(tile_to_shape(SmemAtom{}, make_shape(Int<BN>{},Int<BK>{},kKStage)));

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>; using mma_atom = MMA_Atom<mma_traits>;
    using ash = mma_traits::Shape_MNK;
    static constexpr int PM=1*2*get<0>(ash{}), PN=2*2*get<1>(ash{}), PK=1*1*get<2>(ash{});
    using RepT = decltype(make_layout(make_shape(Int<2>{},Int<2>{},Int<1>{})));
    using PT = Tile<Int<PM>,Int<PN>,Int<PK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{},RepT{},PT{}));

    using g2s_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>,T>;
    using G2SCopyA = decltype(make_tiled_copy(g2s_atom{},
        make_layout(make_shape(Int<32>{},Int<4>{}),make_stride(Int<4>{},Int<1>{})),
        make_layout(make_shape(Int<1>{},Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    using s2r_atom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>,T>;
    using S2RAtomA = s2r_atom; using S2RAtomB = s2r_atom;

    using SmemAtomC = decltype(composition(Swizzle<3,3,3>{},
        make_layout(make_shape(Int<PM>{},Int<PN>{}),make_stride(Int<PN>{},Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(SmemAtomC{},make_shape(Int<PM>{},Int<PN>{},kSCBatch)));

    using R2SAtomC = Copy_Atom<UniversalCopy<int>,T>;
    using S2GAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>,T>;
    using S2GCopyC = decltype(make_tiled_copy(S2GAtomC{},
        make_layout(make_shape(Int<32>{},Int<4>{}),make_stride(Int<4>{},Int<1>{})),
        make_layout(make_shape(Int<1>{},Int<8>{}))));

    int BX=(N+BN-1)/BN, BY=(M+BM-1)/BM;
    dim3 block(size(MMA{})), grid(BX,BY,1);
    static constexpr int smAB=(cute::cosize(SmemLayoutA{})+cute::cosize(SmemLayoutB{}))*sizeof(T);
    static constexpr int smC=cute::cosize(SmemLayoutC{})*sizeof(T);
    int sm = max(smAB,smC);

    using KT = decltype(&hgemm_standard_kernel<T,BM,BN,BK,Stages,MMA,G2SCopyA,G2SCopyB,
        SmemLayoutA,SmemLayoutB,SmemLayoutC,S2RAtomA,S2RAtomB,R2SAtomC,S2GAtomC,S2GCopyC>);
    KT kfn = hgemm_standard_kernel<T,BM,BN,BK,Stages,MMA,G2SCopyA,G2SCopyB,
        SmemLayoutA,SmemLayoutB,SmemLayoutC,S2RAtomA,S2RAtomB,R2SAtomC,S2GAtomC,S2GCopyC>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    kfn<<<grid,block,sm>>>(a, b_tn, c, M, N, K);
}

static float* g_workspace_ptr  = nullptr;
static size_t g_workspace_size = 0;

static float* get_workspace(size_t bytes) {
    if (bytes > g_workspace_size) {
        if (g_workspace_ptr) cudaFree(g_workspace_ptr);
        cudaMalloc(&g_workspace_ptr, bytes);
        g_workspace_size = bytes;
    }
    return g_workspace_ptr;
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* a_ptr    = reinterpret_cast<const half*>(a.data_ptr());
    const half* b_tn_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* c_ptr          = reinterpret_cast<half*>(c.data_ptr());

    if (M == 64 && N == 512 && K == 12288) {
        constexpr int SPLIT_K = 16;
        float* ws = get_workspace((size_t)SPLIT_K * M * N * sizeof(float));
        run_splitk_tiled<SPLIT_K>(a_ptr, b_tn_ptr, c_ptr, M, N, K, ws);
    } else {
        run_standard_hgemm<half>(a_ptr, b_tn_ptr, c_ptr, M, N, K);
    }
}