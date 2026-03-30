#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <ATen/cuda/CUDAContext.h>

using namespace cute;

static constexpr int kBM     = 64;
static constexpr int kBN     = 64;
static constexpr int kBK     = 128;
static constexpr int kStages = 4;
static constexpr int kSplits = 16;

using SmemAtomAB = decltype(composition(
    Swizzle<3, 4, 3>{},
    make_layout(make_shape(Int<8>{}, Int<kBK>{}),
                make_stride(Int<kBK>{}, Int<1>{}))
));
using SmemLayoutA = decltype(tile_to_shape(
    SmemAtomAB{},
    make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{})
));
using SmemLayoutB = decltype(tile_to_shape(
    SmemAtomAB{},
    make_shape(Int<kBN>{}, Int<kBK>{}, Int<kStages>{})
));

using mma_atom_t   = MMA_Atom<MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>>;
using MMA_EURepeat = decltype(make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})));
using MMA_Pertile  = decltype(make_tile(Int<32>{}, Int<32>{}, Int<16>{}));
using TiledMMA_t   = decltype(make_tiled_mma(mma_atom_t{}, MMA_EURepeat{}, MMA_Pertile{}));

using g2s_cacheall_op = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
using g2s_cacheglo_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

using G2SCopyA = decltype(make_tiled_copy(
    Copy_Atom<Copy_Traits<g2s_cacheall_op>, half>{},
    make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
    make_layout(make_shape(Int<1>{}, Int<8>{}))
));
using G2SCopyB = decltype(make_tiled_copy(
    Copy_Atom<Copy_Traits<g2s_cacheglo_op>, half>{},
    make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
    make_layout(make_shape(Int<1>{}, Int<8>{}))
));

using s2r_atom_t = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, half>;

static constexpr int kSmemABElems = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
static constexpr int kSmemBytes   = kSmemABElems * (int)sizeof(half);

__global__ void __launch_bounds__(128, 2)
hgemm_optimized_split_kernel(
    const half* __restrict__ Aptr,
    const half* __restrict__ Bptr,
    float*      __restrict__ partial_out,
    int M, int N, int K,
    int K_per_split)
{
    extern __shared__ half shm_data[];
    half* Ashm = shm_data;
    half* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    const int idx     = threadIdx.x;
    const int warp_id = idx / 32;
    const int ix      = blockIdx.x;
    const int iy      = blockIdx.y;
    const int iz      = blockIdx.z;

    const int K_start  = iz * K_per_split;
    const int K_actual = K_per_split;
    const int ntile    = K_actual / kBK;

    if (ntile <= 0 || iy * kBM >= M || ix * kBN >= N) return;

    float* my_out = partial_out + (size_t)iz * M * N;

    Tensor A = make_tensor(
        make_gmem_ptr(Aptr + K_start),
        make_shape(M, K_actual),
        make_stride(K, Int<1>{})
    );
    Tensor B = make_tensor(
        make_gmem_ptr(Bptr + K_start),
        make_shape(N, K_actual),
        make_stride(K, Int<1>{})
    );
    Tensor D = make_tensor(
        make_gmem_ptr(my_out),
        make_shape(M, N),
        make_stride(N, Int<1>{})
    );

    Tensor gA = local_tile(A, make_tile(Int<kBM>{}, Int<kBK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kBN>{}, Int<kBK>{}), make_coord(ix, _));
    Tensor gD = local_tile(D, make_tile(Int<kBM>{}, Int<kBN>{}), make_coord(iy, ix));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gD);
    clear(tCrD);

    G2SCopyA g2s_copy_a;
    auto g2s_thr_a = g2s_copy_a.get_slice(idx);
    auto tAgA      = g2s_thr_a.partition_S(gA);
    auto tAsA_g2s  = g2s_thr_a.partition_D(sA);

    G2SCopyB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(idx);
    auto tBgB      = g2s_thr_b.partition_S(gB);
    auto tBsB_g2s  = g2s_thr_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(s2r_atom_t{}, tiled_mma);
    auto s2r_thr_a  = s2r_copy_a.get_slice(idx);
    auto tAsA_s2r   = s2r_thr_a.partition_S(sA);
    auto tCrA_view  = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(s2r_atom_t{}, tiled_mma);
    auto s2r_thr_b  = s2r_copy_b.get_slice(idx);
    auto tBsB_s2r   = s2r_thr_b.partition_S(sB);
    auto tCrB_view  = s2r_thr_b.retile_D(tCrB);

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;

    const bool is_memory_warp  = (warp_id == 3);

    if (0 < ntile) {
        cute::copy(g2s_copy_a, tAgA(_, _, _, 0), tAsA_g2s(_, _, _, 0));
        cute::copy(g2s_copy_b, tBgB(_, _, _, 0), tBsB_g2s(_, _, _, 0));
        ++itile_to_read;
        ++ismem_write;
    }
    cp_async_fence();

#pragma unroll
    for (int istage = 1; istage < kStages - 1; ++istage) {
        if (istage < ntile) {
            cute::copy(g2s_copy_a, tAgA(_, _, _, istage), tAsA_g2s(_, _, _, istage));
            cute::copy(g2s_copy_b, tBgB(_, _, _, istage), tBsB_g2s(_, _, _, istage));
            ++itile_to_read;
            ++ismem_write;
        }
        cp_async_fence();
    }

    cp_async_wait<kStages - 2>();
    __syncthreads();

    const int nk = size<2>(tCrA);

    cute::copy(s2r_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB_view(_, _, 0));

#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            const int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStages - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStages;
            }

            cute::copy(s2r_copy_a, tAsA_s2r(_, _, ik_next, ismem_read),
                       tCrA_view(_, _, ik_next));
            cute::copy(s2r_copy_b, tBsB_s2r(_, _, ik_next, ismem_read),
                       tCrB_view(_, _, ik_next));

            const int issue_point = is_memory_warp ? 0 : 2;
            if (ik == issue_point) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_copy_a, tAgA(_, _, _, itile_to_read),
                               tAsA_g2s(_, _, _, ismem_write));
                    cute::copy(g2s_copy_b, tBgB(_, _, _, itile_to_read),
                               tBsB_g2s(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStages;
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

__global__ void __launch_bounds__(256, 4)
split_partial_reduce(
    const float* __restrict__ partial_out,
    __half*      __restrict__ dst,
    int MN, int splits)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = tid * 8;

    if (base + 7 < MN) {
        float4 acc0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1 = {0.f, 0.f, 0.f, 0.f};

        const float4* base_ptr = reinterpret_cast<const float4*>(partial_out) + tid * 2;
        
        const int splits_div4 = splits / 4;
        const int splits_rem = splits % 4;
        
#pragma unroll 4
        for (int s4 = 0; s4 < splits_div4; ++s4) {
            const int s_base = s4 * 4;
            
            const float4* ptr0 = base_ptr + (size_t)(s_base + 0) * MN / 4;
            const float4* ptr1 = base_ptr + (size_t)(s_base + 1) * MN / 4;
            const float4* ptr2 = base_ptr + (size_t)(s_base + 2) * MN / 4;
            const float4* ptr3 = base_ptr + (size_t)(s_base + 3) * MN / 4;
            
            const float4 v00 = __ldg(ptr0);
            const float4 v01 = __ldg(ptr0 + 1);
            const float4 v10 = __ldg(ptr1);
            const float4 v11 = __ldg(ptr1 + 1);
            const float4 v20 = __ldg(ptr2);
            const float4 v21 = __ldg(ptr2 + 1);
            const float4 v30 = __ldg(ptr3);
            const float4 v31 = __ldg(ptr3 + 1);
            
            acc0.x += v00.x + v10.x + v20.x + v30.x;
            acc0.y += v00.y + v10.y + v20.y + v30.y;
            acc0.z += v00.z + v10.z + v20.z + v30.z;
            acc0.w += v00.w + v10.w + v20.w + v30.w;
            acc1.x += v01.x + v11.x + v21.x + v31.x;
            acc1.y += v01.y + v11.y + v21.y + v31.y;
            acc1.z += v01.z + v11.z + v21.z + v31.z;
            acc1.w += v01.w + v11.w + v21.w + v31.w;
        }
        
        for (int s = splits_div4 * 4; s < splits; ++s) {
            const float4* curr_ptr = base_ptr + (size_t)s * MN / 4;
            const float4 v0 = __ldg(curr_ptr);
            const float4 v1 = __ldg(curr_ptr + 1);
            
            acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
            acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
        }

        const __half2 h0 = __float22half2_rn(make_float2(acc0.x, acc0.y));
        const __half2 h1 = __float22half2_rn(make_float2(acc0.z, acc0.w));
        const __half2 h2 = __float22half2_rn(make_float2(acc1.x, acc1.y));
        const __half2 h3 = __float22half2_rn(make_float2(acc1.z, acc1.w));

        int4 out;
        out.x = *reinterpret_cast<const int*>(&h0);
        out.y = *reinterpret_cast<const int*>(&h1);
        out.z = *reinterpret_cast<const int*>(&h2);
        out.w = *reinterpret_cast<const int*>(&h3);
        reinterpret_cast<int4*>(dst)[tid] = out;

    } else {
        for (int i = base; i < MN && i < base + 8; ++i) {
            float acc = 0.f;
#pragma unroll 16
            for (int s = 0; s < splits; ++s)
                acc += __ldg(partial_out + (size_t)s * MN + i);
            dst[i] = __float2half(acc);
        }
    }
}

static float* g_partial_buf   = nullptr;
static size_t g_partial_elems = 0;

static float* get_partial_buf(size_t elems)
{
    if (elems > g_partial_elems) {
        if (g_partial_buf) cudaFree(g_partial_buf);
        cudaMalloc(&g_partial_buf, elems * sizeof(float));
        g_partial_elems = elems;
    }
    return g_partial_buf;
}

static bool g_initialized = false;

static void init_kernels()
{
    if (!g_initialized) {
        cudaFuncSetAttribute(
            hgemm_optimized_split_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSmemBytes
        );
        cudaFuncSetAttribute(
            hgemm_optimized_split_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared
        );
        cudaFuncSetCacheConfig(split_partial_reduce, cudaFuncCachePreferL1);
        g_initialized = true;
    }
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                            \
    if ((T).options().dtype() != (th_type)) {                           \
        std::cout << "Tensor Info:" << (T).options() << std::endl;      \
        throw std::runtime_error("values must be " #th_type);           \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                             \
    if ((T).size(0) != (S0) || (T).size(1) != (S1)) {                  \
        throw std::runtime_error("Tensor size mismatch!");              \
    }

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b,           torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c,           torch::kHalf)

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    init_kernels();

    const int BX          = (N + kBN - 1) / kBN;
    const int BY          = (M + kBM - 1) / kBM;
    const int K_per_split = K / kSplits;

    const size_t partial_elems = (size_t)kSplits * M * N;
    float* partial_buf = get_partial_buf(partial_elems);

    dim3 block_dim(size(TiledMMA_t{}));
    dim3 grid_dim(BX, BY, kSplits);

    hgemm_optimized_split_kernel<<<grid_dim, block_dim, kSmemBytes, stream>>>(
        A_ptr, B_ptr, partial_buf,
        M, N, K, K_per_split
    );

    const int MN            = M * N;
    const int reduce_thr    = 256;
    const int reduce_blocks = (MN / 8 + reduce_thr - 1) / reduce_thr;

    split_partial_reduce<<<reduce_blocks, reduce_thr, 0, stream>>>(
        partial_buf, reinterpret_cast<__half*>(C_ptr), MN, kSplits
    );
}