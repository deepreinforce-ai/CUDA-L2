#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>

using namespace nvcuda::wmma;

static constexpr int SPLIT_K     = 96;
static constexpr int K_CHUNK     = 128;
static constexpr int BM          = 64;
static constexpr int BN          = 128;
static constexpr int BK          = 64;
static constexpr int STAGES      = 2;
static constexpr int SA_STRIDE   = 72;
static constexpr int SB_STRIDE   = 72;

static constexpr int SMEM_A_SIZE = STAGES * BM * SA_STRIDE;
static constexpr int SMEM_B_SIZE = STAGES * BN * SB_STRIDE;
static constexpr int SMEM_BYTES  = (SMEM_A_SIZE + SMEM_B_SIZE) * (int)sizeof(half);

__device__ __forceinline__
void cp_async_ca_16(void* __restrict__ smem_ptr, const void* __restrict__ gmem_ptr)
{
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__global__ void __launch_bounds__(256, 2)
gemm_warp_specialized(
    const half*  __restrict__ A,
    const half*  __restrict__ B_col,
    float*       __restrict__ workspace,
    int M, int N, int K
)
{
    const int split_id = blockIdx.z;
    const int k_start  = split_id * K_CHUNK;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int warp_row = warp_id >> 2;
    const int warp_col = warp_id & 3;

    fragment<accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    extern __shared__ half smem[];
    half (*sA)[BM][SA_STRIDE] = reinterpret_cast<half(*)[BM][SA_STRIDE]>(smem);
    half (*sB)[BN][SB_STRIDE] = reinterpret_cast<half(*)[BN][SB_STRIDE]>(
                                     smem + SMEM_A_SIZE);

    auto load_tile_specialized = [&](int stage, int k_off) __attribute__((always_inline)) {
        if (warp_id < 4) {
            const int lane = tid & 31;
            const int local_warp = warp_id;
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int linear = local_warp * 32 + lane + i * 128;
                const int row = linear >> 3;
                const int col8 = (linear & 7) << 3;
                
                if (row < BM) {
                    half* dst = &sA[stage][row][col8];
                    const half* src = A + (long long)row * K + k_off + col8;
                    cp_async_ca_16(dst, src);
                }
            }
        } else {
            const int lane = tid & 31;
            const int local_warp = warp_id - 4;
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int linear = local_warp * 32 + lane + i * 128;
                const int row = linear >> 3;
                const int col8 = (linear & 7) << 3;
                
                if (row < BN) {
                    half* dst = &sB[stage][row][col8];
                    const half* src = B_col + (long long)row * K + k_off + col8;
                    cp_async_ca_16(dst, src);
                }
            }
        }
    };

    load_tile_specialized(0, k_start);
    cp_async_commit();

    #pragma unroll
    for (int k_tile = 0; k_tile < 2; k_tile++) {
        const int s_read  = k_tile & 1;
        const int s_write = s_read ^ 1;

        cp_async_wait_all();
        __syncthreads();

        if (k_tile + 1 < 2) {
            load_tile_specialized(s_write, k_start + (k_tile + 1) * BK);
            cp_async_commit();
        }

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k_inner = ki * 16;

            fragment<matrix_a, 16, 16, 16, half, row_major> fA[2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                load_matrix_sync(fA[mi],
                    &sA[s_read][warp_row * 32 + mi * 16][k_inner],
                    SA_STRIDE);
            }

            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                fragment<matrix_b, 16, 16, 16, half, col_major> fB;
                load_matrix_sync(fB,
                    &sB[s_read][warp_col * 32 + ni * 16][k_inner],
                    SB_STRIDE);
                
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    mma_sync(acc[mi][ni], fA[mi], fB, acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    float* out = workspace + (long long)split_id * M * N;
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        const int m_base = warp_row * 32 + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const int n_base = warp_col * 32 + ni * 16;
            store_matrix_sync(out + m_base * N + n_base, acc[mi][ni], N, mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(128)
reduce_optimized(
    const float* __restrict__ workspace,
    half*        __restrict__ C,
    int total
)
{
    const int base = blockIdx.x * 4;
    if (base >= total) return;

    const int tid = threadIdx.x;

    float4 p = {0.0f, 0.0f, 0.0f, 0.0f};
    if (tid < SPLIT_K) {
        const float* src = workspace + (long long)tid * total + base;
        if (base + 3 < total) {
            p = __ldg(reinterpret_cast<const float4*>(src));
        } else {
            if (base     < total) p.x = __ldg(src);
            if (base + 1 < total) p.y = __ldg(src + 1);
            if (base + 2 < total) p.z = __ldg(src + 2);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        p.x += __shfl_down_sync(0xffffffff, p.x, off);
        p.y += __shfl_down_sync(0xffffffff, p.y, off);
        p.z += __shfl_down_sync(0xffffffff, p.z, off);
        p.w += __shfl_down_sync(0xffffffff, p.w, off);
    }

    __shared__ float4 warp_sum[4];
    const int warp_id   = tid >> 5;
    const int warp_lane = tid & 31;

    if (warp_lane == 0) {
        warp_sum[warp_id] = p;
    }
    __syncthreads();

    if (tid == 0) {
        float4 r;
        r.x = warp_sum[0].x + warp_sum[1].x + warp_sum[2].x + warp_sum[3].x;
        r.y = warp_sum[0].y + warp_sum[1].y + warp_sum[2].y + warp_sum[3].y;
        r.z = warp_sum[0].z + warp_sum[1].z + warp_sum[2].z + warp_sum[3].z;
        r.w = warp_sum[0].w + warp_sum[1].w + warp_sum[2].w + warp_sum[3].w;

        if (base + 3 < total) {
            *reinterpret_cast<half2*>(C + base) =
                __float22half2_rn(make_float2(r.x, r.y));
            *reinterpret_cast<half2*>(C + base + 2) =
                __float22half2_rn(make_float2(r.z, r.w));
        } else {
            if (base     < total) C[base]     = __float2half(r.x);
            if (base + 1 < total) C[base + 1] = __float2half(r.y);
            if (base + 2 < total) C[base + 2] = __float2half(r.z);
        }
    }
}

static float*  g_workspace    = nullptr;
static size_t  g_workspace_sz = 0;
static bool    g_smem_set     = false;

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c
)
{
    const int M     = (int)a.size(0);
    const int K     = (int)a.size(1);
    const int N     = (int)b.size(1);
    const int total = M * N;

    const size_t needed = (size_t)SPLIT_K * total * sizeof(float);
    if (g_workspace_sz < needed) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, needed);
        g_workspace_sz = needed;
    }

    if (!g_smem_set) {
        cudaFuncSetAttribute(
            gemm_warp_specialized,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_BYTES
        );
        g_smem_set = true;
    }

    const half* Aptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* Bptr = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       Cptr = reinterpret_cast<half*>(c.data_ptr());

    gemm_warp_specialized<<<dim3(1, 1, SPLIT_K), 256, SMEM_BYTES>>>(
        Aptr, Bptr, g_workspace, M, N, K
    );

    reduce_optimized<<<(total + 3) / 4, 128>>>(
        g_workspace, Cptr, total
    );
}