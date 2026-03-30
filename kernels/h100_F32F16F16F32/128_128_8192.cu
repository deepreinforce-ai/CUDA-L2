#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <algorithm>
#include <stdio.h>

namespace cg = cooperative_groups;

#define BM             128
#define BN             128
#define BK             64
#define MMA_M          16
#define MMA_N          8
#define MMA_K          16
#define BLOCK_THREADS  256
#define NUM_WARPS      8
#define WARP_SIZE      32
#define WARP_MMA_N     16

#define NUM_SPLITS     64
#define K_PER_SPLIT    128
#define K_TILES        2

#define SMA_STRIDE     72
#define SMB_STRIDE     136
#define SMA_STAGE_H    (BM * SMA_STRIDE)
#define SMB_STAGE_H    (BK * SMB_STRIDE)

#define SWIZZLE_A(row, col) ((col) ^ (((row) & 3) << 3))

#define CP_ASYNC_A(dst, src) \
    asm volatile( \
        "cp.async.ca.shared.global [%0], [%1], 16;\n" \
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)), \
           "l"((const void*)(src)))

#define CP_ASYNC_B(dst, src) \
    asm volatile( \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" \
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)), \
           "l"((const void*)(src)))

#define CP_ASYNC_COMMIT() \
    asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT(n) \
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n) : "memory")

__device__ __forceinline__
void mma_m16n8k16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

static float*  g_partials   = nullptr;
static size_t  g_partial_sz = 0;

static void ensure_partials(size_t sz) {
    if (g_partial_sz < sz) {
        if (g_partials) cudaFree(g_partials);
        cudaMalloc(&g_partials, sz);
        g_partial_sz = sz;
    }
}

__device__ __forceinline__ void load_smA_async(
    __half* __restrict__ dst,
    const __half* __restrict__ A,
    int K, int k_off)
{
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int c   = threadIdx.x + i * BLOCK_THREADS;
        int row = c >> 3;
        int col = (c & 7) << 3;
        int swz = SWIZZLE_A(row, col);
        CP_ASYNC_A(dst + row * SMA_STRIDE + swz,
                   A + row * K + k_off + col);
    }
}

__device__ __forceinline__ void load_smB_async(
    __half* __restrict__ dst,
    const __half* __restrict__ B,
    int N, int k_off)
{
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int c   = threadIdx.x + i * BLOCK_THREADS;
        int row = c >> 4;
        int col = (c & 15) << 3;
        CP_ASYNC_B(dst + row * SMB_STRIDE + col,
                   B + (k_off + row) * N + col);
    }
}

__device__ __forceinline__
void load_af(uint32_t af[4], const __half* sA, int warp_m_base, int lane_id, int ki)
{
    int a_row = warp_m_base + (lane_id & 15);
    int a_col = ki * MMA_K + ((lane_id >> 4) << 3);
    int a_swz = SWIZZLE_A(a_row, a_col);
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
        sA + a_row * SMA_STRIDE + a_swz);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
        : "r"(addr)
    );
}

__device__ __forceinline__
void load_bf(uint32_t bf[2], const __half* sB, int lane_id, int ki, int ni)
{
    int b_k_row = ki * MMA_K + (lane_id & 15);
    int b_n_col = ni * MMA_N;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
        sB + b_k_row * SMB_STRIDE + b_n_col);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(bf[0]), "=r"(bf[1])
        : "r"(addr)
    );
}

__device__ __forceinline__
void compute_tile(
    float acc[WARP_MMA_N][4],
    const __half* sA,
    const __half* sB,
    int warp_m_base,
    int lane_id)
{
    const int num_ki = BK / MMA_K;

    uint32_t af_cur[4], af_nxt[4];
    load_af(af_cur, sA, warp_m_base, lane_id, 0);

    #pragma unroll
    for (int ki = 0; ki < num_ki; ki++) {
        if (ki + 1 < num_ki) {
            load_af(af_nxt, sA, warp_m_base, lane_id, ki + 1);
        }

        uint32_t bf_cur[2], bf_nxt[2];
        load_bf(bf_cur, sB, lane_id, ki, 0);

        #pragma unroll
        for (int ni = 0; ni < WARP_MMA_N; ni++) {
            if (ni + 1 < WARP_MMA_N) {
                load_bf(bf_nxt, sB, lane_id, ki, ni + 1);
            }
            mma_m16n8k16_f32(
                acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                af_cur[0], af_cur[1], af_cur[2], af_cur[3],
                bf_cur[0], bf_cur[1],
                acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]
            );
            bf_cur[0] = bf_nxt[0];
            bf_cur[1] = bf_nxt[1];
        }

        af_cur[0] = af_nxt[0];
        af_cur[1] = af_nxt[1];
        af_cur[2] = af_nxt[2];
        af_cur[3] = af_nxt[3];
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_coop_split_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ partials,
    __half*       __restrict__ C,
    int M, int N, int K,
    int num_splits)
{
    extern __shared__ __half smem[];

    __half* smA[2];
    __half* smB[2];
    smA[0] = smem;
    smA[1] = smem + SMA_STAGE_H;
    smB[0] = smem + 2 * SMA_STAGE_H;
    smB[1] = smem + 2 * SMA_STAGE_H + SMB_STAGE_H;

    const int split_id    = blockIdx.x;
    const int warp_id     = threadIdx.x >> 5;
    const int lane_id     = threadIdx.x & 31;
    const int warp_m_base = warp_id * MMA_M;

    float acc[WARP_MMA_N][4];
    #pragma unroll
    for (int ni = 0; ni < WARP_MMA_N; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    const int k_start = split_id * K_PER_SPLIT;

    load_smA_async(smA[0], A, K, k_start);
    load_smB_async(smB[0], B, N, k_start);
    CP_ASYNC_COMMIT();

    load_smA_async(smA[1], A, K, k_start + BK);
    load_smB_async(smB[1], B, N, k_start + BK);
    CP_ASYNC_COMMIT();

    CP_ASYNC_WAIT(1);
    __syncthreads();

    compute_tile(acc, smA[0], smB[0], warp_m_base, lane_id);

    CP_ASYNC_WAIT(0);
    __syncthreads();

    compute_tile(acc, smA[1], smB[1], warp_m_base, lane_id);

    const int r0  = warp_m_base + (lane_id >> 2);
    const int r1  = r0 + 8;
    const int dc  = (lane_id & 3) << 1;
    const long long partial_row_stride = (long long)num_splits * N;

    #pragma unroll
    for (int ni = 0; ni < WARP_MMA_N; ni++) {
        const int col = ni * MMA_N + dc;
        float* p0 = partials + (long long)r0 * partial_row_stride + split_id * N + col;
        float* p1 = partials + (long long)r1 * partial_row_stride + split_id * N + col;
        *reinterpret_cast<float2*>(p0) = make_float2(acc[ni][0], acc[ni][1]);
        *reinterpret_cast<float2*>(p1) = make_float2(acc[ni][2], acc[ni][3]);
    }

    cg::this_grid().sync();

    float* warp_sums = reinterpret_cast<float*>(smem);

    const int splits_per_warp = num_splits / NUM_WARPS;
    const int sp_start        = warp_id * splits_per_warp;
    const int sp_end          = sp_start + splits_per_warp;
    const int base_col        = lane_id * 4;

    #pragma unroll
    for (int row_idx = 0; row_idx < 2; row_idx++) {
        const int out_row = split_id * 2 + row_idx;
        if (out_row >= M) break;

        const float* row_base = partials + (long long)out_row * partial_row_stride;

        float4 racc = make_float4(0.f, 0.f, 0.f, 0.f);

        #pragma unroll 4
        for (int sp = sp_start; sp < sp_end; sp++) {
            const float* ptr = row_base + (long long)sp * N + base_col;
            float4 v;
            asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                : "l"(ptr));
            racc.x += v.x;
            racc.y += v.y;
            racc.z += v.z;
            racc.w += v.w;
        }

        float* ws = warp_sums + warp_id * BN + base_col;
        ws[0] = racc.x;
        ws[1] = racc.y;
        ws[2] = racc.z;
        ws[3] = racc.w;
        __syncthreads();

        const int out_col_base = warp_id * (BN / NUM_WARPS);

        if (lane_id < BN / NUM_WARPS) {
            const int out_col = out_col_base + lane_id;
            float final_sum = 0.f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                final_sum += warp_sums[w * BN + out_col];
            }
            C[out_row * BN + out_col] = __float2half(final_sum);
        }

        if (row_idx == 0) __syncthreads();
    }
}

static bool g_attr_configured = false;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* hA = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* hB = reinterpret_cast<const __half*>(b.data_ptr());
    __half*       hC = reinterpret_cast<__half*>(c.data_ptr());

    const int num_splits = NUM_SPLITS;

    const size_t partial_elems = (size_t)M * num_splits * N;
    const size_t partial_bytes = partial_elems * sizeof(float);
    ensure_partials(partial_bytes);

    const size_t smem_bytes =
        (size_t)(2 * SMA_STAGE_H + 2 * SMB_STAGE_H) * sizeof(__half);

    if (!g_attr_configured) {
        cudaFuncSetAttribute(
            hgemm_coop_split_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            98304);
        g_attr_configured = true;
    }

    int iM = M, iN = N, iK = K, iSplits = num_splits;
    void* args[] = {
        (void*)&hA,
        (void*)&hB,
        (void*)&g_partials,
        (void*)&hC,
        (void*)&iM,
        (void*)&iN,
        (void*)&iK,
        (void*)&iSplits
    };

    cudaLaunchCooperativeKernel(
        (void*)hgemm_coop_split_kernel,
        dim3(num_splits),
        dim3(BLOCK_THREADS),
        args,
        smem_bytes,
        nullptr
    );
}