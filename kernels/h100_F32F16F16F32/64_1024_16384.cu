#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <float.h>
#include <stdlib.h>

struct __align__(8) smem_desc_t {
    uint64_t desc;
};

__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_ptr, int stride_dim) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= (uint64_t)(addr >> 4) & 0x3FFF;
    desc |= (uint64_t)(stride_dim) << 16;
    desc |= (uint64_t)(stride_dim) << 32;
    return desc;
}

__device__ __forceinline__ uint64_t make_wgmma_desc(uint32_t smem_addr, uint32_t stride_8byte, uint32_t ld_8byte, int swizzle) {
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_addr >> 4) & 0x3FFF);
    desc |= ((uint64_t)swizzle << 62);
    return desc;
}

#define BM 64
#define BN 128
#define BK 64
#define PAD_A 8
#define PAD_B 8
#define STAGES 3
#define NUM_SPLITS 16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M 4
#define WARP_N 2

__device__ __forceinline__ void mma_f32(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
          "r"(b[0]),"r"(b[1]),
          "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    uint32_t addr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(addr), "l"(src) : "memory");
}

__global__ __launch_bounds__(256, 1)
void hgemm_splitk_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C_split,
    int M, int N, int K, int K_per_split)
{
    const int bn_tile   = blockIdx.x;
    const int split     = blockIdx.z;
    const int bn_start  = bn_tile * BN;
    const int k_start   = split * K_per_split;
    const int num_iters = K_per_split / BK;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    float acc[WARP_M][WARP_N][4];
    #pragma unroll
    for (int mi = 0; mi < WARP_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WARP_N; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    extern __shared__ half smem_raw[];
    half (*smem_A)[BM][BK + PAD_A] = (half (*)[BM][BK + PAD_A])(smem_raw);
    half (*smem_B)[BK][BN + PAD_B] = (half (*)[BK][BN + PAD_B])(
        smem_raw + STAGES * BM * (BK + PAD_A));

    const int a_row0   = tid >> 3;
    const int a_row1   = a_row0 + 32;
    const int a_col    = (tid & 7) << 3;
    const int a_sw0    = a_col ^ (((a_row0 >> 3) & 7) << 3);
    const int a_sw1    = a_col ^ (((a_row1 >> 3) & 7) << 3);

    int b_row[4], b_col[4], b_sw[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx  = tid + i * 256;
        b_row[i] = idx >> 4;
        b_col[i] = (idx & 15) << 3;
        b_sw[i]  = b_col[i] ^ (((b_row[i] >> 1) & 7) << 3);
    }

    auto issue_loads_for_stage = [&](int stage, int iter) __attribute__((always_inline)) {
        const int k_base = k_start + iter * BK;
        const half* A_k = A + k_base;
        const half* B_k = B + k_base * N + bn_start;

        cp_async16(&smem_A[stage][a_row0][a_sw0], &A_k[a_row0 * K + a_col]);
        cp_async16(&smem_A[stage][a_row1][a_sw1], &A_k[a_row1 * K + a_col]);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cp_async16(&smem_B[stage][b_row[i]][b_sw[i]],
                       &B_k[b_row[i] * N + b_col[i]]);
        }
        asm volatile("cp.async.commit_group;\n" :: : "memory");
    };

    issue_loads_for_stage(0, 0);
    if (num_iters > 1) issue_loads_for_stage(1, 1);
    else asm volatile("cp.async.commit_group;\n" :: : "memory");

    uint32_t fa[4][WARP_M][4];
    uint32_t fb[4][WARP_N][2];

    #pragma unroll 1
    for (int iter = 0; iter < num_iters; iter++) {
        const int cur     = iter % STAGES;
        const int pf_iter = iter + 2;

        if (pf_iter < num_iters) issue_loads_for_stage(pf_iter % STAGES, pf_iter);
        else asm volatile("cp.async.commit_group;\n" :: : "memory");

        asm volatile("cp.async.wait_group 2;\n" :: : "memory");
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK / MMA_K; kk++) {
            #pragma unroll
            for (int mi = 0; mi < WARP_M; mi++) {
                const int la_row = mi * MMA_M + (lane_id & 15);
                const int la_col = kk * MMA_K + ((lane_id >> 4) << 3);
                const int la_sw  = la_col ^ (((la_row >> 3) & 7) << 3);
                uint32_t sa = __cvta_generic_to_shared(&smem_A[cur][la_row][la_sw]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(fa[kk][mi][0]),"=r"(fa[kk][mi][1]),
                      "=r"(fa[kk][mi][2]),"=r"(fa[kk][mi][3])
                    : "r"(sa));
            }
            #pragma unroll
            for (int ni = 0; ni < WARP_N; ni++) {
                const int lb_row = kk * MMA_K + (lane_id & 15);
                const int lb_col = warp_id * (WARP_N * MMA_N) + ni * MMA_N;
                const int lb_sw  = lb_col ^ (((lb_row >> 1) & 7) << 3);
                uint32_t sa = __cvta_generic_to_shared(&smem_B[cur][lb_row][lb_sw]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
                    : "=r"(fb[kk][ni][0]),"=r"(fb[kk][ni][1])
                    : "r"(sa));
            }
        }

        #pragma unroll
        for (int kk = 0; kk < BK / MMA_K; kk++) {
            #pragma unroll
            for (int mi = 0; mi < WARP_M; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WARP_N; ni++) {
                    mma_f32(acc[mi][ni], fa[kk][mi], fb[kk][ni], acc[mi][ni]);
                }
            }
        }
    }

    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    __syncthreads();

    float* my_split = C_split + (size_t)split * M * N;
    #pragma unroll
    for (int mi = 0; mi < WARP_M; mi++) {
        const int row0 = mi * MMA_M + (lane_id >> 2);
        const int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WARP_N; ni++) {
            const int col0 = bn_start + warp_id * (WARP_N * MMA_N) + ni * MMA_N + (lane_id & 3) * 2;
            *reinterpret_cast<float2*>(&my_split[row0 * N + col0]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&my_split[row1 * N + col0]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ __launch_bounds__(256)
void hgemm_reduce_kernel(
    const float* __restrict__ C_split,
    half* __restrict__ C,
    int MN, int num_splits)
{
    const int base = (blockIdx.x * 256 + threadIdx.x) * 8;
    if (base >= MN) return;

    float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
    float s4=0.f, s5=0.f, s6=0.f, s7=0.f;

    const size_t MN_sz = (size_t)MN;
    for (int sp = 0; sp < num_splits; sp++) {
        const float4* p = reinterpret_cast<const float4*>(
            C_split + sp * MN_sz + base);
        float4 v0 = __ldg(p);
        float4 v1 = __ldg(p + 1);
        s0 += v0.x; s1 += v0.y; s2 += v0.z; s3 += v0.w;
        s4 += v1.x; s5 += v1.y; s6 += v1.z; s7 += v1.w;
    }

    half* out = C + base;
    *reinterpret_cast<half2*>(out)   = __float22half2_rn(make_float2(s0, s1));
    *reinterpret_cast<half2*>(out+2) = __float22half2_rn(make_float2(s2, s3));
    *reinterpret_cast<half2*>(out+4) = __float22half2_rn(make_float2(s4, s5));
    *reinterpret_cast<half2*>(out+6) = __float22half2_rn(make_float2(s6, s7));
}

static float* g_workspace = nullptr;
static size_t g_workspace_bytes = 0;

static float* get_workspace(size_t needed) {
    if (g_workspace_bytes < needed) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, needed);
        g_workspace_bytes = needed;
    }
    return g_workspace;
}

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half*       C = reinterpret_cast<half*>(c.data_ptr());

    const int MN = M * N;

    const int smem_size = STAGES * (BM * (BK + PAD_A) + BK * (BN + PAD_B)) * (int)sizeof(half);

    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaFuncSetAttribute(hgemm_splitk_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 81920);
        smem_attr_set = true;
    }

    int num_splits = NUM_SPLITS;
    while (num_splits > 1 && (K % (num_splits * BK) != 0)) num_splits >>= 1;
    int K_per_split = K / num_splits;

    size_t ws_bytes = (size_t)num_splits * MN * sizeof(float);
    float* C_split = get_workspace(ws_bytes);

    const int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_n, 1, num_splits);
    hgemm_splitk_kernel<<<grid, 256, smem_size>>>(
        A, B, C_split, M, N, K, K_per_split);

    const int rd_blocks = (MN / 8 + 255) / 256;
    hgemm_reduce_kernel<<<rd_blocks, 256>>>(C_split, C, MN, num_splits);
}