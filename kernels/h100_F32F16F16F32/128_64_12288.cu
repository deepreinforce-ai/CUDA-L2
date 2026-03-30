#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

namespace cg = cooperative_groups;

static constexpr int M_TOTAL = 128;
static constexpr int N_TOTAL = 64;
static constexpr int K_TOTAL = 12288;
static constexpr int MN      = M_TOTAL * N_TOTAL;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

static constexpr int NUM_WARPS    = 4;
static constexpr int WARP_M_TILES = 2;
static constexpr int WARP_N_TILES = 8;

static constexpr int SPLIT_K   = 64;
static constexpr int K_PER_CTA = K_TOTAL / SPLIT_K;
static constexpr int K_STEPS   = K_PER_CTA / MMA_K;

static constexpr int STAGES     = 4;
static constexpr int STAGE_MASK = STAGES - 1;

static constexpr int A_SMEM_STRIDE = MMA_K + 8;
static constexpr int B_SMEM_STRIDE = N_TOTAL + 8;
static constexpr int A_STAGE_SIZE  = M_TOTAL * A_SMEM_STRIDE;
static constexpr int B_STAGE_SIZE  = MMA_K * B_SMEM_STRIDE;

__device__ __forceinline__
void cp_async16_ca(void* dst, const void* src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
           "l"(src) : "memory");
}

__device__ __forceinline__
void cp_async16_cg(void* dst, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
           "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" :: : "memory");
}

__device__ __forceinline__
void mma_m16n8k16(
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
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__
void stcs_float2(float* addr, float x, float y) {
    asm volatile(
        "st.global.cs.v2.f32 [%0], {%1, %2};\n"
        :: "l"(addr), "f"(x), "f"(y) : "memory");
}

__device__ __forceinline__
void issue_load_stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half smem_A[][A_STAGE_SIZE],
    half smem_B[][B_STAGE_SIZE],
    int stage, int k_step, int tid, int k_start)
{
    const int kb = k_start + k_step * MMA_K;

    {
        const half* src = A + (int64_t)tid * K_TOTAL + kb;
        half* dst = smem_A[stage] + tid * A_SMEM_STRIDE;
        cp_async16_ca(dst,     src);
        cp_async16_ca(dst + 8, src + 8);
    }

    {
        const int b_row = tid >> 3;
        const int b_col = (tid & 7) << 3;
        const half* src = B + (int64_t)(kb + b_row) * N_TOTAL + b_col;
        half* dst = smem_B[stage] + b_row * B_SMEM_STRIDE + b_col;
        cp_async16_cg(dst, src);
    }
}

__device__ __forceinline__
void load_a_frag(uint32_t ra[4], const half* smemA_stage, int warp_id, int lane_id, int mi) {
    const int row = warp_id * (WARP_M_TILES * MMA_M) + mi * MMA_M + (lane_id & 15);
    const int col = (lane_id >> 4) << 3;
    const uint32_t addr = __cvta_generic_to_shared(smemA_stage + row * A_SMEM_STRIDE + col);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
        : "r"(addr));
}

__device__ __forceinline__
void load_b_frag(uint32_t rb[2], const half* smemB_stage, int lane_id, int ni) {
    const int k_row = lane_id & 15;
    const int col   = ni * MMA_N;
    const uint32_t addr = __cvta_generic_to_shared(smemB_stage + k_row * B_SMEM_STRIDE + col);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(rb[0]), "=r"(rb[1])
        : "r"(addr));
}

__global__ void __launch_bounds__(128, 2)
gemm_fused_kernel_vec2_reduce(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ C_partial,
    half*       __restrict__ C_out)
{
    cg::grid_group grid = cg::this_grid();

    __shared__ half smem_A[STAGES][A_STAGE_SIZE];
    __shared__ half smem_B[STAGES][B_STAGE_SIZE];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int bid     = blockIdx.x;
    const int k_start = bid * K_PER_CTA;

    float acc[WARP_M_TILES][WARP_N_TILES][4];
    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            acc[mi][ni][0] = 0.f;
            acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f;
            acc[mi][ni][3] = 0.f;
        }
    }

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        issue_load_stage(A, B, smem_A, smem_B, s, s, tid, k_start);
        cp_async_commit();
    }

    #pragma unroll 12
    for (int k = 0; k < K_STEPS; ++k) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        const int cur = k & STAGE_MASK;
        const int nxt = (k + STAGES - 1) & STAGE_MASK;
        const int pfk = k + STAGES - 1;

        if (pfk < K_STEPS) {
            issue_load_stage(A, B, smem_A, smem_B, nxt, pfk, tid, k_start);
        }
        cp_async_commit();

        const half* A_tile = smem_A[cur];
        const half* B_tile = smem_B[cur];

        uint32_t ra[WARP_M_TILES][4];
        #pragma unroll
        for (int mi = 0; mi < WARP_M_TILES; ++mi) {
            load_a_frag(ra[mi], A_tile, warp_id, lane_id, mi);
        }

        uint32_t rb[WARP_N_TILES][2];
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            load_b_frag(rb[ni], B_tile, lane_id, ni);
        }

        #pragma unroll
        for (int mi = 0; mi < WARP_M_TILES; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < WARP_N_TILES; ++ni) {
                mma_m16n8k16(
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                    ra[mi][0], ra[mi][1], ra[mi][2], ra[mi][3],
                    rb[ni][0], rb[ni][1],
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }

    cp_async_wait_all();

    {
        float* c_ptr = C_partial + (int64_t)bid * MN;
        const int row0 = lane_id >> 2;
        const int col0 = (lane_id & 3) << 1;

        #pragma unroll
        for (int mi = 0; mi < WARP_M_TILES; ++mi) {
            const int row_base = warp_id * (WARP_M_TILES * MMA_M) + mi * MMA_M;
            const int r0 = row_base + row0;
            const int r1 = r0 + 8;
            #pragma unroll
            for (int ni = 0; ni < WARP_N_TILES; ++ni) {
                const int c = ni * MMA_N + col0;
                stcs_float2(c_ptr + r0 * N_TOTAL + c, acc[mi][ni][0], acc[mi][ni][1]);
                stcs_float2(c_ptr + r1 * N_TOTAL + c, acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }

    grid.sync();

    const int linear_tid = bid * blockDim.x + tid;
    if (linear_tid < (MN / 2)) {
        const int out0 = linear_tid << 1;
        float s0 = 0.f, s1 = 0.f;

        #pragma unroll
        for (int s = 0; s < SPLIT_K; ++s) {
            const float2 v = reinterpret_cast<const float2*>(C_partial + (int64_t)s * MN + out0)[0];
            s0 += v.x;
            s1 += v.y;
        }

        reinterpret_cast<half2*>(C_out + out0)[0] = __floats2half2_rn(s0, s1);
    }
}

__global__ void __launch_bounds__(128, 2)
gemm_noncoop_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ C_partial)
{
    __shared__ half smem_A[STAGES][A_STAGE_SIZE];
    __shared__ half smem_B[STAGES][B_STAGE_SIZE];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int bid     = blockIdx.x;
    const int k_start = bid * K_PER_CTA;

    float acc[WARP_M_TILES][WARP_N_TILES][4];
    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            acc[mi][ni][0] = 0.f;
            acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f;
            acc[mi][ni][3] = 0.f;
        }
    }

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        issue_load_stage(A, B, smem_A, smem_B, s, s, tid, k_start);
        cp_async_commit();
    }

    #pragma unroll 12
    for (int k = 0; k < K_STEPS; ++k) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        const int cur = k & STAGE_MASK;
        const int nxt = (k + STAGES - 1) & STAGE_MASK;
        const int pfk = k + STAGES - 1;
        if (pfk < K_STEPS) issue_load_stage(A, B, smem_A, smem_B, nxt, pfk, tid, k_start);
        cp_async_commit();

        const half* A_tile = smem_A[cur];
        const half* B_tile = smem_B[cur];

        uint32_t ra[WARP_M_TILES][4];
        #pragma unroll
        for (int mi = 0; mi < WARP_M_TILES; ++mi) load_a_frag(ra[mi], A_tile, warp_id, lane_id, mi);

        uint32_t rb[WARP_N_TILES][2];
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) load_b_frag(rb[ni], B_tile, lane_id, ni);

        #pragma unroll
        for (int mi = 0; mi < WARP_M_TILES; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < WARP_N_TILES; ++ni) {
                mma_m16n8k16(
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                    ra[mi][0], ra[mi][1], ra[mi][2], ra[mi][3],
                    rb[ni][0], rb[ni][1],
                    acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }

    cp_async_wait_all();

    float* c_ptr = C_partial + (int64_t)bid * MN;
    const int row0 = lane_id >> 2;
    const int col0 = (lane_id & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
        const int row_base = warp_id * (WARP_M_TILES * MMA_M) + mi * MMA_M;
        const int r0 = row_base + row0;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            const int c = ni * MMA_N + col0;
            stcs_float2(c_ptr + r0 * N_TOTAL + c, acc[mi][ni][0], acc[mi][ni][1]);
            stcs_float2(c_ptr + r1 * N_TOTAL + c, acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(256)
reduce_fallback_kernel_vec2(
    const float* __restrict__ C_partial,
    half*        __restrict__ C_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (MN / 2)) return;

    const int out0 = tid << 1;
    float s0 = 0.f, s1 = 0.f;

    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        const float2 v = reinterpret_cast<const float2*>(C_partial + (int64_t)s * MN + out0)[0];
        s0 += v.x;
        s1 += v.y;
    }

    reinterpret_cast<half2*>(C_out + out0)[0] = __floats2half2_rn(s0, s1);
}

static float* g_workspace       = nullptr;
static size_t g_workspace_bytes = 0;

static float* ensure_workspace(size_t bytes) {
    if (g_workspace_bytes < bytes) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, bytes);
        g_workspace_bytes = bytes;
    }
    return g_workspace;
}

static int  g_init    = 0;
static bool g_coop_ok = false;

static void init_launch_config() {
    if (g_init) return;
    g_init = 1;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    if (!prop.cooperativeLaunch) {
        g_coop_ok = false;
        return;
    }

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, gemm_fused_kernel_vec2_reduce, 128, 0);

    int max_coop_blocks = prop.multiProcessorCount * blocks_per_sm;
    g_coop_ok = (SPLIT_K <= max_coop_blocks);
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr());

    (void)b_col_major;

    float* workspace = ensure_workspace((size_t)SPLIT_K * MN * sizeof(float));

    init_launch_config();

    if (g_coop_ok) {
        void* args[] = {
            (void*)&A_ptr,
            (void*)&B_ptr,
            (void*)&workspace,
            (void*)&C_ptr
        };
        cudaLaunchCooperativeKernel(
            (void*)gemm_fused_kernel_vec2_reduce,
            dim3(SPLIT_K), dim3(128),
            args, 0, nullptr);
    } else {
        gemm_noncoop_kernel<<<SPLIT_K, 128>>>(A_ptr, B_ptr, workspace);
        reduce_fallback_kernel_vec2<<<((MN / 2) + 255) / 256, 256>>>(workspace, C_ptr);
    }
}