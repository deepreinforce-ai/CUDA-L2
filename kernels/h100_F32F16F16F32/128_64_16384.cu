#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

namespace cg = cooperative_groups;

static constexpr int M_SIZE = 128;
static constexpr int N_SIZE = 64;
static constexpr int K_SIZE = 16384;
static constexpr int TILE_K = 16;

static constexpr int K_SPLITS      = 128;
static constexpr int K_PER_CTA     = K_SIZE / K_SPLITS;
static constexpr int TILES_PER_CTA = K_PER_CTA / TILE_K;

static constexpr int WARPS_M   = 2;
static constexpr int WARPS_N   = 4;
static constexpr int WARP_ROWS = M_SIZE / WARPS_M;
static constexpr int WARP_COLS = N_SIZE / WARPS_N;

static constexpr int NUM_STAGES = 4;

static constexpr int A_ROW_STRIDE  = 24;
static constexpr int A_STAGE_HALFS = M_SIZE * A_ROW_STRIDE;

static constexpr int B_N_STRIDE    = 72;
static constexpr int B_STAGE_HALFS = TILE_K * B_N_STRIDE;

static constexpr int SMEM_BYTES = NUM_STAGES * (A_STAGE_HALFS + B_STAGE_HALFS) * 2;

static constexpr int WS_PER_CTA = M_SIZE * N_SIZE;
static constexpr int WS_TOTAL   = K_SPLITS * WS_PER_CTA;

static constexpr int THREADS_PER_ELEM  = 4;
static constexpr int SPLITS_PER_THREAD = K_SPLITS / THREADS_PER_ELEM;
static constexpr int ELEMS_PER_CTA_RED = WS_PER_CTA / K_SPLITS;

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float* __restrict__ d,
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    const float* __restrict__ c
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

__global__ void __launch_bounds__(256, 2)
hgemm_optimized_final(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ workspace,
    half* __restrict__ C
) {
    cg::grid_group grid = cg::this_grid();

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int cta_id  = blockIdx.x;

    {
        const int k_start  = cta_id * K_PER_CTA;
        const int warp_m   = warp_id >> 2;
        const int warp_n   = warp_id & 3;
        const int warp_row = warp_m * WARP_ROWS;
        const int warp_col = warp_n * WARP_COLS;

        float acc[4][2][4];
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

        extern __shared__ half smem[];
        half* smem_A = smem;
        half* smem_B = smem + NUM_STAGES * A_STAGE_HALFS;

        const int a_row    = tid & 127;
        const int a_koff   = (tid >> 7) * 8;
        const int a_k_swiz = a_koff ^ (((a_row >> 2) & 1) * 8);

        const int b_k_local = tid >> 4;
        const int b_n_start = (tid & 15) * 4;
        const int b_n_swiz  = ((b_n_start & ~7) ^ (((b_k_local >> 2) & 3) << 3)) | (b_n_start & 7);

        const int k_lp = (lane_id & 7) + ((lane_id >> 3) & 1) * 8;

        #pragma unroll
        for (int s = 0; s < NUM_STAGES - 1; s++) {
            const int ks = k_start + s * TILE_K;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(smem_u32(smem_A + s * A_STAGE_HALFS + a_row * A_ROW_STRIDE + a_k_swiz)),
                   "l"(A + a_row * K_SIZE + ks + a_koff) : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                :: "r"(smem_u32(smem_B + s * B_STAGE_HALFS + b_k_local * B_N_STRIDE + b_n_swiz)),
                   "l"(B + (ks + b_k_local) * N_SIZE + b_n_start) : "memory");
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        #pragma unroll 2
        for (int tile = 0; tile < TILES_PER_CTA; tile++) {
            const int cur_s     = tile % NUM_STAGES;
            const int fill_tile = tile + (NUM_STAGES - 1);

            if (fill_tile < TILES_PER_CTA) {
                const int ks = k_start + fill_tile * TILE_K;
                const int fs = fill_tile % NUM_STAGES;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(smem_u32(smem_A + fs * A_STAGE_HALFS + a_row * A_ROW_STRIDE + a_k_swiz)),
                       "l"(A + a_row * K_SIZE + ks + a_koff) : "memory");
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                    :: "r"(smem_u32(smem_B + fs * B_STAGE_HALFS + b_k_local * B_N_STRIDE + b_n_swiz)),
                       "l"(B + (ks + b_k_local) * N_SIZE + b_n_start) : "memory");
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }

            asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2) : "memory");
            __syncthreads();

            uint32_t a_frag[4][4];
            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                const int row      = warp_row + mi * 16 + (lane_id & 15);
                const int col_nat  = (lane_id >> 4) * 8;
                const int col_swiz = col_nat ^ (((row >> 2) & 1) * 8);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                      "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                    : "r"(smem_u32(smem_A + cur_s * A_STAGE_HALFS + row * A_ROW_STRIDE + col_swiz))
                );
            }

            uint32_t b_frag[2][2];
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                const int n_base   = warp_col + ni * 8;
                const int n_stored = n_base ^ (((k_lp >> 2) & 3) << 3);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                    : "r"(smem_u32(smem_B + cur_s * B_STAGE_HALFS + k_lp * B_N_STRIDE + n_stored))
                );
            }

            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                mma_m16n8k16(acc[mi][0], a_frag[mi], b_frag[0], acc[mi][0]);
            }
            #pragma unroll
            for (int mi = 0; mi < 4; mi++) {
                mma_m16n8k16(acc[mi][1], a_frag[mi], b_frag[1], acc[mi][1]);
            }

            __syncthreads();
        }

        asm volatile("cp.async.wait_all;\n" ::: "memory");

        float* __restrict__ my_ws = workspace + (size_t)cta_id * WS_PER_CTA;
        const int row_off = lane_id >> 2;
        const int col_off = 2 * (lane_id & 3);

        #pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            const int row0 = warp_row + mi * 16 + row_off;
            const int row1 = row0 + 8;
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                const int col = warp_col + ni * 8 + col_off;
                *reinterpret_cast<float2*>(&my_ws[row0 * N_SIZE + col]) =
                    make_float2(acc[mi][ni][0], acc[mi][ni][1]);
                *reinterpret_cast<float2*>(&my_ws[row1 * N_SIZE + col]) =
                    make_float2(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }

    grid.sync();

    {
        extern __shared__ half smem_raw[];
        float* red_smem = reinterpret_cast<float*>(smem_raw);

        const int elem_local  = tid >> 2;
        const int split_grp   = tid & 3;
        const int elem_global = cta_id * ELEMS_PER_CTA_RED + elem_local;
        const int split_base  = split_grp * SPLITS_PER_THREAD;

        const float* ws_col = workspace + (size_t)elem_global;

        float a0=0.f, a1=0.f, a2=0.f, a3=0.f, a4=0.f, a5=0.f, a6=0.f, a7=0.f;
        float b0=0.f, b1=0.f, b2=0.f, b3=0.f, b4=0.f, b5=0.f, b6=0.f, b7=0.f;

        const int sb = split_base;
        #pragma unroll 2
        for (int i = 0; i < 2; i++) {
            const int ba = sb + i * 16;
            const int bb = sb + i * 16 + 8;
            a0 += __ldg(ws_col + (size_t)(ba+0) * WS_PER_CTA);
            b0 += __ldg(ws_col + (size_t)(bb+0) * WS_PER_CTA);
            a1 += __ldg(ws_col + (size_t)(ba+1) * WS_PER_CTA);
            b1 += __ldg(ws_col + (size_t)(bb+1) * WS_PER_CTA);
            a2 += __ldg(ws_col + (size_t)(ba+2) * WS_PER_CTA);
            b2 += __ldg(ws_col + (size_t)(bb+2) * WS_PER_CTA);
            a3 += __ldg(ws_col + (size_t)(ba+3) * WS_PER_CTA);
            b3 += __ldg(ws_col + (size_t)(bb+3) * WS_PER_CTA);
            a4 += __ldg(ws_col + (size_t)(ba+4) * WS_PER_CTA);
            b4 += __ldg(ws_col + (size_t)(bb+4) * WS_PER_CTA);
            a5 += __ldg(ws_col + (size_t)(ba+5) * WS_PER_CTA);
            b5 += __ldg(ws_col + (size_t)(bb+5) * WS_PER_CTA);
            a6 += __ldg(ws_col + (size_t)(ba+6) * WS_PER_CTA);
            b6 += __ldg(ws_col + (size_t)(bb+6) * WS_PER_CTA);
            a7 += __ldg(ws_col + (size_t)(ba+7) * WS_PER_CTA);
            b7 += __ldg(ws_col + (size_t)(bb+7) * WS_PER_CTA);
        }
        float sum = (a0+a1+a2+a3+a4+a5+a6+a7) + (b0+b1+b2+b3+b4+b5+b6+b7);

        const unsigned int mask4 = 0xFu << (lane_id & 28u);
        sum += __shfl_down_sync(mask4, sum, 2);
        sum += __shfl_down_sync(mask4, sum, 1);

        if (split_grp == 0) {
            red_smem[elem_local] = sum;
        }
        __syncthreads();

        if (tid < 32) {
            const int out_base = cta_id * ELEMS_PER_CTA_RED;
            const float v0 = red_smem[tid * 2];
            const float v1 = red_smem[tid * 2 + 1];
            *reinterpret_cast<half2*>(C + out_base + tid * 2) =
                __float22half2_rn(make_float2(v0, v1));
        }
    }
}

__global__ void __launch_bounds__(256, 2)
hgemm_fallback_gemm(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ workspace
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int cta_id  = blockIdx.x;
    const int k_start = cta_id * K_PER_CTA;

    const int warp_m   = warp_id >> 2;
    const int warp_n   = warp_id & 3;
    const int warp_row = warp_m * WARP_ROWS;
    const int warp_col = warp_n * WARP_COLS;

    float acc[4][2][4];
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    extern __shared__ half smem[];
    half* smem_A = smem;
    half* smem_B = smem + NUM_STAGES * A_STAGE_HALFS;

    const int a_row    = tid & 127;
    const int a_koff   = (tid >> 7) * 8;
    const int a_k_swiz = a_koff ^ (((a_row >> 2) & 1) * 8);

    const int b_k_local = tid >> 4;
    const int b_n_start = (tid & 15) * 4;
    const int b_n_swiz  = ((b_n_start & ~7) ^ (((b_k_local >> 2) & 3) << 3)) | (b_n_start & 7);

    const int k_lp = (lane_id & 7) + ((lane_id >> 3) & 1) * 8;

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        const int ks = k_start + s * TILE_K;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_u32(smem_A + s * A_STAGE_HALFS + a_row * A_ROW_STRIDE + a_k_swiz)),
               "l"(A + a_row * K_SIZE + ks + a_koff) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
            :: "r"(smem_u32(smem_B + s * B_STAGE_HALFS + b_k_local * B_N_STRIDE + b_n_swiz)),
               "l"(B + (ks + b_k_local) * N_SIZE + b_n_start) : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    #pragma unroll 2
    for (int tile = 0; tile < TILES_PER_CTA; tile++) {
        const int cur_s     = tile % NUM_STAGES;
        const int fill_tile = tile + (NUM_STAGES - 1);

        if (fill_tile < TILES_PER_CTA) {
            const int ks = k_start + fill_tile * TILE_K;
            const int fs = fill_tile % NUM_STAGES;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(smem_u32(smem_A + fs * A_STAGE_HALFS + a_row * A_ROW_STRIDE + a_k_swiz)),
                   "l"(A + a_row * K_SIZE + ks + a_koff) : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                :: "r"(smem_u32(smem_B + fs * B_STAGE_HALFS + b_k_local * B_N_STRIDE + b_n_swiz)),
                   "l"(B + (ks + b_k_local) * N_SIZE + b_n_start) : "memory");
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2) : "memory");
        __syncthreads();

        uint32_t a_frag[4][4];
        #pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            const int row      = warp_row + mi * 16 + (lane_id & 15);
            const int col_nat  = (lane_id >> 4) * 8;
            const int col_swiz = col_nat ^ (((row >> 2) & 1) * 8);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                  "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                : "r"(smem_u32(smem_A + cur_s * A_STAGE_HALFS + row * A_ROW_STRIDE + col_swiz))
            );
        }

        uint32_t b_frag[2][2];
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const int n_base   = warp_col + ni * 8;
            const int n_stored = n_base ^ (((k_lp >> 2) & 3) << 3);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                : "r"(smem_u32(smem_B + cur_s * B_STAGE_HALFS + k_lp * B_N_STRIDE + n_stored))
            );
        }

        #pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            mma_m16n8k16(acc[mi][0], a_frag[mi], b_frag[0], acc[mi][0]);
        }
        #pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            mma_m16n8k16(acc[mi][1], a_frag[mi], b_frag[1], acc[mi][1]);
        }

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");

    float* __restrict__ my_ws = workspace + (size_t)cta_id * WS_PER_CTA;
    const int row_off = lane_id >> 2;
    const int col_off = 2 * (lane_id & 3);

    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        const int row0 = warp_row + mi * 16 + row_off;
        const int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const int col = warp_col + ni * 8 + col_off;
            *reinterpret_cast<float2*>(&my_ws[row0 * N_SIZE + col]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&my_ws[row1 * N_SIZE + col]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

__global__ void __launch_bounds__(256, 4)
hgemm_fallback_reduce(
    const float* __restrict__ workspace,
    half* __restrict__ C
) {
    const int elem = blockIdx.x * 256 + threadIdx.x;
    if (elem >= WS_PER_CTA) return;

    const float* ws = workspace + elem;
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;
    float b0=0.f,b1=0.f,b2=0.f,b3=0.f,b4=0.f,b5=0.f,b6=0.f,b7=0.f;

    #pragma unroll 2
    for (int sp = 0; sp < K_SPLITS; sp += 16) {
        a0 += __ldg(ws + (size_t)(sp+ 0)*WS_PER_CTA);
        a1 += __ldg(ws + (size_t)(sp+ 1)*WS_PER_CTA);
        a2 += __ldg(ws + (size_t)(sp+ 2)*WS_PER_CTA);
        a3 += __ldg(ws + (size_t)(sp+ 3)*WS_PER_CTA);
        a4 += __ldg(ws + (size_t)(sp+ 4)*WS_PER_CTA);
        a5 += __ldg(ws + (size_t)(sp+ 5)*WS_PER_CTA);
        a6 += __ldg(ws + (size_t)(sp+ 6)*WS_PER_CTA);
        a7 += __ldg(ws + (size_t)(sp+ 7)*WS_PER_CTA);
        b0 += __ldg(ws + (size_t)(sp+ 8)*WS_PER_CTA);
        b1 += __ldg(ws + (size_t)(sp+ 9)*WS_PER_CTA);
        b2 += __ldg(ws + (size_t)(sp+10)*WS_PER_CTA);
        b3 += __ldg(ws + (size_t)(sp+11)*WS_PER_CTA);
        b4 += __ldg(ws + (size_t)(sp+12)*WS_PER_CTA);
        b5 += __ldg(ws + (size_t)(sp+13)*WS_PER_CTA);
        b6 += __ldg(ws + (size_t)(sp+14)*WS_PER_CTA);
        b7 += __ldg(ws + (size_t)(sp+15)*WS_PER_CTA);
    }
    C[elem] = __float2half((a0+a1+a2+a3+a4+a5+a6+a7)+(b0+b1+b2+b3+b4+b5+b6+b7));
}

static float* g_workspace    = nullptr;
static size_t g_workspace_sz = 0;
static bool   g_init_done    = false;
static bool   g_coop_ok      = false;
static int    g_coop_max_ctas = 0;

static float* get_workspace_ptr(size_t bytes) {
    if (g_workspace_sz < bytes) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, bytes);
        g_workspace_sz = bytes;
    }
    return g_workspace;
}

static void init_once() {
    if (g_init_done) return;
    cudaFuncSetAttribute(hgemm_optimized_final,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(hgemm_fallback_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    int dev = 0; cudaGetDevice(&dev);
    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
    g_coop_ok = (coop != 0);
    if (g_coop_ok) {
        int bps = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &bps, hgemm_optimized_final, 256, SMEM_BYTES);
        int nsm = 0;
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        g_coop_max_ctas = bps * nsm;
    }
    g_init_done = true;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C_ptr       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    float* workspace = get_workspace_ptr((size_t)WS_TOTAL * sizeof(float));
    init_once();

    if (g_coop_ok && g_coop_max_ctas >= K_SPLITS) {
        void* args[] = {(void*)&A_ptr, (void*)&B_ptr, (void*)&workspace, (void*)&C_ptr};
        cudaLaunchCooperativeKernel(
            (void*)hgemm_optimized_final,
            dim3(K_SPLITS), dim3(256), args, SMEM_BYTES, nullptr);
    } else {
        hgemm_fallback_gemm<<<K_SPLITS, 256, SMEM_BYTES>>>(A_ptr, B_ptr, workspace);
        hgemm_fallback_reduce<<<32, 256>>>(workspace, C_ptr);
    }
}