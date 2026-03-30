#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define M_DIM      64
#define N_DIM      256
#define K_DIM      4096
#define TILE_M     16
#define TILE_N     8
#define BLOCK_K    64
#define MMA_K      16
#define NUM_WARPS  8
#define WARP_K     512
#define WARP_K_ITERS (WARP_K / BLOCK_K)
#define NUM_STAGES 4
#define PAD_A      8
#define PAD_B      8

#define SMA_STRIDE  (BLOCK_K + PAD_A)
#define SMB_STRIDE  (TILE_N  + PAD_B)
#define SMA_STAGE_ELEMS (TILE_M * SMA_STRIDE)
#define SMB_STAGE_ELEMS (BLOCK_K * SMB_STRIDE)
#define SMA_WARP_ELEMS  (NUM_STAGES * SMA_STAGE_ELEMS)
#define SMB_WARP_ELEMS  (NUM_STAGES * SMB_STAGE_ELEMS)
#define SMA_TOTAL       (NUM_WARPS * SMA_WARP_ELEMS)
#define SMB_TOTAL       (NUM_WARPS * SMB_WARP_ELEMS)
#define SMA_BYTES   (SMA_TOTAL * 2)
#define SMB_BYTES   (SMB_TOTAL * 2)
#define SMC_STRIDE  (TILE_N + 2)
#define SMC_WARP_ELEMS (TILE_M * SMC_STRIDE)
#define SMC_TOTAL   (NUM_WARPS * SMC_WARP_ELEMS)

__device__ __forceinline__ uint32_t cvta_smem(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 u64a; cvta.to.shared.u64 u64a, %1; cvt.u32.u64 %0, u64a; }"
                 : "=r"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

__global__ void __launch_bounds__(256, 1)
hgemm_dynamic_smem(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C,
    int dummy)
{
    extern __shared__ char smem_raw[];

    const int tile_m = blockIdx.x >> 5;
    const int tile_n = blockIdx.x & 31;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int row_base = tile_m * TILE_M;
    const int col_base = tile_n * TILE_N;
    const int warp_k_start = warp_id * WARP_K;

    half*  smA_base = reinterpret_cast<half*>(smem_raw);
    half*  smB_base = reinterpret_cast<half*>(smem_raw + SMA_BYTES);
    float* smC_base = reinterpret_cast<float*>(smem_raw + SMA_BYTES + SMB_BYTES);

    auto smA_ptr = [&](int stage, int row, int col) -> half* {
        return smA_base + (warp_id * SMA_WARP_ELEMS + stage * SMA_STAGE_ELEMS
                           + row * SMA_STRIDE + col);
    };

    auto smB_ptr = [&](int stage, int kr, int nc) -> half* {
        return smB_base + (warp_id * SMB_WARP_ELEMS + stage * SMB_STAGE_ELEMS
                           + kr * SMB_STRIDE + nc);
    };

    auto smC_ptr = [&](int w, int row, int col) -> float* {
        return smC_base + (w * SMC_WARP_ELEMS + row * SMC_STRIDE + col);
    };

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    auto load_A = [&](int stage, int k_off) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int g = lane * 4 + i;
            int r = g >> 3;
            int c = (g & 7) << 3;
            cp_async16(cvta_smem(smA_ptr(stage, r, c)),
                       A + (size_t)(row_base + r) * K_DIM + k_off + c);
        }
    };

    auto load_B = [&](int stage, int k_off) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int kr = lane * 2 + i;
            cp_async16(cvta_smem(smB_ptr(stage, kr, 0)),
                       B + (size_t)(k_off + kr) * N_DIM + col_base);
        }
    };

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < WARP_K_ITERS) {
            load_A(s, warp_k_start + s * BLOCK_K);
            load_B(s, warp_k_start + s * BLOCK_K);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    #pragma unroll 1
    for (int iter = 0; iter < WARP_K_ITERS; iter++) {
        int pf = iter + (NUM_STAGES - 1);
        int pf_stage = pf % NUM_STAGES;
        if (pf < WARP_K_ITERS) {
            load_A(pf_stage, warp_k_start + pf * BLOCK_K);
            load_B(pf_stage, warp_k_start + pf * BLOCK_K);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 3;" ::: "memory");
        __syncwarp();

        int cur = iter % NUM_STAGES;

        #pragma unroll
        for (int ki = 0; ki < BLOCK_K; ki += MMA_K) {
            uint32_t a_addr = cvta_smem(smA_ptr(cur, lane & 15, ki + ((lane >> 4) << 3)));
            uint32_t ra[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                         : "=r"(ra[0]),"=r"(ra[1]),"=r"(ra[2]),"=r"(ra[3])
                         : "r"(a_addr));

            uint32_t b_addr = cvta_smem(smB_ptr(cur, ki + (lane & 15), 0));
            uint32_t rb[2];
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                         : "=r"(rb[0]),"=r"(rb[1])
                         : "r"(b_addr));

            mma_m16n8k16(acc0, acc1, acc2, acc3,
                         ra[0], ra[1], ra[2], ra[3],
                         rb[0], rb[1],
                         acc0, acc1, acc2, acc3);
        }
    }

    asm volatile("cp.async.wait_all;" ::: "memory");

    {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        int c  = (lane & 3) << 1;
        *smC_ptr(warp_id, r0, c    ) = acc0;
        *smC_ptr(warp_id, r0, c + 1) = acc1;
        *smC_ptr(warp_id, r1, c    ) = acc2;
        *smC_ptr(warp_id, r1, c + 1) = acc3;
    }

    __syncthreads();

    if (warp_id == 0) {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        int c  = (lane & 3) << 1;

        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            s0 += *smC_ptr(w, r0, c    );
            s1 += *smC_ptr(w, r0, c + 1);
            s2 += *smC_ptr(w, r1, c    );
            s3 += *smC_ptr(w, r1, c + 1);
        }

        int gr0 = row_base + r0;
        int gr1 = row_base + r1;
        int gc  = col_base + c;

        *reinterpret_cast<half2*>(&C[gr0 * N_DIM + gc]) = __float22half2_rn(make_float2(s0, s1));
        *reinterpret_cast<half2*>(&C[gr1 * N_DIM + gc]) = __float22half2_rn(make_float2(s2, s3));
    }
}

#define NW2 2
#define WK2 2048
#define WKI2 (WK2 / BLOCK_K)

__global__ void __launch_bounds__(64, 4)
hgemm_2warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C)
{
    __shared__ half  smA2[NW2][NUM_STAGES][TILE_M][BLOCK_K + PAD_A];
    __shared__ half  smB2[NW2][NUM_STAGES][BLOCK_K][TILE_N + PAD_B];
    __shared__ float smC2[NW2][TILE_M][TILE_N + 2];

    const int tile_m = blockIdx.x >> 5;
    const int tile_n = blockIdx.x & 31;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int row_base = tile_m * TILE_M;
    const int col_base = tile_n * TILE_N;
    const int warp_k_start = warp_id * WK2;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    auto load_A2 = [&](int stage, int k_off) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int g = lane * 4 + i;
            int r = g >> 3;
            int c = (g & 7) << 3;
            cp_async16(cvta_smem(&smA2[warp_id][stage][r][c]),
                       A + (size_t)(row_base + r) * K_DIM + k_off + c);
        }
    };

    auto load_B2 = [&](int stage, int k_off) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int kr = lane * 2 + i;
            cp_async16(cvta_smem(&smB2[warp_id][stage][kr][0]),
                       B + (size_t)(k_off + kr) * N_DIM + col_base);
        }
    };

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < WKI2) {
            load_A2(s, warp_k_start + s * BLOCK_K);
            load_B2(s, warp_k_start + s * BLOCK_K);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    #pragma unroll 1
    for (int iter = 0; iter < WKI2; iter++) {
        int pf = iter + (NUM_STAGES - 1);
        int pf_stage = pf % NUM_STAGES;
        if (pf < WKI2) {
            load_A2(pf_stage, warp_k_start + pf * BLOCK_K);
            load_B2(pf_stage, warp_k_start + pf * BLOCK_K);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 3;" ::: "memory");
        __syncwarp();

        int cur = iter % NUM_STAGES;

        #pragma unroll
        for (int ki = 0; ki < BLOCK_K; ki += MMA_K) {
            uint32_t a_addr = cvta_smem(&smA2[warp_id][cur][lane & 15][ki + ((lane >> 4) << 3)]);
            uint32_t ra[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                         : "=r"(ra[0]),"=r"(ra[1]),"=r"(ra[2]),"=r"(ra[3])
                         : "r"(a_addr));

            uint32_t b_addr = cvta_smem(&smB2[warp_id][cur][ki + (lane & 15)][0]);
            uint32_t rb[2];
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                         : "=r"(rb[0]),"=r"(rb[1])
                         : "r"(b_addr));

            mma_m16n8k16(acc0, acc1, acc2, acc3,
                         ra[0], ra[1], ra[2], ra[3],
                         rb[0], rb[1],
                         acc0, acc1, acc2, acc3);
        }
    }

    asm volatile("cp.async.wait_all;" ::: "memory");

    {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        int c  = (lane & 3) << 1;
        smC2[warp_id][r0][c    ] = acc0;
        smC2[warp_id][r0][c + 1] = acc1;
        smC2[warp_id][r1][c    ] = acc2;
        smC2[warp_id][r1][c + 1] = acc3;
    }

    __syncthreads();

    if (warp_id == 0) {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        int c  = (lane & 3) << 1;

        float s0 = smC2[0][r0][c] + smC2[1][r0][c];
        float s1 = smC2[0][r0][c+1] + smC2[1][r0][c+1];
        float s2 = smC2[0][r1][c] + smC2[1][r1][c];
        float s3 = smC2[0][r1][c+1] + smC2[1][r1][c+1];

        int gr0 = row_base + r0;
        int gr1 = row_base + r1;
        int gc  = col_base + c;

        *reinterpret_cast<half2*>(&C[gr0 * N_DIM + gc]) = __float22half2_rn(make_float2(s0, s1));
        *reinterpret_cast<half2*>(&C[gr1 * N_DIM + gc]) = __float22half2_rn(make_float2(s2, s3));
    }
}

static bool g_smem_configured = false;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const size_t dyn_smem = SMA_BYTES + SMB_BYTES + SMC_TOTAL * sizeof(float);

    if (!g_smem_configured) {
        cudaError_t err = cudaFuncSetAttribute(
            hgemm_dynamic_smem,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)dyn_smem);
        g_smem_configured = (err == cudaSuccess);
    }

    if (g_smem_configured) {
        dim3 grid(128);
        dim3 block(256);
        hgemm_dynamic_smem<<<grid, block, dyn_smem>>>(ptr_A, ptr_B, ptr_C, 0);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        dim3 grid(128);
        dim3 block(64);
        hgemm_2warp<<<grid, block>>>(ptr_A, ptr_B, ptr_C);
    }
}