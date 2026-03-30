#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <cuda.h>

static constexpr int BM         = 128;
static constexpr int BN         = 128;
static constexpr int BK         = 64;
static constexpr int NUM_STAGES = 3;
static constexpr int BLOCK_SIZE = 256;
static constexpr int NUM_SPLITS = 16;
static constexpr int NUM_KK     = BK / 16;

static constexpr int SA_STRIDE = BK;
static constexpr int SB_STRIDE = BN + 8;
static constexpr int SA_STAGE  = BM * SA_STRIDE;
static constexpr int SB_STAGE  = BK * SB_STRIDE;
static constexpr int SA_TOTAL  = NUM_STAGES * SA_STAGE;
static constexpr int SB_TOTAL  = NUM_STAGES * SB_STAGE;

__device__ __forceinline__ int swzA(int row, int col) {
    int chunk = col >> 3;
    int off   = col & 7;
    return ((chunk ^ (row & 7)) << 3) | off;
}

__device__ __forceinline__ void mma16x8x16(
    float &c0, float &c1, float &c2, float &c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float ci0, float ci1, float ci2, float ci3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(c0),"=f"(c1),"=f"(c2),"=f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(ci0),"f"(ci1),"f"(ci2),"f"(ci3)
    );
}

__device__ __forceinline__ void ldm_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldm_x2t(uint32_t &r0, uint32_t &r1, uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
        : "=r"(r0),"=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void cp_async_cg(uint32_t sa, const void* gp) {
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(sa), "l"(gp));
}

__global__ void __launch_bounds__(256, 2)
hgemm_split_k_kernel(
    const half * __restrict__ A,
    const half * __restrict__ B_col,
    float      * __restrict__ C_part,
    int M, int N, int K,
    int k_per_split)
{
    const int bm       = blockIdx.x;
    const int split_id = blockIdx.y;

    const int row_start = bm * BM;
    const int k_start   = split_id * k_per_split;
    const int k_end     = min(k_start + k_per_split, K);
    if (row_start >= M || k_start >= K) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    extern __shared__ half smem[];
    half* smemA = smem;
    half* smemB = smem + SA_TOTAL;

    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    const int la_rb = tid >> 3;
    const int la_c  = (tid & 7) << 3;

    const int lb_kb = tid >> 4;
    const int lb_n  = (tid & 15) << 3;

    int num_k_tiles = (k_end - k_start + BK - 1) / BK;
    if (num_k_tiles <= 0) return;

    auto issue_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = la_rb + r * 32;
            int gr  = row_start + row;
            int gc  = k_off + la_c;
            int sw  = swzA(row, la_c);
            half* dst = smemA + stage * SA_STAGE + row * SA_STRIDE + sw;
            if (gr < M && gc + 7 < K) {
                cp_async_cg(__cvta_generic_to_shared(dst), A + (long)gr * K + gc);
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0,0,0,0);
            }
        }
    };

    auto issue_B_rowmajor = [&](int stage, int k_off, const half* B_rm) __attribute__((always_inline)) {
        const int lb_col_rm = (tid & 15) << 3;
        const int lb_row_rm = tid >> 4;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = lb_row_rm + r * 16;
            int gk  = k_off + row;
            half* dst = smemB + stage * SB_STAGE + row * SB_STRIDE + lb_col_rm;
            if (gk < k_end && lb_col_rm + 7 < N) {
                cp_async_cg(__cvta_generic_to_shared(dst), B_rm + (long)gk * N + lb_col_rm);
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0,0,0,0);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < num_k_tiles) {
            issue_A(s, k_start + s * BK);
            issue_B_rowmajor(s, k_start + s * BK, B_col);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    uint32_t a_reg[2][2][4];
    uint32_t b_reg[2][8][2];

    for (int kt = 0; kt < num_k_tiles; kt++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2));
        __syncthreads();

        int cur = kt % NUM_STAGES;
        half* curA = smemA + cur * SA_STAGE;
        half* curB = smemB + cur * SB_STAGE;

        int pft = kt + NUM_STAGES - 1;
        if (pft < num_k_tiles) {
            issue_A(pft % NUM_STAGES, k_start + pft * BK);
            issue_B_rowmajor(pft % NUM_STAGES, k_start + pft * BK, B_col);
        }
        asm volatile("cp.async.commit_group;\n" ::);

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int base_row = warp_m * 32 + mi * 16;
            int smem_row = base_row + (lane_id & 15);
            int log_col  = (lane_id >> 4) << 3;
            int phy_col  = swzA(smem_row, log_col);
            uint32_t sa  = __cvta_generic_to_shared(curA + smem_row * SA_STRIDE + phy_col);
            ldm_x4(a_reg[0][mi][0], a_reg[0][mi][1], a_reg[0][mi][2], a_reg[0][mi][3], sa);
        }
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int base_col = warp_n * 64 + ni * 8;
            int smem_row = (lane_id & 15);
            int smem_col = base_col + ((lane_id >> 4) << 3);
            uint32_t sb  = __cvta_generic_to_shared(curB + smem_row * SB_STRIDE + smem_col);
            ldm_x2t(b_reg[0][ni][0], b_reg[0][ni][1], sb);
        }

        #pragma unroll
        for (int kk = 0; kk < NUM_KK; kk++) {
            int db  = kk & 1;
            int ndb = 1 - db;

            if (kk < NUM_KK - 1) {
                int nkk = kk + 1;
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int base_row = warp_m * 32 + mi * 16;
                    int smem_row = base_row + (lane_id & 15);
                    int log_col  = nkk * 16 + ((lane_id >> 4) << 3);
                    int phy_col  = swzA(smem_row, log_col);
                    uint32_t sa  = __cvta_generic_to_shared(curA + smem_row * SA_STRIDE + phy_col);
                    ldm_x4(a_reg[ndb][mi][0], a_reg[ndb][mi][1],
                           a_reg[ndb][mi][2], a_reg[ndb][mi][3], sa);
                }
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    int base_col = warp_n * 64 + ni * 8;
                    int smem_row = nkk * 16 + (lane_id & 15);
                    int smem_col = base_col + ((lane_id >> 4) << 3);
                    uint32_t sb  = __cvta_generic_to_shared(curB + smem_row * SB_STRIDE + smem_col);
                    ldm_x2t(b_reg[ndb][ni][0], b_reg[ndb][ni][1], sb);
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    mma16x8x16(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a_reg[db][mi][0], a_reg[db][mi][1],
                        a_reg[db][mi][2], a_reg[db][mi][3],
                        b_reg[db][ni][0], b_reg[db][ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]
                    );
                }
            }
        }
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    float* C_out = C_part + (long)split_id * M * N;
    const int out_row_off = lane_id >> 2;
    const int out_col_off = (lane_id & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        int base_row = row_start + warp_m * 32 + mi * 16;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int base_col = warp_n * 64 + ni * 8;
            int r0 = base_row + out_row_off;
            int r1 = r0 + 8;
            int c0 = base_col + out_col_off;

            if (r0 < M && c0 + 1 < N) {
                *reinterpret_cast<float2*>(C_out + (long)r0 * N + c0) =
                    make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            } else if (r0 < M && c0 < N) {
                C_out[(long)r0 * N + c0] = acc[mi][ni][0];
            }
            if (r1 < M && c0 + 1 < N) {
                *reinterpret_cast<float2*>(C_out + (long)r1 * N + c0) =
                    make_float2(acc[mi][ni][2], acc[mi][ni][3]);
            } else if (r1 < M && c0 < N) {
                C_out[(long)r1 * N + c0] = acc[mi][ni][2];
            }
        }
    }
}

__global__ void __launch_bounds__(256, 4)
hgemm_reduce_kernel(
    const float * __restrict__ C_part,
    half        * __restrict__ C,
    int MN, int num_splits)
{
    int base = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base + 7 < MN) {
        float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        #pragma unroll
        for (int sp = 0; sp < NUM_SPLITS; sp++) {
            long off = (long)sp * MN + base;
            float4 v0 = __ldg(reinterpret_cast<const float4*>(C_part + off));
            float4 v1 = __ldg(reinterpret_cast<const float4*>(C_part + off + 4));
            s0 += v0.x; s1 += v0.y; s2 += v0.z; s3 += v0.w;
            s4 += v1.x; s5 += v1.y; s6 += v1.z; s7 += v1.w;
        }
        *reinterpret_cast<half2*>(C + base + 0) = __float22half2_rn(make_float2(s0, s1));
        *reinterpret_cast<half2*>(C + base + 2) = __float22half2_rn(make_float2(s2, s3));
        *reinterpret_cast<half2*>(C + base + 4) = __float22half2_rn(make_float2(s4, s5));
        *reinterpret_cast<half2*>(C + base + 6) = __float22half2_rn(make_float2(s6, s7));
    } else {
        for (int i = base; i < MN && i < base + 8; i++) {
            float s = 0.f;
            for (int sp = 0; sp < num_splits; sp++)
                s += C_part[(long)sp * MN + i];
            C[i] = __float2half(s);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* A_ptr  = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr  = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half*       C_ptr  = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int num_splits = NUM_SPLITS;
    int k_per_split = (K + num_splits - 1) / num_splits;
    k_per_split = ((k_per_split + BK - 1) / BK) * BK;
    int actual_splits = (K + k_per_split - 1) / k_per_split;

    const int grid_m = (M + BM - 1) / BM;

    constexpr size_t smem_bytes = (SA_TOTAL + SB_TOTAL) * sizeof(half);

    cudaFuncSetAttribute(hgemm_split_k_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    cudaFuncSetAttribute(hgemm_split_k_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    auto partial = torch::empty({(long)actual_splits * M * N}, opts);
    float* partial_ptr = partial.data_ptr<float>();

    dim3 grid(grid_m, actual_splits);
    dim3 block(BLOCK_SIZE);

    hgemm_split_k_kernel<<<grid, block, smem_bytes>>>(
        A_ptr, B_ptr, partial_ptr, M, N, K, k_per_split
    );

    const int MN = M * N;
    const int rthreads = 256;
    const int rblocks  = (MN / 8 + rthreads - 1) / rthreads;
    hgemm_reduce_kernel<<<rblocks, rthreads>>>(partial_ptr, C_ptr, MN, actual_splits);
}