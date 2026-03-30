#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>
#include <cooperative_groups.h>

#define BM 128
#define BN 64
#define BK 64
#define NWARPS 8
#define NTHREADS (NWARPS * 32)
#define SPLITK 33
#define NUM_STAGES 4

#define SA_STRIDE 80
#define SB_STRIDE 72

#define SA_SIZE (BM * SA_STRIDE)
#define SB_SIZE (BK * SB_STRIDE)
#define STAGE_SIZE (SA_SIZE + SB_SIZE)
#define SMEM_TOTAL_HALVES (NUM_STAGES * STAGE_SIZE)
#define SMEM_BYTES (SMEM_TOTAL_HALVES * 2)

__device__ __forceinline__ uint32_t cvt_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    uint32_t d = cvt_smem(dst);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(d), "l"((uint64_t)src)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void smem_zero16(void* dst) {
    uint32_t d = cvt_smem(dst);
    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};\n" :: "r"(d) : "memory");
}

__device__ __forceinline__ void issue_tile_inbounds(
    __half* __restrict__ dA, __half* __restrict__ dB,
    const __half* __restrict__ A, const __half* __restrict__ B,
    int m_base, int n_base, int k_off,
    int K, int tid
) {
    const __half* a_row_base = A + m_base * K + k_off;
    #pragma unroll 4
    for (int idx = tid; idx < BM * (BK / 8); idx += NTHREADS) {
        int row = idx >> 3;
        int col8 = idx & 7;
        cp_async_cg(
            dA + row * SA_STRIDE + col8 * 8,
            a_row_base + row * K + col8 * 8
        );
    }

    const __half* b_row_base = B + k_off * BN + n_base;
    #pragma unroll 2
    for (int idx = tid; idx < BK * (BN / 8); idx += NTHREADS) {
        int ki = idx >> 3;
        int col8 = idx & 7;
        cp_async_cg(
            dB + ki * SB_STRIDE + col8 * 8,
            b_row_base + ki * BN + col8 * 8
        );
    }

    cp_async_commit();
}

__device__ __forceinline__ void issue_tile_boundary(
    __half* __restrict__ dA, __half* __restrict__ dB,
    const __half* __restrict__ A, const __half* __restrict__ B,
    int m_base, int n_base, int k_off,
    int M, int N, int K, int tid
) {
    #pragma unroll 4
    for (int idx = tid; idx < BM * (BK / 8); idx += NTHREADS) {
        int row = idx >> 3;
        int col8 = idx & 7;
        int gk = k_off + col8 * 8;
        void* dst = dA + row * SA_STRIDE + col8 * 8;
        if (gk + 8 <= K) {
            cp_async_cg(dst, A + (m_base + row) * K + gk);
        } else {
            smem_zero16(dst);
            const int rem = K - gk;
            if (rem > 0) {
                __half* dptr = (__half*)dst;
                const __half* sptr = A + (m_base + row) * K + gk;
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    dptr[i] = (i < rem) ? sptr[i] : __float2half(0.f);
            }
        }
    }

    #pragma unroll 2
    for (int idx = tid; idx < BK * (BN / 8); idx += NTHREADS) {
        int ki = idx >> 3;
        int col8 = idx & 7;
        int gk = k_off + ki;
        void* dst = dB + ki * SB_STRIDE + col8 * 8;
        if (gk < K) {
            cp_async_cg(dst, B + gk * N + n_base + col8 * 8);
        } else {
            smem_zero16(dst);
        }
    }

    cp_async_commit();
}

__device__ __forceinline__ void compute_stage(
    const __half* __restrict__ cA,
    const __half* __restrict__ cB,
    int warp_m, int lane_id,
    float acc[8][4]
) {
    #pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
        uint32_t ra[4];
        {
            int row = warp_m + (lane_id & 15);
            int col = ks + ((lane_id >> 4) << 3);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
                : "r"(cvt_smem(cA + row * SA_STRIDE + col))
            );
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            uint32_t rb[2];
            {
                int row = ks + (lane_id & 15);
                int col = ni * 8;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(rb[0]), "=r"(rb[1])
                    : "r"(cvt_smem(cB + row * SB_STRIDE + col))
                );
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
                  "r"(rb[0]), "r"(rb[1])
            );
        }
    }
}

__global__ __launch_bounds__(NTHREADS, 1)
void hgemm_splitk_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ workspace,
    int M, int N, int K,
    int K_chunk
) {
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int bs = blockIdx.z;

    const int m_base = bm * BM;
    const int n_base = bn * BN;
    const int k_start = bs * K_chunk;
    if (k_start >= K) return;
    const int k_end = min(k_start + K_chunk, K);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id * 16;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.f;

    extern __shared__ __half smem[];

    __half* sA[NUM_STAGES];
    __half* sB[NUM_STAGES];
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        sA[s] = smem + s * STAGE_SIZE;
        sB[s] = smem + s * STAGE_SIZE + SA_SIZE;
    }

    const int num_tiles = (k_end - k_start + BK - 1) / BK;
    if (num_tiles == 0) return;

    const bool has_boundary = (k_start + num_tiles * BK > K);
    const int  safe_tiles   = has_boundary ? (num_tiles - 1) : num_tiles;

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < num_tiles) {
            int k_off = k_start + s * BK;
            if (s < safe_tiles) {
                issue_tile_inbounds(sA[s], sB[s], A, B, m_base, n_base, k_off, K, tid);
            } else {
                issue_tile_boundary(sA[s], sB[s], A, B, m_base, n_base, k_off, M, N, K, tid);
            }
        } else {
            cp_async_commit();
        }
    }

    for (int tile = 0; tile < num_tiles; tile++) {
        const int prefetch = tile + NUM_STAGES - 1;
        if (prefetch < num_tiles) {
            int k_off = k_start + prefetch * BK;
            if (prefetch < safe_tiles) {
                issue_tile_inbounds(sA[prefetch % NUM_STAGES], sB[prefetch % NUM_STAGES],
                                    A, B, m_base, n_base, k_off, K, tid);
            } else {
                issue_tile_boundary(sA[prefetch % NUM_STAGES], sB[prefetch % NUM_STAGES],
                                    A, B, m_base, n_base, k_off, M, N, K, tid);
            }
        } else {
            cp_async_commit();
        }

        cp_async_wait<NUM_STAGES - 1>();
        __syncthreads();

        compute_stage(sA[tile % NUM_STAGES], sB[tile % NUM_STAGES], warp_m, lane_id, acc);

        __syncthreads();
    }

    cp_async_wait<0>();

    float* ws = workspace + (size_t)bs * M * N;
    const int row0 = m_base + warp_m + (lane_id >> 2);
    const int row1 = row0 + 8;
    const int col_base = n_base + (lane_id & 3) * 2;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c0 = col_base + ni * 8;
        const int c1 = c0 + 1;
        *reinterpret_cast<float2*>(ws + row0 * N + c0) = make_float2(acc[ni][0], acc[ni][1]);
        *reinterpret_cast<float2*>(ws + row1 * N + c0) = make_float2(acc[ni][2], acc[ni][3]);
    }
}

template<int SK>
__global__ __launch_bounds__(256, 4)
void splitk_reduce_kernel(
    const float* __restrict__ ws,
    __half* __restrict__ C,
    int MN
) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= MN) return;

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

    #pragma unroll
    for (int s = 0; s < SK; s++) {
        const float4 v = *reinterpret_cast<const float4*>(ws + (size_t)s * MN + base);
        s0 += v.x;
        s1 += v.y;
        s2 += v.z;
        s3 += v.w;
    }

    *reinterpret_cast<__half2*>(C + base)     = __floats2half2_rn(s0, s1);
    *reinterpret_cast<__half2*>(C + base + 2) = __floats2half2_rn(s2, s3);
}

static float* g_workspace = nullptr;
static size_t g_workspace_bytes = 0;

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor b_col_major, torch::Tensor c
) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* pA = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* pB = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    __half*       pC = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    constexpr int SK = SPLITK;

    int K_chunk_raw = (K + SK - 1) / SK;
    int K_chunk = ((K_chunk_raw + BK - 1) / BK) * BK;

    const size_t ws_bytes = (size_t)SK * M * N * sizeof(float);
    if (ws_bytes > g_workspace_bytes) {
        if (g_workspace) cudaFree(g_workspace);
        cudaMalloc(&g_workspace, ws_bytes);
        g_workspace_bytes = ws_bytes;
    }

    const int m_tiles = (M + BM - 1) / BM;
    const int n_tiles = (N + BN - 1) / BN;

    dim3 grid(m_tiles, n_tiles, SK);
    dim3 block(NTHREADS);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_splitk_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        attr_set = true;
    }

    hgemm_splitk_kernel<<<grid, block, SMEM_BYTES>>>(
        pA, pB, g_workspace, M, N, K, K_chunk
    );

    const int MN = M * N;
    splitk_reduce_kernel<SK><<<MN / (256 * 4), 256>>>(g_workspace, pC, MN);
}