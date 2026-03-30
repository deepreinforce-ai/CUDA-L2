#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define BM 64
#define BN 64
#define BK 32
#define WARP_M 32
#define WARP_N 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_MMA_M 2
#define WARP_MMA_N 4
#define STAGES 4
#define BLOCK_SIZE 128
#define SMEM_A_STRIDE 40
#define SMEM_B_STRIDE 72

__forceinline__ __device__ uint32_t smem_u32addr(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"((uint64_t)ptr)
    );
    return addr;
}

__global__ void __launch_bounds__(128, 6)
hgemm_optimized_v7(
    const half* __restrict__ A,
    const half* __restrict__ B_row,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __align__(128) half smem_A[STAGES][BM][SMEM_A_STRIDE];
    __shared__ __align__(128) half smem_B[STAGES][BK][SMEM_B_STRIDE];

    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;
    const int block_m = bm * BM;
    const int block_n = bn * BN;

    float acc[WARP_MMA_M][WARP_MMA_N][4];
    #pragma unroll
    for (int i = 0; i < WARP_MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int num_k_tiles = (K + BK - 1) / BK;

    const int a_load_idx0 = tid;
    const int a_load_idx1 = tid + 128;
    const int a_row0 = a_load_idx0 >> 2;
    const int a_col0 = (a_load_idx0 & 3) << 3;
    const int a_row1 = a_load_idx1 >> 2;
    const int a_col1 = (a_load_idx1 & 3) << 3;
    const int a_gr0 = block_m + a_row0;
    const int a_gr1 = block_m + a_row1;
    const uint32_t a_smem_off0_stage0 = smem_u32addr(&smem_A[0][a_row0][a_col0]);
    const uint32_t a_smem_off1_stage0 = smem_u32addr(&smem_A[0][a_row1][a_col1]);
    const uint32_t a_stage_stride = (uint32_t)(BM * SMEM_A_STRIDE * sizeof(half));
    const uint32_t b_stage_stride = (uint32_t)(BK * SMEM_B_STRIDE * sizeof(half));

    const int b_load_idx0 = tid;
    const int b_load_idx1 = tid + 128;
    const int b_row0 = b_load_idx0 >> 3;
    const int b_col0 = (b_load_idx0 & 7) << 3;
    const int b_row1 = b_load_idx1 >> 3;
    const int b_col1 = (b_load_idx1 & 7) << 3;
    const int b_gk0 = b_row0;
    const int b_gk1 = b_row1;
    const int b_gn0 = block_n + b_col0;
    const int b_gn1 = block_n + b_col1;
    const uint32_t b_smem_off0_stage0 = smem_u32addr(&smem_B[0][b_row0][b_col0]);
    const uint32_t b_smem_off1_stage0 = smem_u32addr(&smem_B[0][b_row1][b_col1]);

    const bool a_valid0 = (a_gr0 < M);
    const bool a_valid1 = (a_gr1 < M);
    const bool b_gn0_valid = (b_gn0 + 7 < N);
    const bool b_gn1_valid = (b_gn1 + 7 < N);

    auto load_A_async = [&](int stage, int k_tile) __attribute__((always_inline)) {
        const int k_base = k_tile * BK;
        const int gc0 = k_base + a_col0;
        const int gc1 = k_base + a_col1;
        const uint32_t d0 = a_smem_off0_stage0 + stage * a_stage_stride;
        const uint32_t d1 = a_smem_off1_stage0 + stage * a_stage_stride;

        if (a_valid0 && gc0 + 7 < K) {
            const half* src = A + a_gr0 * K + gc0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(d0), "l"(src));
        } else {
            half tmp[8] = {};
            if (a_valid0) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    if (gc0 + e < K) tmp[e] = A[a_gr0 * K + gc0 + e];
            }
            *reinterpret_cast<float4*>(smem_A[stage][a_row0] + a_col0) =
                *reinterpret_cast<const float4*>(tmp);
        }

        if (a_valid1 && gc1 + 7 < K) {
            const half* src = A + a_gr1 * K + gc1;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(d1), "l"(src));
        } else {
            half tmp[8] = {};
            if (a_valid1) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    if (gc1 + e < K) tmp[e] = A[a_gr1 * K + gc1 + e];
            }
            *reinterpret_cast<float4*>(smem_A[stage][a_row1] + a_col1) =
                *reinterpret_cast<const float4*>(tmp);
        }
    };

    auto load_B_async = [&](int stage, int k_tile) __attribute__((always_inline)) {
        const int k_base = k_tile * BK;
        const int gk0 = k_base + b_gk0;
        const int gk1 = k_base + b_gk1;
        const uint32_t d0 = b_smem_off0_stage0 + stage * b_stage_stride;
        const uint32_t d1 = b_smem_off1_stage0 + stage * b_stage_stride;

        if (gk0 < K && b_gn0_valid) {
            const half* src = B_row + gk0 * N + b_gn0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(d0), "l"(src));
        } else {
            half tmp[8] = {};
            if (gk0 < K) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    if (b_gn0 + e < N) tmp[e] = B_row[gk0 * N + b_gn0 + e];
            }
            *reinterpret_cast<float4*>(smem_B[stage][b_row0] + b_col0) =
                *reinterpret_cast<const float4*>(tmp);
        }

        if (gk1 < K && b_gn1_valid) {
            const half* src = B_row + gk1 * N + b_gn1;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(d1), "l"(src));
        } else {
            half tmp[8] = {};
            if (gk1 < K) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    if (b_gn1 + e < N) tmp[e] = B_row[gk1 * N + b_gn1 + e];
            }
            *reinterpret_cast<float4*>(smem_B[stage][b_row1] + b_col1) =
                *reinterpret_cast<const float4*>(tmp);
        }
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < num_k_tiles) {
            load_A_async(s, s);
            load_B_async(s, s);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    int read_stage = 0;
    int write_stage = STAGES - 1;

    const int a_frag_row_base = warp_m * WARP_M;
    const int lane_row_off = lane_id & 15;
    const int lane_col_off = (lane_id >> 4) << 3;

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();

        int next_k = k_tile + STAGES - 1;
        if (next_k < num_k_tiles) {
            load_A_async(write_stage, next_k);
            load_B_async(write_stage, next_k);
        }
        asm volatile("cp.async.commit_group;\n");
        write_stage = (write_stage + 1 >= STAGES) ? 0 : write_stage + 1;

        uint32_t a0[WARP_MMA_M][4], a1[WARP_MMA_M][4];
        uint32_t b0[WARP_MMA_N][2], b1[WARP_MMA_N][2];

        #pragma unroll
        for (int mi = 0; mi < WARP_MMA_M; mi++) {
            int row = a_frag_row_base + mi * MMA_M + lane_row_off;
            int col = lane_col_off;
            uint32_t addr = smem_u32addr(&smem_A[read_stage][row][col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0[mi][0]),"=r"(a0[mi][1]),"=r"(a0[mi][2]),"=r"(a0[mi][3])
                : "r"(addr));
        }

        #pragma unroll
        for (int ni = 0; ni < WARP_MMA_N; ni++) {
            int row = lane_row_off;
            int col = warp_n * WARP_N + ni * MMA_N;
            uint32_t addr = smem_u32addr(&smem_B[read_stage][row][col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0[ni][0]),"=r"(b0[ni][1])
                : "r"(addr));
        }

        #pragma unroll
        for (int mi = 0; mi < WARP_MMA_M; mi++) {
            int row = a_frag_row_base + mi * MMA_M + lane_row_off;
            int col = MMA_K + lane_col_off;
            uint32_t addr = smem_u32addr(&smem_A[read_stage][row][col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a1[mi][0]),"=r"(a1[mi][1]),"=r"(a1[mi][2]),"=r"(a1[mi][3])
                : "r"(addr));
        }

        #pragma unroll
        for (int ni = 0; ni < WARP_MMA_N; ni++) {
            int row = MMA_K + lane_row_off;
            int col = warp_n * WARP_N + ni * MMA_N;
            uint32_t addr = smem_u32addr(&smem_B[read_stage][row][col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b1[ni][0]),"=r"(b1[ni][1])
                : "r"(addr));
        }

        #pragma unroll
        for (int mi = 0; mi < WARP_MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < WARP_MMA_N; ni++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[mi][ni][0]),"+f"(acc[mi][ni][1]),
                      "+f"(acc[mi][ni][2]),"+f"(acc[mi][ni][3])
                    : "r"(a0[mi][0]),"r"(a0[mi][1]),"r"(a0[mi][2]),"r"(a0[mi][3]),
                      "r"(b0[ni][0]),"r"(b0[ni][1])
                );
            }
        }

        #pragma unroll
        for (int mi = 0; mi < WARP_MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < WARP_MMA_N; ni++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[mi][ni][0]),"+f"(acc[mi][ni][1]),
                      "+f"(acc[mi][ni][2]),"+f"(acc[mi][ni][3])
                    : "r"(a1[mi][0]),"r"(a1[mi][1]),"r"(a1[mi][2]),"r"(a1[mi][3]),
                      "r"(b1[ni][0]),"r"(b1[ni][1])
                );
            }
        }

        read_stage = (read_stage + 1 >= STAGES) ? 0 : read_stage + 1;
        __syncthreads();
    }

    const int warp_row_base = block_m + warp_m * WARP_M;
    const int warp_col_base = block_n + warp_n * WARP_N;
    const bool full_tile = (block_m + BM <= M) && (block_n + BN <= N);

    #pragma unroll
    for (int mi = 0; mi < WARP_MMA_M; mi++) {
        const int r0 = warp_row_base + mi * MMA_M + (lane_id >> 2);
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WARP_MMA_N; ni++) {
            const int c0 = warp_col_base + ni * MMA_N + (lane_id & 3) * 2;
            if (full_tile) {
                *reinterpret_cast<half2*>(&C[r0 * N + c0]) =
                    __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
                *reinterpret_cast<half2*>(&C[r1 * N + c0]) =
                    __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            } else {
                if (r0 < M) {
                    if (c0 + 1 < N) {
                        *reinterpret_cast<half2*>(&C[r0 * N + c0]) =
                            __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
                    } else if (c0 < N) {
                        C[r0 * N + c0] = __float2half(acc[mi][ni][0]);
                    }
                }
                if (r1 < M) {
                    if (c0 + 1 < N) {
                        *reinterpret_cast<half2*>(&C[r1 * N + c0]) =
                            __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
                    } else if (c0 < N) {
                        C[r1 * N + c0] = __float2half(acc[mi][ni][2]);
                    }
                }
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    TORCH_CHECK(a.dtype() == torch::kHalf);
    TORCH_CHECK(b.dtype() == torch::kHalf);
    TORCH_CHECK(c.dtype() == torch::kHalf);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_a = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_b = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_c = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BLOCK_SIZE);
    hgemm_optimized_v7<<<grid, block>>>(ptr_a, ptr_b, ptr_c, M, N, K);
}