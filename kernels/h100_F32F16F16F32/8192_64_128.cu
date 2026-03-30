#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define BM 64
#define BN 64
#define BK 128
#define NWARPS 4
#define NTHREADS 128

#define SA_STRIDE 128
#define SB_STRIDE 136

#define SMEM_A_BYTES (BM * SA_STRIDE * 2)
#define SMEM_B_BYTES (BN * SB_STRIDE * 2)
#define SMEM_TOTAL   (SMEM_A_BYTES + SMEM_B_BYTES)

static __device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src) : "memory");
}

static __device__ __forceinline__ void cp_async16_zfill(uint32_t dst, const void* src, bool valid) {
    if (valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst), "l"(src) : "memory");
    } else {
        asm volatile("st.shared.v4.b32 [%0], {0,0,0,0};\n" :: "r"(dst) : "memory");
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_kernel_bm64(
    const half* __restrict__ A,
    const half* __restrict__ B_nk,
    half* __restrict__ C,
    int M,
    int num_tiles
) {
    extern __shared__ char smem_raw[];

    half (*sA)[SA_STRIDE] = reinterpret_cast<half(*)[SA_STRIDE]>(smem_raw);
    half (*sB)[SB_STRIDE] = reinterpret_cast<half(*)[SB_STRIDE]>(smem_raw + SMEM_A_BYTES);

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;

    {
        const int total = BN * BK / 8;
        #pragma unroll 8
        for (int i = tid; i < total; i += NTHREADS) {
            int flat = i * 8;
            int n    = flat / BK;
            int k    = flat % BK;
            cp_async16(smem_u32(&sB[n][k]), &B_nk[n * BK + k]);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    for (int tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
        const int block_row = tile_idx * BM;

        {
            const int total = BM * BK / 8;
            #pragma unroll 8
            for (int i = tid; i < total; i += NTHREADS) {
                int flat  = i * 8;
                int row   = flat / BK;
                int col   = flat % BK;
                int g_row = block_row + row;
                int chunk = col >> 3;
                int sw_chunk = chunk ^ (row & 15);
                int sw_col = sw_chunk << 3;
                uint32_t dst = smem_u32(&sA[row][sw_col]);
                cp_async16_zfill(dst, &A[g_row * BK + col], g_row < M);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();

        const int warp_row   = warp_id * 16;
        const int g_row_base = block_row + warp_row;

        float acc[8][4];
        #pragma unroll
        for (int ns = 0; ns < 8; ns++)
            acc[ns][0] = acc[ns][1] = acc[ns][2] = acc[ns][3] = 0.0f;

        #pragma unroll 8
        for (int k_tile = 0; k_tile < 8; k_tile++) {
            uint32_t ra[4];
            {
                int row_in_warp = lane_id & 15;
                int abs_row     = warp_row + row_in_warp;
                int col_base    = k_tile * 16 + ((lane_id >> 4) & 1) * 8;
                int chunk       = col_base >> 3;
                int sw_chunk    = chunk ^ (abs_row & 15);
                int sw_col      = sw_chunk << 3;
                uint32_t addr   = smem_u32(&sA[abs_row][sw_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
                    : "r"(addr)
                );
            }

            #pragma unroll 8
            for (int ns = 0; ns < 8; ns++) {
                uint32_t rb[2];
                {
                    int n_row = ns * 8 + (lane_id & 7);
                    int k_col = k_tile * 16 + ((lane_id >> 3) & 1) * 8;
                    uint32_t addr = smem_u32(&sB[n_row][k_col]);
                    asm volatile(
                        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb[0]), "=r"(rb[1])
                        : "r"(addr)
                    );
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(acc[ns][0]), "=f"(acc[ns][1]),
                      "=f"(acc[ns][2]), "=f"(acc[ns][3])
                    : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
                      "r"(rb[0]), "r"(rb[1]),
                      "f"(acc[ns][0]), "f"(acc[ns][1]),
                      "f"(acc[ns][2]), "f"(acc[ns][3])
                );
            }
        }

        {
            const int r0    = (lane_id >> 2);
            const int r1    = r0 + 8;
            const int c_off = (lane_id & 3) * 2;
            const int gR0   = g_row_base + r0;
            const int gR1   = g_row_base + r1;
            const bool v0   = (gR0 < M);
            const bool v1   = (gR1 < M);

            #pragma unroll 8
            for (int ns = 0; ns < 8; ns++) {
                int col = ns * 8 + c_off;
                if (v0) {
                    half2 h = __float22half2_rn(make_float2(acc[ns][0], acc[ns][1]));
                    *reinterpret_cast<half2*>(&C[gR0 * BN + col]) = h;
                }
                if (v1) {
                    half2 h = __float22half2_rn(make_float2(acc[ns][2], acc[ns][3]));
                    *reinterpret_cast<half2*>(&C[gR1 * BN + col]) = h;
                }
            }
        }

        __syncthreads();
    }
}

__global__ void __launch_bounds__(128, 3)
hgemm_kernel_bm64_db(
    const half* __restrict__ A,
    const half* __restrict__ B_nk,
    half* __restrict__ C,
    int M,
    int num_tiles
) {
    const int A_BUF = BM * SA_STRIDE * 2;
    extern __shared__ char smem_raw[];

    half (*sA0)[SA_STRIDE] = reinterpret_cast<half(*)[SA_STRIDE]>(smem_raw);
    half (*sA1)[SA_STRIDE] = reinterpret_cast<half(*)[SA_STRIDE]>(smem_raw + A_BUF);
    half (*sB)[SB_STRIDE]  = reinterpret_cast<half(*)[SB_STRIDE]>(smem_raw + 2 * A_BUF);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    {
        const int total = BN * BK / 8;
        #pragma unroll 8
        for (int i = tid; i < total; i += NTHREADS) {
            int flat = i * 8;
            int n    = flat / BK;
            int k    = flat % BK;
            cp_async16(smem_u32(&sB[n][k]), &B_nk[n * BK + k]);
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    auto load_A = [&](half (*buf)[SA_STRIDE], int block_row) {
        const int total = BM * BK / 8;
        #pragma unroll 8
        for (int i = tid; i < total; i += NTHREADS) {
            int flat     = i * 8;
            int row      = flat / BK;
            int col      = flat % BK;
            int g_row    = block_row + row;
            int sw_col   = (((col >> 3) ^ (row & 15)) << 3);
            uint32_t dst = smem_u32(&buf[row][sw_col]);
            cp_async16_zfill(dst, &A[g_row * BK + col], g_row < M);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    int first_tile = blockIdx.x;
    if (first_tile < num_tiles) {
        load_A(sA0, first_tile * BM);
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    half (*sA_cur)[SA_STRIDE] = sA0;
    half (*sA_nxt)[SA_STRIDE] = sA1;

    for (int tile_idx = first_tile; tile_idx < num_tiles; tile_idx += gridDim.x) {
        const int block_row  = tile_idx * BM;
        const int next_tile  = tile_idx + gridDim.x;

        if (next_tile < num_tiles) {
            load_A(sA_nxt, next_tile * BM);
        }

        const int warp_row   = warp_id * 16;
        const int g_row_base = block_row + warp_row;

        float acc[8][4];
        #pragma unroll
        for (int ns = 0; ns < 8; ns++)
            acc[ns][0] = acc[ns][1] = acc[ns][2] = acc[ns][3] = 0.0f;

        #pragma unroll 8
        for (int k_tile = 0; k_tile < 8; k_tile++) {
            uint32_t ra[4];
            {
                int row_in_warp = lane_id & 15;
                int abs_row     = warp_row + row_in_warp;
                int col_base    = k_tile * 16 + ((lane_id >> 4) & 1) * 8;
                int sw_col      = (((col_base >> 3) ^ (abs_row & 15)) << 3);
                uint32_t addr   = smem_u32(&sA_cur[abs_row][sw_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
                    : "r"(addr)
                );
            }

            #pragma unroll 8
            for (int ns = 0; ns < 8; ns++) {
                uint32_t rb[2];
                {
                    int n_row = ns * 8 + (lane_id & 7);
                    int k_col = k_tile * 16 + ((lane_id >> 3) & 1) * 8;
                    uint32_t addr = smem_u32(&sB[n_row][k_col]);
                    asm volatile(
                        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(rb[0]), "=r"(rb[1])
                        : "r"(addr)
                    );
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(acc[ns][0]), "=f"(acc[ns][1]),
                      "=f"(acc[ns][2]), "=f"(acc[ns][3])
                    : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
                      "r"(rb[0]), "r"(rb[1]),
                      "f"(acc[ns][0]), "f"(acc[ns][1]),
                      "f"(acc[ns][2]), "f"(acc[ns][3])
                );
            }
        }

        {
            const int r0    = (lane_id >> 2);
            const int r1    = r0 + 8;
            const int c_off = (lane_id & 3) * 2;
            const int gR0   = g_row_base + r0;
            const int gR1   = g_row_base + r1;
            const bool v0   = (gR0 < M);
            const bool v1   = (gR1 < M);

            #pragma unroll 8
            for (int ns = 0; ns < 8; ns++) {
                int col = ns * 8 + c_off;
                if (v0) {
                    half2 h = __float22half2_rn(make_float2(acc[ns][0], acc[ns][1]));
                    *reinterpret_cast<half2*>(&C[gR0 * BN + col]) = h;
                }
                if (v1) {
                    half2 h = __float22half2_rn(make_float2(acc[ns][2], acc[ns][3]));
                    *reinterpret_cast<half2*>(&C[gR1 * BN + col]) = h;
                }
            }
        }

        if (next_tile < num_tiles) {
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            __syncthreads();
        }

        half (*tmp)[SA_STRIDE] = sA_cur;
        sA_cur = sA_nxt;
        sA_nxt = tmp;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);

    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int num_tiles = (M + BM - 1) / BM;

    const int SMEM_SB = SMEM_TOTAL;
    const int A_BUF   = BM * SA_STRIDE * 2;
    const int SMEM_DB = 2 * A_BUF + SMEM_B_BYTES;

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(hgemm_kernel_bm64,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_SB);
        cudaFuncSetAttribute(hgemm_kernel_bm64,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(hgemm_kernel_bm64_db,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_DB);
        cudaFuncSetAttribute(hgemm_kernel_bm64_db,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        configured = true;
    }

    hgemm_kernel_bm64_db<<<num_tiles, NTHREADS, SMEM_DB>>>(A_ptr, B_ptr, C_ptr, M, num_tiles);
}