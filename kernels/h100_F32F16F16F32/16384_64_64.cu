#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

__global__ void __launch_bounds__(32, 16)
hgemm_v3_main(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    int M
) {
    const int lane_id   = threadIdx.x;
    const int block_row = blockIdx.x * 16;
    if (block_row >= M) return;

    __shared__ __align__(128) half smem_A[16][80];

    #pragma unroll
    for (int i = lane_id; i < 128; i += 32) {
        const int row   = i >> 3;
        const int col8  = i & 7;
        const int scol8 = col8 ^ (row & 7);
        const int gr    = block_row + row;
        half* dst = &smem_A[row][scol8 * 8];
        if (gr < M) {
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
                   "l"((uint64_t)(A + (uint64_t)gr * 64 + col8 * 8))
                : "memory"
            );
        } else {
            asm volatile(
                "st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
                   "r"(0), "r"(0), "r"(0), "r"(0)
                : "memory"
            );
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    uint32_t b_frag[4][8][2];

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int n = ni * 8 + (lane_id >> 2);
        const half* brow = B_col_major + (uint64_t)n * 64;
        uint4 v0, v1, v2, v3;
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(v0.x),"=r"(v0.y),"=r"(v0.z),"=r"(v0.w)
            : "l"((uint64_t)(brow + 0)));
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(v1.x),"=r"(v1.y),"=r"(v1.z),"=r"(v1.w)
            : "l"((uint64_t)(brow + 16)));
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(v2.x),"=r"(v2.y),"=r"(v2.z),"=r"(v2.w)
            : "l"((uint64_t)(brow + 32)));
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(v3.x),"=r"(v3.y),"=r"(v3.z),"=r"(v3.w)
            : "l"((uint64_t)(brow + 48)));

        const int koff = lane_id & 3;
        (void)v0; (void)v1; (void)v2; (void)v3;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k0 = ki * 16 + koff * 2;
            const int k1 = k0 + 8;
            b_frag[ki][ni][0] = *reinterpret_cast<const uint32_t*>(&brow[k0]);
            b_frag[ki][ni][1] = *reinterpret_cast<const uint32_t*>(&brow[k1]);
        }
    }

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();

    uint32_t a_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        int row, logical_col8;
        if (lane_id < 16) {
            row = lane_id;
            logical_col8 = ki * 2;
        } else {
            row = lane_id - 16;
            logical_col8 = ki * 2 + 1;
        }
        const int actual_col8 = logical_col8 ^ (row & 7);
        const uint32_t smem_addr = __cvta_generic_to_shared(&smem_A[row][actual_col8 * 8]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.row.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_frag[ki][0]), "=r"(a_frag[ki][1]),
              "=r"(a_frag[ki][2]), "=r"(a_frag[ki][3])
            : "r"(smem_addr)
        );
    }

    float acc[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(a_frag[ki][0]), "r"(a_frag[ki][1]),
                  "r"(a_frag[ki][2]), "r"(a_frag[ki][3]),
                  "r"(b_frag[ki][ni][0]), "r"(b_frag[ki][ni][1])
            );
        }
    }

    const int out_row0 = block_row + (lane_id >> 2);
    const int out_row1 = out_row0 + 8;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int out_col = ni * 8 + (lane_id & 3) * 2;
        if (out_row0 < M)
            *reinterpret_cast<half2*>(&C[out_row0 * 64 + out_col]) =
                __floats2half2_rn(acc[ni][0], acc[ni][1]);
        if (out_row1 < M)
            *reinterpret_cast<half2*>(&C[out_row1 * 64 + out_col]) =
                __floats2half2_rn(acc[ni][2], acc[ni][3]);
    }
}

__global__ void __launch_bounds__(32, 16)
hgemm_v3_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    int M,
    int num_tiles
) {
    const int lane_id = threadIdx.x;
    const int stride  = gridDim.x;
    const int koff    = lane_id & 3;

    uint32_t b_frag[4][8][2];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int n = ni * 8 + (lane_id >> 2);
        const half* brow = B_col_major + (uint64_t)n * 64;
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            const int k0 = ki * 16 + koff * 2;
            const int k1 = k0 + 8;
            b_frag[ki][ni][0] = *reinterpret_cast<const uint32_t*>(&brow[k0]);
            b_frag[ki][ni][1] = *reinterpret_cast<const uint32_t*>(&brow[k1]);
        }
    }

    __shared__ __align__(128) half smem_A[2][16][80];

    auto issue_load = [&](int buf, int base_row) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = lane_id; i < 128; i += 32) {
            const int row   = i >> 3;
            const int col8  = i & 7;
            const int scol8 = col8 ^ (row & 7);
            const int gr    = base_row + row;
            half* dst = &smem_A[buf][row][scol8 * 8];
            if (gr < M) {
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
                       "l"((uint64_t)(A + (uint64_t)gr * 64 + col8 * 8))
                    : "memory"
                );
            } else {
                asm volatile(
                    "st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n"
                    :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
                       "r"(0), "r"(0), "r"(0), "r"(0)
                    : "memory"
                );
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    const int first = blockIdx.x;
    if (first < num_tiles) issue_load(0, first * 16);

    int cur_buf = 0;
    for (int cur = first; cur < num_tiles; cur += stride) {
        const int next    = cur + stride;
        const int nxt_buf = cur_buf ^ 1;

        if (next < num_tiles) {
            issue_load(nxt_buf, next * 16);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncwarp();

        uint32_t a_frag[4][4];
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            int row, logical_col8;
            if (lane_id < 16) {
                row = lane_id;
                logical_col8 = ki * 2;
            } else {
                row = lane_id - 16;
                logical_col8 = ki * 2 + 1;
            }
            const int actual_col8 = logical_col8 ^ (row & 7);
            const uint32_t smem_addr = __cvta_generic_to_shared(&smem_A[cur_buf][row][actual_col8 * 8]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.row.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[ki][0]), "=r"(a_frag[ki][1]),
                  "=r"(a_frag[ki][2]), "=r"(a_frag[ki][3])
                : "r"(smem_addr)
            );
        }

        float acc[8][4];
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                      "+f"(acc[ni][2]), "+f"(acc[ni][3])
                    : "r"(a_frag[ki][0]), "r"(a_frag[ki][1]),
                      "r"(a_frag[ki][2]), "r"(a_frag[ki][3]),
                      "r"(b_frag[ki][ni][0]), "r"(b_frag[ki][ni][1])
                );
            }
        }

        const int block_row = cur * 16;
        const int out_row0  = block_row + (lane_id >> 2);
        const int out_row1  = out_row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int out_col = ni * 8 + (lane_id & 3) * 2;
            if (out_row0 < M)
                *reinterpret_cast<half2*>(&C[out_row0 * 64 + out_col]) =
                    __floats2half2_rn(acc[ni][0], acc[ni][1]);
            if (out_row1 < M)
                *reinterpret_cast<half2*>(&C[out_row1 * 64 + out_col]) =
                    __floats2half2_rn(acc[ni][2], acc[ni][3]);
        }

        cur_buf = nxt_buf;
    }
}

__global__ void __launch_bounds__(256, 1)
hgemm_bm128_v3(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    int M
) {
    const int lane_id   = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;
    const int block_row = blockIdx.x * 128;

    __shared__ __align__(128) half smem_BT[64][72];
    __shared__ __align__(128) half smem_A[8][16][72];

    #pragma unroll
    for (int i = threadIdx.x; i < 512; i += 256) {
        const int n    = i >> 3;
        const int col8 = i & 7;
        *reinterpret_cast<float4*>(&smem_BT[n][col8 * 8]) =
            *reinterpret_cast<const float4*>(&B_col_major[n * 64 + col8 * 8]);
    }
    __syncthreads();

    uint32_t b_frag[4][8][2];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        const int k0 = ki * 16 + (lane_id & 3) * 2;
        const int k1 = k0 + 8;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            const int n = ni * 8 + (lane_id >> 2);
            uint32_t r0, r1;
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(r0)
                : "r"((uint32_t)__cvta_generic_to_shared(&smem_BT[n][k0])));
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(r1)
                : "r"((uint32_t)__cvta_generic_to_shared(&smem_BT[n][k1])));
            b_frag[ki][ni][0] = r0;
            b_frag[ki][ni][1] = r1;
        }
    }

    const int warp_row_base = block_row + warp_id * 16;
    #pragma unroll
    for (int i = lane_id; i < 128; i += 32) {
        const int row        = i >> 3;
        const int logical_c8 = i & 7;
        const int actual_c8  = logical_c8 ^ (row & 7);
        const int gr         = warp_row_base + row;
        half* dst = &smem_A[warp_id][row][actual_c8 * 8];
        if (gr < M) {
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
                   "l"((uint64_t)(A + (uint64_t)gr * 64 + logical_c8 * 8))
                : "memory"
            );
        } else {
            asm volatile(
                "st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
                   "r"(0), "r"(0), "r"(0), "r"(0)
                : "memory"
            );
        }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();

    uint32_t a_frag[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        int row, logical_c8;
        if (lane_id < 16) { row = lane_id;      logical_c8 = ki * 2; }
        else               { row = lane_id - 16; logical_c8 = ki * 2 + 1; }
        const int actual_c8  = logical_c8 ^ (row & 7);
        const uint32_t saddr = __cvta_generic_to_shared(&smem_A[warp_id][row][actual_c8 * 8]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.row.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_frag[ki][0]), "=r"(a_frag[ki][1]),
              "=r"(a_frag[ki][2]), "=r"(a_frag[ki][3])
            : "r"(saddr)
        );
    }

    float acc[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(a_frag[ki][0]), "r"(a_frag[ki][1]),
                  "r"(a_frag[ki][2]), "r"(a_frag[ki][3]),
                  "r"(b_frag[ki][ni][0]), "r"(b_frag[ki][ni][1])
            );
        }
    }

    const int out_row0 = warp_row_base + (lane_id >> 2);
    const int out_row1 = out_row0 + 8;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int out_col = ni * 8 + (lane_id & 3) * 2;
        if (out_row0 < M)
            *reinterpret_cast<half2*>(&C[out_row0 * 64 + out_col]) =
                __floats2half2_rn(acc[ni][0], acc[ni][1]);
        if (out_row1 < M)
            *reinterpret_cast<half2*>(&C[out_row1 * 64 + out_col]) =
                __floats2half2_rn(acc[ni][2], acc[ni][3]);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half*       C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int num_tiles_16 = (M + 15) / 16;
    const int max_slots_16 = 132 * 16;

    const int num_tiles_128 = (M + 127) / 128;
    const int max_slots_128 = 132;

    if (num_tiles_128 <= max_slots_128) {
        hgemm_bm128_v3<<<num_tiles_128, 256>>>(A_ptr, B_ptr, C_ptr, M);
    } else if (num_tiles_16 <= max_slots_16) {
        hgemm_v3_main<<<num_tiles_16, 32>>>(A_ptr, B_ptr, C_ptr, M);
    } else {
        const int grid = min(max_slots_16, num_tiles_16);
        hgemm_v3_persistent<<<grid, 32>>>(A_ptr, B_ptr, C_ptr, M, num_tiles_16);
    }
}