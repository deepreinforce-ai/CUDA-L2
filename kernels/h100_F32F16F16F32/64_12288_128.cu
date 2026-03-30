#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>
#include <stdint.h>

__device__ __forceinline__ int swz(int k, int row) {
    return (((k >> 3) ^ (row & 7)) << 3) + (k & 7);
}

__device__ __forceinline__ void st_cs32(__half* ptr, float v0, float v1) {
    __half h0 = __float2half(v0), h1 = __float2half(v1);
    uint32_t p = (uint32_t)(*reinterpret_cast<const uint16_t*>(&h0)) |
                 ((uint32_t)(*reinterpret_cast<const uint16_t*>(&h1)) << 16);
    asm volatile("st.global.cs.b32 [%0], %1;\n" :: "l"(ptr), "r"(p));
}

__device__ __forceinline__ void st_cs64(__half* ptr, float v0, float v1, float v2, float v3) {
    __half h0 = __float2half(v0), h1 = __float2half(v1);
    __half h2 = __float2half(v2), h3 = __float2half(v3);
    uint64_t p = (uint64_t)(*reinterpret_cast<const uint16_t*>(&h0)) |
                 ((uint64_t)(*reinterpret_cast<const uint16_t*>(&h1)) << 16) |
                 ((uint64_t)(*reinterpret_cast<const uint16_t*>(&h2)) << 32) |
                 ((uint64_t)(*reinterpret_cast<const uint16_t*>(&h3)) << 48);
    asm volatile("st.global.cs.b64 [%0], %1;\n" :: "l"(ptr), "l"(p));
}

__global__ void __launch_bounds__(128, 8)
hgemm_persistent_v8(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int N,
    int n_tiles
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int lane_row    = lane >> 2;
    const int lane_k      = (lane & 3) * 2;
    const int warp_n_base = warp_id * 16;

    __shared__ __half sA[64][128];
    __shared__ __half sB[64][128];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int row = idx >> 4, cg = idx & 15;
        uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sA[row][(( cg ^ (row & 7)) << 3)]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(sm), "l"(&A[row * 128 + (cg << 3)]));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    uint32_t a_reg[8][4][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int r0 = mt * 16 + lane_row, r1 = r0 + 8;
            a_reg[k][mt][0] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk0, r0)]);
            a_reg[k][mt][1] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk0, r1)]);
            a_reg[k][mt][2] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk8, r0)]);
            a_reg[k][mt][3] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk8, r1)]);
        }
    }

    for (int bcol = blockIdx.x; bcol < n_tiles; bcol += gridDim.x) {
        const int B_n_base = bcol * 64;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int nl  = idx >> 4, cg = idx & 15;
            int gn  = B_n_base + nl;
            if (gn < N) {
                uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sB[nl][((cg ^ (nl & 7)) << 3)]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                    "r"(sm), "l"(&B_col[gn * 128 + (cg << 3)]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        uint32_t b_reg[8][2][2];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int nc = warp_n_base + nt * 8 + lane_row;
                b_reg[k][nt][0] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk0, nc)]);
                b_reg[k][nt][1] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk8, nc)]);
            }
        }

        float acc[4][2][4];
        #pragma unroll
        for (int mt = 0; mt < 4; mt++)
            #pragma unroll
            for (int nt = 0; nt < 2; nt++)
                acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                #pragma unroll
                for (int mt = 0; mt < 4; mt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                          "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                        : "r"(a_reg[k][mt][0]), "r"(a_reg[k][mt][1]),
                          "r"(a_reg[k][mt][2]), "r"(a_reg[k][mt][3]),
                          "r"(b_reg[k][nt][0]), "r"(b_reg[k][nt][1])
                    );
                }
            }
        }

        const int C_n_base = bcol * 64;
        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int m0 = mt * 16 + lane_row;
            int m1 = m0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int n0 = C_n_base + warp_n_base + nt * 8 + lane_k;
                st_cs32(&C[m0 * N + n0], acc[mt][nt][0], acc[mt][nt][1]);
                st_cs32(&C[m1 * N + n0], acc[mt][nt][2], acc[mt][nt][3]);
            }
        }

        __syncthreads();
    }
}

__global__ void __launch_bounds__(128, 8)
hgemm_static_bn64(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int N
) {
    const int bcol    = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int lane_row    = lane >> 2;
    const int lane_k      = (lane & 3) * 2;
    const int warp_n_base = warp_id * 16;

    __shared__ __half sA[64][128];
    __shared__ __half sB[64][128];

    const int B_n_base = bcol * 64;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int row = idx >> 4, cg = idx & 15;
        uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sA[row][((cg ^ (row & 7)) << 3)]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(sm), "l"(&A[row * 128 + (cg << 3)]));
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int nl  = idx >> 4, cg = idx & 15;
        int gn  = B_n_base + nl;
        if (gn < N) {
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sB[nl][((cg ^ (nl & 7)) << 3)]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(sm), "l"(&B_col[gn * 128 + (cg << 3)]));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    uint32_t a_reg[8][4][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int r0 = mt * 16 + lane_row, r1 = r0 + 8;
            a_reg[k][mt][0] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk0, r0)]);
            a_reg[k][mt][1] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk0, r1)]);
            a_reg[k][mt][2] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk8, r0)]);
            a_reg[k][mt][3] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk8, r1)]);
        }
    }

    uint32_t b_reg[8][2][2];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            int nc = warp_n_base + nt * 8 + lane_row;
            b_reg[k][nt][0] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk0, nc)]);
            b_reg[k][nt][1] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk8, nc)]);
        }
    }

    float acc[4][2][4];
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
        #pragma unroll
        for (int nt = 0; nt < 2; nt++)
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            #pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                      "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                    : "r"(a_reg[k][mt][0]), "r"(a_reg[k][mt][1]),
                      "r"(a_reg[k][mt][2]), "r"(a_reg[k][mt][3]),
                      "r"(b_reg[k][nt][0]), "r"(b_reg[k][nt][1])
                );
            }
        }
    }

    const int C_n_base = bcol * 64;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        int m0 = mt * 16 + lane_row, m1 = m0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            int n0 = C_n_base + warp_n_base + nt * 8 + lane_k;
            st_cs32(&C[m0 * N + n0], acc[mt][nt][0], acc[mt][nt][1]);
            st_cs32(&C[m1 * N + n0], acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

__global__ void __launch_bounds__(128, 6)
hgemm_static_bn128(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int N
) {
    const int bcol    = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int lane_row    = lane >> 2;
    const int lane_k      = (lane & 3) * 2;
    const int warp_n_base = warp_id * 32;

    __shared__ __half sA[64][128];
    __shared__ __half sB[128][128];

    const int B_n_base = bcol * 128;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int row = idx >> 4, cg = idx & 15;
        uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sA[row][((cg ^ (row & 7)) << 3)]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(sm), "l"(&A[row * 128 + (cg << 3)]));
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = tid + i * 128;
        int nl  = idx >> 4, cg = idx & 15;
        int gn  = B_n_base + nl;
        if (gn < N) {
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sB[nl][((cg ^ (nl & 7)) << 3)]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(sm), "l"(&B_col[gn * 128 + (cg << 3)]));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    uint32_t a_reg[8][4][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int r0 = mt * 16 + lane_row, r1 = r0 + 8;
            a_reg[k][mt][0] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk0, r0)]);
            a_reg[k][mt][1] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk0, r1)]);
            a_reg[k][mt][2] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk8, r0)]);
            a_reg[k][mt][3] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk8, r1)]);
        }
    }

    uint32_t b_reg[8][4][2];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            int nc = warp_n_base + nt * 8 + lane_row;
            b_reg[k][nt][0] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk0, nc)]);
            b_reg[k][nt][1] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk8, nc)]);
        }
    }

    float acc[4][4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            #pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                      "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                    : "r"(a_reg[k][mt][0]), "r"(a_reg[k][mt][1]),
                      "r"(a_reg[k][mt][2]), "r"(a_reg[k][mt][3]),
                      "r"(b_reg[k][nt][0]), "r"(b_reg[k][nt][1])
                );
            }
        }
    }

    const int C_n_base = bcol * 128;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        int m0 = mt * 16 + lane_row, m1 = m0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            int n0 = C_n_base + warp_n_base + nt * 8 + lane_k;
            st_cs32(&C[m0 * N + n0], acc[mt][nt][0], acc[mt][nt][1]);
            st_cs32(&C[m1 * N + n0], acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

__global__ void __launch_bounds__(256, 4)
hgemm_static_bn64_256t(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int N
) {
    const int bcol    = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int warp_row = warp_id >> 2;
    const int warp_col = warp_id & 3;

    const int lane_row = lane >> 2;
    const int lane_k   = (lane & 3) * 2;
    const int m_base   = warp_row * 32;
    const int n_base   = warp_col * 16;

    __shared__ __half sA[64][128];
    __shared__ __half sB[64][128];

    const int B_n_base = bcol * 64;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 256;
        int row = idx >> 4, cg = idx & 15;
        uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sA[row][((cg ^ (row & 7)) << 3)]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(sm), "l"(&A[row * 128 + (cg << 3)]));
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 256;
        int nl  = idx >> 4, cg = idx & 15;
        int gn  = B_n_base + nl;
        if (gn < N) {
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sB[nl][((cg ^ (nl & 7)) << 3)]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(sm), "l"(&B_col[gn * 128 + (cg << 3)]));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    uint32_t a_reg[8][2][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            int r0 = m_base + mt * 16 + lane_row, r1 = r0 + 8;
            a_reg[k][mt][0] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk0, r0)]);
            a_reg[k][mt][1] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk0, r1)]);
            a_reg[k][mt][2] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk8, r0)]);
            a_reg[k][mt][3] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk8, r1)]);
        }
    }

    uint32_t b_reg[8][2][2];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            int nc = n_base + nt * 8 + lane_row;
            b_reg[k][nt][0] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk0, nc)]);
            b_reg[k][nt][1] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk8, nc)]);
        }
    }

    float acc[2][2][4];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 2; nt++)
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                      "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                    : "r"(a_reg[k][mt][0]), "r"(a_reg[k][mt][1]),
                      "r"(a_reg[k][mt][2]), "r"(a_reg[k][mt][3]),
                      "r"(b_reg[k][nt][0]), "r"(b_reg[k][nt][1])
                );
            }
        }
    }

    const int C_n_base = bcol * 64;
    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        int m0 = m_base + mt * 16 + lane_row, m1 = m0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            int n0 = C_n_base + n_base + nt * 8 + lane_k;
            st_cs32(&C[m0 * N + n0], acc[mt][nt][0], acc[mt][nt][1]);
            st_cs32(&C[m1 * N + n0], acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

__global__ void __launch_bounds__(256, 3)
hgemm_static_bn128_256t(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int N
) {
    const int bcol    = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int warp_row = warp_id >> 2;
    const int warp_col = warp_id & 3;

    const int lane_row = lane >> 2;
    const int lane_k   = (lane & 3) * 2;
    const int m_base   = warp_row * 32;
    const int n_base   = warp_col * 32;

    __shared__ __half sA[64][128];
    __shared__ __half sB[128][128];

    const int B_n_base = bcol * 128;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * 256;
        int row = idx >> 4, cg = idx & 15;
        uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sA[row][((cg ^ (row & 7)) << 3)]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(sm), "l"(&A[row * 128 + (cg << 3)]));
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 256;
        int nl  = idx >> 4, cg = idx & 15;
        int gn  = B_n_base + nl;
        if (gn < N) {
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sB[nl][((cg ^ (nl & 7)) << 3)]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                "r"(sm), "l"(&B_col[gn * 128 + (cg << 3)]));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    uint32_t a_reg[8][2][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            int r0 = m_base + mt * 16 + lane_row, r1 = r0 + 8;
            a_reg[k][mt][0] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk0, r0)]);
            a_reg[k][mt][1] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk0, r1)]);
            a_reg[k][mt][2] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk8, r0)]);
            a_reg[k][mt][3] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk8, r1)]);
        }
    }

    uint32_t b_reg[8][4][2];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            int nc = n_base + nt * 8 + lane_row;
            b_reg[k][nt][0] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk0, nc)]);
            b_reg[k][nt][1] = *reinterpret_cast<const uint32_t*>(&sB[nc][swz(lk8, nc)]);
        }
    }

    float acc[2][4][4];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                      "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                    : "r"(a_reg[k][mt][0]), "r"(a_reg[k][mt][1]),
                      "r"(a_reg[k][mt][2]), "r"(a_reg[k][mt][3]),
                      "r"(b_reg[k][nt][0]), "r"(b_reg[k][nt][1])
                );
            }
        }
    }

    const int C_n_base = bcol * 128;
    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        int m0 = m_base + mt * 16 + lane_row, m1 = m0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            int n0 = C_n_base + n_base + nt * 8 + lane_k;
            st_cs32(&C[m0 * N + n0], acc[mt][nt][0], acc[mt][nt][1]);
            st_cs32(&C[m1 * N + n0], acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_persistent_db(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half* __restrict__ C,
    int N,
    int n_tiles
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int lane_row    = lane >> 2;
    const int lane_k      = (lane & 3) * 2;
    const int warp_n_base = warp_id * 16;

    __shared__ __half sA[64][128];
    __shared__ __half sB[2][64][128];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128;
        int row = idx >> 4, cg = idx & 15;
        uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sA[row][((cg ^ (row & 7)) << 3)]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(sm), "l"(&A[row * 128 + (cg << 3)]));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    uint32_t a_reg[8][4][4];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int r0 = mt * 16 + lane_row, r1 = r0 + 8;
            a_reg[k][mt][0] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk0, r0)]);
            a_reg[k][mt][1] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk0, r1)]);
            a_reg[k][mt][2] = *reinterpret_cast<const uint32_t*>(&sA[r0][swz(lk8, r0)]);
            a_reg[k][mt][3] = *reinterpret_cast<const uint32_t*>(&sA[r1][swz(lk8, r1)]);
        }
    }

    int cur_buf = 0;
    int start_tile = blockIdx.x;

    if (start_tile >= n_tiles) return;

    {
        int gn_base = start_tile * 64;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int nl  = idx >> 4, cg = idx & 15;
            int gn  = gn_base + nl;
            if (gn < N) {
                uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sB[0][nl][((cg ^ (nl & 7)) << 3)]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                    "r"(sm), "l"(&B_col[gn * 128 + (cg << 3)]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int bcol = start_tile; bcol < n_tiles; bcol += gridDim.x) {
        int next_bcol = bcol + gridDim.x;
        int next_buf  = 1 - cur_buf;

        if (next_bcol < n_tiles) {
            int gn_base = next_bcol * 64;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = tid + i * 128;
                int nl  = idx >> 4, cg = idx & 15;
                int gn  = gn_base + nl;
                if (gn < N) {
                    uint32_t sm = (uint32_t)__cvta_generic_to_shared(&sB[next_buf][nl][((cg ^ (nl & 7)) << 3)]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                        "r"(sm), "l"(&B_col[gn * 128 + (cg << 3)]));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        uint32_t b_reg[8][2][2];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int lk0 = k * 16 + lane_k, lk8 = lk0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int nc = warp_n_base + nt * 8 + lane_row;
                b_reg[k][nt][0] = *reinterpret_cast<const uint32_t*>(&sB[cur_buf][nc][swz(lk0, nc)]);
                b_reg[k][nt][1] = *reinterpret_cast<const uint32_t*>(&sB[cur_buf][nc][swz(lk8, nc)]);
            }
        }

        float acc[4][2][4];
        #pragma unroll
        for (int mt = 0; mt < 4; mt++)
            #pragma unroll
            for (int nt = 0; nt < 2; nt++)
                acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                #pragma unroll
                for (int mt = 0; mt < 4; mt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                          "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                        : "r"(a_reg[k][mt][0]), "r"(a_reg[k][mt][1]),
                          "r"(a_reg[k][mt][2]), "r"(a_reg[k][mt][3]),
                          "r"(b_reg[k][nt][0]), "r"(b_reg[k][nt][1])
                    );
                }
            }
        }

        const int C_n_base = bcol * 64;
        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int m0 = mt * 16 + lane_row, m1 = m0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int n0 = C_n_base + warp_n_base + nt * 8 + lane_k;
                st_cs32(&C[m0 * N + n0], acc[mt][nt][0], acc[mt][nt][1]);
                st_cs32(&C[m1 * N + n0], acc[mt][nt][2], acc[mt][nt][3]);
            }
        }

        cur_buf = next_buf;
        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" ::);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
    if (((T).options().dtype() != (th_type))) { \
        throw std::runtime_error("wrong dtype"); \
    }

static int g_best_v8 = -1;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    int N = (int)b.size(1);

    const __half* A_ptr    = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* Bcol_ptr = reinterpret_cast<const __half*>(b_col_major.data_ptr());
    __half* C_ptr          = reinterpret_cast<__half*>(c.data_ptr());

    const int n_tiles64  = (N + 63) / 64;
    const int n_tiles128 = (N + 127) / 128;
    const int persist_blocks_8 = min(n_tiles64,  132 * 8);
    const int persist_blocks_4 = min(n_tiles64,  132 * 4);

    if (g_best_v8 == -1) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        float times[6];
        for (int i = 0; i < 6; i++) times[i] = 1e30f;
        const int warmup = 5, iters = 50;

        auto run_k = [&](int kid) {
            switch (kid) {
            case 0: hgemm_persistent_v8<<<persist_blocks_8, 128>>>(A_ptr, Bcol_ptr, C_ptr, N, n_tiles64); break;
            case 1: hgemm_static_bn64<<<n_tiles64, 128>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
            case 2: hgemm_static_bn128<<<n_tiles128, 128>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
            case 3: hgemm_static_bn64_256t<<<n_tiles64, 256>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
            case 4: hgemm_static_bn128_256t<<<n_tiles128, 256>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
            case 5: hgemm_persistent_db<<<persist_blocks_4, 128>>>(A_ptr, Bcol_ptr, C_ptr, N, n_tiles64); break;
            }
        };

        for (int kid = 0; kid < 6; kid++) {
            for (int w = 0; w < warmup; w++) run_k(kid);
            cudaDeviceSynchronize();
            cudaEventRecord(ev0);
            for (int i = 0; i < iters; i++) run_k(kid);
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            cudaEventElapsedTime(&times[kid], ev0, ev1);
        }

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        g_best_v8 = 0;
        for (int k = 1; k < 6; k++)
            if (times[k] < times[g_best_v8]) g_best_v8 = k;
    }

    switch (g_best_v8) {
    case 0: hgemm_persistent_v8<<<persist_blocks_8, 128>>>(A_ptr, Bcol_ptr, C_ptr, N, n_tiles64); break;
    case 1: hgemm_static_bn64<<<n_tiles64, 128>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
    case 2: hgemm_static_bn128<<<n_tiles128, 128>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
    case 3: hgemm_static_bn64_256t<<<n_tiles64, 256>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
    case 4: hgemm_static_bn128_256t<<<n_tiles128, 256>>>(A_ptr, Bcol_ptr, C_ptr, N); break;
    case 5: hgemm_persistent_db<<<persist_blocks_4, 128>>>(A_ptr, Bcol_ptr, C_ptr, N, n_tiles64); break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}