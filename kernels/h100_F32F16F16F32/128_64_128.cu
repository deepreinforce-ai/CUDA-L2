#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if ((T).options().dtype() != (th_type)) { \
    throw std::runtime_error("Tensor dtype mismatch"); \
  }

#define M_DIM    128
#define N_DIM    64
#define K_DIM    128
#define BK       16
#define NK_TILES 8
#define BLOCK_M  16

#define SA_STRIDE 16
#define SB_STRIDE 64

static __device__ __forceinline__ uint32_t smem_u32(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__global__ __launch_bounds__(64, 16)
void hgemm_ldmatrix_v3_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int block_row = blockIdx.x * BLOCK_M;
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int warp_n    = warp_id;

    __shared__ __align__(128) half sA[2][BK][SA_STRIDE];
    __shared__ __align__(128) half sB[2][BK][SB_STRIDE];

    float acc[4][4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    auto load_A = [&](int buf, int kt) __attribute__((always_inline)) {
        int k_base   = kt * BK;
        int m_local  = tid & 15;
        int k_off    = (tid >> 4) * 4;
        int m_global = block_row + m_local;
        const half* src = A + m_global * K_DIM + k_base + k_off;
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            int k_idx    = k_off + ki;
            int m_swizzle = m_local ^ ((k_idx >> 1) & 7);
            sA[buf][k_idx][m_swizzle] = src[ki];
        }
    };

    auto load_B_async = [&](int buf, int kt) __attribute__((always_inline)) {
        int k_base = kt * BK;
        int k_row  = tid >> 2;
        int n_base = (tid & 3) * 16;

        uint32_t sm0 = smem_u32(&sB[buf][k_row][n_base]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(sm0), "l"((const void*)(B + (k_base + k_row) * N_DIM + n_base)));
        uint32_t sm1 = smem_u32(&sB[buf][k_row][n_base + 8]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(sm1), "l"((const void*)(B + (k_base + k_row) * N_DIM + n_base + 8)));
    };

    load_A(0, 0);
    load_B_async(0, 0);
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    const int lane_r = lane_id >> 2;
    const int lane_k = (lane_id & 3) << 1;

    const int out_r0 = block_row + lane_r;
    const int out_r1 = out_r0 + 8;

    #pragma unroll 4
    for (int kt = 0; kt < NK_TILES; kt++) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        if (kt + 1 < NK_TILES) {
            load_A(nxt, kt + 1);
            load_B_async(nxt, kt + 1);
            asm volatile("cp.async.commit_group;\n"::);
        }

        uint32_t a_frag[4];
        {
            int k0 = lane_k;
            int k1 = lane_k + 1;
            int k8 = lane_k + 8;
            int k9 = lane_k + 9;
            int r0 = lane_r;
            int r1 = lane_r + 8;

            half a00 = sA[cur][k0][r0 ^ ((k0>>1)&7)];
            half a01 = sA[cur][k1][r0 ^ ((k1>>1)&7)];
            half a10 = sA[cur][k0][r1 ^ ((k0>>1)&7)];
            half a11 = sA[cur][k1][r1 ^ ((k1>>1)&7)];
            half a20 = sA[cur][k8][r0 ^ ((k8>>1)&7)];
            half a21 = sA[cur][k9][r0 ^ ((k9>>1)&7)];
            half a30 = sA[cur][k8][r1 ^ ((k8>>1)&7)];
            half a31 = sA[cur][k9][r1 ^ ((k9>>1)&7)];

            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[0]) : "h"(__half_as_ushort(a00)), "h"(__half_as_ushort(a01)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[1]) : "h"(__half_as_ushort(a10)), "h"(__half_as_ushort(a11)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[2]) : "h"(__half_as_ushort(a20)), "h"(__half_as_ushort(a21)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[3]) : "h"(__half_as_ushort(a30)), "h"(__half_as_ushort(a31)));
        }

        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int n_base_warp = warp_n * 32 + ni * 8;

            uint32_t b_frag[2];
            {
                int n_col = (lane_id >> 2) + n_base_warp;
                int k0    = lane_k;
                int k1    = lane_k + 1;
                int k8    = lane_k + 8;
                int k9    = lane_k + 9;

                half b0 = sB[cur][k0][n_col];
                half b1 = sB[cur][k1][n_col];
                half b2 = sB[cur][k8][n_col];
                half b3 = sB[cur][k9][n_col];

                asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b_frag[0]) : "h"(__half_as_ushort(b0)), "h"(__half_as_ushort(b1)));
                asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b_frag[1]) : "h"(__half_as_ushort(b2)), "h"(__half_as_ushort(b3)));
            }

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1])
            );
        }

        if (kt + 1 < NK_TILES) {
            asm volatile("cp.async.wait_group 0;\n"::);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int n_base = warp_n * 32 + ni * 8;
        int col0   = n_base + lane_k;

        uint32_t v0 = (uint32_t)(__half_as_ushort(__float2half(acc[ni][0]))) |
                      ((uint32_t)(__half_as_ushort(__float2half(acc[ni][1]))) << 16);
        uint32_t v1 = (uint32_t)(__half_as_ushort(__float2half(acc[ni][2]))) |
                      ((uint32_t)(__half_as_ushort(__float2half(acc[ni][3]))) << 16);

        *reinterpret_cast<uint32_t*>(&C[out_r0 * N_DIM + col0]) = v0;
        *reinterpret_cast<uint32_t*>(&C[out_r1 * N_DIM + col0]) = v1;
    }
}

__global__ __launch_bounds__(64, 16)
void hgemm_8block_unrolled_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int block_row = blockIdx.x * BLOCK_M;
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int warp_n    = warp_id;

    __shared__ half sA[2][BK][20];
    __shared__ half sB[2][BK][72];

    float acc[4][4];
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    const int lane_r = lane_id >> 2;
    const int lane_c = (lane_id & 3) << 1;
    const int out_r0 = block_row + lane_r;
    const int out_r1 = out_r0 + 8;

    auto load_A_smem = [&](int buf, int kt) __attribute__((always_inline)) {
        int k_base   = kt * BK;
        int m_local  = tid & 15;
        int k_off    = (tid >> 4) * 4;
        int m_global = block_row + m_local;
        const half* src = A + m_global * K_DIM + k_base + k_off;
        #pragma unroll
        for (int ki = 0; ki < 4; ki++)
            sA[buf][k_off + ki][m_local] = src[ki];
    };

    auto load_B_smem_async = [&](int buf, int kt) __attribute__((always_inline)) {
        int k_base = kt * BK;
        int k_row  = tid >> 2;
        int n_off  = (tid & 3) * 16;
        uint32_t sm0 = smem_u32(&sB[buf][k_row][n_off]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(sm0), "l"((const void*)(B + (k_base + k_row) * N_DIM + n_off)));
        uint32_t sm1 = smem_u32(&sB[buf][k_row][n_off + 8]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(sm1), "l"((const void*)(B + (k_base + k_row) * N_DIM + n_off + 8)));
    };

    load_A_smem(0, 0);
    load_B_smem_async(0, 0);
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    #pragma unroll
    for (int kt = 0; kt < NK_TILES; kt++) {
        int cur = kt & 1;
        int nxt = 1 - cur;

        if (kt + 1 < NK_TILES) {
            load_A_smem(nxt, kt + 1);
            load_B_smem_async(nxt, kt + 1);
            asm volatile("cp.async.commit_group;\n"::);
        }

        uint32_t a_frag[4];
        {
            half a00 = sA[cur][lane_c    ][lane_r    ];
            half a01 = sA[cur][lane_c + 1][lane_r    ];
            half a10 = sA[cur][lane_c    ][lane_r + 8];
            half a11 = sA[cur][lane_c + 1][lane_r + 8];
            half a20 = sA[cur][lane_c + 8][lane_r    ];
            half a21 = sA[cur][lane_c + 9][lane_r    ];
            half a30 = sA[cur][lane_c + 8][lane_r + 8];
            half a31 = sA[cur][lane_c + 9][lane_r + 8];
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[0]) : "h"(__half_as_ushort(a00)), "h"(__half_as_ushort(a01)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[1]) : "h"(__half_as_ushort(a10)), "h"(__half_as_ushort(a11)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[2]) : "h"(__half_as_ushort(a20)), "h"(__half_as_ushort(a21)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(a_frag[3]) : "h"(__half_as_ushort(a30)), "h"(__half_as_ushort(a31)));
        }

        uint32_t b0[2], b1[2], b2[2], b3[2];

        {
            int nc0 = (lane_id >> 2) + warp_n * 32;
            half x0 = sB[cur][lane_c    ][nc0]; half x1 = sB[cur][lane_c + 1][nc0];
            half x2 = sB[cur][lane_c + 8][nc0]; half x3 = sB[cur][lane_c + 9][nc0];
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b0[0]) : "h"(__half_as_ushort(x0)), "h"(__half_as_ushort(x1)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b0[1]) : "h"(__half_as_ushort(x2)), "h"(__half_as_ushort(x3)));
        }
        {
            int nc1 = (lane_id >> 2) + warp_n * 32 + 8;
            half x0 = sB[cur][lane_c    ][nc1]; half x1 = sB[cur][lane_c + 1][nc1];
            half x2 = sB[cur][lane_c + 8][nc1]; half x3 = sB[cur][lane_c + 9][nc1];
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b1[0]) : "h"(__half_as_ushort(x0)), "h"(__half_as_ushort(x1)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b1[1]) : "h"(__half_as_ushort(x2)), "h"(__half_as_ushort(x3)));
        }
        {
            int nc2 = (lane_id >> 2) + warp_n * 32 + 16;
            half x0 = sB[cur][lane_c    ][nc2]; half x1 = sB[cur][lane_c + 1][nc2];
            half x2 = sB[cur][lane_c + 8][nc2]; half x3 = sB[cur][lane_c + 9][nc2];
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b2[0]) : "h"(__half_as_ushort(x0)), "h"(__half_as_ushort(x1)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b2[1]) : "h"(__half_as_ushort(x2)), "h"(__half_as_ushort(x3)));
        }
        {
            int nc3 = (lane_id >> 2) + warp_n * 32 + 24;
            half x0 = sB[cur][lane_c    ][nc3]; half x1 = sB[cur][lane_c + 1][nc3];
            half x2 = sB[cur][lane_c + 8][nc3]; half x3 = sB[cur][lane_c + 9][nc3];
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b3[0]) : "h"(__half_as_ushort(x0)), "h"(__half_as_ushort(x1)));
            asm volatile("mov.b32 %0, {%1,%2};\n" : "=r"(b3[1]) : "h"(__half_as_ushort(x2)), "h"(__half_as_ushort(x3)));
        }

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b0[0]), "r"(b0[1])
        );
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b1[0]), "r"(b1[1])
        );
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b2[0]), "r"(b2[1])
        );
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b3[0]), "r"(b3[1])
        );

        if (kt + 1 < NK_TILES) {
            asm volatile("cp.async.wait_group 0;\n"::);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int col0 = warp_n * 32 + ni * 8 + lane_c;
        uint32_t v0 = (uint32_t)(__half_as_ushort(__float2half(acc[ni][0]))) |
                      ((uint32_t)(__half_as_ushort(__float2half(acc[ni][1]))) << 16);
        uint32_t v1 = (uint32_t)(__half_as_ushort(__float2half(acc[ni][2]))) |
                      ((uint32_t)(__half_as_ushort(__float2half(acc[ni][3]))) << 16);
        *reinterpret_cast<uint32_t*>(&C[out_r0 * N_DIM + col0]) = v0;
        *reinterpret_cast<uint32_t*>(&C[out_r1 * N_DIM + col0]) = v1;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C = reinterpret_cast<half*>(c.data_ptr());

    hgemm_8block_unrolled_kernel<<<8, 64>>>(A, B, C);
}