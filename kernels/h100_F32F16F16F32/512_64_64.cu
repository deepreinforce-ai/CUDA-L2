#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

__device__ __forceinline__ void mma_m16n8k16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__global__ __launch_bounds__(32, 8)
void hgemm_pure_reg_bm8_final(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int lane      = threadIdx.x;
    const int lane_mod4 = lane & 3;
    const int lane_div4 = lane >> 2;
    const int rbase     = blockIdx.x * 8;

    if (rbase >= M) return;

    const int row0 = rbase + lane_div4;
    const bool r0v = (row0 < M);

    uint32_t b_reg[4][8][2];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int k_lo = kk * 16 + lane_mod4 * 2;
        const int k_hi = k_lo + 8;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int n_col = (nt << 3) + lane_div4;
            const int base  = n_col * K;
            b_reg[kk][nt][0] = __ldg(reinterpret_cast<const uint32_t*>(&B_col[base + k_lo]));
            b_reg[kk][nt][1] = __ldg(reinterpret_cast<const uint32_t*>(&B_col[base + k_hi]));
        }
    }

    uint32_t a_reg[4][2];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int k_lo = kk * 16 + lane_mod4 * 2;
        const int k_hi = k_lo + 8;
        a_reg[kk][0] = r0v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row0 * K + k_lo])) : 0u;
        a_reg[kk][1] = r0v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row0 * K + k_hi])) : 0u;
    }

    float acc[8][4] = {};

    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            mma_m16n8k16_f32(
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                a_reg[kk][0], 0u, a_reg[kk][1], 0u,
                b_reg[kk][nt][0], b_reg[kk][nt][1],
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]
            );
        }
    }

    if (r0v) {
        const int c_col_off = lane_mod4 * 2;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int c_col = (nt << 3) + c_col_off;
            *reinterpret_cast<half2*>(&C[row0 * N + c_col]) =
                __floats2half2_rn(acc[nt][0], acc[nt][1]);
        }
    }
}

__global__ __launch_bounds__(32, 8)
void hgemm_pure_reg_bm16_final(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int lane      = threadIdx.x;
    const int lane_mod4 = lane & 3;
    const int lane_div4 = lane >> 2;
    const int rbase     = blockIdx.x * 16;

    if (rbase >= M) return;

    const int row0 = rbase + lane_div4;
    const int row1 = row0 + 8;
    const bool r0v = (row0 < M);
    const bool r1v = (row1 < M);

    uint32_t b_reg[4][8][2];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int k_lo = kk * 16 + lane_mod4 * 2;
        const int k_hi = k_lo + 8;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int n_col = (nt << 3) + lane_div4;
            const int base  = n_col * K;
            b_reg[kk][nt][0] = __ldg(reinterpret_cast<const uint32_t*>(&B_col[base + k_lo]));
            b_reg[kk][nt][1] = __ldg(reinterpret_cast<const uint32_t*>(&B_col[base + k_hi]));
        }
    }

    uint32_t a_reg[4][4];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int k_lo = kk * 16 + lane_mod4 * 2;
        const int k_hi = k_lo + 8;
        a_reg[kk][0] = r0v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row0 * K + k_lo])) : 0u;
        a_reg[kk][1] = r1v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row1 * K + k_lo])) : 0u;
        a_reg[kk][2] = r0v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row0 * K + k_hi])) : 0u;
        a_reg[kk][3] = r1v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row1 * K + k_hi])) : 0u;
    }

    float acc[8][4] = {};
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            mma_m16n8k16_f32(
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                a_reg[kk][0], a_reg[kk][1], a_reg[kk][2], a_reg[kk][3],
                b_reg[kk][nt][0], b_reg[kk][nt][1],
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]
            );
        }
    }

    const int c_col_off = lane_mod4 * 2;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int c_col = (nt << 3) + c_col_off;
        if (r0v)
            *reinterpret_cast<half2*>(&C[row0 * N + c_col]) =
                __floats2half2_rn(acc[nt][0], acc[nt][1]);
        if (r1v)
            *reinterpret_cast<half2*>(&C[row1 * N + c_col]) =
                __floats2half2_rn(acc[nt][2], acc[nt][3]);
    }
}

__global__ __launch_bounds__(64, 4)
void hgemm_2warp_bm8_final(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C,
    int M, int N, int K,
    int num_tiles
) {
    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;
    const int lane_mod4 = lane & 3;
    const int lane_div4 = lane >> 2;

    const int tile = blockIdx.x * 2 + warp_id;
    if (tile >= num_tiles) return;

    const int rbase = tile * 8;
    const int row0  = rbase + lane_div4;
    const bool r0v  = (row0 < M);

    uint32_t b_reg[4][8][2];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int k_lo = kk * 16 + lane_mod4 * 2;
        const int k_hi = k_lo + 8;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int n_col = (nt << 3) + lane_div4;
            const int base  = n_col * K;
            b_reg[kk][nt][0] = __ldg(reinterpret_cast<const uint32_t*>(&B_col[base + k_lo]));
            b_reg[kk][nt][1] = __ldg(reinterpret_cast<const uint32_t*>(&B_col[base + k_hi]));
        }
    }

    uint32_t a_reg[4][2];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int k_lo = kk * 16 + lane_mod4 * 2;
        const int k_hi = k_lo + 8;
        a_reg[kk][0] = r0v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row0 * K + k_lo])) : 0u;
        a_reg[kk][1] = r0v ? __ldg(reinterpret_cast<const uint32_t*>(&A[row0 * K + k_hi])) : 0u;
    }

    float acc[8][4] = {};
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            mma_m16n8k16_f32(
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                a_reg[kk][0], 0u, a_reg[kk][1], 0u,
                b_reg[kk][nt][0], b_reg[kk][nt][1],
                acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]
            );
        }
    }

    if (r0v) {
        const int c_col_off = lane_mod4 * 2;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int c_col = (nt << 3) + c_col_off;
            *reinterpret_cast<half2*>(&C[row0 * N + c_col]) =
                __floats2half2_rn(acc[nt][0], acc[nt][1]);
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    TORCH_CHECK(a.dtype() == torch::kHalf, "a must be half");
    TORCH_CHECK(b.dtype() == torch::kHalf, "b must be half");
    TORCH_CHECK(c.dtype() == torch::kHalf, "c must be half");

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const half* A_ptr     = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_col_ptr = reinterpret_cast<const half*>(b_col_major.data_ptr<at::Half>());
    half* C_ptr           = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const int num_tiles8 = (M + 7) / 8;

    hgemm_pure_reg_bm8_final<<<num_tiles8, 32>>>(A_ptr, B_col_ptr, C_ptr, M, N, K);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "hgemm_pure_reg_bm8_final failed");
}