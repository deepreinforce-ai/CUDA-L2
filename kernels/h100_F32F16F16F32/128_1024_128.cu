#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

static __device__ __forceinline__ uint32_t smem_ptr32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void cp_async16_cg(void* dst, const void* src) {
    uint32_t d = smem_ptr32(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(src) : "memory");
}

__global__ void __launch_bounds__(256, 2)
hgemm_ptx_128x128_db(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int A_STRIDE = BK + 8;
    constexpr int B_STRIDE = BN + 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int base_m = by * BM;
    const int base_n = bx * BN;

    __shared__ __align__(128) half smem_A[2][BM * A_STRIDE];
    __shared__ __align__(128) half smem_B[2][BK * B_STRIDE];

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A_smem = [&](int s, int k_off) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = i * 256 + tid;
            int row = idx >> 2;
            int col = (idx & 3) << 3;
            int g_row = base_m + row;
            int g_col = k_off + col;
            half* dst = &smem_A[s][row * A_STRIDE + col];
            if (g_row < M && g_col + 7 < K) {
                cp_async16_cg(dst, &A[g_row * K + g_col]);
            } else if (g_row < M) {
                #pragma unroll
                for (int jj = 0; jj < 8; jj++)
                    dst[jj] = (g_col + jj < K) ? A[g_row * K + g_col + jj] : __float2half(0.f);
            } else {
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) dst[jj] = __float2half(0.f);
            }
        }
    };

    auto load_B_smem = [&](int s, int k_off) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = i * 256 + tid;
            int row = idx >> 4;
            int col = (idx & 15) << 3;
            int g_row = k_off + row;
            int g_col = base_n + col;
            half* dst = &smem_B[s][row * B_STRIDE + col];
            if (g_row < K && g_col + 7 < N) {
                cp_async16_cg(dst, &B[g_row * N + g_col]);
            } else if (g_row < K) {
                #pragma unroll
                for (int jj = 0; jj < 8; jj++)
                    dst[jj] = (g_col + jj < N) ? B[g_row * N + g_col + jj] : __float2half(0.f);
            } else {
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) dst[jj] = __float2half(0.f);
            }
        }
    };

    auto compute_tile = [&](int s) __attribute__((always_inline)) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int smem_row = warp_m * 32 + mi * 16 + (lane_id & 15);
                int smem_col = k_inner + ((lane_id >> 4) << 3);
                uint32_t addr = smem_ptr32(&smem_A[s][smem_row * A_STRIDE + smem_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                      "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                    : "r"(addr)
                );
            }

            uint32_t b_frag[4][2];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_row = k_inner + (lane_id & 15);
                int b_col = warp_n * 64 + ni * 8;
                uint32_t addr = smem_ptr32(&smem_B[s][b_row * B_STRIDE + b_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b_frag[ni][0]), "r"(b_frag[ni][1])
                    );
                }
            }
        }
    };

    load_A_smem(0, 0);
    load_B_smem(0, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int cur = k_tile & 1;
        int nxt = 1 - cur;

        if (k_tile + 1 < num_k_tiles) {
            load_A_smem(nxt, (k_tile + 1) * BK);
            load_B_smem(nxt, (k_tile + 1) * BK);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();

        compute_tile(cur);
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int lane_row = lane_id >> 2;
    const int lane_col = (lane_id & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        int tile_row = base_m + warp_m * 32 + mi * 16;
        int row0 = tile_row + lane_row;
        int row1 = tile_row + lane_row + 8;

        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int tile_col = base_n + warp_n * 64 + ni * 8;
            int col0 = tile_col + lane_col;

            if (row0 < M && col0 + 1 <= N) {
                *reinterpret_cast<half2*>(&C[row0 * N + col0]) =
                    __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            }
            if (row1 < M && col0 + 1 <= N) {
                *reinterpret_cast<half2*>(&C[row1 * N + col0]) =
                    __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 3)
hgemm_ptx_64x128_db(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int A_STRIDE = BK + 8;
    constexpr int B_STRIDE = BN + 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_m = warp_id >> 2;
    const int warp_n = warp_id & 3;

    const int base_m = by * BM;
    const int base_n = bx * BN;

    __shared__ __align__(128) half smem_A[2][BM * A_STRIDE];
    __shared__ __align__(128) half smem_B[2][BK * B_STRIDE];

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_A_smem = [&](int s, int k_off) __attribute__((always_inline)) {
        int row = tid >> 2;
        int col = (tid & 3) << 3;
        int g_row = base_m + row;
        int g_col = k_off + col;
        half* dst = &smem_A[s][row * A_STRIDE + col];
        if (g_row < M && g_col + 7 < K) {
            cp_async16_cg(dst, &A[g_row * K + g_col]);
        } else if (g_row < M) {
            #pragma unroll
            for (int jj = 0; jj < 8; jj++)
                dst[jj] = (g_col + jj < K) ? A[g_row * K + g_col + jj] : __float2half(0.f);
        } else {
            #pragma unroll
            for (int jj = 0; jj < 8; jj++) dst[jj] = __float2half(0.f);
        }
    };

    auto load_B_smem = [&](int s, int k_off) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = i * 256 + tid;
            int row = idx >> 4;
            int col = (idx & 15) << 3;
            int g_row = k_off + row;
            int g_col = base_n + col;
            half* dst = &smem_B[s][row * B_STRIDE + col];
            if (g_row < K && g_col + 7 < N) {
                cp_async16_cg(dst, &B[g_row * N + g_col]);
            } else if (g_row < K) {
                #pragma unroll
                for (int jj = 0; jj < 8; jj++)
                    dst[jj] = (g_col + jj < N) ? B[g_row * N + g_col + jj] : __float2half(0.f);
            } else {
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) dst[jj] = __float2half(0.f);
            }
        }
    };

    auto compute_tile = [&](int s) __attribute__((always_inline)) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int k_inner = kk * 16;

            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int smem_row = warp_m * 32 + mi * 16 + (lane_id & 15);
                int smem_col = k_inner + ((lane_id >> 4) << 3);
                uint32_t addr = smem_ptr32(&smem_A[s][smem_row * A_STRIDE + smem_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                      "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                    : "r"(addr)
                );
            }

            uint32_t b_frag[4][2];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int b_row = k_inner + (lane_id & 15);
                int b_col = warp_n * 32 + ni * 8;
                uint32_t addr = smem_ptr32(&smem_B[s][b_row * B_STRIDE + b_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[ni][0]), "=r"(b_frag[ni][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b_frag[ni][0]), "r"(b_frag[ni][1])
                    );
                }
            }
        }
    };

    load_A_smem(0, 0);
    load_B_smem(0, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int cur = k_tile & 1;
        int nxt = 1 - cur;

        if (k_tile + 1 < num_k_tiles) {
            load_A_smem(nxt, (k_tile + 1) * BK);
            load_B_smem(nxt, (k_tile + 1) * BK);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();

        compute_tile(cur);
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int lane_row = lane_id >> 2;
    const int lane_col = (lane_id & 3) << 1;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        int tile_row = base_m + warp_m * 32 + mi * 16;
        int row0 = tile_row + lane_row;
        int row1 = tile_row + lane_row + 8;

        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int tile_col = base_n + warp_n * 32 + ni * 8;
            int col0 = tile_col + lane_col;

            if (row0 < M && col0 + 1 <= N) {
                *reinterpret_cast<half2*>(&C[row0 * N + col0]) =
                    __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            }
            if (row1 < M && col0 + 1 <= N) {
                *reinterpret_cast<half2*>(&C[row1 * N + col0]) =
                    __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_a = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_b = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_c = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    {
        constexpr int BM = 64, BN = 128;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(256);
        hgemm_ptx_64x128_db<<<grid, block>>>(ptr_a, ptr_b, ptr_c, M, N, K);
    }
}