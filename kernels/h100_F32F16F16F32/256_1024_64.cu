#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <stdint.h>

using namespace nvcuda::wmma;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 u64addr;\n"
                 "  cvta.to.shared.u64 u64addr, %1;\n"
                 "  cvt.u32.u64 %0, u64addr; }\n"
                 : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__global__ __launch_bounds__(32, 16)
void hgemm_mma_16x64_1warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.y * 16;
    const int block_n = blockIdx.x * 64;
    const int lane = threadIdx.x;

    __shared__ half smA[16][72];
    __shared__ half smB[64][72];

    #pragma unroll
    for (int f = 0; f < 4; f++) {
        int idx = lane * 4 + f;
        int row = idx / 8;
        int col = (idx % 8) * 8;
        int g_row = block_m + row;
        uint32_t sa = smem_u32addr(&smA[row][col]);
        if (g_row < M) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(sa), "l"(&A[g_row * K + col]));
        } else {
            *reinterpret_cast<float4*>(&smA[row][col]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    #pragma unroll
    for (int f = 0; f < 16; f++) {
        int idx = lane * 16 + f;
        int n_local = idx / 8;
        int k_start = (idx % 8) * 8;
        int g_n = block_n + n_local;
        uint32_t sb = smem_u32addr(&smB[n_local][k_start]);
        if (g_n < N) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(sb), "l"(&B[g_n * K + k_start]));
        } else {
            *reinterpret_cast<float4*>(&smB[n_local][k_start]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncwarp();

    float acc[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    const int kp = (lane & 3) * 2;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(&smA[r0][ki*16 + kp    ]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(&smA[r1][ki*16 + kp    ]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(&smA[r0][ki*16 + kp + 8]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(&smA[r1][ki*16 + kp + 8]);

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int n_b = ni * 8 + r0;
            uint32_t b0 = *reinterpret_cast<const uint32_t*>(&smB[n_b][ki*16 + kp    ]);
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(&smB[n_b][ki*16 + kp + 8]);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                : "=f"(acc[ni][0]),"=f"(acc[ni][1]),"=f"(acc[ni][2]),"=f"(acc[ni][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
                  "r"(b0),"r"(b1),
                  "f"(acc[ni][0]),"f"(acc[ni][1]),"f"(acc[ni][2]),"f"(acc[ni][3])
            );
        }
    }

    const int out_r0 = block_m + r0;
    const int out_r1 = block_m + r1;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        int nc = block_n + ni * 8 + kp;
        if (out_r0 < M && nc + 1 <= N)
            *reinterpret_cast<half2*>(&C[out_r0 * N + nc]) =
                __float22half2_rn(make_float2(acc[ni][0], acc[ni][1]));
        if (out_r1 < M && nc + 1 <= N)
            *reinterpret_cast<half2*>(&C[out_r1 * N + nc]) =
                __float22half2_rn(make_float2(acc[ni][2], acc[ni][3]));
    }
}

__global__ __launch_bounds__(128, 4)
void hgemm_mma_64x128_4warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int block_m = blockIdx.y * 64;
    const int block_n = blockIdx.x * 128;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tid = threadIdx.x;

    const int warp_m = (warp_id >> 1) * 32;
    const int warp_n = (warp_id & 1) * 64;

    __shared__ half smA[64][72];
    __shared__ half smB[128][72];

    #pragma unroll
    for (int f = 0; f < 4; f++) {
        int idx = tid * 4 + f;
        int row = idx / 8;
        int col = (idx % 8) * 8;
        int g_row = block_m + row;
        uint32_t sa = smem_u32addr(&smA[row][col]);
        if (g_row < M) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(sa), "l"(&A[g_row * K + col]));
        } else {
            *reinterpret_cast<float4*>(&smA[row][col]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    #pragma unroll
    for (int f = 0; f < 8; f++) {
        int idx = tid * 8 + f;
        int n_local = idx / 8;
        int k_start = (idx % 8) * 8;
        int g_n = block_n + n_local;
        uint32_t sb = smem_u32addr(&smB[n_local][k_start]);
        if (g_n < N) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(sb), "l"(&B[g_n * K + k_start]));
        } else {
            *reinterpret_cast<float4*>(&smB[n_local][k_start]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

    const int r0 = lane >> 2;
    const int kp = (lane & 3) * 2;

    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        uint32_t a[2][4];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int base = warp_m + mi * 16;
            int rr0 = base + r0;
            int rr1 = base + r0 + 8;
            a[mi][0] = *reinterpret_cast<const uint32_t*>(&smA[rr0][ki*16 + kp    ]);
            a[mi][1] = *reinterpret_cast<const uint32_t*>(&smA[rr1][ki*16 + kp    ]);
            a[mi][2] = *reinterpret_cast<const uint32_t*>(&smA[rr0][ki*16 + kp + 8]);
            a[mi][3] = *reinterpret_cast<const uint32_t*>(&smA[rr1][ki*16 + kp + 8]);
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int n_b = warp_n + ni * 8 + r0;
            uint32_t b0 = *reinterpret_cast<const uint32_t*>(&smB[n_b][ki*16 + kp    ]);
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(&smB[n_b][ki*16 + kp + 8]);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(acc[mi][ni][0]),"=f"(acc[mi][ni][1]),"=f"(acc[mi][ni][2]),"=f"(acc[mi][ni][3])
                    : "r"(a[mi][0]),"r"(a[mi][1]),"r"(a[mi][2]),"r"(a[mi][3]),
                      "r"(b0),"r"(b1),
                      "f"(acc[mi][ni][0]),"f"(acc[mi][ni][1]),"f"(acc[mi][ni][2]),"f"(acc[mi][ni][3])
                );
            }
        }
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        int base_r = block_m + warp_m + mi * 16;
        int out_r0 = base_r + r0;
        int out_r1 = base_r + r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int nc = block_n + warp_n + ni * 8 + kp;
            if (out_r0 < M && nc + 1 <= N)
                *reinterpret_cast<half2*>(&C[out_r0 * N + nc]) =
                    __float22half2_rn(make_float2(acc[mi][ni][0], acc[mi][ni][1]));
            if (out_r1 < M && nc + 1 <= N)
                *reinterpret_cast<half2*>(&C[out_r1 * N + nc]) =
                    __float22half2_rn(make_float2(acc[mi][ni][2], acc[mi][ni][3]));
        }
    }
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BM 64
#define BN 128
#define BK 64
#define WM 32
#define WN 64
#define WARP_TILES_M 2
#define WARP_TILES_N 4
#define K_TILES 4

__global__ __launch_bounds__(128, 4)
void hgemm_wmma_fallback(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;
    int warp_id = threadIdx.x / 32;
    int tid = threadIdx.x;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;
    int warp_m_start = warp_row * WM;
    int warp_n_start = warp_col * WN;

    __shared__ half smA_w[BM][BK + 8];
    __shared__ half smB_w[BN][BK + 8];

    #pragma unroll
    for (int f = 0; f < 4; f++) {
        int idx = tid * 4 + f;
        int row = idx / 8;
        int col = (idx % 8) * 8;
        int g_row = block_m + row;
        uint32_t sa = smem_u32addr(&smA_w[row][col]);
        if (g_row < M) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(sa), "l"(&A[g_row * K + col]));
        } else {
            *reinterpret_cast<float4*>(&smA_w[row][col]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    #pragma unroll
    for (int f = 0; f < 8; f++) {
        int idx = tid * 8 + f;
        int n_local = idx / 8;
        int k_start = (idx % 8) * 8;
        int g_n = block_n + n_local;
        uint32_t sb = smem_u32addr(&smB_w[n_local][k_start]);
        if (g_n < N) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(sb), "l"(&B[g_n * K + k_start]));
        } else {
            *reinterpret_cast<float4*>(&smB_w[n_local][k_start]) = make_float4(0.f,0.f,0.f,0.f);
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA[WARP_TILES_M][K_TILES];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB[K_TILES][WARP_TILES_N];

    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; mi++)
        #pragma unroll
        for (int ki = 0; ki < K_TILES; ki++)
            load_matrix_sync(fragA[mi][ki],
                &smA_w[warp_m_start + mi*WMMA_M][ki*WMMA_K], BK+8);

    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++)
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ni++)
            load_matrix_sync(fragB[ki][ni],
                &smB_w[warp_n_start + ni*WMMA_N][ki*WMMA_K], BK+8);

    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ni++)
            #pragma unroll
            for (int ki = 0; ki < K_TILES; ki++)
                mma_sync(acc[mi][ni], fragA[mi][ki], fragB[ki][ni], acc[mi][ni]);

    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ni++) {
            int c_row = block_m + warp_m_start + mi * WMMA_M;
            int c_col = block_n + warp_n_start + ni * WMMA_N;
            if (c_row < M && c_col < N) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_h;
                #pragma unroll
                for (int t = 0; t < acc[mi][ni].num_elements; t++)
                    acc_h.x[t] = __float2half(acc[mi][ni].x[t]);
                store_matrix_sync(&C[c_row * N + c_col], acc_h, N, mem_row_major);
            }
        }
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half* ptr_C = reinterpret_cast<half*>(c.data_ptr());

    {
        dim3 block(32);
        dim3 grid((N + 63) / 64, (M + 15) / 16);
        hgemm_mma_16x64_1warp<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaDeviceSynchronize();
        } else {
            return;
        }
    }

    {
        dim3 block(128);
        dim3 grid((N + 127) / 128, (M + 63) / 64);
        hgemm_mma_64x128_4warp<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaDeviceSynchronize();
        } else {
            return;
        }
    }

    {
        dim3 block(128);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        hgemm_wmma_fallback<<<grid, block>>>(ptr_A, ptr_B, ptr_C, M, N, K);
    }
}