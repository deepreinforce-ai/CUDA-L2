#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_pipeline_primitives.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__global__ void __launch_bounds__(128, 8)
hgemm_bn32_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 128, BN = 32, BK = 64;
    const int BN_PAD = 40;

    __shared__ __align__(128) __half smem_A[BM * BK];
    __shared__ __align__(128) __half smem_B[BK * BN_PAD];

    const int n_start = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx     = tid + i * 128;
            int row     = idx >> 3;
            int col_f4  = idx & 7;
            int phys_f4 = col_f4 ^ (row & 7);
            __half* dst       = smem_A + row * BK + phys_f4 * 8;
            const __half* src = A + row * BK + col_f4 * 8;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx      = tid + i * 128;
            int k_row    = idx >> 2;
            int n_chunk  = idx & 3;
            int n_local  = n_chunk * 8;
            int global_n = n_start + n_local;
            __half* dst       = smem_B + k_row * BN_PAD + n_local;
            const __half* src = B + (int64_t)k_row * N + global_n;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    const int warp_n_off = warp_id * 8;

    float acc[8][4];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;

    uint32_t a_frag[2][8][4];
    uint32_t b_frag[2][2];

    #pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        int row      = mt * 16 + (lane & 15);
        int log_col  = (lane >> 4) << 3;
        int phys_col = log_col ^ ((row & 7) << 3);
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_frag[0][mt][0]), "=r"(a_frag[0][mt][1]),
              "=r"(a_frag[0][mt][2]), "=r"(a_frag[0][mt][3])
            : "r"(addr)
        );
    }
    {
        int b_k_row   = lane & 15;
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + warp_n_off);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b_frag[0][0]), "=r"(b_frag[0][1])
            : "r"(addr)
        );
    }

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        const int cur = kt & 1;
        const int nxt = 1 - cur;

        if (kt < 3) {
            const int nk = (kt + 1) * 16;

            {
                int b_k_row   = nk + (lane & 15);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + warp_n_off);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[nxt][0]), "=r"(b_frag[nxt][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                int row      = mt * 16 + (lane & 15);
                int log_col  = nk + ((lane >> 4) << 3);
                int phys_col = log_col ^ ((row & 7) << 3);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[nxt][mt][0]), "=r"(a_frag[nxt][mt][1]),
                      "=r"(a_frag[nxt][mt][2]), "=r"(a_frag[nxt][mt][3])
                    : "r"(addr)
                );
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(acc[mt][0]), "=f"(acc[mt][1]),
                      "=f"(acc[mt][2]), "=f"(acc[mt][3])
                    : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                      "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                      "r"(b_frag[cur][0]), "r"(b_frag[cur][1]),
                      "f"(acc[mt][0]), "f"(acc[mt][1]),
                      "f"(acc[mt][2]), "f"(acc[mt][3])
                );
            }
        } else {
            #pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(acc[mt][0]), "=f"(acc[mt][1]),
                      "=f"(acc[mt][2]), "=f"(acc[mt][3])
                    : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                      "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                      "r"(b_frag[cur][0]), "r"(b_frag[cur][1]),
                      "f"(acc[mt][0]), "f"(acc[mt][1]),
                      "f"(acc[mt][2]), "f"(acc[mt][3])
                );
            }
        }
    }

    const int out_col = n_start + warp_n_off + (lane & 3) * 2;
    #pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        int row0 = mt * 16 + (lane >> 2);
        int row1 = row0 + 8;
        __half2 h01 = __floats2half2_rn(acc[mt][0], acc[mt][1]);
        __half2 h23 = __floats2half2_rn(acc[mt][2], acc[mt][3]);
        *reinterpret_cast<__half2*>(C + row0 * N + out_col) = h01;
        *reinterpret_cast<__half2*>(C + row1 * N + out_col) = h23;
    }
}

__global__ void __launch_bounds__(128, 7)
hgemm_bn48_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 128, BN = 48, BK = 64;
    const int BN_PAD = 56;

    __shared__ __align__(128) __half smem_A[BM * BK];
    __shared__ __align__(128) __half smem_B[BK * BN_PAD];

    const int n_start = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx     = tid + i * 128;
            int row     = idx >> 3;
            int col_f4  = idx & 7;
            int phys_f4 = col_f4 ^ (row & 7);
            __half* dst       = smem_A + row * BK + phys_f4 * 8;
            const __half* src = A + row * BK + col_f4 * 8;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            int idx      = tid + i * 128;
            int k_row    = idx / 6;
            int n_chunk  = idx % 6;
            int n_local  = n_chunk * 8;
            int global_n = n_start + n_local;
            __half* dst       = smem_B + k_row * BN_PAD + n_local;
            const __half* src = B + (int64_t)k_row * N + global_n;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    const int warp_row    = warp_id >> 1;
    const int warp_col    = warp_id & 1;
    const int warp_m_base = warp_row * 64;
    const int warp_n_base = warp_col * 24;

    float acc[4][3][4];
    #pragma unroll
    for (int m = 0; m < 4; m++)
        #pragma unroll
        for (int n = 0; n < 3; n++)
            acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

    uint32_t a_frag[2][4][4];
    uint32_t b_frag[2][3][2];

    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        int row      = warp_m_base + mt * 16 + (lane & 15);
        int log_col  = (lane >> 4) << 3;
        int phys_col = log_col ^ ((row & 7) << 3);
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_frag[0][mt][0]), "=r"(a_frag[0][mt][1]),
              "=r"(a_frag[0][mt][2]), "=r"(a_frag[0][mt][3])
            : "r"(addr)
        );
    }
    #pragma unroll
    for (int nt = 0; nt < 3; nt++) {
        int b_n_base = warp_n_base + nt * 8;
        int b_k_row  = lane & 15;
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + b_n_base);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b_frag[0][nt][0]), "=r"(b_frag[0][nt][1])
            : "r"(addr)
        );
    }

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        const int cur = kt & 1;
        const int nxt = 1 - cur;

        if (kt < 3) {
            const int nk = (kt + 1) * 16;

            #pragma unroll
            for (int nt = 0; nt < 3; nt++) {
                int b_n_base  = warp_n_base + nt * 8;
                int b_k_row   = nk + (lane & 15);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + b_n_base);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[nxt][nt][0]), "=r"(b_frag[nxt][nt][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int row      = warp_m_base + mt * 16 + (lane & 15);
                int log_col  = nk + ((lane >> 4) << 3);
                int phys_col = log_col ^ ((row & 7) << 3);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[nxt][mt][0]), "=r"(a_frag[nxt][mt][1]),
                      "=r"(a_frag[nxt][mt][2]), "=r"(a_frag[nxt][mt][3])
                    : "r"(addr)
                );
                #pragma unroll
                for (int nt = 0; nt < 3; nt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(acc[mt][nt][0]), "=f"(acc[mt][nt][1]),
                          "=f"(acc[mt][nt][2]), "=f"(acc[mt][nt][3])
                        : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                          "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                          "r"(b_frag[cur][nt][0]), "r"(b_frag[cur][nt][1]),
                          "f"(acc[mt][nt][0]), "f"(acc[mt][nt][1]),
                          "f"(acc[mt][nt][2]), "f"(acc[mt][nt][3])
                    );
                }
            }
        } else {
            #pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                #pragma unroll
                for (int nt = 0; nt < 3; nt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(acc[mt][nt][0]), "=f"(acc[mt][nt][1]),
                          "=f"(acc[mt][nt][2]), "=f"(acc[mt][nt][3])
                        : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                          "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                          "r"(b_frag[cur][nt][0]), "r"(b_frag[cur][nt][1]),
                          "f"(acc[mt][nt][0]), "f"(acc[mt][nt][1]),
                          "f"(acc[mt][nt][2]), "f"(acc[mt][nt][3])
                    );
                }
            }
        }
    }

    const int out_n_base = n_start + warp_n_base;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        int row0 = warp_m_base + mt * 16 + (lane >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 3; nt++) {
            int col = out_n_base + nt * 8 + (lane & 3) * 2;
            __half2 h01 = __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
            __half2 h23 = __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
            *reinterpret_cast<__half2*>(C + row0 * N + col) = h01;
            *reinterpret_cast<__half2*>(C + row1 * N + col) = h23;
        }
    }
}

__global__ void __launch_bounds__(128, 6)
hgemm_bn64_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 128, BN = 64, BK = 64;
    const int BN_PAD = 72;

    __shared__ __align__(128) __half smem_A[BM * BK];
    __shared__ __align__(128) __half smem_B[BK * BN_PAD];

    const int n_start = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx     = tid + i * 128;
            int row     = idx >> 3;
            int col_f4  = idx & 7;
            int phys_f4 = col_f4 ^ (row & 7);
            __half* dst       = smem_A + row * BK + phys_f4 * 8;
            const __half* src = A + row * BK + col_f4 * 8;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx      = tid + i * 128;
            int k_row    = idx >> 3;
            int n_chunk  = idx & 7;
            int n_local  = n_chunk * 8;
            int global_n = n_start + n_local;
            __half* dst       = smem_B + k_row * BN_PAD + n_local;
            const __half* src = B + (int64_t)k_row * N + global_n;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    const int warp_n_off = warp_id * 16;

    float acc[8][2][4];
    #pragma unroll
    for (int m = 0; m < 8; m++)
        #pragma unroll
        for (int n = 0; n < 2; n++)
            acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

    uint32_t a_frag[2][8][4];
    uint32_t b_frag[2][2][2];

    #pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        int row      = mt * 16 + (lane & 15);
        int log_col  = (lane >> 4) << 3;
        int phys_col = log_col ^ ((row & 7) << 3);
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_frag[0][mt][0]), "=r"(a_frag[0][mt][1]),
              "=r"(a_frag[0][mt][2]), "=r"(a_frag[0][mt][3])
            : "r"(addr)
        );
    }
    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        int b_n_base = warp_n_off + nt * 8;
        int b_k_row  = lane & 15;
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + b_n_base);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b_frag[0][nt][0]), "=r"(b_frag[0][nt][1])
            : "r"(addr)
        );
    }

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        const int cur = kt & 1;
        const int nxt = 1 - cur;

        if (kt < 3) {
            const int nk = (kt + 1) * 16;

            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int b_n_base  = warp_n_off + nt * 8;
                int b_k_row   = nk + (lane & 15);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + b_n_base);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[nxt][nt][0]), "=r"(b_frag[nxt][nt][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                int row      = mt * 16 + (lane & 15);
                int log_col  = nk + ((lane >> 4) << 3);
                int phys_col = log_col ^ ((row & 7) << 3);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[nxt][mt][0]), "=r"(a_frag[nxt][mt][1]),
                      "=r"(a_frag[nxt][mt][2]), "=r"(a_frag[nxt][mt][3])
                    : "r"(addr)
                );
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(acc[mt][nt][0]), "=f"(acc[mt][nt][1]),
                          "=f"(acc[mt][nt][2]), "=f"(acc[mt][nt][3])
                        : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                          "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                          "r"(b_frag[cur][nt][0]), "r"(b_frag[cur][nt][1]),
                          "f"(acc[mt][nt][0]), "f"(acc[mt][nt][1]),
                          "f"(acc[mt][nt][2]), "f"(acc[mt][nt][3])
                    );
                }
            }
        } else {
            #pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(acc[mt][nt][0]), "=f"(acc[mt][nt][1]),
                          "=f"(acc[mt][nt][2]), "=f"(acc[mt][nt][3])
                        : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                          "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                          "r"(b_frag[cur][nt][0]), "r"(b_frag[cur][nt][1]),
                          "f"(acc[mt][nt][0]), "f"(acc[mt][nt][1]),
                          "f"(acc[mt][nt][2]), "f"(acc[mt][nt][3])
                    );
                }
            }
        }
    }

    const int out_n_base = n_start + warp_n_off;
    #pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        int row0 = mt * 16 + (lane >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            int col = out_n_base + nt * 8 + (lane & 3) * 2;
            __half2 h01 = __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
            __half2 h23 = __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
            *reinterpret_cast<__half2*>(C + row0 * N + col) = h01;
            *reinterpret_cast<__half2*>(C + row1 * N + col) = h23;
        }
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_bn128_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K
) {
    const int BM = 128, BN = 128, BK = 64;
    const int BN_PAD = 136;

    __shared__ __align__(128) __half smem_A[BM * BK];
    __shared__ __align__(128) __half smem_B[BK * BN_PAD];

    const int n_start = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx     = tid + i * 128;
            int row     = idx >> 3;
            int col_f4  = idx & 7;
            int phys_f4 = col_f4 ^ (row & 7);
            __half* dst       = smem_A + row * BK + phys_f4 * 8;
            const __half* src = A + row * BK + col_f4 * 8;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx      = tid + i * 128;
            int k_row    = idx >> 4;
            int n_chunk  = idx & 15;
            int n_local  = n_chunk * 8;
            int global_n = n_start + n_local;
            __half* dst       = smem_B + k_row * BN_PAD + n_local;
            const __half* src = B + (int64_t)k_row * N + global_n;
            __pipeline_memcpy_async(dst, src, 16);
        }
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    const int warp_row    = warp_id >> 1;
    const int warp_col    = warp_id & 1;
    const int warp_m_base = warp_row * 64;
    const int warp_n_base = warp_col * 64;

    float acc[4][8][4];
    #pragma unroll
    for (int m = 0; m < 4; m++)
        #pragma unroll
        for (int n = 0; n < 8; n++)
            acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;

    uint32_t a_frag[2][4][4];
    uint32_t b_frag[2][8][2];

    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        int row      = warp_m_base + mt * 16 + (lane & 15);
        int log_col  = (lane >> 4) << 3;
        int phys_col = log_col ^ ((row & 7) << 3);
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_frag[0][mt][0]), "=r"(a_frag[0][mt][1]),
              "=r"(a_frag[0][mt][2]), "=r"(a_frag[0][mt][3])
            : "r"(addr)
        );
    }
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int b_n_base = warp_n_base + nt * 8;
        int b_k_row  = lane & 15;
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + b_n_base);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b_frag[0][nt][0]), "=r"(b_frag[0][nt][1])
            : "r"(addr)
        );
    }

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        const int cur = kt & 1;
        const int nxt = 1 - cur;

        if (kt < 3) {
            const int nk = (kt + 1) * 16;

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int b_n_base  = warp_n_base + nt * 8;
                int b_k_row   = nk + (lane & 15);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_B + b_k_row * BN_PAD + b_n_base);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(b_frag[nxt][nt][0]), "=r"(b_frag[nxt][nt][1])
                    : "r"(addr)
                );
            }

            #pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int row      = warp_m_base + mt * 16 + (lane & 15);
                int log_col  = nk + ((lane >> 4) << 3);
                int phys_col = log_col ^ ((row & 7) << 3);
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_A + row * BK + phys_col);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a_frag[nxt][mt][0]), "=r"(a_frag[nxt][mt][1]),
                      "=r"(a_frag[nxt][mt][2]), "=r"(a_frag[nxt][mt][3])
                    : "r"(addr)
                );
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(acc[mt][nt][0]), "=f"(acc[mt][nt][1]),
                          "=f"(acc[mt][nt][2]), "=f"(acc[mt][nt][3])
                        : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                          "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                          "r"(b_frag[cur][nt][0]), "r"(b_frag[cur][nt][1]),
                          "f"(acc[mt][nt][0]), "f"(acc[mt][nt][1]),
                          "f"(acc[mt][nt][2]), "f"(acc[mt][nt][3])
                    );
                }
            }
        } else {
            #pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(acc[mt][nt][0]), "=f"(acc[mt][nt][1]),
                          "=f"(acc[mt][nt][2]), "=f"(acc[mt][nt][3])
                        : "r"(a_frag[cur][mt][0]), "r"(a_frag[cur][mt][1]),
                          "r"(a_frag[cur][mt][2]), "r"(a_frag[cur][mt][3]),
                          "r"(b_frag[cur][nt][0]), "r"(b_frag[cur][nt][1]),
                          "f"(acc[mt][nt][0]), "f"(acc[mt][nt][1]),
                          "f"(acc[mt][nt][2]), "f"(acc[mt][nt][3])
                    );
                }
            }
        }
    }

    const int out_n_base = n_start + warp_n_base;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++) {
        int row0 = warp_m_base + mt * 16 + (lane >> 2);
        int row1 = row0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int col = out_n_base + nt * 8 + (lane & 3) * 2;
            __half2 h01 = __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
            __half2 h23 = __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
            *reinterpret_cast<__half2*>(C + row0 * N + col) = h01;
            *reinterpret_cast<__half2*>(C + row1 * N + col) = h23;
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

    const __half* A_ptr = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* B_ptr = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    __half*       C_ptr = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    {
        const int BN = 32;
        dim3 grid((N + BN - 1) / BN);
        dim3 block(128);
        hgemm_bn32_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    }
}