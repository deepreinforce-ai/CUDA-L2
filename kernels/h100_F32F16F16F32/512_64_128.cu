#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <stdio.h>

#define BM        16
#define BK        16
#define BN        64
#define WTN       8
#define STAGES    8
#define SA_STR    24
#define SB_STR    64

static __device__ __forceinline__ uint32_t cvt_smem(const void* ptr) {
    uint32_t r;
    asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
                 : "=r"(r) : "l"(ptr));
    return r;
}

__device__ __forceinline__ int swz_b(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__global__ __launch_bounds__(32, 16)
void hgemm_kernel_main(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int bm         = blockIdx.x;
    const int lane       = threadIdx.x;
    const int g_row_base = bm * BM;

    extern __shared__ half smem[];
    half* smem_A = smem;
    half* smem_B = smem + STAGES * BM * SA_STR;

    float acc[WTN][4];
    #pragma unroll
    for (int j = 0; j < WTN; j++)
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        {
            half* sa  = smem_A + s * BM * SA_STR;
            int row   = lane >> 1;
            int col   = (lane & 1) << 3;
            int g_row = g_row_base + row;
            uint32_t addr = cvt_smem(sa + row * SA_STR + col);
            if (g_row < M) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr), "l"((const void*)&A[g_row * 128 + s * BK + col]));
            } else {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                             :: "r"(addr), "l"((const void*)A) : "memory");
            }
        }
        {
            half* sb = smem_B + s * BK * SB_STR;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int g     = lane * 4 + i;
                int row   = g >> 3;
                int col8  = (g & 7) << 3;
                int pcol  = swz_b(row, col8);
                uint32_t addr = cvt_smem(sb + row * SB_STR + pcol);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr), "l"((const void*)&B[(s * BK + row) * BN + col8]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    asm volatile("cp.async.wait_all;\n" :: : "memory");
    __syncwarp();

    #pragma unroll
    for (int k = 0; k < STAGES; k++) {
        const half* sa = smem_A + k * BM * SA_STR;
        const half* sb = smem_B + k * BK * SB_STR;

        uint32_t ra[4];
        {
            int r = lane & 15;
            int c = (lane >> 4) << 3;
            uint32_t addr = cvt_smem(sa + r * SA_STR + c);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
                         : "r"(addr));
        }

        uint32_t rb[WTN][2];
        #pragma unroll
        for (int ni = 0; ni < WTN; ni++) {
            int r     = lane & 7;
            int mat   = (lane >> 3) & 1;
            int b_row = r + mat * 8;
            int b_col = ni * 8;
            int pcol  = swz_b(b_row, b_col);
            uint32_t addr = cvt_smem(sb + b_row * SB_STR + pcol);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(rb[ni][0]), "=r"(rb[ni][1])
                         : "r"(addr));
        }

        #pragma unroll
        for (int ni = 0; ni < WTN; ni++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                : "=f"(acc[ni][0]), "=f"(acc[ni][1]),
                  "=f"(acc[ni][2]), "=f"(acc[ni][3])
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
                  "r"(rb[ni][0]), "r"(rb[ni][1]),
                  "f"(acc[ni][0]), "f"(acc[ni][1]),
                  "f"(acc[ni][2]), "f"(acc[ni][3])
            );
        }
    }

    #pragma unroll
    for (int ni = 0; ni < WTN; ni++) {
        int col0 = ni * 8 + (lane & 3) * 2;
        int row0 = g_row_base + (lane >> 2);
        int row1 = g_row_base + (lane >> 2) + 8;
        *reinterpret_cast<half2*>(&C[row0 * BN + col0]) = __floats2half2_rn(acc[ni][0], acc[ni][1]);
        *reinterpret_cast<half2*>(&C[row1 * BN + col0]) = __floats2half2_rn(acc[ni][2], acc[ni][3]);
    }
}

#define BM2       32
#define STAGES2   8

__global__ __launch_bounds__(32, 10)
void hgemm_kernel_bm32(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int bm         = blockIdx.x;
    const int lane       = threadIdx.x;
    const int g_row_base = bm * BM2;

    extern __shared__ half smem2[];
    half* smem_A0 = smem2;
    half* smem_A1 = smem2 + STAGES2 * BM * SA_STR;
    half* smem_B  = smem2 + 2 * STAGES2 * BM * SA_STR;

    float acc0[WTN][4], acc1[WTN][4];
    #pragma unroll
    for (int j = 0; j < WTN; j++) {
        acc0[j][0] = acc0[j][1] = acc0[j][2] = acc0[j][3] = 0.f;
        acc1[j][0] = acc1[j][1] = acc1[j][2] = acc1[j][3] = 0.f;
    }

    const int a_row  = lane >> 1;
    const int a_col  = (lane & 1) << 3;

    #pragma unroll
    for (int s = 0; s < STAGES2; s++) {
        {
            half* sa  = smem_A0 + s * BM * SA_STR;
            int g_row = g_row_base + a_row;
            uint32_t addr = cvt_smem(sa + a_row * SA_STR + a_col);
            if (g_row < M) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr), "l"((const void*)&A[g_row * 128 + s * BK + a_col]));
            } else {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                             :: "r"(addr), "l"((const void*)A) : "memory");
            }
        }
        {
            half* sa  = smem_A1 + s * BM * SA_STR;
            int g_row = g_row_base + 16 + a_row;
            uint32_t addr = cvt_smem(sa + a_row * SA_STR + a_col);
            if (g_row < M) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr), "l"((const void*)&A[g_row * 128 + s * BK + a_col]));
            } else {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                             :: "r"(addr), "l"((const void*)A) : "memory");
            }
        }
        {
            half* sb = smem_B + s * BK * SB_STR;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int g    = lane * 4 + i;
                int row  = g >> 3;
                int col8 = (g & 7) << 3;
                int pcol = swz_b(row, col8);
                uint32_t addr = cvt_smem(sb + row * SB_STR + pcol);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(addr), "l"((const void*)&B[(s * BK + row) * BN + col8]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    asm volatile("cp.async.wait_all;\n" :: : "memory");
    __syncwarp();

    #pragma unroll
    for (int k = 0; k < STAGES2; k++) {
        const half* sa0 = smem_A0 + k * BM * SA_STR;
        const half* sa1 = smem_A1 + k * BM * SA_STR;
        const half* sb  = smem_B  + k * BK * SB_STR;

        uint32_t ra0[4], ra1[4];
        {
            int r = lane & 15;
            int c = (lane >> 4) << 3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(ra0[0]), "=r"(ra0[1]), "=r"(ra0[2]), "=r"(ra0[3])
                         : "r"(cvt_smem(sa0 + r * SA_STR + c)));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(ra1[0]), "=r"(ra1[1]), "=r"(ra1[2]), "=r"(ra1[3])
                         : "r"(cvt_smem(sa1 + r * SA_STR + c)));
        }

        uint32_t rb[WTN][2];
        #pragma unroll
        for (int ni = 0; ni < WTN; ni++) {
            int r     = lane & 7;
            int mat   = (lane >> 3) & 1;
            int b_row = r + mat * 8;
            int b_col = ni * 8;
            int pcol  = swz_b(b_row, b_col);
            uint32_t addr = cvt_smem(sb + b_row * SB_STR + pcol);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(rb[ni][0]), "=r"(rb[ni][1])
                         : "r"(addr));
        }

        #pragma unroll
        for (int ni = 0; ni < WTN; ni++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                : "=f"(acc0[ni][0]), "=f"(acc0[ni][1]),
                  "=f"(acc0[ni][2]), "=f"(acc0[ni][3])
                : "r"(ra0[0]), "r"(ra0[1]), "r"(ra0[2]), "r"(ra0[3]),
                  "r"(rb[ni][0]), "r"(rb[ni][1]),
                  "f"(acc0[ni][0]), "f"(acc0[ni][1]),
                  "f"(acc0[ni][2]), "f"(acc0[ni][3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                : "=f"(acc1[ni][0]), "=f"(acc1[ni][1]),
                  "=f"(acc1[ni][2]), "=f"(acc1[ni][3])
                : "r"(ra1[0]), "r"(ra1[1]), "r"(ra1[2]), "r"(ra1[3]),
                  "r"(rb[ni][0]), "r"(rb[ni][1]),
                  "f"(acc1[ni][0]), "f"(acc1[ni][1]),
                  "f"(acc1[ni][2]), "f"(acc1[ni][3])
            );
        }
    }

    #pragma unroll
    for (int ni = 0; ni < WTN; ni++) {
        int col0 = ni * 8 + (lane & 3) * 2;
        int row0 = g_row_base + (lane >> 2);
        int row1 = g_row_base + (lane >> 2) + 8;
        if (row0 < M)
            *reinterpret_cast<half2*>(&C[row0 * BN + col0]) = __floats2half2_rn(acc0[ni][0], acc0[ni][1]);
        if (row1 < M)
            *reinterpret_cast<half2*>(&C[row1 * BN + col0]) = __floats2half2_rn(acc0[ni][2], acc0[ni][3]);
    }
    #pragma unroll
    for (int ni = 0; ni < WTN; ni++) {
        int col0 = ni * 8 + (lane & 3) * 2;
        int row0 = g_row_base + 16 + (lane >> 2);
        int row1 = g_row_base + 16 + (lane >> 2) + 8;
        if (row0 < M)
            *reinterpret_cast<half2*>(&C[row0 * BN + col0]) = __floats2half2_rn(acc1[ni][0], acc1[ni][1]);
        if (row1 < M)
            *reinterpret_cast<half2*>(&C[row1 * BN + col0]) = __floats2half2_rn(acc1[ni][2], acc1[ni][3]);
    }
}

#define ST2 2

__global__ __launch_bounds__(32, 20)
void hgemm_kernel_2stage(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M
) {
    const int bm         = blockIdx.x;
    const int lane       = threadIdx.x;
    const int g_row_base = bm * BM;
    const int num_k      = 8;

    extern __shared__ half smem_s2[];
    half* smem_A = smem_s2;
    half* smem_B = smem_s2 + ST2 * BM * SA_STR;

    float acc[WTN][4];
    #pragma unroll
    for (int j = 0; j < WTN; j++)
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    auto ldA = [&](int kt, int st) __attribute__((always_inline)) {
        half* sa  = smem_A + st * BM * SA_STR;
        int row   = lane >> 1;
        int col   = (lane & 1) << 3;
        int g_row = g_row_base + row;
        uint32_t addr = cvt_smem(sa + row * SA_STR + col);
        if (g_row < M) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addr), "l"((const void*)&A[g_row * 128 + kt * BK + col]));
        } else {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                         :: "r"(addr), "l"((const void*)A) : "memory");
        }
    };

    auto ldB = [&](int kt, int st) __attribute__((always_inline)) {
        half* sb = smem_B + st * BK * SB_STR;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int g    = lane * 4 + i;
            int row  = g >> 3;
            int col8 = (g & 7) << 3;
            int pcol = swz_b(row, col8);
            uint32_t addr = cvt_smem(sb + row * SB_STR + pcol);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(addr), "l"((const void*)&B[(kt * BK + row) * BN + col8]));
        }
    };

    ldA(0, 0); ldB(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int wst = 1, rst = 0;

    #pragma unroll 1
    for (int k = 0; k < num_k; k++) {
        if (k + 1 < num_k) { ldA(k + 1, wst); ldB(k + 1, wst); }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" :: : "memory");
        __syncwarp();

        const half* sa = smem_A + rst * BM * SA_STR;
        const half* sb = smem_B + rst * BK * SB_STR;

        uint32_t ra[4];
        {
            int r = lane & 15, c = (lane >> 4) << 3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
                         : "r"(cvt_smem(sa + r * SA_STR + c)));
        }

        uint32_t rb[WTN][2];
        #pragma unroll
        for (int ni = 0; ni < WTN; ni++) {
            int r = lane & 7, mat = (lane >> 3) & 1;
            int b_row = r + mat * 8, b_col = ni * 8;
            uint32_t addr = cvt_smem(sb + b_row * SB_STR + swz_b(b_row, b_col));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(rb[ni][0]), "=r"(rb[ni][1]) : "r"(addr));
        }

        #pragma unroll
        for (int ni = 0; ni < WTN; ni++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                : "=f"(acc[ni][0]), "=f"(acc[ni][1]), "=f"(acc[ni][2]), "=f"(acc[ni][3])
                : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
                  "r"(rb[ni][0]), "r"(rb[ni][1]),
                  "f"(acc[ni][0]), "f"(acc[ni][1]), "f"(acc[ni][2]), "f"(acc[ni][3])
            );
        }
        rst ^= 1; wst ^= 1;
    }

    asm volatile("cp.async.wait_all;\n" :: : "memory");

    #pragma unroll
    for (int ni = 0; ni < WTN; ni++) {
        int col0 = ni * 8 + (lane & 3) * 2;
        int row0 = g_row_base + (lane >> 2);
        int row1 = g_row_base + (lane >> 2) + 8;
        if (row0 < M)
            *reinterpret_cast<half2*>(&C[row0 * BN + col0]) = __floats2half2_rn(acc[ni][0], acc[ni][1]);
        if (row1 < M)
            *reinterpret_cast<half2*>(&C[row1 * BN + col0]) = __floats2half2_rn(acc[ni][2], acc[ni][3]);
    }
}

static int s_best_kernel = -1;

void cuda_l2_h100_fp32(
    torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
    const int M = a.size(0);
    const half* A_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* C_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const size_t smem_main   = (size_t)STAGES  * (BM * SA_STR + BK * SB_STR) * sizeof(half);
    const size_t smem_bm32   = (size_t)(2 * STAGES2 * BM * SA_STR + STAGES2 * BK * SB_STR) * sizeof(half);
    const size_t smem_2stage = (size_t)ST2     * (BM * SA_STR + BK * SB_STR) * sizeof(half);

    const int tiles_main   = (M + BM  - 1) / BM;
    const int tiles_bm32   = (M + BM2 - 1) / BM2;

    if (s_best_kernel < 0) {
        cudaFuncSetAttribute(hgemm_kernel_main,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_main);
        cudaFuncSetAttribute(hgemm_kernel_main,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(hgemm_kernel_bm32,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bm32);
        cudaFuncSetAttribute(hgemm_kernel_bm32,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(hgemm_kernel_2stage,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_2stage);
        cudaFuncSetAttribute(hgemm_kernel_2stage,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);

        hgemm_kernel_main<<<tiles_main, 32, smem_main>>>(A_ptr, B_ptr, C_ptr, M);
        hgemm_kernel_bm32<<<tiles_bm32, 32, smem_bm32>>>(A_ptr, B_ptr, C_ptr, M);
        hgemm_kernel_2stage<<<tiles_main, 32, smem_2stage>>>(A_ptr, B_ptr, C_ptr, M);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int NITERS = 200;
        float times[3];

        cudaEventRecord(e0);
        for (int i = 0; i < NITERS; i++)
            hgemm_kernel_main<<<tiles_main, 32, smem_main>>>(A_ptr, B_ptr, C_ptr, M);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&times[0], e0, e1);

        cudaEventRecord(e0);
        for (int i = 0; i < NITERS; i++)
            hgemm_kernel_bm32<<<tiles_bm32, 32, smem_bm32>>>(A_ptr, B_ptr, C_ptr, M);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&times[1], e0, e1);

        cudaEventRecord(e0);
        for (int i = 0; i < NITERS; i++)
            hgemm_kernel_2stage<<<tiles_main, 32, smem_2stage>>>(A_ptr, B_ptr, C_ptr, M);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&times[2], e0, e1);

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);

        s_best_kernel = 0;
        if (times[1] < times[s_best_kernel]) s_best_kernel = 1;
        if (times[2] < times[s_best_kernel]) s_best_kernel = 2;
    }

    if (s_best_kernel == 0) {
        hgemm_kernel_main<<<tiles_main, 32, smem_main>>>(A_ptr, B_ptr, C_ptr, M);
    } else if (s_best_kernel == 1) {
        hgemm_kernel_bm32<<<tiles_bm32, 32, smem_bm32>>>(A_ptr, B_ptr, C_ptr, M);
    } else {
        hgemm_kernel_2stage<<<tiles_main, 32, smem_2stage>>>(A_ptr, B_ptr, C_ptr, M);
    }
}