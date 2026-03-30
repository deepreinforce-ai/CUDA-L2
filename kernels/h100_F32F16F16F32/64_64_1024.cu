#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <iostream>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

static constexpr int CM        = 64;
static constexpr int CN        = 64;
static constexpr int KSTEP     = 64;
static constexpr int STAGES    = 8;
static constexpr int A_STRIDE  = KSTEP + 8;
static constexpr int B_STRIDE  = CN + 8;
static constexpr int A_STAGE   = CM * A_STRIDE;
static constexpr int B_STAGE   = KSTEP * B_STRIDE;

static constexpr size_t SMEM_PIPELINE = (size_t)STAGES * (A_STAGE + B_STAGE) * sizeof(__half);
static constexpr size_t SMEM_OUTPUT   = (size_t)CM * CN * sizeof(float);
static constexpr size_t SMEM_SIZE     = SMEM_PIPELINE > SMEM_OUTPUT ? SMEM_PIPELINE : SMEM_OUTPUT;

__global__ void __launch_bounds__(128, 1)
hgemm_wmma_best(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int K)
{
    extern __shared__ __half smem[];
    __half* smemA = smem;
    __half* smemB = smem + STAGES * A_STAGE;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int wy      = warp_id >> 1;
    const int wx      = warp_id & 1;
    const int wrow    = wy * 32;
    const int wcol    = wx * 32;

    using namespace nvcuda::wmma;
    fragment<accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    const int k_tiles = K / KSTEP;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < k_tiles) {
            const int k_off = s * KSTEP;
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int a_row  = linear / KSTEP;
                int a_col  = linear % KSTEP;
                __half* dstA = smemA + s * A_STAGE + a_row * A_STRIDE + a_col;
                const __half* srcA = A + a_row * K + k_off + a_col;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstA));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcA) : "memory");
            }
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int b_k    = linear / CN;
                int b_n    = linear % CN;
                __half* dstB = smemB + s * B_STAGE + b_k * B_STRIDE + b_n;
                const __half* srcB = B + (k_off + b_k) * CN + b_n;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstB));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcB) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    #pragma unroll 2
    for (int k_iter = 0; k_iter < k_tiles; k_iter++) {
        const int prefetch = k_iter + STAGES - 1;
        if (prefetch < k_tiles) {
            const int sw    = prefetch % STAGES;
            const int k_off = prefetch * KSTEP;
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int a_row  = linear / KSTEP;
                int a_col  = linear % KSTEP;
                __half* dstA = smemA + sw * A_STAGE + a_row * A_STRIDE + a_col;
                const __half* srcA = A + a_row * K + k_off + a_col;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstA));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcA) : "memory");
            }
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int b_k    = linear / CN;
                int b_n    = linear % CN;
                __half* dstB = smemB + sw * B_STAGE + b_k * B_STRIDE + b_n;
                const __half* srcB = B + (k_off + b_k) * CN + b_n;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstB));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcB) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1) : "memory");
        __syncthreads();

        const int cs = k_iter % STAGES;
        const __half* curA = smemA + cs * A_STAGE;
        const __half* curB = smemB + cs * B_STAGE;

        fragment<matrix_a, 16, 16, 16, __half, row_major> fa[2][4];
        fragment<matrix_b, 16, 16, 16, __half, row_major> fb0, fb1;

        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            load_matrix_sync(fa[0][ks], curA + (wrow +  0) * A_STRIDE + ks*16, A_STRIDE);
            load_matrix_sync(fa[1][ks], curA + (wrow + 16) * A_STRIDE + ks*16, A_STRIDE);
        }

        load_matrix_sync(fb0, curB + (0*16) * B_STRIDE + (wcol +  0), B_STRIDE);
        load_matrix_sync(fb1, curB + (0*16) * B_STRIDE + (wcol + 16), B_STRIDE);
        mma_sync(acc[0][0], fa[0][0], fb0, acc[0][0]);
        mma_sync(acc[0][1], fa[0][0], fb1, acc[0][1]);
        mma_sync(acc[1][0], fa[1][0], fb0, acc[1][0]);
        mma_sync(acc[1][1], fa[1][0], fb1, acc[1][1]);

        load_matrix_sync(fb0, curB + (1*16) * B_STRIDE + (wcol +  0), B_STRIDE);
        load_matrix_sync(fb1, curB + (1*16) * B_STRIDE + (wcol + 16), B_STRIDE);
        mma_sync(acc[0][0], fa[0][1], fb0, acc[0][0]);
        mma_sync(acc[0][1], fa[0][1], fb1, acc[0][1]);
        mma_sync(acc[1][0], fa[1][1], fb0, acc[1][0]);
        mma_sync(acc[1][1], fa[1][1], fb1, acc[1][1]);

        load_matrix_sync(fb0, curB + (2*16) * B_STRIDE + (wcol +  0), B_STRIDE);
        load_matrix_sync(fb1, curB + (2*16) * B_STRIDE + (wcol + 16), B_STRIDE);
        mma_sync(acc[0][0], fa[0][2], fb0, acc[0][0]);
        mma_sync(acc[0][1], fa[0][2], fb1, acc[0][1]);
        mma_sync(acc[1][0], fa[1][2], fb0, acc[1][0]);
        mma_sync(acc[1][1], fa[1][2], fb1, acc[1][1]);

        load_matrix_sync(fb0, curB + (3*16) * B_STRIDE + (wcol +  0), B_STRIDE);
        load_matrix_sync(fb1, curB + (3*16) * B_STRIDE + (wcol + 16), B_STRIDE);
        mma_sync(acc[0][0], fa[0][3], fb0, acc[0][0]);
        mma_sync(acc[0][1], fa[0][3], fb1, acc[0][1]);
        mma_sync(acc[1][0], fa[1][3], fb0, acc[1][0]);
        mma_sync(acc[1][1], fa[1][3], fb1, acc[1][1]);

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");

    float* fsmem = reinterpret_cast<float*>(smem);

    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            store_matrix_sync(fsmem + (wrow + mi*16) * CN + (wcol + ni*16),
                              acc[mi][ni], CN, nvcuda::wmma::mem_row_major);

    __syncthreads();

    #pragma unroll 16
    for (int i = tid * 2; i < CM * CN; i += 128 * 2) {
        __half2 h2 = __float22half2_rn(make_float2(fsmem[i], fsmem[i+1]));
        *reinterpret_cast<__half2*>(C + i) = h2;
    }
}

static constexpr int HM       = 32;
static constexpr int HA_STR   = KSTEP + 8;
static constexpr int HB_STR   = CN + 8;
static constexpr int HA_STG   = HM * HA_STR;
static constexpr int HB_STG   = KSTEP * HB_STR;
static constexpr int HSTAGES  = 8;

static constexpr size_t HSMEM_PIPE = (size_t)HSTAGES * (HA_STG + HB_STG) * sizeof(__half);
static constexpr size_t HSMEM_OUT  = (size_t)HM * CN * sizeof(float);
static constexpr size_t HSMEM_SIZE = HSMEM_PIPE > HSMEM_OUT ? HSMEM_PIPE : HSMEM_OUT;

__global__ void __launch_bounds__(128, 1)
hgemm_2cta_best(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int K)
{
    const int m_start = blockIdx.x * HM;

    extern __shared__ __half smem[];
    __half* smemA = smem;
    __half* smemB = smem + HSTAGES * HA_STG;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int wy      = warp_id >> 1;
    const int wx      = warp_id & 1;
    const int wrow    = wy * 16;
    const int wcol    = wx * 32;

    using namespace nvcuda::wmma;
    fragment<accumulator, 16, 16, 16, float> acc[2];
    fill_fragment(acc[0], 0.0f);
    fill_fragment(acc[1], 0.0f);

    const int k_tiles = K / KSTEP;

    #pragma unroll
    for (int s = 0; s < HSTAGES - 1; s++) {
        if (s < k_tiles) {
            const int k_off = s * KSTEP;
            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                int linear = pass * 1024 + tid * 8;
                int a_row  = linear / KSTEP;
                int a_col  = linear % KSTEP;
                if (a_row < HM) {
                    __half* dstA = smemA + s * HA_STG + a_row * HA_STR + a_col;
                    const __half* srcA = A + (m_start + a_row) * K + k_off + a_col;
                    uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstA));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"(sp), "l"(srcA) : "memory");
                }
            }
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int b_k    = linear / CN;
                int b_n    = linear % CN;
                __half* dstB = smemB + s * HB_STG + b_k * HB_STR + b_n;
                const __half* srcB = B + (k_off + b_k) * CN + b_n;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstB));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcB) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    #pragma unroll 2
    for (int k_iter = 0; k_iter < k_tiles; k_iter++) {
        const int prefetch = k_iter + HSTAGES - 1;
        if (prefetch < k_tiles) {
            const int sw    = prefetch % HSTAGES;
            const int k_off = prefetch * KSTEP;
            #pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                int linear = pass * 1024 + tid * 8;
                int a_row  = linear / KSTEP;
                int a_col  = linear % KSTEP;
                if (a_row < HM) {
                    __half* dstA = smemA + sw * HA_STG + a_row * HA_STR + a_col;
                    const __half* srcA = A + (m_start + a_row) * K + k_off + a_col;
                    uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstA));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"(sp), "l"(srcA) : "memory");
                }
            }
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int b_k    = linear / CN;
                int b_n    = linear % CN;
                __half* dstB = smemB + sw * HB_STG + b_k * HB_STR + b_n;
                const __half* srcB = B + (k_off + b_k) * CN + b_n;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstB));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcB) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(HSTAGES - 1) : "memory");
        __syncthreads();

        const int cs = k_iter % HSTAGES;
        const __half* curA = smemA + cs * HA_STG;
        const __half* curB = smemB + cs * HB_STG;

        fragment<matrix_a, 16, 16, 16, __half, row_major> fa[4];
        fragment<matrix_b, 16, 16, 16, __half, row_major> fb0, fb1;

        #pragma unroll
        for (int ks = 0; ks < 4; ks++)
            load_matrix_sync(fa[ks], curA + wrow * HA_STR + ks*16, HA_STR);

        load_matrix_sync(fb0, curB + (0*16) * HB_STR + (wcol +  0), HB_STR);
        load_matrix_sync(fb1, curB + (0*16) * HB_STR + (wcol + 16), HB_STR);
        mma_sync(acc[0], fa[0], fb0, acc[0]);
        mma_sync(acc[1], fa[0], fb1, acc[1]);

        load_matrix_sync(fb0, curB + (1*16) * HB_STR + (wcol +  0), HB_STR);
        load_matrix_sync(fb1, curB + (1*16) * HB_STR + (wcol + 16), HB_STR);
        mma_sync(acc[0], fa[1], fb0, acc[0]);
        mma_sync(acc[1], fa[1], fb1, acc[1]);

        load_matrix_sync(fb0, curB + (2*16) * HB_STR + (wcol +  0), HB_STR);
        load_matrix_sync(fb1, curB + (2*16) * HB_STR + (wcol + 16), HB_STR);
        mma_sync(acc[0], fa[2], fb0, acc[0]);
        mma_sync(acc[1], fa[2], fb1, acc[1]);

        load_matrix_sync(fb0, curB + (3*16) * HB_STR + (wcol +  0), HB_STR);
        load_matrix_sync(fb1, curB + (3*16) * HB_STR + (wcol + 16), HB_STR);
        mma_sync(acc[0], fa[3], fb0, acc[0]);
        mma_sync(acc[1], fa[3], fb1, acc[1]);

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");

    float* fsmem = reinterpret_cast<float*>(smem);
    store_matrix_sync(fsmem + wrow * CN + (wcol +  0), acc[0], CN, nvcuda::wmma::mem_row_major);
    store_matrix_sync(fsmem + wrow * CN + (wcol + 16), acc[1], CN, nvcuda::wmma::mem_row_major);
    __syncthreads();

    __half* C_out = C + m_start * CN;
    #pragma unroll 4
    for (int i = tid * 2; i < HM * CN; i += 128 * 2) {
        __half2 h2 = __float22half2_rn(make_float2(fsmem[i], fsmem[i+1]));
        *reinterpret_cast<__half2*>(C_out + i) = h2;
    }
}

static constexpr int QM       = 16;
static constexpr int QA_STR   = KSTEP + 8;
static constexpr int QB_STR   = CN + 8;
static constexpr int QA_STG   = QM * QA_STR;
static constexpr int QB_STG   = KSTEP * QB_STR;
static constexpr int QSTAGES  = 8;

static constexpr size_t QSMEM_PIPE = (size_t)QSTAGES * (QA_STG + QB_STG) * sizeof(__half);
static constexpr size_t QSMEM_OUT  = (size_t)QM * CN * sizeof(float);
static constexpr size_t QSMEM_SIZE = QSMEM_PIPE > QSMEM_OUT ? QSMEM_PIPE : QSMEM_OUT;

__global__ void __launch_bounds__(128, 1)
hgemm_4cta_best(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int K)
{
    const int m_start = blockIdx.x * QM;

    extern __shared__ __half smem[];
    __half* smemA = smem;
    __half* smemB = smem + QSTAGES * QA_STG;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int wcol    = warp_id * 16;

    using namespace nvcuda::wmma;
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);

    const int k_tiles = K / KSTEP;

    #pragma unroll
    for (int s = 0; s < QSTAGES - 1; s++) {
        if (s < k_tiles) {
            const int k_off = s * KSTEP;
            {
                int linear = tid * 8;
                if (linear < QM * KSTEP) {
                    int a_row = linear / KSTEP;
                    int a_col = linear % KSTEP;
                    __half* dstA = smemA + s * QA_STG + a_row * QA_STR + a_col;
                    const __half* srcA = A + (m_start + a_row) * K + k_off + a_col;
                    uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstA));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"(sp), "l"(srcA) : "memory");
                }
            }
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int b_k    = linear / CN;
                int b_n    = linear % CN;
                __half* dstB = smemB + s * QB_STG + b_k * QB_STR + b_n;
                const __half* srcB = B + (k_off + b_k) * CN + b_n;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstB));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcB) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    #pragma unroll 2
    for (int k_iter = 0; k_iter < k_tiles; k_iter++) {
        const int prefetch = k_iter + QSTAGES - 1;
        if (prefetch < k_tiles) {
            const int sw    = prefetch % QSTAGES;
            const int k_off = prefetch * KSTEP;
            {
                int linear = tid * 8;
                if (linear < QM * KSTEP) {
                    int a_row = linear / KSTEP;
                    int a_col = linear % KSTEP;
                    __half* dstA = smemA + sw * QA_STG + a_row * QA_STR + a_col;
                    const __half* srcA = A + (m_start + a_row) * K + k_off + a_col;
                    uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstA));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                 :: "r"(sp), "l"(srcA) : "memory");
                }
            }
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {
                int linear = pass * 1024 + tid * 8;
                int b_k    = linear / CN;
                int b_n    = linear % CN;
                __half* dstB = smemB + sw * QB_STG + b_k * QB_STR + b_n;
                const __half* srcB = B + (k_off + b_k) * CN + b_n;
                uint32_t sp = static_cast<uint32_t>(__cvta_generic_to_shared(dstB));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(sp), "l"(srcB) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group %0;\n" :: "n"(QSTAGES - 1) : "memory");
        __syncthreads();

        const int cs = k_iter % QSTAGES;
        const __half* curA = smemA + cs * QA_STG;
        const __half* curB = smemB + cs * QB_STG;

        fragment<matrix_a, 16, 16, 16, __half, row_major> fa[4];
        fragment<matrix_b, 16, 16, 16, __half, row_major> fb[4];

        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            load_matrix_sync(fa[ks], curA + 0 * QA_STR + ks*16, QA_STR);
            load_matrix_sync(fb[ks], curB + (ks*16) * QB_STR + wcol, QB_STR);
            mma_sync(acc, fa[ks], fb[ks], acc);
        }

        __syncthreads();
    }

    asm volatile("cp.async.wait_all;\n" ::: "memory");

    float* fsmem = reinterpret_cast<float*>(smem);
    store_matrix_sync(fsmem + 0 * CN + wcol, acc, CN, nvcuda::wmma::mem_row_major);
    __syncthreads();

    __half* C_out = C + m_start * CN;
    #pragma unroll 2
    for (int i = tid * 2; i < QM * CN; i += 128 * 2) {
        __half2 h2 = __float22half2_rn(make_float2(fsmem[i], fsmem[i+1]));
        *reinterpret_cast<__half2*>(C_out + i) = h2;
    }
}

static bool g_attrs_set = false;
static int  g_winner    = -1;

static void setup_attrs() {
    if (g_attrs_set) return;
    cudaFuncSetAttribute(hgemm_wmma_best,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_SIZE);
    cudaFuncSetAttribute(hgemm_wmma_best,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(hgemm_2cta_best,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)HSMEM_SIZE);
    cudaFuncSetAttribute(hgemm_2cta_best,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(hgemm_4cta_best,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)QSMEM_SIZE);
    cudaFuncSetAttribute(hgemm_4cta_best,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    g_attrs_set = true;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c)
{
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int K = static_cast<int>(a.size(1));
    const __half* raw_A = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* raw_B = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    __half*       raw_C = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    setup_attrs();

    if (g_winner == 0) {
        hgemm_wmma_best<<<1, 128, SMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
        return;
    }
    if (g_winner == 1) {
        hgemm_2cta_best<<<2, 128, HSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
        return;
    }
    if (g_winner == 2) {
        hgemm_4cta_best<<<4, 128, QSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
        return;
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    const int WARMUP = 10;
    const int ITERS  = 100;
    float ms[3];

    for (int i = 0; i < WARMUP; i++)
        hgemm_wmma_best<<<1, 128, SMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++)
        hgemm_wmma_best<<<1, 128, SMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms[0], t0, t1);

    for (int i = 0; i < WARMUP; i++)
        hgemm_2cta_best<<<2, 128, HSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++)
        hgemm_2cta_best<<<2, 128, HSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms[1], t0, t1);

    for (int i = 0; i < WARMUP; i++)
        hgemm_4cta_best<<<4, 128, QSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++)
        hgemm_4cta_best<<<4, 128, QSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms[2], t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    g_winner = 0;
    if (ms[1] < ms[g_winner]) g_winner = 1;
    if (ms[2] < ms[g_winner]) g_winner = 2;

    if (g_winner == 0)
        hgemm_wmma_best<<<1, 128, SMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    else if (g_winner == 1)
        hgemm_2cta_best<<<2, 128, HSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);
    else
        hgemm_4cta_best<<<4, 128, QSMEM_SIZE>>>(raw_A, raw_B, raw_C, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("hgemm failed: ") + cudaGetErrorString(err));
}