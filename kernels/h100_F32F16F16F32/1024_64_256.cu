#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <torch/extension.h>
#include <torch/types.h>

using namespace nvcuda;

#define APAD 8
#define A_STRIDE (256 + APAD)

static __device__ __forceinline__
void cp_async16(void* dst, const void* src) {
    uint32_t dst_addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(dst_addr) : "l"(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst_addr), "l"(src) : "memory");
}

static __device__ __forceinline__
uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

__global__ void __launch_bounds__(32, 64)
hgemm_ptx_mma(
    const half* __restrict__ A,
    const half* __restrict__ B_col_major,
    half* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ char smem_raw[];
    half* smem_B = reinterpret_cast<half*>(smem_raw);
    half* smem_A = reinterpret_cast<half*>(smem_raw + 32768);

    const int lane = threadIdx.x;
    const int block_row = blockIdx.x * 16;

    {
        const float4* B_f4 = reinterpret_cast<const float4*>(B_col_major);
        float4* smB_f4 = reinterpret_cast<float4*>(smem_B);
        #pragma unroll 64
        for (int i = lane; i < 2048; i += 32) {
            cp_async16(smB_f4 + i, B_f4 + i);
        }
    }

    {
        const half* A_base = A + block_row * K;
        #pragma unroll 16
        for (int i = lane; i < 512; i += 32) {
            int row  = i >> 5;
            int col8 = (i & 31) * 8;
            cp_async16(smem_A + row * A_STRIDE + col8,
                       A_base + row * K + col8);
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        #pragma unroll
        for (int f = 0; f < 4; f++)
            acc[n][f] = 0.0f;

    #pragma unroll 4
    for (int k = 0; k < 16; k++) {
        uint32_t a_frag[4];
        {
            int ldm_row = lane % 16;
            int ldm_col = (lane / 16) * 8;
            uint32_t a_addr = smem_u32addr(smem_A + ldm_row * A_STRIDE + k * 16 + ldm_col);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(a_addr));
        }

        #pragma unroll
        for (int n8 = 0; n8 < 8; n8++) {
            uint32_t b_frag[2];
            {
                b_frag[0] = 0; b_frag[1] = 0;
            }
            (void)b_frag;
        }
    }

    (void)acc;
}

__global__ void __launch_bounds__(32, 64)
hgemm_wmma_opt(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ char smem_raw[];
    half* smem_B = reinterpret_cast<half*>(smem_raw);
    half* smem_A = reinterpret_cast<half*>(smem_raw + 32768);

    const int lane = threadIdx.x;
    const int block_row = blockIdx.x * 16;

    {
        const float4* B_f4 = reinterpret_cast<const float4*>(B);
        float4* smB_f4 = reinterpret_cast<float4*>(smem_B);
        #pragma unroll 64
        for (int i = lane; i < 2048; i += 32) {
            cp_async16(smB_f4 + i, B_f4 + i);
        }
    }

    {
        const half* A_base = A + block_row * K;
        #pragma unroll 16
        for (int i = lane; i < 512; i += 32) {
            int row  = i >> 5;
            int col8 = (i & 31) * 8;
            cp_async16(smem_A + row * A_STRIDE + col8,
                       A_base + row * K + col8);
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.0f);

    #pragma unroll 16
    for (int k = 0; k < 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A + k * 16, A_STRIDE);
        
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag0, b_frag1, b_frag2, b_frag3;
        wmma::load_matrix_sync(b_frag0, smem_B + k * 16 * 64 + 0 * 16, 64);
        wmma::load_matrix_sync(b_frag1, smem_B + k * 16 * 64 + 1 * 16, 64);
        wmma::load_matrix_sync(b_frag2, smem_B + k * 16 * 64 + 2 * 16, 64);
        wmma::load_matrix_sync(b_frag3, smem_B + k * 16 * 64 + 3 * 16, 64);
        
        wmma::mma_sync(acc[0], a_frag, b_frag0, acc[0]);
        wmma::mma_sync(acc[1], a_frag, b_frag1, acc[1]);
        wmma::mma_sync(acc[2], a_frag, b_frag2, acc[2]);
        wmma::mma_sync(acc[3], a_frag, b_frag3, acc[3]);
    }

    half* C_base = C + block_row * 64;
    #pragma unroll
    for (int n = 0; n < 4; n++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_half;
        const int ne = acc[n].num_elements;
        #pragma unroll
        for (int i = 0; i + 1 < ne; i += 2) {
            __half2 h2 = __float22half2_rn(make_float2(acc[n].x[i], acc[n].x[i+1]));
            acc_half.x[i]   = h2.x;
            acc_half.x[i+1] = h2.y;
        }
        if (ne & 1) acc_half.x[ne-1] = __float2half(acc[n].x[ne-1]);
        wmma::store_matrix_sync(C_base + n * 16, acc_half, 64, wmma::mem_row_major);
    }
}

__global__ void __launch_bounds__(64, 32)
hgemm_wmma_2warp(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ char smem_raw[];
    half* smem_B = reinterpret_cast<half*>(smem_raw);
    half* smem_A = reinterpret_cast<half*>(smem_raw + 32768);

    const int warp_id   = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int block_row = blockIdx.x * 32;
    const int warp_row  = warp_id * 16;

    {
        const float4* B_f4 = reinterpret_cast<const float4*>(B);
        float4* smB_f4 = reinterpret_cast<float4*>(smem_B);
        #pragma unroll 32
        for (int i = threadIdx.x; i < 2048; i += 64) {
            cp_async16(smB_f4 + i, B_f4 + i);
        }
    }

    {
        const half* A_base = A + block_row * K;
        #pragma unroll 8
        for (int i = threadIdx.x; i < 1024; i += 64) {
            int row  = i >> 5;
            int col8 = (i & 31) * 8;
            cp_async16(smem_A + row * A_STRIDE + col8,
                       A_base + row * K + col8);
        }
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.0f);

    #pragma unroll 16
    for (int k = 0; k < 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, smem_A + warp_row * A_STRIDE + k * 16, A_STRIDE);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag0, b_frag1, b_frag2, b_frag3;
        wmma::load_matrix_sync(b_frag0, smem_B + k * 16 * 64 + 0 * 16, 64);
        wmma::load_matrix_sync(b_frag1, smem_B + k * 16 * 64 + 1 * 16, 64);
        wmma::load_matrix_sync(b_frag2, smem_B + k * 16 * 64 + 2 * 16, 64);
        wmma::load_matrix_sync(b_frag3, smem_B + k * 16 * 64 + 3 * 16, 64);

        wmma::mma_sync(acc[0], a_frag, b_frag0, acc[0]);
        wmma::mma_sync(acc[1], a_frag, b_frag1, acc[1]);
        wmma::mma_sync(acc[2], a_frag, b_frag2, acc[2]);
        wmma::mma_sync(acc[3], a_frag, b_frag3, acc[3]);
    }

    half* C_base = C + (block_row + warp_row) * 64;
    #pragma unroll
    for (int n = 0; n < 4; n++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_half;
        const int ne = acc[n].num_elements;
        #pragma unroll
        for (int i = 0; i + 1 < ne; i += 2) {
            __half2 h2 = __float22half2_rn(make_float2(acc[n].x[i], acc[n].x[i+1]));
            acc_half.x[i]   = h2.x;
            acc_half.x[i+1] = h2.y;
        }
        if (ne & 1) acc_half.x[ne-1] = __float2half(acc[n].x[ne-1]);
        wmma::store_matrix_sync(C_base + n * 16, acc_half, 64, wmma::mem_row_major);
    }
}

__global__ void __launch_bounds__(32, 32)
hgemm_wmma_pipelined(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ char smem_raw[];
    half* smem_B  = reinterpret_cast<half*>(smem_raw);
    half* smem_A0 = reinterpret_cast<half*>(smem_raw + 32768);
    half* smem_A1 = smem_A0 + 16 * A_STRIDE;

    const int lane = threadIdx.x;
    const int block_row = blockIdx.x * 16;
    const half* A_base = A + block_row * K;

    {
        const float4* B_f4 = reinterpret_cast<const float4*>(B);
        float4* smB_f4 = reinterpret_cast<float4*>(smem_B);
        #pragma unroll 64
        for (int i = lane; i < 2048; i += 32) {
            cp_async16(smB_f4 + i, B_f4 + i);
        }
    }

    #pragma unroll 16
    for (int i = lane; i < 512; i += 32) {
        int row  = i >> 5;
        int col8 = (i & 31) * 8;
        cp_async16(smem_A0 + row * A_STRIDE + col8,
                   A_base + row * K + col8);
    }

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.0f);

    half* A_bufs[2] = {smem_A0, smem_A1};

    if (16 > 1) {
        #pragma unroll 16
        for (int i = lane; i < 512; i += 32) {
            int row  = i >> 5;
            int col8 = (i & 31) * 8;
            cp_async16(smem_A1 + row * A_STRIDE + col8,
                       A_base + row * K + 16 + col8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    #pragma unroll 1
    for (int k = 0; k < 16; k++) {
        if (k + 2 < 16) {
            half* next_buf = A_bufs[(k + 1) & 1];
            #pragma unroll 16
            for (int i = lane; i < 512; i += 32) {
                int row  = i >> 5;
                int col8 = (i & 31) * 8;
                cp_async16(next_buf + row * A_STRIDE + col8,
                           A_base + row * K + (k + 2) * 16 + col8);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        half* cur = A_bufs[k & 1];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, cur, A_STRIDE);

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag0, b_frag1, b_frag2, b_frag3;
        wmma::load_matrix_sync(b_frag0, smem_B + k * 16 * 64 + 0 * 16, 64);
        wmma::load_matrix_sync(b_frag1, smem_B + k * 16 * 64 + 1 * 16, 64);
        wmma::load_matrix_sync(b_frag2, smem_B + k * 16 * 64 + 2 * 16, 64);
        wmma::load_matrix_sync(b_frag3, smem_B + k * 16 * 64 + 3 * 16, 64);

        wmma::mma_sync(acc[0], a_frag, b_frag0, acc[0]);
        wmma::mma_sync(acc[1], a_frag, b_frag1, acc[1]);
        wmma::mma_sync(acc[2], a_frag, b_frag2, acc[2]);
        wmma::mma_sync(acc[3], a_frag, b_frag3, acc[3]);
    }

    asm volatile("cp.async.wait_all;\n" ::);

    half* C_base = C + block_row * 64;
    #pragma unroll
    for (int n = 0; n < 4; n++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_half;
        const int ne = acc[n].num_elements;
        #pragma unroll
        for (int i = 0; i + 1 < ne; i += 2) {
            __half2 h2 = __float22half2_rn(make_float2(acc[n].x[i], acc[n].x[i+1]));
            acc_half.x[i]   = h2.x;
            acc_half.x[i+1] = h2.y;
        }
        if (ne & 1) acc_half.x[ne-1] = __float2half(acc[n].x[ne-1]);
        wmma::store_matrix_sync(C_base + n * 16, acc_half, 64, wmma::mem_row_major);
    }
}

static bool s_attrs_set = false;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                 torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (!s_attrs_set) {
        cudaFuncSetAttribute(hgemm_wmma_opt,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 41216);
        cudaFuncSetAttribute(hgemm_wmma_opt,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        cudaFuncSetAttribute(hgemm_wmma_2warp,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 49664);
        cudaFuncSetAttribute(hgemm_wmma_2warp,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        cudaFuncSetAttribute(hgemm_wmma_pipelined,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 49664);
        cudaFuncSetAttribute(hgemm_wmma_pipelined,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        s_attrs_set = true;
    }

    hgemm_wmma_opt<<<M / 16, 32, 41216>>>(ptr_A, ptr_B, ptr_C, M, N, K);
}