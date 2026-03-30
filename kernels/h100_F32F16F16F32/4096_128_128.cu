#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <stdint.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

using namespace nvcuda::wmma;

static constexpr int B_STRIDE = 136;

__global__ void __launch_bounds__(64, 6)
hgemm_2w_smemAB(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    const int warp_id   = threadIdx.x >> 5;
    const int lane_id   = threadIdx.x & 31;
    const int block_row = blockIdx.x * 32;
    const int warp_row  = block_row + warp_id * 16;

    __shared__ __align__(128) half  smem_B[128 * B_STRIDE];
    __shared__ __align__(128) half  smem_A[32 * 128];
    __shared__ __align__(16)  float smem_out[2 * 256];

    {
        const char* Bsrc = reinterpret_cast<const char*>(B);
        for (int i = threadIdx.x; i < 128 * 16; i += 64) {
            int row   = i >> 4;
            int chunk = i & 15;
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                smem_B + row * B_STRIDE + chunk * 8);
            const char* src = Bsrc + (row * 128 + chunk * 8) * sizeof(half);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src));
        }
    }

    {
        int a_rows = min(32, M - block_row);
        const char* Asrc = reinterpret_cast<const char*>(A + block_row * 128);
        char* sAptr = reinterpret_cast<char*>(smem_A);
        int total_bytes = a_rows * 128 * (int)sizeof(half);
        for (int i = threadIdx.x * 16; i < total_bytes; i += 64 * 16) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"((unsigned)__cvta_generic_to_shared(sAptr + i)),
                            "l"(Asrc + i));
        }
        if (a_rows < 32) {
            for (int i = total_bytes + threadIdx.x * 2; i < 32*128*2; i += 64*2)
                *reinterpret_cast<half*>(sAptr + i) = __float2half(0.0f);
        }
    }

    asm volatile("cp.async.commit_group;\n" :::);
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    if (warp_row >= M) return;

    fragment<accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) fill_fragment(acc[ni], 0.0f);

    fragment<matrix_a, 16, 16, 16, half, row_major> frag_a[8];
    const half* smA_warp = smem_A + warp_id * 16 * 128;
    #pragma unroll
    for (int ki = 0; ki < 8; ki++) {
        load_matrix_sync(frag_a[ki], smA_warp + ki * 16, 128);
    }

    #pragma unroll
    for (int ki = 0; ki < 8; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
            load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
            mma_sync(acc[ni], frag_a[ki], frag_b, acc[ni]);
        }
    }

    float* warp_out = smem_out + warp_id * 256;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c_col = ni * 16;
        store_matrix_sync(warp_out, acc[ni], 16, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int t = lane_id; t < 128; t += 32) {
            int r   = (t * 2) >> 4;
            int col = (t * 2) & 15;
            int global_row = warp_row + r;
            if (global_row < M) {
                *reinterpret_cast<half2*>(&C[global_row * 128 + c_col + col]) =
                    __floats2half2_rn(warp_out[r * 16 + col], warp_out[r * 16 + col + 1]);
            }
        }
    }
}

__global__ void __launch_bounds__(64, 8)
hgemm_2w_nosmemA(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    const int warp_id   = threadIdx.x >> 5;
    const int lane_id   = threadIdx.x & 31;
    const int block_row = blockIdx.x * 32;
    const int warp_row  = block_row + warp_id * 16;

    __shared__ __align__(128) half  smem_B[128 * B_STRIDE];
    __shared__ __align__(16)  float smem_out[2 * 256];

    {
        const char* Bsrc = reinterpret_cast<const char*>(B);
        for (int i = threadIdx.x; i < 128 * 16; i += 64) {
            int row   = i >> 4;
            int chunk = i & 15;
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                smem_B + row * B_STRIDE + chunk * 8);
            const char* src = Bsrc + (row * 128 + chunk * 8) * sizeof(half);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;\n" :::);
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    if (warp_row >= M) return;

    fragment<accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) fill_fragment(acc[ni], 0.0f);

    if (warp_row + 16 <= M) {
        fragment<matrix_a, 16, 16, 16, half, row_major> frag_a[8];
        const half* A_warp = A + warp_row * 128;
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            load_matrix_sync(frag_a[ki], A_warp + ki * 16, 128);
        }
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
                load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
                mma_sync(acc[ni], frag_a[ki], frag_b, acc[ni]);
            }
        }
    } else {
        half* scratch = reinterpret_cast<half*>(smem_out) + warp_id * 256;
        int rows_avail = M - warp_row;
        for (int ki = 0; ki < 8; ki++) {
            if (lane_id < 16) {
                for (int r = 0; r < 16; r++) {
                    scratch[r * 16 + lane_id] = (r < rows_avail)
                        ? A[(warp_row + r) * 128 + ki * 16 + lane_id]
                        : __float2half(0.0f);
                }
            }
            __syncwarp();
            fragment<matrix_a, 16, 16, 16, half, row_major> fa;
            load_matrix_sync(fa, scratch, 16);
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
                load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
                mma_sync(acc[ni], fa, frag_b, acc[ni]);
            }
        }
    }

    float* warp_out = smem_out + warp_id * 256;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c_col = ni * 16;
        store_matrix_sync(warp_out, acc[ni], 16, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int t = lane_id; t < 128; t += 32) {
            int r   = (t * 2) >> 4;
            int col = (t * 2) & 15;
            int global_row = warp_row + r;
            if (global_row < M) {
                *reinterpret_cast<half2*>(&C[global_row * 128 + c_col + col]) =
                    __floats2half2_rn(warp_out[r * 16 + col], warp_out[r * 16 + col + 1]);
            }
        }
    }
}

__global__ void __launch_bounds__(32, 8)
hgemm_1w_nosmemA(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    const int lane_id = threadIdx.x;
    const int warp_row = blockIdx.x * 16;

    if (warp_row >= M) return;

    __shared__ __align__(128) half  smem_B[128 * B_STRIDE];
    __shared__ __align__(16)  float smem_out[256];

    for (int i = lane_id; i < 128 * 16; i += 32) {
        int row   = i >> 4;
        int chunk = i & 15;
        unsigned dst = (unsigned)__cvta_generic_to_shared(
            smem_B + row * B_STRIDE + chunk * 8);
        const char* src = reinterpret_cast<const char*>(B) + (row * 128 + chunk * 8) * sizeof(half);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst), "l"(src));
    }
    asm volatile("cp.async.commit_group;\n" :::);
    asm volatile("cp.async.wait_all;\n" :::);
    __syncwarp();

    fragment<accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) fill_fragment(acc[ni], 0.0f);

    if (warp_row + 16 <= M) {
        fragment<matrix_a, 16, 16, 16, half, row_major> frag_a[8];
        const half* A_warp = A + warp_row * 128;
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            load_matrix_sync(frag_a[ki], A_warp + ki * 16, 128);
        }
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
                load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
                mma_sync(acc[ni], frag_a[ki], frag_b, acc[ni]);
            }
        }
    } else {
        half* scratch = reinterpret_cast<half*>(smem_out);
        int rows_avail = M - warp_row;
        for (int ki = 0; ki < 8; ki++) {
            if (lane_id < 16) {
                for (int r = 0; r < 16; r++) {
                    scratch[r * 16 + lane_id] = (r < rows_avail)
                        ? A[(warp_row + r) * 128 + ki * 16 + lane_id]
                        : __float2half(0.0f);
                }
            }
            __syncwarp();
            fragment<matrix_a, 16, 16, 16, half, row_major> fa;
            load_matrix_sync(fa, scratch, 16);
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
                load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
                mma_sync(acc[ni], fa, frag_b, acc[ni]);
            }
        }
    }

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c_col = ni * 16;
        store_matrix_sync(smem_out, acc[ni], 16, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int t = lane_id; t < 128; t += 32) {
            int r   = (t * 2) >> 4;
            int col = (t * 2) & 15;
            int global_row = warp_row + r;
            if (global_row < M) {
                *reinterpret_cast<half2*>(&C[global_row * 128 + c_col + col]) =
                    __floats2half2_rn(smem_out[r * 16 + col], smem_out[r * 16 + col + 1]);
            }
        }
    }
}

__global__ void __launch_bounds__(128, 4)
hgemm_4w_smemB(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    const int warp_id   = threadIdx.x >> 5;
    const int lane_id   = threadIdx.x & 31;
    const int block_row = blockIdx.x * 64;
    const int warp_row  = block_row + warp_id * 16;

    __shared__ __align__(128) half  smem_B[128 * B_STRIDE];
    __shared__ __align__(16)  float smem_out[4 * 256];

    {
        const char* Bsrc = reinterpret_cast<const char*>(B);
        for (int i = threadIdx.x; i < 128 * 16; i += 128) {
            int row   = i >> 4;
            int chunk = i & 15;
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                smem_B + row * B_STRIDE + chunk * 8);
            const char* src = Bsrc + (row * 128 + chunk * 8) * sizeof(half);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src));
        }
    }
    asm volatile("cp.async.commit_group;\n" :::);
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    if (warp_row >= M) return;

    fragment<accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) fill_fragment(acc[ni], 0.0f);

    if (warp_row + 16 <= M) {
        fragment<matrix_a, 16, 16, 16, half, row_major> frag_a[8];
        const half* A_warp = A + warp_row * 128;
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            load_matrix_sync(frag_a[ki], A_warp + ki * 16, 128);
        }
        #pragma unroll
        for (int ki = 0; ki < 8; ki++) {
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
                load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
                mma_sync(acc[ni], frag_a[ki], frag_b, acc[ni]);
            }
        }
    } else {
        float* wout_tmp = smem_out + warp_id * 256;
        half* scratch = reinterpret_cast<half*>(wout_tmp);
        int rows_avail = M - warp_row;
        for (int ki = 0; ki < 8; ki++) {
            if (lane_id < 16) {
                for (int r = 0; r < 16; r++) {
                    scratch[r * 16 + lane_id] = (r < rows_avail)
                        ? A[(warp_row + r) * 128 + ki * 16 + lane_id]
                        : __float2half(0.0f);
                }
            }
            __syncwarp();
            fragment<matrix_a, 16, 16, 16, half, row_major> fa;
            load_matrix_sync(fa, scratch, 16);
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
                load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
                mma_sync(acc[ni], fa, frag_b, acc[ni]);
            }
        }
    }

    float* warp_out = smem_out + warp_id * 256;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c_col = ni * 16;
        store_matrix_sync(warp_out, acc[ni], 16, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int t = lane_id; t < 128; t += 32) {
            int r   = (t * 2) >> 4;
            int col = (t * 2) & 15;
            int global_row = warp_row + r;
            if (global_row < M) {
                *reinterpret_cast<half2*>(&C[global_row * 128 + c_col + col]) =
                    __floats2half2_rn(warp_out[r * 16 + col], warp_out[r * 16 + col + 1]);
            }
        }
    }
}

__global__ void __launch_bounds__(64, 6)
hgemm_persistent(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M,
    int* __restrict__ row_counter)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    __shared__ __align__(128) half  smem_B[128 * B_STRIDE];
    __shared__ __align__(128) half  smem_A[32 * 128];
    __shared__ __align__(16)  float smem_out[2 * 256];
    __shared__ int shared_block_row;

    {
        const char* Bsrc = reinterpret_cast<const char*>(B);
        for (int i = threadIdx.x; i < 128 * 16; i += 64) {
            int row   = i >> 4;
            int chunk = i & 15;
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                smem_B + row * B_STRIDE + chunk * 8);
            const char* src = Bsrc + (row * 128 + chunk * 8) * sizeof(half);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(src));
        }
        asm volatile("cp.async.commit_group;\n" :::);
        asm volatile("cp.async.wait_all;\n" :::);
        __syncthreads();
    }

    while (true) {
        if (threadIdx.x == 0) {
            shared_block_row = atomicAdd(row_counter, 32);
        }
        __syncthreads();
        int block_row = shared_block_row;
        if (block_row >= M) break;

        int warp_row = block_row + warp_id * 16;

        {
            int a_rows = min(32, M - block_row);
            const char* Asrc = reinterpret_cast<const char*>(A + block_row * 128);
            char* sAptr = reinterpret_cast<char*>(smem_A);
            int total_bytes = a_rows * 128 * (int)sizeof(half);
            for (int i = threadIdx.x * 16; i < total_bytes; i += 64 * 16) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"((unsigned)__cvta_generic_to_shared(sAptr + i)),
                                "l"(Asrc + i));
            }
            if (a_rows < 32) {
                for (int i = total_bytes + threadIdx.x * 2; i < 32*128*2; i += 64*2)
                    *reinterpret_cast<half*>(sAptr + i) = __float2half(0.0f);
            }
            asm volatile("cp.async.commit_group;\n" :::);
            asm volatile("cp.async.wait_all;\n" :::);
            __syncthreads();
        }

        if (warp_row < M) {
            fragment<accumulator, 16, 16, 16, float> acc[8];
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) fill_fragment(acc[ni], 0.0f);

            fragment<matrix_a, 16, 16, 16, half, row_major> frag_a[8];
            const half* smA_warp = smem_A + warp_id * 16 * 128;
            #pragma unroll
            for (int ki = 0; ki < 8; ki++) {
                load_matrix_sync(frag_a[ki], smA_warp + ki * 16, 128);
            }

            #pragma unroll
            for (int ki = 0; ki < 8; ki++) {
                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    fragment<matrix_b, 16, 16, 16, half, row_major> frag_b;
                    load_matrix_sync(frag_b, smem_B + ki * 16 * B_STRIDE + ni * 16, B_STRIDE);
                    mma_sync(acc[ni], frag_a[ki], frag_b, acc[ni]);
                }
            }

            float* warp_out = smem_out + warp_id * 256;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                const int c_col = ni * 16;
                store_matrix_sync(warp_out, acc[ni], 16, mem_row_major);
                __syncwarp();
                #pragma unroll
                for (int t = lane_id; t < 128; t += 32) {
                    int r   = (t * 2) >> 4;
                    int col = (t * 2) & 15;
                    int global_row = warp_row + r;
                    if (global_row < M) {
                        *reinterpret_cast<half2*>(&C[global_row * 128 + c_col + col]) =
                            __floats2half2_rn(warp_out[r * 16 + col], warp_out[r * 16 + col + 1]);
                    }
                }
            }
        }
        __syncthreads();
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr());

    {
        int grid = (M + 31) / 32;
        hgemm_2w_smemAB<<<grid, 64>>>(ptr_A, ptr_B, ptr_C, M);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        int grid = (M + 31) / 32;
        hgemm_2w_nosmemA<<<grid, 64>>>(ptr_A, ptr_B, ptr_C, M);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        int* row_counter;
        if (cudaMalloc(&row_counter, sizeof(int)) == cudaSuccess) {
            cudaMemset(row_counter, 0, sizeof(int));
            hgemm_persistent<<<132, 64>>>(ptr_A, ptr_B, ptr_C, M, row_counter);
            cudaError_t err = cudaGetLastError();
            cudaFree(row_counter);
            if (err == cudaSuccess) return;
            cudaGetLastError();
        }
    }

    {
        int grid = (M + 15) / 16;
        hgemm_1w_nosmemA<<<grid, 32>>>(ptr_A, ptr_B, ptr_C, M);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    {
        int grid = (M + 63) / 64;
        hgemm_4w_smemB<<<grid, 128>>>(ptr_A, ptr_B, ptr_C, M);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("All GEMM kernels failed: ") + cudaGetErrorString(err));
    }
}