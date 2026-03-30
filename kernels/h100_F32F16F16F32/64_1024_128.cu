#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__global__ __launch_bounds__(32, 8)
void hgemm_128b(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N)
{
    const int lane     = threadIdx.x;
    const int cta_n    = blockIdx.x;
    const int cta_m    = blockIdx.y;
    const int base_row = cta_m * 16;
    const int base_col = cta_n * 32;

    __shared__ __align__(128) half smA[2][16][72];
    __shared__ __align__(128) half smB[2][64][40];

    const half* A_base = A + base_row * 128;
    const half* B_base = B + base_col;

    #pragma unroll
    for (int i = lane; i < 128; i += 32) {
        int r = i / 8; int c8 = i % 8;
        __pipeline_memcpy_async(&smA[0][r][c8*8], A_base + r*128 + c8*8, 16);
    }
    #pragma unroll
    for (int i = lane; i < 256; i += 32) {
        int k = (i*8)/32; int nc = (i*8)%32;
        __pipeline_memcpy_async(&smB[0][k][nc], B_base + k*N + nc, 16);
    }
    __pipeline_commit();

    #pragma unroll
    for (int i = lane; i < 128; i += 32) {
        int r = i / 8; int c8 = i % 8;
        __pipeline_memcpy_async(&smA[1][r][c8*8], A_base + r*128 + 64 + c8*8, 16);
    }
    #pragma unroll
    for (int i = lane; i < 256; i += 32) {
        int k = (i*8)/32; int nc = (i*8)%32;
        __pipeline_memcpy_async(&smB[1][k][nc], B_base + (k+64)*N + nc, 16);
    }
    __pipeline_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fB0, fB1;

    __pipeline_wait_prior(1);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k += 16) {
        wmma::load_matrix_sync(fA,  &smA[0][0][k],  72);
        wmma::load_matrix_sync(fB0, &smB[0][k][0],  40);
        wmma::load_matrix_sync(fB1, &smB[0][k][16], 40);
        wmma::mma_sync(acc0, fA, fB0, acc0);
        wmma::mma_sync(acc1, fA, fB1, acc1);
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k += 16) {
        wmma::load_matrix_sync(fA,  &smA[1][0][k],  72);
        wmma::load_matrix_sync(fB0, &smB[1][k][0],  40);
        wmma::load_matrix_sync(fB1, &smB[1][k][16], 40);
        wmma::mma_sync(acc0, fA, fB0, acc0);
        wmma::mma_sync(acc1, fA, fB1, acc1);
    }

    __syncthreads();

    float* smF = reinterpret_cast<float*>(smA);
    wmma::store_matrix_sync(&smF[0],  acc0, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(&smF[16], acc1, 32, wmma::mem_row_major);
    __syncthreads();

    half* C_out = C + base_row * N + base_col;
    #pragma unroll
    for (int idx = lane; idx < 256; idx += 32) {
        int elem = idx * 2;
        int r = elem / 32; int c = elem % 32;
        *reinterpret_cast<half2*>(&C_out[r * N + c]) =
            __floats2half2_rn(smF[r*32+c], smF[r*32+c+1]);
    }
}

__global__ __launch_bounds__(32, 6)
void hgemm_256b(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N)
{
    const int lane     = threadIdx.x;
    const int cta_n    = blockIdx.x;
    const int cta_m    = blockIdx.y;
    const int base_row = cta_m * 16;
    const int base_col = cta_n * 16;

    __shared__ __align__(128) half smA[2][16][72];
    __shared__ __align__(128) half smB[2][64][24];

    const half* A_base = A + base_row * 128;
    const half* B_base = B + base_col;

    #pragma unroll
    for (int i = lane; i < 128; i += 32) {
        int r = i/8; int c8 = i%8;
        __pipeline_memcpy_async(&smA[0][r][c8*8], A_base + r*128 + c8*8, 16);
    }
    #pragma unroll
    for (int i = lane; i < 128; i += 32) {
        int k = (i*8)/16; int nc = (i*8)%16;
        __pipeline_memcpy_async(&smB[0][k][nc], B_base + k*N + nc, 16);
    }
    __pipeline_commit();

    #pragma unroll
    for (int i = lane; i < 128; i += 32) {
        int r = i/8; int c8 = i%8;
        __pipeline_memcpy_async(&smA[1][r][c8*8], A_base + r*128 + 64 + c8*8, 16);
    }
    #pragma unroll
    for (int i = lane; i < 128; i += 32) {
        int k = (i*8)/16; int nc = (i*8)%16;
        __pipeline_memcpy_async(&smB[1][k][nc], B_base + (k+64)*N + nc, 16);
    }
    __pipeline_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fB;

    __pipeline_wait_prior(1);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k += 16) {
        wmma::load_matrix_sync(fA, &smA[0][0][k], 72);
        wmma::load_matrix_sync(fB, &smB[0][k][0], 24);
        wmma::mma_sync(acc, fA, fB, acc);
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k += 16) {
        wmma::load_matrix_sync(fA, &smA[1][0][k], 72);
        wmma::load_matrix_sync(fB, &smB[1][k][0], 24);
        wmma::mma_sync(acc, fA, fB, acc);
    }
    __syncthreads();

    float* smF = reinterpret_cast<float*>(smB);
    wmma::store_matrix_sync(smF, acc, 16, wmma::mem_row_major);
    __syncthreads();

    half* C_out = C + base_row * N + base_col;
    #pragma unroll
    for (int idx = lane; idx < 128; idx += 32) {
        int elem = idx * 2; int r = elem / 16; int c = elem % 16;
        *reinterpret_cast<half2*>(&C_out[r * N + c]) =
            __floats2half2_rn(smF[r*16+c], smF[r*16+c+1]);
    }
}

__global__ __launch_bounds__(128, 1)
void hgemm_large(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N)
{
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int cta_n    = blockIdx.x;
    const int base_col = cta_n * 128;
    const int warp_row = warp_id * 16;

    __shared__ __align__(128) half smA[64][128];
    __shared__ __align__(128) half smB[128][128];

    {
        const float4* A4 = reinterpret_cast<const float4*>(A);
        float4* sA4 = reinterpret_cast<float4*>(smA);
        #pragma unroll
        for (int i = tid; i < 1024; i += 128) sA4[i] = A4[i];
    }

    #pragma unroll
    for (int i = tid; i < 2048; i += 128) {
        int k  = (i*8)/128; int nc = (i*8)%128;
        *reinterpret_cast<float4*>(&smB[k][nc]) =
            *reinterpret_cast<const float4*>(&B[k * N + base_col + nc]);
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) wmma::fill_fragment(acc[ni], 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fB[8];

    #pragma unroll
    for (int k = 0; k < 128; k += 16) {
        wmma::load_matrix_sync(fA, &smA[warp_row][k], 128);
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            wmma::load_matrix_sync(fB[ni], &smB[k][ni*16], 128);
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            wmma::mma_sync(acc[ni], fA, fB[ni], acc[ni]);
    }
    __syncthreads();

    float* smF = reinterpret_cast<float*>(smB);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        wmma::store_matrix_sync(&smF[warp_row*128 + ni*16], acc[ni], 128, wmma::mem_row_major);
    __syncthreads();

    half* C_out = C + base_col;
    #pragma unroll
    for (int idx = tid; idx < 4096; idx += 128) {
        int elem = idx*2; int r = elem/128; int c = elem%128;
        *reinterpret_cast<half2*>(&C_out[r * N + c]) =
            __floats2half2_rn(smF[r*128+c], smF[r*128+c+1]);
    }
}

__global__ __launch_bounds__(64, 3)
void hgemm_64b(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int N)
{
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int cta_n    = blockIdx.x;
    const int cta_m    = blockIdx.y;
    const int base_row = cta_m * 16;
    const int base_col = cta_n * 64;

    __shared__ __align__(128) half smA[2][16][72];
    __shared__ __align__(128) half smB[2][64][72];

    const half* A_base = A + base_row * 128;
    const half* B_base = B + base_col;

    #pragma unroll
    for (int i = tid; i < 128; i += 64) {
        int r = i/8; int c8 = i%8;
        __pipeline_memcpy_async(&smA[0][r][c8*8], A_base + r*128 + c8*8, 16);
    }
    #pragma unroll
    for (int i = tid; i < 512; i += 64) {
        int k = (i*8)/64; int nc = (i*8)%64;
        __pipeline_memcpy_async(&smB[0][k][nc], B_base + k*N + nc, 16);
    }
    __pipeline_commit();

    #pragma unroll
    for (int i = tid; i < 128; i += 64) {
        int r = i/8; int c8 = i%8;
        __pipeline_memcpy_async(&smA[1][r][c8*8], A_base + r*128 + 64 + c8*8, 16);
    }
    #pragma unroll
    for (int i = tid; i < 512; i += 64) {
        int k = (i*8)/64; int nc = (i*8)%64;
        __pipeline_memcpy_async(&smB[1][k][nc], B_base + (k+64)*N + nc, 16);
    }
    __pipeline_commit();

    const int warp_n_base = warp_id * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fB[2];

    __pipeline_wait_prior(1);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k += 16) {
        wmma::load_matrix_sync(fA,    &smA[0][0][k],              72);
        wmma::load_matrix_sync(fB[0], &smB[0][k][warp_n_base],    72);
        wmma::load_matrix_sync(fB[1], &smB[0][k][warp_n_base+16], 72);
        wmma::mma_sync(acc[0], fA, fB[0], acc[0]);
        wmma::mma_sync(acc[1], fA, fB[1], acc[1]);
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; k += 16) {
        wmma::load_matrix_sync(fA,    &smA[1][0][k],              72);
        wmma::load_matrix_sync(fB[0], &smB[1][k][warp_n_base],    72);
        wmma::load_matrix_sync(fB[1], &smB[1][k][warp_n_base+16], 72);
        wmma::mma_sync(acc[0], fA, fB[0], acc[0]);
        wmma::mma_sync(acc[1], fA, fB[1], acc[1]);
    }
    __syncthreads();

    float* smF = reinterpret_cast<float*>(smB);
    wmma::store_matrix_sync(&smF[warp_n_base],      acc[0], 64, wmma::mem_row_major);
    wmma::store_matrix_sync(&smF[warp_n_base + 16], acc[1], 64, wmma::mem_row_major);
    __syncthreads();

    half* C_out = C + base_row * N + base_col;
    #pragma unroll
    for (int idx = tid; idx < 512; idx += 64) {
        int elem = idx * 2; int r = elem / 64; int c = elem % 64;
        *reinterpret_cast<half2*>(&C_out[r * N + c]) =
            __floats2half2_rn(smF[r*64+c], smF[r*64+c+1]);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    if (M == 64 && K == 128 && N == 1024) {
        dim3 grid(N / 16, M / 16);
        dim3 block(32);
        hgemm_256b<<<grid, block>>>(ptr_A, ptr_B, ptr_C, N);
    } else {
        dim3 grid((N + 31) / 32, (M + 15) / 16);
        dim3 block(32);
        hgemm_128b<<<grid, block>>>(ptr_A, ptr_B, ptr_C, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
    }
}