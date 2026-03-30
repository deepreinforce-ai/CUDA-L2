#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <stdlib.h>
#include <c10/core/TensorImpl.h>

using namespace nvcuda;

constexpr int HGEMM_M = 64;
constexpr int HGEMM_K = 64;
constexpr int HGEMM_N = 12288;

__global__ __launch_bounds__(256, 4)
void hgemm_64x64x12288_bm64_bn128_wmma_bcol_v3_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C)
{
  constexpr int M = HGEMM_M;
  constexpr int K = HGEMM_K;
  constexpr int N = HGEMM_N;

  constexpr int BN = 128;
  constexpr int WARPS = 8;
  constexpr int THREADS = WARPS * 32;
  constexpr int PAD_A = 8;

  __shared__ half  sA[M][K + PAD_A];
  __shared__ float sWarp[WARPS][256];

  const int tid     = threadIdx.x;
  const int lane    = tid & 31;
  const int warp_id = tid >> 5;

  const int tile_n0 = blockIdx.x * BN;
  const int col0    = tile_n0 + warp_id * 16;

  for (int idx = tid; idx < M * K; idx += THREADS) {
    int r = idx / K;
    int k = idx - r * K;
    sA[r][k] = A[r * K + k];
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;
  wmma::fill_fragment(c0, 0.0f);
  wmma::fill_fragment(c1, 0.0f);
  wmma::fill_fragment(c2, 0.0f);
  wmma::fill_fragment(c3, 0.0f);

  #pragma unroll
  for (int k0 = 0; k0 < K; k0 += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0, a1, a2, a3;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;

    const half* a_ptr0 = &sA[0][k0];
    const half* a_ptr1 = &sA[16][k0];
    const half* a_ptr2 = &sA[32][k0];
    const half* a_ptr3 = &sA[48][k0];
    const half* b_ptr  = &B_col[col0 * K + k0];

    wmma::load_matrix_sync(a0, a_ptr0, K + PAD_A);
    wmma::load_matrix_sync(a1, a_ptr1, K + PAD_A);
    wmma::load_matrix_sync(a2, a_ptr2, K + PAD_A);
    wmma::load_matrix_sync(a3, a_ptr3, K + PAD_A);
    wmma::load_matrix_sync(b,  b_ptr,  K);

    wmma::mma_sync(c0, a0, b, c0);
    wmma::mma_sync(c1, a1, b, c1);
    wmma::mma_sync(c2, a2, b, c2);
    wmma::mma_sync(c3, a3, b, c3);
  }

  float* warp_buf = &sWarp[warp_id][0];

  wmma::store_matrix_sync(warp_buf, c0, 16, wmma::mem_row_major);
  __syncwarp();
  #pragma unroll
  for (int p = lane; p < 128; p += 32) {
    int e0 = p << 1;
    int rr = p >> 3;
    int cc = (p & 7) << 1;
    int gr = rr;
    int gc = col0 + cc;
    half2 hv = __floats2half2_rn(warp_buf[e0], warp_buf[e0 + 1]);
    reinterpret_cast<half2*>(&C[gr * N + gc])[0] = hv;
  }

  wmma::store_matrix_sync(warp_buf, c1, 16, wmma::mem_row_major);
  __syncwarp();
  #pragma unroll
  for (int p = lane; p < 128; p += 32) {
    int e0 = p << 1;
    int rr = p >> 3;
    int cc = (p & 7) << 1;
    int gr = 16 + rr;
    int gc = col0 + cc;
    half2 hv = __floats2half2_rn(warp_buf[e0], warp_buf[e0 + 1]);
    reinterpret_cast<half2*>(&C[gr * N + gc])[0] = hv;
  }

  wmma::store_matrix_sync(warp_buf, c2, 16, wmma::mem_row_major);
  __syncwarp();
  #pragma unroll
  for (int p = lane; p < 128; p += 32) {
    int e0 = p << 1;
    int rr = p >> 3;
    int cc = (p & 7) << 1;
    int gr = 32 + rr;
    int gc = col0 + cc;
    half2 hv = __floats2half2_rn(warp_buf[e0], warp_buf[e0 + 1]);
    reinterpret_cast<half2*>(&C[gr * N + gc])[0] = hv;
  }

  wmma::store_matrix_sync(warp_buf, c3, 16, wmma::mem_row_major);
  __syncwarp();
  #pragma unroll
  for (int p = lane; p < 128; p += 32) {
    int e0 = p << 1;
    int rr = p >> 3;
    int cc = (p & 7) << 1;
    int gr = 48 + rr;
    int gc = col0 + cc;
    half2 hv = __floats2half2_rn(warp_buf[e0], warp_buf[e0 + 1]);
    reinterpret_cast<half2*>(&C[gr * N + gc])[0] = hv;
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
  c10::TensorImpl* a_impl  = a.unsafeGetTensorImpl();
  c10::TensorImpl* bc_impl = b_col_major.unsafeGetTensorImpl();
  c10::TensorImpl* c_impl  = c.unsafeGetTensorImpl();

  const half* A_ptr  = reinterpret_cast<const half*>(a_impl->data());
  const half* Bc_ptr = reinterpret_cast<const half*>(bc_impl->data());
  half* C_ptr        = reinterpret_cast<half*>(c_impl->mutable_data());

  dim3 block(256);
  dim3 grid(HGEMM_N / 128, 1);

  hgemm_64x64x12288_bm64_bn128_wmma_bcol_v3_kernel<<<grid, block>>>(A_ptr, Bc_ptr, C_ptr);

  (void)b;
}