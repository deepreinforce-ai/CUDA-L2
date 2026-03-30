#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace nvcuda;

namespace c10 {
namespace detail {
[[noreturn]] void torchInternalAssertFail(const char* file,
                                          const char* func,
                                          unsigned int line,
                                          const char* msg,
                                          const std::string& detail_msg) {
  std::fprintf(stderr,
               "torchInternalAssertFail: %s:%u (%s): %s | %s\n",
               file ? file : "unknown",
               line,
               func ? func : "unknown",
               msg ? msg : "no-msg",
               detail_msg.c_str());
  std::abort();
}

[[noreturn]] void torchInternalAssertFail(const char* file,
                                          const char* func,
                                          unsigned int line,
                                          const char* msg) {
  std::fprintf(stderr,
               "torchInternalAssertFail: %s:%u (%s): %s\n",
               file ? file : "unknown",
               line,
               func ? func : "unknown",
               msg ? msg : "no-msg");
  std::abort();
}
} // namespace detail
} // namespace c10

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t err__ = (x);                                                   \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err__));    \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace {

__global__ __launch_bounds__(256, 3)
void hgemm_64x128x64_fixed_bcol_colmajor_vec128_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C) {

  constexpr int M  = 512;
  constexpr int N  = 1024;
  constexpr int K  = 128;

  constexpr int BM = 64;
  constexpr int BN = 128;
  constexpr int BK = 64;

  constexpr int APAD = 8;
  constexpr int BPAD = 8;

  const int block_m = blockIdx.y;
  const int block_n = blockIdx.x;
  const int m0 = block_m * BM;
  const int n0 = block_n * BN;

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const int warp_m = warp >> 2;
  const int warp_n = warp & 3;

  __shared__ half  As[BM][BK + APAD];
  __shared__ half  Bs[BN][BK + BPAD];
  __shared__ float WarpEpi[8][16 * 16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
  }

#pragma unroll
  for (int k0 = 0; k0 < K; k0 += BK) {

    constexpr int A_VEC_PER_ROW = BK / 8;
    constexpr int A_NUM_VEC     = BM * A_VEC_PER_ROW;
    for (int idx = tid; idx < A_NUM_VEC; idx += 256) {
      int r  = idx / A_VEC_PER_ROW;
      int c8 = idx % A_VEC_PER_ROW;

      const uint4* gmem_ptr = reinterpret_cast<const uint4*>(
          &A[(m0 + r) * K + (k0 + c8 * 8)]);
      uint4* sptr = reinterpret_cast<uint4*>(
          &As[r][c8 * 8]);
      *sptr = *gmem_ptr;
    }

    constexpr int B_VEC_PER_ROW = BK / 8;
    constexpr int B_NUM_VEC     = BN * B_VEC_PER_ROW;
    for (int idx = tid; idx < B_NUM_VEC; idx += 256) {
      int n_local = idx / B_VEC_PER_ROW;
      int k8      = idx % B_VEC_PER_ROW;

      int gn = n0 + n_local;
      int gk = k0 + k8 * 8;

      const uint4* gmem_ptr = reinterpret_cast<const uint4*>(
          &B_col[gn * K + gk]);
      uint4* sptr = reinterpret_cast<uint4*>(
          &Bs[n_local][k8 * 8]);
      *sptr = *gmem_ptr;
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int a_row = warp_m * 32 + i * 16;
        const half* a_ptr = &As[a_row][kk];

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, a_ptr, BK + APAD);

#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int b_col = warp_n * 32 + j * 16;
          const half* b_ptr = &Bs[b_col][kk];

          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, b_ptr, BK + BPAD);

          wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
        }
      }
    }

    __syncthreads();
  }

  float* warp_buf = &WarpEpi[warp][0];

#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int row_base = warp_m * 32 + i * 16;
      int col_base = warp_n * 32 + j * 16;

      wmma::store_matrix_sync(warp_buf, c_frag[i][j], 16, wmma::mem_row_major);
      __syncwarp();

      for (int e2 = lane * 2; e2 < 256; e2 += 64) {
        int rr = e2 >> 4;
        int cc = e2 & 15;

        int gm = m0 + row_base + rr;
        int gn = n0 + col_base + cc;

        float2 fv = *reinterpret_cast<float2*>(&warp_buf[e2]);
        half2 hv = __float22half2_rn(fv);
        *reinterpret_cast<half2*>(&C[gm * N + gn]) = hv;
      }
      __syncwarp();
    }
  }
}

__global__ __launch_bounds__(256, 2)
void hgemm_64x128x16_generic_rowB_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K) {

  constexpr int BM = 64;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int APAD = 8;
  constexpr int BPAD = 8;

  int block_m = blockIdx.y;
  int block_n = blockIdx.x;
  int m0 = block_m * BM;
  int n0 = block_n * BN;
  if (m0 >= M || n0 >= N) return;

  int tid  = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  int warp_m = warp >> 2;
  int warp_n = warp & 3;

  __shared__ half  As[BM][BK + APAD];
  __shared__ half  Bs[BK][BN + BPAD];
  __shared__ float WarpEpi[8][16 * 16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += 256) {
      int r = idx / BK;
      int c = idx % BK;
      int gm = m0 + r;
      int gk = k0 + c;
      As[r][c] = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
    }

    for (int idx = tid; idx < BK * BN; idx += 256) {
      int r = idx / BN;
      int c = idx % BN;
      int gk = k0 + r;
      int gn = n0 + c;
      Bs[r][c] = (gk < K && gn < N) ? B[gk * N + gn] : __float2half(0.0f);
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int a_row = warp_m * 32 + i * 16;
      const half* a_ptr = &As[a_row][0];
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, a_ptr, BK + APAD);

#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int b_col = warp_n * 32 + j * 16;
        const half* b_ptr = &Bs[0][b_col];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, b_ptr, BN + BPAD);
        wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
      }
    }

    __syncthreads();
  }

  float* warp_buf = &WarpEpi[warp][0];

#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int row_base = warp_m * 32 + i * 16;
      int col_base = warp_n * 32 + j * 16;

      wmma::store_matrix_sync(warp_buf, c_frag[i][j], 16, wmma::mem_row_major);
      __syncwarp();

      for (int e = lane; e < 256; e += 32) {
        int rr = e >> 4;
        int cc = e & 15;
        int gm = m0 + row_base + rr;
        int gn = n0 + col_base + cc;
        if (gm < M && gn < N) C[gm * N + gn] = __float2half_rn(warp_buf[e]);
      }
      __syncwarp();
    }
  }
}

} // namespace

void cuda_l2_h100_fp32(torch::Tensor a,
                                torch::Tensor b,
                                torch::Tensor b_col_major,
                                torch::Tensor c) {
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));

  const half* A_ptr   = reinterpret_cast<const half*>(a.data_ptr());
  const half* B_ptr   = reinterpret_cast<const half*>(b.data_ptr());
  const half* Bc_ptr  = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* C_ptr         = reinterpret_cast<half*>(c.data_ptr());

  if (M == 512 && N == 1024 && K == 128) {
    dim3 block(256, 1, 1);
    dim3 grid(1024 / 128, 512 / 64, 1);
    hgemm_64x128x64_fixed_bcol_colmajor_vec128_kernel<<<grid, block>>>(A_ptr, Bc_ptr, C_ptr);
    CHECK_CUDA(cudaGetLastError());
    return;
  }

  dim3 block(256, 1, 1);
  dim3 grid((N + 127) / 128, (M + 63) / 64, 1);
  hgemm_64x128x16_generic_rowB_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);
  CHECK_CUDA(cudaGetLastError());
}