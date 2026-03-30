#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using half = __half;

__device__ __forceinline__
void mma_m16n8k16(
    float &c0, float &c1, float &c2, float &c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0i, float c1i, float c2i, float c3i) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0i), "f"(c1i), "f"(c2i), "f"(c3i));
}

__device__ __forceinline__ void cp_async16_ca(void* dst, const void* src) {
  unsigned dst_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :: "r"(dst_addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async16_cg(void* dst, const void* src) {
  unsigned dst_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
               :: "r"(dst_addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t out;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(hi), "f"(lo));
  return out;
}

static constexpr int M_FIX = 64;
static constexpr int N_FIX = 1024;
static constexpr int K_FIX = 1024;

static constexpr int BM = 16;
static constexpr int BN = 64;
static constexpr int BK = 64;
static constexpr int STAGES = 4;
static constexpr int K_TILES = K_FIX / BK;
static constexpr int STAGE_MASK = STAGES - 1;

static constexpr int SA_STR = 72;
static constexpr int SB_STR = 72;
static constexpr int SA_SZ  = BM * SA_STR;
static constexpr int SB_SZ  = BN * SB_STR;

__device__ __forceinline__ void load_stage_g2s(
    half* __restrict__ sAw,
    half* __restrict__ sBw,
    const half* __restrict__ A_block,
    const half* __restrict__ B_block,
    int k_off,
    int tid) {

  {
    int chunk = tid;
    int row   = chunk >> 3;
    int col8  = chunk & 7;
    cp_async16_ca(
        sAw + row * SA_STR + col8 * 8,
        A_block + row * K_FIX + (k_off + col8 * 8));
  }

  #pragma unroll
  for (int li = 0; li < 4; ++li) {
    int chunk = tid * 4 + li;
    int dn    = chunk >> 3;
    int dk8   = chunk & 7;
    cp_async16_cg(
        sBw + dn * SB_STR + dk8 * 8,
        B_block + (size_t)dn * K_FIX + (k_off + dk8 * 8));
  }
}

__global__ __launch_bounds__(128, 3)
void hgemm_h100_m64n1024k1024_optimized_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half* __restrict__ C) {

  extern __shared__ half smem[];
  half* sA = smem;
  half* sB = smem + STAGES * SA_SZ;

  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane    = tid & 31;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  const half* A_block = A + (size_t)block_m * K_FIX;
  const half* B_block = B_col + (size_t)block_n * K_FIX;

  const int lane_row = lane >> 2;
  const int lane_col = (lane & 3) * 2;

  const int a_row0 = lane_row;
  const int a_row1 = lane_row + 8;
  const int wc     = warp_id;

  float acc[2][4];
  #pragma unroll
  for (int ni = 0; ni < 2; ++ni) {
    acc[ni][0] = 0.f; acc[ni][1] = 0.f; acc[ni][2] = 0.f; acc[ni][3] = 0.f;
  }

  #pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    half* sAw = sA + s * SA_SZ;
    half* sBw = sB + s * SB_SZ;
    load_stage_g2s(sAw, sBw, A_block, B_block, s * BK, tid);
    cp_async_commit();
  }

  int rs = 0;
  int ws = STAGES - 1;

  constexpr int STEADY = K_TILES - (STAGES - 1);

  #pragma unroll 1
  for (int iter = 0; iter < STEADY; ++iter) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    int next_k_off = (iter + (STAGES - 1)) * BK;
    load_stage_g2s(sA + ws * SA_SZ, sB + ws * SB_SZ, A_block, B_block, next_k_off, tid);
    cp_async_commit();

    const half* sAr = sA + rs * SA_SZ;
    const half* sBr = sB + rs * SB_SZ;

    uint32_t ra[4][4];
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp) {
      int col0 = kp * 16 + lane_col;
      int col8 = col0 + 8;
      ra[kp][0] = *reinterpret_cast<const uint32_t*>(sAr + a_row0 * SA_STR + col0);
      ra[kp][1] = *reinterpret_cast<const uint32_t*>(sAr + a_row1 * SA_STR + col0);
      ra[kp][2] = *reinterpret_cast<const uint32_t*>(sAr + a_row0 * SA_STR + col8);
      ra[kp][3] = *reinterpret_cast<const uint32_t*>(sAr + a_row1 * SA_STR + col8);
    }

    uint32_t rb_cur[2][2];
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
      int dn  = wc * 16 + ni * 8 + lane_row;
      int dk0 = lane_col;
      int dk8 = dk0 + 8;
      rb_cur[ni][0] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk0);
      rb_cur[ni][1] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk8);
    }

    #pragma unroll
    for (int kp = 0; kp < 4; ++kp) {
      uint32_t rb_next[2][2];

      if (kp < 3) {
        int dk0n = (kp + 1) * 16 + lane_col;
        int dk8n = dk0n + 8;
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
          int dn = wc * 16 + ni * 8 + lane_row;
          rb_next[ni][0] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk0n);
          rb_next[ni][1] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk8n);
        }
      }

      #pragma unroll
      for (int ni = 0; ni < 2; ++ni) {
        mma_m16n8k16(
            acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
            ra[kp][0], ra[kp][1], ra[kp][2], ra[kp][3],
            rb_cur[ni][0], rb_cur[ni][1],
            acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
      }

      if (kp < 3) {
        rb_cur[0][0] = rb_next[0][0]; rb_cur[0][1] = rb_next[0][1];
        rb_cur[1][0] = rb_next[1][0]; rb_cur[1][1] = rb_next[1][1];
      }
    }

    rs = (rs + 1) & STAGE_MASK;
    ws = (ws + 1) & STAGE_MASK;
  }

  #pragma unroll 1
  for (int iter = STEADY; iter < K_TILES; ++iter) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    const half* sAr = sA + rs * SA_SZ;
    const half* sBr = sB + rs * SB_SZ;

    uint32_t ra[4][4];
    #pragma unroll
    for (int kp = 0; kp < 4; ++kp) {
      int col0 = kp * 16 + lane_col;
      int col8 = col0 + 8;
      ra[kp][0] = *reinterpret_cast<const uint32_t*>(sAr + a_row0 * SA_STR + col0);
      ra[kp][1] = *reinterpret_cast<const uint32_t*>(sAr + a_row1 * SA_STR + col0);
      ra[kp][2] = *reinterpret_cast<const uint32_t*>(sAr + a_row0 * SA_STR + col8);
      ra[kp][3] = *reinterpret_cast<const uint32_t*>(sAr + a_row1 * SA_STR + col8);
    }

    uint32_t rb_cur[2][2];
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
      int dn  = wc * 16 + ni * 8 + lane_row;
      int dk0 = lane_col;
      int dk8 = dk0 + 8;
      rb_cur[ni][0] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk0);
      rb_cur[ni][1] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk8);
    }

    #pragma unroll
    for (int kp = 0; kp < 4; ++kp) {
      uint32_t rb_next[2][2];

      if (kp < 3) {
        int dk0n = (kp + 1) * 16 + lane_col;
        int dk8n = dk0n + 8;
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
          int dn = wc * 16 + ni * 8 + lane_row;
          rb_next[ni][0] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk0n);
          rb_next[ni][1] = *reinterpret_cast<const uint32_t*>(sBr + dn * SB_STR + dk8n);
        }
      }

      #pragma unroll
      for (int ni = 0; ni < 2; ++ni) {
        mma_m16n8k16(
            acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
            ra[kp][0], ra[kp][1], ra[kp][2], ra[kp][3],
            rb_cur[ni][0], rb_cur[ni][1],
            acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
      }

      if (kp < 3) {
        rb_cur[0][0] = rb_next[0][0]; rb_cur[0][1] = rb_next[0][1];
        rb_cur[1][0] = rb_next[1][0]; rb_cur[1][1] = rb_next[1][1];
      }
    }

    rs = (rs + 1) & STAGE_MASK;
    ws = (ws + 1) & STAGE_MASK;
  }

  cp_async_wait_all();
  __syncthreads();

  int out_row0 = block_m + a_row0;
  int out_row1 = block_m + a_row1;

  #pragma unroll
  for (int ni = 0; ni < 2; ++ni) {
    int col = block_n + wc * 16 + ni * 8 + lane_col;
    *reinterpret_cast<uint32_t*>(C + (size_t)out_row0 * N_FIX + col) = pack_f16x2(acc[ni][0], acc[ni][1]);
    *reinterpret_cast<uint32_t*>(C + (size_t)out_row1 * N_FIX + col) = pack_f16x2(acc[ni][2], acc[ni][3]);
  }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c)
{
  (void)b;

  const half* A     = reinterpret_cast<const half*>(a.data_ptr());
  const half* B_col = reinterpret_cast<const half*>(b_col_major.data_ptr());
  half* C           = reinterpret_cast<half*>(c.data_ptr());

  static bool configured = false;
  const int smem_bytes = STAGES * (SA_SZ + SB_SZ) * (int)sizeof(half);

  if (!configured) {
    cudaFuncSetAttribute(hgemm_h100_m64n1024k1024_optimized_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    cudaFuncSetAttribute(hgemm_h100_m64n1024k1024_optimized_kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    configured = true;
  }

  dim3 block(128);
  dim3 grid(N_FIX / BN, M_FIX / BM);
  hgemm_h100_m64n1024k1024_optimized_kernel<<<grid, block, smem_bytes>>>(A, B_col, C);
}