#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define M_GLOBAL 128
#define N_GLOBAL 64
#define K_GLOBAL 512
#define NUM_CTAS 8
#define CTA_M 16
#define BLOCK_K 32
#define NUM_STAGES 4

#define SMEM_A_STRIDE 40
#define SMEM_B_STRIDE 72

static __device__ __forceinline__
void cp_async_ca16(uint32_t dst, const void* __restrict__ src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src) : "memory");
}
static __device__ __forceinline__
void cp_async_cg16(uint32_t dst, const void* __restrict__ src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src) : "memory");
}
static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int N>
static __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

static __device__ __forceinline__
void mma_m16n8k16(uint32_t const* A, uint32_t const* B, float* D) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3])
        : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),
          "r"(B[0]),"r"(B[1])
    );
}

static __device__ __forceinline__
void ldmatrix_x4(uint32_t* r, uint32_t ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(ptr)
    );
}

static __device__ __forceinline__
void ldmatrix_x2_trans(uint32_t* r, uint32_t ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
        : "=r"(r[0]),"=r"(r[1]) : "r"(ptr)
    );
}

__global__ void __launch_bounds__(32, 8)
hgemm_kernel_final(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C)
{
    __shared__ half smem_A[NUM_STAGES][CTA_M][SMEM_A_STRIDE];
    __shared__ half smem_B[NUM_STAGES][BLOCK_K][SMEM_B_STRIDE];

    const int lane = threadIdx.x;
    const int row_start = blockIdx.x * CTA_M;

    float acc[8][4];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    auto load_A = [&](int stage, int k_off) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int linear = lane * 2 + i;
            int r  = linear >> 2;
            int c8 = linear & 3;
            int c8_swz = c8 ^ (r & 3);
            uint32_t dst = __cvta_generic_to_shared(&smem_A[stage][r][c8_swz * 8]);
            cp_async_ca16(dst, &A[(row_start + r) * K_GLOBAL + k_off + c8 * 8]);
        }
    };

    auto load_B = [&](int stage, int k_off) __attribute__((always_inline)) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int linear = lane * 8 + i;
            int r = linear >> 3;
            int c = (linear & 7) * 8;
            uint32_t dst = __cvta_generic_to_shared(&smem_B[stage][r][c]);
            cp_async_cg16(dst, &B[(k_off + r) * N_GLOBAL + c]);
        }
    };

    const int num_k_tiles = K_GLOBAL / BLOCK_K;

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; s++) {
        load_A(s, s * BLOCK_K);
        load_B(s, s * BLOCK_K);
        cp_async_commit();
    }

    uint32_t frag_A[2][4];
    uint32_t frag_B[2][8][2];

    auto load_frag_A_kk = [&](int buf, int cur_stage, int kk) __attribute__((always_inline)) {
        int smem_row  = lane & 15;
        int c8_logical = (lane >> 4) + kk * 2;
        int c8_swz    = c8_logical ^ (smem_row & 3);
        uint32_t ptr  = __cvta_generic_to_shared(&smem_A[cur_stage][smem_row][c8_swz * 8]);
        ldmatrix_x4(frag_A[buf], ptr);
    };

    auto load_frag_B_kk = [&](int buf, int cur_stage, int kk) __attribute__((always_inline)) {
        int base_row = kk * 16 + (lane & 15);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t ptr = __cvta_generic_to_shared(&smem_B[cur_stage][base_row][j * 8]);
            ldmatrix_x2_trans(frag_B[buf][j], ptr);
        }
    };

    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int cur_stage  = k_tile % NUM_STAGES;
        const int load_tile  = k_tile + NUM_STAGES - 1;

        if (load_tile < num_k_tiles) {
            load_A(load_tile % NUM_STAGES, load_tile * BLOCK_K);
            load_B(load_tile % NUM_STAGES, load_tile * BLOCK_K);
        }
        cp_async_commit();
        cp_async_wait<NUM_STAGES - 1>();
        __syncwarp();

        load_frag_A_kk(0, cur_stage, 0);
        load_frag_B_kk(0, cur_stage, 0);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            mma_m16n8k16(frag_A[0], frag_B[0][j], acc[j]);
        }

        load_frag_A_kk(1, cur_stage, 1);
        load_frag_B_kk(1, cur_stage, 1);

        #pragma unroll
        for (int j = 4; j < 8; j++) {
            mma_m16n8k16(frag_A[0], frag_B[0][j], acc[j]);
        }

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            mma_m16n8k16(frag_A[1], frag_B[1][j], acc[j]);
        }
    }

    const int out_r0  = row_start + (lane >> 2);
    const int out_r1  = out_r0 + 8;
    const int out_c   = (lane & 3) * 2;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int c = j * 8 + out_c;
        __half2 h0 = __float22half2_rn(make_float2(acc[j][0], acc[j][1]));
        __half2 h1 = __float22half2_rn(make_float2(acc[j][2], acc[j][3]));
        *reinterpret_cast<__half2*>(&C[out_r0 * N_GLOBAL + c]) = h0;
        *reinterpret_cast<__half2*>(&C[out_r1 * N_GLOBAL + c]) = h1;
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr());

    hgemm_kernel_final<<<NUM_CTAS, 32>>>(ptr_A, ptr_B, ptr_C);
    cudaGetLastError();
}