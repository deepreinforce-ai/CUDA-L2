#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define M_DIM    64
#define N_DIM    128
#define K_DIM    4096

#define TILE_K    16
#define NUM_K_TILES (K_DIM / TILE_K)

#define PAD_A  8
#define PAD_B  8
#define SMEM_A_STRIDE (M_DIM + PAD_A)
#define SMEM_B_STRIDE (N_DIM + PAD_B)

#define NUM_STAGES 4

__device__ __forceinline__ void cp_async16(void* __restrict__ dst, const void* __restrict__ src) {
    uint32_t dst32 = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(dst32), "l"((unsigned long long)src)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t pack2(half a, half b) {
    uint32_t r;
    asm volatile("mov.b32 %0, {%1, %2};\n"
        : "=r"(r) : "h"(__half_as_ushort(a)), "h"(__half_as_ushort(b)));
    return r;
}

__global__ __launch_bounds__(128, 2)
void hgemm_single_block(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half*       __restrict__ C
) {
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id * 16;

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    __shared__ __align__(128) half smem_A[NUM_STAGES][TILE_K][SMEM_A_STRIDE];
    __shared__ __align__(128) half smem_B[NUM_STAGES][TILE_K][SMEM_B_STRIDE];

    const int a_row0   = warp_m + (lane_id >> 2);
    const int a_row1   = a_row0 + 8;
    const int a_k0     = (lane_id & 3) * 2;
    const int a_k8     = a_k0 + 8;
    const int b_n_base = lane_id >> 2;
    const int b_k0     = (lane_id & 3) * 2;
    const int b_k8     = b_k0 + 8;

    auto load_A_tile = [&](int stage, int k_global_base) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx     = tid * 8 + i;
            int k_local = idx >> 6;
            int m       = idx & 63;
            smem_A[stage][k_local][m] = __ldg(&A[(size_t)m * K_DIM + k_global_base + k_local]);
        }
    };

    auto load_B_tile = [&](int stage, int k_global_base) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx     = tid * 2 + i;
            int k_local = idx >> 4;
            int n_off   = (idx & 15) * 8;
            cp_async16(&smem_A[stage][0][0] + (NUM_STAGES * TILE_K * SMEM_A_STRIDE)
                       + stage * TILE_K * SMEM_B_STRIDE + k_local * SMEM_B_STRIDE + n_off,
                       &B[(size_t)(k_global_base + k_local) * N_DIM + n_off]);
        }
    };

    auto load_B_tile2 = [&](int stage, int k_global_base) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx     = tid * 2 + i;
            int k_local = idx >> 4;
            int n_off   = (idx & 15) * 8;
            cp_async16(&smem_B[stage][k_local][n_off],
                       &B[(size_t)(k_global_base + k_local) * N_DIM + n_off]);
        }
    };

    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        load_A_tile(s, s * TILE_K);
        load_B_tile2(s, s * TILE_K);
        cp_async_commit();
    }

    #pragma unroll 4
    for (int tile = 0; tile < NUM_K_TILES; tile++) {
        int stage = tile & (NUM_STAGES - 1);

        cp_async_wait<NUM_STAGES - 1>();
        __syncthreads();

        int next_tile = tile + NUM_STAGES;
        if (next_tile < NUM_K_TILES) {
            int next_stage = next_tile & (NUM_STAGES - 1);
            load_A_tile(next_stage, next_tile * TILE_K);
            load_B_tile2(next_stage, next_tile * TILE_K);
            cp_async_commit();
        }

        uint32_t a_frag[4];
        a_frag[0] = pack2(smem_A[stage][a_k0  ][a_row0], smem_A[stage][a_k0+1][a_row0]);
        a_frag[1] = pack2(smem_A[stage][a_k0  ][a_row1], smem_A[stage][a_k0+1][a_row1]);
        a_frag[2] = pack2(smem_A[stage][a_k8  ][a_row0], smem_A[stage][a_k8+1][a_row0]);
        a_frag[3] = pack2(smem_A[stage][a_k8  ][a_row1], smem_A[stage][a_k8+1][a_row1]);

        uint32_t b0[16], b1[16];
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int b_n = nt * 8 + b_n_base;
            b0[nt] = pack2(smem_B[stage][b_k0][b_n], smem_B[stage][b_k0+1][b_n]);
            b1[nt] = pack2(smem_B[stage][b_k8][b_n], smem_B[stage][b_k8+1][b_n]);
        }

        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(acc[nt][0]),"+f"(acc[nt][1]),"+f"(acc[nt][2]),"+f"(acc[nt][3])
                : "r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
                  "r"(b0[nt]),"r"(b1[nt])
            );
        }
    }

    const int c_row0 = warp_m + (lane_id >> 2);
    const int c_row1 = c_row0 + 8;

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        int c_col = nt * 8 + (lane_id & 3) * 2;
        half2 h01 = __float22half2_rn(make_float2(acc[nt][0], acc[nt][1]));
        half2 h23 = __float22half2_rn(make_float2(acc[nt][2], acc[nt][3]));
        *reinterpret_cast<half2*>(&C[c_row0 * N_DIM + c_col]) = h01;
        *reinterpret_cast<half2*>(&C[c_row1 * N_DIM + c_col]) = h23;
    }
}

#define SPLIT_K   128
#define BLOCK_K   (K_DIM / SPLIT_K)
#define SK_TILE_K 16
#define SK_STAGES 2
#define SK_NUM_TILES (BLOCK_K / SK_TILE_K)

__global__ __launch_bounds__(128, 8)
void hgemm_splitk_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C_acc
) {
    const int split   = blockIdx.z;
    const int k_base  = split * BLOCK_K;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_m  = warp_id * 16;

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    __shared__ __align__(128) half smem_A[SK_STAGES][SK_TILE_K][SMEM_A_STRIDE];
    __shared__ __align__(128) half smem_B[SK_STAGES][SK_TILE_K][SMEM_B_STRIDE];

    const int a_row0   = warp_m + (lane_id >> 2);
    const int a_row1   = a_row0 + 8;
    const int a_k0     = (lane_id & 3) * 2;
    const int a_k8     = a_k0 + 8;
    const int b_n_base = lane_id >> 2;
    const int b_k0     = (lane_id & 3) * 2;
    const int b_k8     = b_k0 + 8;

    auto load_A_sk = [&](int stage, int k_off) {
        int kgb = k_base + k_off;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx     = tid * 8 + i;
            int k_local = idx >> 6;
            int m       = idx & 63;
            smem_A[stage][k_local][m] = __ldg(&A[(size_t)m * K_DIM + kgb + k_local]);
        }
    };

    auto load_B_sk = [&](int stage, int k_off) {
        int kgb = k_base + k_off;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx     = tid * 2 + i;
            int k_local = idx >> 4;
            int n_off   = (idx & 15) * 8;
            uint32_t dst32 = __cvta_generic_to_shared(&smem_B[stage][k_local][n_off]);
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst32), "l"((unsigned long long)(&B[(size_t)(kgb + k_local) * N_DIM + n_off]))
                : "memory"
            );
        }
    };

    load_A_sk(0, 0);
    load_B_sk(0, 0);
    asm volatile("cp.async.commit_group;\n" :: : "memory");

    load_A_sk(1, SK_TILE_K);
    load_B_sk(1, SK_TILE_K);
    asm volatile("cp.async.commit_group;\n" :: : "memory");

    asm volatile("cp.async.wait_group 1;\n" :: : "memory");
    __syncthreads();

    {
        uint32_t a_frag[4];
        a_frag[0] = pack2(smem_A[0][a_k0  ][a_row0], smem_A[0][a_k0+1][a_row0]);
        a_frag[1] = pack2(smem_A[0][a_k0  ][a_row1], smem_A[0][a_k0+1][a_row1]);
        a_frag[2] = pack2(smem_A[0][a_k8  ][a_row0], smem_A[0][a_k8+1][a_row0]);
        a_frag[3] = pack2(smem_A[0][a_k8  ][a_row1], smem_A[0][a_k8+1][a_row1]);
        uint32_t b0[16], b1[16];
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int b_n = nt * 8 + b_n_base;
            b0[nt] = pack2(smem_B[0][b_k0][b_n], smem_B[0][b_k0+1][b_n]);
            b1[nt] = pack2(smem_B[0][b_k8][b_n], smem_B[0][b_k8+1][b_n]);
        }
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(acc[nt][0]),"+f"(acc[nt][1]),"+f"(acc[nt][2]),"+f"(acc[nt][3])
                : "r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
                  "r"(b0[nt]),"r"(b1[nt])
            );
        }
    }

    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
    __syncthreads();

    {
        uint32_t a_frag[4];
        a_frag[0] = pack2(smem_A[1][a_k0  ][a_row0], smem_A[1][a_k0+1][a_row0]);
        a_frag[1] = pack2(smem_A[1][a_k0  ][a_row1], smem_A[1][a_k0+1][a_row1]);
        a_frag[2] = pack2(smem_A[1][a_k8  ][a_row0], smem_A[1][a_k8+1][a_row0]);
        a_frag[3] = pack2(smem_A[1][a_k8  ][a_row1], smem_A[1][a_k8+1][a_row1]);
        uint32_t b0[16], b1[16];
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int b_n = nt * 8 + b_n_base;
            b0[nt] = pack2(smem_B[1][b_k0][b_n], smem_B[1][b_k0+1][b_n]);
            b1[nt] = pack2(smem_B[1][b_k8][b_n], smem_B[1][b_k8+1][b_n]);
        }
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(acc[nt][0]),"+f"(acc[nt][1]),"+f"(acc[nt][2]),"+f"(acc[nt][3])
                : "r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
                  "r"(b0[nt]),"r"(b1[nt])
            );
        }
    }

    float* C_slice = C_acc + (size_t)split * M_DIM * N_DIM;
    const int c_row0 = warp_m + (lane_id >> 2);
    const int c_row1 = c_row0 + 8;

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        int c_col = nt * 8 + (lane_id & 3) * 2;
        *reinterpret_cast<float2*>(&C_slice[c_row0 * N_DIM + c_col]) =
            make_float2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<float2*>(&C_slice[c_row1 * N_DIM + c_col]) =
            make_float2(acc[nt][2], acc[nt][3]);
    }
}

__global__ __launch_bounds__(128)
void reduce_splitk_v2(
    const float* __restrict__ C_acc,
    half*        __restrict__ C,
    int MN
) {
    const int base = blockIdx.x * 8;
    const int sk   = threadIdx.x;

    if (base + 7 >= MN) return;

    const float* src = C_acc + (size_t)sk * MN + base;
    float4 lo = *reinterpret_cast<const float4*>(src);
    float4 hi = *reinterpret_cast<const float4*>(src + 4);

    float v0=lo.x, v1=lo.y, v2=lo.z, v3=lo.w;
    float v4=hi.x, v5=hi.y, v6=hi.z, v7=hi.w;

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        v0 += __shfl_down_sync(0xffffffff, v0, off);
        v1 += __shfl_down_sync(0xffffffff, v1, off);
        v2 += __shfl_down_sync(0xffffffff, v2, off);
        v3 += __shfl_down_sync(0xffffffff, v3, off);
        v4 += __shfl_down_sync(0xffffffff, v4, off);
        v5 += __shfl_down_sync(0xffffffff, v5, off);
        v6 += __shfl_down_sync(0xffffffff, v6, off);
        v7 += __shfl_down_sync(0xffffffff, v7, off);
    }

    __shared__ float smem[4][8];
    if ((sk & 31) == 0) {
        int w = sk >> 5;
        smem[w][0]=v0; smem[w][1]=v1; smem[w][2]=v2; smem[w][3]=v3;
        smem[w][4]=v4; smem[w][5]=v5; smem[w][6]=v6; smem[w][7]=v7;
    }
    __syncthreads();

    if (sk == 0) {
        #pragma unroll
        for (int d = 0; d < 8; d += 2) {
            float s0 = smem[0][d]   + smem[1][d]   + smem[2][d]   + smem[3][d];
            float s1 = smem[0][d+1] + smem[1][d+1] + smem[2][d+1] + smem[3][d+1];
            *reinterpret_cast<half2*>(&C[base + d]) =
                __float22half2_rn(make_float2(s0, s1));
        }
    }
}

static float* g_C_acc       = nullptr;
static size_t g_C_acc_bytes = 0;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<torch::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<torch::Half>());
    half*       ptr_C = reinterpret_cast<half*>(c.data_ptr<torch::Half>());

    const int MN = M_DIM * N_DIM;

    size_t needed = (size_t)SPLIT_K * MN * sizeof(float);
    if (g_C_acc_bytes < needed) {
        if (g_C_acc) cudaFree(g_C_acc);
        cudaMalloc(&g_C_acc, needed);
        g_C_acc_bytes = needed;
    }

    hgemm_splitk_kernel<<<dim3(1, 1, SPLIT_K), 128>>>(ptr_A, ptr_B, g_C_acc);
    reduce_splitk_v2<<<MN / 8, SPLIT_K>>>(g_C_acc, ptr_C, MN);
}