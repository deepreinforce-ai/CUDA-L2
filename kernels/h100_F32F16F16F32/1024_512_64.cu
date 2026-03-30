#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <stdint.h>

__device__ __forceinline__ void cp_async16_ca(void* dst, const void* src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
           "l"((uint64_t)src) : "memory");
}
__device__ __forceinline__ void cp_async16_cg(void* dst, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)),
           "l"((uint64_t)src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait2()  { asm volatile("cp.async.wait_group 2;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait1()  { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait0()  { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

__device__ __forceinline__ void store_cs_b32(half* ptr, float v0, float v1) {
    uint32_t val;
    asm volatile(
        "{\n"
        " .reg .b16 h0, h1;\n"
        " cvt.rn.f16.f32 h0, %1;\n"
        " cvt.rn.f16.f32 h1, %2;\n"
        " mov.b32 %0, {h0, h1};\n"
        "}\n"
        : "=r"(val) : "f"(v0), "f"(v1));
    asm volatile("st.global.cs.b32 [%0], %1;\n" :: "l"((uint64_t)ptr), "r"(val) : "memory");
}

__global__ void __launch_bounds__(128, 4)
hgemm_64x128_triple_prefetchB(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    constexpr int BM = 64, BN = 128, BK = 16;
    constexpr int K = 64, N = 512;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int gm_base = by * BM;
    const int gn_base = bx * BN;

    __shared__ half smem_A[3][BM][BK];
    __shared__ half smem_B[3][BK][BN];

    float acc[16][4];
    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
        acc[ni][0] = 0.f; acc[ni][1] = 0.f; acc[ni][2] = 0.f; acc[ni][3] = 0.f;
    }

    const int la_row      = tid >> 1;
    const int la_col_log  = (tid & 1) << 3;
    const int la_col_phys = la_col_log ^ ((la_row & 1) << 3);
    const int ga_row      = gm_base + la_row;

    const int lb_k0      = tid >> 4;
    const int lb_k1      = lb_k0 + 8;
    const int lb_ngroup  = tid & 15;
    const int lb_n_log   = lb_ngroup << 3;
    const int gn_b       = gn_base + lb_n_log;

    auto load_A_tile = [&](int buf, int kt) {
        const int gc = kt * BK + la_col_log;
        cp_async16_ca(&smem_A[buf][la_row][la_col_phys], A + (size_t)ga_row * K + gc);
    };
    auto load_B_tile = [&](int buf, int kt) {
        const int gk0 = kt * BK + lb_k0;
        const int gk1 = kt * BK + lb_k1;
        const int phys_n0 = (lb_ngroup ^ (lb_k0 & 7)) << 3;
        const int phys_n1 = (lb_ngroup ^ (lb_k1 & 7)) << 3;
        cp_async16_cg(&smem_B[buf][lb_k0][phys_n0], B + (size_t)gk0 * N + gn_b);
        cp_async16_cg(&smem_B[buf][lb_k1][phys_n1], B + (size_t)gk1 * N + gn_b);
    };

    auto do_mma_prefetchB = [&](int buf) {
        const int row_base = warp_id << 4;

        uint32_t a0, a1, a2, a3;
        {
            const int mat   = lane_id >> 3;
            const int r_in  = lane_id & 7;
            const int m_row = row_base + ((mat & 1) << 3) + r_in;
            const int k_log = (mat >> 1) << 3;
            const int k_phy = k_log ^ ((m_row & 1) << 3);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[buf][m_row][k_phy]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
        }

        auto load_b = [&](int ni, uint32_t &b0, uint32_t &b1) {
            const int k_row   = lane_id & 15;
            const int phys_ng = ni ^ (k_row & 7);
            const int phys_n  = phys_ng << 3;
            uint32_t addr = __cvta_generic_to_shared(&smem_B[buf][k_row][phys_n]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0), "=r"(b1) : "r"(addr));
        };

        uint32_t b0_cur, b1_cur, b0_nxt, b1_nxt;
        load_b(0, b0_cur, b1_cur);

        #pragma unroll
        for (int ni = 0; ni < 15; ++ni) {
            load_b(ni + 1, b0_nxt, b1_nxt);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]), "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cur), "r"(b1_cur));
            b0_cur = b0_nxt; b1_cur = b1_nxt;
        }

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[15][0]), "+f"(acc[15][1]), "+f"(acc[15][2]), "+f"(acc[15][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cur), "r"(b1_cur));
    };

    load_A_tile(0, 0); load_B_tile(0, 0); cp_async_commit();
    load_A_tile(1, 1); load_B_tile(1, 1); cp_async_commit();
    load_A_tile(2, 2); load_B_tile(2, 2); cp_async_commit();

    cp_async_wait2(); __syncthreads();
    do_mma_prefetchB(0);
    load_A_tile(0, 3); load_B_tile(0, 3); cp_async_commit();

    cp_async_wait2(); __syncthreads();
    do_mma_prefetchB(1);

    cp_async_wait1(); __syncthreads();
    do_mma_prefetchB(2);

    cp_async_wait0(); __syncthreads();
    do_mma_prefetchB(0);

    const int row_base = warp_id << 4;
    const int lr  = lane_id >> 2;
    const int lc2 = (lane_id & 3) << 1;

    const int r0 = gm_base + row_base + lr;
    const int r1 = r0 + 8;

    half* C_r0 = C + (size_t)r0 * N + gn_base;
    half* C_r1 = C + (size_t)r1 * N + gn_base;

    #pragma unroll
    for (int ni = 0; ni < 16; ++ni) {
        const int c_off = (ni << 3) + lc2;
        store_cs_b32(C_r0 + c_off, acc[ni][0], acc[ni][1]);
        store_cs_b32(C_r1 + c_off, acc[ni][2], acc[ni][3]);
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c) {
    const half* A = reinterpret_cast<const half*>(a.data_ptr());
    const half* B = reinterpret_cast<const half*>(b.data_ptr());
    half* C       = reinterpret_cast<half*>(c.data_ptr());

    dim3 grid(4, 16);
    dim3 block(128);
    hgemm_64x128_triple_prefetchB<<<grid, block>>>(A, B, C);
}