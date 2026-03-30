#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

static constexpr int N_FIXED = 128;
static constexpr int K_FIXED = 64;
static constexpr int MMA_M   = 16;
static constexpr int MMA_N   = 8;
static constexpr int MMA_K   = 16;
static constexpr int K_TILES = K_FIXED / MMA_K;
static constexpr int N_TILES = N_FIXED / MMA_N;
static constexpr int SA_STR  = K_FIXED + 8;
static constexpr int SB_STR  = N_FIXED + 8;

__device__ __forceinline__ uint32_t smem_u32addr(const void* p) {
    uint32_t addr;
    asm volatile(
        "{.reg .u64 u64; cvta.to.shared.u64 u64, %1; cvt.u32.u64 %0, u64;}"
        : "=r"(addr) : "l"(p));
    return addr;
}

__device__ __forceinline__ void store_cs_b32(half* ptr, uint32_t val) {
    asm volatile("st.global.cs.b32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}

__global__ void __launch_bounds__(32, 4)
hgemm_persistent_B_in_regs(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int m_tiles)
{
    __shared__ __align__(128) half smA[MMA_M * SA_STR];
    __shared__ __align__(128) half smB_stage[K_FIXED * SB_STR];

    const int lane      = threadIdx.x;
    const int sm_id     = blockIdx.x;

    #pragma unroll 16
    for (int i = lane; i < K_FIXED * (N_FIXED / 8); i += 32) {
        const int r  = i >> 4;
        const int c8 = i & 15;
        uint32_t dst = smem_u32addr(smB_stage + r * SB_STR + c8 * 8);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst), "l"((const void*)(B + r * N_FIXED + c8 * 8)) : "memory");
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncwarp();

    uint32_t rb[K_TILES][N_TILES][2];
    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        const int row_b = ki * MMA_K + (lane & 15);
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++) {
            uint32_t addr = smem_u32addr(smB_stage + row_b * SB_STR + ni * MMA_N);
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];"
                : "=r"(rb[ki][ni][0]), "=r"(rb[ki][ni][1])
                : "r"(addr));
        }
    }

    for (int m_tile = sm_id; m_tile < m_tiles; m_tile += gridDim.x) {
        const int row_base = m_tile * MMA_M;
        const int valid    = min(MMA_M, M - row_base);

        #pragma unroll 2
        for (int i = lane; i < MMA_M * (K_FIXED / 8); i += 32) {
            const int r  = i >> 3;
            const int c8 = i & 7;
            uint32_t dst = smem_u32addr(smA + r * SA_STR + c8 * 8);
            if (r < valid) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"((const void*)(A + (row_base + r) * K_FIXED + c8 * 8)) : "memory");
            } else {
                asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                             :: "r"(dst), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncwarp();

        uint32_t ra[K_TILES][4];
        #pragma unroll
        for (int ki = 0; ki < K_TILES; ki++) {
            const int r = lane & 15;
            const int c = ki * MMA_K + ((lane >> 4) << 3);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(ra[ki][0]), "=r"(ra[ki][1]), "=r"(ra[ki][2]), "=r"(ra[ki][3])
                : "r"(smem_u32addr(smA + r * SA_STR + c)));
        }

        float acc[N_TILES][4];
        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++) {
            acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;
        }

        #pragma unroll
        for (int ki = 0; ki < K_TILES; ki++) {
            #pragma unroll
            for (int ni = 0; ni < N_TILES; ni++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                      "+f"(acc[ni][2]), "+f"(acc[ni][3])
                    : "r"(ra[ki][0]), "r"(ra[ki][1]), "r"(ra[ki][2]), "r"(ra[ki][3]),
                      "r"(rb[ki][ni][0]), "r"(rb[ki][ni][1]));
            }
        }

        const int or0 = row_base + (lane >> 2);
        const int or1 = or0 + 8;
        const int oc  = (lane & 3) << 1;

        #pragma unroll
        for (int ni = 0; ni < N_TILES; ni++) {
            half2 v0 = __floats2half2_rn(acc[ni][0], acc[ni][1]);
            half2 v1 = __floats2half2_rn(acc[ni][2], acc[ni][3]);
            if (or0 < M) store_cs_b32(C + or0 * N_FIXED + ni * MMA_N + oc,
                                      *reinterpret_cast<uint32_t*>(&v0));
            if (or1 < M) store_cs_b32(C + or1 * N_FIXED + ni * MMA_N + oc,
                                      *reinterpret_cast<uint32_t*>(&v1));
        }
    }
}

static constexpr int SB_STRIP = MMA_N + 8;

__global__ void __launch_bounds__(32, 52)
hgemm_1024blocks_nsplit(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    __shared__ __align__(128) half smA2[MMA_M * SA_STR];
    __shared__ __align__(128) half smB2[K_FIXED * SB_STRIP];

    const int lane     = threadIdx.x;
    const int m_tile   = blockIdx.x;
    const int n_tile   = blockIdx.y;
    const int row_base = m_tile * MMA_M;
    const int col_base = n_tile * MMA_N;

    if (row_base >= M) return;
    const int valid = min(MMA_M, M - row_base);

    #pragma unroll 2
    for (int k = lane; k < K_FIXED; k += 32) {
        uint32_t dst = smem_u32addr(smB2 + k * SB_STRIP);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst), "l"((const void*)(B + k * N_FIXED + col_base)) : "memory");
    }
    asm volatile("cp.async.commit_group;\n");

    #pragma unroll 2
    for (int i = lane; i < MMA_M * (K_FIXED / 8); i += 32) {
        const int r  = i >> 3;
        const int c8 = i & 7;
        uint32_t dst = smem_u32addr(smA2 + r * SA_STR + c8 * 8);
        if (r < valid) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"((const void*)(A + (row_base + r) * K_FIXED + c8 * 8)) : "memory");
        } else {
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                         :: "r"(dst), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncwarp();

    uint32_t ra[K_TILES][4];
    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        const int r = lane & 15;
        const int c = ki * MMA_K + ((lane >> 4) << 3);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(ra[ki][0]), "=r"(ra[ki][1]), "=r"(ra[ki][2]), "=r"(ra[ki][3])
            : "r"(smem_u32addr(smA2 + r * SA_STR + c)));
    }

    uint32_t rb[K_TILES][2];
    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        const int row_b = ki * MMA_K + (lane & 15);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];"
            : "=r"(rb[ki][0]), "=r"(rb[ki][1])
            : "r"(smem_u32addr(smB2 + row_b * SB_STRIP)));
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(ra[ki][0]), "r"(ra[ki][1]), "r"(ra[ki][2]), "r"(ra[ki][3]),
              "r"(rb[ki][0]), "r"(rb[ki][1]));
    }

    const int or0 = row_base + (lane >> 2);
    const int or1 = or0 + 8;
    const int oc  = col_base + ((lane & 3) << 1);

    half2 v0 = __floats2half2_rn(acc[0], acc[1]);
    half2 v1 = __floats2half2_rn(acc[2], acc[3]);
    if (or0 < M) store_cs_b32(C + or0 * N_FIXED + oc, *reinterpret_cast<uint32_t*>(&v0));
    if (or1 < M) store_cs_b32(C + or1 * N_FIXED + oc, *reinterpret_cast<uint32_t*>(&v1));
}

static constexpr int SBC_STR = 16 + 8;

__global__ void __launch_bounds__(32, 49)
hgemm_512blocks_2ntiles(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M)
{
    __shared__ __align__(128) half smA3[MMA_M * SA_STR];
    __shared__ __align__(128) half smB3[K_FIXED * SBC_STR];

    const int lane     = threadIdx.x;
    const int m_tile   = blockIdx.x;
    const int n_block  = blockIdx.y;
    const int row_base = m_tile * MMA_M;
    const int col_base = n_block * 16;

    if (row_base >= M) return;
    const int valid = min(MMA_M, M - row_base);

    #pragma unroll 4
    for (int i = lane; i < K_FIXED * 2; i += 32) {
        const int r  = i >> 1;
        const int c8 = i & 1;
        uint32_t dst = smem_u32addr(smB3 + r * SBC_STR + c8 * 8);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst), "l"((const void*)(B + r * N_FIXED + col_base + c8 * 8)) : "memory");
    }
    asm volatile("cp.async.commit_group;\n");

    #pragma unroll 2
    for (int i = lane; i < MMA_M * (K_FIXED / 8); i += 32) {
        const int r  = i >> 3;
        const int c8 = i & 7;
        uint32_t dst = smem_u32addr(smA3 + r * SA_STR + c8 * 8);
        if (r < valid) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"((const void*)(A + (row_base + r) * K_FIXED + c8 * 8)) : "memory");
        } else {
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                         :: "r"(dst), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncwarp();

    uint32_t ra[K_TILES][4];
    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        const int r = lane & 15;
        const int c = ki * MMA_K + ((lane >> 4) << 3);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(ra[ki][0]), "=r"(ra[ki][1]), "=r"(ra[ki][2]), "=r"(ra[ki][3])
            : "r"(smem_u32addr(smA3 + r * SA_STR + c)));
    }

    uint32_t rb[K_TILES][2][2];
    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        const int row_b = ki * MMA_K + (lane & 15);
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            asm volatile(
                "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];"
                : "=r"(rb[ki][ni][0]), "=r"(rb[ki][ni][1])
                : "r"(smem_u32addr(smB3 + row_b * SBC_STR + ni * MMA_N)));
        }
    }

    float acc[2][4];
    #pragma unroll
    for (int ni = 0; ni < 2; ni++) acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    #pragma unroll
    for (int ki = 0; ki < K_TILES; ki++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]), "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(ra[ki][0]), "r"(ra[ki][1]), "r"(ra[ki][2]), "r"(ra[ki][3]),
                  "r"(rb[ki][ni][0]), "r"(rb[ki][ni][1]));
        }
    }

    const int or0 = row_base + (lane >> 2);
    const int or1 = or0 + 8;
    const int oc  = col_base + ((lane & 3) << 1);

    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        half2 v0 = __floats2half2_rn(acc[ni][0], acc[ni][1]);
        half2 v1 = __floats2half2_rn(acc[ni][2], acc[ni][3]);
        if (or0 < M) store_cs_b32(C + or0 * N_FIXED + oc + ni * MMA_N, *reinterpret_cast<uint32_t*>(&v0));
        if (or1 < M) store_cs_b32(C + or1 * N_FIXED + oc + ni * MMA_N, *reinterpret_cast<uint32_t*>(&v1));
    }
}

static int g_best_kernel = -1;

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M       = a.size(0);
    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr());

    const int m_tiles = (M + MMA_M - 1) / MMA_M;

    const int      grid_persist = 132;
    const dim3     grid_1024(m_tiles, N_TILES);
    const dim3     grid_512 (m_tiles, N_FIXED / (2 * MMA_N));

    if (g_best_kernel < 0) {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        const int WARMUP = 20, BENCH = 60;
        float ms[3];

        for (int i = 0; i < WARMUP; i++)
            hgemm_persistent_B_in_regs<<<grid_persist, 32>>>(ptr_A, ptr_B, ptr_C, M, m_tiles);
        cudaEventRecord(t0);
        for (int i = 0; i < BENCH; i++)
            hgemm_persistent_B_in_regs<<<grid_persist, 32>>>(ptr_A, ptr_B, ptr_C, M, m_tiles);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[0], t0, t1);

        for (int i = 0; i < WARMUP; i++)
            hgemm_1024blocks_nsplit<<<grid_1024, 32>>>(ptr_A, ptr_B, ptr_C, M);
        cudaEventRecord(t0);
        for (int i = 0; i < BENCH; i++)
            hgemm_1024blocks_nsplit<<<grid_1024, 32>>>(ptr_A, ptr_B, ptr_C, M);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[1], t0, t1);

        for (int i = 0; i < WARMUP; i++)
            hgemm_512blocks_2ntiles<<<grid_512, 32>>>(ptr_A, ptr_B, ptr_C, M);
        cudaEventRecord(t0);
        for (int i = 0; i < BENCH; i++)
            hgemm_512blocks_2ntiles<<<grid_512, 32>>>(ptr_A, ptr_B, ptr_C, M);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[2], t0, t1);

        cudaEventDestroy(t0); cudaEventDestroy(t1);

        g_best_kernel = 0;
        if (ms[1] < ms[g_best_kernel]) g_best_kernel = 1;
        if (ms[2] < ms[g_best_kernel]) g_best_kernel = 2;
    }

    if (g_best_kernel == 0)
        hgemm_persistent_B_in_regs<<<grid_persist, 32>>>(ptr_A, ptr_B, ptr_C, M, m_tiles);
    else if (g_best_kernel == 1)
        hgemm_1024blocks_nsplit<<<grid_1024, 32>>>(ptr_A, ptr_B, ptr_C, M);
    else
        hgemm_512blocks_2ntiles<<<grid_512, 32>>>(ptr_A, ptr_B, ptr_C, M);
}