#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

#define BM 64
#define BN 64
#define BK 32
#define SPLIT_K 16
#define NUM_STAGES 4
#define NWARPS 4
#define THREADS (NWARPS * 32)

#define SMEM_A_STRIDE 40
#define SMEM_B_STRIDE 72

static float* s_workspace = nullptr;
static size_t s_workspace_bytes = 0;

static float* ensure_workspace(size_t bytes) {
    if (s_workspace_bytes < bytes) {
        if (s_workspace) cudaFree(s_workspace);
        cudaMalloc(&s_workspace, bytes);
        s_workspace_bytes = bytes;
    }
    return s_workspace;
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    unsigned dst_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"((unsigned long long)(uintptr_t)src)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template<int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
}

__global__ __launch_bounds__(128, 6)
void hgemm_main(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ workspace,
    int M, int N, int K, int K_slice
) {
    const int by      = blockIdx.x;
    const int bk_idx  = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int block_m = by * BM;
    const int k_start = bk_idx * K_slice;
    const int num_steps = K_slice / BK;

    const int warp_m = warp_id * 16;

    extern __shared__ char smem_buf[];
    half* smem_A = (half*)smem_buf;
    half* smem_B = smem_A + NUM_STAGES * BM * SMEM_A_STRIDE;

    float acc[8][4];
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        acc[ni][0] = 0.f; acc[ni][1] = 0.f;
        acc[ni][2] = 0.f; acc[ni][3] = 0.f;
    }

    auto issue_A = [&](int step, int s) __attribute__((always_inline)) {
        const int k_base = k_start + step * BK;
        half* base = smem_A + s * BM * SMEM_A_STRIDE;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int fi  = tid * 2 + i;
            const int row = fi >> 2;
            const int c8  = (fi & 3) << 3;
            cp_async16(base + row * SMEM_A_STRIDE + c8,
                       A + (block_m + row) * K + k_base + c8);
        }
    };

    auto issue_B = [&](int step, int s) __attribute__((always_inline)) {
        const int k_base = k_start + step * BK;
        half* base = smem_B + s * BK * SMEM_B_STRIDE;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int fi  = tid * 2 + i;
            const int row = fi >> 3;
            const int c8  = (fi & 7) << 3;
            cp_async16(base + row * SMEM_B_STRIDE + c8,
                       B + (k_base + row) * N + c8);
        }
    };

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < num_steps) {
            issue_A(s, s);
            issue_B(s, s);
        }
        cp_async_commit();
    }

    #pragma unroll 4
    for (int step = 0; step < num_steps; step++) {
        const int cur_s = step % NUM_STAGES;
        const int nxt   = step + (NUM_STAGES - 1);

        if (nxt < num_steps) {
            issue_A(nxt, nxt % NUM_STAGES);
            issue_B(nxt, nxt % NUM_STAGES);
        }
        cp_async_commit();

        cp_async_wait<NUM_STAGES - 1>();
        __syncthreads();

        const half* cA = smem_A + cur_s * BM * SMEM_A_STRIDE;
        const half* cB = smem_B + cur_s * BK * SMEM_B_STRIDE;

        uint32_t ra0[4], ra1[4];
        {
            const int row = warp_m + (lane_id & 15);
            const int col0 = (lane_id >> 4) << 3;
            const int col1 = 16 + ((lane_id >> 4) << 3);
            const uint32_t addr0 = __cvta_generic_to_shared(cA + row * SMEM_A_STRIDE + col0);
            const uint32_t addr1 = __cvta_generic_to_shared(cA + row * SMEM_A_STRIDE + col1);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra0[0]), "=r"(ra0[1]), "=r"(ra0[2]), "=r"(ra0[3])
                : "r"(addr0)
            );
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(ra1[0]), "=r"(ra1[1]), "=r"(ra1[2]), "=r"(ra1[3])
                : "r"(addr1)
            );
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            uint32_t rb[2];
            {
                const int k_row = lane_id & 15;
                const int n_col = ni * 8;
                const uint32_t addr = __cvta_generic_to_shared(
                    cB + k_row * SMEM_B_STRIDE + n_col);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(rb[0]), "=r"(rb[1])
                    : "r"(addr)
                );
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(ra0[0]), "r"(ra0[1]), "r"(ra0[2]), "r"(ra0[3]),
                  "r"(rb[0]), "r"(rb[1])
            );
        }

        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            uint32_t rb[2];
            {
                const int k_row = 16 + (lane_id & 15);
                const int n_col = ni * 8;
                const uint32_t addr = __cvta_generic_to_shared(
                    cB + k_row * SMEM_B_STRIDE + n_col);
                asm volatile(
                    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(rb[0]), "=r"(rb[1])
                    : "r"(addr)
                );
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(ra1[0]), "r"(ra1[1]), "r"(ra1[2]), "r"(ra1[3]),
                  "r"(rb[0]), "r"(rb[1])
            );
        }
    }

    cp_async_wait<0>();

    float* ws = workspace + (size_t)bk_idx * M * N;
    const int t_row0 = block_m + warp_m + (lane_id >> 2);
    const int t_row1 = t_row0 + 8;
    const int t_col  = (lane_id & 3) << 1;

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        const int c = ni * 8 + t_col;
        *reinterpret_cast<float2*>(ws + t_row0 * N + c) = make_float2(acc[ni][0], acc[ni][1]);
        *reinterpret_cast<float2*>(ws + t_row1 * N + c) = make_float2(acc[ni][2], acc[ni][3]);
    }
}

__global__ __launch_bounds__(256, 8)
void reduce_main(
    const float* __restrict__ ws,
    half* __restrict__ C,
    int MN
) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base >= MN) return;

    float4 s0 = {0.f, 0.f, 0.f, 0.f};
    float4 s1 = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int sk = 0; sk < SPLIT_K; sk++) {
        const float* p = ws + (size_t)sk * MN + base;
        float4 v0 = __ldg(reinterpret_cast<const float4*>(p));
        float4 v1 = __ldg(reinterpret_cast<const float4*>(p + 4));
        s0.x += v0.x; s0.y += v0.y; s0.z += v0.z; s0.w += v0.w;
        s1.x += v1.x; s1.y += v1.y; s1.z += v1.z; s1.w += v1.w;
    }

    half* out = C + base;
    *reinterpret_cast<half2*>(out + 0) = __float22half2_rn(make_float2(s0.x, s0.y));
    *reinterpret_cast<half2*>(out + 2) = __float22half2_rn(make_float2(s0.z, s0.w));
    *reinterpret_cast<half2*>(out + 4) = __float22half2_rn(make_float2(s1.x, s1.y));
    *reinterpret_cast<half2*>(out + 6) = __float22half2_rn(make_float2(s1.z, s1.w));
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const half* ptr_A = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    half* ptr_C       = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    int split_k = SPLIT_K;
    while (split_k > 1 && (K % (split_k * BK)) != 0) split_k >>= 1;
    const int k_slice = K / split_k;

    const size_t ws_bytes = (size_t)split_k * M * N * sizeof(float);
    float* d_ws = ensure_workspace(ws_bytes);

    const size_t smem_size =
        (size_t)NUM_STAGES * (BM * SMEM_A_STRIDE + BK * SMEM_B_STRIDE) * sizeof(half);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_main,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        attr_set = true;
    }

    dim3 grid_main((M + BM - 1) / BM, split_k);
    dim3 block_main(THREADS);

    hgemm_main<<<grid_main, block_main, smem_size>>>(
        ptr_A, ptr_B, d_ws, M, N, K, k_slice);

    const int MN = M * N;
    const int red_tpb = 256;
    const int red_blocks = (MN / 8 + red_tpb - 1) / red_tpb;
    reduce_main<<<red_blocks, red_tpb>>>(d_ws, ptr_C, MN);
}