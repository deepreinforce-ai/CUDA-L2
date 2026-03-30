#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda::wmma;

#define M_FULL   64
#define N_FULL   256
#define K_FULL   8192
#define BN       64
#define BK       32
#define WMMA_K   16
#define STAGES   3
#define THREADS  256
#define NUM_WARPS 8
#define WMMA_M   16
#define WMMA_N   16

#define A_PAD    8
#define B_PAD    8
#define A_S      (BK + A_PAD)
#define B_S      (BN + B_PAD)

__global__ __launch_bounds__(256, 4)
void hgemm_3stage_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C
) {
    const int n_block = blockIdx.x;
    const int n_start = n_block * BN;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_m   = warp_id & 3;
    const int warp_n   = warp_id >> 2;
    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * 32;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    fill_fragment(acc[0], 0.0f);
    fill_fragment(acc[1], 0.0f);

    __shared__ __align__(128) __half smem_A[STAGES][M_FULL][A_S];
    __shared__ __align__(128) __half smem_B[STAGES][BK][B_S];

    const int num_iters = K_FULL / BK;

    auto issue_load = [&](int stage, int k_off) __attribute__((always_inline)) {
        {
            int row = tid >> 2;
            int col = (tid & 3) << 3;
            const __half* src = &A[row * K_FULL + k_off + col];
            __half* dst = &smem_A[stage][row][col];
            uint32_t addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(addr), "l"(src) : "memory");
        }
        {
            int row = tid >> 3;
            int col = (tid & 7) << 3;
            const __half* src = &B[(k_off + row) * N_FULL + n_start + col];
            __half* dst = &smem_B[stage][row][col];
            uint32_t addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(addr), "l"(src) : "memory");
        }
    };

    for (int s = 0; s < STAGES - 1; s++) {
        issue_load(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    for (int iter = 0; iter < num_iters; iter++) {
        if (iter + STAGES - 1 < num_iters) {
            int stage_w = (iter + STAGES - 1) % STAGES;
            issue_load(stage_w, (iter + STAGES - 1) * BK);
            asm volatile("cp.async.commit_group;\n");
        } else {
            asm volatile("cp.async.commit_group;\n");
        }

        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();

        int stage_r = iter % STAGES;

        #pragma unroll
        for (int ki = 0; ki < BK / WMMA_K; ki++) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> frag_a;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> frag_b0, frag_b1;

            load_matrix_sync(frag_a,  &smem_A[stage_r][warp_row][ki * WMMA_K], A_S);
            load_matrix_sync(frag_b0, &smem_B[stage_r][ki * WMMA_K][warp_col],        B_S);
            load_matrix_sync(frag_b1, &smem_B[stage_r][ki * WMMA_K][warp_col + WMMA_N], B_S);

            mma_sync(acc[0], frag_a, frag_b0, acc[0]);
            mma_sync(acc[1], frag_a, frag_b1, acc[1]);
        }
    }

    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    __shared__ float fstage[M_FULL][BN];

    #pragma unroll 4
    for (int i = tid; i < M_FULL * BN; i += THREADS)
        fstage[i / BN][i % BN] = 0.f;
    __syncthreads();

    store_matrix_sync(&fstage[warp_row][warp_col],          acc[0], BN, mem_row_major);
    store_matrix_sync(&fstage[warp_row][warp_col + WMMA_N], acc[1], BN, mem_row_major);
    __syncthreads();

    #pragma unroll 4
    for (int i = tid; i < M_FULL * BN / 2; i += THREADS) {
        int idx2 = i * 2;
        int r = idx2 / BN;
        int c = idx2 % BN;
        float2 fv = make_float2(fstage[r][c], fstage[r][c+1]);
        __half2 hv = __float22half2_rn(fv);
        reinterpret_cast<__half2*>(&C[r * N_FULL + n_start + c])[0] = hv;
    }
}

#define SK2       32
#define K_SLICE2  256

static float* g_workspace2 = nullptr;

__global__ __launch_bounds__(256, 4)
void hgemm_splitk32_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ ws,
    int k_slice_size
) {
    const int n_block = blockIdx.x;
    const int k_slice = blockIdx.y;

    const int n_start = n_block * BN;
    const int k_start = k_slice * k_slice_size;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;

    const int warp_m   = warp_id & 3;
    const int warp_n   = warp_id >> 2;
    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * 32;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    fill_fragment(acc[0], 0.0f);
    fill_fragment(acc[1], 0.0f);

    __shared__ __align__(128) __half smem_A[STAGES][M_FULL][A_S];
    __shared__ __align__(128) __half smem_B[STAGES][BK][B_S];

    const int num_iters = k_slice_size / BK;

    auto issue_load = [&](int stage, int k_off) __attribute__((always_inline)) {
        {
            int row = tid >> 2;
            int col = (tid & 3) << 3;
            const __half* src = &A[row * K_FULL + k_off + col];
            __half* dst = &smem_A[stage][row][col];
            uint32_t addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(addr), "l"(src) : "memory");
        }
        {
            int row = tid >> 3;
            int col = (tid & 7) << 3;
            const __half* src = &B[(k_off + row) * N_FULL + n_start + col];
            __half* dst = &smem_B[stage][row][col];
            uint32_t addr = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "r"(addr), "l"(src) : "memory");
        }
    };

    for (int s = 0; s < STAGES - 1 && s < num_iters; s++) {
        issue_load(s, k_start + s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    for (int iter = 0; iter < num_iters; iter++) {
        if (iter + STAGES - 1 < num_iters) {
            int sw = (iter + STAGES - 1) % STAGES;
            issue_load(sw, k_start + (iter + STAGES - 1) * BK);
            asm volatile("cp.async.commit_group;\n");
        } else {
            asm volatile("cp.async.commit_group;\n");
        }
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();

        int sr = iter % STAGES;
        #pragma unroll
        for (int ki = 0; ki < BK / WMMA_K; ki++) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> fa;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> fb0, fb1;
            load_matrix_sync(fa,  &smem_A[sr][warp_row][ki * WMMA_K], A_S);
            load_matrix_sync(fb0, &smem_B[sr][ki * WMMA_K][warp_col],        B_S);
            load_matrix_sync(fb1, &smem_B[sr][ki * WMMA_K][warp_col + WMMA_N], B_S);
            mma_sync(acc[0], fa, fb0, acc[0]);
            mma_sync(acc[1], fa, fb1, acc[1]);
        }
    }
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    __shared__ float fstage[M_FULL][BN];
    #pragma unroll 4
    for (int i = tid; i < M_FULL * BN; i += THREADS)
        fstage[i / BN][i % BN] = 0.f;
    __syncthreads();
    store_matrix_sync(&fstage[warp_row][warp_col],          acc[0], BN, mem_row_major);
    store_matrix_sync(&fstage[warp_row][warp_col + WMMA_N], acc[1], BN, mem_row_major);
    __syncthreads();

    float* ws_base = ws + (size_t)k_slice * (M_FULL * N_FULL);
    #pragma unroll 4
    for (int i = tid; i < M_FULL * BN; i += THREADS) {
        int r = i / BN;
        int c = i % BN;
        ws_base[r * N_FULL + n_start + c] = fstage[r][c];
    }
}

__global__ __launch_bounds__(256)
void reduce_f32_to_f16(
    const float* __restrict__ ws,
    __half*       __restrict__ C,
    int SK_val,
    int MN
) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= MN) return;

    float s0=0, s1=0, s2=0, s3=0;
    #pragma unroll
    for (int s = 0; s < SK2; s++) {
        const float* p = ws + (size_t)s * MN + base;
        float4 v = *reinterpret_cast<const float4*>(p);
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }
    reinterpret_cast<__half2*>(C)[base/2+0] = __float22half2_rn({s0, s1});
    reinterpret_cast<__half2*>(C)[base/2+1] = __float22half2_rn({s2, s3});
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c
) {
    const __half* A = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* B = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    __half* C       = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    const int MN = M_FULL * N_FULL;

    const size_t ws_bytes = (size_t)SK2 * MN * sizeof(float);
    if (g_workspace2 == nullptr) {
        cudaMalloc(&g_workspace2, ws_bytes);
    }

    dim3 grid(N_FULL / BN, SK2);
    dim3 block(THREADS);
    hgemm_splitk32_kernel<<<grid, block>>>(A, B, g_workspace2, K_SLICE2);

    int nblk = (MN / 4 + 255) / 256;
    reduce_f32_to_f16<<<nblk, 256>>>(g_workspace2, C, SK2, MN);
}