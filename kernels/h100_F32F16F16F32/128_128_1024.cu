#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include <string>
#include <cstdint>

static constexpr int BM         = 32;
static constexpr int BN         = 32;
static constexpr int BK         = 128;
static constexpr int NUM_KTILES = 8;
static constexpr int THREADS    = 128;
static constexpr int MMA_M      = 16, MMA_N = 8, MMA_K = 16;
static constexpr int WM         = 16, WN = 16;
static constexpr int TN         = WN / MMA_N;
static constexpr int TK         = BK / MMA_K;
static constexpr int NUM_STAGES = 3;

static constexpr int SA_STRIDE    = 128;
static constexpr int SA_SIZE      = BM * SA_STRIDE;

static constexpr int SB_STRIDE    = 128;
static constexpr int SB_SIZE      = BN * SB_STRIDE;

static constexpr int SMEM_A_TOTAL = SA_SIZE * NUM_STAGES;
static constexpr int SMEM_B_TOTAL = SB_SIZE * NUM_STAGES;
static constexpr int SMEM_BYTES   = (SMEM_A_TOTAL + SMEM_B_TOTAL) * sizeof(__half);

__device__ __forceinline__
void cp_async_16b(void* dst, const void* src) {
    uint32_t addr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
}

__device__ __forceinline__
void mma_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3)
    );
}

__device__ __forceinline__
void load_stage(
    __half* __restrict__ sA_stage,
    __half* __restrict__ sB_stage,
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    int m_start, int n_start, int k_start, int K, int tid)
{
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        int idx  = tid + s * THREADS;
        int row  = idx >> 4;
        int lcol = (idx & 15) << 3;
        int scol = lcol ^ ((row & 7) << 3);
        cp_async_16b(sA_stage + row * SA_STRIDE + scol,
                     A + (m_start + row) * K + k_start + lcol);
    }
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        int idx  = tid + s * THREADS;
        int n    = idx >> 4;
        int lcol = (idx & 15) << 3;
        int scol = lcol ^ ((n & 7) << 3);
        cp_async_16b(sB_stage + n * SB_STRIDE + scol,
                     B_col + (size_t)(n_start + n) * K + k_start + lcol);
    }
}

__global__
__launch_bounds__(128, 8)
void hgemm_v15_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B_col,
    __half*       __restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int m_start = bm * BM;
    const int n_start = bn * BN;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int warp_m   = warp_id >> 1;
    const int warp_n   = warp_id & 1;
    const int warp_row = warp_m * WM;
    const int warp_col = warp_n * WN;

    extern __shared__ char smem_raw[];
    __half* smem = reinterpret_cast<__half*>(smem_raw);

    __half* sA[NUM_STAGES], *sB[NUM_STAGES];
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        sA[s] = smem + s * SA_SIZE;
        sB[s] = smem + SMEM_A_TOTAL + s * SB_SIZE;
    }

    float acc[TN][4];
    #pragma unroll
    for (int tn = 0; tn < TN; tn++)
        acc[tn][0] = acc[tn][1] = acc[tn][2] = acc[tn][3] = 0.f;

    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        load_stage(sA[s], sB[s], A, B_col, m_start, n_start, s * BK, K, tid);
        cp_async_commit();
    }

    int cur_stage = 0;

    #pragma unroll 1
    for (int tile = 0; tile < NUM_KTILES; tile++) {
        int pf_tile  = tile + (NUM_STAGES - 1);
        int pf_stage = (cur_stage + NUM_STAGES - 1) % NUM_STAGES;

        if (pf_tile < NUM_KTILES) {
            load_stage(sA[pf_stage], sB[pf_stage], A, B_col,
                       m_start, n_start, pf_tile * BK, K, tid);
        }
        cp_async_commit();

        cp_async_wait<NUM_STAGES - 1>();
        __syncthreads();

        __half* csA = sA[cur_stage];
        __half* csB = sB[cur_stage];

        uint32_t a_frag[2][4];
        uint32_t b_frag[2][TN][2];

        {
            int a_row  = warp_row + (lane & 15);
            int a_lcol = (lane >> 4) << 3;
            int a_scol = a_lcol ^ ((a_row & 7) << 3);
            uint32_t addr = __cvta_generic_to_shared(csA + a_row * SA_STRIDE + a_scol);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                : "=r"(a_frag[0][0]),"=r"(a_frag[0][1]),"=r"(a_frag[0][2]),"=r"(a_frag[0][3])
                : "r"(addr)
            );
        }
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int b_n  = warp_col + tn * MMA_N + (lane & 7);
            int b_lk = ((lane >> 3) << 3);
            int b_sk = b_lk ^ ((b_n & 7) << 3);
            uint32_t addr = __cvta_generic_to_shared(csB + b_n * SB_STRIDE + b_sk);
            asm volatile(
                "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];\n"
                : "=r"(b_frag[0][tn][0]),"=r"(b_frag[0][tn][1])
                : "r"(addr)
            );
        }

        #pragma unroll
        for (int tk = 0; tk < TK; tk++) {
            int cur_buf = tk & 1;
            int nxt_buf = 1 - cur_buf;

            if (tk + 1 < TK) {
                int k_off  = (tk + 1) * MMA_K;
                int a_row  = warp_row + (lane & 15);
                int a_lcol = k_off + ((lane >> 4) << 3);
                int a_scol = a_lcol ^ ((a_row & 7) << 3);
                uint32_t addr = __cvta_generic_to_shared(csA + a_row * SA_STRIDE + a_scol);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(a_frag[nxt_buf][0]),"=r"(a_frag[nxt_buf][1]),
                      "=r"(a_frag[nxt_buf][2]),"=r"(a_frag[nxt_buf][3])
                    : "r"(addr)
                );
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    int b_n  = warp_col + tn * MMA_N + (lane & 7);
                    int b_lk = k_off + ((lane >> 3) << 3);
                    int b_sk = b_lk ^ ((b_n & 7) << 3);
                    uint32_t baddr = __cvta_generic_to_shared(csB + b_n * SB_STRIDE + b_sk);
                    asm volatile(
                        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];\n"
                        : "=r"(b_frag[nxt_buf][tn][0]),"=r"(b_frag[nxt_buf][tn][1])
                        : "r"(baddr)
                    );
                }
            }

            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                mma_f32(
                    acc[tn][0], acc[tn][1], acc[tn][2], acc[tn][3],
                    a_frag[cur_buf][0], a_frag[cur_buf][1],
                    a_frag[cur_buf][2], a_frag[cur_buf][3],
                    b_frag[cur_buf][tn][0], b_frag[cur_buf][tn][1],
                    acc[tn][0], acc[tn][1], acc[tn][2], acc[tn][3]
                );
            }
        }

        __syncthreads();
        cur_stage = (cur_stage + 1) % NUM_STAGES;
    }

    cp_async_wait<0>();

    int g_row0 = m_start + warp_row + (lane >> 2);
    int g_row1 = g_row0 + 8;

    #pragma unroll
    for (int tn = 0; tn < TN; tn++) {
        int g_col = n_start + warp_col + tn * MMA_N + (lane & 3) * 2;
        *reinterpret_cast<__half2*>(C + g_row0 * N + g_col) =
            __floats2half2_rn(acc[tn][0], acc[tn][1]);
        *reinterpret_cast<__half2*>(C + g_row1 * N + g_col) =
            __floats2half2_rn(acc[tn][2], acc[tn][3]);
    }
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)b.size(1);

    const __half* pA    = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* pBcol = reinterpret_cast<const __half*>(b_col_major.data_ptr<at::Half>());
    __half*       pC    = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_v15_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        attr_set = true;
    }

    dim3 grid(M / BM, N / BN);
    hgemm_v15_kernel<<<grid, THREADS, SMEM_BYTES>>>(pA, pBcol, pC, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}