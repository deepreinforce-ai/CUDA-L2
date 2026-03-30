#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>

using namespace nvcuda;

static constexpr int TM   = 32;
static constexpr int TN   = 64;
static constexpr int TK   = 64;
static constexpr int WM   = 16;
static constexpr int WN   = 16;
static constexpr int WK   = 16;
static constexpr int NW   = 4;
static constexpr int BT   = NW * 32;
static constexpr int PS   = 3;
static constexpr int SA_S = TK + 8;
static constexpr int SB_S = TK + 8;
static constexpr int KST  = TK / WK;

__device__ __forceinline__
void cp16(half* dst, const half* src) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(src) : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void wait1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
__device__ __forceinline__ void wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__
void load_fast(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half smA[][SA_S],
    half smB[][SB_S],
    int m_base, int k_off, int K, int tid)
{
    const int col8 = (tid & 7) * 8;
    const int r    = tid >> 3;

    cp16(&smA[r][col8],      &A[(m_base + r)      * K + k_off + col8]);
    cp16(&smA[r + 16][col8], &A[(m_base + r + 16) * K + k_off + col8]);

    cp16(&smB[r][col8],      &B_col[(uint64_t)r      * K + k_off + col8]);
    cp16(&smB[r + 16][col8], &B_col[(uint64_t)(r+16) * K + k_off + col8]);
    cp16(&smB[r + 32][col8], &B_col[(uint64_t)(r+32) * K + k_off + col8]);
    cp16(&smB[r + 48][col8], &B_col[(uint64_t)(r+48) * K + k_off + col8]);
}

__device__ __forceinline__
void load_safe(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half smA[][SA_S],
    half smB[][SB_S],
    int m_base, int k_off, int M, int K, int tid)
{
    const int col8 = (tid & 7) * 8;
    const int r    = tid >> 3;

    #pragma unroll
    for (int p = 0; p < 2; p++) {
        const int lr = r + p * 16;
        const int gr = m_base + lr;
        const int gc = k_off + col8;
        if (gr < M && gc + 7 < K) {
            cp16(&smA[lr][col8], &A[gr * K + gc]);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                smA[lr][col8+i] = (gr < M && gc+i < K) ? A[gr*K+gc+i] : __float2half(0.f);
        }
    }

    #pragma unroll
    for (int p = 0; p < 4; p++) {
        const int lr = r + p * 16;
        const int gc = k_off + col8;
        if (lr < TN && gc + 7 < K) {
            cp16(&smB[lr][col8], &B_col[(uint64_t)lr * K + gc]);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                smB[lr][col8+i] = (lr < TN && gc+i < K) ? B_col[(uint64_t)lr*K+gc+i] : __float2half(0.f);
        }
    }
}

__global__ void __launch_bounds__(BT, 8)
hgemm_v70_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_col,
    half*       __restrict__ C,
    int M, int N, int K)
{
    const int m_base  = blockIdx.x * TM;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tid     = threadIdx.x;

    const int warp_m = (warp_id >> 1) * WM;
    const int warp_n = (warp_id & 1)  * 32;

    __shared__ half smem_A[PS][TM][SA_S];
    __shared__ half smem_B[PS][TN][SB_S];

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag[2][2];

    const int k_iters = K / TK;
    const bool fast   = (m_base + TM <= M);

    if (fast) {
        load_fast(A, B_col, smem_A[0], smem_B[0], m_base, 0, K, tid);
    } else {
        load_safe(A, B_col, smem_A[0], smem_B[0], m_base, 0, M, K, tid);
    }
    commit();

    if (k_iters > 1) {
        if (fast) {
            load_fast(A, B_col, smem_A[1], smem_B[1], m_base, TK, K, tid);
        } else {
            load_safe(A, B_col, smem_A[1], smem_B[1], m_base, TK, M, K, tid);
        }
        commit();
    }

    wait1();
    __syncthreads();

    wmma::load_matrix_sync(a_frag[0],    &smem_A[0][warp_m][0],        SA_S);
    wmma::load_matrix_sync(b_frag[0][0], &smem_B[0][warp_n][0],        SB_S);
    wmma::load_matrix_sync(b_frag[0][1], &smem_B[0][warp_n + WN][0],   SB_S);

    for (int ki = 0; ki < k_iters; ki++) {
        const int cs = ki % PS;

        if (ki + 2 < k_iters) {
            const int ns    = (ki + 2) % PS;
            const int k_off = (ki + 2) * TK;
            if (fast) {
                load_fast(A, B_col, smem_A[ns], smem_B[ns], m_base, k_off, K, tid);
            } else {
                load_safe(A, B_col, smem_A[ns], smem_B[ns], m_base, k_off, M, K, tid);
            }
            commit();
        }

        #pragma unroll
        for (int ks = 0; ks < KST; ks++) {
            const int cur = ks & 1;
            const int nxt = cur ^ 1;
            const int nko = (ks + 1) * WK;

            if (ks + 1 < KST) {
                wmma::load_matrix_sync(a_frag[nxt],    &smem_A[cs][warp_m][nko],      SA_S);
                wmma::load_matrix_sync(b_frag[nxt][0], &smem_B[cs][warp_n][nko],      SB_S);
                wmma::load_matrix_sync(b_frag[nxt][1], &smem_B[cs][warp_n + WN][nko], SB_S);
            }

            wmma::mma_sync(acc[0], a_frag[cur], b_frag[cur][0], acc[0]);
            wmma::mma_sync(acc[1], a_frag[cur], b_frag[cur][1], acc[1]);
        }

        if (ki + 1 < k_iters) {
            wait1();
            __syncthreads();

            const int ns = (ki + 1) % PS;
            wmma::load_matrix_sync(a_frag[0],    &smem_A[ns][warp_m][0],       SA_S);
            wmma::load_matrix_sync(b_frag[0][0], &smem_B[ns][warp_n][0],       SB_S);
            wmma::load_matrix_sync(b_frag[0][1], &smem_B[ns][warp_n + WN][0],  SB_S);
        }
    }

    wait0();
    __syncthreads();

    float* smem_C = reinterpret_cast<float*>(&smem_B[0][0][0]);

    #pragma unroll
    for (int i = lane; i < WM * TN; i += 32) {
        smem_C[(warp_m + i / TN) * TN + (i % TN)] = 0.f;
    }
    __syncthreads();

    wmma::store_matrix_sync(&smem_C[warp_m * TN + warp_n],       acc[0], TN, wmma::mem_row_major);
    wmma::store_matrix_sync(&smem_C[warp_m * TN + warp_n + WN],  acc[1], TN, wmma::mem_row_major);
    __syncthreads();

    constexpr int N_U4 = (TM * TN) >> 3;

    #pragma unroll
    for (int i = tid; i < N_U4; i += BT) {
        const int flat  = i << 3;
        const int row   = flat >> 6;
        const int col   = flat & 63;
        const int g_row = m_base + row;

        if (g_row < M) {
            const float* src = &smem_C[row * TN + col];

            const __half2 h01 = __float22half2_rn(make_float2(src[0], src[1]));
            const __half2 h23 = __float22half2_rn(make_float2(src[2], src[3]));
            const __half2 h45 = __float22half2_rn(make_float2(src[4], src[5]));
            const __half2 h67 = __float22half2_rn(make_float2(src[6], src[7]));

            uint4 packed;
            packed.x = *reinterpret_cast<const uint32_t*>(&h01);
            packed.y = *reinterpret_cast<const uint32_t*>(&h23);
            packed.z = *reinterpret_cast<const uint32_t*>(&h45);
            packed.w = *reinterpret_cast<const uint32_t*>(&h67);

            *reinterpret_cast<uint4*>(&C[g_row * N + col]) = packed;
        }
    }
}

void cuda_l2_h100_fp32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor b_col_major,
    torch::Tensor c)
{
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const half* ptr_A     = reinterpret_cast<const half*>(a.data_ptr());
    const half* ptr_B_col = reinterpret_cast<const half*>(b_col_major.data_ptr());
    half*       ptr_C     = reinterpret_cast<half*>(c.data_ptr());

    dim3 grid((M + TM - 1) / TM);
    dim3 block(BT);

    hgemm_v70_kernel<<<grid, block>>>(ptr_A, ptr_B_col, ptr_C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("HGEMM v70 kernel error: ") + cudaGetErrorString(err));
}