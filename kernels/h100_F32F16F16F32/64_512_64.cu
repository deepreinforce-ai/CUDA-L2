#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <torch/types.h>

__global__ void __launch_bounds__(32, 16)
hgemm_v11_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int N
) {
    const int n_chunk = blockIdx.x * 64;
    const int m_base  = blockIdx.y * 16;
    const int lane    = threadIdx.x;

    __shared__ __align__(128) __half smA[16][72];
    __shared__ __align__(128) __half smB[64][72];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx   = lane * 4 + i;
        int row   = idx >> 3;
        int f4col = idx & 7;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"((unsigned)__cvta_generic_to_shared(&smA[row][f4col * 8])),
               "l"((const void*)(A + (m_base + row) * 64 + f4col * 8)));
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx   = lane * 16 + i;
        int k_row = idx >> 3;
        int f4_n  = idx & 7;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"((unsigned)__cvta_generic_to_shared(&smB[k_row][f4_n * 8])),
               "l"((const void*)(B + k_row * N + n_chunk + f4_n * 8)));
    }

    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncwarp();

    uint32_t a0[4], a1[4], a2[4], a3[4];
    {
        int a_row   = lane & 15;
        int lane_hi = (lane >> 4) * 8;
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a0[0]),"=r"(a0[1]),"=r"(a0[2]),"=r"(a0[3])
            : "r"((unsigned)__cvta_generic_to_shared(&smA[a_row][0  + lane_hi])));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a1[0]),"=r"(a1[1]),"=r"(a1[2]),"=r"(a1[3])
            : "r"((unsigned)__cvta_generic_to_shared(&smA[a_row][16 + lane_hi])));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a2[0]),"=r"(a2[1]),"=r"(a2[2]),"=r"(a2[3])
            : "r"((unsigned)__cvta_generic_to_shared(&smA[a_row][32 + lane_hi])));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a3[0]),"=r"(a3[1]),"=r"(a3[2]),"=r"(a3[3])
            : "r"((unsigned)__cvta_generic_to_shared(&smA[a_row][48 + lane_hi])));
    }

    float acc0l[4]={0,0,0,0}, acc0r[4]={0,0,0,0};
    float acc1l[4]={0,0,0,0}, acc1r[4]={0,0,0,0};
    float acc2l[4]={0,0,0,0}, acc2r[4]={0,0,0,0};
    float acc3l[4]={0,0,0,0}, acc3r[4]={0,0,0,0};

    const int k_loc = lane & 15;

#define DO_K_TILE(K_OFF, AREG) \
{ \
    uint32_t bl0[2],br0[2],bl1[2],br1[2],bl2[2],br2[2],bl3[2],br3[2]; \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(bl0[0]),"=r"(bl0[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][0]))); \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(br0[0]),"=r"(br0[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][8]))); \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(bl1[0]),"=r"(bl1[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][16]))); \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(br1[0]),"=r"(br1[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][24]))); \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(bl2[0]),"=r"(bl2[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][32]))); \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(br2[0]),"=r"(br2[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][40]))); \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(bl3[0]),"=r"(bl3[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][48]))); \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" \
        : "=r"(br3[0]),"=r"(br3[1]) \
        : "r"((unsigned)__cvta_generic_to_shared(&smB[(K_OFF)+k_loc][56]))); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc0l[0]),"+f"(acc0l[1]),"+f"(acc0l[2]),"+f"(acc0l[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(bl0[0]),"r"(bl0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc0r[0]),"+f"(acc0r[1]),"+f"(acc0r[2]),"+f"(acc0r[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(br0[0]),"r"(br0[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc1l[0]),"+f"(acc1l[1]),"+f"(acc1l[2]),"+f"(acc1l[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(bl1[0]),"r"(bl1[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc1r[0]),"+f"(acc1r[1]),"+f"(acc1r[2]),"+f"(acc1r[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(br1[0]),"r"(br1[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc2l[0]),"+f"(acc2l[1]),"+f"(acc2l[2]),"+f"(acc2l[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(bl2[0]),"r"(bl2[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc2r[0]),"+f"(acc2r[1]),"+f"(acc2r[2]),"+f"(acc2r[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(br2[0]),"r"(br2[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc3l[0]),"+f"(acc3l[1]),"+f"(acc3l[2]),"+f"(acc3l[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(bl3[0]),"r"(bl3[1])); \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
        :"+f"(acc3r[0]),"+f"(acc3r[1]),"+f"(acc3r[2]),"+f"(acc3r[3]) \
        :"r"(AREG[0]),"r"(AREG[1]),"r"(AREG[2]),"r"(AREG[3]),"r"(br3[0]),"r"(br3[1])); \
}

    DO_K_TILE( 0, a0)
    DO_K_TILE(16, a1)
    DO_K_TILE(32, a2)
    DO_K_TILE(48, a3)

    #undef DO_K_TILE

    {
        const int row0 = m_base + (lane >> 2);
        const int row1 = row0 + 8;
        const int lc   = (lane & 3) * 2;
        const int n0 = n_chunk +  0;
        const int n1 = n_chunk + 16;
        const int n2 = n_chunk + 32;
        const int n3 = n_chunk + 48;

        *reinterpret_cast<__half2*>(C + row0*N + n0+lc)   = __halves2half2(__float2half(acc0l[0]),__float2half(acc0l[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n0+lc)   = __halves2half2(__float2half(acc0l[2]),__float2half(acc0l[3]));
        *reinterpret_cast<__half2*>(C + row0*N + n0+8+lc) = __halves2half2(__float2half(acc0r[0]),__float2half(acc0r[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n0+8+lc) = __halves2half2(__float2half(acc0r[2]),__float2half(acc0r[3]));

        *reinterpret_cast<__half2*>(C + row0*N + n1+lc)   = __halves2half2(__float2half(acc1l[0]),__float2half(acc1l[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n1+lc)   = __halves2half2(__float2half(acc1l[2]),__float2half(acc1l[3]));
        *reinterpret_cast<__half2*>(C + row0*N + n1+8+lc) = __halves2half2(__float2half(acc1r[0]),__float2half(acc1r[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n1+8+lc) = __halves2half2(__float2half(acc1r[2]),__float2half(acc1r[3]));

        *reinterpret_cast<__half2*>(C + row0*N + n2+lc)   = __halves2half2(__float2half(acc2l[0]),__float2half(acc2l[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n2+lc)   = __halves2half2(__float2half(acc2l[2]),__float2half(acc2l[3]));
        *reinterpret_cast<__half2*>(C + row0*N + n2+8+lc) = __halves2half2(__float2half(acc2r[0]),__float2half(acc2r[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n2+8+lc) = __halves2half2(__float2half(acc2r[2]),__float2half(acc2r[3]));

        *reinterpret_cast<__half2*>(C + row0*N + n3+lc)   = __halves2half2(__float2half(acc3l[0]),__float2half(acc3l[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n3+lc)   = __halves2half2(__float2half(acc3l[2]),__float2half(acc3l[3]));
        *reinterpret_cast<__half2*>(C + row0*N + n3+8+lc) = __halves2half2(__float2half(acc3r[0]),__float2half(acc3r[1]));
        *reinterpret_cast<__half2*>(C + row1*N + n3+8+lc) = __halves2half2(__float2half(acc3r[2]),__float2half(acc3r[3]));
    }
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    const int M = a.size(0);
    const int N = b.size(1);

    const __half* ptr_A = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* ptr_B = reinterpret_cast<const __half*>(b.data_ptr());
    __half* ptr_C       = reinterpret_cast<__half*>(c.data_ptr());

    dim3 grid(N / 64, M / 16);
    dim3 block(32, 1);
    hgemm_v11_kernel<<<grid, block>>>(ptr_A, ptr_B, ptr_C, N);
}