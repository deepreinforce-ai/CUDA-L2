#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <stdint.h>
#include <mma.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__device__ __forceinline__ int swz8(int row, int col8) {
    return col8 ^ (row & 7);
}

__device__ __forceinline__ void mma_m16n8k16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void __launch_bounds__(128, 8)
hgemm_128t_4warp(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int tiles_m)
{
    __shared__ __half smem_B[128 * 128];
    __shared__ __half smem_A[16 * 128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx  = tid + i * 128;
        int row  = idx >> 4;
        int col8 = idx & 15;
        int sc   = swz8(row, col8);
        uint32_t dst = __cvta_generic_to_shared(&smem_B[row * 128 + sc * 8]);
        asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
            :: "r"(dst), "l"(reinterpret_cast<const void*>(B + row * 128 + col8 * 8)));
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int wn = warp_id * 4;

    for (int tile = blockIdx.x; tile < tiles_m; tile += gridDim.x) {
        int block_m = tile * 16;

        {
            const __half* A_tile = A + block_m * 128;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx  = tid + i * 128;
                int row  = idx >> 4;
                int col8 = idx & 15;
                int sc   = swz8(row, col8);
                int gr   = block_m + row;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[row * 128 + sc * 8]);
                if (gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"(reinterpret_cast<const void*>(A_tile + row * 128 + col8 * 8)));
                } else {
                    *reinterpret_cast<float4*>(&smem_A[row * 128 + sc * 8]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            __syncthreads();
        }

        float a00=0,a01=0,a02=0,a03=0;
        float a10=0,a11=0,a12=0,a13=0;
        float a20=0,a21=0,a22=0,a23=0;
        float a30=0,a31=0,a32=0,a33=0;

        uint32_t ra0,ra1,ra2,ra3;
        uint32_t rna0,rna1,rna2,rna3;
        uint32_t rb00,rb01, rb10,rb11, rb20,rb21, rb30,rb31;
        uint32_t rnb00,rnb01, rnb10,rnb11, rnb20,rnb21, rnb30,rnb31;

        {
            int ar  = lane_id & 15;
            int ac  = lane_id >> 4;
            int asc = swz8(ar, ac);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[ar * 128 + asc * 8]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                :"=r"(ra0),"=r"(ra1),"=r"(ra2),"=r"(ra3):"r"(addr));
        }
        {
            int br = lane_id & 15;
            uint32_t ba0 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+0)*8]);
            uint32_t ba1 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+1)*8]);
            uint32_t ba2 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+2)*8]);
            uint32_t ba3 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+3)*8]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb00),"=r"(rb01):"r"(ba0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb10),"=r"(rb11):"r"(ba1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb20),"=r"(rb21):"r"(ba2));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb30),"=r"(rb31):"r"(ba3));
        }

#define LDNEXT(nkk) \
        { \
            int ar2  = lane_id & 15; \
            int ac2  = ((nkk) >> 3) + (lane_id >> 4); \
            int asc2 = swz8(ar2, ac2); \
            uint32_t aaddr = __cvta_generic_to_shared(&smem_A[ar2 * 128 + asc2 * 8]); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                :"=r"(rna0),"=r"(rna1),"=r"(rna2),"=r"(rna3):"r"(aaddr)); \
            int br2 = (nkk) + (lane_id & 15); \
            uint32_t bb0 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+0)*8]); \
            uint32_t bb1 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+1)*8]); \
            uint32_t bb2 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+2)*8]); \
            uint32_t bb3 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+3)*8]); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb00),"=r"(rnb01):"r"(bb0)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb10),"=r"(rnb11):"r"(bb1)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb20),"=r"(rnb21):"r"(bb2)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb30),"=r"(rnb31):"r"(bb3)); \
        }

#define DOMMA() \
        mma_m16n8k16(a00,a01,a02,a03, ra0,ra1,ra2,ra3, rb00,rb01, a00,a01,a02,a03); \
        mma_m16n8k16(a10,a11,a12,a13, ra0,ra1,ra2,ra3, rb10,rb11, a10,a11,a12,a13); \
        mma_m16n8k16(a20,a21,a22,a23, ra0,ra1,ra2,ra3, rb20,rb21, a20,a21,a22,a23); \
        mma_m16n8k16(a30,a31,a32,a33, ra0,ra1,ra2,ra3, rb30,rb31, a30,a31,a32,a33);

#define ADVANCE() \
        ra0=rna0; ra1=rna1; ra2=rna2; ra3=rna3; \
        rb00=rnb00; rb01=rnb01; rb10=rnb10; rb11=rnb11; \
        rb20=rnb20; rb21=rnb21; rb30=rnb30; rb31=rnb31;

        LDNEXT(16)  DOMMA() ADVANCE()
        LDNEXT(32)  DOMMA() ADVANCE()
        LDNEXT(48)  DOMMA() ADVANCE()
        LDNEXT(64)  DOMMA() ADVANCE()
        LDNEXT(80)  DOMMA() ADVANCE()
        LDNEXT(96)  DOMMA() ADVANCE()
        LDNEXT(112) DOMMA() ADVANCE()
        DOMMA()

#undef LDNEXT
#undef DOMMA
#undef ADVANCE

        {
            int r0       = block_m + (lane_id >> 2);
            int r1       = r0 + 8;
            int lane_col = (lane_id & 3) * 2;

            if (r0 < M) {
                int c0 = (wn + 0) * 8 + lane_col;
                int c1 = (wn + 1) * 8 + lane_col;
                int c2 = (wn + 2) * 8 + lane_col;
                int c3 = (wn + 3) * 8 + lane_col;
                __half2 h00 = __floats2half2_rn(a00, a01);
                __half2 h10 = __floats2half2_rn(a10, a11);
                __half2 h20 = __floats2half2_rn(a20, a21);
                __half2 h30 = __floats2half2_rn(a30, a31);
                *reinterpret_cast<uint32_t*>(&C[r0 * 128 + c0]) = reinterpret_cast<uint32_t&>(h00);
                *reinterpret_cast<uint32_t*>(&C[r0 * 128 + c1]) = reinterpret_cast<uint32_t&>(h10);
                *reinterpret_cast<uint32_t*>(&C[r0 * 128 + c2]) = reinterpret_cast<uint32_t&>(h20);
                *reinterpret_cast<uint32_t*>(&C[r0 * 128 + c3]) = reinterpret_cast<uint32_t&>(h30);
            }
            if (r1 < M) {
                int c0 = (wn + 0) * 8 + lane_col;
                int c1 = (wn + 1) * 8 + lane_col;
                int c2 = (wn + 2) * 8 + lane_col;
                int c3 = (wn + 3) * 8 + lane_col;
                __half2 h01 = __floats2half2_rn(a02, a03);
                __half2 h11 = __floats2half2_rn(a12, a13);
                __half2 h21 = __floats2half2_rn(a22, a23);
                __half2 h31 = __floats2half2_rn(a32, a33);
                *reinterpret_cast<uint32_t*>(&C[r1 * 128 + c0]) = reinterpret_cast<uint32_t&>(h01);
                *reinterpret_cast<uint32_t*>(&C[r1 * 128 + c1]) = reinterpret_cast<uint32_t&>(h11);
                *reinterpret_cast<uint32_t*>(&C[r1 * 128 + c2]) = reinterpret_cast<uint32_t&>(h21);
                *reinterpret_cast<uint32_t*>(&C[r1 * 128 + c3]) = reinterpret_cast<uint32_t&>(h31);
            }
        }

        __syncthreads();
    }
}

__global__ void __launch_bounds__(64, 10)
hgemm_64t_2warp(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int tiles_m)
{
    __shared__ __half smem_B[128 * 128];
    __shared__ __half smem_A[16 * 128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int idx  = tid + i * 64;
        int row  = idx >> 4;
        int col8 = idx & 15;
        int sc   = swz8(row, col8);
        uint32_t dst = __cvta_generic_to_shared(&smem_B[row * 128 + sc * 8]);
        asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
            :: "r"(dst), "l"(reinterpret_cast<const void*>(B + row * 128 + col8 * 8)));
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int wn = warp_id * 8;

    for (int tile = blockIdx.x; tile < tiles_m; tile += gridDim.x) {
        int block_m = tile * 16;

        {
            const __half* A_tile = A + block_m * 128;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx  = tid + i * 64;
                int row  = idx >> 4;
                int col8 = idx & 15;
                int sc   = swz8(row, col8);
                int gr   = block_m + row;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[row * 128 + sc * 8]);
                if (gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"(reinterpret_cast<const void*>(A_tile + row * 128 + col8 * 8)));
                } else {
                    *reinterpret_cast<float4*>(&smem_A[row * 128 + sc * 8]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            __syncthreads();
        }

        float a00=0,a01=0,a02=0,a03=0;
        float a10=0,a11=0,a12=0,a13=0;
        float a20=0,a21=0,a22=0,a23=0;
        float a30=0,a31=0,a32=0,a33=0;
        float a40=0,a41=0,a42=0,a43=0;
        float a50=0,a51=0,a52=0,a53=0;
        float a60=0,a61=0,a62=0,a63=0;
        float a70=0,a71=0,a72=0,a73=0;

        uint32_t ra0,ra1,ra2,ra3;
        uint32_t rna0,rna1,rna2,rna3;
        uint32_t rb00,rb01, rb10,rb11, rb20,rb21, rb30,rb31;
        uint32_t rb40,rb41, rb50,rb51, rb60,rb61, rb70,rb71;
        uint32_t rnb00,rnb01, rnb10,rnb11, rnb20,rnb21, rnb30,rnb31;
        uint32_t rnb40,rnb41, rnb50,rnb51, rnb60,rnb61, rnb70,rnb71;

        {
            int ar  = lane_id & 15;
            int ac  = lane_id >> 4;
            int asc = swz8(ar, ac);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[ar * 128 + asc * 8]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                :"=r"(ra0),"=r"(ra1),"=r"(ra2),"=r"(ra3):"r"(addr));
        }
        {
            int br = lane_id & 15;
            uint32_t ba0 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+0)*8]);
            uint32_t ba1 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+1)*8]);
            uint32_t ba2 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+2)*8]);
            uint32_t ba3 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+3)*8]);
            uint32_t ba4 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+4)*8]);
            uint32_t ba5 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+5)*8]);
            uint32_t ba6 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+6)*8]);
            uint32_t ba7 = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,wn+7)*8]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb00),"=r"(rb01):"r"(ba0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb10),"=r"(rb11):"r"(ba1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb20),"=r"(rb21):"r"(ba2));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb30),"=r"(rb31):"r"(ba3));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb40),"=r"(rb41):"r"(ba4));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb50),"=r"(rb51):"r"(ba5));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb60),"=r"(rb61):"r"(ba6));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rb70),"=r"(rb71):"r"(ba7));
        }

#define LDNEXT8(nkk) \
        { \
            int ar2  = lane_id & 15; \
            int ac2  = ((nkk) >> 3) + (lane_id >> 4); \
            int asc2 = swz8(ar2, ac2); \
            uint32_t aaddr = __cvta_generic_to_shared(&smem_A[ar2 * 128 + asc2 * 8]); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                :"=r"(rna0),"=r"(rna1),"=r"(rna2),"=r"(rna3):"r"(aaddr)); \
            int br2 = (nkk) + (lane_id & 15); \
            uint32_t bb0 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+0)*8]); \
            uint32_t bb1 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+1)*8]); \
            uint32_t bb2 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+2)*8]); \
            uint32_t bb3 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+3)*8]); \
            uint32_t bb4 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+4)*8]); \
            uint32_t bb5 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+5)*8]); \
            uint32_t bb6 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+6)*8]); \
            uint32_t bb7 = __cvta_generic_to_shared(&smem_B[br2*128 + swz8(br2,wn+7)*8]); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb00),"=r"(rnb01):"r"(bb0)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb10),"=r"(rnb11):"r"(bb1)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb20),"=r"(rnb21):"r"(bb2)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb30),"=r"(rnb31):"r"(bb3)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb40),"=r"(rnb41):"r"(bb4)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb50),"=r"(rnb51):"r"(bb5)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb60),"=r"(rnb61):"r"(bb6)); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(rnb70),"=r"(rnb71):"r"(bb7)); \
        }

#define DOMMA8() \
        mma_m16n8k16(a00,a01,a02,a03, ra0,ra1,ra2,ra3, rb00,rb01, a00,a01,a02,a03); \
        mma_m16n8k16(a10,a11,a12,a13, ra0,ra1,ra2,ra3, rb10,rb11, a10,a11,a12,a13); \
        mma_m16n8k16(a20,a21,a22,a23, ra0,ra1,ra2,ra3, rb20,rb21, a20,a21,a22,a23); \
        mma_m16n8k16(a30,a31,a32,a33, ra0,ra1,ra2,ra3, rb30,rb31, a30,a31,a32,a33); \
        mma_m16n8k16(a40,a41,a42,a43, ra0,ra1,ra2,ra3, rb40,rb41, a40,a41,a42,a43); \
        mma_m16n8k16(a50,a51,a52,a53, ra0,ra1,ra2,ra3, rb50,rb51, a50,a51,a52,a53); \
        mma_m16n8k16(a60,a61,a62,a63, ra0,ra1,ra2,ra3, rb60,rb61, a60,a61,a62,a63); \
        mma_m16n8k16(a70,a71,a72,a73, ra0,ra1,ra2,ra3, rb70,rb71, a70,a71,a72,a73);

#define ADVANCE8() \
        ra0=rna0; ra1=rna1; ra2=rna2; ra3=rna3; \
        rb00=rnb00; rb01=rnb01; rb10=rnb10; rb11=rnb11; \
        rb20=rnb20; rb21=rnb21; rb30=rnb30; rb31=rnb31; \
        rb40=rnb40; rb41=rnb41; rb50=rnb50; rb51=rnb51; \
        rb60=rnb60; rb61=rnb61; rb70=rnb70; rb71=rnb71;

        LDNEXT8(16)  DOMMA8() ADVANCE8()
        LDNEXT8(32)  DOMMA8() ADVANCE8()
        LDNEXT8(48)  DOMMA8() ADVANCE8()
        LDNEXT8(64)  DOMMA8() ADVANCE8()
        LDNEXT8(80)  DOMMA8() ADVANCE8()
        LDNEXT8(96)  DOMMA8() ADVANCE8()
        LDNEXT8(112) DOMMA8() ADVANCE8()
        DOMMA8()

#undef LDNEXT8
#undef DOMMA8
#undef ADVANCE8

        {
            int r0       = block_m + (lane_id >> 2);
            int r1       = r0 + 8;
            int lane_col = (lane_id & 3) * 2;

            if (r0 < M) {
                #define ST0(ni, v0, v1) { \
                    int c0 = (wn + (ni)) * 8 + lane_col; \
                    __half2 h = __floats2half2_rn(v0, v1); \
                    *reinterpret_cast<uint32_t*>(&C[r0 * 128 + c0]) = reinterpret_cast<uint32_t&>(h); }
                ST0(0,a00,a01) ST0(1,a10,a11) ST0(2,a20,a21) ST0(3,a30,a31)
                ST0(4,a40,a41) ST0(5,a50,a51) ST0(6,a60,a61) ST0(7,a70,a71)
                #undef ST0
            }
            if (r1 < M) {
                #define ST1(ni, v2, v3) { \
                    int c0 = (wn + (ni)) * 8 + lane_col; \
                    __half2 h = __floats2half2_rn(v2, v3); \
                    *reinterpret_cast<uint32_t*>(&C[r1 * 128 + c0]) = reinterpret_cast<uint32_t&>(h); }
                ST1(0,a02,a03) ST1(1,a12,a13) ST1(2,a22,a23) ST1(3,a32,a33)
                ST1(4,a42,a43) ST1(5,a52,a53) ST1(6,a62,a63) ST1(7,a72,a73)
                #undef ST1
            }
        }

        __syncthreads();
    }
}

__global__ void __launch_bounds__(64, 4)
hgemm_32x128(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int tiles_m)
{
    __shared__ __half smem_B[128 * 128];
    __shared__ __half smem_A[32 * 128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int idx  = tid + i * 64;
        int row  = idx >> 4;
        int col8 = idx & 15;
        int sc   = swz8(row, col8);
        uint32_t dst = __cvta_generic_to_shared(&smem_B[row * 128 + sc * 8]);
        asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
            :: "r"(dst), "l"(reinterpret_cast<const void*>(B + row * 128 + col8 * 8)));
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();

    const int warp_row_base = warp_id * 16;

    for (int tile = blockIdx.x; tile < tiles_m; tile += gridDim.x) {
        int block_m = tile * 32;

        {
            const __half* A_tile = A + block_m * 128;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx  = tid + i * 64;
                int row  = idx >> 4;
                int col8 = idx & 15;
                int sc   = swz8(row, col8);
                int gr   = block_m + row;
                uint32_t dst = __cvta_generic_to_shared(&smem_A[row * 128 + sc * 8]);
                if (gr < M) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                        :: "r"(dst), "l"(reinterpret_cast<const void*>(A_tile + row * 128 + col8 * 8)));
                } else {
                    *reinterpret_cast<float4*>(&smem_A[row * 128 + sc * 8]) = make_float4(0.f,0.f,0.f,0.f);
                }
            }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            __syncthreads();
        }

        float a00=0,a01=0,a02=0,a03=0;
        float a10=0,a11=0,a12=0,a13=0;
        float a20=0,a21=0,a22=0,a23=0;
        float a30=0,a31=0,a32=0,a33=0;
        float a40=0,a41=0,a42=0,a43=0;
        float a50=0,a51=0,a52=0,a53=0;
        float a60=0,a61=0,a62=0,a63=0;
        float a70=0,a71=0,a72=0,a73=0;
        float a80=0,a81=0,a82=0,a83=0;
        float a90=0,a91=0,a92=0,a93=0;
        float b00=0,b01=0,b02=0,b03=0;
        float b10=0,b11=0,b12=0,b13=0;
        float b20=0,b21=0,b22=0,b23=0;
        float b30=0,b31=0,b32=0,b33=0;
        float b40=0,b41=0,b42=0,b43=0;
        float b50=0,b51=0,b52=0,b53=0;

        uint32_t ra0,ra1,ra2,ra3;
        uint32_t rna0,rna1,rna2,rna3;
        uint32_t rb00,rb01, rb10,rb11, rb20,rb21, rb30,rb31;
        uint32_t rb40,rb41, rb50,rb51, rb60,rb61, rb70,rb71;
        uint32_t rb80,rb81, rb90,rb91, rba0,rba1, rbb0,rbb1;
        uint32_t rbc0,rbc1, rbd0,rbd1, rbe0,rbe1, rbf0,rbf1;
        uint32_t rnb00,rnb01, rnb10,rnb11, rnb20,rnb21, rnb30,rnb31;
        uint32_t rnb40,rnb41, rnb50,rnb51, rnb60,rnb61, rnb70,rnb71;
        uint32_t rnb80,rnb81, rnb90,rnb91, rnba0,rnba1, rnbb0,rnbb1;
        uint32_t rnbc0,rnbc1, rnbd0,rnbd1, rnbe0,rnbe1, rnbf0,rnbf1;

        {
            int ar  = warp_row_base + (lane_id & 15);
            int ac  = lane_id >> 4;
            int asc = swz8(ar, ac);
            uint32_t addr = __cvta_generic_to_shared(&smem_A[ar * 128 + asc * 8]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                :"=r"(ra0),"=r"(ra1),"=r"(ra2),"=r"(ra3):"r"(addr));
        }
        {
            int br = lane_id & 15;
            #define LDB(ni, r0, r1) { \
                uint32_t ba = __cvta_generic_to_shared(&smem_B[br*128 + swz8(br,ni)*8]); \
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" :"=r"(r0),"=r"(r1):"r"(ba)); }
            LDB( 0,rb00,rb01) LDB( 1,rb10,rb11) LDB( 2,rb20,rb21) LDB( 3,rb30,rb31)
            LDB( 4,rb40,rb41) LDB( 5,rb50,rb51) LDB( 6,rb60,rb61) LDB( 7,rb70,rb71)
            LDB( 8,rb80,rb81) LDB( 9,rb90,rb91) LDB(10,rba0,rba1) LDB(11,rbb0,rbb1)
            LDB(12,rbc0,rbc1) LDB(13,rbd0,rbd1) LDB(14,rbe0,rbe1) LDB(15,rbf0,rbf1)
            #undef LDB
        }

#define LDNEXT32(nkk) \
        { \
            int ar2  = warp_row_base + (lane_id & 15); \
            int ac2  = ((nkk) >> 3) + (lane_id >> 4); \
            int asc2 = swz8(ar2, ac2); \
            uint32_t aaddr = __cvta_generic_to_shared(&smem_A[ar2 * 128 + asc2 * 8]); \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                :"=r"(rna0),"=r"(rna1),"=r"(rna2),"=r"(rna3):"r"(aaddr)); \
            int br2 = (nkk) + (lane_id & 15); \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 0)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb00),"=r"(rnb01):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 1)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb10),"=r"(rnb11):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 2)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb20),"=r"(rnb21):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 3)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb30),"=r"(rnb31):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 4)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb40),"=r"(rnb41):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 5)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb50),"=r"(rnb51):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 6)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb60),"=r"(rnb61):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 7)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb70),"=r"(rnb71):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 8)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb80),"=r"(rnb81):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2, 9)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnb90),"=r"(rnb91):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2,10)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnba0),"=r"(rnba1):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2,11)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnbb0),"=r"(rnbb1):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2,12)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnbc0),"=r"(rnbc1):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2,13)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnbd0),"=r"(rnbd1):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2,14)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnbe0),"=r"(rnbe1):"r"(ba)); } \
            { uint32_t ba=__cvta_generic_to_shared(&smem_B[br2*128+swz8(br2,15)*8]); asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n":"=r"(rnbf0),"=r"(rnbf1):"r"(ba)); } \
        }

#define DOMMA32() \
        mma_m16n8k16(a00,a01,a02,a03, ra0,ra1,ra2,ra3, rb00,rb01, a00,a01,a02,a03); \
        mma_m16n8k16(a10,a11,a12,a13, ra0,ra1,ra2,ra3, rb10,rb11, a10,a11,a12,a13); \
        mma_m16n8k16(a20,a21,a22,a23, ra0,ra1,ra2,ra3, rb20,rb21, a20,a21,a22,a23); \
        mma_m16n8k16(a30,a31,a32,a33, ra0,ra1,ra2,ra3, rb30,rb31, a30,a31,a32,a33); \
        mma_m16n8k16(a40,a41,a42,a43, ra0,ra1,ra2,ra3, rb40,rb41, a40,a41,a42,a43); \
        mma_m16n8k16(a50,a51,a52,a53, ra0,ra1,ra2,ra3, rb50,rb51, a50,a51,a52,a53); \
        mma_m16n8k16(a60,a61,a62,a63, ra0,ra1,ra2,ra3, rb60,rb61, a60,a61,a62,a63); \
        mma_m16n8k16(a70,a71,a72,a73, ra0,ra1,ra2,ra3, rb70,rb71, a70,a71,a72,a73); \
        mma_m16n8k16(a80,a81,a82,a83, ra0,ra1,ra2,ra3, rb80,rb81, a80,a81,a82,a83); \
        mma_m16n8k16(a90,a91,a92,a93, ra0,ra1,ra2,ra3, rb90,rb91, a90,a91,a92,a93); \
        mma_m16n8k16(b00,b01,b02,b03, ra0,ra1,ra2,ra3, rba0,rba1, b00,b01,b02,b03); \
        mma_m16n8k16(b10,b11,b12,b13, ra0,ra1,ra2,ra3, rbb0,rbb1, b10,b11,b12,b13); \
        mma_m16n8k16(b20,b21,b22,b23, ra0,ra1,ra2,ra3, rbc0,rbc1, b20,b21,b22,b23); \
        mma_m16n8k16(b30,b31,b32,b33, ra0,ra1,ra2,ra3, rbd0,rbd1, b30,b31,b32,b33); \
        mma_m16n8k16(b40,b41,b42,b43, ra0,ra1,ra2,ra3, rbe0,rbe1, b40,b41,b42,b43); \
        mma_m16n8k16(b50,b51,b52,b53, ra0,ra1,ra2,ra3, rbf0,rbf1, b50,b51,b52,b53);

#define ADVANCE32() \
        ra0=rna0; ra1=rna1; ra2=rna2; ra3=rna3; \
        rb00=rnb00; rb01=rnb01; rb10=rnb10; rb11=rnb11; \
        rb20=rnb20; rb21=rnb21; rb30=rnb30; rb31=rnb31; \
        rb40=rnb40; rb41=rnb41; rb50=rnb50; rb51=rnb51; \
        rb60=rnb60; rb61=rnb61; rb70=rnb70; rb71=rnb71; \
        rb80=rnb80; rb81=rnb81; rb90=rnb90; rb91=rnb91; \
        rba0=rnba0; rba1=rnba1; rbb0=rnbb0; rbb1=rnbb1; \
        rbc0=rnbc0; rbc1=rnbc1; rbd0=rnbd0; rbd1=rnbd1; \
        rbe0=rnbe0; rbe1=rnbe1; rbf0=rnbf0; rbf1=rnbf1;

        LDNEXT32(16)  DOMMA32() ADVANCE32()
        LDNEXT32(32)  DOMMA32() ADVANCE32()
        LDNEXT32(48)  DOMMA32() ADVANCE32()
        LDNEXT32(64)  DOMMA32() ADVANCE32()
        LDNEXT32(80)  DOMMA32() ADVANCE32()
        LDNEXT32(96)  DOMMA32() ADVANCE32()
        LDNEXT32(112) DOMMA32() ADVANCE32()
        DOMMA32()

#undef LDNEXT32
#undef DOMMA32
#undef ADVANCE32

        {
            int r0       = block_m + warp_row_base + (lane_id >> 2);
            int r1       = r0 + 8;
            int lane_col = (lane_id & 3) * 2;
            #define ST(ni, v0, v1, v2, v3) { \
                int c0 = (ni) * 8 + lane_col; \
                __half2 h01 = __floats2half2_rn(v0, v1); \
                __half2 h23 = __floats2half2_rn(v2, v3); \
                if (r0 < M) *reinterpret_cast<uint32_t*>(&C[r0 * 128 + c0]) = reinterpret_cast<uint32_t&>(h01); \
                if (r1 < M) *reinterpret_cast<uint32_t*>(&C[r1 * 128 + c0]) = reinterpret_cast<uint32_t&>(h23); \
            }
            ST( 0, a00,a01,a02,a03) ST( 1, a10,a11,a12,a13)
            ST( 2, a20,a21,a22,a23) ST( 3, a30,a31,a32,a33)
            ST( 4, a40,a41,a42,a43) ST( 5, a50,a51,a52,a53)
            ST( 6, a60,a61,a62,a63) ST( 7, a70,a71,a72,a73)
            ST( 8, a80,a81,a82,a83) ST( 9, a90,a91,a92,a93)
            ST(10, b00,b01,b02,b03) ST(11, b10,b11,b12,b13)
            ST(12, b20,b21,b22,b23) ST(13, b30,b31,b32,b33)
            ST(14, b40,b41,b42,b43) ST(15, b50,b51,b52,b53)
            #undef ST
        }

        __syncthreads();
    }
}

static int g_num_sm = 0;
static int get_num_sm() {
    if (g_num_sm == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_num_sm, cudaDevAttrMultiProcessorCount, dev);
        if (g_num_sm == 0) g_num_sm = 132;
    }
    return g_num_sm;
}

void cuda_l2_h100_fp32(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor b_col_major, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const __half* ptr_A = reinterpret_cast<const __half*>(a.data_ptr());
    const __half* ptr_B = reinterpret_cast<const __half*>(b.data_ptr());
    __half* ptr_C = reinterpret_cast<__half*>(c.data_ptr());

    int num_sm = get_num_sm();

    if (N == 128 && K == 128) {
        int tiles_m_16 = (M + 15) / 16;
        int tiles_m_32 = (M + 31) / 32;

        int grid_128t = min(num_sm * 8, tiles_m_16);
        if (grid_128t < 1) grid_128t = 1;

        hgemm_64t_2warp<<<tiles_m_16, 64>>>(ptr_A, ptr_B, ptr_C, M, tiles_m_16);

    } else {
        int tiles_m = (M + 15) / 16;
        int grid = min(num_sm * 8, tiles_m);
        if (grid < 1) grid = 1;
        hgemm_128t_4warp<<<grid, 128>>>(ptr_A, ptr_B, ptr_C, M, tiles_m);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}