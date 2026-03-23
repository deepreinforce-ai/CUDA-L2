#include "cublas_v2.h"
#include <cublasLt.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdexcept>

// Global or class member variables, created at initialization, destroyed at program end
static cublasLtHandle_t ltHandleV1 = nullptr;
static void* workspaceV1 = nullptr;
static size_t workspace_size = 20 * 1024 * 1024 * 1024; // 20GB

// ========== Added: Cache for NN mode ==========
static cublasLtMatmulDesc_t     nn_operationDescV1 = NULL;
static cublasLtMatrixLayout_t   nn_AdescV1 = NULL;
static cublasLtMatrixLayout_t   nn_BdescV1 = NULL;
static cublasLtMatrixLayout_t   nn_CdescV1 = NULL;
static cublasLtMatmulAlgo_t     nn_algoV1; // Store found algorithm

// ========== Added: Cache for TN mode ==========
static cublasLtMatmulDesc_t     tn_operationDescV1 = NULL;
static cublasLtMatrixLayout_t   tn_AdescV1 = NULL;
static cublasLtMatrixLayout_t   tn_BdescV1 = NULL;
static cublasLtMatrixLayout_t   tn_CdescV1 = NULL;
static cublasLtMatmulAlgo_t     tn_algoV1; // Store found algorithm

// Initialization function (called once at program start)
void init_cublaslt_handle_v1() {
    if (ltHandleV1 == nullptr) {
        cublasLtCreate(&ltHandleV1);
        cudaMalloc(&workspaceV1, workspace_size);
    }
}

// Cleanup function (called at program end)
void destroy_cublaslt_handle_v1() {
    if (ltHandleV1 != nullptr) {
        cublasLtDestroy(ltHandleV1);
        ltHandleV1 = nullptr;
        cudaFree(workspaceV1);
        workspaceV1 = nullptr;
    }
    // ========== Added: Cleanup NN cached descriptors ==========
    if (nn_operationDescV1) cublasLtMatmulDescDestroy(nn_operationDescV1);
    if (nn_AdescV1) cublasLtMatrixLayoutDestroy(nn_AdescV1);
    if (nn_BdescV1) cublasLtMatrixLayoutDestroy(nn_BdescV1);
    if (nn_CdescV1) cublasLtMatrixLayoutDestroy(nn_CdescV1);

    // ========== Added: Cleanup TN cached descriptors ==========
    if (tn_operationDescV1) cublasLtMatmulDescDestroy(tn_operationDescV1);
    if (tn_AdescV1) cublasLtMatrixLayoutDestroy(tn_AdescV1);
    if (tn_BdescV1) cublasLtMatrixLayoutDestroy(tn_BdescV1);
    if (tn_CdescV1) cublasLtMatrixLayoutDestroy(tn_CdescV1);
}

// NN: A/B/C All row major
// Row-major matrices need special handling in cuBLAS (column-major) perspective
void cublaslt_tensor_op_nn_v1(half *a, half *b, half *c, 
                           int M, int N, int K) {

    // Step 1: Check if initialized (lazy singleton)
    // We only check one descriptor; if it's not created, we assume all need to be created.
    if (nn_operationDescV1 == NULL) {
        
        // --- First run: Create, search, and cache descriptors and algorithms ---
        
        cublasLtMatmulPreference_t preference = NULL; // This is a temporary variable, destroyed after use
        
        // 1. Create operation descriptor and cache
        cublasLtMatmulDescCreate(&nn_operationDescV1, CUBLAS_COMPUTE_16F, CUDA_R_16F);
        
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(nn_operationDescV1, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        cublasLtMatmulDescSetAttribute(nn_operationDescV1, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
        
        // 2. Create matrix layout and cache
        // Note cuBLAS (B, A) order
        cublasLtMatrixLayoutCreate(&nn_BdescV1, CUDA_R_16F, N, K, N);
        cublasLtMatrixLayoutCreate(&nn_AdescV1, CUDA_R_16F, K, M, K);
        cublasLtMatrixLayoutCreate(&nn_CdescV1, CUDA_R_16F, N, M, N);
        
        // 3. Create preference (for search)
        cublasLtMatmulPreferenceCreate(&preference);
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));
        
        // 4. Get the best algorithm
        int returnedResults = 0;
        cublasLtMatmulHeuristicResult_t heuristicResult[4];
        cublasLtMatmulAlgoGetHeuristic(ltHandleV1, 
                                       nn_operationDescV1, // Use global variable
                                       nn_BdescV1, nn_AdescV1, nn_CdescV1, nn_CdescV1, // Use global variable
                                       preference,
                                       4,  // Want 4 best
                                       heuristicResult,
                                       &returnedResults);

        if (returnedResults == 0) {
            // If no algorithm is found with fixed M, N, K, this is a serious error and should abort.
            throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic failed to find an algorithm for NN op.");
        }

        // 5. Cache the best algorithm
        nn_algoV1 = heuristicResult[0].algo;
        
        // 6. Cleanup temporary preference object
        cublasLtMatmulPreferenceDestroy(preference);
    }

    // --- Step 2: Execute calculation ---
    // Whether first call or not, use cached descriptors and algorithms for calculation

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    cublasLtMatmul(ltHandleV1,
                   nn_operationDescV1, // Use cached descriptor
                   &alpha,
                   b, nn_BdescV1,     // Use cached layout
                   a, nn_AdescV1,     // Use cached layout
                   &beta,
                   c, nn_CdescV1,     // Use cached layout
                   c, nn_CdescV1,     // Use cached layout
                   &nn_algoV1,         // Use cached algorithm
                   workspaceV1,
                   workspace_size,
                   0);
    
    // --- Step 3: Remove cleanup ---
    // All original Destroy... calls have been removed
}

// TN: A row major MxK, B col major KxN, C row major MxN  
void cublaslt_tensor_op_tn_v1(half *a, half *b_col_major, half *c,
                           int M, int N, int K) {
    
    // Step 1: Check if initialized (lazy singleton)
    if (tn_operationDescV1 == NULL) {
        
        // --- First run: Create, search, and cache descriptors and algorithms ---
        
        cublasLtMatmulPreference_t preference = NULL; // Temporary variable, destroyed after use
        
        // 1. Create operation descriptor and cache
        cublasLtMatmulDescCreate(&tn_operationDescV1, CUBLAS_COMPUTE_16F, CUDA_R_16F);
        
        // For TN mode, we calculate C^T = B^T * A
        cublasOperation_t transa = CUBLAS_OP_T;  // B needs transpose
        cublasOperation_t transb = CUBLAS_OP_N;  // A does not need transpose
        cublasLtMatmulDescSetAttribute(tn_operationDescV1, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        cublasLtMatmulDescSetAttribute(tn_operationDescV1, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
        
        // 2. Create matrix layout and cache
        cublasLtMatrixLayoutCreate(&tn_BdescV1, CUDA_R_16F, K, N, K);  // Column major, leading dimension is K
        cublasLtMatrixLayoutCreate(&tn_AdescV1, CUDA_R_16F, K, M, K);  // leading dimension is K
        cublasLtMatrixLayoutCreate(&tn_CdescV1, CUDA_R_16F, N, M, N);  // leading dimension is N
        
        // 3. Create preference (for search)
        cublasLtMatmulPreferenceCreate(&preference);
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));
        
        // 4. Get the best algorithm
        int returnedResults = 0;
        cublasLtMatmulHeuristicResult_t heuristicResult[4];
        cublasLtMatmulAlgoGetHeuristic(ltHandleV1, 
                                       tn_operationDescV1,  // Use global variable
                                       tn_BdescV1, tn_AdescV1, tn_CdescV1, tn_CdescV1,  // Use global variable
                                       preference,
                                       4,  // Want 4 best
                                       heuristicResult,
                                       &returnedResults);

        if (returnedResults == 0) {
            // If no algorithm is found with fixed M, N, K, this is a serious error and should abort.
            throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic failed to find an algorithm for TN op.");
        }

        // 5. Cache the best algorithm
        tn_algoV1 = heuristicResult[0].algo;
        
        // 6. Cleanup temporary preference object
        cublasLtMatmulPreferenceDestroy(preference);
    }

    // --- Step 2: Execute calculation ---
    // Whether first call or not, use cached descriptors and algorithms for calculation

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    cublasLtMatmul(ltHandleV1,
                   tn_operationDescV1,  // Use cached descriptor
                   &alpha,
                   b_col_major,       // B matrix (column major)
                   tn_BdescV1,          // Use cached layout
                   a,                 // A matrix (row major)
                   tn_AdescV1,          // Use cached layout
                   &beta,
                   c,
                   tn_CdescV1,          // Use cached layout
                   c,
                   tn_CdescV1,          // Use cached layout
                   &tn_algoV1,          // Use cached algorithm
                   workspaceV1,
                   workspace_size,
                   0);
    
    // --- Step 3: Remove cleanup ---
    // All original Destroy... calls have been removed
}

#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// NN: A/B/C All row major
void hgemm_cublaslt_heuristic_nn(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major,
                               torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublaslt_tensor_op_nn_v1(reinterpret_cast<half *>(a.data_ptr()),
                      reinterpret_cast<half *>(b.data_ptr()),
                      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// TN: A row major MxK, B col major KxN, C row major MxN
void hgemm_cublaslt_heuristic_tn(torch::Tensor a, torch::Tensor b,
                               torch::Tensor b_col_major,
                               torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b_col_major.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b_col_major, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublaslt_tensor_op_tn_v1(reinterpret_cast<half *>(a.data_ptr()),
                      reinterpret_cast<half *>(b_col_major.data_ptr()),
                      reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}
