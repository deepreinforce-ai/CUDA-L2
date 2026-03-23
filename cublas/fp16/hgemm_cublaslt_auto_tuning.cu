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
#include <torch/torch.h>
#include <random>    // for std::random_device, std::mt19937
#include <numeric>   // for std::iota
#include <algorithm> // for std::shuffle


// ========== V2 Global Variables ==========
static cublasLtHandle_t ltHandleV2 = nullptr;
static void* workspaceV2 = nullptr;
static size_t workspace_size = 20ULL * 1024 * 1024 * 1024; // 20GB

// Cache for NN mode
static cublasLtMatmulDesc_t     nn_operationDescV2 = NULL;
static cublasLtMatrixLayout_t   nn_AdescV2 = NULL;
static cublasLtMatrixLayout_t   nn_BdescV2 = NULL;
static cublasLtMatrixLayout_t   nn_CdescV2 = NULL;
static cublasLtMatmulAlgo_t     nn_algoV2;

// Cache for TN mode
static cublasLtMatmulDesc_t     tn_operationDescV2 = NULL;
static cublasLtMatrixLayout_t   tn_AdescV2 = NULL;
static cublasLtMatrixLayout_t   tn_BdescV2 = NULL;
static cublasLtMatrixLayout_t   tn_CdescV2 = NULL;
static cublasLtMatmulAlgo_t     tn_algoV2;

// Initialization function
void init_cublaslt_handle_v2() {
    if (ltHandleV2 == nullptr) {
        cublasLtCreate(&ltHandleV2);
        cudaMalloc(&workspaceV2, workspace_size);
        // printf("[V2] Initialized cuBLASLt handle with %.2f GB workspace\n", 
        //        workspace_size / 1024.0 / 1024.0 / 1024.0);
    }
}

// Cleanup function
void destroy_cublaslt_handle_v2() {
    if (ltHandleV2 != nullptr) {
        cublasLtDestroy(ltHandleV2);
        ltHandleV2 = nullptr;
        cudaFree(workspaceV2);
        workspaceV2 = nullptr;
    }
    
    // Cleanup NN cached descriptors
    if (nn_operationDescV2) cublasLtMatmulDescDestroy(nn_operationDescV2);
    if (nn_AdescV2) cublasLtMatrixLayoutDestroy(nn_AdescV2);
    if (nn_BdescV2) cublasLtMatrixLayoutDestroy(nn_BdescV2);
    if (nn_CdescV2) cublasLtMatrixLayoutDestroy(nn_CdescV2);

    // Cleanup TN cached descriptors
    if (tn_operationDescV2) cublasLtMatmulDescDestroy(tn_operationDescV2);
    if (tn_AdescV2) cublasLtMatrixLayoutDestroy(tn_AdescV2);
    if (tn_BdescV2) cublasLtMatrixLayoutDestroy(tn_BdescV2);
    if (tn_CdescV2) cublasLtMatrixLayoutDestroy(tn_CdescV2);
    
    // printf("[V2] Destroyed cuBLASLt handle\n");
}

// ========== Helper function: Calculate median ==========
float median(std::vector<float>& times) {
    const size_t size = times.size();
    if (size == 0) {
        return 0;
    }
    
    std::sort(times.begin(), times.end());
    
    const size_t mid = size / 2;
    if (size % 2 == 0) {
        return (times[mid] + times[mid - 1]) / 2.0f;
    } else {
        return times[mid];
    }
}

// ========== PyTorch style random number generation ==========
void fill_random_half(half* data, int size, cudaStream_t stream) {
    auto options = torch::TensorOptions()
                        .dtype(torch::kHalf)
                        .device(torch::kCUDA);
    
    torch::Tensor tensor = torch::from_blob(data, {size}, options);
    tensor.normal_(0.0, 1.0);  // Standard normal distribution N(0,1)
    
    // Synchronize to specified stream (optional, as PyTorch uses current stream)
    cudaStreamSynchronize(stream);
}
#include <random>    // for std::random_device, std::mt19937
#include <numeric>   // for std::iota
#include <algorithm> // for std::shuffle


// ========== NN mode: Double warmup + shuffle + random init ==========
void find_best_algo_nn_v2(int M, int N, int K) {
    // printf("\n");
    // printf("════════════════════════════════════════════════════════════════\n");
    // printf("  [V2] Finding best algorithm for NN mode (M=%d, N=%d, K=%d)\n", M, N, K);
    // printf("════════════════════════════════════════════════════════════════\n");
    
    // 1. Allocate memory and create descriptors
    half *a, *b, *c;
    cudaMalloc(&a, M * K * sizeof(half));
    cudaMalloc(&b, K * N * sizeof(half));
    cudaMalloc(&c, M * N * sizeof(half));
    
    cublasLtMatmulDescCreate(&nn_operationDescV2, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(nn_operationDescV2, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(nn_operationDescV2, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    
    cublasLtMatrixLayoutCreate(&nn_BdescV2, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&nn_AdescV2, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&nn_CdescV2, CUDA_R_16F, N, M, N);
    
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                        &workspace_size, sizeof(workspace_size));
    
    // 2. Get candidate algorithms
    const int requestedAlgoCount = 100;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount];
    cublasLtMatmulAlgoGetHeuristic(ltHandleV2, 
                                   nn_operationDescV2,
                                   nn_BdescV2, nn_AdescV2, nn_CdescV2, nn_CdescV2,
                                   preference,
                                   requestedAlgoCount,
                                   heuristicResult,
                                   &returnedResults);

    if (returnedResults == 0) {
        cudaFree(a); cudaFree(b); cudaFree(c);
        throw std::runtime_error("[V2] No algorithm found for NN");
    }
    
    // printf("Found %d candidate algorithms, benchmarking...\n", returnedResults);
    
    // 3. Create stream and event
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    // 4. Optimization: warmup=50 rounds
    constexpr int warmupRounds = 50;     // Changed to 50
    constexpr int benchmarkRounds = 100;
    constexpr int totalRounds = warmupRounds + benchmarkRounds;
    
    std::vector<std::vector<float>> algoTimes(returnedResults, 
                                               std::vector<float>(benchmarkRounds));
    
    // printf("  - Warmup rounds: %d\n", warmupRounds);
    // printf("  - Benchmark rounds: %d\n", benchmarkRounds);
    // printf("  - Starting benchmark...\n\n");
    
    // 5. Main loop: warmup + benchmark
    for (int roundIdx = 0; roundIdx < totalRounds; roundIdx++) {
        bool isWarmup = (roundIdx < warmupRounds);
        
        // 5.1 Generate random data
        fill_random_half(a, M * K, stream);
        fill_random_half(b, K * N, stream);
        
        // 5.2 Randomly shuffle algorithm order (before warmup!)
        std::vector<int> algoIndices(returnedResults);
        std::iota(algoIndices.begin(), algoIndices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(algoIndices.begin(), algoIndices.end(), g);
        
        // 5.3 Optimization: Use the [last] algorithm after shuffle for warmup (not timed)
        int warmupAlgoIdx = algoIndices[returnedResults - 1];
        cublasLtMatmul(ltHandleV2,
                      nn_operationDescV2,
                      &alpha,
                      b, nn_BdescV2,
                      a, nn_AdescV2,
                      &beta,
                      c, nn_CdescV2,
                      c, nn_CdescV2,
                      &heuristicResult[warmupAlgoIdx].algo,  // Use the last one
                      workspaceV2,
                      workspace_size,
                      stream);
        cudaStreamSynchronize(stream);
        
        // 5.4 Test all algorithms (from first to last)
        for (int i = 0; i < returnedResults; i++) {
            int algoIdx = algoIndices[i];
            
            cudaEventRecord(start, stream);
            
            cublasLtMatmul(ltHandleV2,
                          nn_operationDescV2,
                          &alpha,
                          b, nn_BdescV2,
                          a, nn_AdescV2,
                          &beta,
                          c, nn_CdescV2,
                          c, nn_CdescV2,
                          &heuristicResult[algoIdx].algo,
                          workspaceV2,
                          workspace_size,
                          stream);
            
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            // Only record data for benchmark rounds
            if (!isWarmup) {
                int benchmarkIdx = roundIdx - warmupRounds;
                algoTimes[algoIdx][benchmarkIdx] = milliseconds;
            }
        }
        
        // Progress display
        if (isWarmup) {
            if ((roundIdx + 1) % 10 == 0 || roundIdx == 0 || roundIdx == warmupRounds - 1) {
                // printf("  Warmup round %d/%d completed\n", roundIdx + 1, warmupRounds);
            }
        } else {
            int benchmarkIdx = roundIdx - warmupRounds + 1;
            if (benchmarkIdx % 10 == 0) {
                // printf("  Benchmark progress: %d/%d rounds completed\n", 
                //        benchmarkIdx, benchmarkRounds);
            }
        }
    }
    
    // 6. Calculate median and select best algorithm
    float best_time = FLT_MAX;
    int best_algo_idx = 0;
    
    // printf("\n");
    // printf("Algorithm Performance Summary:\n");
    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
        float median_time = median(algoTimes[algoIdx]);
        
        // printf("  Algo %2d: %.4f ms (median of %d) | waves=%d, workspace=%zu KB\n", 
        //        algoIdx, median_time, benchmarkRounds,
        //        heuristicResult[algoIdx].wavesCount,
        //        heuristicResult[algoIdx].workspaceSize / 1024);
        
        if (median_time < best_time) {
            best_time = median_time;
            best_algo_idx = algoIdx;
        }
    }
    
    nn_algoV2 = heuristicResult[best_algo_idx].algo;
    
    // printf("\n");
    // printf("Best NN algorithm: index=%d, median time=%.4f ms\n", best_algo_idx, best_time);
    // printf("════════════════════════════════════════════════════════════════\n\n");
    
    // 7. Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cublasLtMatmulPreferenceDestroy(preference);
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

// ========== TN mode: Double warmup + shuffle + random init ==========
void find_best_algo_tn_v2(int M, int N, int K) {
    // printf("\n");
    // printf("════════════════════════════════════════════════════════════════\n");
    // printf("  [V2] Finding best algorithm for TN mode (M=%d, N=%d, K=%d)\n", M, N, K);
    // printf("════════════════════════════════════════════════════════════════\n");
    
    // 1. Allocate memory and create descriptors
    half *a, *b_col_major, *c;
    cudaMalloc(&a, M * K * sizeof(half));
    cudaMalloc(&b_col_major, K * N * sizeof(half));
    cudaMalloc(&c, M * N * sizeof(half));
    
    cublasLtMatmulDescCreate(&tn_operationDescV2, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(tn_operationDescV2, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(tn_operationDescV2, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    
    cublasLtMatrixLayoutCreate(&tn_BdescV2, CUDA_R_16F, K, N, K);
    cublasLtMatrixLayoutCreate(&tn_AdescV2, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&tn_CdescV2, CUDA_R_16F, N, M, N);
    
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                        &workspace_size, sizeof(workspace_size));
    
    // 2. Get candidate algorithms
    const int requestedAlgoCount = 100;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount];
    cublasLtMatmulAlgoGetHeuristic(ltHandleV2, 
                                   tn_operationDescV2,
                                   tn_BdescV2, tn_AdescV2, tn_CdescV2, tn_CdescV2,
                                   preference,
                                   requestedAlgoCount,
                                   heuristicResult,
                                   &returnedResults);

    if (returnedResults == 0) {
        cudaFree(a); cudaFree(b_col_major); cudaFree(c);
        throw std::runtime_error("[V2] No algorithm found for TN");
    }
    
    // printf("Found %d candidate algorithms, benchmarking...\n", returnedResults);
    
    // 3. Create stream and event
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    // 4. Optimization: warmup=50 rounds
    constexpr int warmupRounds = 50;
    constexpr int benchmarkRounds = 100;
    constexpr int totalRounds = warmupRounds + benchmarkRounds;
    
    std::vector<std::vector<float>> algoTimes(returnedResults, 
                                               std::vector<float>(benchmarkRounds));
    
    // printf("  - Warmup rounds: %d\n", warmupRounds);
    // printf("  - Benchmark rounds: %d\n", benchmarkRounds);
    // printf("  - Starting benchmark...\n\n");
    
    // 5. Main loop: warmup + benchmark
    for (int roundIdx = 0; roundIdx < totalRounds; roundIdx++) {
        bool isWarmup = (roundIdx < warmupRounds);
        
        // 5.1 Generate random data
        fill_random_half(a, M * K, stream);
        fill_random_half(b_col_major, K * N, stream);
        
        // 5.2 Randomly shuffle algorithm order (before warmup!)
        std::vector<int> algoIndices(returnedResults);
        std::iota(algoIndices.begin(), algoIndices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(algoIndices.begin(), algoIndices.end(), g);
        
        // 5.3 Optimization: Use the [last] algorithm after shuffle for warmup (not timed)
        int warmupAlgoIdx = algoIndices[returnedResults - 1];
        cublasLtMatmul(ltHandleV2,
                      tn_operationDescV2,
                      &alpha,
                      b_col_major, tn_BdescV2,
                      a, tn_AdescV2,
                      &beta,
                      c, tn_CdescV2,
                      c, tn_CdescV2,
                      &heuristicResult[warmupAlgoIdx].algo,  // Use the last one
                      workspaceV2,
                      workspace_size,
                      stream);
        cudaStreamSynchronize(stream);
        
        // 5.4 Test all algorithms (from first to last)
        for (int i = 0; i < returnedResults; i++) {
            int algoIdx = algoIndices[i];
            
            cudaEventRecord(start, stream);
            
            cublasLtMatmul(ltHandleV2,
                          tn_operationDescV2,
                          &alpha,
                          b_col_major, tn_BdescV2,
                          a, tn_AdescV2,
                          &beta,
                          c, tn_CdescV2,
                          c, tn_CdescV2,
                          &heuristicResult[algoIdx].algo,
                          workspaceV2,
                          workspace_size,
                          stream);
            
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            // Only record data for benchmark rounds
            if (!isWarmup) {
                int benchmarkIdx = roundIdx - warmupRounds;
                algoTimes[algoIdx][benchmarkIdx] = milliseconds;
            }
        }
        
        // Progress display
        if (isWarmup) {
            if ((roundIdx + 1) % 10 == 0 || roundIdx == 0 || roundIdx == warmupRounds - 1) {
                // printf("  Warmup round %d/%d completed\n", roundIdx + 1, warmupRounds);
            }
        } else {
            int benchmarkIdx = roundIdx - warmupRounds + 1;
            if (benchmarkIdx % 10 == 0) {
                // printf("  Benchmark progress: %d/%d rounds completed\n", 
                //        benchmarkIdx, benchmarkRounds);
            }
        }
    }
    
    // 6. Calculate median and select best algorithm
    float best_time = FLT_MAX;
    int best_algo_idx = 0;
    
    // printf("\n");
    // printf("Algorithm Performance Summary:\n");
    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
        float median_time = median(algoTimes[algoIdx]);
        
        // printf("  Algo %2d: %.4f ms (median of %d) | waves=%d, workspace=%zu KB\n", 
        //        algoIdx, median_time, benchmarkRounds,
        //        heuristicResult[algoIdx].wavesCount,
        //        heuristicResult[algoIdx].workspaceSize / 1024);
        
        if (median_time < best_time) {
            best_time = median_time;
            best_algo_idx = algoIdx;
        }
    }
    
    tn_algoV2 = heuristicResult[best_algo_idx].algo;
    
    // printf("\n");
    // printf("Best TN algorithm: index=%d, median time=%.4f ms\n", best_algo_idx, best_time);
    // printf("════════════════════════════════════════════════════════════════\n\n");
    
    // 7. Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cublasLtMatmulPreferenceDestroy(preference);
    
    cudaFree(a);
    cudaFree(b_col_major);
    cudaFree(c);
}


// ========== Simplified NN calculation function (assuming find has been called) ==========
void cublaslt_tensor_op_nn_v2(half *a, half *b, half *c, int M, int N, int K) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    cublasLtMatmul(ltHandleV2,
                   nn_operationDescV2,
                   &alpha,
                   b, nn_BdescV2,
                   a, nn_AdescV2,
                   &beta,
                   c, nn_CdescV2,
                   c, nn_CdescV2,
                   &nn_algoV2,
                   workspaceV2,
                   workspace_size,
                   0);
}

// ========== Simplified TN calculation function (assuming find has been called) ==========
void cublaslt_tensor_op_tn_v2(half *a, half *b_col_major, half *c, int M, int N, int K) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    cublasLtMatmul(ltHandleV2,
                   tn_operationDescV2,
                   &alpha,
                   b_col_major, tn_BdescV2,
                   a, tn_AdescV2,
                   &beta,
                   c, tn_CdescV2,
                   c, tn_CdescV2,
                   &tn_algoV2,
                   workspaceV2,
                   workspace_size,
                   0);
}

// ========== PyTorch Bindings ==========
#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) \
  if (((T).options().dtype() != (th_type))) { \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type); \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1) \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
    throw std::runtime_error("Tensor size mismatch!"); \
  }

void find_best_algo_nn_v2_torch(int M, int N, int K) {
    find_best_algo_nn_v2(M, N, K);
}

void find_best_algo_tn_v2_torch(int M, int N, int K) {
    find_best_algo_tn_v2(M, N, K);
}

// NN: A/B/C All row major
void hgemm_cublaslt_auto_tuning_nn(torch::Tensor a, torch::Tensor b, 
                                    torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublaslt_tensor_op_nn_v2(reinterpret_cast<half *>(a.data_ptr()),
                           reinterpret_cast<half *>(b.data_ptr()),
                           reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}

// TN: A row major MxK, B col major KxN, C row major MxN
void hgemm_cublaslt_auto_tuning_tn(torch::Tensor a, torch::Tensor b,
                                    torch::Tensor b_col_major, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b_col_major, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b_col_major.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b_col_major, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublaslt_tensor_op_tn_v2(reinterpret_cast<half *>(a.data_ptr()),
                           reinterpret_cast<half *>(b_col_major.data_ptr()),
                           reinterpret_cast<half *>(c.data_ptr()), M, N, K);
}
