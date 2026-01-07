#!/bin/bash

# Initialize variables with default values
MNK=""
ACC_PRECISE=""
WARMUP_SECONDS=""
BENCHMARK_SECONDS=""
BASE_DIR=""
GPU_DEVICE_ID=""
MODE=""
TARGET_QPS=""


# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mnk)
            MNK="$2"
            shift 2
            ;;
        --acc_precise)
            ACC_PRECISE="$2"
            shift 2
            ;;
        --warmup_seconds)
            WARMUP_SECONDS="$2"
            shift 2
            ;;
        --benchmark_seconds)
            BENCHMARK_SECONDS="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --gpu_device_id)
            GPU_DEVICE_ID="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --target_qps)
            TARGET_QPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $2"
            exit 1
            ;;
    esac
done

echo "MNK: $MNK"
echo "ACC_PRECISE: $ACC_PRECISE"
echo "WARMUP_SECONDS: $WARMUP_SECONDS"
echo "BENCHMARK_SECONDS: $BENCHMARK_SECONDS"
echo "BASE_DIR: $BASE_DIR"
echo "GPU_DEVICE: $GPU_DEVICE_ID"

rm $BASE_DIR/benchmark*

python zero_one_correctness_check.py \
        --mnk $MNK \
        --acc_precise $ACC_PRECISE \
        --base_dir $BASE_DIR \
        --gpu_device_id $GPU_DEVICE_ID
if [ $? -ne 0 ]; then
    echo "Error: Correctness Check exception. Exiting..."
    exit 1
fi
# All possible perf_funcs
PERF_FUNCS=(
    "hgemm_cublas_tn"
    "hgemm_cublas_nn"
    "hgemm_cublaslt_heuristic_tn"
    "hgemm_cublaslt_heuristic_nn"
    "hgemm_cublaslt_auto_tuning_tn"
    "hgemm_cublaslt_auto_tuning_nn"
    "matmul"
)

echo "Executing hgemm benchmark with shuffled perf_funcs..."
shift  # Remove the first argument which is the script name

# 1. Use shuf -e to randomly sort the array and start the for loop
for func in $(shuf -e "${PERF_FUNCS[@]}"); do
    
    echo "---------------------------------------------------------"
    echo ">>> Running benchmark for: $func"
    
    # 2. Execute Python script (Note: do not use exec here)
    if [ "$MODE" == "server" ]; then
        # need to add --target_qps argument
        python benchmarking_server.py \
            --mnk $MNK \
            --acc_precise $ACC_PRECISE \
            --warmup_seconds $WARMUP_SECONDS \
            --benchmark_seconds $BENCHMARK_SECONDS \
            --base_dir $BASE_DIR \
            --gpu_device_id $GPU_DEVICE_ID \
            --perf_func "$func" \
            --target_qps $TARGET_QPS
    else
        python benchmarking_offline.py \
            --mnk $MNK \
            --acc_precise $ACC_PRECISE \
            --warmup_seconds $WARMUP_SECONDS \
            --benchmark_seconds $BENCHMARK_SECONDS \
            --base_dir $BASE_DIR \
            --gpu_device_id $GPU_DEVICE_ID \
            --perf_func "$func"
    fi

    # 3. Check the exit status of the Python script
    if [ $? -ne 0 ]; then
        echo "Error: Benchmark failed at perf_func: $func. Exiting..."
        exit 1
    fi

done

# 4. Summarize the results
python summarize_result.py --base_dir $BASE_DIR --acc_precise $ACC_PRECISE

echo "All benchmarks completed successfully!"
