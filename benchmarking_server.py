import argparse
import gc
import json
import math
import os
import random
import time

import numpy as np
import pandas
import torch

from benchmarking_utils import run_all_perf_funcs_once
from tools.utils import build_from_sources, extract_bm_bk_bn

torch.set_grad_enabled(False)


print("=====================Benchmarking Script -- Server Mode======================")

parser = argparse.ArgumentParser()
parser.add_argument("--mnk", type=str, required=True)
parser.add_argument("--acc_precise", type=str, required=True)
parser.add_argument("--warmup_seconds", type=float, required=True)
parser.add_argument("--benchmark_seconds", type=float, required=True)
parser.add_argument("--base_dir", type=str, required=True)
parser.add_argument("--gpu_device_id", type=int, required=True)
parser.add_argument("--perf_func", type=str, required=True)
parser.add_argument("--target_qps", type=float, required=True)
args = parser.parse_args()


torch.set_grad_enabled(False)


load_start = time.time()
hgemm = build_from_sources(mnk=args.mnk, acc_precise=args.acc_precise, base_dir=args.base_dir, verbose=False)
load_end = time.time()
print(f"Load hgemm module time: {load_end - load_start:.2f} seconds")

if args.acc_precise == "fp16":
    cuda_l2_func = hgemm.cuda_l2_a100_fp16  # type: ignore
    cuda_l2_func_name = "cuda_l2_a100_fp16"
    kernels_dir_name = "F16F16F16F16"
elif args.acc_precise == "fp32":
    cuda_l2_func = hgemm.cuda_l2_a100_fp32  # type: ignore
    cuda_l2_func_name = "cuda_l2_a100_fp32"
    kernels_dir_name = "F32F16F16F32"
else:
    raise ValueError


def main():
    torch.cuda.set_device(args.gpu_device_id)
    mnk = args.mnk
    m, n, k = map(int, mnk.split("_"))
    acc_precise = args.acc_precise
    warmup_seconds = args.warmup_seconds
    benchmark_seconds = args.benchmark_seconds
    torch.cuda.synchronize()

    start_time = time.time()
    print(f"m={m}, n={n}, k={k}, Warmup={warmup_seconds}s, Benchmark={benchmark_seconds}s")
    torch.cuda.synchronize()


    hgemm.init_cublas_handle()  # type: ignore
    hgemm.init_cublaslt_handle_v1() # type: ignore
    hgemm.init_cublaslt_handle_v2() # type: ignore
    torch.cuda.synchronize()

    if args.perf_func in "hgemm_cublaslt_auto_tuning_tn":
        print("Finding best algo for hgemm_cublaslt_auto_tuning_tn...")
        start_time = time.time()
        hgemm.find_best_algo_tn_v2_torch(m, n, k)  # type: ignore
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Find best algo time: {end_time - start_time:.2f} seconds")
    elif args.perf_func in "hgemm_cublaslt_auto_tuning_nn":
        print("Finding best algo for hgemm_cublaslt_auto_tuning_nn...")
        start_time = time.time()
        hgemm.find_best_algo_nn_v2_torch(m, n, k)  # type: ignore
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Find best algo time: {end_time - start_time:.2f} seconds")

    perf_func_name_to_func = {
        "hgemm_cublas_tn": hgemm.hgemm_cublas_tn,  # type: ignore
        "hgemm_cublas_nn": hgemm.hgemm_cublas_nn,  # type: ignore
        "hgemm_cublaslt_heuristic_tn": hgemm.hgemm_cublaslt_heuristic_tn,  # type: ignore
        "hgemm_cublaslt_heuristic_nn": hgemm.hgemm_cublaslt_heuristic_nn,  # type: ignore
        "hgemm_cublaslt_auto_tuning_tn": hgemm.hgemm_cublaslt_auto_tuning_tn,  # type: ignore
        "hgemm_cublaslt_auto_tuning_nn": hgemm.hgemm_cublaslt_auto_tuning_nn,  # type: ignore
        "matmul": torch.matmul,
    }

    perf_func_list = [
        perf_func_name_to_func[args.perf_func],
        cuda_l2_func,
    ]
    origin_perf_func_list = perf_func_list.copy()

    with open(f"kernels/a100_{kernels_dir_name}/{mnk}.cu", "r") as f:
        code_text = f.read()

    bm, bk, bn = extract_bm_bk_bn(code_text)

    if bm > 0 and bk > 0 and bn > 0:
        padding_m = math.ceil(m / bm) * bm - m
        padding_k = math.ceil(k / bk) * bk - k
        padding_n = math.ceil(n / bn) * bn - n
    else:
        padding_m, padding_k, padding_n = 0, 0, 0
    print(f"Using padding_m={padding_m}, padding_k={padding_k}, padding_n={padding_n}")

    # Warmup
    print("Warmup...")
    warmup_start_time = time.time()
    warmup_count = 0
    while time.time()- warmup_start_time < warmup_seconds:
        record = run_all_perf_funcs_once(
            perf_func_list=perf_func_list, m=m, n=n, k=k, acc_precise=acc_precise,
            padding_m=padding_m, padding_k=padding_k, padding_n=padding_n
        )
        warmup_count += 1
        inter_arrival = np.random.exponential(1.0 / args.target_qps)
        time.sleep(inter_arrival)
    print(f"Warmup done: {warmup_count} iterations in {time.time() - warmup_start_time:.2f} seconds.")

    print("Benchmarking...")
    records = []
    benchmark_start_time = time.time()
    benchmark_count = 0
    while time.time() - benchmark_start_time < benchmark_seconds:
        random.shuffle(perf_func_list)
        record = run_all_perf_funcs_once(
            perf_func_list=perf_func_list, m=m, n=n, k=k, acc_precise=acc_precise,
            padding_m=padding_m, padding_k=padding_k, padding_n=padding_n
        )
        record["idx"] = benchmark_count
        records.append(record)
        benchmark_count += 1
        inter_arrival = np.random.exponential(1.0 / args.target_qps)
        time.sleep(inter_arrival)

    hgemm.destroy_cublas_handle()  # type: ignore
    hgemm.destroy_cublaslt_handle_v1() # type: ignore
    hgemm.destroy_cublaslt_handle_v2() # type: ignore
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    end_time = time.time()
    print(f"Total time: {(end_time - start_time):.2f} seconds, {len(records)} records collected.")

    # Print results
    func_names = [func.__name__ for func in origin_perf_func_list]
    func_names_ms = [func.__name__ + "_ms" for func in origin_perf_func_list]
    df = pandas.DataFrame.from_records(records, columns=["idx"] + func_names + func_names_ms)
    print(df.head().to_markdown())
    print(df.tail().to_markdown())
    mean_tflops = df[func_names].mean()
    merged_result = mean_tflops.to_dict()
    mean_ms = df[func_names_ms].mean()
    merged_result["version"] = "202511261845"
    print(merged_result)
    print(mean_ms)

    our_speed = mean_tflops[cuda_l2_func_name]
    baseline_speed = mean_tflops[args.perf_func]
    print(f"speedup over {args.perf_func}: {our_speed / baseline_speed:.2f}x")
    with open(os.path.join(args.base_dir, f"benchmark_result_{args.perf_func}.json"), "w") as f:
        json.dump({"records": merged_result}, f)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
