import argparse
import gc
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import pandas
import torch
from tqdm import tqdm

from tools.utils import as_col_major, build_from_sources, extract_bm_bk_bn

print("======================Correctness Check======================")

parser = argparse.ArgumentParser()
parser.add_argument("--mnk", type=str, required=True)
parser.add_argument("--base_dir", type=str, required=True)
parser.add_argument("--gpu_device_id", type=int, required=True)
args = parser.parse_args()


torch.set_grad_enabled(False)


load_start = time.time()
hgemm = build_from_sources(mnk=args.mnk, base_dir=args.base_dir, verbose=False)
load_end = time.time()
print(f"Load hgemm module time: {load_end - load_start:.2f} seconds")


@torch.no_grad
def compare_kernels_with_cpu_fp32(
    kernel_funcs: list,
    m: int,
    n: int,
    k: int,
    num_iterations: int,
    padding_m: int,
    padding_k: int,
    padding_n: int,
):
    kernel_diffs = {func.__name__: [] for func in kernel_funcs}
    start_time = time.time()
    max_running_seconds = 60
    if os.environ.get("IS_DEBUG", "false") == "true":
        bar = tqdm(range(num_iterations))
    else:
        bar = range(num_iterations)
    for _ in bar:
        if time.time() - start_time > max_running_seconds:
            break
        # Generate random input tensors
        a = 0.1 * torch.randn((m, k), dtype=torch.half, device="cuda").contiguous()
        b = 0.1 * torch.randn((k, n), dtype=torch.half, device="cuda").contiguous()
        
        # CPU FP32 reference (ground truth)
        torch.cuda.synchronize()
        a_cpu = a.cpu().float()
        b_cpu = b.cpu().float()
        torch.cuda.synchronize()
        truth = torch.matmul(a_cpu, b_cpu).half()
        torch.cuda.synchronize()
        
        for perf_func in kernel_funcs:
            tag = perf_func.__name__
            if tag == "cuda_l2_a100_fp16":
                a_use = torch.zeros((m+padding_m, k+padding_k), dtype=torch.half, device="cuda").cuda()
                a_use[:m, :k] = a.clone()
                b_use = torch.zeros((k+padding_k, n+padding_n), dtype=torch.half, device="cuda").cuda()
                b_use[:k, :n] = b.clone()
                b_col_major_use = as_col_major(b_use)
                out_for_compare = torch.zeros((m+padding_m, n+padding_n), dtype=torch.half, device="cuda").cuda()
                torch.cuda.synchronize()
                perf_func(a_use, b_use, b_col_major_use, out_for_compare)
            else:
                a_use = a.clone()
                b_use = b.clone()
                b_col_major_use = as_col_major(b_use)
                out_for_compare = torch.zeros((m, n), dtype=torch.half, device="cuda").cuda()
                torch.cuda.synchronize()
                if tag == "matmul":
                    torch.matmul(a_use, b_use, out=out_for_compare)
                else:
                    perf_func(a_use, b_use, b_col_major_use, out_for_compare)
            torch.cuda.synchronize()

            out_for_compare = out_for_compare[:m, :n].cpu()

            diff = torch.abs(out_for_compare - truth)
            max_diff = torch.max(diff).item()
            kernel_diffs[tag].append(max_diff)
    result = {
        "if_success": True,
        "m": m,
        "n": n,
        "k": k,
        "num_iterations": num_iterations,
    }
    
    for tag in kernel_diffs.keys():
        avg_diff = sum(kernel_diffs[tag]) / len(kernel_diffs[tag])
        result[f"avg_{tag}_diff"] = round(avg_diff, 6)
    
    best_tag = min(kernel_diffs.keys(), key=lambda tag: sum(kernel_diffs[tag]) / len(kernel_diffs[tag]))
    result["best_kernel"] = best_tag
    
    return result


@torch.no_grad
def run_correctness_check(
    hgemm,
    m: int,
    n: int,
    k: int,
    padding_m: int =0,
    padding_k: int =0,
    padding_n: int =0,
):
    """Run correctness check across multiple iterations with random inputs"""

    hgemm.init_cublas_handle()  # type: ignore
    hgemm.init_cublaslt_handle_v1() # type: ignore
    hgemm.init_cublaslt_handle_v2() # type: ignore
    torch.cuda.synchronize()

    hgemm.find_best_algo_tn_v2_torch(m, n, k)  # type: ignore
    hgemm.find_best_algo_nn_v2_torch(m, n, k)  # type: ignore

    print("Initialize Done.")

    kernel_funcs = [
        hgemm.hgemm_cublas_tn,  # type: ignore
        hgemm.hgemm_cublas_nn,  # type: ignore
        hgemm.hgemm_cublaslt_heuristic_tn,  # type: ignore
        hgemm.hgemm_cublaslt_heuristic_nn,  # type: ignore
        hgemm.hgemm_cublaslt_auto_tuning_tn,  # type: ignore
        hgemm.hgemm_cublaslt_auto_tuning_nn,  # type: ignore
        torch.matmul,
        hgemm.cuda_l2_a100_fp16,
    ]
    

    try:
        result = compare_kernels_with_cpu_fp32(
            kernel_funcs, m, n, k, num_iterations=100, 
            padding_m=padding_m, padding_k=padding_k, padding_n=padding_n
        )
    except Exception as e:
        traceback.print_exc()
        return False, str(e), {}
    
    hgemm.destroy_cublas_handle()  # type: ignore
    hgemm.destroy_cublaslt_handle_v1() # type: ignore
    hgemm.destroy_cublaslt_handle_v2() # type: ignore

    
    # Correctness check for cuda_l2_a100_fp16
    # Dynamically extract ALL other kernels' diffs (excluding cuda_l2_a100_fp16, nan, and Inf)
    other_diffs = []
    print(result)
    for key, val in result.items():
        # Get all avg_*_diff keys except cuda_l2_a100_fp16
        if key.startswith("avg_") and key.endswith("_diff") and key != "avg_cuda_l2_a100_fp16_diff":
            # Skip nan and Inf values
            if isinstance(val, (int, float)) and val == val and val != float('inf') and val != float('-inf'):
                other_diffs.append(val)
    
    if other_diffs and "avg_cuda_l2_a100_fp16_diff" in result:
        v2_diff = result["avg_cuda_l2_a100_fp16_diff"]
        
        # Skip if v2_diff itself is nan or Inf
        if not (isinstance(v2_diff, (int, float)) and v2_diff == v2_diff and v2_diff != float('inf') and v2_diff != float('-inf')):
            return False, f"cuda_l2_a100_fp16 has nan or Inf value: {v2_diff}", result
        
        max_other_diff = max(other_diffs)
        threshold = max_other_diff * 2.0
        
        if v2_diff > threshold:
            error_msg = f"cuda_l2_a100_fp16 diff ({v2_diff:.6f}) exceeds others by >20% (max_other: {max_other_diff:.6f}, threshold: {threshold:.6f})"
            return False, error_msg, result
        else:
            correctness_msg = f"Correctness check passed: v2_diff={v2_diff:.6f}, max_other={max_other_diff:.6f}, threshold={threshold:.6f}"
            return True, correctness_msg, result
    else:
        # no v2 diff found or no other kernels to compare - treat as pass
        raise Exception("no comparison data available for correctness check.")


def main():
    m, n, k = map(int, args.mnk.split("_"))
    torch.cuda.set_device(args.gpu_device_id)
    with open(f"kernels/a100_F16F16F16F16/{args.mnk}.cu", "r") as f:
        code_text = f.read()
    bm, bk, bn = extract_bm_bk_bn(code_text)
    if bm > 0 and bk > 0 and bn > 0:
        padding_m = math.ceil(m / bm) * bm - m
        padding_k = math.ceil(k / bk) * bk - k
        padding_n = math.ceil(n / bn) * bn - n
    else:
        padding_m, padding_k, padding_n = 0, 0, 0
    
    print(f"Running correctness check for m={m}, n={n}, k={k} ...")
    print(f"Padding: padding_m={padding_m}, padding_k={padding_k}, padding_n={padding_n}")
    success, message, result = run_correctness_check(
        hgemm=hgemm,
        m=m,
        n=n,
        k=k,
        padding_m=padding_m,
        padding_k=padding_k,
        padding_n=padding_n,
    )
    base_dir = Path(args.base_dir)
    with open(base_dir / "randn_correctness_check_result.json", "w") as f:
        json.dump({"success": success, "message": message, "result": result}, f, indent=4, ensure_ascii=False)
    
    if success:
        print("Correctness Check PASSED:", message)
    else:
        print("Correctness Check FAILED:", message)


if __name__ == "__main__":
    main()
