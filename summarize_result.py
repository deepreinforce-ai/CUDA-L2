import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, required=True)
args = parser.parse_args()


def summarize_results():
    base_dir = Path(args.base_dir)
    result_files = list(base_dir.glob(f"benchmark_result_*.json"))
    
    name_to_data = {}
    for file in result_files:
        original_method_name = file.stem.replace("benchmark_result_", "")
        if original_method_name == "matmul":
            show_name = "torch.matmul"
        else:
            show_name = original_method_name.replace("hgemm_", "").replace("cublaslt", "cuBLASLt").replace("cublas", "cuBLAS").replace("_", "-")

        with open(file, "r") as f:
            json_data = json.load(f)
            name_to_data[show_name] = {
                "Baseline Method Name": show_name,
                "Baseline TFLOPS": json_data["records"][original_method_name],
                "CUDA-L2 TFLOPS": json_data["records"]["cuda_l2_a100_fp16"],
                "Speedup": json_data["records"]["cuda_l2_a100_fp16"] / json_data["records"][original_method_name],
            }
    print(name_to_data)
    for name in ["cuBLAS", "cuBLASLt-heuristic", "cuBLASLt-auto-tuning"]:
        if name_to_data[f"{name}-tn"]["Speedup"] < name_to_data[f"{name}-nn"]["Speedup"]:
            postfix = "tn"
        else:
            postfix = "nn"
        name_to_data[f"{name}-max"] = {
            "Baseline Method Name": f"{name}-max",
            "Baseline TFLOPS": name_to_data[f"{name}-{postfix}"]["Baseline TFLOPS"],
            "CUDA-L2 TFLOPS": name_to_data[f"{name}-{postfix}"]["CUDA-L2 TFLOPS"],
            "Speedup": name_to_data[f"{name}-{postfix}"]["Speedup"],
        }
    
    name_order = [
        "torch.matmul",
        "cuBLAS-tn",
        "cuBLAS-nn",
        "cuBLAS-max",
        "cuBLASLt-heuristic-tn",
        "cuBLASLt-heuristic-nn",
        "cuBLASLt-heuristic-max",
        "cuBLASLt-auto-tuning-tn",
        "cuBLASLt-auto-tuning-nn",
        "cuBLASLt-auto-tuning-max",
    ]

    data = []
    for name in name_order:
        record = name_to_data[name]
        data.append(record)

    df = pandas.DataFrame.from_records(data)
    # df["Baseline TFLOPS"] = df["Baseline TFLOPS"].astype(float)
    # df["CUDA-L2 TFLOPS"] = df["CUDA-L2 TFLOPS"].astype(float)


    print("Summary of Benchmark Results:")
    print(df.to_markdown(floatfmt=".3f", missingval="-"))

    
if __name__ == "__main__":
    summarize_results()
