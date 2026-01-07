import argparse
import time

import pandas
import torch

from tools.utils import as_col_major

torch.set_grad_enabled(False)

@torch.no_grad
def run_benchmark(
    *,
    perf_func,
    a: torch.Tensor,
    b: torch.Tensor,
    b_col_major: torch.Tensor,
    out: torch.Tensor,
):
    tag = perf_func.__name__
    out.fill_(0)

    if tag != "matmul":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # type: ignore
        perf_func(a, b, b_col_major, out)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
    else:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # type: ignore
        perf_func(a, b, out=out)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
    
    return out, elapsed_time_ms


def run_all_perf_funcs_once(*, perf_func_list, m, n, k, acc_precise, padding_m, padding_k, padding_n):
    a = torch.randn((m, k), dtype=torch.half, device="cuda").cuda()
    b = torch.randn((k, n), dtype=torch.half, device="cuda").cuda()
    a_list, b_list, b_col_major_list, c_list = [], [], [], []
    for perf_func in perf_func_list:
        func_name = perf_func.__name__
        if func_name == f"cuda_l2_a100_{acc_precise}":
            a_use = torch.zeros((m+padding_m, k+padding_k), dtype=torch.half, device="cuda").cuda()
            a_use[:m, :k] = a.clone()
            b_use = torch.zeros((k+padding_k, n+padding_n), dtype=torch.half, device="cuda").cuda()
            b_use[:k, :n] = b.clone()
            b_col_major_use = as_col_major(b_use)
            c_use = torch.randn((m+padding_m, n+padding_n), dtype=torch.half, device="cuda").cuda()
        else:
            a_use = a.clone()
            b_use = b.clone()
            b_col_major_use = as_col_major(b_use)
            c_use = torch.randn((m, n), dtype=torch.half, device="cuda").cuda()
        a_list.append(a_use)
        b_list.append(b_use)
        b_col_major_list.append(b_col_major_use)
        c_list.append(c_use)
    torch.cuda.synchronize()

    record = dict()

    for i, perf_func in enumerate(perf_func_list):
        _, elapsed_time_ms = run_benchmark(
            perf_func=perf_func, a=a_list[i], b=b_list[i], b_col_major=b_col_major_list[i], out=c_list[i],
        )
        func_name = perf_func.__name__
        tflops = (2 * m * n * k) * 1e-12 * 1000 / (elapsed_time_ms)
        record[func_name] = tflops
        record[func_name+"_ms"] = elapsed_time_ms
    return record
