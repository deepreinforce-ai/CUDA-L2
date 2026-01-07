import os
import re

import torch
from torch.utils.cpp_extension import load


def extract_bm_bk_bn(text: str) -> tuple[int, int, int]:
    """
    Scans the code to find the values of BM, BK, and BN.
    """
    bm, bk, bn = extract_bm_bk_bk_use_rules(text)
    if bm > 0 and bk > 0 and bn > 0:
        return bm, bk, bn
    return -1, -1, -1


def extract_bm_bk_bk_use_rules(text: str) -> tuple[int, int, int]:
    """
    Scans the code to find the values of BM, BK, and BN, and the rules used.
    """
    bm, bk, bn = -1, -1, -1
    pattern = r"(BM|BN|BK)\s*=\s*Int<(\d+)>"
    for lines in text.split("\n"):
        line = lines.strip().replace(" ", "")
        match = re.search(pattern, line)
        if not match:
            continue
        var_name, value = match.groups()
        if var_name == "BM":
            bm = int(value)
        elif var_name == "BN":
            bn = int(value)
        elif var_name == "BK":
            bk = int(value)
    return bm, bk, bn


def get_build_sources(mnk, acc_precise):
    if acc_precise == "fp16":
        dir_name = "F16F16F16F16"
    elif acc_precise == "fp32":
        dir_name = "F32F16F16F32"
    else:
        raise ValueError
    build_sources = [
        "cublas/hgemm_cublas.cu",
        "cublas/hgemm_cublaslt_heuristic.cu",
        "cublas/hgemm_cublaslt_auto_tuning.cu",
        f"kernels/a100_{dir_name}/{mnk}.cu",
        f"pybind/hgemm_a100_{acc_precise}.cc",
    ]

    return build_sources


def get_build_cuda_cflags(build_pkg: bool = False):
    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]
    if not build_pkg:
        extra_cuda_cflags += ["-diag-suppress 177", "-Xptxas -v"]
    else:
        extra_cuda_cflags += ["--ptxas-options=-v", "--ptxas-options=-O3"]
    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    extra_cuda_cflags += [
        "-DNO_MMA_HGEMM_BIN",
        "-DNO_WMMA_HGEMM_BIN",
        "-DNO_CUTE_HGEMM_BIN",
        "-DNO_CUBLAS_HGEMM_BIN",
        # add cutlass headers and link cublas.
        f"-I {project_dir}/utils",
        f"-I {project_dir}/cublas",
        f"-I {project_dir}/pybind",
    ]
    cutlass_dir = os.environ["CUTLASS_DIR"]
    extra_cuda_cflags += [
        f"-I {cutlass_dir}/include",
        f"-I {cutlass_dir}/tools/util/include",
        "-lcublas",
    ]
    return extra_cuda_cflags


def build_from_sources(mnk, acc_precise, base_dir: str, verbose: bool):
    torch_arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    device_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    print(f"Loading hgemm lib on device: {device_name} :: {device_capability} :: {torch_arch_list_env}")
    return load(
        name="hgemm_lib",
        sources=get_build_sources(mnk, acc_precise),
        extra_cuda_cflags=get_build_cuda_cflags(),
        extra_cflags=["-std=c++17", "-fuse-ld=lld"],
        verbose=verbose,
        build_directory=base_dir
    )


@torch.no_grad
def as_col_major(x: torch.Tensor):
    # convert a row major tensor -> col major with contiguous storage
    x_trans = x.t()
    x_col_major = x_trans.reshape(x.shape)
    return x_col_major.contiguous()  # must be a contiguous tensor
