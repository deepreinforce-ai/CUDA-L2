import argparse
import os
import time

import torch

from tools.utils import build_from_sources

TORCH_CUDA_ARCH_LIST = os.environ["TORCH_CUDA_ARCH_LIST"]
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, required=True)
parser.add_argument("--mnk", type=str, required=True)
parser.add_argument("--acc_precise", type=str, required=True)
parser.add_argument("--device_type", type=str, required=True, choices=["a100", "3090"])
args = parser.parse_args()

load_start = time.time()
hgemm = build_from_sources(mnk=args.mnk, acc_precise=args.acc_precise, device_type=args.device_type, base_dir=args.base_dir, verbose=False)
load_end = time.time()
print(f"Compile hgemm module time: {load_end - load_start:.2f} seconds")
