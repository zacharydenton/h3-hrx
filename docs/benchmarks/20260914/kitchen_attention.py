"""Measure Comfy Kitchen INT8 attention at a representative 768p H3 shape."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
import comfy_kitchen
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokens', type=int, default=37754)
parser.add_argument('--samples', type=int, default=3)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(1618)
heads, dim = 56, 128
qkv = torch.randn(args.tokens, 3 * heads * dim, device='cuda', dtype=torch.bfloat16)
inputs = [part.view(args.tokens, heads, dim).transpose(0, 1).unsqueeze(0)
          for part in qkv.chunk(3, dim=-1)]
result = {'torch': torch.__version__, 'hip': torch.version.hip,
          'shape': list(inputs[0].shape), 'seconds': [],
          'method': 'Strided bf16 QKV inputs; Comfy Kitchen INT8 including quantization. One warmup; compare against contiguous flash SDPA on identical inputs.'}
with torch.inference_mode():
    for sample in range(args.samples + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = comfy_kitchen.int8_attention(*inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'kitchen sample {sample}: {elapsed:.6f}s', flush=True)
        if sample:
            result['seconds'].append(elapsed)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        reference = torch.nn.functional.scaled_dot_product_attention(*[item.contiguous() for item in inputs])
    a, b = output.float().flatten(), reference.float().flatten()
    result['cosine'] = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    result['max_abs_difference'] = (a - b).abs().max().item()
    result['finite'] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
result['median_seconds'] = statistics.median(result['seconds'])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
