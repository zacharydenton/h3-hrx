"""Measure the H3 attention layout on identical tensors, including repacking cost."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
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
          'shape': list(inputs[0].shape), 'strides': list(inputs[0].stride()),
          'method': 'Alternating layouts on identical bf16 QKV; contiguous timing includes all three copies. Flash SDPA only; one warmup per layout.',
          'seconds': {'strided': [], 'contiguous': []}}
outputs = {}
with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    for sample in range(args.samples + 1):
        for name in (['strided', 'contiguous'] if sample % 2 == 0 else ['contiguous', 'strided']):
            torch.cuda.synchronize()
            start = time.perf_counter()
            operands = inputs if name == 'strided' else [item.contiguous() for item in inputs]
            output = torch.nn.functional.scaled_dot_product_attention(*operands)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f'{name} sample {sample}: {elapsed:.6f}s', flush=True)
            if sample: result['seconds'][name].append(elapsed)
            outputs[name] = output
            del operands
    a, b = (outputs[name].float().flatten() for name in ['strided', 'contiguous'])
    result['output_cosine'] = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    result['output_max_abs_difference'] = (a - b).abs().max().item()
    result['finite'] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
result['median_seconds'] = {key: statistics.median(values) for key, values in result['seconds'].items()}
result['speedup'] = result['median_seconds']['strided'] / result['median_seconds']['contiguous']
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
