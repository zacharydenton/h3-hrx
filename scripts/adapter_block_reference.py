"""Independent CPU block oracle for native Turbo fixtures.

First run the ignored real_adapted_stacks_match_eager_and_recorded_execution
Rust test with H3_ADAPTER_BLOCK_FIXTURE=DIR, then this script with DIR. Uses the
original checkpoint/adapter tensor order; no packed GPU recipes or H3 kernels.
The full block allows 2% relative L2 for propagated INT8 rounding. A second
comparison supplies the independently checked native attention output to isolate
the remaining projections, requiring 0.3% relative L2. Both checks must pass.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

torch.set_num_threads(4)
HID, INNER, FFN = 5376, 7168, 14336
PINS = {
    'Four': ('Comfy-Org/MiniMax-H3', 'a98869194787969724c7425d95d0ed73ce9202af', 'loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors'),
    'Eight': ('lightx2v/Minimax-h3-Turbo', '3ec17a324ced54151364f24f8b5fb6bf7e26414f', 'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'),
}

def cached(repo, revision, name):
    return hf_hub_download(repo, name, revision=revision, local_files_only=True)


def half(x):
    return x.clamp(-65472, 65472).to(torch.float16).float()


def hadamard(base, n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.kron(h, base)
    assert h.shape == (n, n)
    return h


REGULAR = hadamard(torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                                 [1., -1., 1., 1.], [-1., 1., 1., 1.]]), 256) / 16
ATTENTION = hadamard(torch.tensor([[1., 1.], [1., -1.]]), 128)


def quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    return torch.round(x / scale).clamp(-127, 127), scale


def evaluate(directory, native_attention=False):
    meta = json.loads((directory / 'fixture.json').read_text())
    n, classes = meta['rows'], meta['classes']
    assert meta['rope'] == 'identity' and meta['layer'] == 0
    raw = lambda name, shape: torch.from_numpy(np.fromfile(directory / name, np.float32).reshape(shape).copy())
    x = raw('input.f32', (n, HID))
    expected = raw('output.f32', (n, HID))
    table = raw('table.f32', (classes, 2, HID))[torch.arange(n) % classes]
    gate = raw('gate.f32', (classes, HID))[torch.arange(n) % classes]
    intermediates = {}
    def compare(name, actual):
        path = directory / (name + '.f16')
        if path.exists():
            expected = torch.from_numpy(np.fromfile(path, np.float16).astype(np.float32)).reshape(actual.shape)
            intermediates[name] = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
    base_path = cached('Comfy-Org/MiniMax-H3', 'a98869194787969724c7425d95d0ed73ce9202af', 'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors')
    adapter_path = cached(*PINS[meta['preset']])
    prefix = 'token_refiner.blocks.0.' if meta['refiner'] else 'blocks.0.'
    with safe_open(base_path, framework='pt', device='cpu') as base, safe_open(adapter_path, framework='pt', device='cpu') as adapter:
        tensor = lambda name: base.get_tensor(prefix + name).float()
        def norm(x, name, modulate=True):
            y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
            y = y * tensor(name + '.weight')
            return y * (1 + table[:, 0]) + table[:, 1] if modulate else y

        def linear(x, name):
            key = 'diffusion_model.' + prefix + name
            a = adapter.get_tensor(key + '.lora_A.weight').float()
            b = adapter.get_tensor(key + '.lora_B.weight').float()
            scale = adapter.get_tensor(key + '.alpha').item() / a.shape[0]
            ranks = half(x.to(torch.bfloat16).float() @ a.T).to(torch.bfloat16).float()
            delta = half(ranks @ (b * scale).to(torch.bfloat16).float().T)
            w = tensor(name + '.weight')
            if meta['refiner']:
                result = x.to(torch.bfloat16).float() @ w.T
            else:
                rotated = (x.reshape(n, -1, 256) @ REGULAR).reshape(n, -1)
                codes, scales = quant(rotated)
                result = (codes @ w.T) * tensor(name + '.weight_scale').reshape(1, -1) * scales
            return result + delta

        qkv = half(linear(norm(x, 'norm1'), 'attn.qkv_proj')).reshape(n, 3, 56, 128)
        compare('qkv', qkv)
        q = half(norm(qkv[:, 0], 'attn.q_norm', False)).transpose(0, 1)
        k = half(norm(qkv[:, 1], 'attn.k_norm', False)).transpose(0, 1)
        v = qkv[:, 2].transpose(0, 1)
        if meta['refiner']:
            scores = (q @ k.transpose(-1, -2)) / (128 ** 0.5)
        else:
            qi, qs = quant(q @ ATTENTION)
            ki, ks = quant(k @ ATTENTION)
            scores = (qi @ ki.transpose(-1, -2)) * qs * ks.transpose(-1, -2) / (128 * 128 ** 0.5)
        probability = torch.exp(scores - scores.amax(-1, keepdim=True))
        attention = half((probability.to(torch.float16).float() @ v) / probability.sum(-1, keepdim=True)).transpose(0, 1).reshape(n, INNER)
        compare('attention', attention)
        if native_attention:
            attention = torch.from_numpy(np.fromfile(directory / 'attention.f16', np.float16).astype(np.float32)).reshape(n, INNER)
        x = x + gate * linear(attention, 'attn.out_proj')
        gu = linear(norm(x, 'norm2'), 'mlp.fc1')
        hidden = half(torch.nn.functional.silu(gu[:, :FFN]) * gu[:, FFN:])
        compare('hidden', hidden)
        actual = x + gate * linear(hidden, 'mlp.fc2')
    relative = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
    cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {**meta, 'relative_l2': relative, 'cosine': cosine, 'finite': finite,
            'intermediate_relative_l2': intermediates,
            'native_attention_input': native_attention,
            'passed': (finite and relative < (0.003 if native_attention else 0.02) and cosine > 0.999
                       and intermediates.get('qkv', float('inf')) < 0.001
                       and intermediates.get('attention', float('inf')) < 0.002),
            'reference': 'CPU FP32 matrix products with original tensor order, activation quantization and native low-rank rounding boundaries; global softmax permits online-softmax rounding differences.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    fixtures = sorted(args.directory.glob('*/fixture.json'))
    if len(fixtures) != 4:
        parser.error('expected all four DiT/refiner and four/eight-evaluation fixture combinations')
    with torch.inference_mode():
        results = [evaluate(path.parent, native_attention) for path in fixtures for native_attention in (False, True)]
    print(json.dumps(results, indent=2))
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
