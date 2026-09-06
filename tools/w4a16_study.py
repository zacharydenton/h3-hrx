"""Paired weight-only INT4 accuracy study, streaming one block at a time.

The GPTQ cases use the exact exported codes and scales. A16 means BF16 linear
operands, as in the original Torch reference; attention operands are FP16.
This emulates quantized weights with dequantized GEMMs, not a native speed test.

    python tools/w4a16_study.py --out build/w4a16/fixture.json
    python tools/w4a16_study.py --comfy-truth build/comfy_t2va_blocks --out build/w4a16/prompt.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import h3_ref as R


def unpack_i4(packed):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).to(torch.int8)
    return torch.where(codes >= 8, codes - 16, codes)


def undo_gate_interleave(x):
    return x.reshape(-1, 2, 16, *x.shape[1:]).transpose(0, 1).reshape(x.shape)


class GPTQWeights:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.spans = {}
        for line in (self.directory / "manifest.txt").read_text().splitlines():
            name, offset, size, dtype, shape = line.split()
            self.spans[name] = (int(offset), int(size), dtype, tuple(map(int, shape.split("x"))))

    def tensor(self, name, device):
        offset, size, dtype, shape = self.spans[name]
        dt = {"torch.uint8": np.uint8, "torch.float32": np.float32}[dtype]
        with (self.directory / "weights.bin").open("rb") as f:
            f.seek(offset)
            data = f.read(size)
        if len(data) != size:
            raise ValueError(f"short read of {name}")
        return torch.from_numpy(np.frombuffer(data, dtype=dt).copy().reshape(shape)).to(device)

    def linear(self, name, device, dtype):
        tags = {"attn.qkv_proj": "qkv", "attn.out_proj": "out", "mlp.fc1": "gu", "mlp.fc2": "down"}
        prefix, layer, suffix = name.split(".", 2)
        tag = tags[suffix]
        stem = f"{prefix}.{layer}.{tag}"
        q = unpack_i4(self.tensor(stem + ".q", device))
        scales = self.tensor(stem + ".s", device)
        w = (q.float() * scales[:, None]).to(dtype)
        return undo_gate_interleave(w) if tag == "gu" else w


class StudyRef(R.H3Ref):
    def __init__(self, ckpt, mode, weights, attention_dtype):
        super().__init__(ckpt, quant="none" if mode.startswith("gptq_") else mode, cache_linears=False)
        self.mode, self.weights, self.attention_dtype = mode, weights, attention_dtype

    def lin(self, name):
        if not self.mode.startswith("gptq_"):
            return super().lin(name)
        if name not in self._lin:
            linear = R.QuantLinear.__new__(R.QuantLinear)
            linear.h, linear.quant = self.h, "gptq"
            linear.aq = "4" if self.mode == "gptq_a4" else "16"
            linear.w = self.weights.linear(name, self.device, self.dtype)
            self._lin[name] = linear
        return self._lin[name]

    def sdpa(self, q, k, v, layer=0):
        # Query chunking bounds even the math fallback; every chunk sees all keys.
        kt = k.transpose(0, 1)[None].to(self.attention_dtype)
        vt = v.transpose(0, 1)[None].to(self.attention_dtype)
        out = torch.empty_like(v)
        for start in range(0, len(q), 128):
            qt = q[start:start + 128].transpose(0, 1)[None].to(self.attention_dtype)
            out[start:start + 128] = F.scaled_dot_product_attention(qt, kt, vt)[0].transpose(0, 1).to(v.dtype)
        return out.reshape(len(q), R.INNER)


def metrics(x, truth):
    x, truth = x.double().flatten(), truth.double().flatten()
    if not torch.isfinite(x).all() or not torch.isfinite(truth).all():
        raise ValueError("non-finite output")
    return {"cosine": F.cosine_similarity(x, truth, dim=0).item(),
            "relative_error": ((x - truth).norm() / truth.norm().clamp_min(1e-30)).item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=ROOT / "build/fixture.pt")
    ap.add_argument("--weights", type=Path, default=ROOT / "build/weights_gptq")
    ap.add_argument("--comfy-truth", type=Path, help="T2VA dump with blocks/refined_text.npy and noise_video/audio.npy")
    ap.add_argument("--modes", default="none,w8a8,gptq_a4,gptq_a16,w4g128a16,w4g32a16")
    ap.add_argument("--attention-dtype", choices=("fp16", "bf16"), default="fp16")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    modes = a.modes.split(",")
    allowed = {"none", "w8a8", "gptq_a4", "gptq_a16", "w4a16", "w4g128a16", "w4g64a16", "w4g32a16"}
    if not set(modes) <= allowed or modes[0] != "none":
        ap.error("modes must start with none and use supported study modes")
    device = "cuda"
    ckpt = R.Checkpoint(device=device, dtype=torch.bfloat16)
    weights = GPTQWeights(a.weights)
    base = R.H3Ref(ckpt, quant="none", cache_linears=False)
    if a.comfy_truth:
        d = a.comfy_truth
        nv, na = np.load(d / "noise_video.npy"), np.load(d / "noise_audio.npy")
        text = np.load(d / "blocks/refined_text.npy").reshape(-1, R.HIDDEN)
        layout = R.Layout(len(text), *nv.shape[1:], na.shape[-1])
        # Older h_in dumps captured block 0's output after an in-place update.
        # Reconstruct the actual input from saved noise and refined text instead.
        _, nt, nh, nw = nv.shape
        vr = torch.from_numpy(nv).reshape(24, nt, nh // 2, 2, nw // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(-1, 96).to(device)
        ar = torch.from_numpy(na).permute(1, 2, 0).reshape(-1, 32).to(device)
        with torch.inference_mode():
            x0 = torch.cat([torch.from_numpy(text).to(device, torch.bfloat16), base.audio_in(ar), base.video_in(vr)])
        temb = base.t_emb(torch.tensor([0.0, 0.0]))  # first evaluation: both sigmas are 1
        cos, sin = R.rope_tables(layout.position_ids, base.inv_freq, device)
        rows = layout.adaln_rows.to(device)
        if len(x0) != layout.seq_len:
            raise ValueError("dump is not a plain T2VA layout")
    else:
        fx = torch.load(a.fixture, map_location="cpu")
        layout = R.Layout(*fx["layout"])
        x0 = fx["x"].to(device, torch.bfloat16)
        temb, rows, cos, sin = [fx[k].to(device) for k in ("temb", "rows", "cos", "sin")]
    tclass = layout.tclass.to(device)
    report = {"input": str(a.comfy_truth or a.fixture), "weights": str(a.weights), "checkpoint": str(R.CKPT),
              "tokens": layout.seq_len, "linear_dtype": "bf16", "residual_dtype": "bf16",
              "attention_dtype": a.attention_dtype, "purpose": "accuracy emulation, not native throughput",
              "results": {}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{layout.seq_len} tokens; BF16 linears, {a.attention_dtype} attention; exact exported GPTQ weights", flush=True)
    truth = None
    for mode in modes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ref = StudyRef(ckpt, mode, weights, torch.float16 if a.attention_dtype == "fp16" else torch.bfloat16)
        start = time.monotonic()
        with torch.inference_mode():
            x = ref.blocks_forward(x0.clone(), temb, rows, cos, sin)
            video, audio = ref.final(x, temb, tclass, layout)
            result = (video.float().cpu(), audio.float().cpu())
        if truth is None:
            truth = result
        stats = {"video": metrics(result[0], truth[0]), "audio": metrics(result[1], truth[1]),
                 "combined": metrics(torch.cat([z.flatten() for z in result]), torch.cat([z.flatten() for z in truth])),
                 "seconds": time.monotonic() - start, "peak_torch_gb": torch.cuda.max_memory_allocated() / 1e9}
        if a.comfy_truth:
            # At sigma=1, ComfyUI's denoised video equals noise + our velocity.
            expected = torch.from_numpy(np.load(a.comfy_truth / "d_00.npy") - nv)
            expected = expected.reshape(24, nt, nh // 2, 2, nw // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(-1, 96)
            stats["video_vs_comfy"] = metrics(result[0], expected)
        report["results"][mode] = stats
        a.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{mode:14s}: video {stats['video']} audio {stats['audio']} combined {stats['combined']} "
              f"{stats['seconds']:.1f}s peak {stats['peak_torch_gb']:.2f}GB", flush=True)
        del x, video, audio, ref


if __name__ == "__main__":
    main()
