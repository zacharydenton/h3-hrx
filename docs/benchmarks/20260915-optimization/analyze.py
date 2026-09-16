#!/usr/bin/env python3
"""CPU-only static accounting and hypothetical timing scenarios; no GPU benchmark."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "20260914"
run = json.loads((BASE / "results.json").read_text())["runs"]["h3_i8"]
profile = json.loads((BASE / "profiles.json").read_text())["h3"]["denoise"]
shares = {s["stage"]: s["share_of_instrumented_seconds"]
          for s in profile["stages_above_one_percent"]}


def pitch(k):
    return k + 64 if k % 1024 == 0 else k


def gib(n):
    return round(n / 2**30, 6)


def matrices(shapes):
    return sum(n * pitch(k) for n, k in shapes) * 50


# Source: src/model.rs, src/layout.rs, src/dit.rs, src/stack.rs.
hid, inner, video, audio = 5376, 56 * 128, 37 * 24 * 42, 207 * 2
layouts = {}
for name, text, ref in [("alien_fjord", 338, 0), ("glass_leviathan", 1329, 1008)]:
    tokens = text + ref + video + audio
    seq_capacity = ((tokens + 255) // 256) * 256 + 32
    stack_capacity = max(((tokens + 47) // 32) * 32,
                         ((tokens + 127) // 128) * 128,
                         ((tokens + 255) // 256) * 256)
    layouts[name] = {
        "text_rows": text, "reference_rows": ref, "video_rows": video,
        "audio_rows": audio, "total_rows": tokens,
        "sequence_capacity": seq_capacity, "stack_capacity": stack_capacity,
        "text_copy_allocated_gib": gib(seq_capacity * hid * 4),
        "text_copy_prefix_only_gib": gib((text + ref) * hid * 4),
        "text_copy_excess_gib": gib((seq_capacity - text - ref) * hid * 4),
        "split_qkv_gib": gib(stack_capacity * inner * 3 * 2),
        "cache_three_residual_buffers_gib": gib(tokens * hid * 4 * 3),
    }

sampling = run["stages_seconds"]["sampling"]
wall = run["wall_seconds"]
fixed = wall - sampling
attention = shares["attention"]
gemm = sum(v for k, v in shares.items() if k.startswith("gemm"))


def scenario(seconds):
    return {"seconds": round(seconds, 3),
            "minutes": round(seconds / 60, 3),
            "speedup_vs_baseline": round(wall / seconds, 3),
            "wall_time_saved_percent": round(100 * (1 - seconds / wall), 3)}


out = {
    "method": "Static allocation formulas and Amdahl projections, not measured improvements. "
              "Profile shares come from a separate two-evaluation run with graph replay disabled. "
              "Fewer-call scenarios assume unchanged per-call cost and fixed overhead; "
              "adapter overhead, altered sampler, loading and quality are not modeled.",
    "layouts": layouts,
    "main_int8_linear_weights_gib": {
        "text_encoder_excludes_scales_norms_vision_scratch": gib(matrices([
            (10240, 5120), (5120, 8192), (51200, 5120), (5120, 25600)])),
        "dit_excludes_scales_norms_refiner_conditioning_scratch": gib(matrices([
            (21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336)])),
    },
    "baseline_seconds": wall,
    "fixed_non_sampling_seconds": fixed,
    "hypothetical_scenarios": {
        "attention_latency_minus_20_percent": scenario(wall - sampling * attention * .2),
        "all_four_gemm_latency_minus_20_percent": scenario(wall - sampling * gemm * .2),
        "attention_and_gemm_latency_minus_20_percent": scenario(wall - sampling * (attention + gemm) * .2),
        "attention_twice_as_fast": scenario(wall - sampling * attention * .5),
        "decode_twice_as_fast": scenario(wall - run["stages_seconds"]["decode"] * .5),
        **{f"{n}_transformer_calls": scenario(fixed + sampling * n / 20)
           for n in (16, 12, 8, 4)},
        **{f"cache_skips_{n}_of_20_suffixes": scenario(fixed + sampling * (1 - .98 * n / 20))
           for n in (5, 10)},
    },
}
print(json.dumps(out, indent=2))
