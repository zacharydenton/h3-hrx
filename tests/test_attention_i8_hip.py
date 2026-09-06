"""Full-output checks for the >=30 TFLOP/s gfx1151 attention specialization.

Exercises both sides of key/query/workgroup boundaries, including uniform and
sharp softmax distributions. The oracle uses FP32 attention on identical INT8
operands. Timing in these small cases is not a performance claim.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEM = "attention_i8qkhip_mha8_lds_f16_wmma"


def main():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
    shapes = [1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1001]
    cases = [(n, 1.) for n in shapes] + [(257, scale) for scale in (0., 32., 256.)]
    for n, scale in cases:
        output = ROOT / "build/attention30/correctness" / f"scale{scale:g}"
        command = [sys.executable, str(ROOT / "tools/bench_attention_i8.py"), str(n), STEM,
                   "--heads", "3", "--keys", "16", "--queries", "2", "--schedule", "1", "--prefetch",
                   "--rounds", "2", "--repeat", "2", "--full-check", "--score-scale", str(scale),
                   "--output", str(output)]
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        if run.returncode:
            print(run.stdout + run.stderr)
            return run.returncode
        result = json.loads((output / f"n{n}/results.json").read_text())["results"][STEM]
        print(f"PASS N={n:4d}, score scale={scale:g}, full-output relative L2={result['sample_relative_l2']:.6g}, deterministic={result['deterministic']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
