"""Full-output checks for the Loom head-major INT8 attention kernel."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "attention_i8qkhm_mha8_lds_f16_wmma"


def main():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
    shapes = [1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1001]
    cases = [(n, 1.0) for n in shapes] + [(257, scale) for scale in (0.0, 32.0, 256.0)]
    for n, scale in cases:
        output = ROOT / "build/attention30/loom-correctness" / f"scale{scale:g}"
        command = [
            sys.executable,
            str(ROOT / "tools/bench_attention_i8.py"),
            str(n),
            STEM,
            "--heads",
            "3",
            "--head-major",
            "--rounds",
            "2",
            "--repeat",
            "2",
            "--full-check",
            "--score-scale",
            str(scale),
            "--output",
            str(output),
        ]
        run = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        if run.returncode:
            print(run.stdout + run.stderr)
            return run.returncode
        result = json.loads((output / f"n{n}/results.json").read_text())["results"][
            STEM
        ]
        print(
            f"PASS N={n:4d}, score scale={scale:g}, full-output relative L2={result['sample_relative_l2']:.6g}, deterministic={result['deterministic']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
