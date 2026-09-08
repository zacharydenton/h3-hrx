"""Audio convolutions: CPU compilation, exact f32 order, padding and residuals.

--compile-only checks generated-source identity and all decoder configuration
families without initializing HIP. --gpu also compares both kernels bit for
bit, checks output guards, and uses an independent float64 oracle on small cases.
"""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from gen_conv1d4_f32 import generate
from kernel_test import LOOM_COMPILE, compile_kernel, launch, workdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--compile-only", action="store_true")
    action.add_argument("--gpu", action="store_true")
    opt = parser.parse_args()
    # cin, cout, length, taps, dilation, padding, accumulate
    cases = [(3, 5, n, 11, 5, 25, acc)
             for n in (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 769)
             for acc in (0, 1)]
    cases += [(32, 2048, 207, 1, 1, 0, 0), (2048, 1024, 207, 7, 1, 3, 0),
              (512, 512, 1035, 11, 5, 25, 0), (512, 512, 1035, 7, 3, 9, 0),
              (256, 256, 5175, 3, 1, 1, 1), (256, 256, 5175, 7, 1, 3, 0),
              (32, 32, 41400, 11, 5, 25, 1), (4, 1, 165600, 7, 1, 3, 0)]
    with workdir() as directory:
        tmp = Path(directory)
        source = tmp / "generated.loom"
        source.write_text(generate())
        formatter = LOOM_COMPILE.parent.parent / "loom-format/loom-format"
        subprocess.run([str(formatter), "--in-place", str(source)], check=True, capture_output=True)
        assert source.read_bytes() == (ROOT / "h3/kernels/conv1d4_f32.loom").read_bytes(), "stale generated kernel"
        compiled = {}
        # Broad index bounds without allocation, within the backend's existing
        # 32-bit dynamic byte-offset limit for either convolution kernel.
        compile_cases = cases + [(2048, 1024, 524288, 7, 1, 3, 0),
                                 (4, 1, 4194304, 7, 1, 3, 0),
                                 (4096, 4096, 256, 64, 64, 1024, 1)]
        for case in compile_cases:
            ci, co, n, k, d, pad, acc = case
            config = dict(cin=ci, cout=co, ksize=k, dilation=d, pad=pad,
                          accumulate=acc, len_bound=(n + 255) // 256 * 256)
            for stem in ("conv1d_f32", "conv1d4_f32"):
                hsaco = tmp / (stem + "_" + "_".join(map(str, case)) + ".hsaco")
                compile_kernel(ROOT / "h3/kernels" / (stem + ".loom"), "h3_" + stem,
                               {f"h3.{stem}.{key}": value for key, value in config.items()}, hsaco)
                compiled[case, stem] = hsaco
        print(f"PASS generated identity and {len(compile_cases) * 2} CPU compile configurations", flush=True)
        if opt.compile_only:
            return 0
        rng = np.random.default_rng(171)
        for case in cases:
            ci, co, n, k, d, pad, acc = case
            x = rng.normal(0, .25, (ci, n)).astype(np.float32)
            w = rng.normal(0, .25 / np.sqrt(ci * k), (co, ci, k)).astype(np.float32)
            bias = rng.normal(0, .01, co).astype(np.float32)
            prev = rng.normal(0, .02, co * n + 64).astype(np.float32)
            prev[-64:] = 113
            args = [("i32", n), ("in", x), ("in", w), ("in", bias), ("inout", (prev, prev.shape))]
            outputs = []
            for stem, threads in (("conv1d_f32", 256), ("conv1d4_f32", 64)):
                (got,), _ = launch(compiled[case, stem], "h3_" + stem,
                                  ((n + 255) // 256, co, 1), (threads, 1, 1), args, tmp)
                assert np.array_equal(got[-64:], prev[-64:]), (case, stem, "output guard")
                outputs.append(got)
            assert np.array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32)), (case, "f32 order")
            if ci == 3:
                want = np.zeros((co, n), np.float64) + bias[:, None]
                if acc:
                    want += prev[:-64].reshape(co, n)
                for tap in range(k):
                    indices = np.arange(n) + tap * d - pad
                    valid = (indices >= 0) & (indices < n)
                    want[:, valid] += w[:, :, tap].astype(np.float64) @ x[:, indices[valid]].astype(np.float64)
                assert np.max(abs(outputs[1][:-64].reshape(co, n) - want)) < 2e-6, (case, "float64 oracle")
            print("PASS bit-exact convolution and guards", case, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
