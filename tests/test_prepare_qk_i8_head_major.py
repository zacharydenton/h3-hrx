"""The head-major preparation must preserve INT8 codes/scales and padded zeros."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel, launch, workdir


def main():
    rng = np.random.default_rng(3008)
    with workdir() as tmp:
        tmp = Path(tmp)
        for n, heads in [(1, 3), (17, 9), (257, 56)]:
            cap = (n + 31) // 32 * 32
            stride, offset = heads * 128 + 256, 128
            x = (rng.standard_normal((n, stride)) * 0.7).astype(np.float16)
            mean = (rng.standard_normal(stride) * 0.1).astype(np.float32)
            actual = []
            for hm in (False, True):
                stem = "prepare_qk_i8hm" if hm else "prepare_qk_i8"
                config = {
                    "row_stride": stride,
                    "head_offset": offset,
                    "heads": heads,
                    "extra_scale": 1 / (128 * np.sqrt(128)),
                }
                if hm:
                    config["token_capacity"] = cap
                hsaco = tmp / (stem + ".hsaco")
                compile_kernel(
                    ROOT / "h3/kernels" / (stem + ".loom"),
                    "h3_" + stem,
                    {"h3." + stem + "." + k: v for k, v in config.items()},
                    hsaco,
                )
                args = [
                    ("i32", n),
                    ("in_f16", x),
                    ("in", mean),
                    ("out", ((heads, cap, 32) if hm else (n, heads, 32), np.int32)),
                    ("out", ((heads, cap) if hm else (n, heads), np.float32)),
                ]
                (codes, scales), _ = launch(
                    hsaco, "h3_" + stem, (n, 1, 1), (256, 1, 1), args, tmp
                )
                if hm:
                    assert np.all(codes[:, n:] == 0) and np.all(scales[:, n:] == 0)
                    codes, scales = codes[:, :n].transpose(1, 0, 2), scales[:, :n].T
                actual.append((codes, scales))
            assert all(np.array_equal(a, b) for a, b in zip(*actual)), (n, heads)
            print(
                f"PASS N={n}, heads={heads}: bit-identical codes/scales, zero padding",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
