"""Compare plain/residual INT8 GEMM pitches and row groups with float64 checks.

Example: bench_gemm_i8_tuning.py 37723x14336x5376 --groups 1 2 4 --pads 64 192
Timings include scaling and exclude transfers; each has a warmup then repeat
launches. Inputs are identical across candidates; padding contains garbage.
Other observed GPU compute clients invalidate a timing and trigger a retry.
Requires fuser; display work and activity between monitor samples can still
affect results. This is a benchmark monitor, not an exclusive GPU reservation.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from bench_attention_i8 import cpu_busy, cpu_ticks, measure
from kernel_test import LOOM_COMPILE, LOOMRUN, ROOT, TARGET, compile_kernel


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def other_gpu_clients():
    """Exclude display clients and this driver's own loomrun subprocess."""
    devices = [str(p) for p in Path("/dev/dri").glob("renderD*")]
    if not devices:
        return {}
    found = subprocess.run(
        ["fuser", *devices], capture_output=True, text=True, check=False
    )
    clients = {}
    for pid in found.stdout.split():
        try:
            proc = Path("/proc") / pid
            name = (proc / "comm").read_text().strip()
            # Field 4 is PPID; comm may contain spaces or parentheses.
            parent = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if name in ("electron", "niri", "Xwayland") or parent == os.getpid():
            continue
        clients[pid] = name
    return clients


def measured_without_competitors(cmd):
    stop = threading.Event()
    competitors = {}

    def monitor():
        while not stop.is_set():
            competitors.update(other_gpu_clients())
            stop.wait(0.1)

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        timing, environment = measure(cmd)
    finally:
        stop.set()
        thread.join()
    environment["other_gpu_clients"] = competitors
    return timing, environment


def wait_idle():
    paths = list(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
    idle, before = 0, cpu_ticks()
    while idle < 5:
        time.sleep(1)
        after = cpu_ticks()
        quiet = all(int(p.read_text()) < 10 for p in paths) and not other_gpu_clients()
        idle = idle + 1 if quiet and cpu_busy(before, after) < 0.2 else 0
        before = after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shape", help="MxKxN")
    parser.add_argument("--stem", default="gemm_i8_256")
    parser.add_argument("--tile-m", type=int, default=256, choices=[128, 256])
    parser.add_argument("--tile-n", type=int, default=128, choices=[128, 256])
    parser.add_argument("--groups", type=int, nargs="+", default=[4])
    parser.add_argument("--pads", type=int, nargs="+", default=[64])
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-check", action="store_true")
    parser.add_argument("--wait-idle", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "build/gemm-tuning")
    args = parser.parse_args()
    if not shutil.which("fuser"):
        parser.error("fuser is required to monitor competing GPU clients")
    m, k, n = map(int, args.shape.split("x"))
    if m < 1 or k < 64 or k % 64 or n < args.tile_n or n % args.tile_n:
        parser.error("Require M >= 1, K a multiple of 64, N a multiple of tile-n")
    if args.repeat < 2 or args.rounds < 1 or min(args.groups) < 1:
        parser.error("repeat >= 2, rounds >= 1 and groups >= 1 required")
    if any(p < 0 or p % 64 or k + p > 65536 for p in args.pads):
        parser.error("Padding must be a nonnegative multiple of 64; stride <= 65536")
    stem = args.stem
    if "swiglu" in stem:
        parser.error("SwiGLU is not supported by this benchmark")
    residual = "resid" in stem
    src = ROOT / "kernels" / f"{stem}.loom"
    if not src.exists():
        src = ROOT / "experiments" / f"{stem}.loom"
    ns, sym = "h3." + stem, "h3_" + stem
    outdir = args.output / args.shape
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    a = rng.integers(-127, 128, (m, k), dtype=np.int16).astype(np.int8)
    w = rng.integers(-127, 128, (n, k), dtype=np.int16).astype(np.int8)
    ws = (rng.random(n, dtype=np.float32) + 0.5) / k
    acts = rng.random(m, dtype=np.float32) + 0.5
    bias = rng.standard_normal(n).astype(np.float32) if stem.endswith("b") else None
    rows = np.unique(np.clip([0, 1, 15, 16, 255, 256, m // 2, m - 2, m - 1], 0, m - 1))
    cols = np.unique(
        np.clip([0, 1, 15, 16, 63, 64, 127, 128, n // 2, n - 2, n - 1], 0, n - 1)
    )
    if args.full_check:
        rows, cols = np.arange(m), np.arange(n)
    ref = (a[rows].astype(np.float64) @ w[cols].astype(np.float64).T) * ws[cols]
    ref *= acts[rows, None]
    if bias is not None:
        ref += bias[cols]
    if residual:
        gate = rng.uniform(-0.5, 0.5, (12, n)).astype(np.float32)
        cls = rng.integers(0, 12, m, dtype=np.int32)
        initial = rng.standard_normal((m, n), dtype=np.float32)
        # loomrun executes one warmup plus repeat in-place updates. Reset the
        # residual file before each invocation, outside the GPU event interval.
        ref = (
            initial[np.ix_(rows, cols)].astype(np.float64)
            + (args.repeat + 1) * gate[cls[rows]][:, cols] * ref
        )
        gate.tofile(outdir / "gate.bin")
        cls.tofile(outdir / "cls.bin")
    else:
        ref = np.clip(ref, -65472, 65472)
    for name, data in (("ws", ws), ("as", acts), ("bias", bias)):
        if data is not None:
            data.tofile(outdir / f"{name}.bin")
    configs = []
    output = outdir / "output.bin"
    # Compile before timing: CPU compilation shares the APU's power budget.
    for pad in args.pads:
        for name, data in (("a", a), ("w", w)):
            padded = np.full((len(data), k + pad), 113, dtype=np.int8)
            padded[:, :k] = data
            padded.tofile(outdir / f"{name}-pad{pad}.bin")
            del padded
        for group in args.groups:
            key = f"g{group}-pad{pad}"
            cfg = {
                f"{ns}.k_size": k,
                f"{ns}.n_size": n,
                f"{ns}.m_group": group,
                f"{ns}.k_stride": k + pad,
            }
            if residual:
                cfg[f"{ns}.classes"] = 12
            hs = outdir / f"{key}.hsaco"
            compile_kernel(src, sym, cfg, hs)
            gy = ((m + args.tile_m - 1) // args.tile_m + group - 1) // group * group
            cmd = [
                str(LOOMRUN),
                "--hsaco",
                str(hs),
                "--kernel",
                sym,
                "--grid",
                f"{n // args.tile_n},{gy},1",
                "--block",
                "256,1,1",
                "--repeat",
                str(args.repeat),
                "--i32",
                str(m),
            ]
            for name in (f"a-pad{pad}", f"w-pad{pad}", "ws", "as"):
                cmd += ["--in", str(outdir / f"{name}.bin")]
            if residual:
                cmd += [
                    "--inout",
                    str(output),
                    "--in",
                    str(outdir / "gate.bin"),
                    "--in",
                    str(outdir / "cls.bin"),
                ]
            else:
                cmd += ["--out", f"{output}:{m * n * 2}"]
            if bias is not None:
                cmd += ["--in", str(outdir / "bias.bin")]
            configs.append((key, cmd, hs, cfg))
    del a, w
    report = {
        "complete": False,
        "timestamp": time.time(),
        "stem": stem,
        "shape_mkn": [m, k, n],
        "tile_mn": [args.tile_m, args.tile_n],
        "seed": args.seed,
        "repeat": args.repeat,
        "rounds": args.rounds,
        "target": TARGET,
        "source_sha256": digest(src),
        "compiler_sha256": digest(LOOM_COMPILE),
        "metric": "2*M*K*N / seconds / 1e12; kernel only including epilogue",
        "reference": "float64 dot of identical INT8 codes and epilogue; residual includes warmup and repeated updates",
        "epilogue": "residual" if residual else "plain",
        "full_check": args.full_check,
        "atol": 0.002,
        "rtol": 0.002,
        "checked_rows": rows.tolist(),
        "checked_columns": cols.tolist(),
        "results": {},
    }
    for key, _, hs, cfg in configs:
        report["results"][key] = {
            "config": cfg,
            "binary_sha256": digest(hs),
            "samples_ms": [],
            "environment": [],
        }
    for round_id in range(args.rounds):
        for key, cmd, _, _ in configs if round_id % 2 == 0 else configs[::-1]:
            discarded = report["results"][key].setdefault("discarded_contention", [])
            while True:
                if residual:
                    initial.tofile(output)
                if args.wait_idle:
                    wait_idle()
                timing, environment = measured_without_competitors(cmd)
                if not environment["other_gpu_clients"]:
                    break
                discarded.append({"timing": timing, "environment": environment})
                (outdir / "results.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(
                    f"{key}: discarding contention with {environment['other_gpu_clients']}",
                    flush=True,
                )
                wait_idle()
            actual = np.memmap(
                output,
                dtype=np.float32 if residual else np.float16,
                mode="r",
                shape=(m, n),
            )
            sample = np.asarray(actual[np.ix_(rows, cols)], dtype=np.float64)
            finite = bool(np.isfinite(actual).all())
            del actual
            rel_l2 = float(
                np.linalg.norm(sample - ref) / max(np.linalg.norm(ref), 1e-30)
            )
            valid = finite and bool(
                np.all(np.abs(sample - ref) <= 0.002 + 0.002 * np.abs(ref))
            )
            if not valid:
                raise RuntimeError(
                    f"{key}: incorrect output; finite={finite}, relL2={rel_l2}"
                )
            result = report["results"][key]
            sample_hash = hashlib.sha256(sample.tobytes()).hexdigest()
            if result.get("sample_sha256", sample_hash) != sample_hash:
                raise RuntimeError(f"{key}: nondeterministic output")
            result.update(
                finite=finite,
                sample_relative_l2=rel_l2,
                sample_sha256=sample_hash,
                deterministic=bool(result["samples_ms"]),
            )
            ms = timing["per_launch_us"] / 1000
            result["samples_ms"].append(ms)
            result["environment"].append(environment)
            result["median_ms"] = statistics.median(result["samples_ms"])
            result["median_tops"] = 2 * m * k * n / (result["median_ms"] * 1e9)
            print(
                f"{stem} {args.shape} {key} round={round_id + 1} "
                f"{ms:.3f} ms {2 * m * k * n / (ms * 1e9):.3f} TOPS "
                f"relL2={rel_l2:.6f}",
                flush=True,
            )
            (outdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    report["identical_samples_across_configs"] = (
        len({result["sample_sha256"] for result in report["results"].values()}) == 1
    )
    (outdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
