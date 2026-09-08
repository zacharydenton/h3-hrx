"""Reproducible dense INT8-QK / FP16-PV benchmark, including sampled FP32 validation.

All variants share identical inputs. Timings use HIP events, one warmup followed by
--repeat measured launches, alternating kernel order between rounds. FLOPs are
4 * tokens**2 * heads * 128 (both dense matrix products; softmax is not counted).
Kernel-only timing excludes quantization, transpose, allocation, and transfers.
"""

import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from kernel_test import LOOMRUN, ROOT, compile_kernel


def cpu_ticks():
    values = [
        int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
    ]
    return sum(values), values[3] + values[4]


def cpu_busy(before, after):
    total = after[0] - before[0]
    return 1 - (after[1] - before[1]) / max(total, 1)


def measure(cmd):
    """Record clocks during the launch; CPU work shares the APU's power budget."""
    device = next(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")).parent
    clock_path = next(device.glob("hwmon/hwmon*/freq1_input"), None)
    readings = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            if clock_path and int((device / "gpu_busy_percent").read_text()) > 90:
                readings.append(int(clock_path.read_text()) / 1e6)
            stop.wait(0.1)

    before = cpu_ticks()
    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        process = subprocess.run(cmd, check=True, text=True, capture_output=True)
    finally:
        stop.set()
        thread.join()
    timing = json.loads(process.stdout.strip().splitlines()[-1])
    return timing, {
        "cpu_busy_fraction": cpu_busy(before, cpu_ticks()),
        "busy_gpu_mhz": readings,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tokens", type=int)
    p.add_argument("stems", nargs="+")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--wave-size", type=int, default=32, choices=[32, 64])
    p.add_argument("--heads", type=int, default=56)
    p.add_argument("--keys", type=int, default=16, choices=[16, 32, 64])
    p.add_argument(
        "--queries",
        type=int,
        default=1,
        choices=[1, 2],
        help="16-query tiles per wave (must match the selected kernel)",
    )
    p.add_argument("--schedule", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--q-lds", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--pv-group", type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--skip-rescale", action="store_true")
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--buffers", type=int, default=2, choices=[1, 2])
    p.add_argument(
        "--head-major",
        action="store_true",
        help="Store Q, K and their scales with heads first (requires an _hm_ kernel)",
    )
    p.add_argument("--score-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3008)
    p.add_argument(
        "--full-check",
        action="store_true",
        help="Validate every output; intended for small correctness cases",
    )
    p.add_argument(
        "--wait-idle",
        action="store_true",
        help="Wait for five idle GPU / <20%% CPU readings before each measurement",
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse this output directory's inputs and compiled kernels",
    )
    p.add_argument("--output", type=Path, default=ROOT / "build/attention30")
    args = p.parse_args()
    assert args.repeat >= 2 and args.rounds >= 1
    assert all(
        ("_hm_" in stem or "qkhm_" in stem) == args.head_major for stem in args.stems
    ), "Input layout must match the kernel"
    n, h, d = args.tokens, args.heads, 128
    cap = max((n + 47) // 32 * 32, (n + 255) // 256 * 256)
    outdir = args.output / f"n{n}"
    outdir.mkdir(parents=True, exist_ok=True)
    hip_options = {
        key: getattr(args, key)
        for key in (
            "keys",
            "queries",
            "schedule",
            "q_lds",
            "pv_group",
            "skip_rescale",
            "buffers",
            "prefetch",
        )
    }
    previous = json.loads((outdir / "results.json").read_text()) if args.reuse else None
    if previous:
        for key, value in {
            "tokens": n,
            "heads": h,
            "capacity": cap,
            "seed": args.seed,
            "score_scale": args.score_scale,
            "head_major": args.head_major,
            "wave_size": args.wave_size,
            "hip_options": hip_options,
        }.items():
            assert (
                previous.get(key, {"head_major": False, "wave_size": 32}.get(key))
                == value
            ), f"Reuse configuration mismatch: {key}"
    rng = np.random.default_rng(args.seed)
    # Rounded Gaussian codes and positive scales approximate rotated H3 Q/K.
    paths = []
    for name, is_codes in (
        ("q", True),
        ("qs", False),
        ("k", True),
        ("ks", False),
        ("v", None),
    ):
        path = outdir / f"{name}.bin"
        if args.reuse:
            assert path.exists(), path
            paths.append(path)
            continue
        if is_codes:
            data = np.clip(
                np.rint(rng.standard_normal((cap, h, d), dtype=np.float32) * 40),
                -127,
                127,
            ).astype(np.int8)
            data[n:] = 0
        elif is_codes is False:
            data = (
                rng.uniform(0.8, 1.2, (cap, h)) * (1e-4 if name == "qs" else 0.125)
            ).astype(np.float32)
            if name == "qs":
                data *= args.score_scale
            data[n:] = 0
        else:
            data = (rng.standard_normal((h * d, cap), dtype=np.float32) * 0.5).astype(
                np.float16
            )
            data[:, n:] = 0
        if args.head_major and is_codes is not None:
            data = data.swapaxes(0, 1)
        np.ascontiguousarray(data).tofile(path)
        paths.append(path)
        del data
    q = np.memmap(paths[0], mode="r", dtype=np.int8, shape=(cap, h, d))
    qs = np.memmap(paths[1], mode="r", dtype=np.float32, shape=(cap, h))
    k = np.memmap(paths[2], mode="r", dtype=np.int8, shape=(cap, h, d))
    ks = np.memmap(paths[3], mode="r", dtype=np.float32, shape=(cap, h))
    v = np.memmap(paths[4], mode="r", dtype=np.float16, shape=(h, d, cap))
    if args.head_major:
        q = q.reshape(h, cap, d).transpose(1, 0, 2)
        k = k.reshape(h, cap, d).transpose(1, 0, 2)
        qs = qs.reshape(h, cap).T
        ks = ks.reshape(h, cap).T
    rows = np.unique(np.r_[0, 1, 15, 16, n // 2, n - 17, n - 2, n - 1].clip(0, n - 1))
    heads = np.unique([0, h // 2, h - 1])
    if args.full_check:
        rows, heads = np.arange(n), np.arange(h)
    refs = {}
    for head in heads:
        scores = q[rows, head].astype(np.float32) @ k[:n, head].astype(np.float32).T
        scores *= qs[rows, head, None] * ks[None, :n, head]
        prob = np.exp(scores - scores.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        refs[int(head)] = prob @ v[head, :, :n].astype(np.float32).T
    built, results = {}, {}
    for stem in args.stems:
        src = ROOT / "h3/kernels" / f"{stem}.loom"
        if not src.exists():
            src = ROOT / "experiments" / f"{stem}.loom"
        if not src.exists():
            src = ROOT / "experiments" / f"{stem}.cpp"
        wave_match = re.search(r"_mha(\d*)_", stem)
        waves = int(wave_match[1]) if wave_match and wave_match[1] else 4
        sym, ns = "h3_" + stem, "h3." + stem
        hsaco = outdir / f"{stem}.hsaco"
        config = {
            f"{ns}.{key}": val
            for key, val in {
                "q_stride": h * d,
                "kv_stride": h * d,
                "out_stride": h * d,
                "tokens": n,
                "token_capacity": cap,
                "scale": 1.0,
            }.items()
        }
        assert src.suffix != ".cpp" or args.wave_size == 32, (
            "HIP variants require wave32"
        )
        if args.reuse:
            assert hsaco.exists(), hsaco
            assert (
                previous["results"][stem]["source_sha256"]
                == hashlib.sha256(src.read_bytes()).hexdigest()
            ), "Source changed; rebuild without --reuse"
        elif src.suffix == ".cpp":
            subprocess.run(
                [
                    "/opt/rocm/bin/hipcc",
                    "--genco",
                    "--offload-arch=gfx1151",
                    "-O3",
                    "-std=c++17",
                    f"-DTOKENS={n}",
                    f"-DCAPACITY={cap}",
                    f"-DHEADS={h}",
                    f"-DKEY_TILE={args.keys}",
                    f"-DWAVES={waves}",
                    f"-DQUERY_TILES={args.queries}",
                    f"-DQ_LDS={args.q_lds}",
                    f"-DPV_GROUP={args.pv_group}",
                    f"-DSKIP_RESCALE={int(args.skip_rescale)}",
                    f"-DBUFFERS={args.buffers}",
                    f"-DPREFETCH={int(args.prefetch)}",
                    f"-DSCHEDULE_MODE={args.schedule}",
                    f"-DKERNEL_NAME={sym}",
                    str(src),
                    "-o",
                    str(hsaco),
                ],
                check=True,
            )
        else:
            try:
                compile_kernel(src, sym, config, hsaco)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(error.stderr) from error
        output = outdir / f"{stem}.out"
        query_block = 16 * waves * args.queries
        cmd = [
            str(LOOMRUN),
            "--hsaco",
            str(hsaco),
            "--kernel",
            sym,
            "--grid",
            f"{(n + query_block - 1) // query_block},{h},1",
            "--block",
            f"{args.wave_size * waves},1,1",
            "--repeat",
            str(args.repeat),
            "--i32",
            str(n),
            "--i32",
            "0",
        ]
        for path in paths:
            cmd += ["--in", str(path)]
        cmd += ["--out", f"{output}:{n * h * d * 2}"]
        built[stem] = (cmd, output)
        results[stem] = {
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(hsaco.read_bytes()).hexdigest(),
            "samples_ms": [],
            "environment": [],
        }
    valid = True
    for rnd in range(args.rounds):
        for stem in args.stems if rnd % 2 == 0 else args.stems[::-1]:
            if args.wait_idle:
                busy_path = next(
                    Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")
                )
                idle = 0
                before = cpu_ticks()
                while idle < 5:
                    time.sleep(1)
                    after = cpu_ticks()
                    idle = (
                        idle + 1
                        if int(busy_path.read_text()) < 10
                        and cpu_busy(before, after) < 0.2
                        else 0
                    )
                    before = after
            cmd, output = built[stem]
            timing, environment = measure(cmd)
            ms = timing["per_launch_us"] / 1000
            result = results[stem]
            result["samples_ms"].append(ms)
            result["environment"].append(environment)
            actual = np.memmap(output, mode="r", dtype=np.float16, shape=(n, h, d))
            sample = np.concatenate(
                [actual[rows, head].astype(np.float32) for head in heads]
            )
            ref = np.concatenate(list(refs.values()))
            rel = float(np.linalg.norm(sample - ref) / np.linalg.norm(ref))
            finite = bool(np.isfinite(actual).all())
            digest = hashlib.sha256(sample.tobytes()).hexdigest()
            deterministic = result.get("sample_sha256", digest) == digest
            result.update(
                sample_sha256=digest,
                sample_relative_l2=rel,
                finite=finite,
                deterministic=deterministic,
            )
            valid &= finite and deterministic and rel < 0.002
            print(
                f"round {rnd + 1}: {stem}: {ms:.3f} ms, {4 * n * n * h * d / ms / 1e9:.3f} TFLOP/s-eq, sampled rel L2 {rel:.6f}",
                flush=True,
            )
            del actual
    for stem, result in results.items():
        result["median_ms"] = float(np.median(result["samples_ms"]))
        result["median_tflops"] = 4 * n * n * h * d / result["median_ms"] / 1e9
    report = {
        "tokens": n,
        "heads": h,
        "head_dim": d,
        "capacity": cap,
        "repeat": args.repeat,
        "rounds": args.rounds,
        "seed": args.seed,
        "score_scale": args.score_scale,
        "full_check": args.full_check,
        "head_major": args.head_major,
        "wave_size": args.wave_size,
        "hip_options": hip_options,
        "timestamp": time.time(),
        "metric": "4*N*N*heads*D / kernel seconds / 1e12",
        "valid": bool(valid),
        "results": results,
    }
    (outdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
