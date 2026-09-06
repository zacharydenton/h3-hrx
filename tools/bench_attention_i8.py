"""Reproducible dense INT8-QK / FP16-PV benchmark, including sampled FP32 validation.

All variants share identical inputs. Timings use HIP events, one warmup followed by
--repeat measured launches, alternating kernel order between rounds. FLOPs are
4 * tokens**2 * heads * 128 (both dense matrix products; softmax is not counted).
Kernel-only timing excludes quantization, transpose, allocation, and transfers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from kernel_test import ROOT, LOOMRUN, compile_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tokens", type=int)
    p.add_argument("stems", nargs="+")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--heads", type=int, default=56)
    p.add_argument("--keys", type=int, default=16, choices=[16, 32, 64])
    p.add_argument("--queries", type=int, default=1, choices=[1, 2], help="16-query tiles per HIP wave")
    p.add_argument("--wait-idle", action="store_true", help="Wait for five consecutive idle GPU readings before each measurement")
    p.add_argument("--output", type=Path, default=ROOT / "build/attention30")
    args = p.parse_args()
    assert args.repeat >= 2 and args.rounds >= 1
    n, h, d = args.tokens, args.heads, 128
    cap = max((n + 47) // 32 * 32, (n + 255) // 256 * 256)
    outdir = args.output / f"n{n}"
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3008)
    # Rounded Gaussian codes and positive scales approximate rotated H3 Q/K.
    paths = []
    for name, is_codes in (("q", True), ("qs", False), ("k", True), ("ks", False), ("v", None)):
        if is_codes:
            data = np.clip(np.rint(rng.standard_normal((cap, h, d), dtype=np.float32) * 40), -127, 127).astype(np.int8)
            data[n:] = 0
        elif is_codes is False:
            data = (rng.uniform(.8, 1.2, (cap, h)) * (1e-4 if name == "qs" else .125)).astype(np.float32)
            data[n:] = 0
        else:
            data = (rng.standard_normal((h * d, cap), dtype=np.float32) * .5).astype(np.float16)
            data[:, n:] = 0
        path = outdir / f"{name}.bin"
        data.tofile(path)
        paths.append(path)
        del data
    q = np.memmap(paths[0], mode="r", dtype=np.int8, shape=(cap, h, d))
    qs = np.memmap(paths[1], mode="r", dtype=np.float32, shape=(cap, h))
    k = np.memmap(paths[2], mode="r", dtype=np.int8, shape=(cap, h, d))
    ks = np.memmap(paths[3], mode="r", dtype=np.float32, shape=(cap, h))
    v = np.memmap(paths[4], mode="r", dtype=np.float16, shape=(h, d, cap))
    rows = np.unique(np.r_[0, 1, 15, 16, n // 2, n - 17, n - 2, n - 1].clip(0, n - 1))
    heads = np.unique([0, h // 2, h - 1])
    refs = {}
    for head in heads:
        scores = q[rows, head].astype(np.float32) @ k[:n, head].astype(np.float32).T
        scores *= qs[rows, head, None] * ks[None, :n, head]
        prob = np.exp(scores - scores.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        refs[int(head)] = prob @ v[head, :, :n].astype(np.float32).T
    built, results = {}, {}
    for stem in args.stems:
        src = ROOT / "kernels" / f"{stem}.loom"
        if not src.exists(): src = ROOT / "experiments" / f"{stem}.loom"
        if not src.exists(): src = ROOT / "experiments" / f"{stem}.cpp"
        waves = 8 if "mha8" in stem else 16 if "mha16" in stem else 4
        sym, ns = "h3_" + stem, "h3." + stem
        hsaco = outdir / f"{stem}.hsaco"
        config = {f"{ns}.{key}": val for key, val in dict(q_stride=h*d, kv_stride=h*d, out_stride=h*d, tokens=n, token_capacity=cap, scale=1.0).items()}
        if src.suffix == ".cpp":
            subprocess.run(["/opt/rocm/bin/hipcc", "--genco", "--offload-arch=gfx1151", "-O3", "-std=c++17",
                            f"-DTOKENS={n}", f"-DCAPACITY={cap}", f"-DHEADS={h}", f"-DKEY_TILE={args.keys}",
                            f"-DWAVES={waves}", f"-DQUERY_TILES={args.queries}", f"-DKERNEL_NAME={sym}", str(src), "-o", str(hsaco)], check=True)
        else:
            compile_kernel(src, sym, config, hsaco)
        output = outdir / f"{stem}.out"
        query_block = 16 * waves * (args.queries if src.suffix == ".cpp" else 1)
        cmd = [str(LOOMRUN), "--hsaco", str(hsaco), "--kernel", sym,
               "--grid", f"{(n+query_block-1)//query_block},{h},1", "--block", f"{32*waves},1,1",
               "--repeat", str(args.repeat), "--i32", str(n), "--i32", "0"]
        for path in paths: cmd += ["--in", str(path)]
        cmd += ["--out", f"{output}:{n*h*d*2}"]
        built[stem] = (cmd, output)
        results[stem] = dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), samples_ms=[])
    valid = True
    for rnd in range(args.rounds):
        for stem in (args.stems if rnd % 2 == 0 else args.stems[::-1]):
            if args.wait_idle:
                busy_path = next(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
                idle = 0
                while idle < 5:
                    idle = idle + 1 if int(busy_path.read_text()) < 10 else 0
                    time.sleep(1)
            cmd, output = built[stem]
            timing = json.loads(subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip().splitlines()[-1])
            ms = timing["per_launch_us"] / 1000
            result = results[stem]
            result["samples_ms"].append(ms)
            actual = np.memmap(output, mode="r", dtype=np.float16, shape=(n, h, d))
            sample = np.concatenate([actual[rows, head].astype(np.float32) for head in heads])
            ref = np.concatenate(list(refs.values()))
            rel = float(np.linalg.norm(sample-ref) / np.linalg.norm(ref))
            finite = bool(np.isfinite(actual).all())
            digest = hashlib.sha256(sample.tobytes()).hexdigest()
            deterministic = result.get("sample_sha256", digest) == digest
            result.update(sample_sha256=digest, sample_relative_l2=rel, finite=finite, deterministic=deterministic)
            valid &= finite and deterministic and rel < .002
            print(f"round {rnd+1}: {stem}: {ms:.3f} ms, {4*n*n*h*d/ms/1e9:.3f} TFLOP/s-eq, sampled rel L2 {rel:.6f}", flush=True)
            del actual
    for stem, result in results.items():
        result["median_ms"] = float(np.median(result["samples_ms"]))
        result["median_tflops"] = 4*n*n*h*d/result["median_ms"]/1e9
    report = dict(tokens=n, heads=h, head_dim=d, capacity=cap, repeat=args.repeat, rounds=args.rounds,
                  seed=3008, key_tile=args.keys, query_tiles=args.queries, timestamp=time.time(), metric="4*N*N*heads*D / kernel seconds / 1e12", valid=bool(valid), results=results)
    (outdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
