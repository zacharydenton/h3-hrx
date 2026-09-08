"""Compile a Loom kernel, run it on the GPU through loomrun, and compare against torch."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HRX_BUILD = Path(os.environ.get("HRX_BUILD", Path.home() / "code/hrx-system/build-cuda"))
LOOM_COMPILE = HRX_BUILD / "loom/src/loom/tools/loom-compile/loom-compile"
LOOMRUN = Path(os.environ.get("H3_LOOMRUN") or ROOT / "build/loomrun")   # H3_LOOMRUN: compare launchers
TARGET = os.environ.get("LOOM_TARGET", "gfx1151")


def compile_kernel(source: Path, root_symbol: str, config: dict, out: Path) -> None:
    cmd = [str(LOOM_COMPILE), str(source), "--backend=amdgpu-hal", f"--target={TARGET}",
           f"--root=@{root_symbol}", f"--output={out}"]
    cmd += [f"--config={k}={v}" for k, v in config.items()]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def to_bf16(x) -> np.ndarray:
    """f32 -> bf16 bit patterns (uint16), round to nearest even, as torch's .to(bfloat16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def from_bf16(u: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def launch(hsaco: Path, kernel: str, grid, block, args, workdir: Path, repeat: int = 1, rotate_input=None):
    """args: list of ('i32'|'f32', value) or ('in'|'in_f16'|'in_bf16'|'in_u8', ndarray) or ('out'|'out_f16'|'out_bf16', shape/dtype tuple).
    bf16 arrays travel as uint16 bit patterns (in_bf16 takes f32 values and rounds them; out_bf16 comes back as uint16: from_bf16 widens)."""
    cmd = [str(LOOMRUN), "--hsaco", str(hsaco), "--kernel", kernel,
           "--grid", ",".join(map(str, grid)), "--block", ",".join(map(str, block)),
           "--repeat", str(repeat)]
    if rotate_input is not None:
        index, copies = rotate_input
        cmd += ["--rotate-input", f"{index}:{copies}"]
    outputs = []
    for index, (kind, value) in enumerate(args):
        if kind in ("i32", "f32"):
            cmd += [f"--{kind}", str(value)]
        elif kind == "in":
            path = workdir / f"in{index}.bin"
            np.ascontiguousarray(value, dtype=np.float32).tofile(path)
            cmd += ["--in", str(path)]
        elif kind == "in_f16":
            path = workdir / f"in{index}.bin"
            np.ascontiguousarray(value, dtype=np.float16).tofile(path)
            cmd += ["--in", str(path)]
        elif kind == "in_bf16":
            path = workdir / f"in{index}.bin"
            to_bf16(value).tofile(path)
            cmd += ["--in", str(path)]
        elif kind == "in_i32":
            path = workdir / f"in{index}.bin"
            np.ascontiguousarray(value, dtype=np.int32).tofile(path)
            cmd += ["--in", str(path)]
        elif kind == "in_u8":
            path = workdir / f"in{index}.bin"
            np.ascontiguousarray(value, dtype=np.uint8).tofile(path)
            cmd += ["--in", str(path)]
        elif kind == "inout_f16":
            array, shape = value
            path = workdir / f"io{index}.bin"
            np.ascontiguousarray(array, dtype=np.float16).tofile(path)
            cmd += ["--inout", str(path)]
            outputs.append((path, shape, np.float16))
        elif kind == "inout":
            array, shape = value
            path = workdir / f"io{index}.bin"
            np.ascontiguousarray(array, dtype=np.float32).tofile(path)
            cmd += ["--inout", str(path)]
            outputs.append((path, shape, np.float32))
        elif kind == "out_f16":
            shape, _ = value
            path = workdir / f"out{index}.bin"
            nbytes = int(np.prod(shape)) * 2
            cmd += ["--out", f"{path}:{nbytes}"]
            outputs.append((path, shape, np.float16))
        elif kind == "out_bf16":
            shape, _ = value
            path = workdir / f"out{index}.bin"
            nbytes = int(np.prod(shape)) * 2
            cmd += ["--out", f"{path}:{nbytes}"]
            outputs.append((path, shape, np.uint16))
        elif kind == "out":
            shape, dtype = value
            path = workdir / f"out{index}.bin"
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            cmd += ["--out", f"{path}:{nbytes}"]
            outputs.append((path, shape, dtype))
        else:
            raise ValueError(kind)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    timing = json.loads(result.stdout.strip().splitlines()[-1])
    return [np.fromfile(p, dtype=d).reshape(s) for p, s, d in outputs], timing


def report(name: str, actual: np.ndarray, expected: np.ndarray, atol=2e-5, rtol=2e-5) -> bool:
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    absolute = np.abs(actual - expected)
    denom = np.maximum(np.abs(expected), 1e-12)
    relative = absolute / denom
    ok = bool(np.all(absolute <= atol + rtol * np.abs(expected)))
    cos = float(np.dot(actual.ravel(), expected.ravel()) /
                (np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel()) + 1e-30))
    print(f"  {'PASS' if ok else 'FAIL'} {name}: "
          f"max_abs={absolute.max():.3e} max_rel={relative.max():.3e} cosine={cos:.8f}")
    if not ok:
        bad = np.argsort(-absolute.ravel())[:4]
        for i in bad:
            print(f"      [{i}] actual={actual.ravel()[i]:.6g} expected={expected.ravel()[i]:.6g}")
    return ok


def workdir():
    return tempfile.TemporaryDirectory(prefix="loomtest-")
