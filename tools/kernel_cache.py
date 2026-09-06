"""Validate and atomically publish a compiled kernel, including its build identity."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def compile_cached(source, symbol, config, output):
    from kernel_test import LOOM_COMPILE, TARGET, compile_kernel

    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stamp = output.with_suffix(".src.json")
    # A shape can be requested by multiple sessions at once. Neither may consume
    # another process's half-written code object or publish the wrong stamp.
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compiler = Path(LOOM_COMPILE).resolve()
        stat = compiler.stat()
        identity = json.dumps({
            "source": hashlib.sha256(source.read_bytes()).hexdigest(),
            "symbol": symbol, "config": config, "target": TARGET,
            "compiler": [str(compiler), stat.st_size, stat.st_mtime_ns],
        }, sort_keys=True)
        if output.is_file() and output.stat().st_size and stamp.is_file() and stamp.read_text() == identity:
            return
        with tempfile.TemporaryDirectory(prefix=".compile-", dir=output.parent) as tmp:
            binary, metadata = Path(tmp) / "kernel.hsaco", Path(tmp) / "stamp.json"
            compile_kernel(source, symbol, config, binary)
            if not binary.is_file() or not binary.stat().st_size:
                raise RuntimeError(f"compiler produced no code for {source}")
            metadata.write_text(identity)
            os.replace(binary, output)
            os.replace(metadata, stamp)
