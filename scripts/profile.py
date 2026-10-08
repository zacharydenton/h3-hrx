#!/usr/bin/env python3
"""Run one benchmark with host/AMD SMI snapshots; optional ROCprofiler capture.

Raw HRX queues may not appear in ROCprofiler. A successful process is not proof
of GPU trace coverage: report the actual dispatch rows found in exported CSVs.
Use native H3/HRX device-clock profiling for those queues. Never wrap storage or
resident exchange work in kernel replay; external side effects cannot be reset.
"""
import argparse
import csv
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


def capture(argv, path):
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=15, check=False)
        path.write_text(f"command: {argv!r}\nexit: {result.returncode}\n{result.stdout}")
    except (OSError, subprocess.TimeoutExpired) as error:
        path.write_text(f"unavailable: {error}\n")


def snapshot(root, phase):
    for name, argv in [("amd-metric", ["amd-smi", "metric"]),
                       ("amd-process", ["amd-smi", "process"])]:
        capture(argv, root / f"{phase}-{name}.txt")
    paths = [Path("/proc/meminfo"), Path("/proc/vmstat"), Path("/proc/diskstats")]
    paths += list(Path("/sys/devices/system/node").glob("node*/numastat"))
    with (root / f"{phase}-host.txt").open("w") as output:
        for path in paths:
            try:
                output.write(f"[{path}]\n{path.read_text()}\n")
            except OSError as error:
                output.write(f"[{path}] unavailable: {error}\n")


def dispatch_rows(root):
    rows = 0
    for path in root.rglob("*kernel*trace*.csv"):
        with path.open(newline="") as source:
            rows += sum(1 for _ in csv.DictReader(source))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiler", choices=["native", "rocprofv3", "pc-sampling"], default="native")
    parser.add_argument("--sampling-interval", type=int, default=1048576)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a command after --")
    if args.sampling_interval < 1:
        parser.error("sampling interval must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve()
    capture(["uname", "-a"], root / "kernel.txt")
    capture(["amd-smi", "version"], root / "amd-version.txt")
    capture(["amd-smi", "static"], root / "amd-static.txt")
    capture(["amd-smi", "topology"], root / "amd-topology.txt")
    env = os.environ.copy()
    if args.profiler == "native":
        env.setdefault("H3_PROFILE", "device")
    else:
        if not shutil.which("rocprofv3"):
            parser.error("rocprofv3 is not installed")
        capture(["rocprofv3", "--version"], root / "rocprofiler-version.txt")
        wrapper = ["rocprofv3", "--kernel-trace", "--memory-copy-trace", "--output-format", "csv",
                   "--output-directory", str(root / "rocprofiler")]
        if args.profiler == "pc-sampling":
            wrapper += ["--pc-sampling-beta-enabled", "--pc-sampling-unit", "instructions",
                        "--pc-sampling-method", "stochastic", "--pc-sampling-interval",
                        str(args.sampling_interval)]
        command = wrapper + ["--"] + command
    snapshot(root, "before")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with (root / "workload.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    snapshot(root, "after")
    values = {"command": repr(command), "exit": result.returncode, "seconds": elapsed,
              "child_user_seconds": after.ru_utime-before.ru_utime,
              "child_system_seconds": after.ru_stime-before.ru_stime,
              "minor_faults": after.ru_minflt-before.ru_minflt,
              "major_faults": after.ru_majflt-before.ru_majflt,
              "voluntary_switches": after.ru_nvcsw-before.ru_nvcsw,
              "involuntary_switches": after.ru_nivcsw-before.ru_nivcsw}
    if args.profiler != "native":
        values["observed_kernel_dispatch_rows"] = dispatch_rows(root)
        if not values["observed_kernel_dispatch_rows"]:
            values["gpu_trace"] = "no dispatch coverage observed; use native profiling for HRX queues"
    (root / "summary.txt").write_text("".join(f"{key}: {value}\n" for key, value in values.items()))
    print((root / "summary.txt").read_text(), end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
