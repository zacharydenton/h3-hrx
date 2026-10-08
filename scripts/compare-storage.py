#!/usr/bin/env python3
"""Alternate independent lifecycle benchmark processes and compare loading medians."""
import argparse
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="built Criterion lifecycle executable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--filter", default="checkpoint/cold_file")
    parser.add_argument("--progress", choices=["sqpoll", "wait"], default="sqpoll")
    args = parser.parse_args()
    if args.samples < 7:
        parser.error("use at least seven independent samples")
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ["mapped", "native-buffered", "native-direct"]
    timings = {}
    for sample in range(args.samples+1):
        order = modes[sample % len(modes):]+modes[:sample % len(modes)]
        for mode in order:
            env = dict(os.environ, H3_BENCH_WEIGHT_IO=mode, H3_BENCH_STORAGE_PROGRESS=args.progress,
                       H3_BENCH_DETAILS="1")
            env.pop("H3_BENCH_STORAGE_STATISTICS", None)
            path = args.output / f"{mode}-{sample}.log"
            with path.open("w") as log:
                subprocess.run([str(args.binary.resolve()), "--test", args.filter], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            found = re.findall(r"weight-load ([^:]+): ([0-9.]+) ms;", path.read_text())
            if not found:
                raise RuntimeError(f"no loading measurements in {path}")
            if sample:
                for name, value in found:
                    timings.setdefault((mode, name), []).append(float(value))
            print(f"completed {mode} sample {sample}/{args.samples}", flush=True)
    rows = ["route\tcase\tsamples\tmedian_ms\tchange_percent"]
    for (mode, name), samples in sorted(timings.items()):
        median = statistics.median(samples)
        baseline = statistics.median(timings["mapped", name])
        rows.append(f"{mode}\t{name}\t{len(samples)}\t{median:.6f}\t{100*(median/baseline-1):.2f}")
    report = "\n".join(rows)+"\n"
    (args.output / "medians.tsv").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
