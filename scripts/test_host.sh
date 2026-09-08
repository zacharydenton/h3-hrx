#!/usr/bin/env bash
# The host's own unit tests: bounded, CPU-only, no checkpoints, GPU runtime or containers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for crate in h3 hrx cli loomrun; do
  printf '=== %s ===\n' "$crate"
  (cd "$crate" && cargo test --quiet)
done
