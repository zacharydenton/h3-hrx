#!/usr/bin/env bash
# Bounded CPU-only regressions. No checkpoints, GPU runtime or containers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
g++ -std=c++17 -O2 -Wno-subobject-linkage -ffunction-sections -fdata-sections tests/test_host_logic.cpp -Wl,--gc-sections -o build/test_host_logic
build/test_host_logic
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
g++ -std=c++17 -O1 -Wno-subobject-linkage -ffunction-sections -fdata-sections tests/test_pipe_memory.cpp -Wl,--gc-sections -o "$tmpdir/pipe_memory"
"$tmpdir/pipe_memory" "$tmpdir"
g++ -std=c++17 -O1 -Itests/fakes tests/test_runtime_cleanup.cpp -o "$tmpdir/runtime_cleanup"
"$tmpdir/runtime_cleanup"
for variant in DIT TE VAE; do
  g++ -std=c++17 -O1 -Wno-subobject-linkage -Itests/fakes -D"TEST_$variant" tests/test_session_cleanup.cpp -o "$tmpdir/cleanup"
  "$tmpdir/cleanup" "$tmpdir"
done
python3 tests/test_review_regressions.py
python3 -O tests/test_review_regressions.py
