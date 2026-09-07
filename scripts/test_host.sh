#!/usr/bin/env bash
# Bounded CPU-only regressions. No checkpoints, GPU runtime or containers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
PY="${H3_PYTHON:-python3}"
g++ -std=c++17 -O2 -Wno-subobject-linkage -ffunction-sections -fdata-sections tests/test_host_logic.cpp -Wl,--gc-sections -o build/test_host_logic
build/test_host_logic
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -ffunction-sections -fdata-sections tests/test_host_cli.cpp host/h3tok.cpp host/tokenizer_blob.S -Wl,--gc-sections -o "$tmpdir/host_cli"
"$tmpdir/host_cli" "$tmpdir"
g++ -std=c++17 -O1 -Wno-subobject-linkage -ffunction-sections -fdata-sections tests/test_weights.cpp -Wl,--gc-sections -o "$tmpdir/weights"
"$tmpdir/weights" "$tmpdir"
g++ -std=c++17 -O1 -Wno-subobject-linkage -ffunction-sections -fdata-sections tests/test_kernel_cache.cpp -Wl,--gc-sections -o "$tmpdir/kernel_cache" -pthread
mkdir -p "$tmpdir/cache_test"; "$tmpdir/kernel_cache" "$tmpdir/cache_test"
g++ -std=c++17 -O1 -Wno-subobject-linkage -ffunction-sections -fdata-sections tests/test_pipe_memory.cpp -Wl,--gc-sections -o "$tmpdir/pipe_memory"
"$tmpdir/pipe_memory" "$tmpdir"
g++ -std=c++17 -O1 -Itests/fakes tests/test_runtime_cleanup.cpp -o "$tmpdir/runtime_cleanup"
"$tmpdir/runtime_cleanup"
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Itests/fakes tests/test_loomrun_rotation.cpp -o "$tmpdir/loomrun_rotation"
"$tmpdir/loomrun_rotation" "$tmpdir"
g++ -std=c++17 -O1 -g -Wno-subobject-linkage -fsanitize=address,undefined -fno-omit-frame-pointer -ffunction-sections -fdata-sections tests/test_vae_wide_host.cpp -Wl,--gc-sections -o "$tmpdir/vae_wide_host"
"$tmpdir/vae_wide_host" "$tmpdir"
"$PY" tests/test_review_regressions.py
"$PY" -O tests/test_review_regressions.py
