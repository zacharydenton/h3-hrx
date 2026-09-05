#!/usr/bin/env bash
# The one test command. Fills in as the pieces land: format check, generated kernels
# against their generators, host build, reference vs diffusers, kernel tests, blocks vs
# the fixture.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
status=0
step() { local name="$1"; shift; printf '\n=== %s ===\n' "$name"; if "$@"; then printf '  ok\n'; else printf '  FAILED: %s\n' "$name"; status=1; fi; }
if ls kernels/*.loom >/dev/null 2>&1; then step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom'; fi
# ComfyUI imports only inside its image (its runtime packages are not in the shared venv); CPU, toy size
step "reference vs ComfyUI's MiniMaxH3Model (toy)" bash -c 'podman run --rm -v "$HOME:$HOME" -w "$PWD" -e PYTHONPATH=/opt/ComfyUI --entrypoint /opt/venv/bin/python docker.io/kyuz0/amd-strix-halo-comfyui:latest tests/test_ref_vs_comfy.py 2>&1 | grep -E "PASS|FAIL|Error" | tee /dev/stderr | grep -q FAIL && exit 1 || exit 0'
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
