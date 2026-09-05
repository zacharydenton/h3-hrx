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
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
