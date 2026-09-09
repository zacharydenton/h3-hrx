# Loom compiler used by this project

These six patches apply on top of `ROCm/hrx-system` commit
`c9855b47e96e7eb1cbb5b81b1de973762982ae95`. They preserve the compiler and runtime
changes needed by this project without requiring access to a local branch.

1. `0001-amdgpu-fragment-repack.patch` — `v_permlanex16` cross-lane lowering
   and an f32-to-f16 matrix-fragment repack in registers (original commit
   `3fe0a2bb3455ff9e8fb0810ab6b8748e1b09d848`).
2. `0002-preserve-fragment-read-order.patch` — keep reads feeding matrix
   fragments in source order (original commit
   `8b2d1e882d28eb386bccd65fe702bd87119266ce`).
3. `0003-amdgpu-pm4-emulation-query-optional.patch` — a runtime fix, not a
   compiler one. `iree_hal_amdgpu_query_aql_queue_execution_mode` queries
   `HSA_AMD_AGENT_INFO_PM4_EMULATION` (0xA119) and propagates every failure, so
   device initialization fails on any ROCr whose agent attribute enum ends at
   0xA118 — which includes the distribution's HSA 1.18. `INVALID_ARGUMENT` from
   that query means the attribute is unknown, and a runtime that does not know it
   is not emulating PM4, so the patch answers that case from the already-`false`
   default; every other status still propagates. It carries the regression test.
   Everything in this project dispatches through libhrx, so this one is required,
   not optional. Not upstream: it is
   [`pm4-emulation-query-optional`](https://github.com/zacharydenton/hrx-system/tree/pm4-emulation-query-optional)
   on a fork of `ROCm/hrx-system`.
4. `0004-vopd-source-cache-banks.patch` — check VOPD register banks by the
   hardware operand cache. FMAMK addends use SRC2, whose bank mask is 1;
   treating their encoded VSRC1 field as SRC1 admitted illegal dual instructions
   and corrupted vision GELU outputs. The fix covers mixed FMAMK/FMAC pairs in
   both allocation constraints and native emission, while preserving legal
   pairing. Fork commit
   [`675cc43bc`](https://github.com/zacharydenton/hrx-system/commit/675cc43bc).
   See the [reproducer and GPU evidence](../../experiments/vision_gelu_vopd/README.md).

5. `0005-materialize-encoding-config.patch` — materialize exact encoding
   configuration as `encoding.define` before native lowering. Includes i4/i8
   fragment consumers, returned schemas/layouts, and unresolved declarations.
   Fork commit [`2fb396c2f`](https://github.com/zacharydenton/hrx-system/commit/2fb396c2f).
6. `0006-bind-dependent-inline-types.patch` — bind callee dimensions and encoding
   references to call-site values before checking inline argument and return
   types. Exact facts resolve static dimensions without weakening type checks.
   Fork commit [`9e4fff00d`](https://github.com/zacharydenton/hrx-system/commit/9e4fff00d).

From this project's root, create a separate compiler checkout:

```sh
export H3_SOURCE="$PWD"
git clone https://github.com/ROCm/hrx-system.git ../hrx-system-h3
cd ../hrx-system-h3
git checkout --detach c9855b47e96e7eb1cbb5b81b1de973762982ae95
git apply "$H3_SOURCE"/patches/loom/*.patch
```

Follow `BUILDING.md` in that pinned checkout for build prerequisites. Its CMake
entry points support an explicit build directory and AMDGPU target:

```sh
python3 dev.py --cmake-build-dir "$PWD/build" cmake setup
python3 dev.py --cmake-build-dir "$PWD/build" cmake configure \
  -DCMAKE_BUILD_TYPE=Release -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU_TARGETS=gfx1151
python3 dev.py --cmake-build-dir "$PWD/build" cmake build \
  loom-compile loom-format loom-check iree-test-loom iree-benchmark-loom \
  libhrx_src_libhrx_hrx
export LOOM_COMPILE="$PWD/build/loom/src/loom/tools/loom-compile/loom-compile"
cd "$H3_SOURCE"
bash scripts/test.sh
```

`libhrx_src_libhrx_hrx` builds `libhrx.so`. The model normally loads the
pinned HRX bundle, so a local native build is only needed for compiler development.
`LOOM_COMPILE` selects a rebuilt compiler explicitly.

The current bundle contains the compiler built from fork commit
`9e4fff00d244a8b5300d03569addd019ec8e262f` (patches 0001, 0002, 0004–0006).
Compiler SHA-256:
`74a0c9dc5f387e89b85a3cd9d2000644dc0e20a0657d9fe79dfcd627ff5ecdb6`.
Runtime libraries retain their previous bundle bytes; patch 0003 records the
runtime source fix, while the original runtime build provenance remains
unverified. See [bundle setup](../../docs/setup.md).

Validated on 2026-09-09: all six patches apply in order to the pinned revision,
and the resulting compiler sources match the tested fork commits. 550 available
Loom fixture suites pass. Four other source-low suites also fail on a rebuilt
unchanged parent; optional unbuilt test executables were excluded. All 15 H3
GPU kernel tests pass with bitwise source-baseline comparisons and independent
CPU oracles. The patches retain their upstream source license headers.
