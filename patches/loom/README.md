# Loom compiler used by this project

This project uses three local changes on top of `ROCm/hrx-system` commit
`c9855b47e96e7eb1cbb5b81b1de973762982ae95` — two in the compiler, which the
measured kernels need, and one in the runtime, without which nothing here starts
on a stock ROCr. The patches preserve them without requiring access to a local branch:

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

`libhrx_src_libhrx_hrx` builds `libhrx.so`. Nothing here links it, and nothing
here needs this build at all: the model and its tests take a digest-verified
`libhrx` and `loom-compile` from the HRX bundle. What this checkout is for is
rebuilding that compiler — these three patches are the only record of how it
differs from upstream, and two of them are codegen changes the recorded kernel
measurements depend on. `LOOM_COMPILE` substitutes the result for the bundle's.

The three patches reproduce the committed compiler source used for the recorded
measurements. A compiler rebuild from a fresh checkout has not been repeated
recently. The patches retain their upstream source license headers.
