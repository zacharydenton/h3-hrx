# Native patch ownership

Compiler patches, the public upstream pin, build instructions and current
validation live in [hrx.rs](https://github.com/zacharydenton/hrx.rs/tree/main/patches/loom).
H3 loads that compiler in process; `HRX_LOOM_LIBRARY` selects a development
`libloomc.so`. Compiler patches are no longer duplicated here.

The retained `0003-amdgpu-pm4-emulation-query-optional.patch` records the runtime
fix used by the original bundle: an unknown PM4-emulation agent attribute must
not prevent device initialization. Its source branch is
[`pm4-emulation-query-optional`](https://github.com/zacharydenton/hrx-system/tree/pm4-emulation-query-optional).
It is historical source documentation, not the patch set for the current bundle.

The previous compiler bundle used fork commit
[`9e4fff00d`](https://github.com/zacharydenton/hrx-system/commit/9e4fff00d).
Its patches and validation record remain available in this repository's Git
history. See [bundle setup](../../docs/setup.md) for the current release.
