# Sharing GEMM families in Loom

Production packed GEMMs share i4/i8 schedules using encoding providers and
required loop unrolling. Each native float element type shares one multiply
schedule across plain, residual, and SwiGLU epilogues. No model source generator
is needed. See [production validation](../../docs/testing.md#comparing-kernel-families).

## CPU compilation probes

```sh
LOOM_COMPILE=/path/to/loom-compile bash experiments/loom_gemm_family/check.sh
```

The script compiles and inspects gfx1151 artifacts without opening the GPU.
`LLVM_OBJDUMP` selects a disassembler; outputs go to `build/loom-family-probe/check/`.
It checks:

- `packed.loom`: encoding providers select i4/i8 packets inside a shared matrix
  body. Required unrolling emits eight i4 or four i8 WMMAs, with zero or eight
  epilogue adds and no remaining branches or calls.
- `encoding_config.loom`: an exact configured schema feeds a native fragment.
  Both i4 and i8 emit the expected single WMMA instruction.
- `dynamic_argument.loom`: vector arguments cross a dependent template boundary
  at widths two and four.
- Unsupported register widths and epilogue selectors are rejected.

The latter two probes originally exposed compiler bugs. Patches
[0005](https://github.com/zacharydenton/hrx-rs/blob/main/patches/loom/0005-materialize-encoding-config.patch) and
[0006](https://github.com/zacharydenton/hrx-rs/blob/main/patches/loom/0006-bind-dependent-inline-types.patch) fix them in the
current bundle. Exact encoding reads become `encoding.define`; callable types
are compared after SSA binding and exact-fact refinement. Unresolved configs and
runtime dimensions retain their ordinary semantics.

Validated on 2026-09-09 with compiler SHA-256
`74a0c9dc5f387e89b85a3cd9d2000644dc0e20a0657d9fe79dfcd627ff5ecdb6`.
[HRX's patch record](https://github.com/zacharydenton/hrx-rs/tree/main/patches/loom) identifies its source.

These probes establish language and code-generation behavior. Numerical and
performance checks remain necessary for production schedule changes. Native
f16/bf16 element types retain separate typed families; the shared packed-i32
example does not demonstrate arbitrary element-type generics.
