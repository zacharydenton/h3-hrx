# Sharing GEMM families in Loom

The installed compiler can express a shared i4/i8 matrix motif with ordinary
Loom templates, encoding values, shape specialization, and required loop
unrolling. A Python generator or Rust source builder is unnecessary for this
experiment. Two compiler gaps constrain where the template boundaries go.

This is a CPU compilation experiment, not a replacement for the production
256x128 GEMMs. It does not launch kernels, check numerical output, or establish
performance parity with their LDS staging and prefetch schedule.

## Working example

[`packed.loom`](packed.loom) contains one matrix body and one K loop:

- `template.apply` selects an i4 or i8 encoding provider using a `where`
  constraint on the configured register width.
- A shared dot template accepts views and builds
  `vector<[%registers]xi32>` packets inside its body. Both formats use packed
  i32 storage and i32 accumulators.
- `scf.for ... unroll` expands the 64-byte packet traversal into eight i4 or
  four i8 matrix operations.
- A separate epilogue family selects identity or doubling. These deliberately
  small epilogues test composition, not the model's residual/SwiGLU arithmetic.

The actual emitted gfx1151 machine code was disassembled and checked:

| Format | Epilogue | WMMA instructions | Epilogue integer adds | Branches/calls |
| --- | --- | ---: | ---: | ---: |
| i4 | identity | 8 iu4 | 0 | 0 |
| i4 | double | 8 iu4 | 8 | 0 |
| i8 | identity | 4 iu8 | 0 | 0 |
| i8 | double | 4 iu8 | 8 | 0 |

Register width 3 and epilogue 2 are rejected by the configuration contracts.
The templates and loop have disappeared by target lowering.

Run from the repository root, with `llvm-objdump` on PATH:

```sh
LOOM_COMPILE=/path/to/loom-compile bash experiments/loom_gemm_family/check.sh
```

`LLVM_OBJDUMP` can select a different disassembler. The script saves compiler
logs, pass dumps, artifacts, and disassembly in `build/loom-family-probe/check/`.
It compiles the same source for all four combinations; it emits no Loom source.

Validated on 2026-09-09 with compiler SHA-256
`a2902bba66bec779d95d15f6bac573072c1940dccd34215663c9a59842941dfa`,
from native bundle
`750f265ce4fd6a194fbac12a795c96cb19cc9ed3696fd5123c5edd5589a4cd05`.
The sibling checkout's `build-cuda` compiler has the same digest. These findings
describe that binary; the bundle manifest does not establish a reproducible
source-to-binary provenance.

## Compiler gaps and reproducers

These files parse and verify with `loom-format`, but fail native compilation
with the tested compiler. They are standalone inputs suitable for compiler
bug reports; they are not included in the successful experiment's check script.

### Exact encoding configuration survives into target lowering

```sh
"$LOOM_COMPILE" experiments/loom_gemm_family/encoding_config.loom \
  --backend=amdgpu-hal --target=gfx1151 \
  --config=probe.registers=2 \
  '--config=probe.schema=#encoding.operand<element_format=i4, payload_elements=16, payload_registers=2>' \
  --output=/tmp/encoding-config.hal
```

Fails with `TARGET/003`, `config.get`, and `no_ordinary_uses`. The exact
schema is accepted as configuration, but the encoding-valued read remains in
the IR at `source-to-low`. Replacing that read with the corresponding
`encoding.define` allows compilation, including the dynamic packet width.

The compiler needs to carry an exact configured encoding through its consumers
and eliminate the compile-time read before executable emission. Existing
encoding-config canonicalization coverage in Loom tests `encoding.isa` folding;
it does not exercise a fragment consumer end to end. The working example's
schema providers return `encoding.define` and avoid this gap.

### Specialized vectors do not cross a dynamic template argument boundary

```sh
"$LOOM_COMPILE" experiments/loom_gemm_family/dynamic_argument.loom \
  --backend=amdgpu-hal --target=gfx1151 --config=probe.registers=2 \
  --output=/tmp/dynamic-argument.hal
```

Fails with `LOWERING/044`, `inline-callables`, and `operand_type_mismatch`.
The caller's vector becomes concrete while the provider argument retains its
SSA-bound width. The inliner in
`loom/src/loom/transforms/symbol/inline_callables.c` checks structural type
equality before inlining. This path needs compatible shape specialization and
binding substitution at the callable boundary, with an end-to-end regression.
Passing fixed-shape views and doing the dynamic-width load inside the template
works, as demonstrated by `packed.loom`.

## Applying this to the model

Start with the i4/i8 256x128 pair: share the workgroup schedule and K traversal,
keep packing in small Loom providers, and compose epilogues as templates.
Preserve the existing LDS layout, grouped rasterization, prefetch lifetime,
fragment read order, and accumulator reuse. The motif only proves the relevant
language mechanisms; it does not yet prove the full schedule survives that
refactor. Compare emitted instructions and resource use, then run numerical
and performance checks before switching production dispatch.

This does not demonstrate arbitrary element-type generics. In particular,
f16/bf16 families also change native vector element types, and floating GEMMs
use f32 accumulation. They need a separate composition experiment, potentially
with concrete typed providers around a shared schedule. They should not be
assumed covered by the packed i32 example.

The demonstrated compiler gaps justify focused Loom fixes. They do not justify
reintroducing a model source generator: a useful shared form already compiles.
