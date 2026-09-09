# Native test coverage

`scripts/test.sh --cpu` runs formatting, Clippy, workspace tests and C-header drift checks.
`scripts/test.sh --gpu` additionally runs `h3/tests/kernels.rs` and the resident
Euler and conditioning unit tests on gfx1151.
The native tests compile through `hrx::loom::Compiler`, upload owned buffers,
dispatch through `hrx::Stream`, and read results back through HRX staging. They
are ignored by default; explicitly running them requires working hardware and
the provisioned native bundle. No Python or Torch dependency is involved.

The GPU suite compares independent scalar CPU references against:

- GroupNorm and SiLU at zero, small and normal variance.
- Four experimental 32-key attention layouts with the softmax maximum in the upper key tile.
- The attention stems the host actually selects — `mha`, `mha8`, `mha64`, `mha648`, `mha64t32` and
  the text encoder's causal `gqa8c` — against scaled dot-product attention in f64.
- `prepare_qk_i8`: the Hadamard rotation, int8 quantisation, packing and scales that the int8
  attention consumes, and `attention_i8qk_mha` against the attention those operands define.
- Scalar and four-output audio convolutions at eleven boundary lengths, with
  padding, dilation, residual accumulation and untouched output guards.
- Float32 matrix multiplication past the former 32,768-row limit.
- BF16 vision matmul with bias, residual scaling, tanh GELU and erf GELU.
- RMSNorm, LayerNorm and plain FP16/BF16 preparation at three widths, including
  large inputs, per-row classes and padded output strides.
- All three rotary Q/K normalization layouts, including grouped heads and exact V copying.
- INT4/INT8 GEMM families, both tile sizes, bias, residual classes, SwiGLU ordering
  and padded input strides on kernels that support them.
- The f16 and BF16 GEMM families across all three modes and both epilogues.
- Packed INT4/INT8 attention with four and eight waves, including the INT4 skip decisions.
- INT4/INT8 preparation against a dense Hadamard reference, including scales and packing.
- Video convolution with causal time padding, reflected spatial padding, strides,
  residual addition and untouched output guards.

Workspace unit tests cover model shapes, checkpoint layouts, CPU sampling,
tokenization, compiler/source behavior, dispatch bounds, and C ABI contracts.
The C smoke test in the shared HRX repository builds against generated H3 and
Krea headers and loads both model libraries in one process. Rustler remains an
application adapter under `examples/rustler`.

Loom sources are maintained directly. Python generators, reference model
implementations, wrappers and one-off studies are retired. Historical reports
remain historical; they are not automatically revalidated by this suite.
Whole-model parity lives in `scripts/parity.py`, outside this suite and outside
`scripts/test.sh`: it needs the checkpoints, a device and dumps produced inside the
ComfyUI container by `scripts/comfy_dump.py`, and it takes about eleven minutes.
Run `python3 scripts/parity.py gate --require` before a release. This suite
establishes the numerical cases above; that script establishes full-model parity.

Still uncovered here, in rough order of how much they matter: the large-token int4 and
head-major attention variants (`attention_i4qkl*`, `attention_i4qksl*`, `attention_i8qkhm_mha8_k64`,
`attention_mha64hm32`) with their `prepare_qk_i4` and `prepare_qk_i8hm` operands;
the wide and fast decoder GEMMs; the fused decoder QKV GEMM; `norm_mod_f32`; and
the smaller shape kernels (`layernorm_f32`, `layernorm_f16_f32`, `transpose_f16`,
`transpose_f32`, `gn_stats_f16`, `prepare_plain16_i8`, `conv1d_s_f32`,
`rope2d_qkv_f16`, `matmul_bias_f16_wmma_af16_cf16`). The resident Euler sampler is checked bit-for-bit
against the CPU as a unit test, and its five reference denoise cases were compared
byte-for-byte against the host sampler it replaced; neither check is in a committed
harness, so re-run the comparison by hand when that path changes.

## Comparing kernel families

Production sources use native Loom templates and specialization. The packed
256×128 GEMMs share INT4/INT8 bodies through schema providers and required
unrolling. Float GEMMs share bias epilogues within each element type; preparation
shares its narrowing or Hadamard/packing finish. Four/eight-wave attention,
head-64/head-128 rotary normalization and residual video convolution also share
bodies. The host selects a module and export explicitly; removed filenames have
no aliases or fallback lookup.

This replaces 45 sources with 14 modules, removing about 8,100 Loom lines.
Tile geometry, LDS staging, prefetch and exported binding/configuration contracts
remain the same. Separate families retain different native element types,
specialized wide/fast/fused schedules and floating SwiGLU epilogues. A shared
vision GELU candidate failed bitwise comparison and was excluded.

To reproduce the comparison, extract the pre-consolidation sources:

```sh
mkdir -p build/kernel-baseline
git archive 62f837d h3/kernels | tar -x -C build/kernel-baseline
export H3_KERNEL_BASELINE="$PWD/build/kernel-baseline/h3/kernels"
cargo test -p h3 --test kernels -- --ignored --test-threads=1 --nocapture
H3_KERNEL_TIMING=1 cargo test -p h3 --test kernels preparation -- --ignored --test-threads=1 --nocapture
cargo run --release -p h3 --example compare_gemm -- "$H3_KERNEL_BASELINE"
```

The test harness compares every binding bit-for-bit before the independent CPU
oracle check. Optional timings use ten alternating pairs of resident sequences.
`compare_gemm` checks decoder-sized matrices with both repeated weights and a
ring exceeding 64 MiB; its optional second argument filters export names.
`H3_COMPARE_BATCHES=40` extends its default ten paired batches when investigating
small differences. Run timings with both CPU and GPU idle: they share memory
bandwidth on this device. Investigate repeatable slowdowns above 2%.

For a resident decoder comparison, group the original independent exports into
the new module filenames in the **baseline fixture only**:

```sh
for module in h3/kernels/*_family.loom h3/kernels/gemm_packed_256.loom; do
  sed -n 's/.*export("h3_\([^"]*\)").*/\1/p' "$module" |
    while IFS= read -r entry; do cat "$H3_KERNEL_BASELINE/$entry.loom"; done |
    awk '!/^amdgpu.target/ || !seen[$0]++' > "$H3_KERNEL_BASELINE/$(basename "$module")"
done
cargo run --release -p h3 --example compare_decode -- \
  "$H3_KERNEL_BASELINE" "$VAE_CHECKPOINT" "$LATENTS_F32" 480 864 22
```

Latents are little-endian f32 in `[24, 7, 30, 54]` order for that shape. The
diagnostic loads both decoders once, warms each, alternates ten timed pairs and
requires identical decoded RGB for every pair. It isolates source changes with
the same host and compiler; it does not replace whole-model parity testing.

On gfx1151, the consolidation passed all 15 kernel tests with bitwise baseline
comparison, plus the two resident conditioning/sampler tests. The 480×864,
22-frame decoder comparison produced identical RGB in all ten pairs: median
4.542 s before and 4.544 s after (+0.05%). Decoder-sized GEMM comparisons covered
120 export/shape/cache cases; extending the noisy INT4 SwiGLU cases to 40 pairs
left no repeatable slowdown above 2%. These measurements used native bundle
`750f265ce4fd6a194fbac12a795c96cb19cc9ed3696fd5123c5edd5589a4cd05`, compiler SHA-256
`a2902bba66bec779d95d15f6bac573072c1940dccd34215663c9a59842941dfa`, on 2026-09-09.
