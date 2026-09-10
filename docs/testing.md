# Native test coverage

Three tiers, each needing more of the machine than the last:

| | needs | runs |
| --- | --- | --- |
| `scripts/test.sh --cpu` | nothing | formatting, Clippy, the unit tests |
| `scripts/test.sh --gpu` | gfx1151 and a provisioned HRX | the above, plus `tests/kernels.rs` and the resident Euler and conditioning tests |
| `scripts/test.sh --full` | the checkpoints as well | the above, plus `tests/differentials.rs`, about three minutes |

The native tests compile through `hrx::loom::Compiler`, upload owned buffers,
dispatch through `hrx::Stream`, and read results back through HRX staging. They
are ignored by default; explicitly running them requires working hardware and
the provisioned native bundle. No Python or Torch dependency is involved.

## Whole-pipeline digests

`tests/differentials.rs` runs thirty-one cases — seven video decodes across every tiling the
decoder chooses, fourteen audio conversions at its padding boundaries, and five denoising
trajectories over both samplers and the step cache — and compares a SHA-256 of each result against a
constant in the file.

These are the checks that catch what types cannot: a recorded graph missing an edge, a stage moved
to another stream, a path that resolves to the wrong tree. Inputs are generated from a seed by a
generator written out in the test, so nothing is stored; the outputs would be gigabytes and a digest
compares them exactly as well.

`H3_GRAPH=1 cargo test --features internals --test differentials --release -- --ignored --test-threads=1` runs the same cases through the
recorded graphs, which is how the recordings are known to be faithful.

A failed assertion reports the actual and expected digests. Before updating an expected digest
for an intentional numerical change, validate the new output independently with
`scripts/parity.py` against diffusers, then edit the constant explicitly.

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
tokenization, compiler/source behavior and dispatch bounds.
The C smoke test in the shared HRX repository builds against generated H3 and
Krea headers and loads both model libraries in one process. Rustler remains an
application adapter under `clients/rustler`.

Loom sources are maintained directly. Python generators, reference model
implementations, wrappers and one-off studies are retired. Historical reports
remain historical; they are not automatically revalidated by this suite.
Two checks are anchored genuinely upstream, against the weights MiniMax released
and the implementation diffusers ships, rather than against ComfyUI's conversion
or a reimplementation of either.

`python3 scripts/parity.py dit` runs the host's packed rows through the released
**unquantized bf16** blocks, one block resident at a time so it costs about a
gigabyte rather than sixty-two. The host runs ComfyUI's int8 ConvRot conversion of
those same weights, so the agreement is the quantisation and this implementation
together: 0.99999 after one block and 0.99865 after all fifty (text 0.9992, audio
0.9996, video 0.9971). It needs the released `transformer/`, a 62 GB download.

`python3 scripts/parity.py te` walks the host's embedding rows through the
released bf16 Qwen3-VL, the 50 of 64 decoder layers MiniMax-H3 reads before taking
the unnormalised hidden state: 0.99995 after fifty. One layer resident at a time.
It needs the released `text_encoder/`, a 62 GB download.

`python3 scripts/parity.py convert` answers the tensor-level question the other
two cannot: whether ComfyUI's int8 ConvRot conversion carries the weights MiniMax
released. The conversion rotates within 256-channel groups, quantises, and
reorders rows for the kernels, so the stored numbers are deliberately not the
released ones — but a rotation is orthogonal, so row norms survive it, and
compared as a multiset they survive the reordering too. All 200 quantised tensors
agree to 1.1e-4; the 150 attention projections keep their rows in place, so their
group norms match as well, while the SwiGLU projections are the same weights
permuted. It needs no GPU.

`python3 scripts/parity.py vae` is the other end: the host's tiled f16 decoder against diffusers' `AutoencoderKLMiniMaxH3` reading the
weights MiniMax released, rather than against ComfyUI's conversion or a
reimplementation. It measures the f16 narrowing plus whatever the Loom decoder
does differently, and clears 62.5 dB PSNR at 384x320x22. It needs `diffusers`,
`torch` and the released VAE in diffusers layout (`~/h3-models/vae`); no
transformer, so it costs 10 GB rather than 62.

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
unrolling. Float GEMMs share their multiply loop across plain, residual, and
both SwiGLU orderings within each native element type; preparation shares its
narrowing or Hadamard/packing finish. Four/eight-wave attention,
head-64/head-128 rotary normalization and residual video convolution also share
bodies. The host selects a module and export explicitly; removed filenames have
no aliases or fallback lookup.

Across both passes, 52 sources become 15 modules, removing 10,086 Loom lines.
Tile geometry, LDS staging, prefetch and exported binding/configuration contracts
remain the same. Separate families retain different native element types and
specialized wide/fast/fused schedules. Vision bias, tanh GELU, and erf GELU
now share a BF16 module. That step required the
[VOPD compiler fix](../experiments/vision_gelu_vopd/README.md), which is included
in the pinned native bundle.

To reproduce the comparison, extract the pre-consolidation sources:

```sh
mkdir -p build/kernel-baseline
git archive 62f837d kernels | tar -x -C build/kernel-baseline
export H3_KERNEL_BASELINE="$PWD/build/kernel-baseline/kernels"
cargo test --features internals --test kernels -- --ignored --test-threads=1 --nocapture
H3_KERNEL_TIMING=1 cargo test --features internals --test kernels preparation -- --ignored --test-threads=1 --nocapture
cargo run --release --features internals --bin h3-dev -- compare-gemm "$H3_KERNEL_BASELINE"
cargo run --release --features internals --bin h3-dev -- compare-vision "$H3_KERNEL_BASELINE"
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
for module in kernels/*_family.loom kernels/gemm_packed_256.loom; do
  sed -n 's/.*export("h3_\([^"]*\)").*/\1/p' "$module" |
    while IFS= read -r entry; do cat "$H3_KERNEL_BASELINE/$entry.loom"; done |
    awk '!/^amdgpu.target/ || !seen[$0]++' > "$H3_KERNEL_BASELINE/$(basename "$module")"
done
cargo run --release --features internals --bin h3-dev -- compare-decode \
  "$H3_KERNEL_BASELINE" "$VAE_CHECKPOINT" "$LATENTS_F32" 480 864 22
```

Latents are little-endian f32 in `[24, 7, 30, 54]` order for that shape. The
diagnostic loads both decoders once, warms each, alternates ten timed pairs and
requires identical decoded RGB for every pair. By default it compares source
changes with the same host and compiler. Set `H3_BASELINE_LOOM_LIBRARY` to the
previous compiler shared library to include a compiler upgrade in the comparison.
Neither mode replaces whole-model parity testing.

The first consolidation pass on gfx1151 passed all 15 kernel tests with bitwise
baseline comparison, plus the two resident conditioning/sampler tests. The 480×864,
22-frame decoder comparison produced identical RGB in all ten pairs: median
4.542 s before and 4.544 s after (+0.05%). Decoder-sized GEMM comparisons covered
120 export/shape/cache cases; extending the noisy INT4 SwiGLU cases to 40 pairs
left no repeatable slowdown above 2%. These measurements used native bundle
`750f265ce4fd6a194fbac12a795c96cb19cc9ed3696fd5123c5edd5589a4cd05`, compiler SHA-256
`a2902bba66bec779d95d15f6bac573072c1940dccd34215663c9a59842941dfa`, on 2026-09-09.

The second pass uses bundle
`34591d78d625f9c696657820d04615f3f55a134010a9354c0e455a4c2e60caa0`, compiler SHA-256
`74a0c9dc5f387e89b85a3cd9d2000644dc0e20a0657d9fe79dfcd627ff5ecdb6`.
All 15 GPU kernel tests pass again, including every float GEMM export and all
three shared vision epilogues, with bitwise baseline and CPU-oracle checks.
Workspace tests, formatting and Clippy pass. The compiler's available fixture
corpus passed 550 suites in that historical bundle; its validation record is
preserved in Git history. Current compiler validation lives in
[HRX](https://github.com/zacharydenton/hrx-rs/blob/main/patches/loom/README.md).

Resource checks across 36 float export/shape combinations retain the same VGPR
counts, 55,296 bytes of LDS, no scratch memory, and 64 static WMMA instructions.
SGPR counts are unchanged except unbiased SwiGLU, which drops from 24 to 22.

Second-pass resident timings on 2026-09-09 compare original sources with shared
modules using the same corrected compiler. Every case checks bitwise output:

| Family | Shape/cache cases | Candidate time change |
| --- | ---: | ---: |
| Vision bias / GELU / erf GELU | 30 | −2.56% to +1.20% |
| FP16 GEMM | 36 | −1.08% to +0.43% |
| BF16 GEMM | 36 | −0.39% to +1.54% |

Vision and FP16 use ten paired batches; BF16 uses forty. Competing memory-heavy
jobs caused larger BF16 outliers in earlier runs. The table uses the final full
BF16 run with its plain exports rechecked as a group after competing GPU work
ended. No slowdown above 2% repeated. These are source-consolidation comparisons,
not peak-throughput measurements.

The resident 480×864, 22-frame decoder also produces identical RGB in all ten
pairs when comparing the original sources and previous bundle's compiler with
the new sources and compiler. Median times were 4.884 s before and 4.862 s after
(−0.44%). One baseline execution took 31.533 s on the shared machine, so this
run establishes output equivalence and gives only a coarse timing check; it
does not establish an end-to-end speedup.
