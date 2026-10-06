# Native test coverage

Run commands from the repository root. GPU tests are ignored by default and
require working hardware when explicitly selected.

| Command | Needs | Coverage |
| --- | --- | --- |
| `scripts/test.sh --cpu` | Rust toolchain | Formatting, Clippy, workspace tests |
| `scripts/test.sh --gpu` | gfx1151 and HRX bundle | CPU tier plus kernel, conditioning, sampler and graph checks |
| `scripts/test.sh --full` | Base/Ref2VA checkpoints and pinned world adapter | GPU tier plus pipeline digests, session lifecycle, RefMod and world tests |
| `scripts/test.sh --adapters` | Base DiT, both Turbo adapters and [Orbit LoRA](loras.md) | GPU tier plus low-rank projection and adapted eager/graph tests |

Native tests use HRX directly and need no Python. Optional independent model
references use Python, Torch and cached checkpoints.

## Kernel and host coverage

`tests/kernels.rs` compares GPU output against scalar CPU oracles for:

- GroupNorm/SiLU, RMSNorm, LayerNorm, rotary Q/K normalization and V copying.
- FP16/BF16 GEMMs, INT4/INT8 GEMMs, bias, residual gates, SwiGLU ordering and
  padded inputs crossing workgroup boundaries.
- FP32 feed-forward range, wide Hadamard preparation, quantization scales,
  invalid rows and activation binding sizes.
- BF16 vision projections with bias, residual scaling and both GELU variants.
- Causal/grouped and packed attention, skip decisions, directed world routing,
  ragged rows and head-major INT8 sequences through 8193 tokens.
- Audio convolution, upsampling and Snake FIR phases, with padding, dilation,
  residuals, output guards and ordered FP32 accumulation.
- Video convolution with causal time padding and reflected spatial padding.
- Graph replay, fused operand preparation and reused padded capacity.

Host tests cover shapes, checkpoint layouts, sampling, tokenization, dispatch
bounds, model selection, cancellation, residency and prompt preparation. The
[ComfyUI tile fixtures](../tests/fixtures/tiles/README.md) exercise complete
spatial composition, including intersecting/triple overlaps and temporal lookahead.

Coverage is incomplete for large-token INT4 and FP16 head-major attention,
wide/fast/fused decoder GEMMs, and several small layout kernels. Passing these
cases does not establish arbitrary-shape correctness or full-model parity.

## Whole-pipeline digests

`tests/differentials.rs` compares deterministic video/audio decodes and denoising
trajectories against recorded SHA-256 digests. Cache tests check forced-full
execution and actual residual reuse. Run the same differential cases through
graphs with:

```sh
H3_GRAPH=1 cargo test --locked --test differentials --release -- --ignored --test-threads=1
```

A digest change is a numerical change. Before updating expected values, compare
with an independent reference and document why the new result is correct.
Historical validation is not rerun automatically by the CPU suite.

### Decoder and reference checks

The October 2 spatial-compositor and five-frame lookahead fixes reduced a
320×320×22 ComfyUI comparison from 0.462 to 0.167 RGB8 RMSE. Other same-latent
checks at 39/56/73 frames stayed below 0.19 RMSE with maximum error 3; stereo
audio at 65/255/256/257 latent frames differed by less than 9e-7 per sample.

The library's October 3 checkpoint-selection fix is tested against explicit
Ref2VA and across all residency policies. A two-image/audio reference case at
320×320×124, seed 7 and 20 ResMultistep evaluations used identical noise and
shared text embeddings with ComfyUI `a7169322`:

| Native checkpoint | Video latent cosine | Audio latent cosine |
| --- | ---: | ---: |
| Previous FL2VA default | 0.991144 | 0.726789 |
| Corrected Ref2VA selection | 0.999855 | 0.981867 |

That synthetic comparison did not reproduce the reported flashing blocks.
Precision and reference-augmentation RNG differences remain comparison limits.

### Periodic artifacts and FP32 range

The saved 384×256×56 portrait/voice clip reproduces frame-17/34 pulses in both
ComfyUI's FP16 decoder (`a7169322`) and MiniMax's original FP32 decoder
(`42ed227e`). Native RGB8 RMSE was 0.1985 against ComfyUI and 0.1469 against the
original decoder. Tracing confirmed matching chunk trimming and blending;
a repeated-static-frame control also had smaller periodic reconstruction changes.
These results exclude the conversion and FP16 decoder precision as explanations
for this clip, but do not establish denoiser parity. The
[comparison record](benchmarks/20261004-refmod-flicker.json) retains measurements
and source hashes.

The October 6 FP32 feed-forward fixes preserve activations reaching 287,332,
remove FP16 Hadamard scratch and stop clamps from masking NaN/Inf. The real
replay produced finite latents under a 32 GiB budget. Fixed baseline latents
decode byte-identically; the temporal pulses remain. See the
[FP32 validation record](benchmarks/20261006-fp32-feedforward.json).

## Independent model references

`scripts/parity.py` is separate from the native suite. It uses cached checkpoints
and does not download them during validation. Install `numpy` and
`huggingface_hub`; model-based references additionally need Torch and, where
applicable, diffusers.

| Command | Reference / requirement |
| --- | --- |
| `python3 scripts/parity.py gate --require` | Full-model gate against ComfyUI dumps from `scripts/comfy_dump.py`; missing fixtures fail |
| `python3 scripts/parity.py dit` | Released unquantized BF16 blocks; cached `transformer/` weights (~62 GB), one layer resident at a time |
| `python3 scripts/parity.py te` | Released Qwen3-VL layers; cached `text_encoder/` weights (~62 GB) |
| `python3 scripts/parity.py convert` | ConvRot weight conversion against released weights; no GPU |
| `python3 scripts/parity.py vae` | FP16 native decoder against diffusers and the released VAE (~10 GB) |

Recorded checks reached 0.99865 cosine after 50 DiT blocks, 0.99995 after 50 text
layers, and over 62.5 dB decoder PSNR at 384×320×22. All 200 quantized tensors
passed the conversion norm check within 1.1e-4. These historical measurements
include quantization differences and apply to their tested inputs and versions.

For an independent whole-block adapter check:

```sh
H3_ADAPTER_BLOCK_FIXTURE=build/adapter-blocks scripts/test.sh --adapters
python3 scripts/adapter_block_reference.py build/adapter-blocks
```

The second command uses Torch on the CPU with original tensor order. It compares
QKV, attention, complete blocks and projections supplied with native attention
input. Use a Python environment with the dependencies of the reference tool.

## Comparing kernel families

Production Loom modules share schedules across compatible exports. To compare
an arithmetic-preserving change, snapshot a baseline with matching bindings and
configuration, then make one source file per export for the test harness:

```sh
# Set BASELINE_COMMIT to the commit before your change.
mkdir -p build/kernel-baseline
git archive "$BASELINE_COMMIT" kernels | tar -x -C build/kernel-baseline
export H3_KERNEL_BASELINE="$PWD/build/kernel-baseline/kernels"
python3 - <<'PY'
import os, re
from pathlib import Path
root = Path(os.environ['H3_KERNEL_BASELINE'])
for path in list(root.glob('*.loom')):
    source = path.read_text()
    for name in re.findall(r'export\("h3_(\w+)"\)', source):
        (root / f'{name}.loom').write_text(source)
PY
cargo test --locked --test kernels --release -- --ignored --test-threads=1 --nocapture
```

The harness compares every binding byte-for-byte before its CPU-oracle check.
Use a filtered test if only some exports have a compatible baseline. Intentional
precision changes need independent validation instead of byte equality.

```sh
H3_KERNEL_TIMING=1 cargo test --locked --test kernels --release preparation -- --ignored --test-threads=1
cargo run --release --bin h3-dev -- compare-gemm "$H3_KERNEL_BASELINE"
cargo run --release --bin h3-dev -- compare-vision "$H3_KERNEL_BASELINE"
cargo run --release --bin h3-dev -- compare-decode \
  "$H3_KERNEL_BASELINE" "$VAE_CHECKPOINT" "$LATENTS_F32" 480 864 22
```

Timings use alternating resident runs with output checks. `compare-gemm` accepts
an export-name filter and tests repeated weights plus a ring exceeding 64 MiB.
`H3_COMPARE_BATCHES=40` extends its default ten pairs. The decoder example expects
little-endian FP32 latents shaped `[24,7,30,54]`; its baseline must contain the
module filenames selected by the current host. `H3_BASELINE_LOOM_LIBRARY` selects
a different baseline compiler. Measure on an idle CPU/GPU.

The [September consolidation measurements](https://github.com/zacharydenton/h3-hrx/blob/3399b13/docs/testing.md#comparing-kernel-families)
retain old compiler hashes and timing data. Compiler regression probes remain
under `experiments/`, including the
[VOPD reproduction](../experiments/vision_gelu_vopd/README.md).

## Optional prompt generation and reference presentation

CPU tests use local HTTP stubs for prompt formatting, repair limits, multimodal
payloads, endpoint errors and credential redaction. CLI tests run against an
empty offline model cache. No live LLM service is required.

Pinned [presentation fixtures](../tests/fixtures/presentation/README.md) check
video patches, token IDs, odd-frame padding, timestamps and modality tags.
RefMod checks cover strengths, disabled members, copy sharing, ordering,
original sources, capability preflight and media budgets.

```sh
cargo test --locked --lib --release prompt::tests::native_refmod_ -- --ignored --test-threads=1
cargo test --release --test refmod effective_refmod_reconstruction -- --ignored --test-threads=1
cargo test --release --test refmod presented_video -- --ignored --test-threads=1
python3 scripts/test_optimization_benchmark.py
```

Hardware checks reconstruct visual/audio members and verify that changing a
presented video frame changes generation. They validate wiring and numerical
behavior; prompt quality needs evaluation with the configured endpoint.

## H3-World

`tests/world.rs` checks all 512 keyboard states against pinned upstream tokens,
resize fixtures and CLI validation. GPU tests cover directed attention and
left/right/left requests in one session. Rollout tests cover persistence,
branching, last-frame handoff, seed/frame accounting, rollback and writer locks.

`world_rollout_decodes_and_resumes` runs two decoded segments through a budgeted
session. These tests are in the full tier and require the pinned world adapter.
See [world validation](world.md#validation) for observed action response and
remaining visual-quality limits.
