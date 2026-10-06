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

## Independent model references

`scripts/parity.py` is separate from the native suite. It uses cached checkpoints
and invokes the ignored `tests/parity_dump.rs` fixture test through Cargo. It does
not download checkpoints during validation. Install `numpy` and
`huggingface_hub`; model-based references additionally need Torch and, where
applicable, diffusers.

| Command | Reference / requirement |
| --- | --- |
| `python3 scripts/parity.py gate --require` | Full-model gate against ComfyUI dumps from `scripts/comfy_dump.py`; missing fixtures fail |
| `python3 scripts/parity.py dit` | Released unquantized BF16 blocks; cached `transformer/` weights (~62 GB), one layer resident at a time |
| `python3 scripts/parity.py te` | Released Qwen3-VL layers; cached `text_encoder/` weights (~62 GB) |
| `python3 scripts/parity.py convert` | ConvRot weight conversion against released weights; no GPU |
| `python3 scripts/parity.py vae` | FP16 native decoder against diffusers and the released VAE (~10 GB) |

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

Use [Criterion](performance.md) for timing comparisons. Compiler regression probes
remain under `experiments/`, including the
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
python3 scripts/test_cache_calibrate.py
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
See [world validation](world.md#validation) for visual qualification guidance.
