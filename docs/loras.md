# H3 LoRAs

Pass a safetensors checkpoint with `--lora`. Repeat the flag to combine adapters;
each defaults to strength 1. `--lora-strength INDEX=VALUE` selects a one-based
slot. Strengths may be positive or negative; zero disables that file entirely.

```sh
h3 --lora camera.safetensors --lora style.safetensors \
  --lora-strength 1=0.8 --lora-strength 2=-0.25 \
  --first-frame input.png --out clip.mp4 < prompt.txt
```

The loader accepts the native H3 keys:

```text
diffusion_model.blocks.{0..49}.{projection}.lora_A.weight
diffusion_model.blocks.{0..49}.{projection}.lora_B.weight
diffusion_model.token_refiner.blocks.{0..1}.{projection}.lora_A.weight
diffusion_model.token_refiner.blocks.{0..1}.{projection}.lora_B.weight
```

`projection` is `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, or `mlp.fc2`.
Any subset is supported, with independent positive ranks and FP16, BF16, or FP32
weights. A is `[rank, input]`, B is `[output, rank]`. An optional floating scalar
`{prefix}.alpha` supplies `alpha / rank`; without it, the multiplier is 1.
The effective multiplier also includes the chosen strength. Unknown tensors,
unpaired factors, invalid dimensions, and nonfinite values are errors. Other
model architectures and naming conventions are not translated.

The original quantized base weights remain intact. Native BF16 branches evaluate
the weighted low-rank updates in the original weight basis before the projection's
activation or residual gate. Multiple adapters concatenate their ranks, giving
the sum of their updates. Rank projections, deltas and SwiGLU products retain
FP32 range; matrix operands narrow to BF16. Ranks are padded for the kernels; the combined rank per projection is limited to 32,768.
Missing projections use their original base path. Files must remain unchanged
for the lifetime of a session. Custom LoRAs cannot be combined with a Turbo preset.

## 360 Orbit example

The [360 Orbit LoRA](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA)
uses FL2VA and matching first and last keyframes. Download the tested revision
into the standard Hugging Face cache with the optional Hub CLI:

```sh
ORBIT_LORA=$(hf download pablodawson/MiniMax-H3-360-Orbit-LoRA \
  minimax_h3_flf2v_lora_v1.safetensors \
  --revision 5ddbc2dbbe95edbbdaf5017c3e934b1d01791697)
```

Save the author's prompt from the model card as `orbit-prompt.txt`, then run:

```sh
h3 --lora "$ORBIT_LORA" \
  --first-frame input.png --last-frame input.png \
  --width 768 --height 768 --frames 73 --steps 29 --seed 2026 \
  --out orbit-with-audio.mp4 < orbit-prompt.txt
ffmpeg -i orbit-with-audio.mp4 -map 0:v:0 -c:v copy -an orbit.mp4
```

`--steps 29` means 28 model evaluations. `--last-frame` requires a first frame
and more than one output frame; its index follows the rounded output length.
Different first/last images are also supported. h3 jointly samples video and
audio; the final command removes audio from the MP4 to make a silent preview.
It does not change audio conditioning during sampling.

## Rust API

```rust,no_run
use h3_hrx::{adapter::Lora, Config, Session};

let config = Config {
    loras: vec![Lora::new("orbit.safetensors", 1.0)],
    ..Default::default()
};
// Safety: the application keeps every configured checkpoint immutable.
let mut session = unsafe { Session::new(config) }?;
# Ok::<(), h3_hrx::Error>(())
```

Condition the first and last frames with two `Keyframe` values whose
`frame_index` values are 0 and the output frame count minus one. LoRAs are loaded
with the DiT on demand and reopened with it after residency eviction.

## Validation

`scripts/test.sh --adapters` requires the base DiT, both pinned Turbo adapters,
and the Orbit revision above in the cache. It checks the actual A/B products
against independent CPU sums and checks eager execution against graph replay
for both DiT and refiner stacks. A mixed, partial adapter case exercises signed
strengths, overlapping projections, different ranks, and the base fallback.
CPU tests cover headers, payload validation, alpha, disabled files, and CLI
arguments. See [test coverage](testing.md) for the whole-block CPU oracle.
