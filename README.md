# minimax-h3-loom

MiniMax H3 (Hailuo 3.0), the open-weights text/image/audio-to-video-with-sound model, running on
AMD's Strix Halo (Radeon 8060S, `gfx1151`) with **every GPU kernel written in
[Loom](https://github.com/ROCm/hrx-system)**, AMD's kernel language, and a C host. No PyTorch,
no Triton, no vendor libraries at run time: the runtime dependency is the HIP runtime API for
module load, memory and launch.

It produces the same clips as ComfyUI's reference implementation (residual stream cosine 0.999
through all 50 blocks against ComfyUI's own run) and, on the same GPU, it is 3.9x faster on a
5-second 480p clip and 7.5x faster at 768p.

```sh
h3 ref.jpg voice.wav < prompt.txt        # references by extension, prompt on stdin, clip.mp4 out
```

![fox](docs/media/fox_480p_5s_strip.jpg)

## Numbers

Per model evaluation, int8 path, Radeon 8060S, idle box (`H3_PROFILE=1`; ComfyUI measured on
the same GPU with `tools/bench_comfyui_h3.py` in the Strix Halo ComfyUI image):

| clip | this repo | ComfyUI, same GPU | speedup |
| --- | ---: | ---: | ---: |
| 1344x768, 124 frames (5 s) | 101 s | 771 s | 7.5x |
| 864x480, 124 frames (5 s) | 26.7 s | 103 s | 3.9x |
| 864x480, 22 frames (1 s) | 4.2 s | 3.8 s | parity |

A full 5-second 480p clip at the default 30 evaluations is about 15 minutes end to end; the same at
768p about 55 minutes. The only public Strix Halo report for this model, ComfyUI on the same
checkpoint ([Comfy-Org/MiniMax-H3 discussion 33](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/33)),
is 70 to 89 s per iteration at 480p against 26.7 s here.

Kernel-level, 56 heads x 128, measured in `docs/`:

| kernel | this repo | reference on the same GPU |
| --- | ---: | ---: |
| attention, int8 QK^T, f16 PV, 37.7k tokens | 36.9 TFLOP/s | aotriton flash (torch SDPA, contiguous) 30.8; CK-tile fp16 38.7 |
| int8 GEMMs, H3 shapes at 37.7k rows | 41 to 45 TOPS | comfy_kitchen's int8 kernel 27 to 42; measured peak 54 |

## Requirements

- **Hardware:** an AMD Strix Halo APU (Ryzen AI Max, Radeon 8060S). The int8 model, its text
  encoder and the VAEs occupy about 48 GB of memory before activations; 128 GB is what this was
  developed and measured on, 64 GB is the realistic floor.
- **ROCm** with a HIP runtime for `gfx1151` (ROCm 7.1x; `scripts/env.sh` documents the runtime
  quirk on Arch).
- **Loom** from [ROCm/hrx-system](https://github.com/ROCm/hrx-system), built with `loom-compile`
  (`scripts/env.sh` points at the build directory). Kernels are compiled on first use for each
  shape into `build/kernel_cache`.
- **ffmpeg** for the `h3` command (input decoding, output muxing).
- **Python 3 with ROCm PyTorch, diffusers and transformers** only for exporting the weights and
  running the reference tests. The clip itself never touches Python.

## Weights

Two downloads, then the exports. The DiT blocks come from ComfyUI's int8 ConvRot checkpoint
(`pruned_int8_convrot` from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3),
21 GB) and its int8 Qwen3-VL-32B text encoder; the VAEs, tokenizer and schedules come from the
original [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) release
(`scripts/download.sh` fetches the FL2VA partition into `~/h3-models`).

| export | tool | on disk |
| --- | --- | ---: |
| DiT blocks, int8 rows | `tools/export_weights.py --bits 8` | 18 GB |
| text encoder, int8 | `tools/export_te.py` | 23 GB |
| conditioning tables, embedders, refiner, heads, audio decoder | `tools/export_glue.py` | 2.7 GB |
| video VAE decoder, int8 | `tools/export_vae.py --bits 8` | 2.3 GB |
| video VAE encoder, vision tower, audio encoder (references) | `tools/export_vae_encoder.py`, `tools/export_vision.py`, `tools/export_audio_encoder.py` | 2 GB |
| ref2va blocks and glue (reference-conditioned clips) | the same two tools with `H3_CKPT=<ref2va checkpoint>` and `--out ..._ref2va` | 21 GB |

Optional: `tools/export_weights.py --bits 16` for the pruned bf16 checkpoint's rows in f16 (36 GB,
`--precision bf16`), `tools/gptq_export.py` for GPTQ int4 blocks (9 GB, `--precision int4`, preview
quality only).

The weights are under the MiniMax H3 Community License, which permits open-weight use in the USA,
EU, UK and South Korea; other regions apply to MiniMax for a licence. Nothing in this repository is
derived from ComfyUI's code, which is GPL; ComfyUI is used only as the measurement oracle inside
its own container image.

## Build

```sh
source scripts/env.sh          # LOOM_COMPILE, the ROCm runtime, LOOM_TARGET=gfx1151
bash scripts/build_host.sh     # build/libh3pipe.so, build/h3, build/h3pipe, build/loomrun
ln -s "$PWD/build/h3" ~/.local/bin/h3
```

## Use

**The `h3` command.** Positional files are references by extension: images become `<Picture i>`
(encoded by the video VAE's encoder, presented to the text encoder through the vision tower),
audio files become `<Audio j>` (the audio VAE's encoder). The prompt comes from stdin or `-p`.
Any format ffmpeg reads; audio is resampled to 32 kHz stereo.

```sh
h3 ref1.jpg voice.wav < prompt.txt
h3 --first-frame fox.png -p "A red fox on a mossy log turns to the camera ..." --out fox.mp4
h3 -p "..." --frames 124 --steps 31 --width 864 --height 480 --seed 0
h3 --help
```

Defaults are the stock ComfyUI workflow settings: `res_multistep` on the `simple` schedule, 30
evaluations (`--steps 31` sigma grid points; ComfyUI's workflows use 20), no CFG, shifts 12/3,
`--precision int8`. The clip lands as `<out>.mp4` with `<out>.wav` beside it. Sizes are multiples
of 32; frame counts snap to 17n + 5 (22, 39, ..., 124).

**From Python.** `tools/pipeline_c.py` drives the same library through ctypes
(`h3pipe_loom.py`) with the same flags (`--ref-image`, `--ref-audio`, `--first-frame`,
`--latents-out`, `--no-decode`).

**From C.** `host/h3pipe.h` is the whole pipeline behind a C ABI: `h3pipe_create` (weight
directories, kernel sources, cache, the `loom-compile` path), `h3tok_encode` (text to ids, the
Qwen2 byte-level BPE in C), `h3pipe_encode_video` / `h3pipe_encode_audio` / `h3pipe_vision_embed`
(references), `h3pipe_denoise` and `h3pipe_denoise_refs` (ids and references to model-space
latents, with a progress callback), `h3pipe_decode_video` and `h3pipe_decode_audio`.
`docs/abi.md` is the contract (sizes, layouts, references, errors, threading); `examples/` has the same
minimal client in C, Rust (no bindgen) and Go (cgo), which produce byte-identical clips; `host/h3_cli.cpp`
is the complete client.

**Knobs.** `H3_PROFILE=1` prints per-stage times after every step; `H3_TRACE=1` prints and
synchronises every launch; `--precision bf16` runs the pruned bf16 checkpoint's rows in f16 at
about the int8 speed; `--precision int4` is a 2x-per-step preview path that ghosts keyframe and
reference clips (do not use it for conditioned generation); `--attn f16` restores f16 attention.

## Quality

The gate is ComfyUI's own run of the same step, dumped from inside its image
(`tools/comfy_clip.py --dump-blocks`, `tests/test_comfy_parity.py`): the residual stream after
each block, video rows only, cosine against ComfyUI.

| block | int8 rows, int8 QK^T attention (default) | int8 rows, f16 attention | bf16 rows, f16 attention |
| ---: | ---: | ---: | ---: |
| 0 to 10 | 1.0000 | 1.0000 | 1.0000 |
| 20 | 0.9999 | 0.9999 | 0.9999 |
| 30 | 0.9988 | 0.9990 | 0.9995 |
| 40 | 0.9929 | 0.9935 | 0.9968 |
| 49 | 0.9991 | 0.9992 | 0.9996 |

The 20-evaluation trajectory matches ComfyUI's to a relative error of 0.01 after five
evaluations; the final latents end at a cosine of about 0.90, the same figure ComfyUI's own bf16
and int8 runs reach against each other, because the last sigmas amplify any rounding difference.
Bit-exactness is not possible: ComfyUI dequantises the int8 rows to bf16 and runs bf16 matmuls;
this pipeline runs true int8 WMMA products with int32 accumulation and per-row scales, and int8
QK^T attention on rotated, per-token quantised Q and K.

Every encoder is checked against ComfyUI's too: audio 1.0000000, vision 0.99996, video VAE
0.9994; the tokenizer against transformers; the text encoder's 50 layers at cosine 1.0000 after
50 layers against transformers bf16.

## How it works

H3 is a 33B dense single-stream transformer over one packed sequence of text, video and audio
rows: 50 blocks of hidden 5376, 56 attention heads of 128 with no key-value sharing, SwiGLU
14336, AdaLN modulation per (timestep, modality) row from tables precomputed once per schedule.
Qwen3-VL-32B is the text and image encoder; a causal 24-channel video VAE and a 40 Hz audio VAE
are the tokenizers.

**In Loom** (84 kernels in `kernels/`, most written by generators in `tools/gen_*.py`):
the DiT blocks (int8, f16 and int4 GEMM families with fused SwiGLU and class-gated residual
epilogues; the AdaLN prepare kernels with the ConvRot Hadamard and row quantisation; RoPE with
q/k RMSNorm; attention), the text encoder's 50 layers, the token refiner, embedders and final
layer, the video VAE decoder's 36 blocks and heads, the three reference encoders, and the audio
vocoder. **On the host in C++:** layout, the AdaLN curve tables, the sampler, noise, patching,
chunk blending and the pixel mapping, together well under a second per clip.

The design decisions that carry the numbers, each with its measurement in `docs/notes.md`:

- **An f32 residual stream and f32 LDS in the prepare kernels.** H3's residual passes f16's
  range by block 23 and reaches 3e6 by block 36.
- **Int8 QK^T attention** on the part's `iu8` WMMA (SageAttention's idea with this part's
  arithmetic): Q and K per token and head, rotated by a Hadamard in the prepare kernel, f16 PV,
  f32 online softmax. The shipped kernel is head-major with a 64-key LDS tile shared by eight
  query tiles ([docs/attention-int8-45.md](docs/attention-int8-45.md)).
- **256x128 GEMM tiles of 64x64 wave tiles** with unconditional staging loads and a padded
  operand pitch whenever a row's byte pitch is a multiple of 1024 (rows at such a pitch alias in
  the cache: the out and down projections gained 12% and 24%). Row-group selection per token
  count ([docs/gemm-int8-tuning.md](docs/gemm-int8-tuning.md)).
- **ComfyUI's sampler and layout**, reproduced exactly: `res_multistep` on the `simple` schedule,
  the audio carried as (sigma_v / sigma_a) x_a, keyframes and references packed between the text
  and target streams.
- **Why ComfyUI is slow at video sizes**, for the record: its H3 model hands torch's flash
  attention head-strided views, on which aotriton runs at 5.8 TFLOP/s instead of the 30.8 it
  reaches on contiguous tensors.

The levers that lost are in `docs/notes.md` with their numbers too: int4 QK^T (ghosts conditioned
clips), the step cache (0.94 latent cosine at the first threshold that skips anything), tile
skipping, deeper LDS prefetch, 72-byte stage rows, wider raster groups, and the direct-load
attention form.

## Repository

| | |
| --- | --- |
| `host/` | the C++ host: `h3pipe.cpp` (the pipeline behind `h3pipe.h`), `h3_cli.cpp` (`h3`), `h3tok.cpp` (tokenizer), the text encoder, VAE and runtime bindings |
| `kernels/` | the Loom kernels; `experiments/` the measured losers, kept with their numbers |
| `tools/` | kernel generators (`gen_*.py`), weight exports (`export_*.py`, `gptq_export.py`), the Python driver, benches, the ComfyUI harness |
| `tests/` | kernel tests against float64 references, host tests, the ComfyUI parity gate |
| `reference/` | a NumPy/torch reference of the block stack and the VAE decoder |
| `docs/` | `abi.md` (the C ABI contract), `notes.md` (every measured lever, won or lost), the attention and GEMM reports, the reference-conditioning plan |
| `examples/` | the minimal client in C, Rust and Go |
| `scripts/` | `env.sh`, `build_host.sh`, `test.sh`, `download.sh` |

## Tests

```sh
bash scripts/test.sh --quick    # format, generators, host build, CPU host tests, kernel tests, tokenizer
bash scripts/test.sh            # plus the decoder, encoder and block comparisons against the references
python3 tests/test_comfy_parity.py   # the ComfyUI parity gate (needs the dumps from tools/comfy_clip.py)
```

`bash scripts/test_host.sh` runs the CPU-only host regressions without weights or a GPU.

## Limitations

- One GPU family. The kernels target `gfx1151`'s wave32 WMMA; other RDNA3/3.5 parts would need
  the tile budgets revisited and nothing has been run on them.
- Short clips are not faster than ComfyUI: at 22 frames the sequence is short enough that launch
  overhead and the short-sequence attention path dominate.
- No classifier-free guidance, matching the stock workflows (cfg 1).
- The first run of a new shape compiles its kernels (tens of seconds); they are cached after that.
- Text-to-video, first-frame (fl2va) and reference-conditioned (ref2va) generation are supported;
  video references are exposed by the C ABI but not by the `h3` command yet.

## License and acknowledgements

Apache-2.0 (see `LICENSE`). The model weights are MiniMax's, under their community license.

Built on [Loom](https://github.com/ROCm/hrx-system) from AMD, whose kernel language and compiler
this work depends on (a fork with a fragment-repack strategy and a scheduling fix is referenced in
`docs/notes.md`). MiniMax's H3 release and paper; ComfyUI's day-0 implementation as the reference
behaviour; SageAttention (arXiv 2410.02367) for the int8 attention idea; CK-tile's flash forward
as the measured ceiling. A sibling of
[krea2-loom](https://github.com/zacharydenton/krea2-loom), whose kernels, runtime and test
discipline this repository started from.
