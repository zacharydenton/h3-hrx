# RefMods

RefMods store encoded image and audio references for reuse during generation.
Creation uses the native H3 VAEs. Loading needs no source media or VAE encoding.
The output decoders still need their respective VAEs when producing video/audio.

Only current embedded-metadata standalone v4 and bundle v5 safetensors files are
supported. Legacy sidecars, older format versions, and LoRA hybrid containers
are unsupported. Compatibility follows ComfyUI-MiniMaxH3Mod commit
[`f9462081`](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/f9462081e28794389b5a6c5067eb327412ad8ee7).

## Create

```sh
h3 refmod create ./character-images --out character.safetensors
h3 refmod create ./character-images --audio voice.wav \
  --name character --description 'A character with a low, soft voice' \
  --out character-with-voice.safetensors
h3 refmod create --audio voice.wav --out voice.safetensors
h3 refmod inspect character-with-voice.safetensors
```

Directories are scanned nonrecursively in lexical order; explicit image files
retain argument order. Images use the existing CLI's supported image formats.
The first image determines the canvas. Its short edge is capped at 1024 pixels,
with dimensions rounded to multiples of 32. A single image is resized without
cropping; a stack is center-cropped to the shared canvas aspect ratio.
Native bilinear resampling differs from the upstream
extractor's Lanczos resampling, so independently encoded files need not be
byte-identical. Preprocessing settings are recorded in metadata.

Each image is encoded separately and stacked along the latent time axis. The
visual budget defaults to 8192 tokens. When necessary, near-duplicate latent
frames are removed, then remaining frames are uniformly sampled. Spatial grids
are preserved. The report includes retained image indices (zero-based into the
printed source list); omitted indices identify dropped images. If one image
alone exceeds the budget, reduce `--resolution` or increase `--max-tokens`.

Audio is resampled to 32 kHz stereo and limited to the first 30 seconds by default.
It is encoded in 10-second chunks, with a default 5120-token audio budget.
Use `--audio-max-seconds`, `--audio-max-tokens`, and `--truncate-audio` to control
this. Exceeding the audio budget fails unless truncation is explicitly enabled.
The optional audio member's concept type is `voice`; the visual concept type
defaults to `identity` and can be set with `--concept-type`.

Images alone produce a standalone file, as does audio alone. Images plus audio
produce a single bundle containing a visual member followed by an audio member.
Bundling does not assign a voice to a character or encode synchronization.

`--dry-run` decodes and validates media and reports estimated costs, without
opening models, initializing a GPU, or writing output. It cannot predict latent
deduplication. `--force` allows atomic replacement of an existing output.
`--video-vae`, `--audio-vae`, `--offline`, and `--root` apply after `create`.
Creation resolves only the requested encoders and retains their weights for reuse.

## Generate

```sh
h3 --refmod character-with-voice.safetensors --out clip.mp4 < prompt.txt
h3 --refmod character-with-voice.safetensors \
  --refmod-visual-strength 1=0.8 --refmod-audio-strength 1=0 \
  --refmod-copies 1=2 --refmod-max-total-tokens 16384 \
  --out clip.mp4 < prompt.txt
```

Repeat `--refmod FILE` for multiple concepts. Override indices are one-based
positions in that list. Visual/audio strengths default to 1; copies default to 1.
Strength must be between 0 and 1. Zero removes that modality; intermediate values
blend toward a blurred latent using upstream constant-strength semantics. Copies
repeat reference blocks and increase attention cost, while sharing host latent
storage. The optional total token limit counts selected members and copies.

Refmods are appended after raw image/audio references in file/member order.
Active refmods select ref2va weights by default. `--dit` and `--base-weights`
retain their usual meaning; Turbo cannot use active refmods. Pre-encoded visual
members have no original pixels to send through the vision tower, so no picture
placeholders or prompt text are inserted. Describe the desired subjects and
sounds explicitly in the prompt; member descriptions are printed as hints.

Saved upstream configuration is preserved and reported, but is not automatically
applied. Animated curves, training/refinement, and extraction from source videos
are outside this implementation. Existing encoded visual members, including
pooled or video-origin members, can still be loaded.

## Rust

Use `refmod::{RefMod, RefModMember, ApplyOptions, CreateOptions, ImageInput,
AudioInput}`. `Session::create_refmod` accepts decoded RGB8 images and optional
planar stereo samples; use a retaining session to reuse encoder weights.
Filesystem decoding is a CLI concern.

With `Config::dit = None`, `Session::denoise` selects Ref2VA whenever its reference
list is nonempty. Without references it selects FL2VA, including keyframe-only
requests. A reused session switches checkpoints as needed, including when models
are held in the budgeted residency cache. An explicit `Config::dit` path overrides
this selection. Standalone `Session::text_in` uses FL2VA by default.

```rust,no_run
use h3_hrx::refmod::{ApplyOptions, RefMod};
# fn main() -> h3_hrx::Result<()> {
let file = RefMod::load("character-with-voice.safetensors")?;
let prepared = file.prepare(ApplyOptions::default())?;
let references = prepared.references();
// Pass &references to Session::denoise; keep prepared alive for the call.
# Ok(())
# }
```

Members own normalized latents. Visual storage is `[24,T,H,W]`; native audio is
`[2,32,T]`. File import/export transposes audio to/from upstream `[1,32,2,T]`.
Import supports F16, BF16, and F32. Export uses F16 visuals and F32 audio; saving
higher-precision visual imports therefore quantizes them. Metadata and unknown
metadata fields are preserved; arbitrary extra tensors are not re-exported.

## Checks

The CPU suite checks pinned upstream files and strength outputs, layout conversion,
format rejection, temporal selection, CLI parsing, and atomic writes. Hardware
checks additionally exercise native creation and generation equivalence:

```sh
scripts/test.sh --cpu
cargo test --release --test refmod -- --ignored --test-threads=1
```
