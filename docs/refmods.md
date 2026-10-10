# Reuse subjects and voices with RefMods

Encode image and audio references once, then reuse them across renders. RefMods
store those latents in safetensors files; loading them skips reference VAE
encoding. Output decoding still uses the video/audio VAEs.

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

RefMods follow raw references in file/member order and select Ref2VA by default.
`--dit` and `--base-weights` override model selection; Turbo cannot use active RefMods.
In the default latent-only mode, visual members have no original pixels to send
through the vision tower, so no picture
placeholders or prompt text are inserted. Describe the desired subjects and
sounds explicitly in the prompt; member descriptions are printed as hints.

Saved upstream configuration is preserved and reported, but is not automatically
applied. Animated curves, training/refinement, and extraction from source videos
are outside this implementation. Existing encoded visual members, including
pooled or video-origin members, can still be loaded.

### First frame + RefMod with FL2VA

Use `--first-frame` for the opening composition, `--refmod` for separate reference
conditioning, and `--base-weights` to keep the FL2VA checkpoint:

```sh
h3 --base-weights --first-frame opening.png \
  --refmod character.safetensors --out clip.mp4 < prompt.txt

# Endpoint prompting with an original image for a single-image RefMod:
h3 --base-weights --first-frame opening.png \
  --refmod character.safetensors --refmod-source '1:1=portrait.png' \
  --generate-prompt --prompt-images \
  -p 'Start from the opening image; the referenced character turns and waves' \
  --out clip.mp4
```

The keyframe anchors frame zero; RefMods supply separate reference blocks.
Upstream presentation sends both keyframe and reference visuals to the encoder.
Endpoint rewriting uses the six-section reference format and distinguishes keyframe composition
from the RefMod's requested identity, appearance or sound. Supply the manifest's
actual labels when writing a prompt manually; stacks use `<Video N>` labels.
For a bundle containing audio, endpoint prompting also needs `--prompt-audio`.

The combined path has CPU regression coverage; reference fidelity with stock
FL2VA still needs visual qualification. RefMod tokens add attention work with
either checkpoint.

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
retain argument order. Images use the formats supported by the CLI.
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

## Upstream presentation

`--refmod-presentation upstream` reconstructs active references through the H3
VAEs and presents numbered media to H3's text/vision encoder. `--generate-prompt`
and `h3 prompt` select this path automatically; combining them with
`--refmod-presentation latent-only` is an error. Ordinary generation defaults to latent-only conditioning.

```sh
h3 --refmod character.safetensors --refmod-presentation upstream \
  --out greeting.mp4 < greeting.txt
```

The implementation follows ComfyUI-MiniMaxH3Mod `f9462081` (`prompt.py`,
`nodes.py`) and ComfyUI `e9027f2b` (`comfy/text_encoders/minimax.py`). Strength
transforms and disabled-member filtering happen before reconstruction. Raw
references precede RefMod members and copies; each media kind has its own counter.
Each copy receives a consecutive label, while decoded host pixels are shared.
A single image member is a `<Picture N>`, an image stack/video member is a
`<Video N>`, and an audio member is an independent `<Audio N>`.

Video presentation samples at 2 fps using `--reference-fps` (default 24) as the
reconstructed playback rate, pairs frames into temporal patches, and repeats an
odd final frame. Stack/compressed references do not preserve source chronology;
the prompt processor treats these timestamps as synthetic. A bundle does not
implicitly synchronize audio or bind a voice to a subject.

The prepared-presentation path uses upstream Qwen3-VL image/video resizing and
0.5 normalization. The direct-generation image path uses different preprocessing for compatibility.

Visual reconstruction retains float pixels for H3 conditioning. Audio is decoded
for the optional LLM's analysis; H3's text encoder receives its label, while its
DiT receives the already-encoded audio. Reference latents are attached once.
This path requires the relevant VAEs, consumes additional vision tokens and can
fail the text budget even when latent-only loading succeeds. The prompt-only
command also needs these VAEs and a GPU when it reconstructs RefMods.

### Use original files without VAE reconstruction

Supply `--refmod-source SLOT:MEMBER=PATH` to use original media for prompt analysis
and H3 text/vision presentation. Both indices are one-based: `SLOT` is the position
in the `--refmod` list, and `MEMBER` is the original member number printed by
`h3 refmod inspect`. Member numbers stay fixed when another modality is disabled;
copies inherit the same source and share its decoded buffers.

```sh
h3 prompt --prompt-images --refmod character.safetensors \
  --refmod-source '1:1=original portrait.png' \
  -p 'The referenced character greets the viewer' > greeting.txt

# A visual-stack member followed by an audio member in the same bundle:
h3 --generate-prompt --prompt-images --prompt-audio \
  --refmod character-with-voice.safetensors \
  --refmod-source '1:1=front.png' --refmod-source '1:1=profile.png' \
  --refmod-source '1:2=voice.wav' \
  -p 'The referenced character greets the viewer' --out greeting.mp4
```

An image member accepts one image, an audio member accepts one audio file, and a
video/stack member accepts either one video file or repeated image paths in the
desired order. Video files are sampled at 2 fps; all supplied stack images are
shown at synthetic half-second intervals. Later stack images are center-cropped
to the first image's aspect ratio. Visuals fit the output canvas pixel budget on
the 32-pixel grid. Video soundtracks are not implicitly included; use a separate
audio member and source mapping. Paths with spaces should be quoted.

The flag enables upstream presentation automatically and conflicts with explicit
`--refmod-presentation latent-only`. It works with endpoint prompting
and with an existing handwritten prompt. Sources replace presentation evidence rather than adding
positional references: the RefMod latents, strengths and conditioning copies
remain unchanged, and originals are never VAE-encoded.

Unmapped active members still use VAE reconstruction. When every active member
has an original source, endpoint `h3 prompt` needs no H3 checkpoints or GPU.
Video generation still needs its model and output decoder. Source decoding is bounded by the shared
`--refmod-media-budget-mib` limit (default 1024 MiB), excluding codec subprocess
memory, models and endpoint payloads. Oversized sources fail rather than truncate.

Supply files corresponding to the intended members: correspondence cannot be
verified automatically, especially for pooled or optimized RefMods. Provenance
records supplied paths as `original_file`, separately from reconstructed latent
evidence. Originals do not reflect latent strength adjustments.

## File compatibility

Only current embedded-metadata standalone v4 and bundle v5 safetensors files are
supported. Legacy sidecars, older format versions, and LoRA hybrid containers
are unsupported. Compatibility follows ComfyUI-MiniMaxH3Mod commit
[`f9462081`](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/f9462081e28794389b5a6c5067eb327412ad8ee7).

## Rust

Use `refmod::{RefMod, RefModMember, ApplyOptions, CreateOptions, ImageInput,
AudioInput}`. `Session::create_refmod` accepts decoded RGB8 images and optional
planar stereo samples; use a retaining session to reuse encoder weights.
Filesystem decoding is a CLI concern.

`Session::refmod_entries(&prepared_mods, RefModPresentationOptions::default())`
reconstructs the active members as ordered `MediaEntry` values for upstream
presentation, including copy labels and synthetic timing. With the optional
`prompt-generation` feature, `PromptGenerator::generate_refmods` performs this
preparation and endpoint rewriting together, returning the prompt and matching
H3 presentation. See the [native Rust example](prompt-generation.md#native-rust-refmod-prompting).

For originals, pass decoded `RefModSource` values to
`Session::refmod_entries_with_sources`; only unmapped members are reconstructed.
`refmod::entries_from_sources` assembles fully supplied references without a
session. Both return the same ordered `MediaEntry` list for prompting and H3
presentation. `PreparedRefMod::indexed_members()` exposes stable original member
numbers, including copies.
For endpoint prompting in one call, use
`PromptGenerator::generate_refmods_with_sources(None, request, &sources)` when
all active members have originals, or pass `Some(&mut session)` for partial
coverage. See the [Rust original-media example](prompt-generation.md#native-rust-refmod-prompting).

With `Config::dit = None` and the default `base_weights: false`, `Session::denoise`
selects Ref2VA whenever its reference list is nonempty. Set `base_weights: true`
to use FL2VA with references, including a RefMod plus first frame. Without
references it selects FL2VA, including keyframe-only
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

For FL2VA plus a first frame, construct the session with
`Config { base_weights: true, ..Default::default() }`, then pass the first-frame
latents and RefMod references together:

```rust,no_run
use h3_hrx::{Clip, DenoiseParams, Keyframe, Latents, Noise, Presented, Session};
use h3_hrx::refmod::PreparedRefMod;

fn first_frame_with_refmod(
    session: &mut Session, // Config.base_weights = true
    ids: &[i32],
    params: &DenoiseParams,
    first_rgb: &[f32], // Interleaved RGB [0,1], resized to the output canvas.
    prepared: &PreparedRefMod,
) -> h3_hrx::Result<Latents> {
    let clip = Clip {
        pixels: first_rgb, frames: 1,
        height: params.height as usize, width: params.width as usize,
    };
    let (latents, _) = session.encode_video(clip)?;
    let first = Keyframe {
        frame_index: 0, latents: &latents, audio: None,
        presented: Some(Presented {
            pixels: first_rgb, height: clip.height, width: clip.width,
        }),
    };
    session.denoise(ids, params, Noise::default(), &prepared.references(), &[first], None)
}
```

For endpoint prompting, include that same RGB frame in
`RefModPromptRequest.entries` as `MediaEntry { role: "first_frame".into(),
media: Media::Picture(frame), metadata: serde_json::json!({"frame_index": 0}) }`.
Pass the returned presentation to `Session::denoise_presented` with the same
RefMod references and `Keyframe { frame_index: 0, ... }`. The explicit
presentation already contains the first-frame vision block, so its keyframe
`presented` field may be `None`; the keyframe latents are still required.

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
cargo test
cargo test --release --test refmod -- --test-threads=1
```
