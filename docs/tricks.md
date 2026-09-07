# What else H3 does, and what to take from this repository

Measured 2026-09-07 on the Radeon 8060S with `h3`, int8 path, unless stated. "Per evaluation"
is one pass of the 50 blocks; a run pays about 40 s once per process for the 18 GB block
upload and the text encoder, which a resident session (the C ABI keeps one) removes.

## Voice cloning and text-to-speech

H3 is an audio-video model, but the video stream can be reduced to a single 2x2 latent patch:
a 32x32 canvas is 37 video rows against 207 audio rows for 5 seconds. `--audio-only` skips
the video decoder and writes the WAV.

```sh
echo 'The voice of <Audio 1> says: "Hello from Strix Halo."' | h3 voice.wav --width 32 --height 32 --frames 124 --audio-only --out hello
```

| | rows | per evaluation | 30 evaluations | 20 evaluations |
| --- | ---: | ---: | ---: | ---: |
| 5 s of voice from a 5 s reference, 32x32 | 413 | 1.1 s | 33 s | 22 s |
| 15 s of voice from a 5 s reference, 32x32 (`--frames 362`) | 742 | 2.2 s | | 44 s |
| 5 s of sound from text alone (rain, thunder), 32x32 | 206 | 0.6 s | | 12 s |
| 5 s of voice, 64x64 | 780 | 2.0 s | 60 s | |

The reference audio is any file ffmpeg reads, 2 to 15 seconds; the ref2va checkpoint exports are
selected automatically. Zero video is not possible: the model's sequence always carries a
video stream, and 32x32 is its minimum (one patch per latent frame). At that size the video
rows are 9% of the sequence and the per-evaluation time is launch-bound (about 500 kernel
launches per evaluation across the 50 blocks), so the lever for speech throughput is a
resident session, not a smaller canvas: with the 40 s of per-process setup gone, 5 seconds of
cloned speech is 12 to 33 seconds of GPU time.

## Single images: H3 as an image generator and editor

Frame counts snap to 17n + 5, so 22 frames is the smallest clip; `--still frame.png` writes one
frame (`--still-frame N` chooses it). A prompt that asks for a still photograph and a static
camera gives a photograph.

```sh
h3 -p "A still photograph, static camera: a red fox on a mossy log in a birch forest, golden hour." \
   --width 1344 --height 768 --frames 22 --steps 21 --still fox.png
h3 scene.jpg -p "<Picture 1> is the scene. The same scene and subject at night under neon signs and rain, static camera." \
   --width 864 --height 480 --frames 22 --steps 21 --still night.png
```

| | per evaluation | 20 evaluations | total |
| --- | ---: | ---: | ---: |
| text to image, 1344x768 | 13 s | 261 s | 5.3 min |
| image edit from a reference, 864x480 | 4.7 s | 94 s | 2.4 min |

Against [krea2-loom](https://github.com/zacharydenton/krea2-loom) at 22 s per image, H3 is
about 15x slower as a plain text-to-image model and is not the tool for that. What it does that
Krea 2 cannot: edit and compose from up to nine reference images and audio, with Qwen3-VL-32B
reading the prompt (the night-neon edit above kept the rider, the bike and the snow ramp of the
reference and relit everything), and the image comes with a second of matching video and
sound. The prompt-understanding class is different; the speed class is too.

`docs/media/fox_still_1344x768.jpg` and `docs/media/edit_night_neon.jpg` are the two stills.

## Other modes the model card lists that this host reaches

- **First-frame animation** (`--first-frame`): the fl2va checkpoint, a keyframe pinned at frame 0.
- **Reference-driven video** (`h3 ref1.jpg ref2.jpg voice.wav < prompt`): up to nine images and
  three audio clips through the ref2va checkpoint; a talking subject with a cloned voice is one
  image plus one audio reference.
- **Video references** (video-to-audio-video: foley for a silent clip, restyling a clip) are
  exposed by the C ABI (`h3pipe_ref.kind` 2, `h3pipe_encode_video` on 17-frame chunks) but not
  by the `h3` command yet.
- **Last-frame and first-and-last-frame** generation need the keyframe at `frames - 1`, which
  the ABI accepts (`h3pipe_keyframe.frame_index`); untested here.
- **2K regeneration** (the model regenerating its own low-resolution output in context) is not
  open-sourced by MiniMax.

## Pieces worth taking elsewhere

Everything below is self-contained and has a test.

| piece | where | why it matters beyond H3 |
| --- | --- | --- |
| int8 QK^T flash attention, head-major, 64-key tiles | `kernels/attention_i8qkhm_mha8_lds_f16_wmma.loom`, `tools/gen_attention_i8_head_major.py`, `docs/attention-int8-45.md` | 37 TFLOP/s-equivalent on this GPU, 20% over aotriton's flash kernel and within 5% of CK-tile; any transformer inference on Strix Halo |
| int8 and int4 WMMA GEMM family with fused epilogues | `tools/gen_gemm.py`, `kernels/gemm_*` | 41 to 45 TOPS at LLM shapes, ahead of comfy_kitchen's HIP kernel at every shape measured; the padded-pitch rule (`gemm_pitch`) applies to any GEMM on this APU |
| the aliasing finding itself | `docs/notes.md`, "rows whose pitch is a multiple of 1024" | a row pitch that is a multiple of 1024 bytes loses 12 to 32% in any strided kernel here; pad it |
| Qwen3-VL-32B in Loom, int8 | `host/h3te.cpp`, `kernels/`, `tools/export_te.py` | a 32B vision-language encoder as a C library: embeddings, prefill, the vision tower with DeepStack |
| the Qwen2 byte-level BPE tokenizer in C | `host/h3tok.cpp` (one file, tested against transformers) | any Qwen-family model without Python |
| ConvRot row quantiser | `tools/gen_prepare.py`, `kernels/prepare_*` | Hadamard-rotated per-row int8 and int4 activation quantisation with f32 LDS, the W8A8 / W4A4 front end |
| the video VAE (causal 3-D conv encoder and decoder) and the audio VAE in Loom | `host/h3vae.cpp`, `tools/export_vae*.py`, `tools/export_audio_encoder.py` | standalone codecs: the decoder tiles as ComfyUI does, 52 dB frame PSNR at int8 |
| the C ABI pattern | `host/h3pipe.h`, `docs/abi.md`, `examples/` | opaque session, caller-owned buffers, error strings, progress callback; bindings in C, Rust and Go that produce identical output |
| Loom compiler work | `~/code/hrx-system/loom`, branch `loom-swizzler` | `v_permlanex16` cross-lane recipes, an f32-to-f16 rhs fragment repack in registers, and `sink-single-use-reads` leaving fragment loads in source order |
| the measurement protocol | `docs/notes.md` | a theory earns a kernel only after a cheap falsifier; every lever with its number, including the ones that lost; never time with a compile running (the APU's shared power budget costs 40%) |
