# The C ABI: `libh3pipe.so`

[`host/h3pipe.h`](../host/h3pipe.h) and [`host/h3tok.h`](../host/h3tok.h) declare the pipeline and tokenizer.
Every language with a C foreign-function interface can drive it; `examples/` has working
programs in C, Rust (no bindgen) and Go (cgo), and `h3pipe_loom.py` is the ctypes binding the
Python tools use. `host/h3_cli.cpp` is the complete client: references, keyframes, decoding and
muxing.

```
h3pipe_abi_version                                   -> 7
h3tok_create / h3tok_encode / h3tok_vocab_size / h3tok_destroy      text -> token ids
h3pipe_create / h3pipe_destroy                       a session: weights resident, kernels cached
h3pipe_shape_for                                     parameters -> latent and frame counts (no session needed)
h3pipe_denoise / h3pipe_denoise_refs                 ids (+ keyframes, references) -> model-space latents
h3pipe_decode_video / h3pipe_decode_audio            latents -> RGB8 frames / 32 kHz stereo samples
h3pipe_encode_video / h3pipe_encode_audio / h3pipe_vision_embed     references in
h3pipe_text_in                                       the refined text rows (inspection)
```

## Contract

**Versioning.** `h3pipe_abi_version()` returns `H3PIPE_ABI_VERSION` (7). Check it before using
the structs; the structs' layouts are frozen per version and every field is a fixed-width C
type or a pointer, so no language needs a bindings generator.

**Return codes.** Every call returns `H3PIPE_OK` (0) on success, `H3PIPE_INVALID_ARGUMENT` (64)
for a bad parameter or a buffer that is too small, `H3PIPE_CANCELLED` (2) when the progress
callback cancelled, `H3PIPE_ERROR` (1) otherwise. On failure the `error` buffer holds a NUL-terminated
message (pass its capacity; 4096 bytes is plenty). `h3tok_encode` returns the id count or -1: the count is
what the text needs even when it exceeds the buffer's capacity, and only the first `capacity` ids are
written, so check the count against the capacity (or call with capacity 0 to size the buffer).
`h3pipe_shape_for` returns `H3PIPE_INVALID_ARGUMENT` for a canvas that is not multiples of 32 in 32..8192
or a frame count outside 1..1048576; check it before allocating from the shape.

**Ownership.** The caller allocates every buffer and keeps it alive for the duration of the call;
the library never keeps a pointer past the call except the session's own state. Latent, frame and
sample buffers must be at least the sizes below; pass their element counts, and the library
refuses short ones rather than writing past them. A larger buffer (a pooled one, say) is accepted
and exactly the required count is read or written; the rest is untouched.

**Threading.** A session serialises its calls with an internal mutex: concurrent calls from several
threads are safe and run one at a time. The progress callback runs on the calling thread between
denoising steps; returning nonzero from it cancels the run, which returns `H3PIPE_CANCELLED` with
the output buffers unspecified. Reuse a session across requests to retain loaded weights and compiled kernels.

**Kernels.** The first call at a new shape spawns `loom-compile` (path in the config) for the kernels
that shape needs and caches the binaries in `cache_dir`; later runs at the same shape load from the
cache. Expect tens of seconds the first time a size is used.

**Weights.** The checkpoints are memory-mapped and each tensor is uploaded to the device the first
time a stage asks for it, so creating a session costs milliseconds and only what a call touches is
resident.

## Sizes and layouts

Fill an `h3pipe_params` (height and width multiples of 32; frames snaps up to `17n + 5`; steps is
the number of sigma grid points, evaluations + 1; `seed`; `sampler` 1 for ComfyUI's
`res_multistep`, 0 for Euler; shifts and `cache_threshold` 0 for the defaults) and call
`h3pipe_shape_for` to get the counts:

| buffer | element type | layout | element count |
| --- | --- | --- | --- |
| video latents | f32 | `[24][latent_t][lat_h][lat_w]` | `24 * latent_t * lat_h * lat_w` |
| audio latents | f32 | `[2][32][audio_t]` | `64 * audio_t` |
| decoded frames | u8 | `[frames][height][width][3]`, RGB | `frames * height * width * 3` |
| decoded samples | f32 | `[2][audio_t * 800]`, planar stereo at 32 kHz | `1600 * audio_t` |
| optional caller noise | f32 | same as the latents | same |

`lat_h = height / 16`, `lat_w = width / 16`; 24 frames per second of video; 800 samples per audio
latent (40 Hz).

## Prompts and references

Ids come from `h3tok_encode` on the prompt text (no special tokens). For references, build the
presentation in this order, then the prompt: keyframes first, then reference images, each as
`"<Picture i>: "` tokenised, id 151652 (`<|vision_start|>`), `(height/32) * (width/32)` placeholder
ids of value -1, id 151653 (`<|vision_end|>`); then `"<Audio j>: "` per reference audio. The host
replaces the placeholder rows with the vision tower's embeds of the pixels you pass in the
`h3pipe_ref` / `h3pipe_keyframe` structs.

- **Reference image** (`h3pipe_ref.kind` 0): resize so its pixel count is at most the canvas' and
  both sides are multiples of 32; `h3pipe_encode_video` on the one frame gives `video_latent`
  (`[24][1][h/16][w/16]`, `latent_t` 1); pass the same pixels (f32 `[h][w][3]` in [0, 1]) as
  `pixels`. The reference-conditioned checkpoint (`..._ref2va_...safetensors`) is the one to load.
- **Reference audio** (kind 1): `h3pipe_encode_audio` on planar stereo f32 samples at 32 kHz gives
  `audio_latent` and `audio_t`.
- **Keyframe** (`h3pipe_keyframe`, first-frame generation): the frame resized to the canvas,
  encoded the same way, `frame_index` 0.

`h3pipe_denoise_refs` takes the keyframe and reference arrays; `h3pipe_denoise` is the plain
text-to-video call.

## Configuration files

`h3pipe_config` names ComfyUI's four checkpoints, listed in the [setup guide](setup.md#checkpoints):
`dit_file` (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, or the `ref2va` one for
reference-conditioned clips), `te_file`, `video_vae_file` and `audio_vae_file`. There is no export
step: the DiT blocks' and text encoder's int8 ConvRot rows run on the
int8 GEMMs with their stored scales, the bf16 refiner, condition projection and vision tower on
bf16 kernels, the video VAE's f16 and the audio VAE's f32 tensors in their own types. A file may
be NULL; the calls that need it then fail with a message naming it, and each file is opened on
first use. `kernel_sources` is the repository's `kernels/`, `cache_dir` any writable directory,
`loom_compile` the compiler binary. `attn_qk_bits` chooses the DiT attention's QK^T operands: 8
(the default and the parity path), 16 for f16, or 4 for int4.

`h3tok_create(NULL, ...)` uses the tokenizer compiled into the library (`H3_TOKENIZER=<file>`
overrides it); passing a path reads that file instead.

## Clients

[Complete examples](../examples/README.md) cover C, Rust, Go, and Python,
including error handling and output. The C++ CLI in
[`host/h3_cli.cpp`](../host/h3_cli.cpp) also demonstrates reference preparation
and muxing. Use [structured prompts](prompting.md) for generation.

Link with `-L build -lh3pipe` and put the library directory on the runtime
search path (`-Wl,-rpath` or `LD_LIBRARY_PATH`). See [setup](setup.md) for the
ROCm and Loom toolchain configuration.
