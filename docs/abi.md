# The C ABI: `libh3.so`

[`include/h3.h`](../include/h3.h) declares the whole surface. It is generated from
[`h3/src/capi.rs`](../h3/src/capi.rs) by cbindgen and checked in; `scripts/test.sh` fails if the
committed copy no longer matches the code that implements it.

Every language with a C foreign-function interface can drive it; `examples/c` and `examples/go` are
working clients. A **Rust** caller does
not need the C ABI at all: the `h3` crate's `Session` is the same API with slices, borrows and
`Result`, which is what `cli/` and `examples/rust` use.

```
h3_abi_version                                            -> 9
h3_last_error                                             the last failure on this thread
h3_tokenizer_create / _encode / _vocab_size / _destroy    text -> token ids
h3_create / h3_destroy                                    a session: weights resident, kernels cached
h3_shape_for                                              parameters -> latent and frame counts (no session)
h3_denoise                                                ids (+ keyframes, references) -> model-space latents
h3_decode_video / h3_decode_audio                         latents -> RGB8 frames / 32 kHz stereo samples
h3_encode_video / h3_encode_audio / h3_vision_embed       references in
h3_text_in                                                the refined text rows (inspection)
```

**Errors.** A call returns `H3_OK` or a status; `h3_last_error()` gives the reason for the last
failure on the calling thread, valid until the next one. Signatures carry no error buffer.

## Contract

**Versioning.** `h3_abi_version()` returns `H3_ABI_VERSION` (9). Check it before using
the structs; the structs' layouts are frozen per version and every field is a fixed-width C
type or a pointer, so no language needs a bindings generator.

**Return codes.** Every call returns `H3_OK` (0) on success, `H3_INVALID_ARGUMENT` (64)
for a bad parameter or a buffer that is too small, `H3_CANCELLED` (2) when the progress
callback cancelled, `H3_ERROR` (1) otherwise. On failure the `error` buffer holds a NUL-terminated
message (pass its capacity; 4096 bytes is plenty). `h3tok_encode` returns the id count or -1: the count is
what the text needs even when it exceeds the buffer's capacity, and only the first `capacity` ids are
written, so check the count against the capacity (or call with capacity 0 to size the buffer).
`h3_shape_for` returns `H3_INVALID_ARGUMENT` for a canvas that is not multiples of 32 in 32..8192
or a frame count outside 1..1048576; check it before allocating from the shape.

**Ownership.** The caller allocates every buffer and keeps it alive for the duration of the call;
the library never keeps a pointer past the call except the session's own state. Inputs are read from
the caller's memory rather than copied, so a call's input and output buffers must not overlap. Latent, frame and
sample buffers must be at least the sizes below; pass their element counts, and the library
refuses short ones rather than writing past them. A larger buffer (a pooled one, say) is accepted
and exactly the required count is read or written; the rest is untouched. A pointer must also be
aligned for the type it names and its span must not wrap the address space; a count of zero is the
one case where NULL is accepted, and yields an empty slice. What the library cannot check — that
the pointer really names the memory it claims — remains the caller's promise.

**On failure.** Arguments are validated before any output is written, so `H3_INVALID_ARGUMENT`
always leaves the output buffers as they were. A failure raised once a call is under way — a device
error, or a cancellation — can leave a decode's output partly written, because the video decoder
commits each temporal chunk as it finishes rather than staging a whole clip. Read an output buffer
only after `H3_OK`.

**Threading.** A session serialises its calls with an internal mutex: concurrent calls from several
threads are safe and run one at a time. The progress callback runs on the calling thread between
denoising steps; returning nonzero from it cancels the run, which returns `H3_CANCELLED` with
the output buffers unspecified. Reuse a session across requests to retain loaded weights and compiled kernels.

**Kernels.** The first call at a new shape compiles the kernels in process through HRX’s `libloomc`
that shape needs and caches the binaries in `cache_dir`; later runs at the same shape load from the
cache. Expect tens of seconds the first time a size is used.

**Weights.** The checkpoints are memory-mapped and each tensor is uploaded to the device the first
time a stage asks for it, so creating a session costs milliseconds and only what a call touches is
resident.

## Sizes and layouts

Fill an `h3_params` (height and width multiples of 32; frames snaps up to `17n + 5`; steps is
the number of sigma grid points, evaluations + 1; `seed`; `sampler` 1 for ComfyUI's
`res_multistep`, 0 for Euler; shifts and `cache_threshold` 0 for the defaults) and call
`h3_shape_for` to get the counts:

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
`h3_ref` / `h3_keyframe` structs.

- **Reference image** (`h3_ref.kind` 0): resize so its pixel count is at most the canvas' and
  both sides are multiples of 32; `h3_encode_video` on the one frame gives `video_latent`
  (`[24][1][h/16][w/16]`, `latent_t` 1); pass the same pixels (f32 `[h][w][3]` in [0, 1]) as
  `pixels`. The reference-conditioned checkpoint (`..._ref2va_...safetensors`) is the one to load.
- **Reference audio** (kind 1): `h3_encode_audio` on planar stereo f32 samples at 32 kHz gives
  `audio_latent` and `audio_t`.
- **Keyframe** (`h3_keyframe`, first-frame generation): the frame resized to the canvas,
  encoded the same way, `frame_index` 0.

`h3_denoise` takes the keyframe and reference arrays, which may be NULL with a count of zero for the plain
text-to-video call.

## Configuration files

`h3_config` names ComfyUI's four checkpoints, listed in the [setup guide](setup.md#checkpoints):
`dit_file` (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, or the `ref2va` one for
reference-conditioned clips), `te_file`, `video_vae_file` and `audio_vae_file`. There is no export
step: the DiT blocks' and text encoder's int8 ConvRot rows run on the
int8 GEMMs with their stored scales, the bf16 refiner, condition projection and vision tower on
bf16 kernels, the video VAE's f16 and the audio VAE's f32 tensors in their own types. A file may
be NULL; the calls that need it then fail with a message naming it, and each file is opened on
first use. The three remaining strings are all optional, and NULL or `""` selects a default:
`kernel_sources` the Loom sources built into the library (pass the repository's `h3/kernels/` to
compile from a working tree instead), `cache_dir` the shared per-user HRX cache under
`$XDG_CACHE_HOME/hrx` (pass any writable directory to keep a cache of your own), and `loom_library`
whatever `HRX_LOOM_LIBRARY` names or, failing that, the compiler in the pinned native bundle.
`attn_qk_bits` chooses the DiT attention's QK^T operands: 8 (the default and the parity path), 16
for f16, or 4 for int4.

`h3_tokenizer_create(NULL)` uses the vocabulary compiled into the library (`H3_TOKENIZER=<file>`
overrides it); passing a path reads that file instead.

## Clients

[Complete examples](../examples/README.md) cover C, Rust, Go, and Elixir,
including error handling and output. The `h3` command in
[`cli/src/main.rs`](../cli/src/main.rs) is the fullest client: reference
preparation, keyframes, decoding and muxing. Use [structured
prompts](prompting.md) for generation.

Link with `-L build -lh3` and put the library directory on the runtime
search path (`-Wl,-rpath` or `LD_LIBRARY_PATH`). See [setup](setup.md) for the
Loom toolchain configuration.
