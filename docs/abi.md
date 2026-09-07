# The C ABI: `libh3pipe.so`

`host/h3pipe.h` and `host/h3tok.h` declare the whole pipeline behind sixteen plain C functions.
Every language with a C foreign-function interface can drive it; `examples/` has working
programs in C, Rust (no bindgen) and Go (cgo), and `h3pipe_loom.py` is the ctypes binding the
Python tools use. `host/h3_cli.cpp` is the complete client: references, keyframes, decoding and
muxing.

```
h3pipe_abi_version                                   -> 6
h3tok_create / h3tok_encode / h3tok_vocab_size / h3tok_destroy      text -> token ids
h3pipe_create / h3pipe_destroy                       a session: weights resident, kernels cached
h3pipe_shape_for                                     parameters -> latent and frame counts (no session needed)
h3pipe_denoise / h3pipe_denoise_refs                 ids (+ keyframes, references) -> model-space latents
h3pipe_decode_video / h3pipe_decode_audio            latents -> RGB8 frames / 32 kHz stereo samples
h3pipe_encode_video / h3pipe_encode_audio / h3pipe_vision_embed     references in
h3pipe_text_in                                       the refined text rows (inspection)
```

## Contract

**Versioning.** `h3pipe_abi_version()` returns `H3PIPE_ABI_VERSION` (6). Check it before using
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
the output buffers unspecified. Creating a session loads about 48 GB of weights (int8 path); create
one per process and reuse it.

**Kernels.** The first call at a new shape spawns `loom-compile` (path in the config) for the kernels
that shape needs and caches the binaries in `cache_dir`; later runs at the same shape load from the
cache. Expect tens of seconds the first time a size is used.

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
  `pixels`. The reference-conditioned checkpoint (`*_ref2va` exports) is the one to load.
- **Reference audio** (kind 1): `h3pipe_encode_audio` on planar stereo f32 samples at 32 kHz gives
  `audio_latent` and `audio_t`.
- **Keyframe** (`h3pipe_keyframe`, first-frame generation): the frame resized to the canvas,
  encoded the same way, `frame_index` 0.

`h3pipe_denoise_refs` takes the keyframe and reference arrays; `h3pipe_denoise` is the plain
text-to-video call.

## Configuration directories

`h3pipe_config` names the exported weight directories (`docs/../README.md`, "Weights"): `blocks_dir`
(`build/weights_i8`, or `weights_f16` with `attn_qk_bits` 16, or `weights_gptq` with 4), `glue_dir`,
`te_dir`, `vae_dir` (`weights_vae_i8`, `vae_bits` 8), and, for references, `venc_dir`, `vision_dir`,
`aenc_dir` (NULL disables the corresponding encoder). `kernel_sources` is the repository's `kernels/`,
`cache_dir` any writable directory, `loom_compile` the compiler binary. `attn_qk_bits` 8 is the
parity path.

## Minimal client, in C

```c
#include "h3pipe.h"
#include "h3tok.h"

char err[4096];
h3tok *tok = h3tok_create("~/h3-models/tokenizer/tokenizer.json", err, sizeof err);
int32_t ids[4096]; int n = h3tok_encode(tok, "A red fox on a mossy log ...", ids, 4096);   /* n > 4096: the buffer was too small */

h3pipe_config cfg = { "build/weights_glue", "build/weights_i8", "build/weights_te", "build/weights_vae_i8",
                      "kernels", "build/kernel_cache", "loom-compile", 8, NULL, NULL, NULL, 8 };
h3pipe_session *s; h3pipe_create(&cfg, &s, err, sizeof err);

h3pipe_params p = { 480, 864, 124, 31, 0, 0, 0, 1, 0 };
h3pipe_shape sh; if (h3pipe_shape_for(&p, &sh)) return 64;
float *video = malloc(sizeof(float) * 24 * sh.latent_t * sh.lat_h * sh.lat_w), *audio = malloc(sizeof(float) * 64 * sh.audio_t);
h3pipe_denoise(s, ids, n, &p, NULL, NULL, video, 24 * sh.latent_t * sh.lat_h * sh.lat_w, audio, 64 * sh.audio_t, NULL, NULL, err, sizeof err);

uint8_t *frames = malloc((size_t)sh.frames * p.height * p.width * 3); float *samples = malloc(sizeof(float) * 1600 * sh.audio_t);
h3pipe_decode_video(s, &p, video, 24 * sh.latent_t * sh.lat_h * sh.lat_w, frames, (size_t)sh.frames * p.height * p.width * 3, err, sizeof err);
h3pipe_decode_audio(s, audio, 64 * sh.audio_t, sh.audio_t, samples, 1600 * sh.audio_t, err, sizeof err);
h3pipe_destroy(s); h3tok_destroy(tok);
```

`examples/c/minimal.c` is this with error handling and file output; `examples/rust` and
`examples/go` are the same program in those languages. Linking: `-L build -lh3pipe` with the
library's directory on the runtime path (`-Wl,-rpath` or `LD_LIBRARY_PATH`), plus the ROCm
runtime from `scripts/env.sh`.
