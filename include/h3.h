// C ABI for the whole MiniMax H3 text -> video + audio pipeline with every kernel in Loom:
// prompt token ids in, latents / frames / samples out. Everything between the kernels runs on the
// host in Rust — layout, AdaLN curves, scheduler, noise, patching, blending.
//
// Kernels are compiled on first use for a shape by spawning `loom-compile` into cache_dir; the
// runtime surface is libhrx, and there is no device code in this library.
//
// A call returns H3_OK or a status; h3_last_error() gives the reason for the last failure on the
// calling thread. H3_INVALID_ARGUMENT means nothing was written to the output buffers: arguments
// are checked before any work starts. A failure raised once a call is under way, a cancellation
// included, can leave a decode's output partly written — the video decoder commits each temporal
// chunk as it finishes — so read an output buffer only after H3_OK.
//
// What every entry point requires of its caller, once:
//
//   - A session pointer is one h3_create returned and h3_destroy has not been called on, or NULL,
//     which is refused. It may be used from any thread but not from two at once.
//   - Every buffer pointer is either NULL, or points to at least the number of elements its paired
//     count says, correctly aligned and — for inputs — initialised. A NULL with a positive count is
//     refused rather than dereferenced; a *short* buffer cannot be detected and is undefined.
//   - Every string is NUL-terminated and stays valid for the duration of the call.
//   - Pointers inside h3_ref and h3_keyframe follow the same rules, with the lengths their own
//     fields imply.
//   - A call's input and output buffers do not overlap: inputs are read from the caller's memory
//     rather than copied into the library.
//   - A checkpoint file must not be modified or truncated while a session holds it: it is mapped,
//     not copied.
//
// GENERATED from h3/src/capi.rs by cbindgen. Do not edit; run scripts/build_host.sh.

#ifndef H3_LOOM_H
#define H3_LOOM_H

#include <stddef.h>
#include <stdint.h>
#define H3_ABI_VERSION 8u

// What an entry point returns. The distinction is the caller's: `INVALID_ARGUMENT` means the request
// was not one this library serves, `CANCELLED` that a progress callback stopped the run, and `ERROR`
// that something failed on the way.
typedef enum h3_status {
  H3_OK = 0,
  H3_ERROR = 1,
  H3_CANCELLED = 2,
  H3_INVALID_ARGUMENT = 64,
} h3_status;

// The session as the ABI hands it out. The lock is here rather than in `Session` because it exists
// for the ABI's sake: a C caller may use one session from any thread, while a Rust caller gets
// `&mut self` and needs no lock at all.
typedef struct h3_session h3_session;

// A loaded vocabulary.
typedef struct h3_tokenizer h3_tokenizer;

// Where the checkpoints live and how kernels are built. Every checkpoint is optional: one is opened
// only when a call needs it, and a NULL leaves the calls that would use it unavailable.
typedef struct h3_config {
  // ComfyUI's minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors, read as it is
  const char *dit_file;
  // qwen3vl_32b_minimax_h3_int8_convrot.safetensors: the encoder's layers, its embedding table,
  // and the vision tower
  const char *te_file;
  // minimax_h3_video_vae_fp16.safetensors: the video decoder and encoder
  const char *video_vae_file;
  // minimax_h3_audio_vae_fp32.safetensors: the vocoder and the audio encoder
  const char *audio_vae_file;
  // Optional directory of .loom sources; NULL or empty selects embedded sources
  const char *kernel_sources;
  // Optional cache directory; NULL or empty selects the shared per-user HRX cache
  const char *cache_dir;
  // Optional compiler override; NULL or empty selects LOOM_COMPILE or the pinned bundle
  const char *loom_compile;
  // the DiT attention's QK^T operands: 16 (f16), 8 (int8, the parity path) or 4 (int4); 0 means 8
  int attn_qk_bits;
} h3_config;

// What a run asks for.
typedef struct h3_params {
  // pixels, multiples of 32
  int height;
  int width;
  // snapped up to the next 17n + 5
  int frames;
  // sigma grid points, so steps - 1 model evaluations, as diffusers counts them
  int steps;
  uint64_t seed;
  // 0 means the model's defaults, 12 and 3
  float video_shift;
  float audio_shift;
  // 0 = Euler per stream schedule; 1 = res_multistep on the video sigma grid with the audio
  // carried as (sigma_v / sigma_a) x_a, which is what the stock workflows use
  int sampler;
  // first-block step cache: skip blocks 1..49 while the accumulated relative change of block 0's
  // output stays below this. 0 turns it off; 0.05 to 0.15 is the useful range
  float cache_threshold;
} h3_params;

// The shapes a request produces, after snapping: video latents `[24][latent_t][lat_h][lat_w]` and
// audio latents `[2][32][audio_t]`.
typedef struct h3_shape {
  int frames;
  int latent_t;
  int lat_h;
  int lat_w;
  int audio_t;
  int text_rows_max;
} h3_shape;

// A keyframe (fl2va): one latent frame pinned at a frame index, presented before any reference.
typedef struct h3_keyframe {
  // 0 for the first frame, or frames - 1 after snapping for the last
  int frame_index;
  // [24][1][lat_h][lat_w] on the generation's own latent grid
  const float *video_latent;
  // the same frame as pixels for the encoder's presentation, f32 [height][width][3]
  const float *pixels;
  int height;
  int width;
  // optional, and never denoised
  const float *audio_latent;
  int audio_t;
} h3_keyframe;

// A reference block (ref2va), in presentation order. Unused pointers are NULL and unused counts 0.
typedef struct h3_ref {
  // 0 = image (one latent frame), 1 = audio, 2 = video (with an optional soundtrack)
  int kind;
  const float *video_latent;
  int latent_t;
  int lat_h;
  int lat_w;
  const float *audio_latent;
  int audio_t;
  // images: the reference as presented to the text encoder, f32 [height][width][3] in [0, 1] with
  // both sides multiples of 32. The ids carry (height/32)*(width/32) placeholders (-1) for it
  const float *pixels;
  int height;
  int width;
} h3_ref;

// Called after every denoising step; return nonzero to cancel.
typedef int (*h3_progress)(void *user, int step, int steps, double seconds);

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// The last failing call's message on this thread, or an empty string.
//
// The pointer stays valid until the next failing call on the same thread; copy it if you need it
// longer. It is never NULL.
const char *h3_last_error(void);

uint32_t h3_abi_version(void);

// # Safety
//
// `config` points to one initialised `h3_config` whose strings are NUL-terminated, and `out_session`
// to one writable pointer. See the requirements in the header.
int h3_create(const struct h3_config *config, struct h3_session **out_session);

// # Safety
//
// `s` is a session from [`h3_create`] that has not been destroyed, or NULL. After this returns the
// pointer is dangling and must not be used again.
void h3_destroy(struct h3_session *s);

// # Safety
//
// `params` points to one initialised `h3_params` and `out` to one writable `h3_shape`.
int h3_shape_for(const struct h3_params *params, struct h3_shape *out);

// # Safety
//
// See the requirements in the header: `ids` holds `n_ids` values and `out` at least `out_elements`.
int h3_text_in(struct h3_session *s, const int32_t *ids, int n_ids, float *out, uintptr_t out_elements);

// Prompt ids to model-space latents.
//
// `keyframes` and `refs` may be NULL with a count of zero, which is the plain text-to-video case:
// there is one entry point rather than two, because a reference-free run is not a different call.
// `noise_video` and `noise_audio` are optional standard-normal draws in the output layouts, for
// reproducible comparisons; without them the seed's own generator is used.
// # Safety
//
// See the requirements in the header: every pointer holds at least what its count says, and the reference and
// keyframe arrays hold `n_refs` and `n_keyframes` initialised structs whose own pointers follow the
// same rules.
int h3_denoise(struct h3_session *s,
               const int32_t *ids,
               int n_ids,
               const struct h3_params *params,
               const struct h3_keyframe *keyframes,
               int n_keyframes,
               const struct h3_ref *refs,
               int n_refs,
               const float *noise_video,
               const float *noise_audio,
               float *video_latents,
               uintptr_t video_elements,
               float *audio_latents,
               uintptr_t audio_elements,
               h3_progress progress,
               void *user);

// # Safety
//
// See the requirements in the header.
int h3_decode_video(struct h3_session *s,
                    const struct h3_params *params,
                    const float *video_latents,
                    uintptr_t video_elements,
                    uint8_t *frames,
                    uintptr_t frame_bytes);

// # Safety
//
// See the requirements in the header: `pixels` holds `frames * height * width * 3` floats.
int h3_encode_video(struct h3_session *s,
                    const float *pixels,
                    int frames,
                    int height,
                    int width,
                    float *latents,
                    uintptr_t latent_elements,
                    int *latent_t);

// # Safety
//
// See the requirements in the header.
int h3_decode_audio(struct h3_session *s,
                    const float *audio_latents,
                    uintptr_t audio_elements,
                    int audio_t,
                    float *samples,
                    uintptr_t sample_elements);

// # Safety
//
// See the requirements in the header: `samples` holds `2 * n_samples` floats, planar stereo.
int h3_encode_audio(struct h3_session *s,
                    const float *samples,
                    int n_samples,
                    float *latents,
                    uintptr_t latent_elements,
                    int *audio_t);

// # Safety
//
// See the requirements in the header: `pixels` holds `height * width * 3` floats.
int h3_vision_embed(struct h3_session *s,
                    const float *pixels,
                    int height,
                    int width,
                    float *merged,
                    uintptr_t merged_elements,
                    float *deepstack,
                    uintptr_t deepstack_elements,
                    int *tokens);

// `tokenizer_json`: an HF tokenizer.json, or NULL for the one compiled into the library
// (`H3_TOKENIZER=<file>` overrides that). Returns NULL on failure; `h3_last_error` says why.
// # Safety
//
// `tokenizer_json` is NUL-terminated or NULL.
struct h3_tokenizer *h3_tokenizer_create(const char *tokenizer_json);

// # Safety
//
// `t` is a tokenizer from [`h3_tokenizer_create`] that has not been destroyed, or NULL.
void h3_tokenizer_destroy(struct h3_tokenizer *t);

// The number of ids the text encodes to, writing up to `capacity` of them; -1 on failure.
//
// The count is returned even when it exceeds the buffer, so a caller can size a second call to it.
// # Safety
//
// `t` is a live tokenizer, `utf8` NUL-terminated, and `ids` holds at least `capacity` values.
int h3_tokenizer_encode(const struct h3_tokenizer *t, const char *utf8, int32_t *ids, uintptr_t capacity);

// The vocabulary size: the largest id plus one among the model's tokens.
// # Safety
//
// `t` is a live tokenizer, or NULL.
int h3_tokenizer_vocab_size(const struct h3_tokenizer *t);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* H3_LOOM_H */
