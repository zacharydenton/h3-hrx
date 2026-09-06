// C ABI for the whole MiniMax H3 text -> video + audio pipeline with every kernel in Loom:
// prompt token ids in, latents / frames / samples out. Everything between the kernels runs
// on the host in plain C++ (layout, AdaLN curves, scheduler, noise, patching, blending).
//
// Kernels are compiled on first use for a shape by spawning `loom-compile` into cache_dir;
// the runtime surface is the HIP runtime API (module load, memory, launch), no device code.
#ifndef H3PIPE_LOOM_H
#define H3PIPE_LOOM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define H3PIPE_ABI_VERSION 4u
enum { H3PIPE_OK = 0, H3PIPE_ERROR = 1, H3PIPE_CANCELLED = 2, H3PIPE_INVALID_ARGUMENT = 64 };
typedef struct h3pipe_session h3pipe_session;

typedef struct {
    const char *glue_dir;        // tools/export_glue.py: conditioning tables, embedders, refiner, VAE heads, audio decoder, embedding table
    const char *blocks_dir;      // tools/gptq_export.py (int4 GPTQ) or tools/export_weights.py: the 50 DiT blocks
    const char *te_dir;          // tools/export_te.py: the text encoder's 50 layers (int8)
    const char *vae_dir;         // tools/export_vae.py --bits 8 (or gptq_export_vae.py with vae_bits 4): the video decoder's 36 blocks
    const char *kernel_sources;  // the repo's kernels/ directory (.loom files)
    const char *cache_dir;       // where compiled .hsaco files live (created)
    const char *loom_compile;    // path to the loom-compile binary
    int vae_bits;                // 8 or 4, matching vae_dir
    const char *aenc_dir;        // tools/export_audio_encoder.py: the audio VAE's encoder (reference audio); NULL -> h3pipe_encode_audio unavailable
    const char *vision_dir;      // tools/export_vision.py: Qwen3-VL's vision tower (reference images in the prompt); NULL -> unavailable
    const char *venc_dir;        // tools/export_vae_encoder.py: the video VAE's encoder (reference images/videos, keyframes); NULL -> unavailable
} h3pipe_config;

typedef struct {
    int height, width;           // pixels, multiples of 32
    int frames;                  // snapped up to the next 17n + 5
    int steps;                   // sigma grid points (steps - 1 model evaluations), as diffusers
    uint64_t seed;
    float video_shift, audio_shift;   // 0 -> the model's defaults (12, 3)
    float cache_threshold;            // first-block step cache: skip blocks 1..49 while the accumulated relative change of block 0's output stays below this (0 = off; 0.05-0.15 typical)
} h3pipe_params;

typedef struct {
    int frames, latent_t, lat_h, lat_w, audio_t;       // after snapping; latents [24][latent_t][lat_h][lat_w], audio latents [2][32][audio_t]
    int text_rows_max;
} h3pipe_shape;

// A reference block (ref2va), in presentation order: kind 0 = image (video_latent [24][1][lat_h][lat_w], model space, from
// h3pipe_encode_image), 1 = audio (audio_latent [2][32][audio_t] from h3pipe_encode_audio), 2 = video (video_latent
// [24][latent_t][lat_h][lat_w], audio_latent optional: its soundtrack). Unused pointers NULL, unused counts 0.
typedef struct {
    int kind;
    const float *video_latent; int latent_t, lat_h, lat_w;
    const float *audio_latent; int audio_t;
    const float *pixels; int height, width;   // images: the reference image as presented to the text encoder, f32 [height][width][3] in [0, 1],
                                              // height and width multiples of 32; the ids carry (height/32)*(width/32) placeholders (-1) for it
} h3pipe_ref;

// Called after every denoising step; return nonzero to cancel.
typedef int (*h3pipe_progress)(void *user, int step, int steps, double seconds);

uint32_t h3pipe_abi_version(void);
int h3pipe_create(const h3pipe_config *config, h3pipe_session **out_session, char *error, size_t error_capacity);
void h3pipe_destroy(h3pipe_session *s);
int h3pipe_shape_for(const h3pipe_params *params, h3pipe_shape *out);

// Prompt ids -> the model-space latents. noise_video / noise_audio are optional standard-normal
// draws in the output layouts (for reproducible comparisons); otherwise the seed's own RNG.
int h3pipe_denoise(h3pipe_session *s, const int32_t *ids, int n_ids, const h3pipe_params *params,
                   const float *noise_video, const float *noise_audio,
                   float *video_latents, size_t video_elements, float *audio_latents, size_t audio_elements,
                   h3pipe_progress progress, void *user, char *error, size_t error_capacity);

// As h3pipe_denoise with reference blocks packed between the text and the target streams (the ids must carry the
// matching presentation: "<Picture i>: " + vision span, "<Audio j>: ", "<Video k>: " blocks, then the prompt).
int h3pipe_denoise_refs(h3pipe_session *s, const int32_t *ids, int n_ids, const h3pipe_params *params, const h3pipe_ref *refs, int n_refs,
                        const float *noise_video, const float *noise_audio,
                        float *video_latents, size_t video_elements, float *audio_latents, size_t audio_elements,
                        h3pipe_progress progress, void *user, char *error, size_t error_capacity);

// Inspection: the refined text rows the blocks see, f32 [n_ids][5376].
int h3pipe_text_in(h3pipe_session *s, const int32_t *ids, int n_ids, float *out, size_t out_elements, char *error, size_t error_capacity);

// Model-space video latents [24][latent_t][lat_h][lat_w] -> RGB8 frames [frames][height][width][3] (the
// pixel count is frames*height*width*3 with frames from h3pipe_shape_for).
int h3pipe_decode_video(h3pipe_session *s, const h3pipe_params *params, const float *video_latents, size_t video_elements,
                        uint8_t *frames, size_t frame_bytes, char *error, size_t error_capacity);

// Model-space audio latents [2][32][audio_t] -> stereo float samples [2][audio_t * 800] at 32 kHz.
int h3pipe_decode_audio(h3pipe_session *s, const float *audio_latents, size_t audio_elements, int audio_t,
                        float *samples, size_t sample_elements, char *error, size_t error_capacity);
// The vision tower on one image: pixels f32 [height][width][3] in [0, 1] (height and width multiples of 32, as the
// reference sizing produces) -> the merged vision embeds [tokens][5120] and the three DeepStack embeds [3][tokens][5120]
// with tokens = (height / 32) * (width / 32); the element counts must be at least those sizes; *tokens is written.
int h3pipe_vision_embed(h3pipe_session *s, const float *pixels, int height, int width, float *merged, size_t merged_elements, float *deepstack, size_t deepstack_elements, int *tokens, char *error, size_t error_capacity);

// The video VAE's encoder. Pixels f32 [frames][height][width][3] in [0, 1] (height and width multiples of 32, up to 2048).
// frames = 1 encodes an image -> latents [24][1][height/16][width/16]; frames > 1 encodes a clip in 17-frame chunks
// (the last repeat-padded) -> [24][latent_t][height/16][width/16] with latent_t = 5 * ceil(frames / 17) - 3, written to *latent_t.
int h3pipe_encode_video(h3pipe_session *s, const float *pixels, int frames, int height, int width, float *latents, size_t latent_elements, int *latent_t, char *error, size_t error_capacity);

// Stereo float samples [2][n_samples] at 32 kHz (right-padded to a multiple of 800 inside) -> model-space audio latents
// [2][32][audio_t] with audio_t = ceil(n_samples / 800) written to *audio_t; latent_elements must be at least 2*32*audio_t.
int h3pipe_encode_audio(h3pipe_session *s, const float *samples, int n_samples, float *latents, size_t latent_elements, int *audio_t, char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
