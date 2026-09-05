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
#define H3PIPE_ABI_VERSION 1u
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
} h3pipe_config;

typedef struct {
    int height, width;           // pixels, multiples of 32
    int frames;                  // snapped up to the next 17n + 5
    int steps;                   // sigma grid points (steps - 1 model evaluations), as diffusers
    uint64_t seed;
    float video_shift, audio_shift;   // 0 -> the model's defaults (12, 3)
} h3pipe_params;

typedef struct {
    int frames, latent_t, lat_h, lat_w, audio_t;       // after snapping; latents [24][latent_t][lat_h][lat_w], audio latents [2][32][audio_t]
    int text_rows_max;
} h3pipe_shape;

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

// Inspection: the refined text rows the blocks see, f32 [n_ids][5376].
int h3pipe_text_in(h3pipe_session *s, const int32_t *ids, int n_ids, float *out, size_t out_elements, char *error, size_t error_capacity);

// Model-space video latents [24][latent_t][lat_h][lat_w] -> RGB8 frames [frames][height][width][3] (the
// pixel count is frames*height*width*3 with frames from h3pipe_shape_for).
int h3pipe_decode_video(h3pipe_session *s, const h3pipe_params *params, const float *video_latents, size_t video_elements,
                        uint8_t *frames, size_t frame_bytes, char *error, size_t error_capacity);

// Model-space audio latents [2][32][audio_t] -> stereo float samples [2][audio_t * 800] at 32 kHz.
int h3pipe_decode_audio(h3pipe_session *s, const float *audio_latents, size_t audio_elements, int audio_t,
                        float *samples, size_t sample_elements, char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
