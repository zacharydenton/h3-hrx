// C ABI for the resident video-VAE decoder session (the 36 ViT blocks in Loom).
#ifndef H3VAE_LOOM_H
#define H3VAE_LOOM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define H3VAE_ABI_VERSION 1u
enum { H3VAE_OK = 0, H3VAE_ERROR = 1, H3VAE_INVALID_ARGUMENT = 64 };
typedef struct h3vae_session h3vae_session;
uint32_t h3vae_abi_version(void);
// kernels_dir: scripts/build_kernels_vae.py for exactly `tokens`; weights_dir: tools/export_vae.py.
int h3vae_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers,
                 h3vae_session **out_session, char *error, size_t error_capacity);
// x: f32 [tokens][2048] in and out (the token stream after the blocks); cos/sin: f32 [tokens][24].
int h3vae_run(h3vae_session *s, float *x, size_t x_elements, const float *cos, const float *sin, size_t rope_elements,
              char *error, size_t error_capacity);
int h3vae_profile(h3vae_session *s, int enable);
void h3vae_destroy(h3vae_session *s);
#ifdef __cplusplus
}
#endif
#endif
