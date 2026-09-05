// C ABI for the resident text-encoder session (Qwen3-VL-32B's 50 language-model layers in Loom).
#ifndef H3TE_LOOM_H
#define H3TE_LOOM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define H3TE_ABI_VERSION 1u
enum { H3TE_OK = 0, H3TE_ERROR = 1, H3TE_INVALID_ARGUMENT = 64 };
typedef struct h3te_session h3te_session;
uint32_t h3te_abi_version(void);
// kernels_dir: scripts/build_kernels_te.py for exactly `tokens`; weights_dir: tools/export_te.py.
int h3te_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers,
                h3te_session **out_session, char *error, size_t error_capacity);
// x: f32 [tokens][5120] in (the token embeddings) and out (the unnormalised hidden state after
// `layers` layers); cos/sin: f32 [tokens][64], the rotary tables over the 128 head channels.
int h3te_run(h3te_session *s, float *x, size_t x_elements, const float *cos, const float *sin, size_t rope_elements,
             char *error, size_t error_capacity);
int h3te_profile(h3te_session *s, int enable);
void h3te_destroy(h3te_session *s);
#ifdef __cplusplus
}
#endif
#endif
