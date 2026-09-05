// C ABI for the resident MiniMax H3 transformer-block session (the 50 DiT blocks in Loom).
#ifndef H3_LOOM_H
#define H3_LOOM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define H3_ABI_VERSION 1u
#define H3_CLASSES 12u
enum { H3_OK = 0, H3_ERROR = 1, H3_INVALID_ARGUMENT = 64 };
typedef struct h3_session h3_session;
uint32_t h3_abi_version(void);
// kernels_dir holds the HSACOs compiled for exactly `tokens` (attention specialises on the
// sequence length); weights_dir holds weights.bin + manifest.txt from tools/export_weights.py.
int h3_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers,
              h3_session **out_session, char *error, size_t error_capacity);
// x: f16 [tokens][5376] in and out (the residual stream after all blocks);
// cls: i32 [tokens], each row's AdaLN class (timestep class * 3 + modality), < H3_CLASSES;
// mods: f32 [layers][6 * H3_CLASSES][5376], per layer four tables in this order:
//   [H3_CLASSES][2][5376] (scale_msa, shift_msa), [H3_CLASSES][5376] gate_msa,
//   [H3_CLASSES][2][5376] (scale_mlp, shift_mlp), [H3_CLASSES][5376] gate_mlp;
// cos/sin: f32 [tokens][48] (the angle of each rotated pair).
int h3_run(h3_session *s, uint16_t *x, size_t x_elements, const int32_t *cls, size_t cls_elements,
           const float *mods, size_t mods_elements, const float *cos, const float *sin, size_t rope_elements,
           char *error, size_t error_capacity);
int h3_profile(h3_session *s, int enable);
void h3_destroy(h3_session *s);
#ifdef __cplusplus
}
#endif
#endif
