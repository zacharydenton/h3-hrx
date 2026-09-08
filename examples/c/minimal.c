// The smallest complete client of libh3.so: prompt -> frames + samples, written as <out>.rgb and <out>.wav.
// The library is Rust; this is the standing proof that what it exports is callable from plain C.
//   gcc -O2 -I../../include minimal.c -L../../build -lh3 -Wl,-rpath,$(cd ../../build && pwd) -o minimal
//   ./minimal "A red fox ..." [frames] [steps] [out]        (run from the repository root, or set H3_ROOT; models: $H3_MODELS or ~/comfy-models)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h3.h"

static int progress(void *user, int step, int steps, double seconds) {
    (void)user; fprintf(stderr, "  step %d/%d  %.1f s\n", step, steps, seconds); return 0;   /* nonzero cancels */
}
static char *join(const char *a, const char *b) { char *s = malloc(strlen(a) + strlen(b) + 2); sprintf(s, "%s/%s", a, b); return s; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: minimal \"prompt\" [frames] [steps] [out]\n"); return 64; }
    const char *root = getenv("H3_ROOT") ? getenv("H3_ROOT") : ".";
    const char *home = getenv("HOME") ? getenv("HOME") : ".";
    const char *out = argc > 4 ? argv[4] : "minimal";
    if (h3_abi_version() != H3_ABI_VERSION) { fprintf(stderr, "libh3.so ABI %u, header %u\n", h3_abi_version(), H3_ABI_VERSION); return 1; }

    /* text -> ids (NULL: the tokenizer compiled into libh3.so) */
    h3_tokenizer *tok = h3_tokenizer_create(NULL);
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", h3_last_error()); return 1; }
    int32_t ids[4096]; const int n_ids = h3_tokenizer_encode(tok, argv[1], ids, 4096);
    h3_tokenizer_destroy(tok);
    if (n_ids < 0 || n_ids > 4096) { fprintf(stderr, "cannot tokenize the prompt\n"); return 1; }

    /* a session: weights resident, kernels compiled on first use into cache_dir */
    const char *loom = getenv("LOOM_COMPILE") ? getenv("LOOM_COMPILE") : "loom-compile";
    /* ComfyUI's models directory: the four checkpoints, read as they are */
    const char *models = getenv("H3_MODELS") ? getenv("H3_MODELS") : join(home, "comfy-models");
    h3_config cfg = {join(models, "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"), join(models, "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
                         join(models, "vae/minimax_h3_video_vae_fp16.safetensors"), join(models, "vae/minimax_h3_audio_vae_fp32.safetensors"),
                         join(root, "kernels"), join(root, "build/kernel_cache"), loom, 8};
    h3_session *s = NULL;
    if (h3_create(&cfg, &s)) { fprintf(stderr, "create: %s\n", h3_last_error()); return 1; }

    /* sizes from the parameters */
    h3_params p = {480, 864, argc > 2 ? atoi(argv[2]) : 124, argc > 3 ? atoi(argv[3]) : 31, 0, 0.0f, 0.0f, 1, 0.0f};
    h3_shape sh; if (h3_shape_for(&p, &sh)) { fprintf(stderr, "invalid parameters\n"); return 64; }
    const size_t nv = (size_t)24 * sh.latent_t * sh.lat_h * sh.lat_w, na = (size_t)64 * sh.audio_t;
    float *video = malloc(nv * sizeof(float)), *audio = malloc(na * sizeof(float));
    fprintf(stderr, "%d frames, %dx%dx%d latents, %d audio latents, %d prompt tokens\n", sh.frames, sh.latent_t, sh.lat_h, sh.lat_w, sh.audio_t, n_ids);

    /* no keyframes and no references: the plain text-to-video case */
    if (h3_denoise(s, ids, n_ids, &p, NULL, 0, NULL, 0, NULL, NULL, video, nv, audio, na, progress, NULL)) {
        fprintf(stderr, "denoise: %s\n", h3_last_error()); return 1;
    }

    const size_t nf = (size_t)sh.frames * p.height * p.width * 3, ns = (size_t)1600 * sh.audio_t;
    uint8_t *frames = malloc(nf); float *samples = malloc(ns * sizeof(float));
    if (h3_decode_video(s, &p, video, nv, frames, nf)) { fprintf(stderr, "decode video: %s\n", h3_last_error()); return 1; }
    if (h3_decode_audio(s, audio, na, sh.audio_t, samples, ns)) { fprintf(stderr, "decode audio: %s\n", h3_last_error()); return 1; }
    h3_destroy(s);

    char path[4096];
    snprintf(path, sizeof path, "%s.rgb", out); FILE *f = fopen(path, "wb");
    if (!f || fwrite(frames, 1, nf, f) != nf || fclose(f) != 0) { perror(path); return 1; }
    snprintf(path, sizeof path, "%s.wav", out); f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    {   /* 16-bit PCM, interleaved stereo, 32 kHz */
        const uint32_t n = (uint32_t)sh.audio_t * 800, bytes = n * 4, rate = 32000, fmt = 16; const uint16_t pcm = 1, ch = 2, align = 4, bits = 16;
        fwrite("RIFF", 1, 4, f); uint32_t riff = 36 + bytes; fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
        fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); uint32_t bps = rate * 4; fwrite(&bps, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
        fwrite("data", 1, 4, f); fwrite(&bytes, 4, 1, f);
        for (uint32_t i = 0; i < n; ++i) for (int c = 0; c < 2; ++c) { float v = samples[(size_t)c * n + i]; v = v < -1 ? -1 : (v > 1 ? 1 : v); int16_t q = (int16_t)(v * 32767.0f); fwrite(&q, 2, 1, f); }
        if (ferror(f) || fclose(f) != 0) { perror(path); return 1; }
    }
    fprintf(stderr, "wrote %s.rgb (%d x %dx%d rgb24) and %s.wav; mux: ffmpeg -f rawvideo -pix_fmt rgb24 -s %dx%d -r 24 -i %s.rgb -i %s.wav %s.mp4\n",
            out, sh.frames, p.width, p.height, out, p.width, p.height, out, out, out);
    return 0;
}
