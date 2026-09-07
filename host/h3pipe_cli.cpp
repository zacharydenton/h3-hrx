// h3pipe: prompt token ids -> raw RGB frames + 16-bit stereo WAV through libh3pipe (every kernel in Loom).
//   h3pipe --ids 32 1234 ... [--frames 124] [--steps 31] [--width 864] [--height 480] [--seed 0] [--out build/clip] [--attn i8|f16|i4] [--models DIR]
//     writes <out>.rgb (frames x height x width x 3, uint8) and <out>.wav (32 kHz stereo);
//     ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r 24 -i out.rgb -i out.wav out.mp4 muxes them.
//   --prompt "text" tokenizes in C (host/h3tok.cpp, the Qwen3-VL tokenizer.json); --ids takes ids from elsewhere (tools/prompt_ids.py).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "h3pipe.h"
#include "h3tok.h"

static const char *arg(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int progress(void *, int step, int steps, double sec) { fprintf(stderr, "  step %d/%d  %.1f s\n", step, steps, sec); return 0; }

int main(int argc, char **argv) {
    std::vector<int32_t> ids;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--ids")) { for (int j = i + 1; j < argc && argv[j][0] != '-'; ++j) ids.push_back(atoi(argv[j])); }
    const std::string root = arg(argc, argv, "--root", "."), out = arg(argc, argv, "--out", "build/clip");
    if (const char *prompt = arg(argc, argv, "--prompt", nullptr)) {          // the tokenizer in C (host/h3tok.cpp)
        const char *tok_path = arg(argc, argv, "--tokenizer", nullptr);   // NULL: the tokenizer compiled into libh3pipe
        char terr[512]; h3tok *tok = h3tok_create(tok_path, terr, sizeof terr);
        if (!tok) { fprintf(stderr, "tokenizer: %s\n", terr); return 1; }
        ids.resize(4096); int n = h3tok_encode(tok, prompt, ids.data(), ids.size());   // the count the text needs, even past the buffer
        if (n >= 0 && size_t(n) > ids.size()) { ids.resize(size_t(n)); if (h3tok_encode(tok, prompt, ids.data(), ids.size()) != n) n = -1; }
        h3tok_destroy(tok);
        if (n < 0) { fprintf(stderr, "tokenizer: cannot encode the prompt\n"); return 1; } ids.resize(size_t(n));
    }
    if (ids.empty()) { fprintf(stderr, "usage: h3pipe (--prompt \"text\" | --ids <token ids...>) [--frames N] [--steps N] [--width W] [--height H] [--seed S] [--out prefix] [--tokenizer tokenizer.json]\n"); return 64; }
    const char *models_env = getenv("H3_MODELS"), *home0 = getenv("HOME");
    const std::string models = arg(argc, argv, "--models", models_env && *models_env ? models_env : (std::string(home0 ? home0 : ".") + "/comfy-models").c_str());
    const std::string dit = arg(argc, argv, "--dit", (models + "/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors").c_str());
    const std::string te = arg(argc, argv, "--te", (models + "/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors").c_str());
    const std::string video_vae = arg(argc, argv, "--video-vae", (models + "/vae/minimax_h3_video_vae_fp16.safetensors").c_str());
    const std::string audio_vae = arg(argc, argv, "--audio-vae", (models + "/vae/minimax_h3_audio_vae_fp32.safetensors").c_str());
    const char *loom = getenv("LOOM_COMPILE");
    const std::string sources = root + "/kernels", cache = root + "/build/kernel_cache";
    h3pipe_config cfg = {dit.c_str(), te.c_str(), video_vae.c_str(), audio_vae.c_str(), sources.c_str(), cache.c_str(), loom ? loom : "loom-compile", std::string(arg(argc, argv, "--attn", "i8")) == "f16" ? 16 : (std::string(arg(argc, argv, "--attn", "i8")) == "i4" ? 4 : 8)};
    h3pipe_params p = {atoi(arg(argc, argv, "--height", "480")), atoi(arg(argc, argv, "--width", "864")), atoi(arg(argc, argv, "--frames", "124")), atoi(arg(argc, argv, "--steps", "31")), (uint64_t)atoll(arg(argc, argv, "--seed", "0")), 0.0f, 0.0f, std::string(arg(argc, argv, "--sampler", "res_multistep")) == "euler" ? 0 : 1, (float)atof(arg(argc, argv, "--cache", "0"))};
    char err[4096]; h3pipe_session *s = nullptr;
    h3pipe_shape sh; if (h3pipe_shape_for(&p, &sh)) { fprintf(stderr, "invalid parameters: width and height are multiples of 32, frames >= 1\n"); return 64; }
    if (h3pipe_create(&cfg, &s, err, sizeof err)) { fprintf(stderr, "create: %s\n", err); return 1; }
    fprintf(stderr, "%d frames at %dx%d: %dx%dx%d latents, %d audio latents, %zu prompt tokens\n", sh.frames, p.width, p.height, sh.latent_t, sh.lat_h, sh.lat_w, sh.audio_t, ids.size());
    std::vector<float> video(size_t(24) * sh.latent_t * sh.lat_h * sh.lat_w), audio(size_t(2) * 32 * sh.audio_t);
    if (h3pipe_denoise(s, ids.data(), int(ids.size()), &p, nullptr, nullptr, video.data(), video.size(), audio.data(), audio.size(), progress, nullptr, err, sizeof err)) { fprintf(stderr, "denoise: %s\n", err); return 1; }
    std::vector<uint8_t> frames(size_t(sh.frames) * p.height * p.width * 3);
    if (h3pipe_decode_video(s, &p, video.data(), video.size(), frames.data(), frames.size(), err, sizeof err)) { fprintf(stderr, "decode video: %s\n", err); return 1; }
    std::vector<float> samples(size_t(2) * sh.audio_t * 800);
    if (h3pipe_decode_audio(s, audio.data(), audio.size(), sh.audio_t, samples.data(), samples.size(), err, sizeof err)) { fprintf(stderr, "decode audio: %s\n", err); return 1; }
    h3pipe_destroy(s);
    { FILE *f = fopen((out + ".rgb").c_str(), "wb"); if (!f) { perror("rgb"); return 1; } const bool ok = fwrite(frames.data(), 1, frames.size(), f) == frames.size(); if (fclose(f) != 0 || !ok) { perror("rgb"); return 1; } }
    {   // 16-bit PCM WAV, interleaved stereo, 32 kHz
        const uint32_t n = uint32_t(sh.audio_t) * 800, data_bytes = n * 4, rate = 32000;
        FILE *f = fopen((out + ".wav").c_str(), "wb"); if (!f) { perror("wav"); return 1; }
        auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); }; auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
        fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVEfmt ", 1, 8, f); u32(16); u16(1); u16(2); u32(rate); u32(rate * 4); u16(4); u16(16); fwrite("data", 1, 4, f); u32(data_bytes);
        for (uint32_t i = 0; i < n; ++i) for (int c = 0; c < 2; ++c) { float v = samples[size_t(c) * n + i]; v = v < -1 ? -1 : (v > 1 ? 1 : v); u16(uint16_t(int16_t(v * 32767.0f))); }
        if (ferror(f) || fclose(f) != 0) { perror("wav"); return 1; }
    }
    fprintf(stderr, "wrote %s.rgb (%d x %dx%d) and %s.wav\n", out.c_str(), sh.frames, p.width, p.height, out.c_str());
    return 0;
}
