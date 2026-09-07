// h3: the whole MiniMax H3 pipeline from the shell, through libh3pipe alone (every kernel in Loom).
//
//   h3 [ref1.jpg ref2.png voice.wav ...] [-p "prompt"] [options] < prompt
//
// Positional files are references by extension: images (.jpg .jpeg .png .webp .bmp .gif .tif .tiff) are presented as
// <Picture i> and encoded by the video VAE's encoder; audio (.wav .mp3 .flac .ogg .m4a .aac .opus) as <Audio j> through
// the audio VAE's encoder. The prompt is read from stdin unless -p is given. ffmpeg decodes the inputs and muxes the
// output; images are resized the way tools/pipeline_c.py did (PIL bilinear, antialiased): references to at most the
// canvas' pixel count on a 32-pixel grid, the keyframe to the canvas.
//
//   --first-frame img   fl2va keyframe          --out clip.mp4    (default h3_out.mp4; <out>.wav is kept next to it)
//   --frames 124        --steps 31 (= 30 evaluations)   --width 864  --height 480   --seed 0
//   --precision int8|bf16|int4  (int8: the checkpoint's int8 rows, int8 QK^T)   --attn f16|i8|i4   --sampler res_multistep|euler
//   --root DIR          the repository (default: the binary's parent's parent)   --no-decode   --audio-only (voice/sound: 32x32 canvas, wav only)   --still frame.png [--still-frame N] (one frame as an image)   --latents prefix
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include <unistd.h>

#include "h3pipe.h"
#include "h3tok.h"

namespace {

constexpr int32_t VISION_START = 151652, VISION_END = 151653;
constexpr int FPS = 24, RATE = 32000;

const char *arg(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
bool flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return true;
    return false;
}
std::string lower_ext(const std::string &path) {
    const size_t dot = path.rfind('.'), slash = path.rfind('/');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return "";
    std::string e = path.substr(dot + 1); for (char &c : e) c = char(tolower(c)); return e;
}
bool is_image(const std::string &e) { return e == "jpg" || e == "jpeg" || e == "png" || e == "webp" || e == "bmp" || e == "gif" || e == "tif" || e == "tiff"; }
bool is_audio(const std::string &e) { return e == "wav" || e == "mp3" || e == "flac" || e == "ogg" || e == "m4a" || e == "aac" || e == "opus"; }
bool exists(const std::string &path) { return access(path.c_str(), R_OK) == 0; }
std::string shell_quote(const std::string &s) { std::string q = "'"; for (char c : s) q += c == '\'' ? "'\\''" : std::string(1, c); return q + "'"; }

// Everything a child process writes to stdout.
std::vector<uint8_t> capture(const std::string &cmd, bool *ok) {
    std::vector<uint8_t> out; FILE *p = popen(cmd.c_str(), "r"); *ok = false;
    if (!p) return out;
    uint8_t buf[1 << 16]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.insert(out.end(), buf, buf + n);
    *ok = pclose(p) == 0; return out;
}

// ffprobe -> (width, height) of the first video stream (an image is one).
bool probe_size(const std::string &path, int *w, int *h) {
    bool ok; auto out = capture("ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 " + shell_quote(path), &ok);
    if (!ok) return false;
    return sscanf(std::string(out.begin(), out.end()).c_str(), "%d,%d", w, h) == 2 && *w > 0 && *h > 0;
}
// ffmpeg -> RGB8 [h][w][3] of an image file (any format ffmpeg reads).
bool decode_image(const std::string &path, std::vector<uint8_t> *rgb, int *w, int *h) {
    if (!probe_size(path, w, h)) return false;
    bool ok; *rgb = capture("ffmpeg -v error -i " + shell_quote(path) + " -frames:v 1 -f rawvideo -pix_fmt rgb24 -", &ok);
    return ok && rgb->size() == size_t(*w) * size_t(*h) * 3;
}
// ffmpeg -> stereo float samples [2][n] at 32 kHz of any audio file (mono duplicated, other rates resampled).
bool decode_audio(const std::string &path, std::vector<float> *samples, int *n) {
    bool ok; auto raw = capture("ffmpeg -v error -i " + shell_quote(path) + " -vn -f f32le -ac 2 -ar 32000 -", &ok);
    if (!ok || raw.size() < 8) return false;
    *n = int(raw.size() / 8); samples->assign(size_t(2) * *n, 0.0f);
    const float *inter = reinterpret_cast<const float *>(raw.data());
    for (int i = 0; i < *n; ++i) { (*samples)[i] = inter[2 * i]; (*samples)[size_t(*n) + i] = inter[2 * i + 1]; }
    return true;
}

// PIL's bilinear resample (ImagingResample): a triangle filter whose support grows with the downscale factor, weights
// normalised per output pixel; separable, horizontal then vertical, in float. Input RGB8 [sh][sw][3] -> f32 [dh][dw][3] in [0, 1].
void pil_weights(int in_len, int out_len, int x, int *lo, int *hi, std::vector<double> *w) {
    const double scale = double(in_len) / out_len, ss = scale >= 1.0 ? scale : 1.0, support = ss;   // bilinear: support 1.0 x the filter scale
    const double center = (x + 0.5) * scale;
    *lo = std::max(0, int(center - support + 0.5)); *hi = std::min(in_len, int(center + support + 0.5));
    w->assign(size_t(*hi - *lo), 0.0); double total = 0;
    for (int i = *lo; i < *hi; ++i) { const double t = std::fabs((i - center + 0.5) / ss); const double v = t < 1.0 ? 1.0 - t : 0.0; (*w)[size_t(i - *lo)] = v; total += v; }
    for (double &v : *w) v = total > 0 ? v / total : 0;
}
std::vector<float> resize_pil_bilinear(const std::vector<uint8_t> &rgb, int sw, int sh, int dw, int dh) {
    std::vector<float> horiz(size_t(sh) * dw * 3), out(size_t(dh) * dw * 3);   // [sh][dw][3], [dh][dw][3]
    std::vector<double> w; int lo, hi;
    for (int x = 0; x < dw; ++x) {
        pil_weights(sw, dw, x, &lo, &hi, &w);
        for (int y = 0; y < sh; ++y)
            for (int c = 0; c < 3; ++c) {
                double acc = 0; for (int i = lo; i < hi; ++i) acc += w[size_t(i - lo)] * rgb[(size_t(y) * sw + i) * 3 + c];
                horiz[(size_t(y) * dw + x) * 3 + c] = float(acc / 255.0);
            }
    }
    for (int y = 0; y < dh; ++y) {
        pil_weights(sh, dh, y, &lo, &hi, &w);
        for (int x = 0; x < dw; ++x)
            for (int c = 0; c < 3; ++c) {
                double acc = 0; for (int i = lo; i < hi; ++i) acc += w[size_t(i - lo)] * horiz[(size_t(i) * dw + x) * 3 + c];
                const float v = float(acc); out[(size_t(y) * dw + x) * 3 + c] = v < 0 ? 0 : (v > 1 ? 1 : v);
            }
    }
    return out;
}

std::string exe_root() {
    char buf[4096]; ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return ".";
    std::string p(buf, size_t(n));
    for (int up = 0; up < 2; ++up) { size_t s = p.rfind('/'); if (s == std::string::npos) return "."; p = p.substr(0, s); }
    return p.empty() ? "/" : p;
}

int progress(void *user, int step, int steps, double sec) {
    auto *t0 = static_cast<std::chrono::steady_clock::time_point *>(user);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - *t0).count();
    fprintf(stderr, "  step %d/%d  %.1f s  (%.0f s left)\n", step, steps, sec, step > 0 ? wall / step * (steps - step) : 0.0);
    return 0;
}

struct Tok {
    h3tok *t = nullptr;
    explicit Tok(const std::string &path) { char e[512]; t = h3tok_create(path.c_str(), e, sizeof e); if (!t) fprintf(stderr, "tokenizer: %s\n", e); }
    ~Tok() { if (t) h3tok_destroy(t); }
    bool encode(const std::string &text, std::vector<int32_t> *ids) const {
        std::vector<int32_t> buf(1 << 15); const int n = h3tok_encode(t, text.c_str(), buf.data(), buf.size());
        if (n < 0) return false; ids->insert(ids->end(), buf.begin(), buf.begin() + n); return true;
    }
};

}  // namespace

int main(int argc, char **argv) {
    std::vector<std::string> image_files, audio_files;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') { if (strcmp(argv[i], "--no-decode") && strcmp(argv[i], "--audio-only") && strcmp(argv[i], "-h") && strcmp(argv[i], "--help")) ++i; continue; }
        const std::string e = lower_ext(argv[i]);
        if (is_image(e)) image_files.push_back(argv[i]);
        else if (is_audio(e)) audio_files.push_back(argv[i]);
        else { fprintf(stderr, "h3: %s: not an image or audio file by extension\n", argv[i]); return 64; }
    }
    if (flag(argc, argv, "-h") || flag(argc, argv, "--help")) {
        fprintf(stderr, "usage: h3 [ref.jpg ... ref.wav ...] [-p \"prompt\" | < prompt] [--first-frame img] [--out clip.mp4] [--frames 124] [--steps 31]\n"
                        "          [--width 864] [--height 480] [--seed 0] [--precision int8|bf16|int4] [--attn f16|i8|i4] [--sampler res_multistep|euler]\n"
                        "          [--root DIR] [--no-decode] [--audio-only] [--still frame.png [--still-frame N]] [--latents prefix]\n"); return 64;
    }
    std::string prompt = arg(argc, argv, "-p", "");
    if (prompt.empty()) { char buf[1 << 16]; size_t n; while ((n = fread(buf, 1, sizeof buf, stdin)) > 0) prompt.append(buf, n); }
    while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r' || prompt.back() == ' ')) prompt.pop_back();
    if (prompt.empty()) { fprintf(stderr, "h3: no prompt (give -p \"...\" or pipe it on stdin)\n"); return 64; }

    const std::string root = arg(argc, argv, "--root", exe_root().c_str());
    std::string out = arg(argc, argv, "--out", "h3_out.mp4");
    if (lower_ext(out) != "mp4") out += ".mp4";
    const std::string out_stem = out.substr(0, out.size() - 4);
    const std::string precision = arg(argc, argv, "--precision", "int8");
    const int bits = precision == "bf16" ? 16 : precision == "int4" ? 4 : 8;
    const std::string wdir = bits == 16 ? "weights_f16" : bits == 4 ? "weights_gptq" : "weights_i8";
    const bool ref2va = (!image_files.empty() || !audio_files.empty()) && exists(root + "/build/" + wdir + "_ref2va/manifest.txt") && exists(root + "/build/weights_glue_ref2va/manifest.txt");
    const std::string blocks = root + "/build/" + wdir + (ref2va ? "_ref2va" : ""), glue = root + "/build/weights_glue" + (ref2va ? "_ref2va" : "");
    const std::string te = root + "/build/weights_te", vae = root + "/build/weights_vae_i8", aenc = root + "/build/weights_aenc", vision = root + "/build/weights_vision", venc = root + "/build/weights_venc";
    const std::string sources = root + "/kernels", cache = root + "/build/kernel_cache";
    const char *loom_env = getenv("LOOM_COMPILE");
    const std::string attn = arg(argc, argv, "--attn", bits == 16 ? "f16" : bits == 4 ? "i4" : "i8");
    const char *home = getenv("HOME");
    const std::string tok_path = getenv("H3_TOKENIZER") ? getenv("H3_TOKENIZER") : std::string(home ? home : ".") + "/h3-models/tokenizer/tokenizer.json";

    h3pipe_params p = {atoi(arg(argc, argv, "--height", "480")), atoi(arg(argc, argv, "--width", "864")), atoi(arg(argc, argv, "--frames", "124")), atoi(arg(argc, argv, "--steps", "31")),
                       (uint64_t)atoll(arg(argc, argv, "--seed", "0")), 0.0f, 0.0f, std::string(arg(argc, argv, "--sampler", "res_multistep")) == "euler" ? 0 : 1, 0.0f};
    if (p.height % 32 || p.width % 32 || p.height < 32 || p.width < 32) { fprintf(stderr, "h3: --width and --height must be multiples of 32\n"); return 64; }
    h3pipe_shape sh; if (h3pipe_shape_for(&p, &sh)) { fprintf(stderr, "h3: invalid parameters\n"); return 64; }

    // inputs first (cheap, and they fail fast): the keyframe to the canvas, references to at most the canvas' pixel count
    struct Image { std::vector<float> pixels; int w, h; };
    std::vector<Image> ref_images; Image keyframe; bool have_keyframe = false;
    if (const char *ff = arg(argc, argv, "--first-frame", nullptr)) {
        std::vector<uint8_t> rgb; int w, h;
        if (!decode_image(ff, &rgb, &w, &h)) { fprintf(stderr, "h3: cannot decode %s (ffmpeg)\n", ff); return 1; }
        keyframe = {resize_pil_bilinear(rgb, w, h, p.width, p.height), p.width, p.height}; have_keyframe = true;
    }
    for (const std::string &path : image_files) {
        std::vector<uint8_t> rgb; int w, h;
        if (!decode_image(path, &rgb, &w, &h)) { fprintf(stderr, "h3: cannot decode %s (ffmpeg)\n", path.c_str()); return 1; }
        const double scale = std::fmin(1.0, std::sqrt(double(p.width) * p.height / (double(w) * h)));
        const int tw = std::max(32, int(std::lround(w * scale / 32.0)) * 32), th = std::max(32, int(std::lround(h * scale / 32.0)) * 32);
        ref_images.push_back({resize_pil_bilinear(rgb, w, h, tw, th), tw, th});
    }
    std::vector<std::vector<float>> ref_audio; std::vector<int> ref_audio_n;
    for (const std::string &path : audio_files) {
        std::vector<float> s; int n;
        if (!decode_audio(path, &s, &n)) { fprintf(stderr, "h3: cannot decode %s (ffmpeg)\n", path.c_str()); return 1; }
        ref_audio.push_back(std::move(s)); ref_audio_n.push_back(n);
    }

    // the presentation: keyframe, then reference images ("<Picture i>: " + a vision span), then "<Audio j>: ", then the prompt
    Tok tok(tok_path); if (!tok.t) return 1;
    std::vector<int32_t> ids; int picture = 0;
    auto vision_span = [&](int w, int h) {
        char label[64]; snprintf(label, sizeof label, "<Picture %d>: ", ++picture);
        if (!tok.encode(label, &ids)) return false;
        ids.push_back(VISION_START); ids.insert(ids.end(), size_t(h / 32) * size_t(w / 32), -1); ids.push_back(VISION_END); return true;
    };
    if (have_keyframe && !vision_span(keyframe.w, keyframe.h)) return 1;
    for (const Image &im : ref_images) if (!vision_span(im.w, im.h)) return 1;
    for (size_t j = 0; j < ref_audio.size(); ++j) { char label[64]; snprintf(label, sizeof label, "<Audio %zu>: ", j + 1); if (!tok.encode(label, &ids)) return 1; }
    if (!tok.encode(prompt, &ids)) { fprintf(stderr, "h3: cannot tokenize the prompt\n"); return 1; }

    fprintf(stderr, "%d frames at %dx%d: %dx%dx%d latents, %d audio latents; %zu prompt tokens (%d keyframe, %zu reference images, %zu reference audio)%s\n",
            sh.frames, p.width, p.height, sh.latent_t, sh.lat_h, sh.lat_w, sh.audio_t, ids.size(), have_keyframe ? 1 : 0, ref_images.size(), ref_audio.size(), ref2va ? ", ref2va weights" : "");

    h3pipe_config cfg = {glue.c_str(), blocks.c_str(), te.c_str(), vae.c_str(), sources.c_str(), cache.c_str(), loom_env ? loom_env : "loom-compile", 8,
                         exists(aenc + "/manifest.txt") ? aenc.c_str() : nullptr, exists(vision + "/manifest.txt") ? vision.c_str() : nullptr, exists(venc + "/manifest.txt") ? venc.c_str() : nullptr,
                         attn == "f16" ? 16 : attn == "i8" ? 8 : 4};
    char err[4096]; h3pipe_session *s = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if (h3pipe_create(&cfg, &s, err, sizeof err)) { fprintf(stderr, "h3: create: %s\n", err); return 1; }
    fprintf(stderr, "session in %.1f s (%s, %s attention)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), blocks.c_str() + root.size() + 7, attn.c_str());

    // encoders: latents for the keyframe and the references
    std::vector<h3pipe_keyframe> kfs; std::vector<h3pipe_ref> refs;
    std::vector<std::vector<float>> latent_store;   // keeps the latent buffers alive behind the raw pointers
    latent_store.reserve(1 + ref_images.size() + ref_audio.size());
    if (have_keyframe) {
        latent_store.emplace_back(size_t(24) * (p.height / 16) * (p.width / 16)); int lt = 0;
        if (h3pipe_encode_video(s, keyframe.pixels.data(), 1, keyframe.h, keyframe.w, latent_store.back().data(), latent_store.back().size(), &lt, err, sizeof err)) { fprintf(stderr, "h3: encode keyframe: %s\n", err); return 1; }
        kfs.push_back({0, latent_store.back().data(), keyframe.pixels.data(), keyframe.h, keyframe.w, nullptr, 0});
    }
    for (const Image &im : ref_images) {
        latent_store.emplace_back(size_t(24) * (im.h / 16) * (im.w / 16)); int lt = 0;
        if (h3pipe_encode_video(s, im.pixels.data(), 1, im.h, im.w, latent_store.back().data(), latent_store.back().size(), &lt, err, sizeof err)) { fprintf(stderr, "h3: encode reference image: %s\n", err); return 1; }
        h3pipe_ref r = {}; r.kind = 0; r.video_latent = latent_store.back().data(); r.latent_t = 1; r.lat_h = im.h / 16; r.lat_w = im.w / 16; r.pixels = im.pixels.data(); r.height = im.h; r.width = im.w;
        refs.push_back(r);
    }
    for (size_t j = 0; j < ref_audio.size(); ++j) {
        const int T = (ref_audio_n[j] + 799) / 800; latent_store.emplace_back(size_t(2) * 32 * T); int at = 0;
        if (h3pipe_encode_audio(s, ref_audio[j].data(), ref_audio_n[j], latent_store.back().data(), latent_store.back().size(), &at, err, sizeof err)) { fprintf(stderr, "h3: encode reference audio: %s\n", err); return 1; }
        h3pipe_ref r = {}; r.kind = 1; r.audio_latent = latent_store.back().data(); r.audio_t = at;
        refs.push_back(r);
    }

    std::vector<float> video(size_t(24) * sh.latent_t * sh.lat_h * sh.lat_w), audio(size_t(2) * 32 * sh.audio_t);
    t0 = std::chrono::steady_clock::now();
    const int rc = (kfs.empty() && refs.empty())
        ? h3pipe_denoise(s, ids.data(), int(ids.size()), &p, nullptr, nullptr, video.data(), video.size(), audio.data(), audio.size(), progress, &t0, err, sizeof err)
        : h3pipe_denoise_refs(s, ids.data(), int(ids.size()), &p, kfs.data(), int(kfs.size()), refs.data(), int(refs.size()), nullptr, nullptr, video.data(), video.size(), audio.data(), audio.size(), progress, &t0, err, sizeof err);
    if (rc) { fprintf(stderr, "h3: denoise: %s\n", err); return 1; }
    fprintf(stderr, "denoised in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (const char *lat = arg(argc, argv, "--latents", nullptr)) {
        FILE *f = fopen((std::string(lat) + ".video.f32").c_str(), "wb"); if (f) { fwrite(video.data(), 4, video.size(), f); fclose(f); }
        f = fopen((std::string(lat) + ".audio.f32").c_str(), "wb"); if (f) { fwrite(audio.data(), 4, audio.size(), f); fclose(f); }
    }
    if (flag(argc, argv, "--no-decode")) { h3pipe_destroy(s); return 0; }

    const bool audio_only = flag(argc, argv, "--audio-only");   // voice and sound: skip the video decoder, write <out>.wav only (use a 32x32 canvas)
    t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> frames(audio_only ? 0 : size_t(sh.frames) * p.height * p.width * 3);
    if (!audio_only && h3pipe_decode_video(s, &p, video.data(), video.size(), frames.data(), frames.size(), err, sizeof err)) { fprintf(stderr, "h3: decode video: %s\n", err); return 1; }
    std::vector<float> samples(size_t(2) * sh.audio_t * 800);
    if (h3pipe_decode_audio(s, audio.data(), audio.size(), sh.audio_t, samples.data(), samples.size(), err, sizeof err)) { fprintf(stderr, "h3: decode audio: %s\n", err); return 1; }
    h3pipe_destroy(s);
    fprintf(stderr, "decoded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    {   // <out>.wav: 16-bit PCM, interleaved stereo, 32 kHz
        const uint32_t n = uint32_t(sh.audio_t) * 800, data_bytes = n * 4;
        FILE *f = fopen((out_stem + ".wav").c_str(), "wb"); if (!f) { perror("wav"); return 1; }
        auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); }; auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
        fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVEfmt ", 1, 8, f); u32(16); u16(1); u16(2); u32(RATE); u32(RATE * 4); u16(4); u16(16); fwrite("data", 1, 4, f); u32(data_bytes);
        for (uint32_t i = 0; i < n; ++i) for (int c = 0; c < 2; ++c) { float v = samples[size_t(c) * n + i]; v = v < -1 ? -1 : (v > 1 ? 1 : v); u16(uint16_t(int16_t(v * 32767.0f))); }
        fclose(f);
    }
    if (const char *still = arg(argc, argv, "--still", nullptr)) {   // one frame as an image (any format ffmpeg writes by extension): H3 as an image generator or editor
        const int idx = std::min(sh.frames - 1, std::max(0, atoi(arg(argc, argv, "--still-frame", "0"))));
        char size[32]; snprintf(size, sizeof size, "%dx%d", p.width, p.height);
        const std::string cmd = "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " + std::string(size) + " -i - -frames:v 1 " + shell_quote(still);
        FILE *ff = popen(cmd.c_str(), "w"); if (!ff) { perror("ffmpeg"); return 1; }
        const size_t fb = size_t(p.height) * p.width * 3; fwrite(frames.data() + size_t(idx) * fb, 1, fb, ff);
        if (pclose(ff) != 0) { fprintf(stderr, "h3: ffmpeg failed writing %s\n", still); return 1; }
        fprintf(stderr, "wrote %s (frame %d)\n", still, idx);
    }
    if (audio_only) { fprintf(stderr, "wrote %s.wav (%.2f s, 32 kHz stereo)\n", out_stem.c_str(), double(sh.audio_t) * 800 / RATE); printf("%s.wav\n", out_stem.c_str()); return 0; }
    {   // frames straight into ffmpeg's stdin
        char size[32]; snprintf(size, sizeof size, "%dx%d", p.width, p.height);
        const std::string cmd = "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " + std::string(size) + " -r " + std::to_string(FPS) + " -i - -i " + shell_quote(out_stem + ".wav") +
                                " -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest " + shell_quote(out);
        FILE *ff = popen(cmd.c_str(), "w"); if (!ff) { perror("ffmpeg"); return 1; }
        if (fwrite(frames.data(), 1, frames.size(), ff) != frames.size()) { fprintf(stderr, "h3: short write to ffmpeg\n"); pclose(ff); return 1; }
        if (pclose(ff) != 0) { fprintf(stderr, "h3: ffmpeg failed\n"); return 1; }
    }
    fprintf(stderr, "wrote %s (%d frames, %dx%d, %.2f s) and %s.wav\n", out.c_str(), sh.frames, p.width, p.height, double(sh.frames) / FPS, out_stem.c_str());
    printf("%s\n", out.c_str());
    return 0;
}
