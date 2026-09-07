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
//   --attn f16|i8|i4    the DiT attention's QK^T operands (i8: the parity path)   --sampler res_multistep|euler
//   --models DIR        ComfyUI's models directory (default $H3_MODELS, else ~/comfy-models): diffusion_models/, text_encoders/, vae/
//   --dit FILE  --te FILE  --video-vae FILE  --audio-vae FILE      the four checkpoints, overriding --models
//   --root DIR          the repository (default: the binary's parent's parent)   --no-decode   --audio-only (voice/sound: 32x32 canvas, wav only)   --still frame.png [--still-frame N] (one frame as an image)   --latents prefix
//   --base-weights      run reference files on the base checkpoint when the ref2va checkpoint is absent (otherwise an error)
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "h3pipe.h"
#include "h3tok.h"

namespace {

constexpr int32_t VISION_START = 151652, VISION_END = 151653;
constexpr int FPS = 24, RATE = 32000;

// The command line, parsed once against a fixed table: unknown options and missing values are errors, and a value
// (a prompt of "--help", say) is never mistaken for an option.
struct Options {
    std::map<std::string, std::string> values; std::set<std::string> flags; std::vector<std::string> files;
    const char *get(const char *name, const char *dflt = nullptr) const { auto it = values.find(name); return it == values.end() ? dflt : it->second.c_str(); }
    bool has(const char *name) const { return flags.count(name) || values.count(name); }
};
const std::set<std::string> VALUED = {"-p", "--first-frame", "--out", "--frames", "--steps", "--width", "--height", "--seed", "--attn", "--sampler", "--root", "--still", "--still-frame", "--latents", "--models", "--dit", "--te", "--video-vae", "--audio-vae"};
const std::set<std::string> FLAGS = {"--no-decode", "--audio-only", "--base-weights", "-h", "--help"};
bool parse_options(int argc, char **argv, Options *o, std::string *error) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.size() > 1 && a[0] == '-') {
            if (FLAGS.count(a)) { o->flags.insert(a); continue; }
            if (!VALUED.count(a)) { *error = "unknown option " + a; return false; }
            if (i + 1 >= argc) { *error = "option " + a + " needs a value"; return false; }
            o->values[a] = argv[++i]; continue;
        }
        o->files.push_back(a);
    }
    return true;
}
// A whole decimal integer in [lo, hi], or an error naming the option.
bool parse_int(const Options &o, const char *name, int dflt, int lo, int hi, int *out, std::string *error) {
    const char *v = o.get(name); if (!v) { *out = dflt; return true; }
    char *end = nullptr; errno = 0; const long long n = strtoll(v, &end, 10);
    if (*v == 0 || *end != 0 || errno != 0 || n < lo || n > hi) { *error = std::string(name) + " must be an integer in " + std::to_string(lo) + ".." + std::to_string(hi) + ", got \"" + v + "\""; return false; }
    *out = int(n); return true;
}
bool parse_choice(const Options &o, const char *name, const char *dflt, const std::set<std::string> &choices, std::string *out, std::string *error) {
    *out = o.get(name, dflt);
    if (choices.count(*out)) return true;
    *error = std::string(name) + " must be one of"; for (const std::string &c : choices) *error += " " + c; *error += ", got \"" + *out + "\""; return false;
}
std::string lower_ext(const std::string &path) {
    const size_t dot = path.rfind('.'), slash = path.rfind('/');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return "";
    std::string e = path.substr(dot + 1); for (char &c : e) c = char(tolower(c)); return e;
}
bool is_image(const std::string &e) { return e == "jpg" || e == "jpeg" || e == "png" || e == "webp" || e == "bmp" || e == "gif" || e == "tif" || e == "tiff"; }
bool is_audio(const std::string &e) { return e == "wav" || e == "mp3" || e == "flac" || e == "ogg" || e == "m4a" || e == "aac" || e == "opus"; }
bool exists(const std::string &path) { return access(path.c_str(), R_OK) == 0; }
// The directory a path would be written into exists and is writable (checked before the long generation begins).
bool dir_writable(const std::string &path) {
    const size_t slash = path.rfind('/'); const std::string dir = slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash);
    struct stat st; return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && access(dir.c_str(), W_OK) == 0;
}
// Every byte to the file, or false with errno set.
bool write_file(const std::string &path, const void *data, size_t bytes) {
    FILE *f = fopen(path.c_str(), "wb"); if (!f) return false;
    const bool written = fwrite(data, 1, bytes, f) == bytes; const int saved = errno;
    if (fclose(f) != 0) return false;
    if (!written) errno = saved;
    return written;
}
// 16-bit PCM WAV bytes: interleaved stereo at RATE from planar float samples [2][n].
std::vector<uint8_t> wav_bytes(const float *samples, uint32_t n) {
    std::vector<uint8_t> w; w.reserve(44 + size_t(n) * 4);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) w.push_back(uint8_t(v >> (8 * i))); }; auto u16 = [&](uint16_t v) { w.push_back(uint8_t(v)); w.push_back(uint8_t(v >> 8)); };
    auto tag = [&](const char *t) { w.insert(w.end(), t, t + 4); };
    const uint32_t data_bytes = n * 4;
    tag("RIFF"); u32(36 + data_bytes); tag("WAVE"); tag("fmt "); u32(16); u16(1); u16(2); u32(RATE); u32(RATE * 4); u16(4); u16(16); tag("data"); u32(data_bytes);
    for (uint32_t i = 0; i < n; ++i) for (int c = 0; c < 2; ++c) { float v = samples[size_t(c) * n + i]; v = v < -1 ? -1 : (v > 1 ? 1 : v); u16(uint16_t(int16_t(v * 32767.0f))); }
    return w;
}
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
    explicit Tok(const std::string &path) { char e[512]; t = h3tok_create(path.empty() ? nullptr : path.c_str(), e, sizeof e); if (!t) fprintf(stderr, "tokenizer: %s\n", e); }
    ~Tok() { if (t) h3tok_destroy(t); }
    // h3tok_encode returns the count the text needs even when the buffer is smaller: size the buffer to it.
    bool encode(const std::string &text, std::vector<int32_t> *ids) const {
        std::vector<int32_t> buf(4096); int n = h3tok_encode(t, text.c_str(), buf.data(), buf.size());
        if (n < 0) return false;
        if (size_t(n) > buf.size()) { buf.resize(size_t(n)); if (h3tok_encode(t, text.c_str(), buf.data(), buf.size()) != n) return false; }
        ids->insert(ids->end(), buf.begin(), buf.begin() + n); return true;
    }
};

}  // namespace

int main(int argc, char **argv) {
    Options opt; std::string perr;
    if (!parse_options(argc, argv, &opt, &perr)) { fprintf(stderr, "h3: %s (h3 --help)\n", perr.c_str()); return 64; }
    if (opt.has("-h") || opt.has("--help")) {
        fprintf(stderr, "usage: h3 [ref.jpg ... ref.wav ...] [-p \"prompt\" | < prompt] [--first-frame img] [--out clip.mp4] [--frames 124] [--steps 31]\n"
                        "          [--width 864] [--height 480] [--seed 0] [--attn f16|i8|i4] [--sampler res_multistep|euler]\n"
                        "          [--models DIR] [--dit FILE] [--te FILE] [--video-vae FILE] [--audio-vae FILE]\n"
                        "          [--root DIR] [--no-decode] [--audio-only] [--still frame.png [--still-frame N]] [--latents prefix] [--base-weights]\n"); return 0;
    }
    std::vector<std::string> image_files, audio_files;
    for (const std::string &f : opt.files) {
        const std::string e = lower_ext(f);
        if (is_image(e)) image_files.push_back(f);
        else if (is_audio(e)) audio_files.push_back(f);
        else { fprintf(stderr, "h3: %s: not an image or audio file by extension\n", f.c_str()); return 64; }
    }
    std::string prompt = opt.get("-p", "");
    if (prompt.empty()) { char buf[1 << 16]; size_t n; while ((n = fread(buf, 1, sizeof buf, stdin)) > 0) prompt.append(buf, n); }
    while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r' || prompt.back() == ' ')) prompt.pop_back();
    if (prompt.empty()) { fprintf(stderr, "h3: no prompt (give -p \"...\" or pipe it on stdin)\n"); return 64; }

    // the options, validated before anything expensive
    std::string attn, sampler; int height, width, n_frames, steps, still_frame;
    if (!parse_choice(opt, "--attn", "i8", {"f16", "i8", "i4"}, &attn, &perr) ||
        !parse_choice(opt, "--sampler", "res_multistep", {"res_multistep", "euler"}, &sampler, &perr) ||
        !parse_int(opt, "--height", 480, 32, 8192, &height, &perr) || !parse_int(opt, "--width", 864, 32, 8192, &width, &perr) ||
        !parse_int(opt, "--frames", 124, 1, 1 << 20, &n_frames, &perr) || !parse_int(opt, "--steps", 31, 2, 1000, &steps, &perr) ||
        !parse_int(opt, "--still-frame", 0, 0, 1 << 20, &still_frame, &perr)) { fprintf(stderr, "h3: %s\n", perr.c_str()); return 64; }
    uint64_t seed = 0;
    if (const char *sv = opt.get("--seed")) { char *end = nullptr; errno = 0; seed = strtoull(sv, &end, 10); if (*sv == 0 || *end != 0 || errno != 0) { fprintf(stderr, "h3: --seed must be an unsigned integer, got \"%s\"\n", sv); return 64; } }
    if (height % 32 || width % 32) { fprintf(stderr, "h3: --width and --height must be multiples of 32\n"); return 64; }
    const bool no_decode = opt.has("--no-decode"), audio_only = opt.has("--audio-only");   // audio only (voice and sound): skip the video decoder, write <out>.wav only (use a 32x32 canvas)
    const char *still = opt.get("--still"), *latents_prefix = opt.get("--latents");
    if (audio_only && still) { fprintf(stderr, "h3: --audio-only decodes no frames; it cannot be combined with --still\n"); return 64; }
    if (no_decode && (still || audio_only)) { fprintf(stderr, "h3: --no-decode decodes nothing; it cannot be combined with --still or --audio-only\n"); return 64; }

    const std::string root = opt.get("--root", exe_root().c_str());
    std::string out = opt.get("--out", "h3_out.mp4");
    if (lower_ext(out) != "mp4") out += ".mp4";
    const std::string out_stem = out.substr(0, out.size() - 4);
    if (!no_decode && !dir_writable(out)) { fprintf(stderr, "h3: cannot write %s: the directory is missing or not writable\n", out.c_str()); return 64; }
    if (latents_prefix && !dir_writable(latents_prefix)) { fprintf(stderr, "h3: cannot write %s.video.f32: the directory is missing or not writable\n", latents_prefix); return 64; }
    if (still && !dir_writable(still)) { fprintf(stderr, "h3: cannot write %s: the directory is missing or not writable\n", still); return 64; }
    // ComfyUI's models directory: the four checkpoints, read as they are
    const char *models_env = getenv("H3_MODELS"), *home0 = getenv("HOME");
    const std::string models = opt.get("--models", models_env && *models_env ? models_env : (std::string(home0 ? home0 : ".") + "/comfy-models").c_str());
    // references (ref2va) need the reference-conditioned checkpoint; the base one only on request
    const bool want_refs = !image_files.empty() || !audio_files.empty();
    const std::string dit_base = models + "/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
    const std::string dit_ref2va = models + "/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors";
    const bool have_ref2va = exists(dit_ref2va), ref2va = want_refs && have_ref2va && !opt.has("--base-weights");
    if (want_refs && !have_ref2va && !opt.has("--base-weights") && !opt.get("--dit")) {
        fprintf(stderr, "h3: reference files need the ref2va checkpoint, %s (README, Weights); --base-weights runs the base checkpoint anyway\n", dit_ref2va.c_str()); return 64;
    }
    const std::string dit = opt.get("--dit", (ref2va ? dit_ref2va : dit_base).c_str());
    const std::string te = opt.get("--te", (models + "/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors").c_str());
    const std::string video_vae = opt.get("--video-vae", (models + "/vae/minimax_h3_video_vae_fp16.safetensors").c_str());
    const std::string audio_vae = opt.get("--audio-vae", (models + "/vae/minimax_h3_audio_vae_fp32.safetensors").c_str());
    if (!exists(dit)) { fprintf(stderr, "h3: %s not found (README, Weights; --models or --dit)\n", dit.c_str()); return 64; }
    const std::string sources = root + "/kernels", cache = root + "/build/kernel_cache";
    const char *loom_env = getenv("LOOM_COMPILE");
    const char *home = getenv("HOME");
    (void)home;

    h3pipe_params p = {height, width, n_frames, steps, seed, 0.0f, 0.0f, sampler == "euler" ? 0 : 1, 0.0f};
    h3pipe_shape sh; if (h3pipe_shape_for(&p, &sh)) { fprintf(stderr, "h3: invalid parameters\n"); return 64; }

    // inputs first (cheap, and they fail fast): the keyframe to the canvas, references to at most the canvas' pixel count
    struct Image { std::vector<float> pixels; int w, h; };
    std::vector<Image> ref_images; Image keyframe; bool have_keyframe = false;
    if (const char *ff = opt.get("--first-frame")) {
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
    Tok tok{std::string()}; if (!tok.t) return 1;   // the tokenizer compiled into libh3pipe (H3_TOKENIZER overrides it)
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
    if (ids.size() > size_t(sh.text_rows_max)) { fprintf(stderr, "h3: the presentation is %zu tokens; the model takes at most %d\n", ids.size(), sh.text_rows_max); return 64; }

    fprintf(stderr, "%d frames at %dx%d: %dx%dx%d latents, %d audio latents; %zu prompt tokens (%d keyframe, %zu reference images, %zu reference audio)%s\n",
            sh.frames, p.width, p.height, sh.latent_t, sh.lat_h, sh.lat_w, sh.audio_t, ids.size(), have_keyframe ? 1 : 0, ref_images.size(), ref_audio.size(), ref2va ? ", ref2va weights" : "");

    h3pipe_config cfg = {dit.c_str(), te.c_str(), video_vae.c_str(), audio_vae.c_str(), sources.c_str(), cache.c_str(), loom_env ? loom_env : "loom-compile",
                         attn == "f16" ? 16 : attn == "i8" ? 8 : 4};
    char err[4096]; h3pipe_session *s = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if (h3pipe_create(&cfg, &s, err, sizeof err)) { fprintf(stderr, "h3: create: %s\n", err); return 1; }
    fprintf(stderr, "session in %.1f s (%s, %s attention)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), dit.substr(dit.find_last_of('/') + 1).c_str(), attn.c_str());

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
    if (latents_prefix) {
        for (const auto &part : {std::make_pair(std::string(latents_prefix) + ".video.f32", &video), std::make_pair(std::string(latents_prefix) + ".audio.f32", &audio)})
            if (!write_file(part.first, part.second->data(), part.second->size() * 4)) { fprintf(stderr, "h3: cannot write %s: %s\n", part.first.c_str(), strerror(errno)); return 1; }
        fprintf(stderr, "wrote %s.video.f32 and %s.audio.f32\n", latents_prefix, latents_prefix);
    }
    if (no_decode) { h3pipe_destroy(s); return 0; }

    t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> frames(audio_only ? 0 : size_t(sh.frames) * p.height * p.width * 3);
    if (!audio_only && h3pipe_decode_video(s, &p, video.data(), video.size(), frames.data(), frames.size(), err, sizeof err)) { fprintf(stderr, "h3: decode video: %s\n", err); return 1; }
    std::vector<float> samples(size_t(2) * sh.audio_t * 800);
    if (h3pipe_decode_audio(s, audio.data(), audio.size(), sh.audio_t, samples.data(), samples.size(), err, sizeof err)) { fprintf(stderr, "h3: decode audio: %s\n", err); return 1; }
    h3pipe_destroy(s);
    fprintf(stderr, "decoded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    {   // <out>.wav: 16-bit PCM, interleaved stereo, 32 kHz
        const std::vector<uint8_t> w = wav_bytes(samples.data(), uint32_t(sh.audio_t) * 800);
        if (!write_file(out_stem + ".wav", w.data(), w.size())) { fprintf(stderr, "h3: cannot write %s.wav: %s\n", out_stem.c_str(), strerror(errno)); return 1; }
    }
    if (still) {   // one frame as an image (any format ffmpeg writes by extension): H3 as an image generator or editor
        const int idx = std::min(sh.frames - 1, still_frame);
        char size[32]; snprintf(size, sizeof size, "%dx%d", p.width, p.height);
        const std::string cmd = "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " + std::string(size) + " -i - -frames:v 1 " + shell_quote(still);
        FILE *ff = popen(cmd.c_str(), "w"); if (!ff) { perror("ffmpeg"); return 1; }
        const size_t fb = size_t(p.height) * p.width * 3;
        if (fwrite(frames.data() + size_t(idx) * fb, 1, fb, ff) != fb) { fprintf(stderr, "h3: short write to ffmpeg\n"); pclose(ff); return 1; }
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
