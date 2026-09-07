// The MiniMax H3 pipeline as one C library: every kernel in Loom (compiled on first use through
// loom-compile into a cache), the host doing only what is under a megabyte per step. One generic
// transformer stack serves the text encoder, the token refiner, the 50 DiT blocks and the video
// decoder; the embedders and the final layer are the same int8 GEMMs with padded K / N.
//
// Build: ./scripts/build_host.sh  (host-only code against the HIP runtime API)
#include "rt.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>

#include "checkpoint.h"
#include "h3pipe.h"

extern char **environ;

namespace {

// --- the model's shapes --------------------------------------------------------------------
constexpr int HID = 5376, HEADS = 56, HEAD_DIM = 128, FFN = 14336, ROPE_DIM = 96, ROPE_HALF = 48;
constexpr int TEXT_DIM = 5120, VIDEO_PATCH = 96, AUDIO_CH = 32, FINAL_N = 128;
constexpr int CLASSES = 12, MODALITIES = 3, MODS_ROWS = 6 * CLASSES;
constexpr float VISUAL_COND_AUG = 0.999f;                                  // ComfyUI's VISUAL_COND_TIMESTEP: reference latents at 0.999 * z + 0.001 * noise, timestep class max(t_v, 0.999)      // the AdaLN table rows per layer
constexpr int TE_HID = 5120, TE_HEADS = 64, TE_KV = 8, TE_FFN = 25600, TE_ROPE_HALF = 64;
constexpr int LATENT_CH = 24, FPS = 24, AUDIO_LATENTS_PER_S = 40;
constexpr int VAE_HID = 2048, VAE_HEADS = 32, VAE_D = 64, VAE_FFN = 8192, VAE_ROPE_HALF = 24, VAE_PT = 4, VAE_PS = 16, VAE_OUT = 3 * VAE_PT * VAE_PS * VAE_PS, VAE_REG = 4;
constexpr int VAE_KIN = 64;   // the decoder's 24 latent channels padded to the f16 GEMM's multiple of 64
constexpr int VAE_CHUNK = 5, VAE_OVERLAP = 2, VAE_TOKEN_DROP = 3, VAE_TRATIO = 4, VAE_CLIP = 17;   // tokens_chunk_size, token_overlap, token_drop, temporal ratio, clip_length
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f}, IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};
constexpr double FRAME_RESCALE = 5.0 / 3.0, SPATIAL_SCALE = 32.0;
constexpr int FRAME_PER_TOKEN[5] = {1, 4, 4, 4, 4};
constexpr int THREADS = 256;


// --- small host helpers --------------------------------------------------------------------
int mrope_axis(int pair) { return pair < 60 ? pair % 3 : 0; }   // interleaved Qwen3-VL [24, 20, 20]

struct DecoderGrid {
    int frames = 0, height = 0, width = 0;
    bool matches(int f, int h, int w) const { return frames == f && height == h && width == w; }
};

int decoder_chunks(int tokens, int padding) {
    return std::max(1, (tokens + VAE_TOKEN_DROP + padding) / VAE_CHUNK - 1);
}

uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000; int exp = int((x >> 23) & 0xff) - 127 + 15; uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return uint16_t(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000; const int shift = 14 - exp;
        uint32_t half = mant >> shift; const uint32_t rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = (uint32_t(exp) << 10) | (mant >> 13); const uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
    return uint16_t(sign | half);
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000) << 16; uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff; uint32_t x;
    if (exp == 0) { if (mant == 0) x = sign; else { float f = std::ldexp(float(mant), -24); memcpy(&x, &f, 4); x |= sign; } }
    else if (exp == 31) x = sign | 0x7f800000 | (mant << 13);
    else x = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &x, 4); return f;
}
float bf16_to_f32(uint16_t b) { const uint32_t x = uint32_t(b) << 16; float f; memcpy(&f, &x, 4); return f; }

struct Rng {                      // splitmix64 -> Box-Muller normals
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { uint64_t z = (s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
    double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
    float normal() { const double u1 = std::max(uniform(), 1e-300), u2 = uniform(); return float(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2)); }
};

std::string fmt(const char *f, ...) { char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b; }
std::string num(double v) { char b[64]; snprintf(b, sizeof b, "%.17g", v); return b; }
bool exists(const std::string &p) { struct stat st; return stat(p.c_str(), &st) == 0; }
void mkdirs(const std::string &p) { std::string cur; for (size_t i = 0; i <= p.size(); ++i) { if (i == p.size() || p[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); } if (i < p.size()) cur += p[i]; } }

// --- weights ---------------------------------------------------------------------------------
// Own device allocations even when a constructor or a stage throws.
class DeviceBuffers {
    std::vector<void *> pointers_;
public:
    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers &) = delete;
    DeviceBuffers &operator=(const DeviceBuffers &) = delete;
    ~DeviceBuffers() { clear(); }
    void *alloc(size_t bytes) {
        void *p = rt().alloc(bytes);
        try { pointers_.push_back(p); } catch (...) { rt().free(p); throw; }
        return p;
    }
    void free(void *p) noexcept {
        auto it = std::find(pointers_.begin(), pointers_.end(), p);
        if (it != pointers_.end()) { rt().free(p); pointers_.erase(it); }
    }
    void clear() noexcept { for (void *p : pointers_) rt().free(p); pointers_.clear(); }
};

// A checkpoint tensor's home on the device and where its bytes come from: ordered row segments of the mapped file
// (a fused operand is several tensors' rows; the gate/up interleave is 16-row runs of two halves), written at a pitch
// whose pad bytes stay zero (the GEMMs read k_size of the k_stride pitch); or a small host-built array (per-row scales
// in the same order, bf16/f16 vectors widened to f32 for the kernels that take f32, exp of the Snake parameters).
// Nothing here rotates or quantises: the checkpoint's dtypes are what the kernels run.
struct Segment { const char *src; size_t rows; };
struct Recipe {
    size_t rows = 0, row_bytes = 0, pitch_bytes = 0;
    std::vector<Segment> segments;
    std::function<std::vector<char>()> build; size_t built_bytes = 0;
    size_t device_bytes() const { return build ? built_bytes : rows * pitch_bytes; }
};

class Weights {
    std::unique_ptr<Checkpoint> file_; std::map<std::string, Recipe> recipes_;
    mutable DeviceBuffers memory_; mutable std::map<std::string, char *> tensors_;
public:
    bool is_open() const { return bool(file_); }
    const Checkpoint &file() const { if (!file_) throw std::runtime_error("no checkpoint open"); return *file_; }
    // Map the file and build its recipe table (the plan validates every source tensor's dtype and shape up front).
    void open(const std::string &path, void (*plan)(const Checkpoint &, Weights &)) {
        auto next = std::make_unique<Checkpoint>(path);
        std::map<std::string, Recipe> keep; keep.swap(recipes_);
        try { plan(*next, *this); } catch (...) { recipes_.swap(keep); throw; }
        memory_.clear(); tensors_.clear(); file_ = std::move(next);
    }
    void add(const std::string &name, Recipe r) { recipes_[name] = std::move(r); }
    bool has(const std::string &name) const { return recipes_.count(name) != 0; }
    const Recipe &recipe(const std::string &name) const {
        auto it = recipes_.find(name);
        if (it == recipes_.end()) throw std::runtime_error("no recipe for tensor " + name + (file_ ? " in " + file_->path : std::string()));
        return it->second;
    }
    // The recipe's bytes on the host: the built array, or the segments gathered at the pitch.
    std::vector<char> assemble(const Recipe &r) const {
        if (r.build) { std::vector<char> h = r.build(); if (h.size() != r.built_bytes) throw std::runtime_error("a built tensor's size does not match its recipe"); return h; }
        std::vector<char> h(r.rows * r.pitch_bytes, 0); size_t row = 0;
        for (const Segment &sg : r.segments) for (size_t i = 0; i < sg.rows; ++i, ++row) memcpy(h.data() + row * r.pitch_bytes, sg.src + i * r.row_bytes, r.row_bytes);
        if (row != r.rows) throw std::runtime_error("a recipe's segments do not add up to its rows");
        return h;
    }
    char *at(const std::string &name, size_t bytes) const {
        const Recipe &r = recipe(name);
        if (r.device_bytes() != bytes) throw std::runtime_error("tensor " + name + " is " + std::to_string(r.device_bytes()) + " bytes on the device, expected " + std::to_string(bytes));
        auto it = tensors_.find(name); if (it != tensors_.end()) return it->second;
        char *p = (char *)memory_.alloc(std::max(bytes, size_t(1)));
        try {
            const size_t chunk = size_t(16) << 20;
            if (r.build) { const std::vector<char> h = assemble(r); rt().h2d(p, h.data(), h.size()); }
            else if (r.segments.size() == 1 && r.pitch_bytes == r.row_bytes) {   // one straight run of the map
                const char *src = r.segments[0].src; file_->will_need(src, bytes);
                for (size_t done = 0; done < bytes;) { const size_t n = std::min(chunk, bytes - done); rt().h2d(p + done, src + done, n); done += n; }
            } else {                                                                 // rows gathered at the pitch through a zeroed staging buffer
                const size_t per = std::max<size_t>(1, chunk / r.pitch_bytes); std::vector<char> stage(per * r.pitch_bytes, 0); size_t staged = 0, written = 0;
                auto flush = [&] { rt().h2d(p + written * r.pitch_bytes, stage.data(), staged * r.pitch_bytes); written += staged; staged = 0; };
                for (const Segment &sg : r.segments) {
                    file_->will_need(sg.src, sg.rows * r.row_bytes);
                    for (size_t i = 0; i < sg.rows; ++i) { memcpy(stage.data() + staged * r.pitch_bytes, sg.src + i * r.row_bytes, r.row_bytes); if (++staged == per) flush(); }
                }
                if (staged) flush();
                if (written != r.rows) throw std::runtime_error("a recipe's segments do not add up to its rows");
            }
            tensors_.emplace(name, p);
        } catch (...) { memory_.free(p); throw; }
        return p;
    }
    char *rows(const std::string &name, size_t n, size_t row_bytes, size_t pitch_bytes) const {
        const Recipe &r = recipe(name);
        if (r.build || r.rows != n || r.row_bytes != row_bytes || r.pitch_bytes != pitch_bytes)
            throw std::runtime_error("tensor " + name + " is not " + std::to_string(n) + " rows of " + std::to_string(row_bytes) + " bytes at pitch " + std::to_string(pitch_bytes));
        return at(name, n * pitch_bytes);
    }
    std::vector<float> host_f32(const std::string &name, size_t count) const {
        const std::vector<char> h = assemble(recipe(name));
        if (h.size() != count * 4) throw std::runtime_error("tensor " + name + " is " + std::to_string(h.size()) + " bytes, expected " + std::to_string(count * 4));
        std::vector<float> v(count); memcpy(v.data(), h.data(), h.size()); return v;
    }
};

// --- recipe primitives: layout only, bit for bit ---------------------------------------------
using Entry = Checkpoint::Entry;
// The rows of one or more tensors of the same row width, in order, at pitch_bytes (0: the row width).
Recipe rows_of(const Checkpoint &ck, std::initializer_list<const Entry *> parts, size_t pitch_bytes = 0) {
    Recipe r;
    for (const Entry *e : parts) {
        if (r.segments.empty()) r.row_bytes = e->row_bytes(); else if (e->row_bytes() != r.row_bytes) throw std::runtime_error("concatenated tensors differ in row width");
        r.segments.push_back({ck.data(*e), e->rows()}); r.rows += e->rows();
    }
    r.pitch_bytes = pitch_bytes ? pitch_bytes : r.row_bytes;
    if (r.pitch_bytes < r.row_bytes) throw std::runtime_error("a pitch narrower than the rows");
    return r;
}
// 16-row runs alternating the two halves (first rows i..i+16, then second rows i..i+16): the gate|up operand of the fused
// SwiGLU GEMM (tools/export_weights.py's interleave_gate_up). Each half is `rows_each` rows starting at its tensor's row0.
Recipe interleave16(const Checkpoint &ck, const Entry &first, size_t first_row0, const Entry &second, size_t second_row0, size_t rows_each, size_t pitch_bytes = 0) {
    if (first.row_bytes() != second.row_bytes() || rows_each % 16 || first_row0 + rows_each > first.rows() || second_row0 + rows_each > second.rows()) throw std::runtime_error("interleave: the halves do not fit");
    Recipe r; r.row_bytes = first.row_bytes(); r.pitch_bytes = pitch_bytes ? pitch_bytes : r.row_bytes; r.rows = 2 * rows_each;
    for (size_t i = 0; i < rows_each; i += 16) { r.segments.push_back({ck.data(first) + (first_row0 + i) * r.row_bytes, 16}); r.segments.push_back({ck.data(second) + (second_row0 + i) * r.row_bytes, 16}); }
    return r;
}
float widen_one(const Entry &e, const char *p) {
    if (e.dtype == "F32") { float f; memcpy(&f, p, 4); return f; }
    if (e.dtype == "F16") { uint16_t h; memcpy(&h, p, 2); return f16_to_f32(h); }
    if (e.dtype == "BF16") { uint16_t h; memcpy(&h, p, 2); return bf16_to_f32(h); }
    throw std::runtime_error("cannot widen " + e.dtype + " to f32");
}
size_t elem_bytes(const Entry &e) { return e.dtype == "F32" ? 4 : (e.dtype == "F16" || e.dtype == "BF16") ? 2 : e.dtype == "I8" ? 1 : 0; }
// `pad_rows` zero rows after the tensor's own (a padded N: the vision MLP's 4304 -> 4352).
Recipe rows_padded(const Checkpoint &ck, const Entry &e, size_t pad_rows, size_t pitch_bytes = 0) {
    Recipe r = rows_of(ck, {&e}, pitch_bytes); const size_t row_bytes = r.row_bytes, pitch = r.pitch_bytes, rows = r.rows;
    const char *src = ck.data(e);
    r.rows += pad_rows; r.built_bytes = r.rows * pitch;
    r.build = [src, rows, pad_rows, row_bytes, pitch] { std::vector<char> h((rows + pad_rows) * pitch, 0);
        for (size_t i = 0; i < rows; ++i) memcpy(h.data() + i * pitch, src + i * row_bytes, row_bytes); return h; };
    r.segments.clear();
    return r;
}
using Run = std::pair<size_t, size_t>;   // (first, count) of rows or elements taken from a tensor, in destination order
// The tensor's rows in an explicit order (the VAE attention's head-interleaved [q_h|k_h|v_h] rows as [Q|K|V]).
Recipe rows_permuted(const Checkpoint &ck, const Entry &e, const std::vector<Run> &runs, size_t pitch_bytes = 0) {
    Recipe r; r.row_bytes = e.row_bytes(); r.pitch_bytes = pitch_bytes ? pitch_bytes : r.row_bytes;
    for (const Run &run : runs) {
        if (run.first + run.second > e.rows()) throw std::runtime_error("a row run past the tensor");
        r.segments.push_back({ck.data(e) + run.first * r.row_bytes, run.second}); r.rows += run.second;
    }
    return r;
}
// A float vector's elements in an explicit order, as f32 (a bias permuted the same way as its rows).
Recipe widen_runs(const Checkpoint &ck, const Entry &e, const std::vector<Run> &runs) {
    const size_t eb = elem_bytes(e); if (!eb || e.dtype == "I8") throw std::runtime_error("widen_runs on " + e.dtype);
    size_t n = 0; for (const Run &run : runs) { if (run.first + run.second > e.elements()) throw std::runtime_error("an element run past the tensor"); n += run.second; }
    Recipe r; r.built_bytes = n * 4; const char *src = ck.data(e); const std::string dtype = e.dtype; const Entry copy = e;
    r.build = [src, runs, eb, n, copy] { std::vector<char> h(n * 4); float *o = (float *)h.data(); size_t k = 0;
        for (const Run &run : runs) for (size_t i = 0; i < run.second; ++i) o[k++] = widen_one(copy, src + (run.first + i) * eb); return h; };
    return r;
}
// Runs of `size` alternating two halves of one tensor, `first` first (the fused SwiGLU operand's 16-row groups).
std::vector<Run> interleaved_runs(size_t first, size_t second, size_t count, size_t size) {
    std::vector<Run> runs;
    for (size_t i = 0; i < count; i += size) { runs.push_back({first + i, size}); runs.push_back({second + i, size}); }
    return runs;
}
// Each row's `groups` groups of `in_elems` zero-extended to `out_elems` (the attention output projection's head
// dim 72 -> 128, or a plain K pad with one group).
Recipe regroup(const Checkpoint &ck, const Entry &e, size_t groups, size_t in_elems, size_t out_elems, size_t out_rows = 0) {
    const size_t eb = elem_bytes(e); if (!eb) throw std::runtime_error("regroup on " + e.dtype);
    if (e.row_bytes() != groups * in_elems * eb || out_elems < in_elems) throw std::runtime_error("regroup: the row does not hold the groups");
    Recipe r; const size_t rows = e.rows(); const char *src = ck.data(e);
    if (out_rows && out_rows < rows) throw std::runtime_error("regroup: fewer rows than the tensor");
    r.rows = out_rows ? out_rows : rows; r.row_bytes = r.pitch_bytes = groups * out_elems * eb; r.built_bytes = r.rows * r.pitch_bytes;
    const size_t in_bytes = in_elems * eb, out_bytes = out_elems * eb, row_in = groups * in_bytes, row_out = r.pitch_bytes, all = r.rows;
    r.build = [src, rows, all, groups, in_bytes, out_bytes, row_in, row_out] { std::vector<char> h(all * row_out, 0);
        for (size_t i = 0; i < rows; ++i) for (size_t g = 0; g < groups; ++g) memcpy(h.data() + i * row_out + g * out_bytes, src + i * row_in + g * in_bytes, in_bytes); return h; };
    return r;
}
// A float tensor as f32, element for element (F32 verbatim; F16 and BF16 widened, which is exact).
Recipe widen_f32(const Checkpoint &ck, std::initializer_list<const Entry *> parts) {
    Recipe r; size_t n = 0; std::vector<std::pair<const Entry *, const char *>> srcs;
    for (const Entry *e : parts) { if (!elem_bytes(*e) || e->dtype == "I8") throw std::runtime_error("widen_f32 on " + e->dtype); srcs.push_back({e, ck.data(*e)}); n += e->elements(); }
    r.built_bytes = n * 4;
    r.build = [srcs, n] { std::vector<char> h(n * 4); float *o = (float *)h.data(); size_t k = 0;
        for (auto &se : srcs) { const size_t eb = elem_bytes(*se.first); for (size_t i = 0; i < se.first->elements(); ++i) o[k++] = widen_one(*se.first, se.second + i * eb); } return h; };
    return r;
}
// A float vector as f32, zero-extended to `n_out` elements (a bias beside a zero-padded weight).
Recipe widen_padded(const Checkpoint &ck, const Entry &e, size_t n_out) {
    if (n_out < e.elements()) throw std::runtime_error("widen_padded: shorter than the tensor");
    const size_t eb = elem_bytes(e), n = e.elements(); const char *src = ck.data(e); const Entry copy = e;
    Recipe r; r.built_bytes = n_out * 4;
    r.build = [src, n, n_out, eb, copy] { std::vector<char> h(n_out * 4, 0); float *o = (float *)h.data();
        for (size_t i = 0; i < n; ++i) o[i] = widen_one(copy, src + i * eb); return h; };
    return r;
}
// The f32 [N,1] per-row scales of concatenated int8 operands, or of the two halves of an interleaved one, in the rows' order.
Recipe scales_rows(const Checkpoint &ck, std::initializer_list<const Entry *> parts) { return widen_f32(ck, parts); }
Recipe scales_interleave16(const Checkpoint &ck, const Entry &first, size_t first_row0, const Entry &second, size_t second_row0, size_t rows_each) {
    if (first.dtype != "F32" || second.dtype != "F32" || first.row_bytes() != 4 || second.row_bytes() != 4) throw std::runtime_error("scales must be f32 [N, 1]");
    Recipe r; r.built_bytes = 2 * rows_each * 4; const char *a = ck.data(first) + first_row0 * 4, *b = ck.data(second) + second_row0 * 4;
    r.build = [a, b, rows_each] { std::vector<char> h(2 * rows_each * 4);
        for (size_t i = 0; i < rows_each; i += 16) { memcpy(h.data() + (2 * i) * 4, a + i * 4, 64); memcpy(h.data() + (2 * i + 16) * 4, b + i * 4, 64); } return h; };
    return r;
}

// --- kernels: compile through loom-compile into the cache, load, launch ---------------------
struct Kernel {
    RtKernel *k = nullptr;
    void load(const std::string &path, const std::string &symbol) { k = rt().load(path, symbol); }
    ~Kernel() { if (k) rt().unload(k); }
};
using Cfg = std::vector<std::pair<std::string, std::string>>;

struct Compiler {
    std::string exe, sources, cache;
    static constexpr const char *BACKEND = "amdgpu-hal", *TARGET = "gfx1151";
    std::map<std::string, std::shared_ptr<Kernel>> loaded;
    std::string compiler_id_;   // computed once per Compiler
    // One kernel per (stem, source text, symbol, backend, target, config, compiler binary): the cache file name carries a hash
    // of all of them, so neither an edited kernel source nor a replaced loom-compile ever reuses a stale binary
    // (tools/kernel_cache.py keeps the same policy for the Python harnesses).
    static uint64_t fnv(const std::string &text, uint64_t h = 1469598103934665603ull) { for (unsigned char c : text) { h ^= c; h *= 1099511628211ull; } return h; }
    static std::string hex(uint64_t h, size_t digits) { char buf[17]; snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h); return std::string(buf, digits); }
    static std::string source_hash(const std::string &path) {
        std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("missing kernel source " + path);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return hex(fnv(text), 10);
    }
    // The compiler binary as posix_spawnp finds it (a bare name searches PATH), with its size and modification time.
    std::string compiler_id() {
        if (!compiler_id_.empty()) return compiler_id_;
        std::string path = exe;
        if (exe.find('/') == std::string::npos) {
            const char *env = getenv("PATH"); std::string dirs = env ? env : "";
            for (size_t b = 0; b <= dirs.size();) { const size_t e = dirs.find(':', b); const std::string d = dirs.substr(b, e == std::string::npos ? std::string::npos : e - b); const std::string cand = (d.empty() ? "." : d) + "/" + exe; if (access(cand.c_str(), X_OK) == 0) { path = cand; break; } if (e == std::string::npos) break; b = e + 1; }
        }
        struct stat st; if (stat(path.c_str(), &st) != 0) throw std::runtime_error("loom-compile not found: " + exe);
        compiler_id_ = path + ":" + std::to_string((long long)st.st_size) + ":" + std::to_string((long long)st.st_mtim.tv_sec) + "." + std::to_string((long long)st.st_mtim.tv_nsec);
        return compiler_id_;
    }
    std::shared_ptr<Kernel> get(const std::string &stem, const std::string &symbol, const Cfg &cfg) {
        std::string tag = stem + "__s" + source_hash(sources + "/" + stem + ".loom");
        std::string identity = std::string(BACKEND) + "\n" + TARGET + "\n" + symbol + "\n" + compiler_id() + "\n";
        for (auto &c : cfg) { tag += "__" + c.first.substr(c.first.rfind('.') + 1) + "_" + c.second; identity += c.first + "=" + c.second + "\n"; }
        tag += "__i" + hex(fnv(identity), 10);
        for (char &ch : tag) if (!isalnum((unsigned char)ch) && ch != '_' && ch != '-' && ch != '.') ch = '_';
        auto it = loaded.find(tag);
        if (it != loaded.end()) return it->second;
        const std::string path = cache + "/" + tag + ".hsaco";
        if (!exists(path)) compile(stem, symbol, cfg, path);
        auto k = std::make_shared<Kernel>(); k->load(path, symbol); loaded[tag] = k; return k;
    }
    // Compile into a unique temporary and rename it into place under a lock on <path>.lock, so several sessions asking for the
    // same kernel at once neither load a half-written binary nor clobber each other's temporary.
    void compile(const std::string &stem, const std::string &symbol, const Cfg &cfg, const std::string &path) {
        mkdirs(cache);
        const int lock = open((path + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
        if (lock < 0) throw std::runtime_error("cannot create " + path + ".lock");
        struct Unlock { int fd; ~Unlock() { flock(fd, LOCK_UN); close(fd); } } unlock{lock};
        if (flock(lock, LOCK_EX) != 0) throw std::runtime_error("cannot lock " + path + ".lock");
        if (exists(path)) return;   // another session published it while we waited
        std::random_device rd; const std::string tmp = path + ".tmp." + std::to_string(getpid()) + "." + hex((uint64_t(rd()) << 32) | rd(), 16);
        std::vector<std::string> args = {exe, sources + "/" + stem + ".loom", std::string("--backend=") + BACKEND, std::string("--target=") + TARGET, "--root=@" + symbol, "--output=" + tmp};
        for (auto &c : cfg) args.push_back("--config=" + c.first + "=" + c.second);
        std::vector<char *> argv; for (auto &a : args) argv.push_back(const_cast<char *>(a.c_str())); argv.push_back(nullptr);
        pid_t pid; if (posix_spawnp(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ) != 0) throw std::runtime_error("cannot spawn " + exe);
        int status = 0; waitpid(pid, &status, 0);
        struct stat st; const bool produced = stat(tmp.c_str(), &st) == 0 && st.st_size > 0;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || !produced) {
            unlink(tmp.c_str());
            std::string cmd; for (auto &a : args) cmd += a + " ";
            throw std::runtime_error("loom-compile failed for " + stem + ": " + cmd);
        }
        if (rename(tmp.c_str(), path.c_str()) != 0) { unlink(tmp.c_str()); throw std::runtime_error("cannot rename " + tmp); }
    }
};

struct Profile { bool on = false; std::map<std::string, double> us; };

void launch(Kernel &k, Profile *prof, const char *stage, unsigned gx, unsigned gy, unsigned bx, KernArgs &args) {
    std::chrono::steady_clock::time_point t0;
    if (prof && prof->on) { rt().sync(); t0 = std::chrono::steady_clock::now(); }
    static const bool trace = std::getenv("H3_TRACE") && *std::getenv("H3_TRACE");   // H3_TRACE=1: print and synchronize every launch (fault localisation)
    if (trace) fprintf(stderr, "launch %-12s grid %u x %u block %u args %d+%d\n", stage, gx, gy, bx, args.nscalars, args.nptrs);
    rt().launch(k.k, gx, gy, bx, args);
    if (trace) rt().sync();
    if (prof && prof->on) { rt().sync(); prof->us[stage] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); }
}

int lanes_for(int width) {
    for (int l : {320, 256, 160, 128, 96, 64, 32}) if (width % (8 * l) == 0 && (width / 4) % l == 0) return l;
    throw std::runtime_error("no prepare lane count for width " + std::to_string(width));
}
unsigned m_group_for(size_t tokens) {          // the raster group rule shared with scripts/build_kernels*.py
    const size_t tiles = (tokens + 255) / 256;
    if (tiles == 1) return 1;
    unsigned best = 4; size_t best_pad = (tiles + 3) / 4 * 4;
    for (unsigned g : {3u, 2u}) { const size_t pad = (tiles + g - 1) / g * g; if (pad < best_pad) { best = g; best_pad = pad; } }
    return best;
}
unsigned gemm_m_group_for(size_t tokens, int k, int n, int bits) {
    // The long H3 INT8 down projection benefits from a smaller row group.
    // Keep this shape-specific: the one-row group and a wider tile both lost.
    if (bits == 8 && k == FFN && n == HID && tokens >= 32768) return 2;
    return m_group_for(tokens);
}
// GEMM operand row pitch in k elements: K itself unless the row's byte pitch is a multiple of 1024, when the 384 rows a k step touches alias in
// the cache (int8 out projection K = 7168: 36.7 -> 41.2 TOPS, K = 21504 benchmark: 33.4 -> 41.4 with the pad); the pad is
// one k step (64 int8, 128 int4) so the kernels' step constraints hold. f16 GEMMs take no pitch.
size_t gemm_pitch(size_t k, int bits) {
    const bool aliases = bits == 8 ? k % 1024 == 0 : bits == 4 ? k % 2048 == 0 : false;   // the byte pitch (int4 rows are k/2 bytes) a multiple of 1024
    return aliases ? k + (bits == 4 ? 128 : 64) : k;
}
unsigned gemm_grid_y(size_t tokens, unsigned group) { return unsigned(((tokens + 255) / 256 + group - 1) / group * group); }

// The GEMM operand element types the stacks run: the checkpoint's int8 ConvRot rows (rotated activations, per-token
// scales), or its f16 / bf16 rows as stored (unrotated activations narrowed to the same type, no scales).
bool quantised(const std::string &elem) { return elem == "i8"; }
int elem_bits(const std::string &elem) { return quantised(elem) ? 8 : 16; }

// A prepare kernel (norm / lnorm / plain) for one width and operand type, ready to launch.
struct Prepare {
    std::shared_ptr<Kernel> k; int lanes = 0; std::string form, elem = "i8";
    void build(Compiler &c, const std::string &form_, const std::string &elem_, int width, float eps = 1e-5f, int classes = 1, int out_stride = 0) {
        form = form_; elem = elem_;
        std::string stem = "prepare_" + form + "_" + elem;
        if (form == "plain" && quantised(elem) && size_t(width) * 4 > 65536) stem = "prepare_plain16_i8";   // f16 LDS for rows past 64 KB of f32
        lanes = lanes_for(width);
        const std::string ns = "h3." + stem + ".";
        Cfg cfg = {{ns + "width", std::to_string(width)}, {ns + "lanes", std::to_string(lanes)}};
        if (form != "plain") { cfg.push_back({ns + "eps", num(eps)}); cfg.push_back({ns + "classes", std::to_string(classes)}); }
        cfg.push_back({ns + "out_stride", std::to_string(out_stride ? out_stride : width)});   // the operand rows' pitch = the GEMM's k_stride
        k = c.get(stem, "h3_" + stem, cfg);
    }
    // norm forms: (x f32, weight, table, cls) -> a_q[, a_s]; plain: (h f16) -> a_q[, a_s]
    void run(Profile *p, const char *stage, unsigned tokens, const void *x, const void *weight, const void *table, const void *cls, void *a_q, void *a_s) {
        KernArgs a; a.i32(int(tokens)).ptr(x);
        if (form != "plain") a.ptr(weight).ptr(table).ptr(cls);
        a.ptr(a_q); if (quantised(elem)) a.ptr(a_s);   // the float forms write rows, no token scale
        launch(*k, p, stage, tokens, 1, unsigned(lanes), a);
    }
};

// A GEMM of the int8 / f16 / bf16 family for one (K, N, m_group).
struct Gemm {
    std::shared_ptr<Kernel> k; int n = 0; bool resid = false, bias = false; std::string elem = "i8";
    unsigned m_group = 1;
    void build(Compiler &c, const std::string &mode, const std::string &elem_, bool bias_, bool gate_first, int k_size, int n_size, size_t tokens, int classes = 1, int k_stride = 0) {
        resid = mode == "resid"; bias = bias_; n = n_size; elem = elem_;
        m_group = gemm_m_group_for(tokens, k_size, n_size, elem_bits(elem));
        std::string stem = "gemm_" + elem + (mode == "plain" ? "" : "_" + mode) + "_256" + (bias ? "b" : "") + (mode == "swiglu" && !gate_first ? "_gs" : "");
        const std::string ns = "h3." + stem + ".";
        Cfg cfg = {{ns + "k_size", std::to_string(k_size)}, {ns + "n_size", std::to_string(n_size)}, {ns + "m_group", std::to_string(m_group)}};
        if (resid) cfg.push_back({ns + "classes", std::to_string(classes)});
        cfg.push_back({ns + "k_stride", std::to_string(k_stride ? k_stride : k_size)});   // operand row pitch (see gemm_pitch)
        k = c.get(stem, "h3_" + stem, cfg);
    }
    void run(Profile *p, const char *stage, unsigned tokens, const void *a_q, const void *w_q, const void *w_s, const void *a_s, void *out, const void *gate = nullptr, const void *cls = nullptr, const void *b = nullptr) {
        KernArgs a; a.i32(int(tokens)).ptr(a_q).ptr(w_q); if (quantised(elem)) a.ptr(w_s).ptr(a_s); a.ptr(out);   // float operands carry no scales
        if (resid) a.ptr(gate).ptr(cls);
        if (bias) a.ptr(b);
        launch(*k, p, stage, unsigned(n / 128), gemm_grid_y(tokens, m_group), THREADS, a);
    }
};

// --- the generic transformer stack -----------------------------------------------------------
struct StackDims {
    int hidden, heads, kv_heads, head_dim, ffn, rope_dim, classes, wbits; float eps; bool bias, gate_first, causal; bool attn_i4 = false; int attn_qk_bits = 16;   // attn_i4: QK^T in int4 (prepare_qk_i4 operands); attn_qk_bits 8: int8 operands (prepare_qk_i8, attention_i8qk_*), 4 = attn_i4
    bool bf16 = false;                                     // wbits 8: the checkpoint's int8 rows; 16: its f16 rows, or bf16 rows with bf16 set
    std::string elem() const { return wbits == 8 ? "i8" : bf16 ? "bf16" : "f16"; }
    int inner() const { return heads * head_dim; }
    int kv_inner() const { return kv_heads * head_dim; }
    int qkv() const { return inner() + 2 * kv_inner(); }
};
struct LayerCond { const float *table_msa, *gate_msa, *table_mlp, *gate_mlp; };

class Stack {
    DeviceBuffers memory_;
public:
    Stack(Compiler &c, const StackDims &d, size_t tokens, int layers, const Weights &w, const std::string &prefix_fmt, bool qk_weights, const float *ones_head, const std::string &tag = "stack")
        : d_(d), tokens_(tokens), layers_(layers), tag_(tag) {
        const std::string elem = d.elem(); const int bits = d.wbits; const bool quant = quantised(elem);
        auto wbytes = [&](size_t n, size_t k) { return quant ? n * k : n * k * 2; };
        auto scale = [&](const std::string &name, size_t n) -> char * { return quant ? w.at(name, n * 4) : nullptr; };
        auto pitch = [&](size_t k) { return gemm_pitch(k, bits); };
        auto wpad = [&](const std::string &name, size_t n, size_t k) { return w.rows(name, n, wbytes(1, k), wbytes(1, pitch(k))); };
        waves_ = d.causal ? 8 : (tokens >= 4096 ? 8 : 4);
        capacity_ = std::max<size_t>((tokens + 16 + 31) / 32 * 32, (tokens + 16 * waves_ - 1) / (16 * waves_) * (16 * waves_));
        capacity_ = std::max<size_t>(capacity_, (tokens + 255) / 256 * 256);
        for (int i = 0; i < layers; ++i) {
            const std::string p = fmt(prefix_fmt.c_str(), i);
            Block b;
            b.qkv_q = wpad(p + "qkv.q", d.qkv(), d.hidden);   b.qkv_s = scale(p + "qkv.s", d.qkv());
            b.out_q = wpad(p + "out.q", d.hidden, d.inner()); b.out_s = scale(p + "out.s", d.hidden);
            b.gu_q = wpad(p + "gu.q", 2 * d.ffn, d.hidden);   b.gu_s = scale(p + "gu.s", 2 * d.ffn);
            b.down_q = wpad(p + "down.q", d.hidden, d.ffn);   b.down_s = scale(p + "down.s", d.hidden);
            if (d.bias) { b.qkv_b = w.at(p + "qkv.b", size_t(d.qkv()) * 4); b.out_b = w.at(p + "out.b", size_t(d.hidden) * 4); b.gu_b = w.at(p + "gu.b", size_t(2 * d.ffn) * 4); b.down_b = w.at(p + "down.b", size_t(d.hidden) * 4); }
            b.norm1 = w.at(p + "norm1", size_t(d.hidden) * 4); b.norm2 = w.at(p + "norm2", size_t(d.hidden) * 4);
            if (qk_weights) { b.qnorm = w.at(p + "qnorm", size_t(d.head_dim) * 4); b.knorm = w.at(p + "knorm", size_t(d.head_dim) * 4); }
            else b.qnorm = b.knorm = (char *)ones_head;
            if (w.has(p + "scale1")) { b.scale1 = w.at(p + "scale1", size_t(d.hidden) * 4); b.scale2 = w.at(p + "scale2", size_t(d.hidden) * 4); }
            blocks_.push_back(b);
        }
        prep_norm_.build(c, "norm", elem, d.hidden, d.eps, d.classes, int(pitch(d.hidden)));
        direct_ = elem == "f16";   // f16 rows: the f16 attention output and gate|up product are the out / down operands as they are
        if (!direct_) { prep_attn_.build(c, "plain", elem, d.inner(), 1e-5f, 1, int(pitch(d.inner()))); prep_down_.build(c, "plain", elem, d.ffn, 1e-5f, 1, int(pitch(d.ffn))); }
        gemm_qkv_.build(c, "plain", elem, d.bias, true, d.hidden, d.qkv(), tokens, 1, int(pitch(d.hidden)));
        gemm_gu_.build(c, "swiglu", elem, d.bias, d.gate_first, d.hidden, 2 * d.ffn, tokens, 1, int(pitch(d.hidden)));
        gemm_out_.build(c, "resid", elem, d.bias, true, d.inner(), d.hidden, tokens, d.classes, int(pitch(d.inner())));
        gemm_down_.build(c, "resid", elem, d.bias, true, d.ffn, d.hidden, tokens, d.classes, int(pitch(d.ffn)));
        {
            const std::string stem = d.head_dim == 64 ? "rope64_qknorm_f16" : (d.rope_dim == 128 ? "rope128_qknorm_f16" : "rope_qknorm_f16");
            const std::string ns = "h3." + stem + ".";
            rope_ = c.get(stem, "h3_" + stem, {{ns + "row_stride", std::to_string(d.qkv())}, {ns + "heads", std::to_string(d.heads)}, {ns + "kv_heads", std::to_string(d.kv_heads)}, {ns + "k_offset", std::to_string(d.inner())}, {ns + "eps", num(d.eps)}});
        }
        const bool qk_int = d.attn_i4 || d.attn_qk_bits == 8;
        const bool qk_head_major = !d.attn_i4 && d.attn_qk_bits == 8 && waves_ == 8;
        const std::string pqk = d.attn_i4 ? "prepare_qk_i4" : (qk_head_major ? "prepare_qk_i8hm" : "prepare_qk_i8");
        if (qk_int) {
            if (d.causal || d.head_dim != 128) throw std::runtime_error("integer QK^T attention: MHA with head 128 only");
            const std::string pq = "h3." + pqk + ".";
            colmean_ = c.get("colmean_f32", "h3_colmean_f32", {{"h3.colmean_f32.width", std::to_string(d.inner())}});
            Cfg prep_cfg{{pq + "row_stride", std::to_string(d.inner())}, {pq + "head_offset", "0"}, {pq + "heads", std::to_string(d.heads)}};
            if (qk_head_major) prep_cfg.emplace_back(pq + "token_capacity", std::to_string(capacity_));
            prep_cfg.emplace_back(pq + "extra_scale", num(1.0 / std::sqrt(double(d.head_dim)) / 128.0));
            prep_q_ = c.get(pqk, "h3_" + pqk, prep_cfg);
            prep_cfg.back().second = "1";
            prep_k_ = c.get(pqk, "h3_" + pqk, prep_cfg);
            transpose_ = c.get("transpose_f16", "h3_transpose_f16", {{"h3.transpose_f16.width", std::to_string(d.inner())}, {"h3.transpose_f16.row_capacity", std::to_string(capacity_)}});
        }
        {
            std::string stem = d.causal ? "attention_gqa8c_lds_f16_wmma" : (d.head_dim == 64 ? (waves_ == 8 ? "attention_mha648_lds_f16_wmma" : "attention_mha64_lds_f16_wmma") : (waves_ == 8 ? "attention_mha8_lds_f16_wmma" : "attention_mha_lds_f16_wmma"));
            if (qk_int) {
                if (qk_head_major) {
                    // Head-major operands; 64 shared keys amortize softmax and loop overhead.
                    stem = "attention_i8qkhm_mha8_k64_lds_f16_wmma";
                } else if (d.attn_i4) {
                    stem = tokens >= 20000 ? "attention_i4qkl_mha8_lds_f16_wmma" :
                           (waves_ == 8 ? "attention_i4qk_mha8_lds_f16_wmma" : "attention_i4qk_mha_lds_f16_wmma");
                } else {
                    stem = "attention_i8qk_mha_lds_f16_wmma";
                }
            }
            // tile skip: H3_ATTN_SKIP_TAU=<tau> selects the skip twin (i4qk -> i4qks) at that tau (tau 4: no measured velocity cost,
            // about 3% on the 768 step against the carried-scale plain kernel); unset or 'off' -> the plain kernel
            double tau = 0.0;
            if (const char *v = std::getenv("H3_ATTN_SKIP_TAU")) { std::string sv(v); for (auto &ch : sv) ch = char(tolower(ch)); tau = (sv == "off" || sv == "none" || sv == "0" || sv == "1e30" || sv.empty()) ? 0.0 : std::atof(v); }
            const bool skip = d.attn_i4 && tau > 0.0;   // the skip twins exist for int4 only
            if (skip) { const size_t at = stem.find("i4qk"); stem.replace(at, 4, "i4qks"); }
            const std::string ns = "h3." + stem + ".";
            Cfg acfg = {{ns + "q_stride", std::to_string(d.inner())}, {ns + "kv_stride", std::to_string(d.kv_inner())}, {ns + "tokens", std::to_string(tokens)}, {ns + "token_capacity", std::to_string(capacity_)}, {ns + "scale", num(1.0 / std::sqrt(double(d.head_dim)))}, {ns + "out_stride", std::to_string(d.inner())}};
            if (skip) acfg.push_back({ns + "skip_tau", num(tau)});
            attention_ = c.get(stem, "h3_" + stem, acfg);
        }
        const size_t T = capacity_;
        a_q_ = memory_.alloc(T * std::max(pitch(d.ffn), std::max(pitch(d.hidden), pitch(d.inner()))) * (quant ? 1 : 2));
        a_s_ = memory_.alloc(T * 4);
        fused_ = memory_.alloc(T * size_t(d.qkv()) * 2);
        q_ = memory_.alloc(T * size_t(d.inner()) * 2);
        k_ = memory_.alloc(T * size_t(d.kv_inner()) * 2);
        v_ = memory_.alloc(T * size_t(d.kv_inner()) * 2);
        attn_ = memory_.alloc(T * size_t(d.inner()) * 2);
        gu_ = memory_.alloc(T * size_t(d.ffn) * 2);
        for (auto p : {fused_, q_, k_, v_, attn_}) rt().memset(p, 0, T * size_t(p == fused_ ? d.qkv() : (p == q_ || p == attn_ ? d.inner() : d.kv_inner())) * 2);
        if (d.attn_i4 || d.attn_qk_bits == 8) {
            const size_t code_bytes = d.attn_i4 ? 64 : 128; qi_ = memory_.alloc(T * size_t(d.heads) * code_bytes); ki_ = memory_.alloc(T * size_t(d.heads) * code_bytes); qs_ = memory_.alloc(T * size_t(d.heads) * 4); ks_ = memory_.alloc(T * size_t(d.heads) * 4);
            kmean_ = memory_.alloc(size_t(d.inner()) * 4); zmean_ = memory_.alloc(size_t(d.inner()) * 4); rt().memset(zmean_, 0, size_t(d.inner()) * 4);
            vt_ = memory_.alloc(size_t(d.inner()) * T * 2); rt().memset(vt_, 0, size_t(d.inner()) * T * 2);
            for (auto p : {qi_, ki_}) rt().memset(p, 0, T * size_t(d.heads) * code_bytes);
            for (auto p : {qs_, ks_}) rt().memset(p, 0, T * size_t(d.heads) * 4);
        }
    }

    size_t capacity() const { return capacity_; }
    size_t tokens() const { return tokens_; }
    const std::vector<struct Block_> *dummy = nullptr;

    struct Block { char *qkv_q, *qkv_s, *out_q, *out_s, *gu_q, *gu_s, *down_q, *down_s, *qkv_b = nullptr, *out_b = nullptr, *gu_b = nullptr, *down_b = nullptr, *norm1, *norm2, *qnorm, *knorm, *scale1 = nullptr, *scale2 = nullptr; };
    const Block &block(int i) const { return blocks_[i]; }

    // x: f32 [capacity][hidden] (rows past tokens untouched); cls: i32 [tokens]; cos/sin: f32 [tokens][rope_dim/2]
    void forward(Profile *prof, void *x, const void *cls, const void *cos, const void *sin, const std::function<LayerCond(int)> &cond, int first = 0, int last = -1) {
        const unsigned T = unsigned(tokens_);
        if (last < 0) last = layers_;
        // H3_DUMP_BLOCKS=<dir>: this stack's x before the first block (<tag>_h_in.f32) and after every block (<tag>_blk_NN.f32), [tokens][hidden]
        // f32, on its H3_DUMP_CALL-th forward (default 0; the DiT's first step, the decoder's first clip)
        static const char *dump_blocks = std::getenv("H3_DUMP_BLOCKS"); static const int dump_call = std::getenv("H3_DUMP_CALL") ? atoi(std::getenv("H3_DUMP_CALL")) : 0;
        const bool dumping = dump_blocks && calls_++ == dump_call && first == 0;
        auto dump_x = [&](const std::string &name) { if (!dumping) return; std::vector<float> hbuf(size_t(T) * d_.hidden); rt().sync(); rt().d2h(hbuf.data(), x, hbuf.size() * 4);
            if (FILE *f = fopen((std::string(dump_blocks) + "/" + tag_ + "_" + name + ".f32").c_str(), "wb")) { fwrite(hbuf.data(), 4, hbuf.size(), f); fclose(f); } };
        dump_x("h_in");
        for (int i = first; i < last; ++i) {
            const Block &b = blocks_[i]; const LayerCond lc = cond(i);
            prep_norm_.run(prof, "prepare norm", T, x, b.norm1, lc.table_msa, cls, a_q_, a_s_);
            gemm_qkv_.run(prof, "gemm qkv", T, a_q_, b.qkv_q, b.qkv_s, a_s_, fused_, nullptr, nullptr, b.qkv_b);
            { KernArgs a; a.i32(int(T)).ptr(fused_).ptr(b.qnorm).ptr(b.knorm).ptr(cos).ptr(sin).ptr(q_).ptr(k_).ptr(v_); launch(*rope_, prof, "qk norm + rope", T, 1, THREADS, a); }
            if (d_.attn_i4 || d_.attn_qk_bits == 8) {
                static const bool smooth = std::getenv("H3_KSMOOTH") && std::string(std::getenv("H3_KSMOOTH")) == "1";   // K mean smoothing: off by default (measured worse)
                if (smooth) { KernArgs a; a.i32(int(T)).ptr(k_).ptr(kmean_); launch(*colmean_, prof, "attention operands", unsigned(d_.inner() / 256), 1, THREADS, a); }
                { KernArgs a; a.i32(int(T)).ptr(q_).ptr(zmean_).ptr(qi_).ptr(qs_); launch(*prep_q_, prof, "attention operands", T, 1, THREADS, a); }
                { KernArgs a; a.i32(int(T)).ptr(k_).ptr(smooth ? kmean_ : zmean_).ptr(ki_).ptr(ks_); launch(*prep_k_, prof, "attention operands", T, 1, THREADS, a); }
                { KernArgs a; a.i32(int(T)).ptr(v_).ptr(vt_); launch(*transpose_, prof, "attention operands", (T + 31) / 32, unsigned(d_.inner() / 32), THREADS, a); }
                KernArgs a; a.i32(int(T)).ptr(qi_).ptr(qs_).ptr(ki_).ptr(ks_).ptr(vt_).ptr(attn_);
                const unsigned qb = 16 * unsigned(waves_); launch(*attention_, prof, "attention", (T + qb - 1) / qb, unsigned(d_.heads), 32 * unsigned(waves_), a);
            } else {
              KernArgs a; a.i32(int(T)).ptr(q_).ptr(k_).ptr(v_).ptr(attn_);
              if (d_.causal) launch(*attention_, prof, "attention", (T + 15) / 16, unsigned(d_.kv_heads), THREADS, a);
              else { const unsigned qb = 16 * unsigned(waves_); launch(*attention_, prof, "attention", (T + qb - 1) / qb, unsigned(d_.heads), 32 * unsigned(waves_), a); } }
            if (!direct_) prep_attn_.run(prof, "prepare out input", T, attn_, nullptr, nullptr, nullptr, a_q_, a_s_);
            gemm_out_.run(prof, "gemm out + residual", T, direct_ ? attn_ : a_q_, b.out_q, b.out_s, a_s_, x, lc.gate_msa, cls, b.out_b);
            prep_norm_.run(prof, "prepare norm", T, x, b.norm2, lc.table_mlp, cls, a_q_, a_s_);
            gemm_gu_.run(prof, "gemm ff + swiglu", T, a_q_, b.gu_q, b.gu_s, a_s_, gu_, nullptr, nullptr, b.gu_b);
            if (!direct_) prep_down_.run(prof, "prepare down input", T, gu_, nullptr, nullptr, nullptr, a_q_, a_s_);
            gemm_down_.run(prof, "gemm down + residual", T, direct_ ? gu_ : a_q_, b.down_q, b.down_s, a_s_, x, lc.gate_mlp, cls, b.down_b);
            dump_x(fmt("blk_%02d", i));
        }
    }

private:
    StackDims d_; size_t tokens_, capacity_ = 0; int layers_, waves_ = 4; std::string tag_; int calls_ = 0; bool direct_ = false;
    std::vector<Block> blocks_;
    Prepare prep_norm_, prep_attn_, prep_down_;
    Gemm gemm_qkv_, gemm_gu_, gemm_out_, gemm_down_;
    std::shared_ptr<Kernel> rope_, attention_, colmean_, prep_q_, prep_k_, transpose_;
    void *a_q_ = nullptr, *a_s_ = nullptr, *fused_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *attn_ = nullptr, *gu_ = nullptr,
         *qi_ = nullptr, *ki_ = nullptr, *qs_ = nullptr, *ks_ = nullptr, *kmean_ = nullptr, *zmean_ = nullptr, *vt_ = nullptr;
};

// --- the packed sequence layout (reference/h3_ref.py Layout) --------------------------------
struct RefSeg { int kind; size_t row0, rows; int latent_t, lat_h, lat_w, audio_t; bool audio; int ref; };   // one packed segment of a reference block

struct Layout {
    // The packed sequence [text | reference blocks | audio | video] as ComfyUI's PackedLayout: positions (t, h, w), the
    // AdaLN row (timestep class * 3 + modality tag) and the timestep class per row. Classes: 0 video, 1 audio,
    // 2 cond video (t = max(t_v, 0.999)), 3 cond audio (t = max(t_a, 1.0)). Tags: 0 video, 1 text, 2 audio.
    int text_len, latent_t, lat_h, lat_w, audio_t; size_t ref_rows = 0, audio_rows, video_rows, seq_len;
    std::vector<double> pos;                 // [seq][3] (t, h, w)
    std::vector<int32_t> adaln_rows, tclass;
    std::vector<RefSeg> ref_segs;            // the reference segments in row order (a video block with sound gives two)
    void mark_vision(size_t start, size_t count) {
        if (start > size_t(text_len) || count > size_t(text_len) - start) throw std::invalid_argument("vision span outside presentation");
        const size_t begin = start ? start - 1 : 0, end = std::min(size_t(text_len), start + count + 1);
        std::fill(adaln_rows.begin() + begin, adaln_rows.begin() + end, 0);   // embeddings and flanking vision tokens
    }
    static std::vector<double> axis(int dim, double sqrt_area) {
        const double ratio = dim / sqrt_area; const int n = dim / 2; std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = (i * (ratio / n) + (1.0 - ratio) / 2.0) * SPATIAL_SCALE;
        return v;
    }
    static double video_span(int n) { double s = 0; for (int k = 0; k < n; ++k) s += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5]; return s; }
    Layout(int text_len_, int latent_t_, int lat_h_, int lat_w_, int audio_t_, const std::vector<h3pipe_ref> &refs = {}, const std::vector<h3pipe_keyframe> &kfs = {}) : text_len(text_len_), latent_t(latent_t_), lat_h(lat_h_), lat_w(lat_w_), audio_t(audio_t_) {
        const double area = std::sqrt(double(lat_h) * lat_w);
        const std::vector<double> ah = axis(lat_h, area), aw = axis(lat_w, area);
        audio_rows = size_t(audio_t) * 2; video_rows = size_t(latent_t) * ah.size() * aw.size();
        for (const h3pipe_ref &rf : refs) {
            if (rf.kind == 1) ref_rows += size_t(rf.audio_t) * 2;
            else { const int hh = rf.lat_h / 2, ww = rf.lat_w / 2; ref_rows += size_t(rf.kind == 0 ? 1 : rf.latent_t) * hh * ww; if (rf.kind == 2 && rf.audio_latent && rf.audio_t > 0) ref_rows += size_t(rf.audio_t) * 2; }
        }
        for (const h3pipe_keyframe &kf : kfs) { ref_rows += ah.size() * aw.size(); if (kf.audio_latent && kf.audio_t > 0) ref_rows += size_t(kf.audio_t) * 2; }
        seq_len = text_len + ref_rows + audio_rows + video_rows;
        pos.assign(seq_len * 3, 0.0); adaln_rows.assign(seq_len, 0); tclass.assign(seq_len, 0);
        size_t r = 0;
        for (int i = 0; i < text_len; ++i, ++r) { pos[3 * r] = i; adaln_rows[r] = 1; tclass[r] = 0; }
        double cursor = text_len;
        auto audio_grid = [&](double origin, int t, double w_low, double w_high, int cls) {
            for (int c = 0; c < 2; ++c) for (int k = 0; k < t; ++k, ++r) { pos[3 * r] = origin + k; pos[3 * r + 2] = c == 0 ? w_low : w_high; adaln_rows[r] = cls * MODALITIES + 2; tclass[r] = cls; } };
        auto video_grid = [&](double origin, int vt, const std::vector<double> &gh, const std::vector<double> &gw, int cls) {
            double acc = origin;
            for (int k = 0; k < vt; ++k) { for (double hv : gh) for (double wv : gw) { pos[3 * r] = acc; pos[3 * r + 1] = hv; pos[3 * r + 2] = wv; adaln_rows[r] = cls * MODALITIES + 0; tclass[r] = cls; ++r; } acc += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5]; } };
        // keyframes: cond rows on the target grid at cond_t = cursor + FRAME_RESCALE * frame_index, where cursor already skips the references
        { double after_refs = cursor;
          for (const h3pipe_ref &rf : refs) after_refs += rf.kind == 0 ? 1.0 : (rf.kind == 1 ? double(rf.audio_t) : std::max(rf.audio_latent && rf.audio_t > 0 ? double(rf.audio_t) : 0.0, video_span(rf.latent_t)));
          for (size_t ki = 0; ki < kfs.size(); ++ki) {
              const h3pipe_keyframe &kf = kfs[ki]; const double cond_t = after_refs + FRAME_RESCALE * kf.frame_index;
              ref_segs.push_back({3, r, ah.size() * aw.size(), 1, lat_h, lat_w, 0, false, int(ki)});
              for (double hv : ah) for (double wv : aw) { pos[3 * r] = cond_t; pos[3 * r + 1] = hv; pos[3 * r + 2] = wv; adaln_rows[r] = 2 * MODALITIES + 0; tclass[r] = 2; ++r; }
              if (kf.audio_latent && kf.audio_t > 0) { ref_segs.push_back({3, r, size_t(kf.audio_t) * 2, 0, 0, 0, kf.audio_t, true, int(ki)}); audio_grid(cond_t, kf.audio_t, aw.front(), aw.back(), 3); }
          } }
        for (size_t ri = 0; ri < refs.size(); ++ri) {
            const h3pipe_ref &rf = refs[ri];
            if (rf.kind == 0 || rf.kind == 2) {
                const double rarea = std::sqrt(double(rf.lat_h) * rf.lat_w); const std::vector<double> rh = axis(rf.lat_h, rarea), rw = axis(rf.lat_w, rarea);
                const int vt = rf.kind == 0 ? 1 : rf.latent_t; const bool sound = rf.kind == 2 && rf.audio_latent && rf.audio_t > 0;
                if (sound) { ref_segs.push_back({rf.kind, r, size_t(rf.audio_t) * 2, 0, 0, 0, rf.audio_t, true, int(ri)}); audio_grid(cursor, rf.audio_t, rw.front(), rw.back(), 3); }
                ref_segs.push_back({rf.kind, r, size_t(vt) * rh.size() * rw.size(), vt, rf.lat_h, rf.lat_w, 0, false, int(ri)});
                video_grid(cursor, vt, rh, rw, 2);
                cursor += rf.kind == 0 ? 1.0 : std::max(sound ? double(rf.audio_t) : 0.0, video_span(vt));
            } else {
                if (rf.audio_t > 0) { ref_segs.push_back({1, r, size_t(rf.audio_t) * 2, 0, 0, 0, rf.audio_t, true, int(ri)}); audio_grid(cursor, rf.audio_t, aw.front(), aw.back(), 3); }
                cursor += rf.audio_t;
            }
        }
        audio_grid(cursor, audio_t, aw.front(), aw.back(), 1);
        video_grid(cursor, latent_t, ah, aw, 0);
        if (r != seq_len) throw std::runtime_error("layout row count mismatch");
    }
};

int align_frames(int n) { while (n % 17 != 5) ++n; return n; }
constexpr int MAX_SIDE = 8192, MAX_FRAMES = 1 << 20;   // canvas sides (multiples of 32) and frame counts h3pipe_shape_for accepts: every derived count fits an int
bool valid_canvas(int height, int width) { return height >= 32 && width >= 32 && height <= MAX_SIDE && width <= MAX_SIDE && height % 32 == 0 && width % 32 == 0; }
h3pipe_shape shape_or_throw(const h3pipe_params &p) {
    h3pipe_shape sh; if (h3pipe_shape_for(&p, &sh)) throw std::invalid_argument("height and width must be multiples of 32 up to 8192, frames 1..1048576");
    return sh;
}

struct Schedule {                            // diffusers' MiniMaxH3Scheduler
    std::vector<float> sigmas, timesteps;
    Schedule(int steps, double shift) {
        std::vector<float> s;
        for (int i = 0; i < steps; ++i) { const double base = steps == 1 ? 1.0 : 1.0 - double(i) / (steps - 1); s.push_back(float(shift * base / (1.0 + (shift - 1.0) * base))); }
        for (float v : s) if (sigmas.empty() || v != sigmas.back()) sigmas.push_back(v);
        for (size_t i = 0; i + 1 < sigmas.size(); ++i) timesteps.push_back(1.0f - sigmas[i]);
    }
};

// --- ComfyUI's H3 checkpoint (minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors) as the DiT stack, the token
// refiner, the embedders, the final layer and the conditioning tables read: the 50 blocks' int8 ConvRot rows with their
// scales, everything else in its stored float type (bf16 refiner and condition_proj, f32 patch projections and heads, f16
// AdaLN projections widened for the CPU), laid out for the kernels but never converted.
void dit_plan(const Checkpoint &ck, Weights &w) {
    auto I8 = [&](const std::string &n, int64_t rows, int64_t cols) -> const Entry & { return ck.at(n + ".weight", "I8", {rows, cols}); };
    auto S = [&](const std::string &n, int64_t rows) -> const Entry & { return ck.at(n + ".weight_scale", "F32", {rows, 1}); };
    auto vec = [&](const std::string &n, const char *dtype, int64_t len) -> const Entry & { return ck.at(n, dtype, {len}); };
    const int inner = HEADS * HEAD_DIM, qkv = 3 * inner;
    for (int i = 0; i < 50; ++i) {
        const std::string p = fmt("blocks.%d.", i), q = p + "attn.qkv_proj", o = p + "attn.out_proj", f1 = p + "mlp.fc1", f2 = p + "mlp.fc2";
        w.add(p + "qkv.q", rows_of(ck, {&I8(q, qkv, HID)}, gemm_pitch(HID, 8)));      w.add(p + "qkv.s", scales_rows(ck, {&S(q, qkv)}));
        w.add(p + "out.q", rows_of(ck, {&I8(o, HID, inner)}, gemm_pitch(inner, 8)));  w.add(p + "out.s", scales_rows(ck, {&S(o, HID)}));
        const Entry &gu = I8(f1, 2 * FFN, HID), &gus = S(f1, 2 * FFN);               // gate rows first, then up: interleaved in 16-row runs for the fused SwiGLU
        w.add(p + "gu.q", interleave16(ck, gu, 0, gu, FFN, FFN, gemm_pitch(HID, 8))); w.add(p + "gu.s", scales_interleave16(ck, gus, 0, gus, FFN, FFN));
        w.add(p + "down.q", rows_of(ck, {&I8(f2, HID, FFN)}, gemm_pitch(FFN, 8)));   w.add(p + "down.s", scales_rows(ck, {&S(f2, HID)}));
        w.add(p + "norm1", widen_f32(ck, {&vec(p + "norm1.weight", "BF16", HID)}));   w.add(p + "norm2", widen_f32(ck, {&vec(p + "norm2.weight", "BF16", HID)}));
        w.add(p + "qnorm", widen_f32(ck, {&vec(p + "attn.q_norm.weight", "BF16", HEAD_DIM)})); w.add(p + "knorm", widen_f32(ck, {&vec(p + "attn.k_norm.weight", "BF16", HEAD_DIM)}));
        w.add(fmt("h3.blocks.%d.adaln.w", i), widen_f32(ck, {&ck.at(p + "adaln_proj.linear.weight", "F16", {3 * 6 * HID, 8})}));
        w.add(fmt("h3.blocks.%d.adaln.b", i), widen_f32(ck, {&vec(p + "adaln_proj.linear.bias", "F16", 3 * 6 * HID)}));
    }
    w.add("h3.adaln_t_table", widen_f32(ck, {&ck.at("adaln_t_table", "F32", {1025, 8})}));
    w.add("h3.rope_inv_freq", widen_f32(ck, {&vec("rope.inv_freq", "F32", 16)}));
    w.add("h3.final.adaln.w", widen_f32(ck, {&ck.at("final_layer.adaln_proj.linear.weight", "F16", {2 * HID, 8})}));
    w.add("h3.final.adaln.b", widen_f32(ck, {&vec("final_layer.adaln_proj.linear.bias", "F16", 2 * HID)}));
    w.add("h3.final.norm", widen_f32(ck, {&vec("final_layer.norm.weight", "BF16", HID)}));
    // the two heads stacked to N = 128 (video 96 | audio 32) for one f32 matmul
    w.add("h3.final.out.w", rows_of(ck, {&ck.at("final_layer.video_out.weight", "F32", {VIDEO_PATCH, HID}), &ck.at("final_layer.audio_out.weight", "F32", {AUDIO_CH, HID})}));
    w.add("h3.final.out.b", widen_f32(ck, {&vec("final_layer.video_out.bias", "F32", VIDEO_PATCH), &vec("final_layer.audio_out.bias", "F32", AUDIO_CH)}));
    w.add("h3.cond.w", rows_of(ck, {&ck.at("condition_proj.weight", "BF16", {HID, TEXT_DIM})}));          w.add("h3.cond.b", widen_f32(ck, {&vec("condition_proj.bias", "BF16", HID)}));
    w.add("h3.video_in.w", rows_of(ck, {&ck.at("video_patch_proj.weight", "F32", {HID, VIDEO_PATCH})})); w.add("h3.video_in.b", widen_f32(ck, {&vec("video_patch_proj.bias", "F32", HID)}));
    w.add("h3.audio_in.w", rows_of(ck, {&ck.at("audio_patch_proj.weight", "F32", {HID, AUDIO_CH})}));    w.add("h3.audio_in.b", widen_f32(ck, {&vec("audio_patch_proj.bias", "F32", HID)}));
    for (int j = 0; j < 2; ++j) {   // the token refiner: bf16 rows as stored (unrotated), on the bf16 GEMMs
        const std::string p = fmt("h3.refiner.%d.", j), src = fmt("token_refiner.blocks.%d.", j);
        w.add(p + "qkv.q", rows_of(ck, {&ck.at(src + "attn.qkv_proj.weight", "BF16", {qkv, HID})}, gemm_pitch(HID, 16) * 2));
        w.add(p + "out.q", rows_of(ck, {&ck.at(src + "attn.out_proj.weight", "BF16", {HID, inner})}, gemm_pitch(inner, 16) * 2));
        const Entry &gu = ck.at(src + "mlp.fc1.weight", "BF16", {2 * FFN, HID});
        w.add(p + "gu.q", interleave16(ck, gu, 0, gu, FFN, FFN, gemm_pitch(HID, 16) * 2));
        w.add(p + "down.q", rows_of(ck, {&ck.at(src + "mlp.fc2.weight", "BF16", {HID, FFN})}, gemm_pitch(FFN, 16) * 2));
        w.add(p + "norm1", widen_f32(ck, {&vec(src + "norm1.weight", "BF16", HID)}));  w.add(p + "norm2", widen_f32(ck, {&vec(src + "norm2.weight", "BF16", HID)}));
        w.add(p + "qnorm", widen_f32(ck, {&vec(src + "attn.q_norm.weight", "BF16", HEAD_DIM)})); w.add(p + "knorm", widen_f32(ck, {&vec(src + "attn.k_norm.weight", "BF16", HEAD_DIM)}));
    }
    w.add("h3.refiner.final_norm", widen_f32(ck, {&vec("token_refiner.final_norm.weight", "BF16", HID)}));
}

// ComfyUI's Qwen3-VL-32B text encoder (qwen3vl_32b_minimax_h3_int8_convrot.safetensors): the 50 layers' int8 ConvRot
// rows with their scales, the bf16 vision tower as stored (zero-padded where a kernel's K or N demands it), the norms
// widened to f32. The embedding table is not a recipe: text_in reads one row per token straight from the mapping.
void te_plan(const Checkpoint &ck, Weights &w) {
    auto I8 = [&](const std::string &n, int64_t rows, int64_t cols) -> const Entry & { return ck.at(n + ".weight", "I8", {rows, cols}); };
    auto S = [&](const std::string &n, int64_t rows) -> const Entry & { return ck.at(n + ".weight_scale", "F32", {rows, 1}); };
    auto bf = [&](const std::string &n, std::initializer_list<int64_t> shape) -> const Entry & { return ck.at(n, "BF16", shape); };
    const int inner = TE_HEADS * HEAD_DIM, kv = TE_KV * HEAD_DIM, pitch = int(gemm_pitch(TE_HID, 8));
    for (int i = 0; i < 50; ++i) {
        const std::string p = fmt("blocks.%d.", i), l = fmt("model.layers.%d.", i);
        const std::string q = l + "self_attn.q_proj", k = l + "self_attn.k_proj", v = l + "self_attn.v_proj", o = l + "self_attn.o_proj";
        w.add(p + "qkv.q", rows_of(ck, {&I8(q, inner, TE_HID), &I8(k, kv, TE_HID), &I8(v, kv, TE_HID)}, pitch));
        w.add(p + "qkv.s", scales_rows(ck, {&S(q, inner), &S(k, kv), &S(v, kv)}));
        w.add(p + "out.q", rows_of(ck, {&I8(o, TE_HID, inner)}, gemm_pitch(inner, 8)));   w.add(p + "out.s", scales_rows(ck, {&S(o, TE_HID)}));
        const std::string g = l + "mlp.gate_proj", u = l + "mlp.up_proj", d = l + "mlp.down_proj";
        const Entry &ge = I8(g, TE_FFN, TE_HID), &ue = I8(u, TE_FFN, TE_HID);            // gate and up are separate tensors: interleaved in 16-row runs
        w.add(p + "gu.q", interleave16(ck, ge, 0, ue, 0, TE_FFN, pitch));                 w.add(p + "gu.s", scales_interleave16(ck, S(g, TE_FFN), 0, S(u, TE_FFN), 0, TE_FFN));
        w.add(p + "down.q", rows_of(ck, {&I8(d, TE_HID, TE_FFN)}, gemm_pitch(TE_FFN, 8))); w.add(p + "down.s", scales_rows(ck, {&S(d, TE_HID)}));
        w.add(p + "norm1", widen_f32(ck, {&bf(l + "input_layernorm.weight", {TE_HID})}));  w.add(p + "norm2", widen_f32(ck, {&bf(l + "post_attention_layernorm.weight", {TE_HID})}));
        w.add(p + "qnorm", widen_f32(ck, {&bf(l + "self_attn.q_norm.weight", {HEAD_DIM})})); w.add(p + "knorm", widen_f32(ck, {&bf(l + "self_attn.k_norm.weight", {HEAD_DIM})}));
    }
    // the vision tower: bf16 rows on the bf16 matmuls, padded only where a kernel's K or N demands it
    constexpr int VHID = 1152, VHEADS = 16, VHD = 72, VHDP = 128, VMLP_SRC = 4304, VMLP = 4352, VOUT = 5120, VMERGE = 4 * VHID;
    w.add("vis.patch.w", rows_of(ck, {&ck.at("visual.patch_embed.proj.weight", "BF16", {VHID, 3, 2, 16, 16})}));   // [1152][3*2*16*16 = 1536]
    w.add("vis.patch.b", widen_f32(ck, {&bf("visual.patch_embed.proj.bias", {VHID})}));
    w.add("vis.pos", widen_f32(ck, {&bf("visual.pos_embed.weight", {2304, VHID})}));
    for (int i = 0; i < 27; ++i) {
        const std::string b = fmt("vis.b%d.", i), src = fmt("visual.blocks.%d.", i);
        for (const char *n : {"norm1", "norm2"}) { w.add(b + n + ".w", widen_f32(ck, {&bf(src + n + ".weight", {VHID})})); w.add(b + n + ".b", widen_f32(ck, {&bf(src + n + ".bias", {VHID})})); }
        w.add(b + "qkv.w", rows_of(ck, {&bf(src + "attn.qkv.weight", {3 * VHEADS * VHD, VHID})}));   w.add(b + "qkv.b", widen_f32(ck, {&bf(src + "attn.qkv.bias", {3 * VHEADS * VHD})}));
        w.add(b + "proj.w", regroup(ck, bf(src + "attn.proj.weight", {VHID, VHEADS * VHD}), VHEADS, VHD, VHDP));   // the rope kernel writes heads at 128
        w.add(b + "proj.b", widen_f32(ck, {&bf(src + "attn.proj.bias", {VHID})}));
        w.add(b + "fc1.w", rows_padded(ck, bf(src + "mlp.linear_fc1.weight", {VMLP_SRC, VHID}), VMLP - VMLP_SRC));   // N to a multiple of 64
        w.add(b + "fc1.b", widen_padded(ck, bf(src + "mlp.linear_fc1.bias", {VMLP_SRC}), VMLP));   // beside the padded rows
        w.add(b + "fc2.w", regroup(ck, bf(src + "mlp.linear_fc2.weight", {VHID, VMLP_SRC}), 1, VMLP_SRC, VMLP));     // K to match
        w.add(b + "fc2.b", widen_f32(ck, {&bf(src + "mlp.linear_fc2.bias", {VHID})}));
    }
    for (int j = 0; j < 4; ++j) {   // the three DeepStack mergers and the patch merger
        const std::string d = j < 3 ? fmt("vis.ds%d.", j) : "vis.merger.", src = j < 3 ? fmt("visual.deepstack_merger_list.%d.", j) : "visual.merger.";
        const int nw = j < 3 ? VMERGE : VHID;   // the mergers view [n][1152] as [n/4][4608]; only the DeepStack norms are 4608 wide
        w.add(d + "norm.w", widen_f32(ck, {&bf(src + "norm.weight", {nw})}));   w.add(d + "norm.b", widen_f32(ck, {&bf(src + "norm.bias", {nw})}));
        w.add(d + "fc1.w", rows_of(ck, {&bf(src + "linear_fc1.weight", {VMERGE, VMERGE})}));  w.add(d + "fc1.b", widen_f32(ck, {&bf(src + "linear_fc1.bias", {VMERGE})}));
        w.add(d + "fc2.w", rows_of(ck, {&bf(src + "linear_fc2.weight", {VOUT, VMERGE})}));    w.add(d + "fc2.b", widen_f32(ck, {&bf(src + "linear_fc2.bias", {VOUT})}));
    }
}

// ComfyUI's video VAE (minimax_h3_video_vae_fp16.safetensors), f16 throughout and run as f16: the decoder's 36
// blocks (its fused to_qkv rows are head-interleaved, its gate|up half is w1's first half: both reordered here, not
// converted), the decoder's heads, and the encoder's causal 3-D convs as the implicit-GEMM operands the conv kernels take.
void vvae_plan(const Checkpoint &ck, Weights &w) {
    auto f16 = [&](const std::string &n, std::initializer_list<int64_t> shape) -> const Entry & { return ck.at(n, "F16", shape); };
    const int inner = VAE_HEADS * VAE_D;
    for (int i = 0; i < 36; ++i) {
        const std::string p = fmt("blocks.%d.", i), src = fmt("decoder.transformer_blocks.%d.", i);
        // to_qkv holds [q | k | v] per head of 64; the rope kernel wants [Q | K | V]
        std::vector<Run> qkv_rows; for (int part = 0; part < 3; ++part) for (int head = 0; head < VAE_HEADS; ++head) qkv_rows.push_back({size_t(head * 3 + part) * VAE_D, VAE_D});
        w.add(p + "qkv.q", rows_permuted(ck, f16(src + "attn.to_qkv.weight", {3 * inner, VAE_HID}), qkv_rows));
        w.add(p + "qkv.b", widen_runs(ck, f16(src + "attn.to_qkv.bias", {3 * inner}), qkv_rows));
        w.add(p + "out.q", rows_of(ck, {&f16(src + "attn.to_out.weight", {VAE_HID, inner})}));   w.add(p + "out.b", widen_f32(ck, {&f16(src + "attn.to_out.bias", {VAE_HID})}));
        // w1 is gate | linear (comfy/ldm/minimax/vae.py: gate, x = w1(x).chunk(2)); the epilogue takes the linear half first
        const std::vector<Run> gu = interleaved_runs(VAE_FFN, 0, VAE_FFN, 16);
        w.add(p + "gu.q", rows_permuted(ck, f16(src + "ff.w1.weight", {2 * VAE_FFN, VAE_HID}), gu));
        w.add(p + "gu.b", widen_runs(ck, f16(src + "ff.w1.bias", {2 * VAE_FFN}), gu));
        w.add(p + "down.q", rows_of(ck, {&f16(src + "ff.w2.weight", {VAE_HID, VAE_FFN})}));      w.add(p + "down.b", widen_f32(ck, {&f16(src + "ff.w2.bias", {VAE_HID})}));
        for (const char *n : {"norm1", "norm2"}) w.add(p + n, widen_f32(ck, {&f16(src + n + ".weight", {VAE_HID})}));
        for (const char *n : {"scale1", "scale2"}) w.add(p + n, widen_f32(ck, {&f16(src + n, {VAE_HID})}));
    }
    w.add("vae.proj_in.w", regroup(ck, f16("decoder.x_embedder.weight", {VAE_HID, LATENT_CH}), 1, LATENT_CH, VAE_KIN));   // K to the f16 GEMM's multiple of 64
    w.add("vae.proj_in.b", widen_f32(ck, {&f16("decoder.x_embedder.bias", {VAE_HID})}));
    w.add("vae.register_tokens", widen_f32(ck, {&ck.at("decoder.register_tokens", "F16", {1, VAE_REG, VAE_HID})}));
    w.add("vae.norm_out.w", widen_f32(ck, {&f16("decoder.norm_out.weight", {VAE_HID})}));  w.add("vae.norm_out.b", widen_f32(ck, {&f16("decoder.norm_out.bias", {VAE_HID})}));
    w.add("vae.proj_out.w", rows_of(ck, {&f16("decoder.proj_out.weight", {VAE_OUT, VAE_HID})})); w.add("vae.proj_out.b", widen_f32(ck, {&f16("decoder.proj_out.bias", {VAE_OUT})}));
    w.add("vae.post_quant_conv.w", widen_f32(ck, {&ck.at("post_quant_conv.weight", "F16", {LATENT_CH, LATENT_CH, 1, 1, 1})}));
    w.add("vae.post_quant_conv.b", widen_f32(ck, {&f16("post_quant_conv.bias", {LATENT_CH})}));
    w.add("vae.latents_mean", widen_f32(ck, {&f16("latents_mean", {LATENT_CH})}));  w.add("vae.latents_std", widen_f32(ck, {&f16("latents_std", {LATENT_CH})}));

    // the encoder: every conv weight [Cout][Cin][3][3][3] as the implicit GEMM's operand rows [Cout_pad][taps * Cin_pad],
    // taps in (t, y, x) order and the channel innermost, zero-padded; ".w2" is the last temporal tap alone (a single image)
    auto up = [](int n, int m) { return (n + m - 1) / m * m; };
    auto conv = [&](const std::string &out, const std::string &src, int cout, int cin, bool down = false) {
        const Entry &e = ck.at(src + ".weight", "F16", {cout, cin, 3, 3, 3});
        const int cin_pad = up(cin, 8), cout_pad = up(cout, 64);
        for (int taps : {27, 9}) {
            const int k = up(taps * cin_pad, 32); const char *p = ck.data(e);
            Recipe r; r.rows = size_t(cout_pad); r.row_bytes = r.pitch_bytes = size_t(k) * 2; r.built_bytes = r.rows * r.pitch_bytes;
            r.build = [p, cout, cin, cin_pad, k, taps, cout_pad] {
                std::vector<char> h(size_t(cout_pad) * k * 2, 0); uint16_t *o = (uint16_t *)h.data(); const uint16_t *in = (const uint16_t *)p;
                for (int oc = 0; oc < cout; ++oc) for (int ic = 0; ic < cin; ++ic) for (int t = 0; t < 3; ++t) for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x) {
                    if (taps == 9 && t != 2) continue;                                        // the image form keeps the last temporal tap
                    const int tap = taps == 27 ? (t * 3 + y) * 3 + x : y * 3 + x;
                    o[size_t(oc) * k + size_t(tap) * cin_pad + ic] = in[(((size_t(oc) * cin + ic) * 3 + t) * 3 + y) * 3 + x];
                }
                return h; };
            w.add(out + (taps == 27 ? ".w3" : ".w2"), r);
        }
        w.add(out + ".b", widen_padded(ck, ck.at(src + ".bias", "F16", {cout}), size_t(cout_pad)));   // the conv writes cout_pad channels
        (void)down;
    };
    auto matmul_w = [&](const std::string &out, const std::string &src, int cout, int cin) {   // a 1x1x1 conv as [Cout_pad][K]
        const Entry &e = ck.at(src + ".weight", "F16", {cout, cin, 1, 1, 1});
        w.add(out + ".wm", regroup(ck, e, 1, size_t(cin), size_t(up(up(cin, 8), 32)), size_t(up(cout, 64))));
        w.add(out + ".b", widen_padded(ck, ck.at(src + ".bias", "F16", {cout}), size_t(up(cout, 64))));
    };
    static const int mid[6] = {128, 256, 256, 512, 512, 1024};
    conv("venc.conv_in", "encoder.conv_in", 128, 3);
    int c = 128;
    for (int l = 0; l < 6; ++l) {
        for (int r = 0; r < 2; ++r) {
            const std::string b = fmt("venc.l%d.r%d.", l, r), src = fmt("encoder.down.%d.block.%d.", l, r);
            for (const char *n : {"norm1", "norm2"}) {
                const int ch = n[4] == '1' ? c : mid[l];
                w.add(b + n + ".g", widen_f32(ck, {&ck.at(src + n + ".weight", "F16", {ch})})); w.add(b + n + ".b", widen_f32(ck, {&ck.at(src + n + ".bias", "F16", {ch})}));
            }
            conv(b + "conv1", src + "conv1", mid[l], c);
            conv(b + "conv2", src + "conv2", mid[l], mid[l]);
            if (c != mid[l]) matmul_w(b + "nin", src + "nin_shortcut", mid[l], c);
            c = mid[l];
        }
        if (ck.has(fmt("encoder.down.%d.downsample.conv.weight", l))) conv(fmt("venc.l%d.down", l), fmt("encoder.down.%d.downsample.conv", l), c, c, true);
    }
    w.add("venc.norm_out.g", widen_f32(ck, {&ck.at("encoder.norm_out.weight", "F16", {c})})); w.add("venc.norm_out.b", widen_f32(ck, {&ck.at("encoder.norm_out.bias", "F16", {c})}));
    conv("venc.conv_out", "encoder.conv_out", 48, c);
    matmul_w("venc.quant", "quant_conv", 48, 48);
    w.add("venc.latents_mean", widen_f32(ck, {&f16("latents_mean", {LATENT_CH})})); w.add("venc.latents_std", widen_f32(ck, {&f16("latents_std", {LATENT_CH})}));
}

// ComfyUI's audio VAE (minimax_h3_audio_vae_fp32.safetensors), f32 throughout with its weight norm already folded:
// the BigVGAN vocoder and the encoder's DAC stack, verbatim except for the conv reshapes the kernels' layouts want and
// exp() of the decoder's SnakeBeta parameters, which the module stores as logs (comfy/ldm/minimax/audio_vae.py).
void avae_plan(const Checkpoint &ck, Weights &w) {
    auto f32 = [&](const std::string &n, std::initializer_list<int64_t> shape) -> const Entry & { return ck.at(n, "F32", shape); };
    // [Cout][Cin][k] (or the transposed conv's [Cin][Cout][k]) as the kernels' flat [Cout][Cin * k] rows; bias_width 0: no bias
    auto flat = [&](const std::string &out, const std::string &src, std::initializer_list<int64_t> shape, int64_t bias_width = -1) {
        const Entry &e = ck.at(src + ".weight", "F32", shape);
        Recipe r; r.rows = 1; r.row_bytes = r.pitch_bytes = e.bytes; r.segments.push_back({ck.data(e), 1});
        w.add(out + ".w", r);
        const int64_t bw = bias_width < 0 ? *(shape.begin()) : bias_width;
        if (bw) w.add(out + ".b", widen_f32(ck, {&ck.at(src + ".bias", "F32", {bw})}));
    };
    auto exp_f32 = [&](const std::string &out, const std::string &src, int64_t n) {   // the SnakeBeta parameters are stored as logs
        const Entry &e = ck.at(src, "F32", {n}); const char *p = ck.data(e);
        Recipe r; r.built_bytes = size_t(n) * 4;
        r.build = [p, n] { std::vector<char> h(size_t(n) * 4); float *o = (float *)h.data();
            for (int64_t i = 0; i < n; ++i) { float v; memcpy(&v, p + i * 4, 4); o[i] = std::exp(v); } return h; };
        w.add(out, r);
    };
    static const int UPK[7] = {9, 9, 4, 4, 4, 4, 4}, RESK[3] = {3, 7, 11};   // the decoder rates {5,5,2,2,2,2,2} halve the channels, which the loop tracks
    w.add("audio.latents_mean", widen_f32(ck, {&f32("latents_mean", {AUDIO_CH})})); w.add("audio.latents_std", widen_f32(ck, {&f32("latents_std", {AUDIO_CH})}));
    flat("audio.dec_in_proj", "dec_in_proj", {2048, AUDIO_CH, 1});
    flat("audio.conv_pre", "decoder.conv_pre", {1024, 2048, 7});
    int c = 1024;
    for (int i = 0; i < 7; ++i) {
        const int cout = c / 2;
        flat(fmt("audio.ups.%d", i), fmt("decoder.ups.%d.0", i), {c, cout, UPK[i]}, cout);   // a transposed conv: [Cin][Cout][k], bias Cout
        for (int j = 0; j < 3; ++j) {
            const int r = i * 3 + j; const std::string src = fmt("decoder.resblocks.%d", r);
            for (int d = 0; d < 3; ++d) {
                flat(fmt("audio.res.%d.c1.%d", r, d), src + fmt(".convs1.%d", d), {cout, cout, RESK[j]});
                flat(fmt("audio.res.%d.c2.%d", r, d), src + fmt(".convs2.%d", d), {cout, cout, RESK[j]});
            }
            for (int act = 0; act < 6; ++act) {
                exp_f32(fmt("audio.res.%d.act.%d.alpha", r, act), src + fmt(".activations.%d.act.alpha", act), cout);
                exp_f32(fmt("audio.res.%d.act.%d.beta", r, act), src + fmt(".activations.%d.act.beta", act), cout);
            }
        }
        c = cout;
    }
    exp_f32("audio.post.alpha", "decoder.activation_post.act.alpha", c); exp_f32("audio.post.beta", "decoder.activation_post.act.beta", c);
    { const Entry &e = ck.at("decoder.activation_post.upsample.filter", "F32", {1, 1, 12});   // the same 12-tap Kaiser-sinc in every activation
      for (const auto &kv : ck.entries) if (kv.first.size() > 7 && kv.first.compare(kv.first.size() - 7, 7, ".filter") == 0)
          if (kv.second.bytes != e.bytes || memcmp(ck.data(kv.second), ck.data(e), e.bytes) != 0) throw std::runtime_error("the resampling filters differ: " + kv.first);
      Recipe r; r.rows = 1; r.row_bytes = r.pitch_bytes = e.bytes; r.segments.push_back({ck.data(e), 1}); w.add("audio.fir", r); }
    flat("audio.conv_post", "decoder.conv_post", {1, c, 7}, 0);   // no bias

    // the encoder's DAC stack and posterior head
    static const int ARATES[5] = {2, 4, 4, 5, 5};
    flat("aenc.conv_in", "encoder.block.0", {64, 1, 7});
    int dim = 64;
    for (int i = 1; i <= 5; ++i) {
        const std::string p = fmt("encoder.block.%d", i);
        for (int r = 0; r < 3; ++r) {
            const std::string q = p + fmt(".block.%d.block", r);
            w.add(fmt("aenc.b%d.r%d.act0", i, r), widen_f32(ck, {&ck.at(q + ".0.alpha", "F32", {1, dim, 1})}));   // the encoder's Snake takes alpha as stored
            flat(fmt("aenc.b%d.r%d.c1", i, r), q + ".1", {dim, dim, 7});
            w.add(fmt("aenc.b%d.r%d.act1", i, r), widen_f32(ck, {&ck.at(q + ".2.alpha", "F32", {1, dim, 1})}));
            flat(fmt("aenc.b%d.r%d.c2", i, r), q + ".3", {dim, dim, 1});
        }
        w.add(fmt("aenc.b%d.act", i), widen_f32(ck, {&ck.at(p + ".block.3.alpha", "F32", {1, dim, 1})}));
        flat(fmt("aenc.b%d.down", i), p + ".block.4", {2 * dim, dim, 2 * ARATES[i - 1]});
        dim *= 2;
    }
    w.add("aenc.act_out", widen_f32(ck, {&ck.at("encoder.block.6.alpha", "F32", {1, dim, 1})}));
    flat("aenc.conv_out", "encoder.block.7", {dim, dim, 3});
    for (const char *n : {"norm1", "norm2", "norm3"}) { const int64_t width = std::string(n) == "norm2" ? AUDIO_CH : 2048;
        w.add(fmt("aenc.pre.%s.w", n), widen_f32(ck, {&f32(fmt("pre_block.%s.weight", n), {width})})); w.add(fmt("aenc.pre.%s.b", n), widen_f32(ck, {&f32(fmt("pre_block.%s.bias", n), {width})})); }
    { const Entry &e = ck.at("pre_block.attn.qkv.weight", "F32", {6144, 2048});
      Recipe r; r.rows = e.rows(); r.row_bytes = r.pitch_bytes = e.row_bytes(); r.segments.push_back({ck.data(e), e.rows()}); w.add("aenc.pre.qkv.w", r); }
    w.add("aenc.pre.qkv.b", widen_f32(ck, {&f32("pre_block.attn.q_bias", {2048}), &f32("pre_block.attn.zero_k_bias", {2048}), &f32("pre_block.attn.v_bias", {2048})}));
    flat("aenc.pre.attn_proj", "pre_block.attn.proj", {AUDIO_CH, AUDIO_CH});
    flat("aenc.pre.proj", "pre_block.proj", {AUDIO_CH, 2048});
    flat("aenc.pre.mlp.norm", "pre_block.mlp.norm", {AUDIO_CH});
    flat("aenc.pre.mlp.w0", "pre_block.mlp.w0", {64, AUDIO_CH}); flat("aenc.pre.mlp.w1", "pre_block.mlp.w1", {64, AUDIO_CH}); flat("aenc.pre.mlp.w2", "pre_block.mlp.w2", {AUDIO_CH, 64});
    flat("aenc.mean_proj", "mean_proj", {AUDIO_CH, AUDIO_CH, 1});
    w.add("aenc.latents_mean", widen_f32(ck, {&f32("latents_mean", {AUDIO_CH})})); w.add("aenc.latents_std", widen_f32(ck, {&f32("latents_std", {AUDIO_CH})}));
}

// --- the pipeline ----------------------------------------------------------------------------
class Pipe {
    DeviceBuffers memory_;
public:
    explicit Pipe(const h3pipe_config &cfg) {
        if (!cfg.kernel_sources || !cfg.cache_dir || !cfg.loom_compile) throw std::invalid_argument("kernel_sources, cache_dir and loom_compile are required");
        rt();
        comp_.exe = cfg.loom_compile; comp_.sources = cfg.kernel_sources; comp_.cache = cfg.cache_dir;
        dit_file_ = cfg.dit_file ? cfg.dit_file : ""; te_file_ = cfg.te_file ? cfg.te_file : "";
        video_vae_file_ = cfg.video_vae_file ? cfg.video_vae_file : ""; audio_vae_file_ = cfg.audio_vae_file ? cfg.audio_vae_file : "";
        attn_qk_bits_ = cfg.attn_qk_bits ? cfg.attn_qk_bits : 8; if (attn_qk_bits_ != 4 && attn_qk_bits_ != 8 && attn_qk_bits_ != 16) throw std::invalid_argument("attn_qk_bits must be 4, 8 or 16");
        ones_ = memory_.alloc(size_t(HID) * 4); { std::vector<float> o(HID, 1.0f); rt().h2d(ones_, o.data(), o.size() * 4); }
        zeros_ = memory_.alloc(size_t(2 * TE_FFN) * 4); rt().memset(zeros_, 0, size_t(2 * TE_FFN) * 4);
    }


    Profile prof{std::getenv("H3_PROFILE") != nullptr && *std::getenv("H3_PROFILE") != 0};   // H3_PROFILE=1: synchronized per-stage times printed after every step

    // --- the prompt: embedding lookup on the host, the encoder, condition_proj, the refiner ---
    void ensure_seq(size_t seq) {
        if (seq <= seq_cap_) return;
        for (void **p : {&x_, &cls_, &cls0_, &tcls_, &cos_, &sin_, &in32_, &out32_, &text_copy_}) { if (*p) memory_.free(*p); *p = nullptr; }
        seq_cap_ = 0;
        const size_t T = (seq + 255) / 256 * 256 + 32;
        x_ = memory_.alloc(T * HID * 4); rt().memset(x_, 0, T * HID * 4);
        cls_ = memory_.alloc(T * 4); rt().memset(cls_, 0, T * 4);
        cls0_ = memory_.alloc(T * 4); rt().memset(cls0_, 0, T * 4);       // the single-class GEMMs (embedders, condition proj) index their gate table with this
        tcls_ = memory_.alloc(T * 4); rt().memset(tcls_, 0, T * 4);
        cos_ = memory_.alloc(T * ROPE_HALF * 4); sin_ = memory_.alloc(T * ROPE_HALF * 4);
        in32_ = memory_.alloc(T * VIDEO_PATCH * 4);        // the embedders' f32 rows [rows][96 | 32]
        out32_ = memory_.alloc(T * FINAL_N * 4);           // the final layer's f32 rows [rows][128]
        seq_cap_ = T;
    }

    // The DiT checkpoint, opened on first use (the conditioning tables, the refiner, the embedders, the blocks, the final layer).
    void ensure_dit() {
        if (dit_w_.is_open()) return;
        if (dit_file_.empty()) throw std::invalid_argument("no DiT checkpoint: h3pipe_config.dit_file is NULL");
        dit_w_.open(dit_file_, dit_plan);
    }
    // The text encoder checkpoint, opened on first use (its 50 layers, its embedding table, the vision tower).
    // The video VAE checkpoint, opened on first use (the decoder's 36 blocks and heads, the encoder's convs).
    void ensure_vvae() {
        if (vvae_w_.is_open()) return;
        if (video_vae_file_.empty()) throw std::invalid_argument("no video VAE checkpoint: h3pipe_config.video_vae_file is NULL");
        vvae_w_.open(video_vae_file_, vvae_plan);
    }
    // The audio VAE checkpoint, opened on first use (the vocoder and the audio encoder).
    void ensure_avae() {
        if (avae_w_.is_open()) return;
        if (audio_vae_file_.empty()) throw std::invalid_argument("no audio VAE checkpoint: h3pipe_config.audio_vae_file is NULL");
        avae_w_.open(audio_vae_file_, avae_plan);
    }
    void ensure_te() {
        if (te_w_.is_open()) return;
        if (te_file_.empty()) throw std::invalid_argument("no text encoder checkpoint: h3pipe_config.te_file is NULL");
        te_w_.open(te_file_, te_plan);
        embed_ = &te_w_.file().at("model.embed_tokens.weight", "BF16", {-1, TEXT_DIM});
    }
    // x rows [row0, row0 + rows) = W · in + b on the checkpoint's f32 patch projection as stored: in f32 [rows][k] on in32_
    void embed_f32(const char *stage, size_t row0, size_t rows, int k, const std::string &wname) {
        matmul(stage, rows, k, HID, in32_, dit_w_.at(wname + ".w", size_t(HID) * k * 4), dit_w_.at(wname + ".b", size_t(HID) * 4), (float *)x_ + row0 * HID);
    }

    struct VisionSpan { size_t start, count; int merged_h, merged_w; const float *merged, *deepstack; };   // an image's rows in the presentation
    void text_in(const int32_t *ids, int n, const std::vector<VisionSpan> &spans = {}) {
        // embedding rows from the file (bf16) -> f32; vision spans take the merged vision embeds
        std::vector<float> emb(size_t(n) * TEXT_DIM);
        for (const VisionSpan &sp : spans) memcpy(emb.data() + sp.start * TEXT_DIM, sp.merged, sp.count * TEXT_DIM * 4);
        ensure_te();
        { const char *table = te_w_.file().data(*embed_); const size_t rows = embed_->rows();
          for (int i = 0; i < n; ++i) { if (ids[i] < 0) { bool in_span = false; for (const VisionSpan &sp : spans) in_span |= size_t(i) >= sp.start && size_t(i) < sp.start + sp.count; if (in_span) continue; }
              if (ids[i] < 0 || size_t(ids[i]) >= rows) throw std::invalid_argument("token id out of range: " + std::to_string(ids[i]));
              const char *src = table + size_t(ids[i]) * TEXT_DIM * 2;   // the table's bf16 row, widened
              for (int j = 0; j < TEXT_DIM; ++j) { uint16_t h; memcpy(&h, src + j * 2, 2); emb[size_t(i) * TEXT_DIM + j] = bf16_to_f32(h); } } }
        // the encoder's 50 layers
        std::string span_sig; for (const VisionSpan &sp : spans) span_sig += fmt("%zu:%zu:%d:%d;", sp.start, sp.count, sp.merged_h, sp.merged_w);
        if (!te_ready_ || !te_ || te_->tokens() != size_t(n) || te_span_sig_ != span_sig) {
            te_span_sig_.clear(); te_ready_ = false;
            te_.reset(); te_ = std::make_unique<Stack>(comp_, StackDims{TE_HID, TE_HEADS, TE_KV, HEAD_DIM, TE_FFN, 128, 1, 8, 1e-6f, false, true, true}, size_t(n), 50, te_w_, "blocks.%d.", true, nullptr, "te");
            for (void **q : {&te_x_, &te_cos_, &te_sin_, &te_cls_}) { memory_.free(*q); *q = nullptr; }
            const size_t T = te_->capacity();
            te_x_ = memory_.alloc(T * TE_HID * 4); rt().memset(te_x_, 0, T * TE_HID * 4);
            te_cos_ = memory_.alloc(T * TE_ROPE_HALF * 4); te_sin_ = memory_.alloc(T * TE_ROPE_HALF * 4);
            te_cls_ = memory_.alloc(T * 4); rt().memset(te_cls_, 0, T * 4);
            std::vector<float> c(size_t(n) * TE_ROPE_HALF), s(size_t(n) * TE_ROPE_HALF);
            std::vector<double> pos(size_t(n) * 3);   // (t, h, w) per token: text runs sequentially, each image span its grid (qwen2vl_mrope_position_ids)
            { size_t cursor = 0; double offset = 0; std::vector<VisionSpan> ordered = spans; std::sort(ordered.begin(), ordered.end(), [](const VisionSpan &x, const VisionSpan &y) { return x.start < y.start; });
              for (const VisionSpan &sp : ordered) {
                  for (size_t i = cursor; i < sp.start; ++i) for (int ax = 0; ax < 3; ++ax) pos[i * 3 + ax] = double(i) + offset;
                  for (size_t k = 0; k < sp.count; ++k) { pos[(sp.start + k) * 3] = double(sp.start) + offset; pos[(sp.start + k) * 3 + 1] = double(sp.start) + offset + double(k / size_t(sp.merged_w)); pos[(sp.start + k) * 3 + 2] = double(sp.start) + offset + double(k % size_t(sp.merged_w)); }
                  const double len_max = std::max(sp.merged_h, sp.merged_w);
                  offset += len_max - double(sp.count); cursor = sp.start + sp.count;
              }
              for (size_t i = cursor; i < size_t(n); ++i) for (int ax = 0; ax < 3; ++ax) pos[i * 3 + ax] = double(i) + offset; }
            for (int t = 0; t < n; ++t) for (int j = 0; j < TE_ROPE_HALF; ++j) {
                const int ax = mrope_axis(j);   // interleaved mrope sections [24, 20, 20]: pair j takes the t / h / w position
                const double inv = std::pow(5000000.0, -double(2 * j) / HEAD_DIM), ang = pos[size_t(t) * 3 + ax] * inv;
                c[size_t(t) * TE_ROPE_HALF + j] = float(std::cos(ang)); s[size_t(t) * TE_ROPE_HALF + j] = float(std::sin(ang)); }
            rt().h2d(te_cos_, c.data(), c.size() * 4); rt().h2d(te_sin_, s.data(), s.size() * 4);
            te_span_sig_ = span_sig; te_ready_ = true;
        }
        rt().h2d(te_x_, emb.data(), emb.size() * 4);
        auto te_cond = [&](int) { return LayerCond{(const float *)zeros_, (const float *)ones_, (const float *)zeros_, (const float *)ones_}; };
        if (spans.empty()) te_->forward(&prof, te_x_, te_cls_, te_cos_, te_sin_, te_cond);
        else {   // DeepStack: the vision tower's features from blocks 8, 16, 24 added at the image rows after the first three layers
            if (!ds_buf_) ds_buf_ = memory_.alloc(size_t(4096) * TEXT_DIM * 4);
            for (int layer = 0; layer < 3; ++layer) {
                te_->forward(&prof, te_x_, te_cls_, te_cos_, te_sin_, te_cond, layer, layer + 1);
                for (const VisionSpan &sp : spans) {
                    if (sp.count > 4096) throw std::invalid_argument("vision span too long");
                    rt().h2d(ds_buf_, sp.deepstack + size_t(layer) * sp.count * TEXT_DIM, sp.count * TEXT_DIM * 4);
                    axpy(1.0f, 1.0f, sp.count * TEXT_DIM, ds_buf_, (float *)te_x_ + sp.start * TEXT_DIM);
                }
            }
            te_->forward(&prof, te_x_, te_cls_, te_cos_, te_sin_, te_cond, 3, 50);
        }
        // condition_proj: the encoder's f32 hidden rows through the bf16 matmul on the checkpoint's bf16 rows, into x rows [0, n)
        ensure_dit();
        { const std::string stem = "matmul_bias_bf16_wmma", ns = "h3." + stem + ".";
          auto k = comp_.get(stem, "h3_" + stem, {{ns + "k_size", std::to_string(TEXT_DIM)}, {ns + "n_size", std::to_string(HID)}});
          KernArgs a; a.i32(n).ptr(te_x_).ptr(dit_w_.at("h3.cond.w", size_t(HID) * TEXT_DIM * 2)).ptr(dit_w_.at("h3.cond.b", size_t(HID) * 4)).ptr(x_);
          launch(*k, &prof, "condition proj", unsigned(HID / 64), unsigned((n + 63) / 64), 256, a); }
        // the token refiner: two H3-shaped blocks on bf16 rows without rope (identity tables), then its final norm
        if (!refiner_ready_ || !refiner_ || refiner_->tokens() != size_t(n)) {
            refiner_ready_ = false;
            StackDims rd{HID, HEADS, HEADS, HEAD_DIM, FFN, ROPE_DIM, 1, 16, 1e-5f, false, true, false}; rd.bf16 = true;
            refiner_.reset(); refiner_ = std::make_unique<Stack>(comp_, rd, size_t(n), 2, dit_w_, "h3.refiner.%d.", true, nullptr, "refiner");
            std::vector<float> c(size_t(n) * ROPE_HALF, 1.0f), s(size_t(n) * ROPE_HALF, 0.0f);
            for (void **q : {&ref_cos_, &ref_sin_}) { memory_.free(*q); *q = nullptr; }
            ref_cos_ = memory_.alloc(c.size() * 4); ref_sin_ = memory_.alloc(s.size() * 4);
            rt().h2d(ref_cos_, c.data(), c.size() * 4); rt().h2d(ref_sin_, s.data(), s.size() * 4);
            const std::string ns = "h3.norm_mod_f32.";
            norm_f32_ = comp_.get("norm_mod_f32", "h3_norm_mod_f32", {{ns + "width", std::to_string(HID)}, {ns + "lanes", std::to_string(lanes_for(HID))}, {ns + "eps", num(1e-5)}, {ns + "classes", "1"}});
        }
        refiner_ready_ = true;
        refiner_->forward(&prof, x_, cls0_, ref_cos_, ref_sin_, [&](int) { return LayerCond{(const float *)zeros_, (const float *)ones_, (const float *)zeros_, (const float *)ones_}; });
        { KernArgs a; a.i32(n).ptr(x_).ptr(dit_w_.at("h3.refiner.final_norm", size_t(HID) * 4)).ptr(zeros_).ptr(cls0_); launch(*norm_f32_, &prof, "refiner final norm", unsigned(n), 1, unsigned(lanes_for(HID)), a); }
    }

    void get_text_in(const int32_t *ids, int n, float *out) {
        ensure_dit(); ensure_seq(size_t(n));
        text_in(ids, n);
        rt().sync(); rt().d2h(out, x_, size_t(n) * HID * 4);
    }

    // --- conditioning per step ---
    void temb(float t, float *out8) const {
        const double pos = std::min(std::max(double(t), 0.0), 1.0) * 1024.0; const int i0 = std::min(int(std::floor(pos)), 1023); const double f = pos - i0;
        for (int j = 0; j < 8; ++j) out8[j] = float(curve_[size_t(i0) * 8 + j] * (1.0 - f) + curve_[size_t(i0 + 1) * 8 + j] * f);
    }
    // the per-layer tables in the runtime's layout: rows [0,2C) (scale_msa, shift_msa) per class, [2C,3C) gate_msa, [3C,5C) (scale_mlp, shift_mlp), [5C,6C) gate_mlp;
    // class = timestep index * 3 + modality; the projection's chunks per (modality d, j): shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    void upload_mods(const float tv[8], const float ta[8], const float tcv[8], const float tca[8]) {
        std::vector<float> table(size_t(50) * MODS_ROWS * HID, 0.0f), proj(size_t(3 * 6 * HID));
        for (int i = 0; i < 50; ++i) {
            const float *W = adaln_w_[i].data(), *B = adaln_b_[i].data();
            for (int m = 0; m < 4; ++m) {
                const float *te = m == 0 ? tv : (m == 1 ? ta : (m == 2 ? tcv : tca));
                for (size_t r = 0; r < proj.size(); ++r) { float acc = B[r]; for (int j = 0; j < 8; ++j) acc += W[r * 8 + j] * te[j]; proj[r] = acc; }
                for (int d = 0; d < 3; ++d) {
                    const int cls = m * 3 + d; float *t = table.data() + size_t(i) * MODS_ROWS * HID;
                    const float *chunk = [&](int j) { return proj.data() + (size_t(d) * 6 + j) * HID; }(0);
                    (void)chunk;
                    auto ch = [&](int j) { return proj.data() + (size_t(d) * 6 + j) * HID; };
                    memcpy(t + size_t(2 * cls) * HID, ch(1), HID * 4); memcpy(t + size_t(2 * cls + 1) * HID, ch(0), HID * 4);
                    memcpy(t + size_t(2 * CLASSES + cls) * HID, ch(2), HID * 4);
                    memcpy(t + size_t(3 * CLASSES + 2 * cls) * HID, ch(4), HID * 4); memcpy(t + size_t(3 * CLASSES + 2 * cls + 1) * HID, ch(3), HID * 4);
                    memcpy(t + size_t(5 * CLASSES + cls) * HID, ch(5), HID * 4);
                }
            }
        }
        rt().h2d(mods_, table.data(), table.size() * 4);
        // the final layer's (scale, shift) per timestep class: rows 2c = scale, 2c + 1 = shift; the projection gives (shift | scale)
        std::vector<float> ft(size_t(4) * HID);
        for (int m = 0; m < 2; ++m) {
            const float *te = m == 0 ? tv : ta;
            for (int r = 0; r < 2 * HID; ++r) { float acc = final_b_[r]; for (int j = 0; j < 8; ++j) acc += final_w_[size_t(r) * 8 + j] * te[j]; (r < HID ? ft[size_t(2 * m + 1) * HID + r] : ft[size_t(2 * m) * HID + r - HID]) = acc; }
        }
        rt().h2d(final_table_, ft.data(), ft.size() * 4);
    }
    LayerCond cond(int i) const {
        const float *mod = (const float *)mods_ + size_t(i) * MODS_ROWS * HID;
        return LayerCond{mod, mod + size_t(CLASSES) * 2 * HID, mod + size_t(CLASSES) * 3 * HID, mod + size_t(CLASSES) * 5 * HID};
    }

    void ensure_conditioning() {
        if (conditioning_ready_) return;
        adaln_w_.clear(); adaln_b_.clear();
        ensure_dit();
        // host-side conditioning tables (the checkpoint's f32 curve and f16 projections, widened)
        curve_ = dit_w_.host_f32("h3.adaln_t_table", 1025 * 8);
        inv_freq_ = dit_w_.host_f32("h3.rope_inv_freq", 16);
        for (int i = 0; i < 50; ++i) { adaln_w_.push_back(dit_w_.host_f32(fmt("h3.blocks.%d.adaln.w", i), size_t(3 * 6 * HID) * 8)); adaln_b_.push_back(dit_w_.host_f32(fmt("h3.blocks.%d.adaln.b", i), size_t(3 * 6 * HID))); }
        final_w_ = dit_w_.host_f32("h3.final.adaln.w", size_t(2 * HID) * 8); final_b_ = dit_w_.host_f32("h3.final.adaln.b", size_t(2 * HID));
        if (!mods_) mods_ = memory_.alloc(size_t(50) * MODS_ROWS * HID * 4);
        if (!final_table_) final_table_ = memory_.alloc(size_t(4) * HID * 4);
        conditioning_ready_ = true;
    }

    // --- denoising ---
    void denoise(const int32_t *ids, int n, const h3pipe_params &p, const float *noise_video, const float *noise_audio, float *video_out, float *audio_out, h3pipe_progress progress, void *user, const std::vector<h3pipe_ref> &refs = {}, const std::vector<h3pipe_keyframe> &kfs = {}) {
        const h3pipe_shape sh = shape_or_throw(p);   // 32 = one 2x2 latent patch: the audio-only canvas
        if (p.steps < 2 || p.steps > 1000) throw std::invalid_argument("steps must be 2..1000");
        ensure_conditioning();
        Layout lay(n, sh.latent_t, sh.lat_h, sh.lat_w, sh.audio_t, refs, kfs);
        const size_t L = size_t(n), Rr = lay.ref_rows, LR = L + Rr, Na = lay.audio_rows, Nv = lay.video_rows, S = lay.seq_len;
        ensure_seq(S);
        // image references presented to the text encoder: the vision tower per image, in the order of the placeholder runs
        std::vector<VisionSpan> spans; std::vector<std::vector<float>> vmerged, vdeep;
        { std::vector<std::pair<size_t, size_t>> runs; for (int i = 0; i < n; ++i) if (ids[i] < 0) { if (runs.empty() || runs.back().first + runs.back().second != size_t(i)) runs.push_back({size_t(i), 0}); ++runs.back().second; }
          size_t ri = 0;
          std::vector<std::tuple<const float *, int, int>> pics;   // keyframes first, then image references, as the presentation orders them
          for (const h3pipe_keyframe &kf : kfs) if (kf.pixels) pics.emplace_back(kf.pixels, kf.height, kf.width);
          for (const h3pipe_ref &rf : refs) if (rf.kind == 0 && rf.pixels) pics.emplace_back(rf.pixels, rf.height, rf.width);
          for (const auto &pc : pics) {
              const float *px = std::get<0>(pc); const int ph = std::get<1>(pc), pw = std::get<2>(pc);
              if (ri >= runs.size()) throw std::invalid_argument("more images than placeholder runs in the ids");
              const int mh = ph / 32, mw = pw / 32;
              if (runs[ri].second != size_t(mh) * mw) throw std::invalid_argument(fmt("placeholder run %zu has %zu ids, the image needs %d", ri, runs[ri].second, mh * mw));
              vmerged.emplace_back(); vdeep.emplace_back(); vision_embed(px, ph, pw, vmerged.back(), vdeep.back());
              spans.push_back({runs[ri].first, runs[ri].second, mh, mw, nullptr, nullptr}); ++ri;
          }
          if (ri != runs.size()) throw std::invalid_argument("placeholder runs in the ids without an image reference with pixels");
          for (size_t i = 0; i < spans.size(); ++i) { spans[i].merged = vmerged[i].data(); spans[i].deepstack = vdeep[i].data(); } }
        for (const VisionSpan &sp : spans) lay.mark_vision(sp.start, sp.count);
        text_in(ids, n, spans);
        // the packed layout's tables
        rt().h2d(cls_, lay.adaln_rows.data(), S * 4); rt().h2d(tcls_, lay.tclass.data(), S * 4);
        { std::vector<float> c(S * ROPE_HALF), s(S * ROPE_HALF);
          for (size_t r = 0; r < S; ++r) for (int ax = 0; ax < 3; ++ax) for (int j = 0; j < 16; ++j) { const float ang = float(lay.pos[3 * r + ax]) * inv_freq_[j]; c[r * ROPE_HALF + ax * 16 + j] = std::cos(ang); s[r * ROPE_HALF + ax * 16 + j] = std::sin(ang); }
          rt().h2d(cos_, c.data(), c.size() * 4); rt().h2d(sin_, s.data(), s.size() * 4); }
        if (!dit_ || dit_->tokens() != S) {
            const char *qk = std::getenv("H3_ATTN_QK"); StackDims dd{HID, HEADS, HEADS, HEAD_DIM, FFN, ROPE_DIM, CLASSES, 8, 1e-5f, false, true, false}; const int qkb = qk ? (std::string(qk) == "f16" ? 16 : (std::string(qk) == "i8" ? 8 : 4)) : attn_qk_bits_; dd.attn_i4 = qkb == 4; dd.attn_qk_bits = qkb;   // H3_ATTN_QK=f16|i8|i4 overrides the config
            dit_.reset(); dit_ = std::make_unique<Stack>(comp_, dd, S, 50, dit_w_, "blocks.%d.", true, nullptr, "dit"); }
        if (!final_norm_) { const std::string ns = "h3.norm_mod_f32."; final_norm_ = comp_.get("norm_mod_f32", "h3_norm_mod_f32", {{ns + "width", std::to_string(HID)}, {ns + "lanes", std::to_string(lanes_for(HID))}, {ns + "eps", num(1e-5)}, {ns + "classes", "2"}}); }
        const size_t generated_rows = Na + Nv;   // the final head has only the video/audio timestep classes
        // latents as rows: video [Nv][96] (row = (t*(H/2) + hh)*(W/2) + ww, column = c*4 + dy*2 + dx), audio [2*audio_t][32] (row = c*audio_t + t)
        std::vector<float> vrows(Nv * VIDEO_PATCH), arows(Na * AUDIO_CH);
        const bool res = p.sampler == 1;   // ComfyUI's res_multistep over the pack on the video sigma grid; arows is what the network sees, yrows the carried audio variable
        const double shift_v = p.video_shift > 0 ? p.video_shift : 12.0, shift_a = p.audio_shift > 0 ? p.audio_shift : 3.0, ascale = shift_v / shift_a;
        std::vector<float> yrows, den_v, den_a, old_v, old_a;
        const int T = sh.latent_t, H = sh.lat_h, W = sh.lat_w, A = sh.audio_t;
        Rng rng(p.seed);
        if (noise_video) for (size_t i = 0; i < vrows.size(); ++i) vrows[i] = 0; // filled below from the tensor layout
        auto tensor_to_rows = [&](const float *lat) { for (int c = 0; c < LATENT_CH; ++c) for (int t = 0; t < T; ++t) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const size_t row = (size_t(t) * (H / 2) + y / 2) * (W / 2) + x / 2, col = size_t(c) * 4 + (y % 2) * 2 + (x % 2);
            vrows[row * VIDEO_PATCH + col] = lat[((size_t(c) * T + t) * H + y) * W + x]; } };
        auto rows_to_tensor = [&](float *lat) { for (int c = 0; c < LATENT_CH; ++c) for (int t = 0; t < T; ++t) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const size_t row = (size_t(t) * (H / 2) + y / 2) * (W / 2) + x / 2, col = size_t(c) * 4 + (y % 2) * 2 + (x % 2);
            lat[((size_t(c) * T + t) * H + y) * W + x] = vrows[row * VIDEO_PATCH + col]; } };
        if (noise_video) tensor_to_rows(noise_video); else { std::vector<float> lat(size_t(LATENT_CH) * T * H * W); for (float &v : lat) v = rng.normal(); tensor_to_rows(lat.data()); }
        if (noise_audio) for (int c = 0; c < 2; ++c) for (int t = 0; t < A; ++t) for (int k = 0; k < AUDIO_CH; ++k) arows[(size_t(c) * A + t) * AUDIO_CH + k] = noise_audio[(size_t(c) * AUDIO_CH + k) * A + t];
        else for (float &v : arows) v = rng.normal();
        if (res) yrows = arows;   // carry = sigma_a / sigma_v = 1 at sigma_v = 1
        const char *dump_dir = std::getenv("H3_DUMP_DIR");   // per-step video latents [24][T][H][W] f32: x_00 = noise, x_k = the state after evaluation k
        auto dump = [&](size_t k) { if (!dump_dir) return; std::vector<float> lat(size_t(LATENT_CH) * T * H * W); rows_to_tensor(lat.data());
            if (FILE *f = fopen(fmt("%s/x_%02zu.f32", dump_dir, k).c_str(), "wb")) { fwrite(lat.data(), 4, lat.size(), f); fclose(f); } };
        dump(0);
        Schedule sv(p.steps, p.video_shift > 0 ? p.video_shift : 12.0), sa(p.steps, p.audio_shift > 0 ? p.audio_shift : 3.0);
        cache_acc_ = 0.0; have_cache_ = false; cache_skipped_ = 0;
        if (sv.timesteps.size() != sa.timesteps.size()) throw std::runtime_error("the two schedules differ in length");
        size_t in_rows = std::max(Na, Nv); for (const RefSeg &sg : lay.ref_segs) in_rows = std::max(in_rows, sg.rows);
        std::vector<float> in32(in_rows * VIDEO_PATCH); std::vector<float> out32(generated_rows * FINAL_N);
        const auto t_start = std::chrono::steady_clock::now();
        auto hnow = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        for (size_t step = 0; step < sv.timesteps.size(); ++step) {
            const double h0 = hnow(); double h_mods = 0, h_embed = 0, h_dit = 0, h_final = 0;
            float tv[8], ta[8], tcv[8], tca[8]; temb(sv.timesteps[step], tv); temb(sa.timesteps[step], ta);
            temb(std::max(sv.timesteps[step], VISUAL_COND_AUG), tcv); temb(std::max(sa.timesteps[step], 1.0f), tca); upload_mods(tv, ta, tcv, tca);
            h_mods = hnow();
            // audio rows -> x[L, L+Na), video rows -> x[L+Na, S), each through the f32 patch projection as stored
            if (res) { const float carry = sa.sigmas[step] / sv.sigmas[step]; for (size_t i = 0; i < arows.size(); ++i) arows[i] = yrows[i] * carry; }
            rt().h2d(in32_, arows.data(), Na * AUDIO_CH * 4);
            embed_f32("audio in", LR, Na, AUDIO_CH, "h3.audio_in");
            rt().h2d(in32_, vrows.data(), Nv * VIDEO_PATCH * 4);
            embed_f32("video in", LR + Na, Nv, VIDEO_PATCH, "h3.video_in");
            // the text rows are refreshed from a copy each step (the blocks update x in place)
            if (step == 0) {
                // reference rows (constant across steps, re-set every step like the text): the packed latents through the
                // patch projections; visual ones mixed with seeded noise at 0.999 as ComfyUI's condition augmentation
                for (const RefSeg &sg : lay.ref_segs) {
                    h3pipe_ref rf{};
                    if (sg.kind == 3) { const h3pipe_keyframe &kf = kfs[size_t(sg.ref)]; rf.kind = 0; rf.video_latent = kf.video_latent; rf.latent_t = 1; rf.lat_h = sh.lat_h; rf.lat_w = sh.lat_w; rf.audio_latent = kf.audio_latent; rf.audio_t = kf.audio_t; }
                    else rf = refs[size_t(sg.ref)];
                    if (sg.audio) {
                        for (int c = 0; c < 2; ++c) for (int t = 0; t < sg.audio_t; ++t) for (int k = 0; k < AUDIO_CH; ++k) in32[(size_t(c) * sg.audio_t + t) * AUDIO_CH + k] = rf.audio_latent[(size_t(c) * AUDIO_CH + k) * sg.audio_t + t];
                        rt().h2d(in32_, in32.data(), sg.rows * AUDIO_CH * 4); embed_f32("ref audio in", sg.row0, sg.rows, AUDIO_CH, "h3.audio_in");
                    } else {
                        Rng arng(p.seed); const int vt = sg.latent_t, hh = sg.lat_h, ww = sg.lat_w;
                        for (int c = 0; c < LATENT_CH; ++c) for (int t = 0; t < vt; ++t) for (int y = 0; y < hh; ++y) for (int x = 0; x < ww; ++x) {
                            const size_t row = (size_t(t) * (hh / 2) + y / 2) * (ww / 2) + x / 2, col = size_t(c) * 4 + (y % 2) * 2 + (x % 2);
                            const float z = rf.video_latent[((size_t(c) * vt + t) * hh + y) * ww + x];
                            in32[row * VIDEO_PATCH + col] = VISUAL_COND_AUG * z + (1.0f - VISUAL_COND_AUG) * arng.normal(); }
                        rt().h2d(in32_, in32.data(), sg.rows * VIDEO_PATCH * 4); embed_f32("ref video in", sg.row0, sg.rows, VIDEO_PATCH, "h3.video_in");
                    }
                }
                if (!text_copy_) text_copy_ = memory_.alloc(seq_cap_ * HID * 4); rt().d2d(text_copy_, x_, LR * HID * 4);
            } else rt().d2d(x_, text_copy_, LR * HID * 4);
            h_embed = hnow();
            if (p.cache_threshold > 0.0f) {                                          // first-block cache (TeaCache / FBCache style)
                const size_t n = S * size_t(HID), groups = (n + 2047) / 2048;
                if (cache_cap_ < n) {
                    cache_cap_ = 0;
                    for (void **q : {&xb0_, &prev_b0_, &cache_resid_, &partials_}) { memory_.free(*q); *q = nullptr; }
                    xb0_ = memory_.alloc(n * 4); prev_b0_ = memory_.alloc(n * 4); cache_resid_ = memory_.alloc(n * 4); partials_ = memory_.alloc(groups * 8); cache_cap_ = n;
                    absdiff_ = comp_.get("absdiff_sum_f32", "h3_absdiff_sum_f32", {});
                }
                dit_->forward(&prof, x_, cls_, cos_, sin_, [&](int i) { return cond(i); }, 0, 1);
                bool skip = false;
                if (step > 0) {
                    KernArgs a; a.i32(int(n)).ptr(x_).ptr(prev_b0_).ptr(partials_); launch(*absdiff_, &prof, "cache metric", unsigned(groups), 1, THREADS, a);
                    std::vector<float> ps(groups * 2); rt().sync(); rt().d2h(ps.data(), partials_, ps.size() * 4);
                    double d = 0, m = 0; for (size_t i = 0; i < groups; ++i) { d += ps[2 * i]; m += ps[2 * i + 1]; }
                    const double rel = d / std::max(m, 1e-30); cache_acc_ += rel;
                    skip = cache_acc_ < p.cache_threshold && have_cache_;
                    if (std::getenv("H3_CACHE_TRACE")) fprintf(stderr, "  step %zu: block-0 change %.4f, accumulated %.4f -> %s\n", step + 1, rel, cache_acc_, skip ? "cached" : "full");
                }
                rt().d2d(prev_b0_, x_, n * 4);
                if (skip) { axpy(1.0f, 1.0f, n, cache_resid_, x_); ++cache_skipped_; }
                else {
                    rt().d2d(xb0_, x_, n * 4);
                    dit_->forward(&prof, x_, cls_, cos_, sin_, [&](int i) { return cond(i); }, 1, 50);
                    rt().d2d(cache_resid_, x_, n * 4); axpy(-1.0f, 1.0f, n, xb0_, cache_resid_);   // residual of blocks 1..49
                    have_cache_ = true; cache_acc_ = 0.0;
                }
            } else dit_->forward(&prof, x_, cls_, cos_, sin_, [&](int i) { return cond(i); });
            if (prof.on) rt().sync(); h_dit = hnow();
            // the final layer: the modulated RMSNorm in place on the generated rows (re-embedded next step), then the two f32 heads
            { KernArgs a; a.i32(int(generated_rows)).ptr((float *)x_ + LR * HID).ptr(dit_w_.at("h3.final.norm", size_t(HID) * 4)).ptr(final_table_).ptr((int32_t *)tcls_ + LR);
              launch(*final_norm_, &prof, "final norm", unsigned(generated_rows), 1, unsigned(lanes_for(HID)), a); }
            matmul("final out", generated_rows, HID, FINAL_N, (float *)x_ + LR * HID, dit_w_.at("h3.final.out.w", size_t(FINAL_N) * HID * 4), dit_w_.at("h3.final.out.b", size_t(FINAL_N) * 4), out32_);
            rt().sync(); rt().d2h(out32.data(), out32_, generated_rows * FINAL_N * 4);
            h_final = hnow();
            const float sg_v = sv.sigmas[step], sg_next = sv.sigmas[step + 1], r_v = sg_next / sg_v, sg_a = sa.sigmas[step], r_a = sa.sigmas[step + 1] / sg_a;
            if (!res) {
                // Euler step per schedule: x0 = x + sigma * v, x' = r x + (1 - r) x0
                for (size_t r = 0; r < Nv; ++r) for (int k = 0; k < VIDEO_PATCH; ++k) { float &x = vrows[r * VIDEO_PATCH + k]; const float v = out32[(Na + r) * FINAL_N + k]; x = r_v * x + (1.0f - r_v) * (x + sg_v * v); }
                for (size_t r = 0; r < Na; ++r) for (int k = 0; k < AUDIO_CH; ++k) { float &x = arows[r * AUDIO_CH + k]; const float v = out32[r * FINAL_N + VIDEO_PATCH + k]; x = r_a * x + (1.0f - r_a) * (x + sg_a * v); }
            } else {
                // ComfyUI (comfy/k_diffusion/sampling.py res_multistep, eta 0): denoised D = X - sigma_v * OUT over the pack. The video's OUT is
                // -v. The audio's carried variable y = (sigma_v / sigma_a) x_a sees OUT_a = (1 - scale) x_a + (1 + (scale - 1) sigma_a) (-v_a),
                // scale = shift_v / shift_a (comfy/ldm/minimax/model.py forward); the network itself sees x_a and t_a = 1 - sigma_a.
                den_v.resize(vrows.size()); den_a.resize(arows.size());
                for (size_t i = 0; i < vrows.size(); ++i) { const size_t r = i / VIDEO_PATCH, k = i % VIDEO_PATCH; den_v[i] = vrows[i] + sg_v * out32[(Na + r) * FINAL_N + k]; }
                for (size_t i = 0; i < arows.size(); ++i) { const size_t r = i / AUDIO_CH, k = i % AUDIO_CH; const float v = out32[r * FINAL_N + VIDEO_PATCH + k];
                    const double out = (1.0 - ascale) * arows[i] - (1.0 + (ascale - 1.0) * sg_a) * v; den_a[i] = float(yrows[i] - sg_v * out); }
                auto advance = [&](std::vector<float> &x, const std::vector<float> &d, const std::vector<float> &od) {
                    if (sg_next == 0.0f || od.empty()) { for (size_t i = 0; i < x.size(); ++i) x[i] = r_v * x[i] + (1.0f - r_v) * d[i]; }   // Euler: x + (x - D) / sigma * (sigma_next - sigma)
                    else {   // second-order multistep (arXiv 2308.02157) in t = -log sigma with the previous denoised
                        const double t = -std::log(double(sg_v)), t_next = -std::log(double(sg_next)), t_prev = -std::log(double(sv.sigmas[step - 1]));
                        const double h = t_next - t, c2 = (t_prev - t) / h, phi1 = std::expm1(-h) / (-h), phi2 = (phi1 - 1.0) / (-h);
                        const double b1 = phi1 - phi2 / c2, b2 = phi2 / c2, decay = std::exp(-h);
                        for (size_t i = 0; i < x.size(); ++i) x[i] = float(decay * x[i] + h * (b1 * d[i] + b2 * od[i]));
                    }
                };
                advance(vrows, den_v, old_v); advance(yrows, den_a, old_a); old_v.swap(den_v); old_a.swap(den_a);
            }
            dump(step + 1);
            if (prof.on) fprintf(stderr, "  step %zu host phases: mods %.2fs  embed %.2fs  blocks %.2fs  final+d2h %.2fs  euler %.2fs\n", step + 1, h_mods - h0, h_embed - h_mods, h_dit - h_embed, h_final - h_dit, hnow() - h_final);
            if (prof.on) { double tot = 0; for (auto &kv : prof.us) tot += kv.second; fprintf(stderr, "  step %zu stages (%.1f s):", step + 1, tot * 1e-6); for (auto &kv : prof.us) if (kv.second > 0.01 * tot) fprintf(stderr, "  %s %.2fs", kv.first.c_str(), kv.second * 1e-6); fprintf(stderr, "\n"); prof.us.clear(); }
            if (progress && progress(user, int(step + 1), int(sv.timesteps.size()), std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count())) throw Cancelled();
        }
        if (p.cache_threshold > 0.0f && std::getenv("H3_CACHE_TRACE")) fprintf(stderr, "  step cache: %d of %zu evaluations skipped\n", cache_skipped_, sv.timesteps.size());
        rows_to_tensor(video_out);
        for (int c = 0; c < 2; ++c) for (int t = 0; t < A; ++t) for (int k = 0; k < AUDIO_CH; ++k) audio_out[(size_t(c) * AUDIO_CH + k) * A + t] = res ? float(yrows[(size_t(c) * A + t) * AUDIO_CH + k] / ascale) : arows[(size_t(c) * A + t) * AUDIO_CH + k];   // the carried variable ends at scale * x_a
    }
    struct Cancelled {};

    // --- the video VAE encoder: causal 3-D convs as implicit GEMMs on channels-last f16, GroupNorm+SiLU, 256-px tiles blended
    // in latent space and 17-frame chunks as ComfyUI's tiled_encode / encode_temporal ---
    struct VConv { std::shared_ptr<Kernel> k; int cout_pad, tout, ho, wo; };
    VConv vconv_kernel(bool add, int frames, int H, int W, int stride, int tstride, int taps_t, int cin_pad, int cin_stride, int k_size, int cout_pad) {
        const std::string stem = add ? "conv3d_f16_wmma_add" : "conv3d_f16_wmma", ns = "h3." + stem + ".";
        const size_t rows = size_t(frames) * H * W, rb = (rows + 63) / 64 * 64;
        VConv c; c.k = comp_.get(stem, "h3_" + stem, {{ns + "frames", std::to_string(frames)}, {ns + "height", std::to_string(H)}, {ns + "width", std::to_string(W)}, {ns + "stride", std::to_string(stride)}, {ns + "tstride", std::to_string(tstride)},
                                               {ns + "taps_t", std::to_string(taps_t)}, {ns + "cin_pad", std::to_string(cin_pad)}, {ns + "cin_stride", std::to_string(cin_stride)}, {ns + "rows_bound", std::to_string(rb)}, {ns + "k_size", std::to_string(k_size)}, {ns + "n_size", std::to_string(cout_pad)}});
        c.cout_pad = cout_pad; c.tout = (frames - 1) / tstride + 1; c.ho = H / stride; c.wo = W / stride; return c;
    }
    void vconv_run(const VConv &c, const char *stage, const void *a, const void *w, const void *b, void *out, const void *residual = nullptr) {
        const size_t m = size_t(c.tout) * c.ho * c.wo; KernArgs args; args.i32(int(m)).ptr(a).ptr(w).ptr(b).ptr(out); if (residual) args.ptr(residual);
        launch(*c.k, &prof, stage, unsigned(c.cout_pad / 64), unsigned((m + 63) / 64), 256, args);
    }
    void gn_silu(int frames, int H, int W, int channels, const void *x, const void *gamma, const void *beta, void *stats, void *out) {
        const size_t plane = size_t(H) * W, rows = size_t(frames) * plane, rb = (rows + 63) / 64 * 64;
        const std::string ns = "h3.gn_stats_f16.", na = "h3.gn_silu_f16.";
        auto ks = comp_.get("gn_stats_f16", "h3_gn_stats_f16", {{ns + "channels", std::to_string(channels)}, {ns + "groups", "32"}, {ns + "plane", std::to_string(plane)}, {ns + "rows_bound", std::to_string(rb)}});
        auto ka = comp_.get("gn_silu_f16", "h3_gn_silu_f16", {{na + "channels", std::to_string(channels)}, {na + "groups", "32"}, {na + "plane", std::to_string(plane)}, {na + "rows_bound", std::to_string(rb)}, {na + "eps", num(1e-6)}});
        { KernArgs a; a.i32(frames).ptr(x).ptr(stats); launch(*ks, &prof, "venc groupnorm", unsigned(frames), 32, 32, a); }
        { KernArgs a; a.i32(frames).ptr(x).ptr(stats).ptr(gamma).ptr(beta).ptr(out); launch(*ka, &prof, "venc groupnorm", unsigned((rows * channels + 255) / 256), 1, 256, a); }
    }
    // one tile (or whole image) of `frames` frames -> moments rows [T_lat*h*w][64] f16 on the device (channels 0..23 the mean)
    void venc_tile(const float *pixels, int frames, int H, int W, int y0, int x0, int TH, int TW, int fullW, std::vector<float> &latent, int &T_lat) {
        auto Wt = [&](const std::string &nm, size_t bytes) { return vvae_w_.at(nm, bytes); };
        DeviceBuffers temporary; auto buf = [&](size_t bytes) { return temporary.alloc(bytes); };
        const bool image = frames == 1; const int taps_t = image ? 1 : 3;
        // input rows [frames*TH*TW][8] f16: ImageNet-normalised pixels, channels 3..7 zero
        const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
        std::vector<uint16_t> in(size_t(frames) * TH * TW * 8, 0);
        for (int t = 0; t < frames; ++t) for (int y = 0; y < TH; ++y) for (int x = 0; x < TW; ++x) for (int c = 0; c < 3; ++c)
            in[((size_t(t) * TH + y) * TW + x) * 8 + c] = f32_to_f16((pixels[((size_t(t) * H + y0 + y) * fullW + x0 + x) * 3 + c] - mean[c]) / stdv[c]);
        const size_t rows0 = size_t(frames) * TH * TW;
        void *x = buf(rows0 * 8 * 2); rt().h2d(x, in.data(), in.size() * 2);
        void *h = buf(rows0 * 128 * 2), *y = buf(rows0 * 128 * 2), *tmp = buf(rows0 * 128 * 2), *sc = buf(rows0 * 128 * 2), *stats = buf(size_t(frames) * 32 * 2 * 4);
        auto W3 = [&](const std::string &nm, int cout_pad, int k) { return Wt(nm + (image ? ".w2" : ".w3"), size_t(cout_pad) * k * 2); };
        auto ksz = [&](int cin_pad) { return (int)(((image ? 9 : 27) * cin_pad + 31) / 32 * 32); };
        // conv_in: 3 -> 128
        int T = frames, th = TH, tw = TW, C = 128;
        { VConv c = vconv_kernel(false, T, th, tw, 1, 1, taps_t, 8, 8, ksz(8), 128); vconv_run(c, "venc conv_in", x, W3("venc.conv_in", 128, ksz(8)), Wt("venc.conv_in.b", 128 * 4), h); }
        static const int mid[6] = {128, 256, 256, 512, 512, 1024}, sdown[6] = {2, 2, 2, 2, 1, 1}, tdown[6] = {1, 2, 2, 1, 1, 1};
        for (int l = 0; l < 6; ++l) {
            for (int r = 0; r < 2; ++r) {
                const std::string b = fmt("venc.l%d.r%d.", l, r); const int cin = C, cout = mid[l];
                gn_silu(T, th, tw, cin, h, Wt(b + "norm1.g", cin * 4), Wt(b + "norm1.b", cin * 4), stats, y);
                VConv c1 = vconv_kernel(false, T, th, tw, 1, 1, taps_t, cin, cin, ksz(cin), cout); vconv_run(c1, "venc conv", y, W3(b + "conv1", cout, ksz(cin)), Wt(b + "conv1.b", cout * 4), tmp);
                gn_silu(T, th, tw, cout, tmp, Wt(b + "norm2.g", cout * 4), Wt(b + "norm2.b", cout * 4), stats, y);
                const void *resid = h;
                if (cin != cout) {   // 1x1 shortcut: a matmul over channels
                    const int k = (cin + 31) / 32 * 32; const std::string stem = "matmul_bias_f16_wmma_af16_cf16", ns = "h3." + stem + ".";
                    auto kk = comp_.get(stem, "h3_" + stem, {{ns + "k_size", std::to_string(k)}, {ns + "n_size", std::to_string(cout)}});
                    const size_t m = size_t(T) * th * tw; KernArgs a; a.i32(int(m)).ptr(h).ptr(Wt(b + "nin.wm", size_t(cout) * k * 2)).ptr(Wt(b + "nin.b", cout * 4)).ptr(sc);
                    launch(*kk, &prof, "venc shortcut", unsigned(cout / 64), unsigned((m + 63) / 64), 256, a); resid = sc;
                }
                VConv c2 = vconv_kernel(true, T, th, tw, 1, 1, taps_t, cout, cout, ksz(cout), cout); vconv_run(c2, "venc conv", y, W3(b + "conv2", cout, ksz(cout)), Wt(b + "conv2.b", cout * 4), tmp, resid);
                std::swap(h, tmp); C = cout;
            }
            if (sdown[l] * tdown[l] > 1) {
                const std::string b = fmt("venc.l%d.down", l);
                VConv c = vconv_kernel(false, T, th, tw, sdown[l], image ? 1 : tdown[l], taps_t, C, C, ksz(C), C); vconv_run(c, "venc down", h, W3(b, C, ksz(C)), Wt(b + ".b", C * 4), tmp);
                std::swap(h, tmp); T = c.tout; th = c.ho; tw = c.wo;
            }
        }
        gn_silu(T, th, tw, C, h, Wt("venc.norm_out.g", C * 4), Wt("venc.norm_out.b", C * 4), stats, y);
        { VConv c = vconv_kernel(false, T, th, tw, 1, 1, taps_t, C, C, ksz(C), 64); vconv_run(c, "venc conv_out", y, W3("venc.conv_out", 64, ksz(C)), Wt("venc.conv_out.b", 64 * 4), tmp); }
        { const std::string stem = "matmul_bias_f16_wmma_af16_cf16", ns = "h3." + stem + ".";
          auto kk = comp_.get(stem, "h3_" + stem, {{ns + "k_size", "64"}, {ns + "n_size", "64"}});
          const size_t m = size_t(T) * th * tw; KernArgs a; a.i32(int(m)).ptr(tmp).ptr(Wt("venc.quant.wm", size_t(64) * 64 * 2)).ptr(Wt("venc.quant.b", 64 * 4)).ptr(y);
          launch(*kk, &prof, "venc quant", 1, unsigned((m + 63) / 64), 256, a); }
        const size_t m = size_t(T) * th * tw; std::vector<uint16_t> mom(m * 64); rt().sync(); rt().d2h(mom.data(), y, mom.size() * 2);
        const std::vector<float> lmean = vvae_w_.host_f32("venc.latents_mean", 24), lstd = vvae_w_.host_f32("venc.latents_std", 24);
        T_lat = T; latent.assign(size_t(24) * T * th * tw, 0.0f);
        for (int t = 0; t < T; ++t) for (int yy = 0; yy < th; ++yy) for (int xx = 0; xx < tw; ++xx) for (int c = 0; c < 24; ++c)
            latent[((size_t(c) * T + t) * th + yy) * tw + xx] = (f16_to_f32(mom[((size_t(t) * th + yy) * tw + xx) * 64 + c]) - lmean[c]) / lstd[c];

    }
    static void split_tiles(int len, std::vector<int> &starts, std::vector<int> &overlaps) {   // ComfyUI's split_tiles: 256-px tiles, overlaps >= 64 in 16-px units
        starts.clear(); overlaps.clear(); const int tile = 256, omin = 64, ratio = 16;
        if (tile >= len) { starts = {0}; return; }
        int N = (len + tile - 1) / tile;
        while (true) { overlaps.assign(N - 1, omin); const int remaining = tile * N - omin * (N - 1) - len; if (remaining < 0) ++N; else { for (int i = 0; i < remaining / ratio; ++i) overlaps[i % (N - 1)] += ratio; break; } }
        starts = {0}; for (int i = 0; i < N - 1; ++i) starts.push_back(starts.back() + tile - overlaps[i]);
    }
    // one clip of `frames` frames (an image when frames == 1): tiled encode with linear latent blends across the overlaps
    void venc_clip(const float *pixels, int frames, int H, int W, std::vector<float> &out, int &T_lat) {
        std::vector<int> ys, yo, xs, xo; split_tiles(H, ys, yo); split_tiles(W, xs, xo);
        const int ny = int(ys.size()), nx = int(xs.size()); std::vector<std::vector<float>> tiles(size_t(ny) * nx); std::vector<int> th(ny), tw(nx); int T = 0;
        for (int i = 0; i < ny; ++i) for (int j = 0; j < nx; ++j) {
            const int TH = ny == 1 ? H : 256, TW = nx == 1 ? W : 256; th[i] = TH / 16; tw[j] = TW / 16;
            venc_tile(pixels, frames, H, W, ys[i], xs[j], TH, TW, W, tiles[size_t(i) * nx + j], T);
        }
        // ComfyUI's tiled_encode, literally: for tile (i, j), blend(raw above, tile) over the y overlap (the first ov_y rows of the
        // tile become the linear blend of the upper tile's last ov_y rows and its own), then blend(raw left, tile) over the x overlap
        // on the result, crop the trailing overlaps, concatenate. Latent tiles are [24][T][h][w].
        const int LH = H / 16, LW = W / 16; out.assign(size_t(24) * T * LH * LW, 0.0f);
        auto blend = [&](const std::vector<float> &av, int ah, int aw, const std::vector<float> &bv, int bh, int bw, int ext, bool ydim) {
            std::vector<float> r(bv); const int e = std::min(ext, ydim ? std::min(ah, bh) : std::min(aw, bw));
            for (int c = 0; c < 24; ++c) for (int t = 0; t < T; ++t) for (int y = 0; y < bh; ++y) for (int x = 0; x < bw; ++x) {
                const int k = ydim ? y : x; if (k >= e) continue; const float wb = float(k) / e, wa = 1.0f - wb;
                const int ay = ydim ? ah - e + y : y, ax = ydim ? x : aw - e + x;
                r[((size_t(c) * T + t) * bh + y) * bw + x] = wa * av[((size_t(c) * T + t) * ah + ay) * aw + ax] + wb * bv[((size_t(c) * T + t) * bh + y) * bw + x]; }
            return r; };
        int oy = 0;
        for (int i = 0; i < ny; ++i) {
            int ox = 0; const int h = th[i];
            for (int j = 0; j < nx; ++j) {
                const int w = tw[j]; std::vector<float> tile = tiles[size_t(i) * nx + j];
                if (i > 0) tile = blend(tiles[size_t(i - 1) * nx + j], th[i - 1], w, tile, h, w, yo[i - 1] / 16, true);
                if (j > 0) tile = blend(tiles[size_t(i) * nx + j - 1], h, tw[j - 1], tile, h, w, xo[j - 1] / 16, false);
                const int keep_y = i < ny - 1 ? h - yo[i] / 16 : h, keep_x = j < nx - 1 ? w - xo[j] / 16 : w;
                for (int c = 0; c < 24; ++c) for (int t = 0; t < T; ++t) for (int y = 0; y < keep_y; ++y) for (int x = 0; x < keep_x; ++x)
                    out[((size_t(c) * T + t) * LH + oy + y) * LW + ox + x] = tile[((size_t(c) * T + t) * h + y) * w + x];
                ox += keep_x;
            }
            oy += (i < ny - 1 ? h - yo[i] / 16 : h);
        }
        T_lat = T;
    }
    void encode_video(const float *pixels, int frames, int H, int W, float *latents, int &latent_t) {
        ensure_vvae();
        const int LH = H / 16, LW = W / 16;
        if (frames == 1) { std::vector<float> z; int T = 0; venc_clip(pixels, 1, H, W, z, T); if (T != 1) throw std::runtime_error("image encode produced more than one latent frame"); memcpy(latents, z.data(), z.size() * 4); latent_t = 1; return; }
        const int chunks = (frames + 16) / 17, TL = chunks * 5; std::vector<float> all(size_t(24) * TL * LH * LW); std::vector<float> clip(size_t(17) * H * W * 3);
        for (int ch = 0; ch < chunks; ++ch) {
            for (int f = 0; f < 17; ++f) { const int src = std::min(ch * 17 + f, frames - 1); memcpy(clip.data() + size_t(f) * H * W * 3, pixels + size_t(src) * H * W * 3, size_t(H) * W * 3 * 4); }
            std::vector<float> z; int T = 0; venc_clip(clip.data(), 17, H, W, z, T); if (T != 5) throw std::runtime_error("a 17-frame chunk must give 5 latent frames");
            for (int c = 0; c < 24; ++c) for (int t = 0; t < 5; ++t) memcpy(all.data() + ((size_t(c) * TL + ch * 5 + t) * LH) * LW, z.data() + ((size_t(c) * 5 + t) * LH) * LW, size_t(LH) * LW * 4);
        }
        latent_t = TL - 3;   // token_drop
        for (int c = 0; c < 24; ++c) for (int t = 0; t < latent_t; ++t) memcpy(latents + ((size_t(c) * latent_t + t) * LH) * LW, all.data() + ((size_t(c) * TL + t) * LH) * LW, size_t(LH) * LW * 4);
    }

    // --- the vision tower: Qwen3-VL's ViT (27 blocks, hidden 1152, 16 heads of 72 padded to 128) in f16-weight WMMA GEMMs on
    // an f16 residual stream, the H3 f16 attention per image, the patch merger and the DeepStack mergers ---
    static constexpr int VHID = 1152, VHEADS = 16, VHD = 72, VHDP = 128, VMLP = 4352, VOUT = 5120, VBLOCKS = 27;
    std::shared_ptr<Kernel> f16gemm(const char *kind, int k, int n) {
        const std::string stem = std::string("matmul_") + kind + "_bf16_wmma", ns = "h3." + stem + ".";   // the tower's rows are bf16 as stored
        return comp_.get(stem, "h3_" + stem, {{ns + "k_size", std::to_string(k)}, {ns + "n_size", std::to_string(n)}});
    }
    void gemm16(const char *kind, const char *stage, size_t m, int k, int n, const void *a, const void *w, const void *bias, void *c, const void *lambda = nullptr) {
        auto kern = f16gemm(kind, k, n); KernArgs args; args.i32(int(m)).ptr(a).ptr(w).ptr(bias).ptr(c); if (lambda) args.ptr(lambda);
        launch(*kern, &prof, stage, unsigned(n / 64), unsigned((m + 63) / 64), 256, args);
    }
    void layernorm16(size_t rows, int width, const void *x16, const void *w, const void *b, void *out32) {
        const std::string ns = "h3.layernorm_f16_f32.";
        auto k = comp_.get("layernorm_f16_f32", "h3_layernorm_f16_f32", {{ns + "width", std::to_string(width)}, {ns + "eps", num(1e-6)}});
        KernArgs a; a.i32(int(rows)).ptr(x16).ptr(w).ptr(b).ptr(out32); launch(*k, &prof, "vision layernorm", unsigned(rows), 1, 32, a);
    }
    // pixels [H][W][3] in [0, 1] -> merged [n/4][5120], deepstack [3][n/4][5120]; n = (H/16) * (W/16) patches in 2x2 merge order
    void vision_embed(const float *pixels, int H, int W, std::vector<float> &merged, std::vector<float> &deepstack) {
        ensure_te();
        if (H % 32 || W % 32 || H < 32 || W < 32) throw std::invalid_argument("vision images need height and width multiples of 32");
        const int gh = H / 16, gw = W / 16, n = gh * gw, m = n / 4; const size_t cap = (size_t(n) + 16 + 31) / 32 * 32;
        auto V = [&](const std::string &nm, size_t bytes) { return te_w_.at(nm, bytes); };
        DeviceBuffers temporary; auto buf = [&](size_t bytes) { return temporary.alloc(bytes); };
        // patches in merge order, content (c, t, py, px) with the image in both temporal slots, CLIP's mean/std (process_qwen2vl_images)
        static const float pmean[3] = {0.48145466f, 0.4578275f, 0.40821073f}, pstd[3] = {0.26862954f, 0.26130258f, 0.27577711f};
        std::vector<float> patches(size_t(n) * 1536);
        for (int bh = 0; bh < gh / 2; ++bh) for (int bw = 0; bw < gw / 2; ++bw) for (int ih = 0; ih < 2; ++ih) for (int iw = 0; iw < 2; ++iw) {
            const size_t pi = ((size_t(bh) * (gw / 2) + bw) * 2 + ih) * 2 + iw; float *dst = patches.data() + pi * 1536;
            for (int c = 0; c < 3; ++c) for (int t = 0; t < 2; ++t) for (int py = 0; py < 16; ++py) for (int px = 0; px < 16; ++px) {
                const int y = (bh * 2 + ih) * 16 + py, x = (bw * 2 + iw) * 16 + px;
                dst[((c * 2 + t) * 16 + py) * 16 + px] = (pixels[(size_t(y) * W + x) * 3 + c] - pmean[c]) / pstd[c]; } }
        static const bool vdebug = std::getenv("H3_VISION_DEBUG") && *std::getenv("H3_VISION_DEBUG");
        auto probe = [&](const char *what, const void *ptr, size_t count, bool half) { if (!vdebug) return; rt().sync(); std::vector<float> v(count);
            if (half) { std::vector<uint16_t> h(count); rt().d2h(h.data(), ptr, count * 2); for (size_t i = 0; i < count; ++i) v[i] = f16_to_f32(h[i]); } else rt().d2h(v.data(), ptr, count * 4);
            size_t bad = 0; double mx = 0; for (float f : v) { if (!std::isfinite(f)) ++bad; else mx = std::max(mx, double(std::fabs(f))); }
            fprintf(stderr, "  vision %-18s non-finite %zu / %zu  max|x| %.3g\n", what, bad, count, mx); };
        void *pa = buf(patches.size() * 4); rt().h2d(pa, patches.data(), patches.size() * 4);
        void *x32 = buf(size_t(n) * VHID * 4); gemm16("bias", "vision patch", n, 1536, VHID, pa, V("vis.patch.w", size_t(VHID) * 1536 * 2), V("vis.patch.b", VHID * 4), x32);
        // the learned 48x48 position table, bilinearly resampled to the grid then permuted into merge order (fast_pos_embed_interpolate)
        std::vector<float> pos(te_w_.host_f32("vis.pos", size_t(2304) * VHID)), x0(size_t(n) * VHID); rt().d2h(x0.data(), x32, x0.size() * 4);
        const int G = 48;
        for (int hy = 0; hy < gh; ++hy) for (int wx = 0; wx < gw; ++wx) {
            const float fh = gh == 1 ? 0.0f : float(hy) * (G - 1) / float(gh - 1), fw = gw == 1 ? 0.0f : float(wx) * (G - 1) / float(gw - 1);
            const int h0 = int(fh), w0 = int(fw), h1 = std::min(h0 + 1, G - 1), w1 = std::min(w0 + 1, G - 1); const float dh = fh - h0, dw = fw - w0;
            const size_t pi = ((size_t(hy / 2) * (gw / 2) + wx / 2) * 2 + hy % 2) * 2 + wx % 2; float *row = x0.data() + pi * VHID;
            const float *r00 = pos.data() + (size_t(h0) * G + w0) * VHID, *r01 = pos.data() + (size_t(h0) * G + w1) * VHID, *r10 = pos.data() + (size_t(h1) * G + w0) * VHID, *r11 = pos.data() + (size_t(h1) * G + w1) * VHID;
            for (int c = 0; c < VHID; ++c) row[c] += (1 - dh) * (1 - dw) * r00[c] + (1 - dh) * dw * r01[c] + dh * (1 - dw) * r10[c] + dh * dw * r11[c]; }
        probe("patch gemm", x32, x0.size(), false);
        std::vector<uint16_t> x16h(x0.size()); for (size_t i = 0; i < x0.size(); ++i) x16h[i] = f32_to_f16(x0[i]);
        void *x16 = buf(x16h.size() * 2); rt().h2d(x16, x16h.data(), x16h.size() * 2);
        // 2-D rope tables: pair j < 36 -> h with inv_freq[j], j >= 36 -> w with inv_freq[j - 36]; theta 10000 over dim 36
        std::vector<float> cosv(size_t(n) * 36), sinv(size_t(n) * 36);
        for (int hy = 0; hy < gh; ++hy) for (int wx = 0; wx < gw; ++wx) { const size_t pi = ((size_t(hy / 2) * (gw / 2) + wx / 2) * 2 + hy % 2) * 2 + wx % 2;
            for (int j = 0; j < 36; ++j) { const double inv = std::pow(10000.0, -double(2 * (j % 18)) / 36.0), ang = double(j < 18 ? hy : wx) * inv; cosv[pi * 36 + j] = float(std::cos(ang)); sinv[pi * 36 + j] = float(std::sin(ang)); } }
        void *cosb = buf(cosv.size() * 4), *sinb = buf(sinv.size() * 4); rt().h2d(cosb, cosv.data(), cosv.size() * 4); rt().h2d(sinb, sinv.data(), sinv.size() * 4);
        void *ln = buf(size_t(n) * VHID * 4), *qkv = buf(size_t(n) * 3 * VHEADS * VHD * 4), *q16 = buf(cap * VHEADS * VHDP * 2), *k16 = buf(cap * VHEADS * VHDP * 2), *v16 = buf(cap * VHEADS * VHDP * 2);
        void *att16 = buf(cap * VHEADS * VHDP * 2), *hid = buf(size_t(n) * VMLP * 4), *ln4 = buf(size_t(m) * 4608 * 4), *mid = buf(size_t(m) * 4608 * 4), *out5 = buf(size_t(m) * VOUT * 4);
        std::vector<float> ones(4608, 1.0f); void *lam = buf(ones.size() * 4); rt().h2d(lam, ones.data(), ones.size() * 4);
        rt().memset(k16, 0, cap * VHEADS * VHDP * 2); rt().memset(v16, 0, cap * VHEADS * VHDP * 2); rt().memset(q16, 0, cap * VHEADS * VHDP * 2);
        const std::string ans = "h3.attention_mha_lds_f16_wmma.";
        auto attn = comp_.get("attention_mha_lds_f16_wmma", "h3_attention_mha_lds_f16_wmma", {{ans + "q_stride", std::to_string(VHEADS * VHDP)}, {ans + "kv_stride", std::to_string(VHEADS * VHDP)}, {ans + "tokens", std::to_string(n)}, {ans + "token_capacity", std::to_string(cap)}, {ans + "scale", num(1.0 / std::sqrt(double(VHD)))}, {ans + "out_stride", std::to_string(VHEADS * VHDP)}});
        const std::string rns = "h3.rope2d_qkv_f16.";
        auto rope = comp_.get("rope2d_qkv_f16", "h3_rope2d_qkv_f16", {{rns + "heads", std::to_string(VHEADS)}, {rns + "hd", std::to_string(VHD)}, {rns + "hd_pad", std::to_string(VHDP)}});
        auto cast = comp_.get("cast_f32_f16", "h3_cast_f32_f16", {});
        void *hid16 = buf(size_t(n) * VMLP * 2);
        merged.assign(size_t(m) * VOUT, 0.0f); deepstack.assign(size_t(3) * m * VOUT, 0.0f);
        static const int DS[3] = {8, 16, 24};
        for (int i = 0; i < VBLOCKS; ++i) {
            const std::string b = fmt("vis.b%d.", i);
            layernorm16(n, VHID, x16, V(b + "norm1.w", VHID * 4), V(b + "norm1.b", VHID * 4), ln);
            gemm16("bias", "vision qkv", n, VHID, 3 * VHEADS * VHD, ln, V(b + "qkv.w", size_t(3 * VHEADS * VHD) * VHID * 2), V(b + "qkv.b", size_t(3 * VHEADS * VHD) * 4), qkv);
            { KernArgs a; a.i32(n).ptr(qkv).ptr(cosb).ptr(sinb).ptr(q16).ptr(k16).ptr(v16); launch(*rope, &prof, "vision rope", unsigned(n), unsigned(VHEADS), unsigned(VHDP), a); }
            { KernArgs a; a.i32(n).ptr(q16).ptr(k16).ptr(v16).ptr(att16); launch(*attn, &prof, "vision attention", unsigned((n + 63) / 64), unsigned(VHEADS), 128, a); }
            gemm16("resid", "vision proj", n, VHEADS * VHDP, VHID, att16, V(b + "proj.w", size_t(VHID) * VHEADS * VHDP * 2), V(b + "proj.b", VHID * 4), x16, lam);   // the residual GEMM takes A as f16
            layernorm16(n, VHID, x16, V(b + "norm2.w", VHID * 4), V(b + "norm2.b", VHID * 4), ln);
            gemm16("gelu", "vision fc1", n, VHID, VMLP, ln, V(b + "fc1.w", size_t(VMLP) * VHID * 2), V(b + "fc1.b", VMLP * 4), hid);
            { KernArgs a; a.i32(n * VMLP).ptr(hid).ptr(hid16); launch(*cast, &prof, "vision cast", unsigned((size_t(n) * VMLP + 255) / 256), 1, 256, a); }
            gemm16("resid", "vision fc2", n, VMLP, VHID, hid16, V(b + "fc2.w", size_t(VHID) * VMLP * 2), V(b + "fc2.b", VHID * 4), x16, lam);
            if (i < 2 || i == VBLOCKS - 1) { probe(fmt("b%d ln1", i).c_str(), ln, size_t(n) * VHID, false); probe(fmt("b%d qkv", i).c_str(), qkv, size_t(n) * 3 * VHEADS * VHD, false); probe(fmt("b%d attn", i).c_str(), att16, size_t(n) * VHEADS * VHDP, true); probe(fmt("b%d fc1", i).c_str(), hid, size_t(n) * VMLP, false); probe(fmt("b%d x", i).c_str(), x16, size_t(n) * VHID, true); }
            for (int j = 0; j < 3; ++j) if (i == DS[j]) {
                const std::string d = fmt("vis.ds%d.", j);
                layernorm16(m, 4608, x16, V(d + "norm.w", 4608 * 4), V(d + "norm.b", 4608 * 4), ln4);
                gemm16("gelu_erf", "vision deepstack", m, 4608, 4608, ln4, V(d + "fc1.w", size_t(4608) * 4608 * 2), V(d + "fc1.b", 4608 * 4), mid);
                gemm16("bias", "vision deepstack", m, 4608, VOUT, mid, V(d + "fc2.w", size_t(VOUT) * 4608 * 2), V(d + "fc2.b", VOUT * 4), out5);
                rt().d2h(deepstack.data() + size_t(j) * m * VOUT, out5, size_t(m) * VOUT * 4);
            }
        }
        layernorm16(n, VHID, x16, V("vis.merger.norm.w", VHID * 4), V("vis.merger.norm.b", VHID * 4), ln);   // [n][1152] -> viewed [m][4608]
        gemm16("gelu_erf", "vision merger", m, 4608, 4608, ln, V("vis.merger.fc1.w", size_t(4608) * 4608 * 2), V("vis.merger.fc1.b", 4608 * 4), mid);
        gemm16("bias", "vision merger", m, 4608, VOUT, mid, V("vis.merger.fc2.w", size_t(VOUT) * 4608 * 2), V("vis.merger.fc2.b", VOUT * 4), out5);
        rt().d2h(merged.data(), out5, merged.size() * 4);
        rt().sync();
    }

    // --- the audio encoder: the audio VAE's DAC conv stack and posterior head, f32 SIMT Loom kernels (reference audio) ---
    static size_t pow2_bound(size_t n) { size_t b = 256; while (b < n) b *= 2; return b; }
    std::shared_ptr<Kernel> conv_s_kernel(int cin, int cout, int ksize, int dil, int pad, int stride, size_t in_len, size_t out_len) {
        const std::string ns = "h3.conv1d_s_f32.";
        return comp_.get("conv1d_s_f32", "h3_conv1d_s_f32", {{ns + "cin", std::to_string(cin)}, {ns + "cout", std::to_string(cout)}, {ns + "ksize", std::to_string(ksize)}, {ns + "dilation", std::to_string(dil)}, {ns + "pad", std::to_string(pad)},
                                                                {ns + "stride", std::to_string(stride)}, {ns + "in_bound", std::to_string(pow2_bound(in_len))}, {ns + "out_bound", std::to_string(pow2_bound(out_len))}});
    }
    void conv_s(const char *stage, int cin, int cout, int ksize, int dil, int pad, int stride, size_t in_len, size_t out_len, const void *x, const void *w, const void *b, void *out) {
        auto k = conv_s_kernel(cin, cout, ksize, dil, pad, stride, in_len, out_len);
        KernArgs a; a.i32(int(out_len)).i32(int(in_len)).ptr(x).ptr(w).ptr(b).ptr(out); launch(*k, &prof, stage, unsigned((out_len + 255) / 256), unsigned(cout), THREADS, a);
    }
    void snake_plain(int channels, size_t len, const void *x, const void *alpha, void *out) {
        const std::string ns = "h3.snake_f32.";
        auto k = comp_.get("snake_f32", "h3_snake_f32", {{ns + "channels", std::to_string(channels)}, {ns + "len_bound", std::to_string(pow2_bound(len))}});
        KernArgs a; a.i32(int(len)).ptr(x).ptr(alpha).ptr(out); launch(*k, &prof, "aenc snake", unsigned((len + 255) / 256), unsigned(channels), THREADS, a);
    }
    void layernorm(size_t rows, int width, const void *x, const void *w, const void *b, void *out) {
        const std::string ns = "h3.layernorm_f32.";
        auto k = comp_.get("layernorm_f32", "h3_layernorm_f32", {{ns + "width", std::to_string(width)}, {ns + "eps", num(1e-5)}});
        KernArgs a; a.i32(int(rows)).ptr(x).ptr(w).ptr(b).ptr(out); launch(*k, &prof, "aenc layernorm", unsigned(rows), 1, 32, a);
    }
    // out [m][n] f32 = x [m][k] f32 . w [n][k] f32 + b: one lane per output element (the audio encoder's heads, the patch projections, the final layer)
    void matmul(const char *stage, size_t m, int kdim, int n, const void *x, const void *w, const void *b, void *out) {
        const std::string ns = "h3.matmul_f32.";
        auto k = comp_.get("matmul_f32", "h3_matmul_f32", {{ns + "k", std::to_string(kdim)}, {ns + "n", std::to_string(n)}});
        KernArgs a; a.i32(int(m)).ptr(x).ptr(w).ptr(b).ptr(out); launch(*k, &prof, stage, unsigned((n + 255) / 256), unsigned(m), THREADS, a);
    }
    void transpose_f32(size_t rows, int cols, const void *x, void *out) {
        const std::string ns = "h3.transpose_f32.";
        auto k = comp_.get("transpose_f32", "h3_transpose_f32", {{ns + "cols", std::to_string(cols)}});
        KernArgs a; a.i32(int(rows)).ptr(x).ptr(out); launch(*k, &prof, "aenc transpose", unsigned((rows * size_t(cols) + 255) / 256), 1, THREADS, a);
    }
    void encode_audio(const float *samples, int n, float *out, int &audio_t) {
        ensure_avae();
        const size_t Lp = (size_t(n) + 799) / 800 * 800; const int T = int(Lp / 800); audio_t = T;
        auto W = [&](const std::string &nm, size_t count) { return avae_w_.at(nm, count * 4); };
        DeviceBuffers temporary; auto buf = [&](size_t floats) { return temporary.alloc(floats * 4); };
        const size_t plane = size_t(64) * Lp;
        void *x0 = buf(Lp), *h = buf(plane), *h2 = buf(plane), *y = buf(plane), *y2 = buf(plane);
        void *rows = buf(size_t(T) * 2048), *n1 = buf(size_t(T) * 2048), *qkv = buf(size_t(T) * 6144), *pattn = buf(size_t(8) * T * T), *pool = buf(size_t(T) * 32);
        void *xa = buf(size_t(T) * 32), *xb = buf(size_t(T) * 32), *xc = buf(size_t(T) * 32), *a0 = buf(size_t(T) * 64), *a1 = buf(size_t(T) * 64), *g = buf(size_t(T) * 64);
        static const int rates[5] = {2, 4, 4, 5, 5};
        std::vector<float> host(Lp), zrow(size_t(T) * 32);
        const float *lmean = nullptr; std::vector<float> mean_h(32), std_h(32);
        mean_h = avae_w_.host_f32("aenc.latents_mean", 32); std_h = avae_w_.host_f32("aenc.latents_std", 32); (void)lmean;
        for (int ch = 0; ch < 2; ++ch) {
            std::fill(host.begin(), host.end(), 0.0f); std::copy(samples + size_t(ch) * n, samples + size_t(ch) * n + n, host.begin());
            rt().h2d(x0, host.data(), Lp * 4);
            size_t len = Lp; int dim = 64;
            conv_s("aenc conv_in", 1, 64, 7, 1, 3, 1, Lp, Lp, x0, W("aenc.conv_in.w", 64 * 7), W("aenc.conv_in.b", 64), h);
            for (int i = 1; i <= 5; ++i) {
                for (int r = 0; r < 3; ++r) {
                    const int dil = r == 0 ? 1 : (r == 1 ? 3 : 9); const std::string p = fmt("aenc.b%d.r%d.", i, r);
                    snake_plain(dim, len, h, W(p + "act0", dim), y);
                    conv_s("aenc res conv7", dim, dim, 7, dil, 3 * dil, 1, len, len, y, W(p + "c1.w", size_t(dim) * dim * 7), W(p + "c1.b", dim), y2);
                    snake_plain(dim, len, y2, W(p + "act1", dim), y);
                    conv_s("aenc res conv1", dim, dim, 1, 1, 0, 1, len, len, y, W(p + "c2.w", size_t(dim) * dim), W(p + "c2.b", dim), y2);
                    axpy(1.0f, 1.0f, size_t(dim) * len, y2, h);
                }
                const int s = rates[i - 1]; const size_t out_len = len / s; const std::string p = fmt("aenc.b%d.", i);
                snake_plain(dim, len, h, W(p + "act", dim), y);
                conv_s("aenc down", dim, 2 * dim, 2 * s, 1, (s + 1) / 2, s, len, out_len, y, W(p + "down.w", size_t(2 * dim) * dim * 2 * s), W(p + "down.b", 2 * dim), h2);
                std::swap(h, h2); dim *= 2; len = out_len;
            }
            if (dim != 2048 || len != size_t(T)) throw std::runtime_error("audio encoder shape mismatch");
            snake_plain(2048, len, h, W("aenc.act_out", 2048), y);
            conv_s("aenc conv_out", 2048, 2048, 3, 1, 1, 1, len, len, y, W("aenc.conv_out.w", size_t(2048) * 2048 * 3), W("aenc.conv_out.b", 2048), h);
            transpose_f32(2048, T, h, rows);                                   // [2048][T] -> [T][2048]
            // AttnProjection: x = proj(norm3(x)) + attn(norm1(x)); x += mlp(norm2(x))
            layernorm(T, 2048, rows, W("aenc.pre.norm3.w", 2048), W("aenc.pre.norm3.b", 2048), n1);
            matmul("aenc matmul", T, 2048, 32, n1, W("aenc.pre.proj.w", size_t(32) * 2048), W("aenc.pre.proj.b", 32), xa);
            layernorm(T, 2048, rows, W("aenc.pre.norm1.w", 2048), W("aenc.pre.norm1.b", 2048), n1);
            matmul("aenc matmul", T, 2048, 6144, n1, W("aenc.pre.qkv.w", size_t(6144) * 2048), W("aenc.pre.qkv.b", 6144), qkv);
            { const std::string ns = "h3.attn_scores_f32."; auto k = comp_.get("attn_scores_f32", "h3_attn_scores_f32", {{ns + "heads", "8"}, {ns + "hd", "256"}, {ns + "scale", num(1.0 / 16.0)}});
              KernArgs a; a.i32(T).ptr(qkv).ptr(pattn); launch(*k, &prof, "aenc attention", unsigned((T + 255) / 256), 8, THREADS, a); }
            { const std::string ns = "h3.attn_pv_pool_f32."; auto k = comp_.get("attn_pv_pool_f32", "h3_attn_pv_pool_f32", {{ns + "heads", "8"}, {ns + "hd", "256"}, {ns + "pool", "8"}});
              KernArgs a; a.i32(T).ptr(qkv).ptr(pattn).ptr(pool); launch(*k, &prof, "aenc attention", unsigned(T), 1, 32, a); }
            matmul("aenc matmul", T, 32, 32, pool, W("aenc.pre.attn_proj.w", 32 * 32), W("aenc.pre.attn_proj.b", 32), xb);
            axpy(1.0f, 1.0f, size_t(T) * 32, xb, xa);                                                       // xa = proj + attn
            layernorm(T, 32, xa, W("aenc.pre.norm2.w", 32), W("aenc.pre.norm2.b", 32), xb);
            layernorm(T, 32, xb, W("aenc.pre.mlp.norm.w", 32), W("aenc.pre.mlp.norm.b", 32), xc);
            matmul("aenc matmul", T, 32, 64, xc, W("aenc.pre.mlp.w0.w", 64 * 32), W("aenc.pre.mlp.w0.b", 64), a0);
            matmul("aenc matmul", T, 32, 64, xc, W("aenc.pre.mlp.w1.w", 64 * 32), W("aenc.pre.mlp.w1.b", 64), a1);
            { auto k = comp_.get("geglu_tanh_f32", "h3_geglu_tanh_f32", {}); KernArgs a; a.i32(T * 64).ptr(a0).ptr(a1).ptr(g); launch(*k, &prof, "aenc geglu", unsigned((T * 64 + 255) / 256), 1, THREADS, a); }
            matmul("aenc matmul", T, 64, 32, g, W("aenc.pre.mlp.w2.w", 32 * 64), W("aenc.pre.mlp.w2.b", 32), xb);
            axpy(1.0f, 1.0f, size_t(T) * 32, xb, xa);                                                       // xa += mlp
            matmul("aenc matmul", T, 32, 32, xa, W("aenc.mean_proj.w", 32 * 32), W("aenc.mean_proj.b", 32), xc);
            rt().d2h(zrow.data(), xc, size_t(T) * 32 * 4);
            for (int t = 0; t < T; ++t) for (int c = 0; c < 32; ++c) out[(size_t(ch) * 32 + c) * T + t] = (zrow[size_t(t) * 32 + c] - mean_h[c]) / std_h[c];
        }
        rt().sync();
    }

    // --- the audio decoder: BigVGAN in f32 SIMT Loom kernels ---
    struct Conv { std::shared_ptr<Kernel> k; int cin, cout, ksize, dil, pad; };
    Conv conv_kernel(int cin, int cout, int ksize, int dil, int pad, bool accumulate, size_t len_bound) {
        const std::string ns = "h3.conv1d_f32.";
        return Conv{comp_.get("conv1d_f32", "h3_conv1d_f32", {{ns + "cin", std::to_string(cin)}, {ns + "cout", std::to_string(cout)}, {ns + "ksize", std::to_string(ksize)}, {ns + "dilation", std::to_string(dil)}, {ns + "pad", std::to_string(pad)}, {ns + "accumulate", accumulate ? "1" : "0"}, {ns + "len_bound", std::to_string(len_bound)}}), cin, cout, ksize, dil, pad};
    }
    void conv_run(const Conv &c, const char *stage, size_t len, const void *x, const void *w, const void *b, void *out) {
        KernArgs a; a.i32(int(len)).ptr(x).ptr(w).ptr(b).ptr(out); launch(*c.k, &prof, stage, unsigned((len + 255) / 256), unsigned(c.cout), THREADS, a);
    }
    static size_t round256(size_t n) { return (n + 255) / 256 * 256; }
    void axpy(float av, float bv, size_t count, const void *x, void *y) {
        const std::string ns = "h3.axpy_f32.";
        auto k = comp_.get("axpy_f32", "h3_axpy_f32", {{ns + "a", num(av)}, {ns + "b", num(bv)}});
        KernArgs a; a.i32(int(count)).ptr(x).ptr(y); launch(*k, &prof, "audio axpy", unsigned((count + 255) / 256), 1, THREADS, a);
    }
    // the anti-aliased SnakeBeta: x [C][len] -> out [C][len] through the 2x buffer
    void snake(int channels, size_t len, const void *x, const void *alpha, const void *beta, void *tmp2, void *out) {
        const std::string nu = "h3.up2_snake_f32.", nd = "h3.down2_f32.";
        auto up = comp_.get("up2_snake_f32", "h3_up2_snake_f32", {{nu + "channels", std::to_string(channels)}, {nu + "len_bound", std::to_string(round256(len))}});
        auto down = comp_.get("down2_f32", "h3_down2_f32", {{nd + "channels", std::to_string(channels)}, {nd + "len_bound", std::to_string(round256(2 * len))}});
        const void *fir = avae_w_.at("audio.fir", 12 * 4);
        { KernArgs a; a.i32(int(len)).ptr(x).ptr(fir).ptr(alpha).ptr(beta).ptr(tmp2); launch(*up, &prof, "audio snake up", unsigned((2 * len + 255) / 256), unsigned(channels), THREADS, a); }
        { KernArgs a; a.i32(int(2 * len)).ptr(tmp2).ptr(fir).ptr(out); launch(*down, &prof, "audio snake down", unsigned((len + 255) / 256), unsigned(channels), THREADS, a); }
    }
    void decode_audio(const float *latents, int audio_t, float *samples) {
        static const int RATES[7] = {5, 5, 2, 2, 2, 2, 2}, UPK[7] = {9, 9, 4, 4, 4, 4, 4}, RESK[3] = {3, 7, 11}, DIL[3] = {1, 3, 5};
        const int T = audio_t; const size_t L_out = size_t(T) * 800;
        const size_t cap = std::max<size_t>(size_t(2048) * T, size_t(8) * L_out) + 4096;    // the widest [C][len] plane of the stack
        if (audio_cap_ < cap) {
            audio_cap_ = 0;
            for (void **q : {&ah_, &aacc_, &ahj_, &ar_, &ar2_, &atmp_, &ain_}) { memory_.free(*q); *q = nullptr; }
            ah_ = memory_.alloc(cap * 4); aacc_ = memory_.alloc(cap * 4); ahj_ = memory_.alloc(cap * 4); ar_ = memory_.alloc(cap * 4); ar2_ = memory_.alloc(cap * 4);
            atmp_ = memory_.alloc(cap * 8); ain_ = memory_.alloc(size_t(32) * T * 4 + 4096); audio_cap_ = cap;
        }
        ensure_avae();
        std::vector<float> lmean = avae_w_.host_f32("audio.latents_mean", 32), lstd = avae_w_.host_f32("audio.latents_std", 32);
        std::vector<float> in(size_t(32) * T), out(L_out);
        for (int ch = 0; ch < 2; ++ch) {
            for (int c = 0; c < 32; ++c) for (int t = 0; t < T; ++t) in[size_t(c) * T + t] = latents[(size_t(ch) * 32 + c) * T + t] * lstd[c] + lmean[c];
            rt().h2d(ain_, in.data(), in.size() * 4);
            // dec_in_proj (32 -> 2048, k 1), conv_pre (2048 -> 1024, k 7)
            conv_run(conv_kernel(32, 2048, 1, 1, 0, false, round256(T)), "audio dec_in_proj", T, ain_, avae_w_.at("audio.dec_in_proj.w", size_t(2048) * 32 * 4), avae_w_.at("audio.dec_in_proj.b", 2048 * 4), ar_);
            conv_run(conv_kernel(2048, 1024, 7, 1, 3, false, round256(T)), "audio conv_pre", T, ar_, avae_w_.at("audio.conv_pre.w", size_t(1024) * 2048 * 7 * 4), avae_w_.at("audio.conv_pre.b", 1024 * 4), ah_);
            size_t len = size_t(T); int C = 1024;
            for (int i = 0; i < 7; ++i) {
                const int Cout = C / 2, k = UPK[i], rate = RATES[i], pad = (k - rate) / 2; const size_t olen = (len - 1) * rate + k - 2 * pad;
                { const std::string ns = "h3.convt1d_f32.";
                  auto kt = comp_.get("convt1d_f32", "h3_convt1d_f32", {{ns + "cin", std::to_string(C)}, {ns + "cout", std::to_string(Cout)}, {ns + "ksize", std::to_string(k)}, {ns + "stride", std::to_string(rate)}, {ns + "pad", std::to_string(pad)}, {ns + "len_bound", std::to_string(round256(olen))}});
                  KernArgs a; a.i32(int(len)).i32(int(olen)).ptr(ah_).ptr(avae_w_.at(fmt("audio.ups.%d.w", i), size_t(C) * Cout * k * 4)).ptr(avae_w_.at(fmt("audio.ups.%d.b", i), size_t(Cout) * 4)).ptr(ar_);
                  launch(*kt, &prof, "audio upsample", unsigned((olen + 255) / 256), unsigned(Cout), THREADS, a); }
                len = olen; C = Cout; const size_t plane = size_t(C) * len;
                rt().d2d(ah_, ar_, plane * 4); rt().memset(aacc_, 0, plane * 4);
                for (int j = 0; j < 3; ++j) {                                   // the three AMP blocks, averaged
                    const int r = i * 3 + j, kk = RESK[j];
                    rt().d2d(ahj_, ah_, plane * 4);
                    for (int d = 0; d < 3; ++d) {
                        const std::string act1 = fmt("audio.res.%d.act.%d.", r, 2 * d), act2 = fmt("audio.res.%d.act.%d.", r, 2 * d + 1);
                        snake(C, len, ahj_, avae_w_.at(act1 + "alpha", size_t(C) * 4), avae_w_.at(act1 + "beta", size_t(C) * 4), atmp_, ar_);
                        conv_run(conv_kernel(C, C, kk, DIL[d], (kk * DIL[d] - DIL[d]) / 2, false, round256(len)), "audio res conv1", len, ar_, avae_w_.at(fmt("audio.res.%d.c1.%d.w", r, d), size_t(C) * C * kk * 4), avae_w_.at(fmt("audio.res.%d.c1.%d.b", r, d), size_t(C) * 4), ar2_);
                        snake(C, len, ar2_, avae_w_.at(act2 + "alpha", size_t(C) * 4), avae_w_.at(act2 + "beta", size_t(C) * 4), atmp_, ar_);
                        conv_run(conv_kernel(C, C, kk, 1, (kk - 1) / 2, true, round256(len)), "audio res conv2", len, ar_, avae_w_.at(fmt("audio.res.%d.c2.%d.w", r, d), size_t(C) * C * kk * 4), avae_w_.at(fmt("audio.res.%d.c2.%d.b", r, d), size_t(C) * 4), ahj_);
                    }
                    axpy(1.0f, 1.0f, plane, ahj_, aacc_);
                }
                axpy(1.0f / 3.0f, 0.0f, plane, aacc_, ah_);
            }
            snake(C, len, ah_, avae_w_.at("audio.post.alpha", size_t(C) * 4), avae_w_.at("audio.post.beta", size_t(C) * 4), atmp_, ar_);
            conv_run(conv_kernel(C, 1, 7, 1, 3, false, round256(len)), "audio conv_post", len, ar_, avae_w_.at("audio.conv_post.w", size_t(C) * 7 * 4), zeros_, ar2_);
            if (len != L_out) throw std::runtime_error("audio length " + std::to_string(len) + " != " + std::to_string(L_out));
            rt().sync(); rt().d2h(out.data(), ar2_, L_out * 4);
            for (size_t i = 0; i < L_out; ++i) samples[size_t(ch) * L_out + i] = std::min(std::max(out[i], -1.0f), 1.0f);
        }
    }

    // --- the video decoder: diffusers' chunking and heads around the 36 blocks in Loom ---
    // one clip: model-space latents z [24][ft][h][w] (already * std + mean) -> ImageNet-space frames [3][ft*4][h*16][w*16]
    void decode_clip(const float *z, int ft, int h, int w, std::vector<float> &frames) {
        const size_t N = size_t(ft) * h * w, NT = N + VAE_REG + 1;
        ensure_vvae();
        if (!vae_ || !vae_grid_.matches(ft, h, w)) {
            vae_grid_ = {};
            vae_.reset(); vae_ = std::make_unique<Stack>(comp_, StackDims{VAE_HID, VAE_HEADS, VAE_HEADS, VAE_D, VAE_FFN, 48, 1, 16, 1e-5f, true, false, false}, NT, 36, vvae_w_, "blocks.%d.", false, (const float *)ones_, "vae");
            for (void **q : {&vx_, &vcos_, &vsin_, &vin16_, &va_q_, &va_s_, &vout16_, &vcls_}) { if (*q) memory_.free(*q); *q = nullptr; }
            const size_t T = vae_->capacity();
            vx_ = memory_.alloc(T * VAE_HID * 4); rt().memset(vx_, 0, T * VAE_HID * 4);
            vcos_ = memory_.alloc(T * VAE_ROPE_HALF * 4); vsin_ = memory_.alloc(T * VAE_ROPE_HALF * 4);
            vin16_ = memory_.alloc(T * VAE_KIN * 2); rt().memset(vin16_, 0, T * VAE_KIN * 2);
            va_q_ = memory_.alloc(T * VAE_HID * 2); va_s_ = memory_.alloc(T * 4);
            vout16_ = memory_.alloc(T * VAE_OUT * 2); vcls_ = memory_.alloc(T * 4); rt().memset(vcls_, 0, T * 4);
            // the rotary tables: coordinates 2 * ((i + 0.5) / size) - 1 per axis, angles 2 pi * pos * inv_freq (8 frequencies per axis), zero for the register / cls rows
            std::vector<float> c(T * VAE_ROPE_HALF, 1.0f), s(T * VAE_ROPE_HALF, 0.0f);
            const int sizes[3] = {ft, h, w};
            for (int t = 0; t < ft; ++t) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                const size_t r = (size_t(t) * h + y) * w + x; const int idx[3] = {t, y, x};
                for (int ax = 0; ax < 3; ++ax) for (int j = 0; j < 8; ++j) {
                    const double pos = 2.0 * ((idx[ax] + 0.5) / sizes[ax]) - 1.0, inv = std::pow(100.0, -double(j) * 6.0 / 48.0), ang = 2.0 * M_PI * pos * inv;
                    c[r * VAE_ROPE_HALF + ax * 8 + j] = float(std::cos(ang)); s[r * VAE_ROPE_HALF + ax * 8 + j] = float(std::sin(ang)); } }
            rt().h2d(vcos_, c.data(), c.size() * 4); rt().h2d(vsin_, s.data(), s.size() * 4);
            // the checkpoint's f16 rows: x_embedder takes the host's f16 latent rows as they are, no prepare
            vproj_in_.build(comp_, "resid", "f16", true, true, VAE_KIN, VAE_HID, N, 1);
            vnorm_out_.build(comp_, "lnorm", "f16", VAE_HID, 1e-5f, 1); vproj_out_.build(comp_, "plain", "f16", true, true, VAE_HID, VAE_OUT, NT);
            if (!vnorm_table_) { vnorm_table_ = memory_.alloc(size_t(2) * VAE_HID * 4); rt().memset(vnorm_table_, 0, size_t(VAE_HID) * 4);
                rt().d2d(((float *)vnorm_table_ + VAE_HID), vvae_w_.at("vae.norm_out.b", size_t(VAE_HID) * 4), size_t(VAE_HID) * 4); }
        }
        vae_grid_ = {ft, h, w};
        // post_quant_conv (a 24x24 matrix per voxel) on the host, then the tokens padded to K = 256 in f16
        const auto pq_w = vvae_w_.host_f32("vae.post_quant_conv.w", 24 * 24), pq_b = vvae_w_.host_f32("vae.post_quant_conv.b", 24);
        std::vector<uint16_t> in16(N * VAE_KIN, 0);
        for (size_t v = 0; v < N; ++v) for (int o = 0; o < LATENT_CH; ++o) { float acc = pq_b[o]; for (int i = 0; i < LATENT_CH; ++i) acc += pq_w[o * 24 + i] * z[size_t(i) * N + v]; in16[v * VAE_KIN + o] = f32_to_f16(acc); }
        rt().h2d(vin16_, in16.data(), in16.size() * 2);
        rt().memset(vx_, 0, NT * VAE_HID * 4);
        vproj_in_.run(&prof, "vae proj_in", unsigned(N), vin16_, vvae_w_.at("vae.proj_in.w", size_t(VAE_HID) * VAE_KIN * 2), nullptr, nullptr, vx_, ones_, vcls_, vvae_w_.at("vae.proj_in.b", size_t(VAE_HID) * 4));
        rt().d2d(((float *)vx_ + N * VAE_HID), vvae_w_.at("vae.register_tokens", size_t(VAE_REG) * VAE_HID * 4), size_t(VAE_REG) * VAE_HID * 4);   // then a zero cls row
        vae_->forward(&prof, vx_, vcls_, vcos_, vsin_, [&](int i) { const Stack::Block &b = vae_->block(i); return LayerCond{(const float *)zeros_, (const float *)b.scale1, (const float *)zeros_, (const float *)b.scale2}; });
        vnorm_out_.run(&prof, "vae norm_out", unsigned(NT), vx_, vvae_w_.at("vae.norm_out.w", size_t(VAE_HID) * 4), vnorm_table_, vcls_, va_q_, va_s_);
        vproj_out_.run(&prof, "vae proj_out", unsigned(NT), va_q_, vvae_w_.at("vae.proj_out.w", size_t(VAE_OUT) * VAE_HID * 2), nullptr, nullptr, vout16_, nullptr, nullptr, vvae_w_.at("vae.proj_out.b", size_t(VAE_OUT) * 4));
        std::vector<uint16_t> out16(N * VAE_OUT); rt().sync(); rt().d2h(out16.data(), vout16_, out16.size() * 2);
        // unpatchify: token (t, y, x) holds [3][4][16][16] -> frames [3][ft*4][h*16][w*16]
        const size_t FH = size_t(h) * VAE_PS, FW = size_t(w) * VAE_PS, FT = size_t(ft) * VAE_PT;
        frames.assign(size_t(3) * FT * FH * FW, 0.0f);
        for (int t = 0; t < ft; ++t) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const uint16_t *tok = out16.data() + ((size_t(t) * h + y) * w + x) * VAE_OUT;
            for (int c = 0; c < 3; ++c) for (int pt = 0; pt < VAE_PT; ++pt) for (int py = 0; py < VAE_PS; ++py) for (int px = 0; px < VAE_PS; ++px)
                frames[((size_t(c) * FT + size_t(t) * VAE_PT + pt) * FH + size_t(y) * VAE_PS + py) * FW + size_t(x) * VAE_PS + px] = f16_to_f32(tok[((size_t(c) * VAE_PT + pt) * VAE_PS + py) * VAE_PS + px]);
        }
    }

    // Decode spatial tiles before the temporal chunk blend, as the released VAE does.
    void decode_spatial(const float *z, int ft, int h, int w, std::vector<float> &frames) {
        const int H = h * VAE_PS, W = w * VAE_PS, F = ft * VAE_PT;
        std::vector<int> ys, yo, xs, xo; split_tiles(H, ys, yo); split_tiles(W, xs, xo);
        if (ys.size() == 1 && xs.size() == 1) { decode_clip(z, ft, h, w, frames); return; }
        frames.assign(size_t(3) * F * H * W, 0.0f);
        std::vector<std::vector<float>> above(xs.size()), row(xs.size());
        const int TH = std::min(H, 256), TW = std::min(W, 256), lh = TH / VAE_PS, lw = TW / VAE_PS;
        std::vector<float> latent(size_t(LATENT_CH) * ft * lh * lw);
        for (size_t iy = 0; iy < ys.size(); ++iy) {
            for (size_t ix = 0; ix < xs.size(); ++ix) {
                for (int c = 0; c < LATENT_CH; ++c) for (int t = 0; t < ft; ++t) for (int y = 0; y < lh; ++y)
                    memcpy(latent.data() + ((size_t(c) * ft + t) * lh + y) * lw,
                           z + ((size_t(c) * ft + t) * h + ys[iy] / VAE_PS + y) * w + xs[ix] / VAE_PS, size_t(lw) * 4);
                decode_clip(latent.data(), ft, lh, lw, row[ix]);
                std::vector<float> tile = row[ix];
                auto blend = [&](const std::vector<float> &a, int extent, bool vertical) {
                    for (int c = 0; c < 3; ++c) for (int t = 0; t < F; ++t)
                        for (int y = 0; y < (vertical ? extent : TH); ++y) for (int x = 0; x < (vertical ? TW : extent); ++x) {
                            const float wb = float(vertical ? y : x) / extent, wa = 1.0f - wb;
                            const size_t dst = ((size_t(c) * F + t) * TH + y) * TW + x;
                            const size_t src = ((size_t(c) * F + t) * TH + (vertical ? TH - extent + y : y)) * TW + (vertical ? x : TW - extent + x);
                            tile[dst] = wa * a[src] + wb * tile[dst];
                        }
                };
                if (iy) blend(above[ix], yo[iy - 1], true);
                if (ix) blend(row[ix - 1], xo[ix - 1], false);
                const int keep_h = TH - (iy + 1 < ys.size() ? yo[iy] : 0), keep_w = TW - (ix + 1 < xs.size() ? xo[ix] : 0);
                for (int c = 0; c < 3; ++c) for (int t = 0; t < F; ++t) for (int y = 0; y < keep_h; ++y)
                    memcpy(frames.data() + ((size_t(c) * F + t) * H + ys[iy] + y) * W + xs[ix],
                           tile.data() + ((size_t(c) * F + t) * TH + y) * TW, size_t(keep_w) * 4);
            }
            above.swap(row);
        }
    }

    // the clip loop of diffusers' _decode: 5-token chunks with a 2-token overlap, 17 kept frames per chunk (3 dropped in front), 5-frame cross-fades
    void decode_video(const h3pipe_params &p, const float *latents, uint8_t *out) {
        const h3pipe_shape sh = shape_or_throw(p);
        const int T = sh.latent_t, H = sh.lat_h, W = sh.lat_w, F = sh.frames;
        const size_t FH = size_t(H) * VAE_PS, FW = size_t(W) * VAE_PS, plane = FH * FW;
        ensure_vvae();
        std::vector<float> lmean = vvae_w_.host_f32("vae.latents_mean", 24), lstd = vvae_w_.host_f32("vae.latents_std", 24);
        const int num_tokens = T + VAE_TOKEN_DROP, pad_tokens = ((-num_tokens) % VAE_CHUNK + VAE_CHUNK) % VAE_CHUNK, num_chunks = decoder_chunks(T, pad_tokens);
        const int Tp = T + pad_tokens;
        std::vector<float> zp(size_t(LATENT_CH) * Tp * H * W);
        for (int c = 0; c < LATENT_CH; ++c) for (int t = 0; t < Tp; ++t) for (size_t i = 0; i < size_t(H) * W; ++i)
            zp[(size_t(c) * Tp + t) * H * W + i] = latents[(size_t(c) * T + std::min(t, T - 1)) * H * W + i] * lstd[c] + lmean[c];
        const int chunk_frames = VAE_CHUNK * VAE_TRATIO, pre = ((-VAE_CLIP) % VAE_TRATIO + VAE_TRATIO) % VAE_TRATIO, overlap_frames = std::max(VAE_OVERLAP * VAE_TRATIO - pre, 0);
        std::vector<float> overlap; bool have_overlap = false;
        size_t dec_frames = 0;
        auto append = [&](const std::vector<float> &chunk, size_t nf, size_t src_ft) {
            const size_t take = dec_frames < size_t(F) ? std::min(nf, size_t(F) - dec_frames) : 0;
            for (size_t f = 0; f < take; ++f) for (size_t q = 0; q < plane; ++q) for (int c = 0; c < 3; ++c) {
                const float v = chunk[(size_t(c) * src_ft + f) * plane + q] * IMAGENET_STD[c] + IMAGENET_MEAN[c];
                out[((dec_frames + f) * plane + q) * 3 + c] = uint8_t(std::lround(std::min(std::max(v, 0.0f), 1.0f) * 255.0f));
            }
            dec_frames += nf;
        };
        std::vector<float> clip, chunk;
        for (int i = 0; i < num_chunks; ++i) {
            const int start = i * VAE_CHUNK, ft = std::min(VAE_CHUNK + VAE_OVERLAP, Tp - start);
            std::vector<float> z(size_t(LATENT_CH) * ft * H * W);
            for (int c = 0; c < LATENT_CH; ++c) memcpy(z.data() + size_t(c) * ft * H * W, zp.data() + (size_t(c) * Tp + start) * H * W, size_t(ft) * H * W * 4);
            decode_spatial(z.data(), ft, H, W, clip);
            const size_t clip_frames = size_t(ft) * VAE_TRATIO;
            for (int j = 0; j < 2; ++j) {
                const size_t f0 = size_t(j) * chunk_frames + pre, f1 = std::min(size_t(j + 1) * chunk_frames, clip_frames);
                if (f0 >= f1) { if (j == 1) have_overlap = false; continue; }
                const size_t nf = f1 - f0;
                chunk.assign(size_t(3) * nf * plane, 0.0f);
                for (int c = 0; c < 3; ++c) memcpy(chunk.data() + size_t(c) * nf * plane, clip.data() + (size_t(c) * clip_frames + f0) * plane, nf * plane * 4);
                if (j == 0) {
                    if (have_overlap) {       // cross-fade the overlap's tail into the chunk's head
                        const size_t ov = overlap.size() / (3 * plane), be = std::min(std::min(ov, nf), size_t(overlap_frames));
                        for (int c = 0; c < 3; ++c) for (size_t k = 0; k < be; ++k) { const float wb = float(k) / be, wa = 1.0f - wb;
                            float *dst = chunk.data() + (size_t(c) * nf + k) * plane; const float *src = overlap.data() + (size_t(c) * ov + ov - be + k) * plane;
                            for (size_t q = 0; q < plane; ++q) dst[q] = wa * src[q] + wb * dst[q]; }
                    }
                    append(chunk, nf, nf);
                } else { overlap = chunk; have_overlap = true; }
            }
        }
        if (have_overlap) append(overlap, overlap.size() / (3 * plane), overlap.size() / (3 * plane));
        size_t keep = dec_frames;
        if (pad_tokens > 0) {
            const int intra_tail = VAE_CLIP % VAE_TRATIO; size_t pad_frames = 0;
            for (int k = 0; k < pad_tokens; ++k) pad_frames += (intra_tail && (T + k) % VAE_CHUNK == 0) ? size_t(intra_tail) : size_t(VAE_TRATIO);
            keep = dec_frames - pad_frames;
        }
        if (keep != size_t(F)) throw std::runtime_error("decoded " + std::to_string(keep) + " frames, expected " + std::to_string(F));

    }

private:
    Compiler comp_;
    Weights dit_w_, te_w_, vvae_w_, avae_w_; std::string dit_file_, te_file_, video_vae_file_, audio_vae_file_; const Entry *embed_ = nullptr; bool conditioning_ready_ = false, te_ready_ = false, refiner_ready_ = false;
    std::vector<float> curve_, inv_freq_, final_w_, final_b_; std::vector<std::vector<float>> adaln_w_, adaln_b_;
    std::unique_ptr<Stack> te_, refiner_, dit_, vae_;
    DecoderGrid vae_grid_;
    Prepare vnorm_out_; Gemm vproj_in_, vproj_out_;
    std::shared_ptr<Kernel> absdiff_; void *xb0_ = nullptr, *prev_b0_ = nullptr, *cache_resid_ = nullptr, *partials_ = nullptr; size_t cache_cap_ = 0;
    double cache_acc_ = 0.0; bool have_cache_ = false; int cache_skipped_ = 0;
    size_t audio_cap_ = 0; void *ah_ = nullptr, *aacc_ = nullptr, *ahj_ = nullptr, *ar_ = nullptr, *ar2_ = nullptr, *atmp_ = nullptr, *ain_ = nullptr;
    void *vx_ = nullptr, *vcos_ = nullptr, *vsin_ = nullptr, *vin16_ = nullptr, *va_q_ = nullptr, *va_s_ = nullptr, *vout16_ = nullptr, *vcls_ = nullptr, *vnorm_table_ = nullptr;
    std::shared_ptr<Kernel> final_norm_;
    std::string te_span_sig_; void *ds_buf_ = nullptr;
    int attn_qk_bits_ = 8;
    std::shared_ptr<Kernel> norm_f32_;
    size_t seq_cap_ = 0;
    void *ones_ = nullptr, *zeros_ = nullptr, *mods_ = nullptr, *final_table_ = nullptr, *x_ = nullptr, *cls_ = nullptr, *cls0_ = nullptr, *tcls_ = nullptr, *cos_ = nullptr, *sin_ = nullptr,
         *in32_ = nullptr, *out32_ = nullptr, *text_copy_ = nullptr, *ref_cos_ = nullptr, *ref_sin_ = nullptr, *te_x_ = nullptr, *te_cos_ = nullptr, *te_sin_ = nullptr, *te_cls_ = nullptr;
};

void write_error(char *error, size_t cap, const char *m) noexcept { if (error && cap) std::snprintf(error, cap, "%s", m ? m : "unknown error"); }

}  // namespace

struct h3pipe_session { Pipe value; std::mutex mutex; explicit h3pipe_session(const h3pipe_config &c) : value(c) {} };

extern "C" uint32_t h3pipe_abi_version(void) { return H3PIPE_ABI_VERSION; }
extern "C" void h3pipe_destroy(h3pipe_session *s) { delete s; }

extern "C" int h3pipe_shape_for(const h3pipe_params *p, h3pipe_shape *out) {
    if (!p || !out) return H3PIPE_INVALID_ARGUMENT;
    if (!valid_canvas(p->height, p->width) || p->frames < 1 || p->frames > MAX_FRAMES) return H3PIPE_INVALID_ARGUMENT;
    out->frames = align_frames(std::max(p->frames, 5)); out->latent_t = (out->frames - 5) / 17 * 5 + 2;
    out->lat_h = p->height / 16; out->lat_w = p->width / 16; out->audio_t = int(std::lround(double(out->frames) / FPS * AUDIO_LATENTS_PER_S));
    out->text_rows_max = 4096;
    return H3PIPE_OK;
}

#define GUARD(body) \
    if (error && cap) error[0] = 0; \
    try { body; return H3PIPE_OK; } \
    catch (const Pipe::Cancelled &) { write_error(error, cap, "cancelled"); return H3PIPE_CANCELLED; } \
    catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return H3PIPE_INVALID_ARGUMENT; } \
    catch (const std::exception &e) { write_error(error, cap, e.what()); return H3PIPE_ERROR; } \
    catch (...) { write_error(error, cap, "unknown C++ exception"); return H3PIPE_ERROR; }

extern "C" int h3pipe_create(const h3pipe_config *config, h3pipe_session **out, char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    if (!out || !config) { write_error(error, cap, "config and out_session are required"); return H3PIPE_INVALID_ARGUMENT; }
    *out = nullptr;
    GUARD({ *out = new h3pipe_session(*config); })
}

extern "C" int h3pipe_text_in(h3pipe_session *s, const int32_t *ids, int n, float *outp, size_t out_elements, char *error, size_t cap) {
    GUARD({
        if (!s || !ids || !outp || n < 1) throw std::invalid_argument("session, ids (n >= 1) and out are required");
        if (out_elements < size_t(n) * HID) throw std::invalid_argument("out must hold at least n_ids * 5376 floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.get_text_in(ids, n, outp);
    })
}

extern "C" int h3pipe_denoise_refs(h3pipe_session *s, const int32_t *ids, int n, const h3pipe_params *params, const h3pipe_keyframe *keyframes, int n_keyframes, const h3pipe_ref *refs, int n_refs, const float *noise_video, const float *noise_audio,
                                   float *video, size_t video_elements, float *audio, size_t audio_elements, h3pipe_progress progress, void *user, char *error, size_t cap) {
    GUARD({
        if (!s || !ids || !params || !video || !audio || n < 1 || (n_refs > 0 && !refs) || (n_keyframes > 0 && !keyframes)) throw std::invalid_argument("session, ids, params, keyframes, refs and outputs are required");
        const h3pipe_shape sh = shape_or_throw(*params);
        if (video_elements < size_t(LATENT_CH) * sh.latent_t * sh.lat_h * sh.lat_w) throw std::invalid_argument("video_latents must hold at least 24 * latent_t * lat_h * lat_w floats");
        if (audio_elements < size_t(2) * AUDIO_CH * sh.audio_t) throw std::invalid_argument("audio_latents must hold at least 2 * 32 * audio_t floats");
        std::vector<h3pipe_ref> rv(refs, refs + n_refs); std::vector<h3pipe_keyframe> kv(keyframes, keyframes + n_keyframes);
        for (const h3pipe_keyframe &kf : kv) { if (!kf.video_latent || kf.frame_index < 0 || kf.frame_index >= sh.frames) throw std::invalid_argument("keyframes need a video latent and a frame index inside the clip"); if (kf.audio_latent && kf.audio_t < 1) throw std::invalid_argument("keyframe audio needs audio_t >= 1"); }
        for (const h3pipe_ref &rf : rv) {
            if (rf.kind < 0 || rf.kind > 2) throw std::invalid_argument("ref kind must be 0 (image), 1 (audio) or 2 (video)");
            if (rf.kind != 1 && (!rf.video_latent || rf.lat_h < 2 || rf.lat_w < 2 || rf.lat_h % 2 || rf.lat_w % 2 || (rf.kind == 2 && rf.latent_t < 1))) throw std::invalid_argument("image/video refs need a video latent with even lat_h, lat_w");
            if (rf.kind == 1 && (!rf.audio_latent || rf.audio_t < 1)) throw std::invalid_argument("audio refs need an audio latent with audio_t >= 1");
        }
        std::lock_guard<std::mutex> lock(s->mutex); s->value.denoise(ids, n, *params, noise_video, noise_audio, video, audio, progress, user, rv, kv);
    })
}
extern "C" int h3pipe_denoise(h3pipe_session *s, const int32_t *ids, int n, const h3pipe_params *params, const float *noise_video, const float *noise_audio,
                              float *video, size_t video_elements, float *audio, size_t audio_elements, h3pipe_progress progress, void *user, char *error, size_t cap) {
    GUARD({
        if (!s || !ids || !params || !video || !audio || n < 1) throw std::invalid_argument("session, ids, params and outputs are required");
        const h3pipe_shape sh = shape_or_throw(*params);
        if (video_elements < size_t(LATENT_CH) * sh.latent_t * sh.lat_h * sh.lat_w) throw std::invalid_argument("video_latents must hold at least 24 * latent_t * lat_h * lat_w floats");
        if (audio_elements < size_t(2) * AUDIO_CH * sh.audio_t) throw std::invalid_argument("audio_latents must hold at least 2 * 32 * audio_t floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.denoise(ids, n, *params, noise_video, noise_audio, video, audio, progress, user);
    })
}

extern "C" int h3pipe_decode_video(h3pipe_session *s, const h3pipe_params *params, const float *latents, size_t video_elements, uint8_t *frames, size_t frame_bytes, char *error, size_t cap) {
    GUARD({
        if (!s || !params || !latents || !frames) throw std::invalid_argument("session, params, latents and frames are required");
        const h3pipe_shape sh = shape_or_throw(*params);
        if (video_elements < size_t(LATENT_CH) * sh.latent_t * sh.lat_h * sh.lat_w) throw std::invalid_argument("video_latents must hold at least 24 * latent_t * lat_h * lat_w floats");
        if (frame_bytes < size_t(sh.frames) * params->height * params->width * 3) throw std::invalid_argument("frames must hold at least frames * height * width * 3 bytes");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.decode_video(*params, latents, frames);
    })
}
extern "C" int h3pipe_encode_video(h3pipe_session *s, const float *pixels, int frames, int height, int width, float *latents, size_t latent_elements, int *latent_t, char *error, size_t cap) {
    GUARD({
        if (!s || !pixels || !latents || !latent_t || frames < 1 || frames > MAX_FRAMES) throw std::invalid_argument("session, pixels (frames 1..1048576), latents and latent_t are required");
        if (height % 32 || width % 32 || height < 32 || width < 32 || height > 2048 || width > 2048) throw std::invalid_argument("height and width must be multiples of 32 up to 2048");
        const int TL = frames == 1 ? 1 : 5 * ((frames + 16) / 17) - 3;
        if (latent_elements < size_t(24) * TL * (height / 16) * (width / 16)) throw std::invalid_argument("latents must hold 24 * latent_t * height/16 * width/16 floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.encode_video(pixels, frames, height, width, latents, *latent_t);
    })
}
extern "C" int h3pipe_vision_embed(h3pipe_session *s, const float *pixels, int height, int width, float *merged, size_t merged_elements, float *deepstack, size_t deepstack_elements, int *tokens, char *error, size_t cap) {
    GUARD({
        if (!s || !pixels || !merged || !deepstack || !tokens) throw std::invalid_argument("session, pixels, merged, deepstack and tokens are required");
        if (!valid_canvas(height, width)) throw std::invalid_argument("height and width must be multiples of 32 up to 8192");
        const size_t m = size_t(height / 32) * size_t(width / 32);
        if (merged_elements < m * 5120 || deepstack_elements < 3 * m * 5120) throw std::invalid_argument("merged needs tokens * 5120 floats, deepstack 3 * tokens * 5120");
        std::lock_guard<std::mutex> lock(s->mutex); std::vector<float> mg; std::vector<float> ds; s->value.vision_embed(pixels, height, width, mg, ds);
        memcpy(merged, mg.data(), mg.size() * 4); memcpy(deepstack, ds.data(), ds.size() * 4); *tokens = int(m);
    })
}
extern "C" int h3pipe_encode_audio(h3pipe_session *s, const float *samples, int n_samples, float *latents, size_t latent_elements, int *audio_t, char *error, size_t cap) {
    GUARD({
        if (!s || !samples || n_samples < 1 || n_samples > (1 << 30) || !latents || !audio_t) throw std::invalid_argument("session, samples (n_samples 1..2^30), latents and audio_t are required");
        const int T = (n_samples + 799) / 800;
        if (latent_elements < size_t(2) * AUDIO_CH * T) throw std::invalid_argument("latents must hold at least 2 * 32 * ceil(n_samples / 800) floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.encode_audio(samples, n_samples, latents, *audio_t);
    })
}
extern "C" int h3pipe_decode_audio(h3pipe_session *s, const float *latents, size_t audio_elements, int audio_t, float *samples, size_t sample_elements, char *error, size_t cap) {
    GUARD({
        if (!s || !latents || !samples || audio_t < 1 || audio_t > (1 << 24)) throw std::invalid_argument("session, latents (audio_t 1..2^24) and samples are required");
        if (audio_elements < size_t(2) * AUDIO_CH * audio_t) throw std::invalid_argument("audio_latents must hold at least 2 * 32 * audio_t floats");
        if (sample_elements < size_t(2) * audio_t * 800) throw std::invalid_argument("samples must hold at least 2 * audio_t * 800 floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.decode_audio(latents, audio_t, samples);
    })
}
