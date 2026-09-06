// The MiniMax H3 pipeline as one C library: every kernel in Loom (compiled on first use through
// loom-compile into a cache), the host doing only what is under a megabyte per step. One generic
// transformer stack serves the text encoder, the token refiner, the 50 DiT blocks and the video
// decoder; the embedders and the final layer are the same int8 GEMMs with padded K / N.
//
// Build: ./scripts/build_host.sh  (host-only code against the HIP runtime API)
#include "rt.h"

#include <spawn.h>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>

#include "h3pipe.h"

extern char **environ;

namespace {

// --- the model's shapes --------------------------------------------------------------------
constexpr int HID = 5376, HEADS = 56, HEAD_DIM = 128, FFN = 14336, ROPE_DIM = 96, ROPE_HALF = 48;
constexpr int TEXT_DIM = 5120, VIDEO_PATCH = 96, AUDIO_CH = 32, KPAD = 256, FINAL_N = 128;
constexpr int CLASSES = 12, MODALITIES = 3, MODS_ROWS = 6 * CLASSES;
constexpr float VISUAL_COND_AUG = 0.999f;                                  // ComfyUI's VISUAL_COND_TIMESTEP: reference latents at 0.999 * z + 0.001 * noise, timestep class max(t_v, 0.999)      // the AdaLN table rows per layer
constexpr int TE_HID = 5120, TE_HEADS = 64, TE_KV = 8, TE_FFN = 25600, TE_ROPE_HALF = 64;
constexpr int LATENT_CH = 24, FPS = 24, AUDIO_LATENTS_PER_S = 40;
constexpr int VAE_HID = 2048, VAE_HEADS = 32, VAE_D = 64, VAE_FFN = 8192, VAE_ROPE_HALF = 24, VAE_PT = 4, VAE_PS = 16, VAE_OUT = 3 * VAE_PT * VAE_PS * VAE_PS, VAE_REG = 4;
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

// --- weights: manifest + device blob --------------------------------------------------------
struct Span { size_t offset, bytes; std::string dtype, shape; };

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

struct Blob {
    // File offsets remain unchanged for host reads. Device tensors are loaded on first use.
    std::string path; size_t size = 0; std::map<std::string, Span> spans;
    mutable DeviceBuffers memory;
    mutable std::map<std::string, char *> tensors;
    bool is_open() const { return !path.empty(); }
    void open(const std::string &dir) {
        const std::string next_path = dir + "/weights.bin";
        std::ifstream in(dir + "/manifest.txt");
        if (!in) throw std::runtime_error("cannot read " + dir + "/manifest.txt");
        std::map<std::string, Span> next_spans; std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line); std::string name, dtype, shape; size_t offset, bytes;
            if (ls >> name >> offset >> bytes >> dtype >> shape) next_spans[name] = {offset, bytes, dtype, shape};
        }
        std::ifstream f(next_path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() < 0) throw std::runtime_error("cannot read " + next_path);
        const size_t next_size = size_t(f.tellg());
        for (auto &e : next_spans) if (e.second.offset > next_size || e.second.bytes > next_size - e.second.offset)
            throw std::runtime_error("manifest span '" + e.first + "' runs past " + next_path);
        memory.clear(); tensors.clear(); spans.swap(next_spans); path = next_path; size = next_size;
    }
    const Span &span(const std::string &name) const {
        auto it = spans.find(name);
        if (it == spans.end()) throw std::runtime_error("missing tensor " + name + " in " + path);
        return it->second;
    }
    char *at(const std::string &name, size_t bytes) const {
        const Span &s = span(name);
        if (s.bytes != bytes) throw std::runtime_error("tensor " + name + " has " + std::to_string(s.bytes) + " bytes, expected " + std::to_string(bytes));
        auto it = tensors.find(name); if (it != tensors.end()) return it->second;
        std::ifstream f(path, std::ios::binary); f.seekg(std::streamoff(s.offset));
        std::vector<char> chunk(std::min(bytes, size_t(16) << 20));
        char *p = (char *)memory.alloc(std::max(bytes, size_t(1)));
        try {
            for (size_t done = 0; done < bytes;) {
                const size_t n = std::min(chunk.size(), bytes - done);
                f.read(chunk.data(), std::streamsize(n));
                if (!f) throw std::runtime_error("short read of " + name);
                rt().h2d(p + done, chunk.data(), n); done += n;
            }
            tensors.emplace(name, p);
        } catch (...) { memory.free(p); throw; }
        return p;
    }
    std::vector<float> host_f32(const std::string &name, size_t count) const {
        const Span &s = span(name);
        if (s.bytes != count * 4) throw std::runtime_error("tensor " + name + " has " + std::to_string(s.bytes) + " bytes, expected " + std::to_string(count * 4));
        std::vector<float> v(count); std::ifstream f(path, std::ios::binary); f.seekg(std::streamoff(s.offset)); f.read((char *)v.data(), std::streamsize(s.bytes));
        if (!f) throw std::runtime_error("short read of " + name); return v;
    }
};

// --- kernels: compile through loom-compile into the cache, load, launch ---------------------
struct Kernel {
    RtKernel *k = nullptr;
    void load(const std::string &path, const std::string &symbol) { k = rt().load(path, symbol); }
    ~Kernel() { if (k) rt().unload(k); }
};
using Cfg = std::vector<std::pair<std::string, std::string>>;

struct Compiler {
    std::string exe, sources, cache;
    std::map<std::string, std::shared_ptr<Kernel>> loaded;
    // one kernel per (stem, source, config); the cache file name carries a hash of the .loom text and the config,
    // so an edited kernel source never reuses a stale binary
    static std::string source_hash(const std::string &path) {
        std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("missing kernel source " + path);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        uint64_t h = 1469598103934665603ull; for (unsigned char c : text) { h ^= c; h *= 1099511628211ull; }
        char buf[17]; snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h); return std::string(buf, 10);
    }
    std::shared_ptr<Kernel> get(const std::string &stem, const std::string &symbol, const Cfg &cfg) {
        std::string tag = stem + "__s" + source_hash(sources + "/" + stem + ".loom");
        for (auto &c : cfg) { tag += "__" + c.first.substr(c.first.rfind('.') + 1) + "_" + c.second; }
        for (char &ch : tag) if (!isalnum((unsigned char)ch) && ch != '_' && ch != '-' && ch != '.') ch = '_';
        auto it = loaded.find(tag);
        if (it != loaded.end()) return it->second;
        const std::string path = cache + "/" + tag + ".hsaco";
        if (!exists(path)) compile(stem, symbol, cfg, path);
        auto k = std::make_shared<Kernel>(); k->load(path, symbol); loaded[tag] = k; return k;
    }
    void compile(const std::string &stem, const std::string &symbol, const Cfg &cfg, const std::string &path) {
        mkdirs(cache);
        const std::string tmp = path + ".tmp." + std::to_string(getpid());
        std::vector<std::string> args = {exe, sources + "/" + stem + ".loom", "--backend=amdgpu-hal", "--target=gfx1151", "--root=@" + symbol, "--output=" + tmp};
        for (auto &c : cfg) args.push_back("--config=" + c.first + "=" + c.second);
        std::vector<char *> argv; for (auto &a : args) argv.push_back(const_cast<char *>(a.c_str())); argv.push_back(nullptr);
        pid_t pid; if (posix_spawnp(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ) != 0) throw std::runtime_error("cannot spawn " + exe);
        int status = 0; waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || !exists(tmp)) {
            std::string cmd; for (auto &a : args) cmd += a + " ";
            throw std::runtime_error("loom-compile failed for " + stem + ": " + cmd);
        }
        if (rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + tmp);
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
unsigned gemm_grid_y(size_t tokens) { const unsigned g = m_group_for(tokens); return unsigned(((tokens + 255) / 256 + g - 1) / g * g); }

// A prepare kernel (norm / lnorm / plain) for one width, ready to launch.
struct Prepare {
    std::shared_ptr<Kernel> k; int lanes = 0; std::string form;
    void build(Compiler &c, const std::string &form_, int bits, int width, float eps = 1e-5f, int classes = 1) {
        form = form_;
        std::string stem = "prepare_" + form + "_i" + std::to_string(bits);
        if (form == "plain" && size_t(width) * 4 > 65536) stem = "prepare_plain16_i" + std::to_string(bits);   // f16 LDS for rows past 64 KB of f32
        lanes = lanes_for(width);
        const std::string ns = "h3." + stem + ".";
        Cfg cfg = {{ns + "width", std::to_string(width)}, {ns + "lanes", std::to_string(lanes)}};
        if (form != "plain") { cfg.push_back({ns + "eps", num(eps)}); cfg.push_back({ns + "classes", std::to_string(classes)}); }
        k = c.get(stem, "h3_" + stem, cfg);
    }
    // norm forms: (x f32, weight, table, cls) -> a_q, a_s; plain: (h f16) -> a_q, a_s
    void run(Profile *p, const char *stage, unsigned tokens, const void *x, const void *weight, const void *table, const void *cls, void *a_q, void *a_s) {
        KernArgs a; a.i32(int(tokens)).ptr(x);
        if (form != "plain") a.ptr(weight).ptr(table).ptr(cls);
        a.ptr(a_q).ptr(a_s);
        launch(*k, p, stage, tokens, 1, unsigned(lanes), a);
    }
};

// A GEMM of the int4/int8 family for one (K, N, m_group).
struct Gemm {
    std::shared_ptr<Kernel> k; int n = 0; bool resid = false, bias = false;
    void build(Compiler &c, const std::string &mode, int bits, bool bias_, bool gate_first, int k_size, int n_size, size_t tokens, int classes = 1) {
        resid = mode == "resid"; bias = bias_; n = n_size;
        std::string stem = "gemm_i" + std::to_string(bits) + (mode == "plain" ? "" : "_" + mode) + "_256" + (bias ? "b" : "") + (mode == "swiglu" && !gate_first ? "_gs" : "");
        const std::string ns = "h3." + stem + ".";
        Cfg cfg = {{ns + "k_size", std::to_string(k_size)}, {ns + "n_size", std::to_string(n_size)}, {ns + "m_group", std::to_string(m_group_for(tokens))}};
        if (resid) cfg.push_back({ns + "classes", std::to_string(classes)});
        k = c.get(stem, "h3_" + stem, cfg);
    }
    void run(Profile *p, const char *stage, unsigned tokens, const void *a_q, const void *w_q, const void *w_s, const void *a_s, void *out, const void *gate = nullptr, const void *cls = nullptr, const void *b = nullptr) {
        KernArgs a; a.i32(int(tokens)).ptr(a_q).ptr(w_q).ptr(w_s).ptr(a_s).ptr(out);
        if (resid) a.ptr(gate).ptr(cls);
        if (bias) a.ptr(b);
        launch(*k, p, stage, unsigned(n / 128), gemm_grid_y(tokens), THREADS, a);
    }
};

// --- the generic transformer stack -----------------------------------------------------------
struct StackDims {
    int hidden, heads, kv_heads, head_dim, ffn, rope_dim, classes, bits; float eps; bool bias, gate_first, causal; bool attn_i4 = false;   // attn_i4: QK^T in int4 (prepare_qk_i4 operands)
    int inner() const { return heads * head_dim; }
    int kv_inner() const { return kv_heads * head_dim; }
    int qkv() const { return inner() + 2 * kv_inner(); }
};
struct LayerCond { const float *table_msa, *gate_msa, *table_mlp, *gate_mlp; };

class Stack {
    DeviceBuffers memory_;
public:
    Stack(Compiler &c, const StackDims &d, size_t tokens, int layers, const Blob &w, const std::string &prefix_fmt, bool qk_weights, const float *ones_head)
        : d_(d), tokens_(tokens), layers_(layers) {
        const int bits = d.bits, per = bits == 4 ? 2 : 1;
        waves_ = d.causal ? 8 : (tokens >= 4096 ? 8 : 4);
        capacity_ = std::max<size_t>((tokens + 16 + 31) / 32 * 32, (tokens + 16 * waves_ - 1) / (16 * waves_) * (16 * waves_));
        capacity_ = std::max<size_t>(capacity_, (tokens + 255) / 256 * 256);
        for (int i = 0; i < layers; ++i) {
            const std::string p = fmt(prefix_fmt.c_str(), i);
            Block b;
            b.qkv_q = w.at(p + "qkv.q", size_t(d.qkv()) * d.hidden / per);   b.qkv_s = w.at(p + "qkv.s", size_t(d.qkv()) * 4);
            b.out_q = w.at(p + "out.q", size_t(d.hidden) * d.inner() / per); b.out_s = w.at(p + "out.s", size_t(d.hidden) * 4);
            b.gu_q = w.at(p + "gu.q", size_t(2 * d.ffn) * d.hidden / per);   b.gu_s = w.at(p + "gu.s", size_t(2 * d.ffn) * 4);
            b.down_q = w.at(p + "down.q", size_t(d.hidden) * d.ffn / per);   b.down_s = w.at(p + "down.s", size_t(d.hidden) * 4);
            if (d.bias) { b.qkv_b = w.at(p + "qkv.b", size_t(d.qkv()) * 4); b.out_b = w.at(p + "out.b", size_t(d.hidden) * 4); b.gu_b = w.at(p + "gu.b", size_t(2 * d.ffn) * 4); b.down_b = w.at(p + "down.b", size_t(d.hidden) * 4); }
            b.norm1 = w.at(p + "norm1", size_t(d.hidden) * 4); b.norm2 = w.at(p + "norm2", size_t(d.hidden) * 4);
            if (qk_weights) { b.qnorm = w.at(p + "qnorm", size_t(d.head_dim) * 4); b.knorm = w.at(p + "knorm", size_t(d.head_dim) * 4); }
            else b.qnorm = b.knorm = (char *)ones_head;
            if (w.spans.count(p + "scale1")) { b.scale1 = w.at(p + "scale1", size_t(d.hidden) * 4); b.scale2 = w.at(p + "scale2", size_t(d.hidden) * 4); }
            blocks_.push_back(b);
        }
        prep_norm_.build(c, "norm", bits, d.hidden, d.eps, d.classes);
        prep_attn_.build(c, "plain", bits, d.inner());
        prep_down_.build(c, "plain", bits, d.ffn);
        gemm_qkv_.build(c, "plain", bits, d.bias, true, d.hidden, d.qkv(), tokens);
        gemm_gu_.build(c, "swiglu", bits, d.bias, d.gate_first, d.hidden, 2 * d.ffn, tokens);
        gemm_out_.build(c, "resid", bits, d.bias, true, d.inner(), d.hidden, tokens, d.classes);
        gemm_down_.build(c, "resid", bits, d.bias, true, d.ffn, d.hidden, tokens, d.classes);
        {
            const std::string stem = d.head_dim == 64 ? "rope64_qknorm_f16" : (d.rope_dim == 128 ? "rope128_qknorm_f16" : "rope_qknorm_f16");
            const std::string ns = "h3." + stem + ".";
            rope_ = c.get(stem, "h3_" + stem, {{ns + "row_stride", std::to_string(d.qkv())}, {ns + "heads", std::to_string(d.heads)}, {ns + "kv_heads", std::to_string(d.kv_heads)}, {ns + "k_offset", std::to_string(d.inner())}, {ns + "eps", num(d.eps)}});
        }
        if (d.attn_i4) {
            if (d.causal || d.head_dim != 128) throw std::runtime_error("int4 QK^T attention: MHA with head 128 only");
            const std::string pq = "h3.prepare_qk_i4.";
            colmean_ = c.get("colmean_f32", "h3_colmean_f32", {{"h3.colmean_f32.width", std::to_string(d.inner())}});
            prep_q_ = c.get("prepare_qk_i4", "h3_prepare_qk_i4", {{pq + "row_stride", std::to_string(d.inner())}, {pq + "head_offset", "0"}, {pq + "heads", std::to_string(d.heads)}, {pq + "extra_scale", num(1.0 / std::sqrt(double(d.head_dim)) / 128.0)}});
            prep_k_ = c.get("prepare_qk_i4", "h3_prepare_qk_i4", {{pq + "row_stride", std::to_string(d.inner())}, {pq + "head_offset", "0"}, {pq + "heads", std::to_string(d.heads)}, {pq + "extra_scale", "1"}});
            transpose_ = c.get("transpose_f16", "h3_transpose_f16", {{"h3.transpose_f16.width", std::to_string(d.inner())}, {"h3.transpose_f16.row_capacity", std::to_string(capacity_)}});
        }
        {
            std::string stem = d.causal ? "attention_gqa8c_lds_f16_wmma" : (d.head_dim == 64 ? (waves_ == 8 ? "attention_mha648_lds_f16_wmma" : "attention_mha64_lds_f16_wmma") : (waves_ == 8 ? "attention_mha8_lds_f16_wmma" : "attention_mha_lds_f16_wmma"));
            if (d.attn_i4) stem = tokens >= 20000 ? "attention_i4qkl_mha8_lds_f16_wmma" : (waves_ == 8 ? "attention_i4qk_mha8_lds_f16_wmma" : "attention_i4qk_mha_lds_f16_wmma");   // one workgroup per CU past ~20k rows
            // tile skip: H3_ATTN_SKIP_TAU=<tau> selects the skip twin (i4qk -> i4qks) at that tau (tau 4: no measured velocity cost,
            // about 3% on the 768 step against the carried-scale plain kernel); unset or 'off' -> the plain kernel
            double tau = 0.0;
            if (const char *v = std::getenv("H3_ATTN_SKIP_TAU")) { std::string sv(v); for (auto &ch : sv) ch = char(tolower(ch)); tau = (sv == "off" || sv == "none" || sv == "0" || sv == "1e30" || sv.empty()) ? 0.0 : std::atof(v); }
            const bool skip = d.attn_i4 && tau > 0.0;
            if (skip) { const size_t at = stem.find("i4qk"); stem.replace(at, 4, "i4qks"); }
            const std::string ns = "h3." + stem + ".";
            Cfg acfg = {{ns + "q_stride", std::to_string(d.inner())}, {ns + "kv_stride", std::to_string(d.kv_inner())}, {ns + "tokens", std::to_string(tokens)}, {ns + "token_capacity", std::to_string(capacity_)}, {ns + "scale", num(1.0 / std::sqrt(double(d.head_dim)))}, {ns + "out_stride", std::to_string(d.inner())}};
            if (skip) acfg.push_back({ns + "skip_tau", num(tau)});
            attention_ = c.get(stem, "h3_" + stem, acfg);
        }
        const size_t T = capacity_;
        a_q_ = memory_.alloc(T * size_t(std::max(d.ffn, std::max(d.hidden, d.inner()))) / per);
        a_s_ = memory_.alloc(T * 4);
        fused_ = memory_.alloc(T * size_t(d.qkv()) * 2);
        q_ = memory_.alloc(T * size_t(d.inner()) * 2);
        k_ = memory_.alloc(T * size_t(d.kv_inner()) * 2);
        v_ = memory_.alloc(T * size_t(d.kv_inner()) * 2);
        attn_ = memory_.alloc(T * size_t(d.inner()) * 2);
        gu_ = memory_.alloc(T * size_t(d.ffn) * 2);
        for (auto p : {fused_, q_, k_, v_, attn_}) rt().memset(p, 0, T * size_t(p == fused_ ? d.qkv() : (p == q_ || p == attn_ ? d.inner() : d.kv_inner())) * 2);
        if (d.attn_i4) {
            qi_ = memory_.alloc(T * size_t(d.heads) * 64); ki_ = memory_.alloc(T * size_t(d.heads) * 64); qs_ = memory_.alloc(T * size_t(d.heads) * 4); ks_ = memory_.alloc(T * size_t(d.heads) * 4);
            kmean_ = memory_.alloc(size_t(d.inner()) * 4); zmean_ = memory_.alloc(size_t(d.inner()) * 4); rt().memset(zmean_, 0, size_t(d.inner()) * 4);
            vt_ = memory_.alloc(size_t(d.inner()) * T * 2); rt().memset(vt_, 0, size_t(d.inner()) * T * 2);
            for (auto p : {qi_, ki_}) rt().memset(p, 0, T * size_t(d.heads) * 64);
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
        static const char *dump_blocks = std::getenv("H3_DUMP_BLOCKS");   // H3_DUMP_BLOCKS=<dir>: x before the first block (h_in) and after every block (blk_NN), [tokens][hidden] f32, first call only
        static int dump_calls = 0; const bool dumping = dump_blocks && dump_calls == 0 && layers_ == 50 && d_.classes > 1;   // the DiT, not the 50-layer text encoder if (dumping) ++dump_calls;
        auto dump_x = [&](const std::string &name) { if (!dumping) return; std::vector<float> hbuf(size_t(T) * d_.hidden); rt().sync(); rt().d2h(hbuf.data(), x, hbuf.size() * 4);
            if (FILE *f = fopen((std::string(dump_blocks) + "/" + name + ".f32").c_str(), "wb")) { fwrite(hbuf.data(), 4, hbuf.size(), f); fclose(f); } };
        dump_x("h_in");
        for (int i = first; i < last; ++i) {
            const Block &b = blocks_[i]; const LayerCond lc = cond(i);
            prep_norm_.run(prof, "prepare norm", T, x, b.norm1, lc.table_msa, cls, a_q_, a_s_);
            gemm_qkv_.run(prof, "gemm qkv", T, a_q_, b.qkv_q, b.qkv_s, a_s_, fused_, nullptr, nullptr, b.qkv_b);
            { KernArgs a; a.i32(int(T)).ptr(fused_).ptr(b.qnorm).ptr(b.knorm).ptr(cos).ptr(sin).ptr(q_).ptr(k_).ptr(v_); launch(*rope_, prof, "qk norm + rope", T, 1, THREADS, a); }
            if (d_.attn_i4) {
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
            prep_attn_.run(prof, "prepare out input", T, attn_, nullptr, nullptr, nullptr, a_q_, a_s_);
            gemm_out_.run(prof, "gemm out + residual", T, a_q_, b.out_q, b.out_s, a_s_, x, lc.gate_msa, cls, b.out_b);
            prep_norm_.run(prof, "prepare norm", T, x, b.norm2, lc.table_mlp, cls, a_q_, a_s_);
            gemm_gu_.run(prof, "gemm ff + swiglu", T, a_q_, b.gu_q, b.gu_s, a_s_, gu_, nullptr, nullptr, b.gu_b);
            prep_down_.run(prof, "prepare down input", T, gu_, nullptr, nullptr, nullptr, a_q_, a_s_);
            gemm_down_.run(prof, "gemm down + residual", T, a_q_, b.down_q, b.down_s, a_s_, x, lc.gate_mlp, cls, b.down_b);
            dump_x(fmt("blk_%02d", i));
        }
    }

private:
    StackDims d_; size_t tokens_, capacity_ = 0; int layers_, waves_ = 4;
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

struct Schedule {                            // diffusers' MiniMaxH3Scheduler
    std::vector<float> sigmas, timesteps;
    Schedule(int steps, double shift) {
        std::vector<float> s;
        for (int i = 0; i < steps; ++i) { const double base = steps == 1 ? 1.0 : 1.0 - double(i) / (steps - 1); s.push_back(float(shift * base / (1.0 + (shift - 1.0) * base))); }
        for (float v : s) if (sigmas.empty() || v != sigmas.back()) sigmas.push_back(v);
        for (size_t i = 0; i + 1 < sigmas.size(); ++i) timesteps.push_back(1.0f - sigmas[i]);
    }
};

// --- the pipeline ----------------------------------------------------------------------------
class Pipe {
    DeviceBuffers memory_;
public:
    explicit Pipe(const h3pipe_config &cfg) {
        if (!cfg.glue_dir || !cfg.blocks_dir || !cfg.te_dir || !cfg.kernel_sources || !cfg.cache_dir || !cfg.loom_compile) throw std::invalid_argument("every directory and the loom-compile path are required");
        rt();
        comp_.exe = cfg.loom_compile; comp_.sources = cfg.kernel_sources; comp_.cache = cfg.cache_dir;
        vae_dir_ = cfg.vae_dir ? cfg.vae_dir : ""; vae_bits_ = cfg.vae_bits ? cfg.vae_bits : 8;
        aenc_dir_ = cfg.aenc_dir ? cfg.aenc_dir : ""; vision_dir_ = cfg.vision_dir ? cfg.vision_dir : ""; venc_dir_ = cfg.venc_dir ? cfg.venc_dir : "";
        attn_qk_bits_ = cfg.attn_qk_bits ? cfg.attn_qk_bits : 4; if (attn_qk_bits_ != 4 && attn_qk_bits_ != 16) throw std::invalid_argument("attn_qk_bits must be 4 or 16");
        glue_.open(cfg.glue_dir);
        embed_ = glue_.span("te.embed");
        blocks_dir_ = cfg.blocks_dir;
        te_dir_ = cfg.te_dir;
        ones_ = memory_.alloc(size_t(HID) * 4); { std::vector<float> o(HID, 1.0f); rt().h2d(ones_, o.data(), o.size() * 4); }
        zeros_ = memory_.alloc(size_t(2 * TE_FFN) * 4); rt().memset(zeros_, 0, size_t(2 * TE_FFN) * 4);
    }


    Profile prof{std::getenv("H3_PROFILE") != nullptr && *std::getenv("H3_PROFILE") != 0};   // H3_PROFILE=1: synchronized per-stage times printed after every step

    // --- the prompt: embedding lookup on the host, the encoder, condition_proj, the refiner ---
    void ensure_seq(size_t seq) {
        if (seq <= seq_cap_) return;
        for (void **p : {&x_, &cls_, &cls0_, &tcls_, &cos_, &sin_, &in16_, &a_q_, &a_s_, &out16_, &text_copy_}) { if (*p) memory_.free(*p); *p = nullptr; }
        seq_cap_ = 0;
        const size_t T = (seq + 255) / 256 * 256 + 32;
        x_ = memory_.alloc(T * HID * 4); rt().memset(x_, 0, T * HID * 4);
        cls_ = memory_.alloc(T * 4); rt().memset(cls_, 0, T * 4);
        cls0_ = memory_.alloc(T * 4); rt().memset(cls0_, 0, T * 4);       // the single-class GEMMs (embedders, condition proj) index their gate table with this
        tcls_ = memory_.alloc(T * 4); rt().memset(tcls_, 0, T * 4);
        cos_ = memory_.alloc(T * ROPE_HALF * 4); sin_ = memory_.alloc(T * ROPE_HALF * 4);
        in16_ = memory_.alloc(T * TEXT_DIM * 2); rt().memset(in16_, 0, T * TEXT_DIM * 2);
        a_q_ = memory_.alloc(T * size_t(std::max(TEXT_DIM, HID))); a_s_ = memory_.alloc(T * 4);   // the final norm writes HID-wide rows, the embedders TEXT_DIM-wide
        out16_ = memory_.alloc(T * FINAL_N * 2);
        seq_cap_ = T;
    }

    // x rows [row0, row0 + rows) = W · in + b through the int8 family: in f16 [rows][k] on in16_, rows zeroed first
    void embed_linear(const char *stage, size_t row0, size_t rows, int k, const std::string &wname) {
        Prepare &prep = prepares_[fmt("plain_%d_%zu", k, rows)]; if (!prep.k) prep.build(comp_, "plain", 8, k);
        Gemm &g = gemms_[fmt("resid_%d_%zu", k, rows)]; if (!g.k) g.build(comp_, "resid", 8, true, true, k, HID, rows, 1);
        prep.run(&prof, stage, unsigned(rows), in16_, nullptr, nullptr, nullptr, a_q_, a_s_);
        float *xr = (float *)x_ + row0 * HID;
        rt().memset(xr, 0, rows * HID * 4);
        g.run(&prof, stage, unsigned(rows), a_q_, glue_.at(wname + ".q", size_t(HID) * k), glue_.at(wname + ".s", size_t(HID) * 4), a_s_, xr, ones_, cls0_, glue_.at(wname + ".b", size_t(HID) * 4));
    }

    struct VisionSpan { size_t start, count; int merged_h, merged_w; const float *merged, *deepstack; };   // an image's rows in the presentation
    void text_in(const int32_t *ids, int n, const std::vector<VisionSpan> &spans = {}) {
        // embedding rows from the file (bf16) -> f32; vision spans take the merged vision embeds
        std::vector<float> emb(size_t(n) * TEXT_DIM);
        for (const VisionSpan &sp : spans) memcpy(emb.data() + sp.start * TEXT_DIM, sp.merged, sp.count * TEXT_DIM * 4);
        { std::ifstream f(glue_.path, std::ios::binary); std::vector<uint16_t> row(TEXT_DIM);
          for (int i = 0; i < n; ++i) { if (ids[i] < 0) { bool in_span = false; for (const VisionSpan &sp : spans) in_span |= size_t(i) >= sp.start && size_t(i) < sp.start + sp.count; if (in_span) continue; }
              if (ids[i] < 0 || size_t(ids[i]) * TEXT_DIM * 2 >= embed_.bytes) throw std::invalid_argument("token id out of range: " + std::to_string(ids[i]));
              f.seekg(std::streamoff(embed_.offset + size_t(ids[i]) * TEXT_DIM * 2)); f.read((char *)row.data(), TEXT_DIM * 2); if (!f) throw std::runtime_error("short read of the embedding table");
              for (int j = 0; j < TEXT_DIM; ++j) emb[size_t(i) * TEXT_DIM + j] = bf16_to_f32(row[j]); } }
        // the encoder's 50 layers
        std::string span_sig; for (const VisionSpan &sp : spans) span_sig += fmt("%zu:%zu:%d:%d;", sp.start, sp.count, sp.merged_h, sp.merged_w);
        if (!te_ready_ || !te_ || te_->tokens() != size_t(n) || te_span_sig_ != span_sig) {
            te_span_sig_.clear(); te_ready_ = false;
            if (!te_blob_.is_open()) te_blob_.open(te_dir_);
            te_.reset(); te_ = std::make_unique<Stack>(comp_, StackDims{TE_HID, TE_HEADS, TE_KV, HEAD_DIM, TE_FFN, 128, 1, 8, 1e-6f, false, true, true}, size_t(n), 50, te_blob_, "blocks.%d.", true, nullptr);
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
        // hidden -> f16 -> condition_proj into x rows [0, n)
        std::vector<float> hid(size_t(n) * TE_HID); rt().sync(); rt().d2h(hid.data(), te_x_, hid.size() * 4);
        std::vector<uint16_t> h16(hid.size()); for (size_t i = 0; i < hid.size(); ++i) h16[i] = f32_to_f16(hid[i]);
        rt().h2d(in16_, h16.data(), h16.size() * 2);
        embed_linear("condition proj", 0, size_t(n), TEXT_DIM, "h3.cond");
        // the token refiner: two H3-shaped blocks without rope (identity tables), then its final norm
        if (!refiner_ready_ || !refiner_ || refiner_->tokens() != size_t(n)) {
            refiner_ready_ = false;
            refiner_.reset(); refiner_ = std::make_unique<Stack>(comp_, StackDims{HID, HEADS, HEADS, HEAD_DIM, FFN, ROPE_DIM, 1, 8, 1e-5f, false, true, false}, size_t(n), 2, glue_, "h3.refiner.%d.", true, nullptr);
            std::vector<float> c(size_t(n) * ROPE_HALF, 1.0f), s(size_t(n) * ROPE_HALF, 0.0f);
            for (void **q : {&ref_cos_, &ref_sin_}) { memory_.free(*q); *q = nullptr; }
            ref_cos_ = memory_.alloc(c.size() * 4); ref_sin_ = memory_.alloc(s.size() * 4);
            rt().h2d(ref_cos_, c.data(), c.size() * 4); rt().h2d(ref_sin_, s.data(), s.size() * 4);
            const std::string ns = "h3.norm_mod_f32.";
            norm_f32_ = comp_.get("norm_mod_f32", "h3_norm_mod_f32", {{ns + "width", std::to_string(HID)}, {ns + "lanes", std::to_string(lanes_for(HID))}, {ns + "eps", num(1e-5)}, {ns + "classes", "1"}});
        }
        refiner_ready_ = true;
        refiner_->forward(&prof, x_, cls0_, ref_cos_, ref_sin_, [&](int) { return LayerCond{(const float *)zeros_, (const float *)ones_, (const float *)zeros_, (const float *)ones_}; });
        { KernArgs a; a.i32(n).ptr(x_).ptr(glue_.at("h3.refiner.final_norm", size_t(HID) * 4)).ptr(zeros_).ptr(cls0_); launch(*norm_f32_, &prof, "refiner final norm", unsigned(n), 1, unsigned(lanes_for(HID)), a); }
    }

    void get_text_in(const int32_t *ids, int n, float *out) {
        ensure_seq(size_t(n));
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
        // host-side conditioning tables
        curve_ = glue_.host_f32("h3.adaln_t_table", 1025 * 8);
        inv_freq_ = glue_.host_f32("h3.rope_inv_freq", 16);
        for (int i = 0; i < 50; ++i) { adaln_w_.push_back(glue_.host_f32(fmt("h3.blocks.%d.adaln.w", i), size_t(3 * 6 * HID) * 8)); adaln_b_.push_back(glue_.host_f32(fmt("h3.blocks.%d.adaln.b", i), size_t(3 * 6 * HID))); }
        final_w_ = glue_.host_f32("h3.final.adaln.w", size_t(2 * HID) * 8); final_b_ = glue_.host_f32("h3.final.adaln.b", size_t(2 * HID));
        if (!mods_) mods_ = memory_.alloc(size_t(50) * MODS_ROWS * HID * 4);
        if (!final_table_) final_table_ = memory_.alloc(size_t(4) * HID * 4);
        if (!blocks_.is_open()) blocks_.open(blocks_dir_);
        conditioning_ready_ = true;
    }

    // --- denoising ---
    void denoise(const int32_t *ids, int n, const h3pipe_params &p, const float *noise_video, const float *noise_audio, float *video_out, float *audio_out, h3pipe_progress progress, void *user, const std::vector<h3pipe_ref> &refs = {}, const std::vector<h3pipe_keyframe> &kfs = {}) {
        if (p.height % 32 || p.width % 32 || p.height < 64 || p.width < 64) throw std::invalid_argument("height and width must be multiples of 32");
        if (p.steps < 2 || p.steps > 1000) throw std::invalid_argument("steps must be 2..1000");
        h3pipe_shape sh; h3pipe_shape_for(&p, &sh);
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
            const char *qk = std::getenv("H3_ATTN_QK"); StackDims dd{HID, HEADS, HEADS, HEAD_DIM, FFN, ROPE_DIM, CLASSES, 4, 1e-5f, false, true, false}; dd.attn_i4 = qk ? !(std::string(qk) == "f16") : attn_qk_bits_ == 4;   // H3_ATTN_QK=f16|i4 overrides the config
            dd.bits = blocks_.span("blocks.0.qkv.q").bytes == size_t(dd.qkv()) * dd.hidden ? 8 : 4;   // the export's width: int8 rows verbatim (export_weights.py --bits 8) or packed int4
            dit_.reset(); dit_ = std::make_unique<Stack>(comp_, dd, S, 50, blocks_, "blocks.%d.", true, nullptr); }
        if (!final_prep_.k) { final_prep_.build(comp_, "norm", 8, HID, 1e-5f, 2); }
        const size_t generated_rows = Na + Nv;   // the final head has only the video/audio timestep classes
        Gemm &final_gemm = gemms_[fmt("final_%zu", generated_rows)]; if (!final_gemm.k) final_gemm.build(comp_, "plain", 8, true, true, HID, FINAL_N, generated_rows);
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
        std::vector<uint16_t> in16(in_rows * KPAD); std::vector<uint16_t> out16(generated_rows * FINAL_N);
        const auto t_start = std::chrono::steady_clock::now();
        auto hnow = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        for (size_t step = 0; step < sv.timesteps.size(); ++step) {
            const double h0 = hnow(); double h_mods = 0, h_embed = 0, h_dit = 0, h_final = 0;
            float tv[8], ta[8], tcv[8], tca[8]; temb(sv.timesteps[step], tv); temb(sa.timesteps[step], ta);
            temb(std::max(sv.timesteps[step], VISUAL_COND_AUG), tcv); temb(std::max(sa.timesteps[step], 1.0f), tca); upload_mods(tv, ta, tcv, tca);
            h_mods = hnow();
            // audio rows -> x[L, L+Na), video rows -> x[L+Na, S), each through prepare(256) + the padded-K int8 GEMM
            std::fill(in16.begin(), in16.end(), 0);
            if (res) { const float carry = sa.sigmas[step] / sv.sigmas[step]; for (size_t i = 0; i < arows.size(); ++i) arows[i] = yrows[i] * carry; }
            for (size_t r = 0; r < Na; ++r) for (int k = 0; k < AUDIO_CH; ++k) in16[r * KPAD + k] = f32_to_f16(arows[r * AUDIO_CH + k]);
            rt().h2d(in16_, in16.data(), Na * KPAD * 2);
            embed_linear("audio in", LR, Na, KPAD, "h3.audio_in");
            std::fill(in16.begin(), in16.end(), 0);
            for (size_t r = 0; r < Nv; ++r) for (int k = 0; k < VIDEO_PATCH; ++k) in16[r * KPAD + k] = f32_to_f16(vrows[r * VIDEO_PATCH + k]);
            rt().h2d(in16_, in16.data(), Nv * KPAD * 2);
            embed_linear("video in", LR + Na, Nv, KPAD, "h3.video_in");
            // the text rows are refreshed from a copy each step (the blocks update x in place)
            if (step == 0) {
                // reference rows (constant across steps, re-set every step like the text): the packed latents through the
                // patch projections; visual ones mixed with seeded noise at 0.999 as ComfyUI's condition augmentation
                for (const RefSeg &sg : lay.ref_segs) {
                    h3pipe_ref rf{};
                    if (sg.kind == 3) { const h3pipe_keyframe &kf = kfs[size_t(sg.ref)]; rf.kind = 0; rf.video_latent = kf.video_latent; rf.latent_t = 1; rf.lat_h = sh.lat_h; rf.lat_w = sh.lat_w; rf.audio_latent = kf.audio_latent; rf.audio_t = kf.audio_t; }
                    else rf = refs[size_t(sg.ref)];
                    std::fill(in16.begin(), in16.end(), 0);
                    if (sg.audio) {
                        for (int c = 0; c < 2; ++c) for (int t = 0; t < sg.audio_t; ++t) for (int k = 0; k < AUDIO_CH; ++k) in16[(size_t(c) * sg.audio_t + t) * KPAD + k] = f32_to_f16(rf.audio_latent[(size_t(c) * AUDIO_CH + k) * sg.audio_t + t]);
                        rt().h2d(in16_, in16.data(), sg.rows * KPAD * 2); embed_linear("ref audio in", sg.row0, sg.rows, KPAD, "h3.audio_in");
                    } else {
                        Rng arng(p.seed); const int vt = sg.latent_t, hh = sg.lat_h, ww = sg.lat_w;
                        for (int c = 0; c < LATENT_CH; ++c) for (int t = 0; t < vt; ++t) for (int y = 0; y < hh; ++y) for (int x = 0; x < ww; ++x) {
                            const size_t row = (size_t(t) * (hh / 2) + y / 2) * (ww / 2) + x / 2, col = size_t(c) * 4 + (y % 2) * 2 + (x % 2);
                            const float z = rf.video_latent[((size_t(c) * vt + t) * hh + y) * ww + x];
                            in16[row * KPAD + col] = f32_to_f16(VISUAL_COND_AUG * z + (1.0f - VISUAL_COND_AUG) * arng.normal()); }
                        rt().h2d(in16_, in16.data(), sg.rows * KPAD * 2); embed_linear("ref video in", sg.row0, sg.rows, KPAD, "h3.video_in");
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
            final_prep_.run(&prof, "final norm", unsigned(generated_rows), (float *)x_ + LR * HID, glue_.at("h3.final.norm", size_t(HID) * 4), final_table_, (int32_t *)tcls_ + LR, a_q_, a_s_);
            final_gemm.run(&prof, "final out", unsigned(generated_rows), a_q_, glue_.at("h3.final.out.q", size_t(FINAL_N) * HID), glue_.at("h3.final.out.s", size_t(FINAL_N) * 4), a_s_, out16_, nullptr, nullptr, glue_.at("h3.final.out.b", size_t(FINAL_N) * 4));
            rt().sync(); rt().d2h(out16.data(), out16_, generated_rows * FINAL_N * 2);
            h_final = hnow();
            const float sg_v = sv.sigmas[step], sg_next = sv.sigmas[step + 1], r_v = sg_next / sg_v, sg_a = sa.sigmas[step], r_a = sa.sigmas[step + 1] / sg_a;
            if (!res) {
                // Euler step per schedule: x0 = x + sigma * v, x' = r x + (1 - r) x0
                for (size_t r = 0; r < Nv; ++r) for (int k = 0; k < VIDEO_PATCH; ++k) { float &x = vrows[r * VIDEO_PATCH + k]; const float v = f16_to_f32(out16[(Na + r) * FINAL_N + k]); x = r_v * x + (1.0f - r_v) * (x + sg_v * v); }
                for (size_t r = 0; r < Na; ++r) for (int k = 0; k < AUDIO_CH; ++k) { float &x = arows[r * AUDIO_CH + k]; const float v = f16_to_f32(out16[r * FINAL_N + VIDEO_PATCH + k]); x = r_a * x + (1.0f - r_a) * (x + sg_a * v); }
            } else {
                // ComfyUI (comfy/k_diffusion/sampling.py res_multistep, eta 0): denoised D = X - sigma_v * OUT over the pack. The video's OUT is
                // -v. The audio's carried variable y = (sigma_v / sigma_a) x_a sees OUT_a = (1 - scale) x_a + (1 + (scale - 1) sigma_a) (-v_a),
                // scale = shift_v / shift_a (comfy/ldm/minimax/model.py forward); the network itself sees x_a and t_a = 1 - sigma_a.
                den_v.resize(vrows.size()); den_a.resize(arows.size());
                for (size_t i = 0; i < vrows.size(); ++i) { const size_t r = i / VIDEO_PATCH, k = i % VIDEO_PATCH; den_v[i] = vrows[i] + sg_v * f16_to_f32(out16[(Na + r) * FINAL_N + k]); }
                for (size_t i = 0; i < arows.size(); ++i) { const size_t r = i / AUDIO_CH, k = i % AUDIO_CH; const float v = f16_to_f32(out16[r * FINAL_N + VIDEO_PATCH + k]);
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
        auto Wt = [&](const std::string &nm, size_t bytes) { return venc_.at(nm, bytes); };
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
        std::vector<float> lmean(24), lstd(24); rt().d2h(lmean.data(), Wt("venc.latents_mean", 24 * 4), 24 * 4); rt().d2h(lstd.data(), Wt("venc.latents_std", 24 * 4), 24 * 4);
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
        if (!venc_open_) { if (venc_dir_.empty()) throw std::runtime_error("no video encoder weights: h3pipe_config.venc_dir is NULL"); venc_.open(venc_dir_); venc_open_ = true; }
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
        const std::string stem = std::string("matmul_") + kind + "_f16_wmma", ns = "h3." + stem + ".";
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
        if (!vision_open_) { if (vision_dir_.empty()) throw std::runtime_error("no vision tower weights: h3pipe_config.vision_dir is NULL"); vision_.open(vision_dir_); vision_open_ = true; }
        if (H % 32 || W % 32 || H < 32 || W < 32) throw std::invalid_argument("vision images need height and width multiples of 32");
        const int gh = H / 16, gw = W / 16, n = gh * gw, m = n / 4; const size_t cap = (size_t(n) + 16 + 31) / 32 * 32;
        auto V = [&](const std::string &nm, size_t bytes) { return vision_.at(nm, bytes); };
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
        std::vector<float> pos(vision_.host_f32("vis.pos", size_t(2304) * VHID)), x0(size_t(n) * VHID); rt().d2h(x0.data(), x32, x0.size() * 4);
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
    void matmul(size_t m, int kdim, int n, const void *x, const void *w, const void *b, void *out) {
        const std::string ns = "h3.matmul_f32.";
        auto k = comp_.get("matmul_f32", "h3_matmul_f32", {{ns + "k", std::to_string(kdim)}, {ns + "n", std::to_string(n)}});
        KernArgs a; a.i32(int(m)).ptr(x).ptr(w).ptr(b).ptr(out); launch(*k, &prof, "aenc matmul", unsigned((n + 255) / 256), unsigned(m), THREADS, a);
    }
    void transpose_f32(size_t rows, int cols, const void *x, void *out) {
        const std::string ns = "h3.transpose_f32.";
        auto k = comp_.get("transpose_f32", "h3_transpose_f32", {{ns + "cols", std::to_string(cols)}});
        KernArgs a; a.i32(int(rows)).ptr(x).ptr(out); launch(*k, &prof, "aenc transpose", unsigned((rows * size_t(cols) + 255) / 256), 1, THREADS, a);
    }
    void encode_audio(const float *samples, int n, float *out, int &audio_t) {
        if (!aenc_open_) { if (aenc_dir_.empty()) throw std::runtime_error("no audio encoder weights: h3pipe_config.aenc_dir is NULL"); aenc_.open(aenc_dir_); aenc_open_ = true; }
        const size_t Lp = (size_t(n) + 799) / 800 * 800; const int T = int(Lp / 800); audio_t = T;
        auto W = [&](const std::string &nm, size_t count) { return aenc_.at(nm, count * 4); };
        DeviceBuffers temporary; auto buf = [&](size_t floats) { return temporary.alloc(floats * 4); };
        const size_t plane = size_t(64) * Lp;
        void *x0 = buf(Lp), *h = buf(plane), *h2 = buf(plane), *y = buf(plane), *y2 = buf(plane);
        void *rows = buf(size_t(T) * 2048), *n1 = buf(size_t(T) * 2048), *qkv = buf(size_t(T) * 6144), *pattn = buf(size_t(8) * T * T), *pool = buf(size_t(T) * 32);
        void *xa = buf(size_t(T) * 32), *xb = buf(size_t(T) * 32), *xc = buf(size_t(T) * 32), *a0 = buf(size_t(T) * 64), *a1 = buf(size_t(T) * 64), *g = buf(size_t(T) * 64);
        static const int rates[5] = {2, 4, 4, 5, 5};
        std::vector<float> host(Lp), zrow(size_t(T) * 32);
        const float *lmean = nullptr; std::vector<float> mean_h(32), std_h(32);
        rt().d2h(mean_h.data(), W("aenc.latents_mean", 32), 32 * 4); rt().d2h(std_h.data(), W("aenc.latents_std", 32), 32 * 4); (void)lmean;
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
            matmul(T, 2048, 32, n1, W("aenc.pre.proj.w", size_t(32) * 2048), W("aenc.pre.proj.b", 32), xa);
            layernorm(T, 2048, rows, W("aenc.pre.norm1.w", 2048), W("aenc.pre.norm1.b", 2048), n1);
            matmul(T, 2048, 6144, n1, W("aenc.pre.qkv.w", size_t(6144) * 2048), W("aenc.pre.qkv.b", 6144), qkv);
            { const std::string ns = "h3.attn_scores_f32."; auto k = comp_.get("attn_scores_f32", "h3_attn_scores_f32", {{ns + "heads", "8"}, {ns + "hd", "256"}, {ns + "scale", num(1.0 / 16.0)}});
              KernArgs a; a.i32(T).ptr(qkv).ptr(pattn); launch(*k, &prof, "aenc attention", unsigned((T + 255) / 256), 8, THREADS, a); }
            { const std::string ns = "h3.attn_pv_pool_f32."; auto k = comp_.get("attn_pv_pool_f32", "h3_attn_pv_pool_f32", {{ns + "heads", "8"}, {ns + "hd", "256"}, {ns + "pool", "8"}});
              KernArgs a; a.i32(T).ptr(qkv).ptr(pattn).ptr(pool); launch(*k, &prof, "aenc attention", unsigned(T), 1, 32, a); }
            matmul(T, 32, 32, pool, W("aenc.pre.attn_proj.w", 32 * 32), W("aenc.pre.attn_proj.b", 32), xb);
            axpy(1.0f, 1.0f, size_t(T) * 32, xb, xa);                                                       // xa = proj + attn
            layernorm(T, 32, xa, W("aenc.pre.norm2.w", 32), W("aenc.pre.norm2.b", 32), xb);
            layernorm(T, 32, xb, W("aenc.pre.mlp.norm.w", 32), W("aenc.pre.mlp.norm.b", 32), xc);
            matmul(T, 32, 64, xc, W("aenc.pre.mlp.w0.w", 64 * 32), W("aenc.pre.mlp.w0.b", 64), a0);
            matmul(T, 32, 64, xc, W("aenc.pre.mlp.w1.w", 64 * 32), W("aenc.pre.mlp.w1.b", 64), a1);
            { auto k = comp_.get("geglu_tanh_f32", "h3_geglu_tanh_f32", {}); KernArgs a; a.i32(T * 64).ptr(a0).ptr(a1).ptr(g); launch(*k, &prof, "aenc geglu", unsigned((T * 64 + 255) / 256), 1, THREADS, a); }
            matmul(T, 64, 32, g, W("aenc.pre.mlp.w2.w", 32 * 64), W("aenc.pre.mlp.w2.b", 32), xb);
            axpy(1.0f, 1.0f, size_t(T) * 32, xb, xa);                                                       // xa += mlp
            matmul(T, 32, 32, xa, W("aenc.mean_proj.w", 32 * 32), W("aenc.mean_proj.b", 32), xc);
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
        const void *fir = glue_.at("audio.fir", 12 * 4);
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
        std::vector<float> lmean = glue_.host_f32("audio.latents_mean", 32), lstd = glue_.host_f32("audio.latents_std", 32);
        std::vector<float> in(size_t(32) * T), out(L_out);
        for (int ch = 0; ch < 2; ++ch) {
            for (int c = 0; c < 32; ++c) for (int t = 0; t < T; ++t) in[size_t(c) * T + t] = latents[(size_t(ch) * 32 + c) * T + t] * lstd[c] + lmean[c];
            rt().h2d(ain_, in.data(), in.size() * 4);
            // dec_in_proj (32 -> 2048, k 1), conv_pre (2048 -> 1024, k 7)
            conv_run(conv_kernel(32, 2048, 1, 1, 0, false, round256(T)), "audio dec_in_proj", T, ain_, glue_.at("audio.dec_in_proj.w", size_t(2048) * 32 * 4), glue_.at("audio.dec_in_proj.b", 2048 * 4), ar_);
            conv_run(conv_kernel(2048, 1024, 7, 1, 3, false, round256(T)), "audio conv_pre", T, ar_, glue_.at("audio.conv_pre.w", size_t(1024) * 2048 * 7 * 4), glue_.at("audio.conv_pre.b", 1024 * 4), ah_);
            size_t len = size_t(T); int C = 1024;
            for (int i = 0; i < 7; ++i) {
                const int Cout = C / 2, k = UPK[i], rate = RATES[i], pad = (k - rate) / 2; const size_t olen = (len - 1) * rate + k - 2 * pad;
                { const std::string ns = "h3.convt1d_f32.";
                  auto kt = comp_.get("convt1d_f32", "h3_convt1d_f32", {{ns + "cin", std::to_string(C)}, {ns + "cout", std::to_string(Cout)}, {ns + "ksize", std::to_string(k)}, {ns + "stride", std::to_string(rate)}, {ns + "pad", std::to_string(pad)}, {ns + "len_bound", std::to_string(round256(olen))}});
                  KernArgs a; a.i32(int(len)).i32(int(olen)).ptr(ah_).ptr(glue_.at(fmt("audio.ups.%d.w", i), size_t(C) * Cout * k * 4)).ptr(glue_.at(fmt("audio.ups.%d.b", i), size_t(Cout) * 4)).ptr(ar_);
                  launch(*kt, &prof, "audio upsample", unsigned((olen + 255) / 256), unsigned(Cout), THREADS, a); }
                len = olen; C = Cout; const size_t plane = size_t(C) * len;
                rt().d2d(ah_, ar_, plane * 4); rt().memset(aacc_, 0, plane * 4);
                for (int j = 0; j < 3; ++j) {                                   // the three AMP blocks, averaged
                    const int r = i * 3 + j, kk = RESK[j];
                    rt().d2d(ahj_, ah_, plane * 4);
                    for (int d = 0; d < 3; ++d) {
                        const std::string act1 = fmt("audio.res.%d.act.%d.", r, 2 * d), act2 = fmt("audio.res.%d.act.%d.", r, 2 * d + 1);
                        snake(C, len, ahj_, glue_.at(act1 + "alpha", size_t(C) * 4), glue_.at(act1 + "beta", size_t(C) * 4), atmp_, ar_);
                        conv_run(conv_kernel(C, C, kk, DIL[d], (kk * DIL[d] - DIL[d]) / 2, false, round256(len)), "audio res conv1", len, ar_, glue_.at(fmt("audio.res.%d.c1.%d.w", r, d), size_t(C) * C * kk * 4), glue_.at(fmt("audio.res.%d.c1.%d.b", r, d), size_t(C) * 4), ar2_);
                        snake(C, len, ar2_, glue_.at(act2 + "alpha", size_t(C) * 4), glue_.at(act2 + "beta", size_t(C) * 4), atmp_, ar_);
                        conv_run(conv_kernel(C, C, kk, 1, (kk - 1) / 2, true, round256(len)), "audio res conv2", len, ar_, glue_.at(fmt("audio.res.%d.c2.%d.w", r, d), size_t(C) * C * kk * 4), glue_.at(fmt("audio.res.%d.c2.%d.b", r, d), size_t(C) * 4), ahj_);
                    }
                    axpy(1.0f, 1.0f, plane, ahj_, aacc_);
                }
                axpy(1.0f / 3.0f, 0.0f, plane, aacc_, ah_);
            }
            snake(C, len, ah_, glue_.at("audio.post.alpha", size_t(C) * 4), glue_.at("audio.post.beta", size_t(C) * 4), atmp_, ar_);
            conv_run(conv_kernel(C, 1, 7, 1, 3, false, round256(len)), "audio conv_post", len, ar_, glue_.at("audio.conv_post.w", size_t(C) * 7 * 4), zeros_, ar2_);
            if (len != L_out) throw std::runtime_error("audio length " + std::to_string(len) + " != " + std::to_string(L_out));
            rt().sync(); rt().d2h(out.data(), ar2_, L_out * 4);
            for (size_t i = 0; i < L_out; ++i) samples[size_t(ch) * L_out + i] = std::min(std::max(out[i], -1.0f), 1.0f);
        }
    }

    // --- the video decoder: diffusers' chunking and heads around the 36 blocks in Loom ---
    // one clip: model-space latents z [24][ft][h][w] (already * std + mean) -> ImageNet-space frames [3][ft*4][h*16][w*16]
    void decode_clip(const float *z, int ft, int h, int w, std::vector<float> &frames) {
        const size_t N = size_t(ft) * h * w, NT = N + VAE_REG + 1;
        if (!vae_blob_.is_open()) { if (vae_dir_.empty()) throw std::invalid_argument("vae_dir is required for video decoding"); vae_blob_.open(vae_dir_); }
        if (!vae_ || !vae_grid_.matches(ft, h, w)) {
            vae_grid_ = {};
            vae_.reset(); vae_ = std::make_unique<Stack>(comp_, StackDims{VAE_HID, VAE_HEADS, VAE_HEADS, VAE_D, VAE_FFN, 48, 1, vae_bits_, 1e-5f, true, false, false}, NT, 36, vae_blob_, "blocks.%d.", false, (const float *)ones_);
            for (void **q : {&vx_, &vcos_, &vsin_, &vin16_, &va_q_, &va_s_, &vout16_, &vcls_}) { if (*q) memory_.free(*q); *q = nullptr; }
            const size_t T = vae_->capacity();
            vx_ = memory_.alloc(T * VAE_HID * 4); rt().memset(vx_, 0, T * VAE_HID * 4);
            vcos_ = memory_.alloc(T * VAE_ROPE_HALF * 4); vsin_ = memory_.alloc(T * VAE_ROPE_HALF * 4);
            vin16_ = memory_.alloc(T * KPAD * 2); rt().memset(vin16_, 0, T * KPAD * 2);
            va_q_ = memory_.alloc(T * VAE_HID); va_s_ = memory_.alloc(T * 4);
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
            vproj_in_prep_.build(comp_, "plain", 8, KPAD); vproj_in_.build(comp_, "resid", 8, true, true, KPAD, VAE_HID, N, 1);
            vnorm_out_.build(comp_, "lnorm", 8, VAE_HID, 1e-5f, 1); vproj_out_.build(comp_, "plain", 8, true, true, VAE_HID, VAE_OUT, NT);
            if (!vnorm_table_) { vnorm_table_ = memory_.alloc(size_t(2) * VAE_HID * 4); rt().memset(vnorm_table_, 0, size_t(VAE_HID) * 4);
                rt().d2d(((float *)vnorm_table_ + VAE_HID), glue_.at("vae.norm_out.b", size_t(VAE_HID) * 4), size_t(VAE_HID) * 4); }
        }
        vae_grid_ = {ft, h, w};
        // post_quant_conv (a 24x24 matrix per voxel) on the host, then the tokens padded to K = 256 in f16
        const auto pq_w = glue_.host_f32("vae.post_quant_conv.w", 24 * 24), pq_b = glue_.host_f32("vae.post_quant_conv.b", 24);
        std::vector<uint16_t> in16(N * KPAD, 0);
        for (size_t v = 0; v < N; ++v) for (int o = 0; o < LATENT_CH; ++o) { float acc = pq_b[o]; for (int i = 0; i < LATENT_CH; ++i) acc += pq_w[o * 24 + i] * z[size_t(i) * N + v]; in16[v * KPAD + o] = f32_to_f16(acc); }
        rt().h2d(vin16_, in16.data(), in16.size() * 2);
        vproj_in_prep_.run(&prof, "vae proj_in", unsigned(N), vin16_, nullptr, nullptr, nullptr, va_q_, va_s_);
        rt().memset(vx_, 0, NT * VAE_HID * 4);
        vproj_in_.run(&prof, "vae proj_in", unsigned(N), va_q_, glue_.at("vae.proj_in.q", size_t(VAE_HID) * KPAD), glue_.at("vae.proj_in.s", size_t(VAE_HID) * 4), va_s_, vx_, ones_, vcls_, glue_.at("vae.proj_in.b", size_t(VAE_HID) * 4));
        rt().d2d(((float *)vx_ + N * VAE_HID), glue_.at("vae.register_tokens", size_t(VAE_REG) * VAE_HID * 4), size_t(VAE_REG) * VAE_HID * 4);   // then a zero cls row
        vae_->forward(&prof, vx_, vcls_, vcos_, vsin_, [&](int i) { const Stack::Block &b = vae_->block(i); return LayerCond{(const float *)zeros_, (const float *)b.scale1, (const float *)zeros_, (const float *)b.scale2}; });
        vnorm_out_.run(&prof, "vae norm_out", unsigned(NT), vx_, glue_.at("vae.norm_out.w", size_t(VAE_HID) * 4), vnorm_table_, vcls_, va_q_, va_s_);
        vproj_out_.run(&prof, "vae proj_out", unsigned(NT), va_q_, glue_.at("vae.proj_out.q", size_t(VAE_OUT) * VAE_HID), glue_.at("vae.proj_out.s", size_t(VAE_OUT) * 4), va_s_, vout16_, nullptr, nullptr, glue_.at("vae.proj_out.b", size_t(VAE_OUT) * 4));
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
        h3pipe_shape sh; h3pipe_shape_for(&p, &sh);
        const int T = sh.latent_t, H = sh.lat_h, W = sh.lat_w, F = sh.frames;
        const size_t FH = size_t(H) * VAE_PS, FW = size_t(W) * VAE_PS, plane = FH * FW;
        std::vector<float> lmean = glue_.host_f32("vae.latents_mean", 24), lstd = glue_.host_f32("vae.latents_std", 24);
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
    Blob glue_, blocks_, te_blob_; Span embed_; std::string te_dir_, vae_dir_, blocks_dir_; bool conditioning_ready_ = false, te_ready_ = false, refiner_ready_ = false; int vae_bits_ = 8;
    std::vector<float> curve_, inv_freq_, final_w_, final_b_; std::vector<std::vector<float>> adaln_w_, adaln_b_;
    std::unique_ptr<Stack> te_, refiner_, dit_, vae_; Blob vae_blob_;
    DecoderGrid vae_grid_;
    Prepare vproj_in_prep_, vnorm_out_; Gemm vproj_in_, vproj_out_;
    std::shared_ptr<Kernel> absdiff_; void *xb0_ = nullptr, *prev_b0_ = nullptr, *cache_resid_ = nullptr, *partials_ = nullptr; size_t cache_cap_ = 0;
    double cache_acc_ = 0.0; bool have_cache_ = false; int cache_skipped_ = 0;
    size_t audio_cap_ = 0; void *ah_ = nullptr, *aacc_ = nullptr, *ahj_ = nullptr, *ar_ = nullptr, *ar2_ = nullptr, *atmp_ = nullptr, *ain_ = nullptr;
    void *vx_ = nullptr, *vcos_ = nullptr, *vsin_ = nullptr, *vin16_ = nullptr, *va_q_ = nullptr, *va_s_ = nullptr, *vout16_ = nullptr, *vcls_ = nullptr, *vnorm_table_ = nullptr;
    std::map<std::string, Prepare> prepares_; std::map<std::string, Gemm> gemms_; Prepare final_prep_;
    Blob aenc_; bool aenc_open_ = false; std::string aenc_dir_;
    Blob vision_; bool vision_open_ = false; std::string vision_dir_; std::string te_span_sig_; void *ds_buf_ = nullptr;
    Blob venc_; bool venc_open_ = false; std::string venc_dir_; int attn_qk_bits_ = 4;
    std::shared_ptr<Kernel> norm_f32_;
    size_t seq_cap_ = 0;
    void *ones_ = nullptr, *zeros_ = nullptr, *mods_ = nullptr, *final_table_ = nullptr, *x_ = nullptr, *cls_ = nullptr, *cls0_ = nullptr, *tcls_ = nullptr, *cos_ = nullptr, *sin_ = nullptr,
         *in16_ = nullptr, *a_q_ = nullptr, *a_s_ = nullptr, *out16_ = nullptr, *text_copy_ = nullptr, *ref_cos_ = nullptr, *ref_sin_ = nullptr, *te_x_ = nullptr, *te_cos_ = nullptr, *te_sin_ = nullptr, *te_cls_ = nullptr;
};

void write_error(char *error, size_t cap, const char *m) noexcept { if (error && cap) std::snprintf(error, cap, "%s", m ? m : "unknown error"); }

}  // namespace

struct h3pipe_session { Pipe value; std::mutex mutex; explicit h3pipe_session(const h3pipe_config &c) : value(c) {} };

extern "C" uint32_t h3pipe_abi_version(void) { return H3PIPE_ABI_VERSION; }
extern "C" void h3pipe_destroy(h3pipe_session *s) { delete s; }

extern "C" int h3pipe_shape_for(const h3pipe_params *p, h3pipe_shape *out) {
    if (!p || !out) return H3PIPE_INVALID_ARGUMENT;
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
        if (out_elements != size_t(n) * HID) throw std::invalid_argument("out must hold n_ids * 5376 floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.get_text_in(ids, n, outp);
    })
}

extern "C" int h3pipe_denoise_refs(h3pipe_session *s, const int32_t *ids, int n, const h3pipe_params *params, const h3pipe_keyframe *keyframes, int n_keyframes, const h3pipe_ref *refs, int n_refs, const float *noise_video, const float *noise_audio,
                                   float *video, size_t video_elements, float *audio, size_t audio_elements, h3pipe_progress progress, void *user, char *error, size_t cap) {
    GUARD({
        if (!s || !ids || !params || !video || !audio || n < 1 || (n_refs > 0 && !refs) || (n_keyframes > 0 && !keyframes)) throw std::invalid_argument("session, ids, params, keyframes, refs and outputs are required");
        h3pipe_shape sh; h3pipe_shape_for(params, &sh);
        if (video_elements != size_t(LATENT_CH) * sh.latent_t * sh.lat_h * sh.lat_w) throw std::invalid_argument("video_latents must hold 24 * latent_t * lat_h * lat_w floats");
        if (audio_elements != size_t(2) * AUDIO_CH * sh.audio_t) throw std::invalid_argument("audio_latents must hold 2 * 32 * audio_t floats");
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
        h3pipe_shape sh; h3pipe_shape_for(params, &sh);
        if (video_elements != size_t(LATENT_CH) * sh.latent_t * sh.lat_h * sh.lat_w) throw std::invalid_argument("video_latents must hold 24 * latent_t * lat_h * lat_w floats");
        if (audio_elements != size_t(2) * AUDIO_CH * sh.audio_t) throw std::invalid_argument("audio_latents must hold 2 * 32 * audio_t floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.denoise(ids, n, *params, noise_video, noise_audio, video, audio, progress, user);
    })
}

extern "C" int h3pipe_decode_video(h3pipe_session *s, const h3pipe_params *params, const float *latents, size_t video_elements, uint8_t *frames, size_t frame_bytes, char *error, size_t cap) {
    GUARD({
        if (!s || !params || !latents || !frames) throw std::invalid_argument("session, params, latents and frames are required");
        h3pipe_shape sh; h3pipe_shape_for(params, &sh);
        if (video_elements != size_t(LATENT_CH) * sh.latent_t * sh.lat_h * sh.lat_w) throw std::invalid_argument("video_latents must hold 24 * latent_t * lat_h * lat_w floats");
        if (frame_bytes != size_t(sh.frames) * params->height * params->width * 3) throw std::invalid_argument("frames must hold frames * height * width * 3 bytes");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.decode_video(*params, latents, frames);
    })
}
extern "C" int h3pipe_encode_video(h3pipe_session *s, const float *pixels, int frames, int height, int width, float *latents, size_t latent_elements, int *latent_t, char *error, size_t cap) {
    GUARD({
        if (!s || !pixels || !latents || !latent_t || frames < 1) throw std::invalid_argument("session, pixels (frames >= 1), latents and latent_t are required");
        if (height % 32 || width % 32 || height < 32 || width < 32 || height > 2048 || width > 2048) throw std::invalid_argument("height and width must be multiples of 32 up to 2048");
        const int TL = frames == 1 ? 1 : 5 * ((frames + 16) / 17) - 3;
        if (latent_elements < size_t(24) * TL * (height / 16) * (width / 16)) throw std::invalid_argument("latents must hold 24 * latent_t * height/16 * width/16 floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.encode_video(pixels, frames, height, width, latents, *latent_t);
    })
}
extern "C" int h3pipe_vision_embed(h3pipe_session *s, const float *pixels, int height, int width, float *merged, size_t merged_elements, float *deepstack, size_t deepstack_elements, int *tokens, char *error, size_t cap) {
    GUARD({
        if (!s || !pixels || !merged || !deepstack || !tokens) throw std::invalid_argument("session, pixels, merged, deepstack and tokens are required");
        if (height % 32 || width % 32 || height < 32 || width < 32) throw std::invalid_argument("height and width must be multiples of 32");
        const size_t m = size_t(height / 32) * size_t(width / 32);
        if (merged_elements < m * 5120 || deepstack_elements < 3 * m * 5120) throw std::invalid_argument("merged needs tokens * 5120 floats, deepstack 3 * tokens * 5120");
        std::lock_guard<std::mutex> lock(s->mutex); std::vector<float> mg; std::vector<float> ds; s->value.vision_embed(pixels, height, width, mg, ds);
        memcpy(merged, mg.data(), mg.size() * 4); memcpy(deepstack, ds.data(), ds.size() * 4); *tokens = int(m);
    })
}
extern "C" int h3pipe_encode_audio(h3pipe_session *s, const float *samples, int n_samples, float *latents, size_t latent_elements, int *audio_t, char *error, size_t cap) {
    GUARD({
        if (!s || !samples || n_samples < 1 || !latents || !audio_t) throw std::invalid_argument("session, samples (n_samples >= 1), latents and audio_t are required");
        const int T = (n_samples + 799) / 800;
        if (latent_elements < size_t(2) * AUDIO_CH * T) throw std::invalid_argument("latents must hold at least 2 * 32 * ceil(n_samples / 800) floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.encode_audio(samples, n_samples, latents, *audio_t);
    })
}
extern "C" int h3pipe_decode_audio(h3pipe_session *s, const float *latents, size_t audio_elements, int audio_t, float *samples, size_t sample_elements, char *error, size_t cap) {
    GUARD({
        if (!s || !latents || !samples || audio_t < 1) throw std::invalid_argument("session, latents (audio_t >= 1) and samples are required");
        if (audio_elements != size_t(2) * AUDIO_CH * audio_t) throw std::invalid_argument("audio_latents must hold 2 * 32 * audio_t floats");
        if (sample_elements != size_t(2) * audio_t * 800) throw std::invalid_argument("samples must hold 2 * audio_t * 800 floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.decode_audio(latents, audio_t, samples);
    })
}
