// The MiniMax H3 pipeline as one C library: every kernel in Loom (compiled on first use through
// loom-compile into a cache), the host doing only what is under a megabyte per step. One generic
// transformer stack serves the text encoder, the token refiner, the 50 DiT blocks and the video
// decoder; the embedders and the final layer are the same int8 GEMMs with padded K / N.
//
// Build: ./scripts/build_host.sh  (host-only code against the HIP runtime API)
#include <hip/hip_runtime.h>

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

#include "h3pipe.h"

extern char **environ;

namespace {

// --- the model's shapes --------------------------------------------------------------------
constexpr int HID = 5376, HEADS = 56, HEAD_DIM = 128, FFN = 14336, ROPE_DIM = 96, ROPE_HALF = 48;
constexpr int TEXT_DIM = 5120, VIDEO_PATCH = 96, AUDIO_CH = 32, KPAD = 256, FINAL_N = 128;
constexpr int CLASSES = 12, MODALITIES = 3, MODS_ROWS = 6 * CLASSES;      // the AdaLN table rows per layer
constexpr int TE_HID = 5120, TE_HEADS = 64, TE_KV = 8, TE_FFN = 25600, TE_ROPE_HALF = 64;
constexpr int LATENT_CH = 24, FPS = 24, AUDIO_LATENTS_PER_S = 40;
constexpr int VAE_HID = 2048, VAE_HEADS = 32, VAE_D = 64, VAE_FFN = 8192, VAE_ROPE_HALF = 24, VAE_PT = 4, VAE_PS = 16, VAE_OUT = 3 * VAE_PT * VAE_PS * VAE_PS, VAE_REG = 4;
constexpr int VAE_CHUNK = 5, VAE_OVERLAP = 2, VAE_TOKEN_DROP = 3, VAE_TRATIO = 4, VAE_CLIP = 17;   // tokens_chunk_size, token_overlap, token_drop, temporal ratio, clip_length
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f}, IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};
constexpr double FRAME_RESCALE = 5.0 / 3.0, SPATIAL_SCALE = 32.0;
constexpr int FRAME_PER_TOKEN[5] = {1, 4, 4, 4, 4};
constexpr int THREADS = 256;

#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e_)); } while (0)

// --- small host helpers --------------------------------------------------------------------
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

struct Blob {
    std::string path; size_t size = 0; void *dev = nullptr; std::map<std::string, Span> spans;
    void open(const std::string &dir, const std::function<bool(const std::string &)> &skip = {}) {
        path = dir + "/weights.bin";
        std::ifstream in(dir + "/manifest.txt");
        if (!in) throw std::runtime_error("cannot read " + dir + "/manifest.txt");
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line); std::string name, dtype, shape; size_t offset, bytes;
            if (ls >> name >> offset >> bytes >> dtype >> shape) spans[name] = {offset, bytes, dtype, shape};
        }
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot read " + path);
        size = size_t(f.tellg());
        for (auto &e : spans) if (e.second.offset + e.second.bytes > size) throw std::runtime_error("manifest span '" + e.first + "' runs past " + path);
        HIP_CHECK(hipMalloc(&dev, size));
        std::vector<char> chunk(size_t(256) << 20);
        // upload every span except the skipped ones (uploading the whole file keeps it simple; skipped spans stay uninitialised)
        for (auto &e : spans) {
            if (skip && skip(e.first)) continue;
            f.seekg(std::streamoff(e.second.offset));
            for (size_t done = 0; done < e.second.bytes;) {
                const size_t n = std::min(chunk.size(), e.second.bytes - done);
                f.read(chunk.data(), std::streamsize(n));
                if (!f) throw std::runtime_error("short read of " + path);
                HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)((char *)dev + e.second.offset + done), chunk.data(), n));
                done += n;
            }
        }
    }
    const Span &span(const std::string &name) const {
        auto it = spans.find(name);
        if (it == spans.end()) throw std::runtime_error("missing tensor " + name + " in " + path);
        return it->second;
    }
    char *at(const std::string &name, size_t bytes) const {
        const Span &s = span(name);
        if (s.bytes != bytes) throw std::runtime_error("tensor " + name + " has " + std::to_string(s.bytes) + " bytes, expected " + std::to_string(bytes));
        return (char *)dev + s.offset;
    }
    std::vector<float> host_f32(const std::string &name, size_t count) const {
        const Span &s = span(name);
        if (s.bytes != count * 4) throw std::runtime_error("tensor " + name + " has " + std::to_string(s.bytes) + " bytes, expected " + std::to_string(count * 4));
        std::vector<float> v(count); std::ifstream f(path, std::ios::binary); f.seekg(std::streamoff(s.offset)); f.read((char *)v.data(), std::streamsize(s.bytes));
        if (!f) throw std::runtime_error("short read of " + name); return v;
    }
    ~Blob() { if (dev) (void)hipFree(dev); }
};

// --- kernels: compile through loom-compile into the cache, load, launch ---------------------
struct Kernel {
    hipModule_t module = nullptr; hipFunction_t function = nullptr;
    void load(const std::string &path, const std::string &symbol) {
        HIP_CHECK(hipModuleLoad(&module, path.c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module, symbol.c_str()));
    }
    ~Kernel() { if (module) (void)hipModuleUnload(module); }
};
using Cfg = std::vector<std::pair<std::string, std::string>>;

struct Compiler {
    std::string exe, sources, cache;
    std::map<std::string, std::shared_ptr<Kernel>> loaded;
    // one kernel per (stem, config); the cache file name carries the config
    std::shared_ptr<Kernel> get(const std::string &stem, const std::string &symbol, const Cfg &cfg) {
        std::string tag = stem;
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

struct KernArgs {
    alignas(16) unsigned char bytes[192]; size_t size = 0;
    KernArgs &i32(int v) { size = (size + 3) & ~size_t(3); memcpy(bytes + size, &v, 4); size += 4; return *this; }
    KernArgs &ptr(const void *p) { size = (size + 7) & ~size_t(7); memcpy(bytes + size, &p, 8); size += 8; return *this; }
};

struct Profile { bool on = false; std::map<std::string, double> us; };

void launch(Kernel &k, Profile *prof, const char *stage, unsigned gx, unsigned gy, unsigned bx, KernArgs &args) {
    std::chrono::steady_clock::time_point t0;
    if (prof && prof->on) { HIP_CHECK(hipDeviceSynchronize()); t0 = std::chrono::steady_clock::now(); }
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size, HIP_LAUNCH_PARAM_END};
    HIP_CHECK(hipModuleLaunchKernel(k.function, gx, gy, 1, bx, 1, 1, 0, nullptr, nullptr, config));
    if (prof && prof->on) { HIP_CHECK(hipDeviceSynchronize()); prof->us[stage] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); }
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
            const std::string ns = "h3." + stem + ".";
            attention_ = c.get(stem, "h3_" + stem, {{ns + "q_stride", std::to_string(d.inner())}, {ns + "kv_stride", std::to_string(d.kv_inner())}, {ns + "tokens", std::to_string(tokens)}, {ns + "token_capacity", std::to_string(capacity_)}, {ns + "scale", num(1.0 / std::sqrt(double(d.head_dim)))}, {ns + "out_stride", std::to_string(d.inner())}});
        }
        const size_t T = capacity_;
        HIP_CHECK(hipMalloc(&a_q_, T * size_t(std::max(d.ffn, std::max(d.hidden, d.inner()))) / per));
        HIP_CHECK(hipMalloc(&a_s_, T * 4));
        HIP_CHECK(hipMalloc(&fused_, T * size_t(d.qkv()) * 2));
        HIP_CHECK(hipMalloc(&q_, T * size_t(d.inner()) * 2));
        HIP_CHECK(hipMalloc(&k_, T * size_t(d.kv_inner()) * 2));
        HIP_CHECK(hipMalloc(&v_, T * size_t(d.kv_inner()) * 2));
        HIP_CHECK(hipMalloc(&attn_, T * size_t(d.inner()) * 2));
        HIP_CHECK(hipMalloc(&gu_, T * size_t(d.ffn) * 2));
        for (auto p : {fused_, q_, k_, v_, attn_}) HIP_CHECK(hipMemset(p, 0, T * size_t(p == fused_ ? d.qkv() : (p == q_ || p == attn_ ? d.inner() : d.kv_inner())) * 2));
        if (d.attn_i4) {
            HIP_CHECK(hipMalloc(&qi_, T * size_t(d.heads) * 64)); HIP_CHECK(hipMalloc(&ki_, T * size_t(d.heads) * 64)); HIP_CHECK(hipMalloc(&qs_, T * size_t(d.heads) * 4)); HIP_CHECK(hipMalloc(&ks_, T * size_t(d.heads) * 4));
            HIP_CHECK(hipMalloc(&kmean_, size_t(d.inner()) * 4)); HIP_CHECK(hipMalloc(&zmean_, size_t(d.inner()) * 4)); HIP_CHECK(hipMemset(zmean_, 0, size_t(d.inner()) * 4));
            HIP_CHECK(hipMalloc(&vt_, size_t(d.inner()) * T * 2)); HIP_CHECK(hipMemset(vt_, 0, size_t(d.inner()) * T * 2));
            for (auto p : {qi_, ki_}) HIP_CHECK(hipMemset(p, 0, T * size_t(d.heads) * 64));
            for (auto p : {qs_, ks_}) HIP_CHECK(hipMemset(p, 0, T * size_t(d.heads) * 4));
        }
    }
    ~Stack() { for (void *p : {a_q_, a_s_, fused_, q_, k_, v_, attn_, gu_, qi_, ki_, qs_, ks_, kmean_, zmean_, vt_}) if (p) (void)hipFree(p); }
    size_t capacity() const { return capacity_; }
    size_t tokens() const { return tokens_; }
    const std::vector<struct Block_> *dummy = nullptr;

    struct Block { char *qkv_q, *qkv_s, *out_q, *out_s, *gu_q, *gu_s, *down_q, *down_s, *qkv_b = nullptr, *out_b = nullptr, *gu_b = nullptr, *down_b = nullptr, *norm1, *norm2, *qnorm, *knorm, *scale1 = nullptr, *scale2 = nullptr; };
    const Block &block(int i) const { return blocks_[i]; }

    // x: f32 [capacity][hidden] (rows past tokens untouched); cls: i32 [tokens]; cos/sin: f32 [tokens][rope_dim/2]
    void forward(Profile *prof, void *x, const void *cls, const void *cos, const void *sin, const std::function<LayerCond(int)> &cond) {
        const unsigned T = unsigned(tokens_);
        for (int i = 0; i < layers_; ++i) {
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
                KernArgs a; a.i32(int(T)).i32(d_.heads).ptr(qi_).ptr(qs_).ptr(ki_).ptr(ks_).ptr(vt_).ptr(attn_);
                const unsigned qb = 16 * unsigned(waves_); launch(*attention_, prof, "attention", (T + qb - 1) / qb, unsigned(d_.heads), 32 * unsigned(waves_), a);
            } else {
              KernArgs a; a.i32(int(T)).i32(d_.causal ? d_.kv_heads : d_.heads).ptr(q_).ptr(k_).ptr(v_).ptr(attn_);
              if (d_.causal) launch(*attention_, prof, "attention", (T + 15) / 16, unsigned(d_.kv_heads), THREADS, a);
              else { const unsigned qb = 16 * unsigned(waves_); launch(*attention_, prof, "attention", (T + qb - 1) / qb, unsigned(d_.heads), 32 * unsigned(waves_), a); } }
            prep_attn_.run(prof, "prepare out input", T, attn_, nullptr, nullptr, nullptr, a_q_, a_s_);
            gemm_out_.run(prof, "gemm out + residual", T, a_q_, b.out_q, b.out_s, a_s_, x, lc.gate_msa, cls, b.out_b);
            prep_norm_.run(prof, "prepare norm", T, x, b.norm2, lc.table_mlp, cls, a_q_, a_s_);
            gemm_gu_.run(prof, "gemm ff + swiglu", T, a_q_, b.gu_q, b.gu_s, a_s_, gu_, nullptr, nullptr, b.gu_b);
            prep_down_.run(prof, "prepare down input", T, gu_, nullptr, nullptr, nullptr, a_q_, a_s_);
            gemm_down_.run(prof, "gemm down + residual", T, a_q_, b.down_q, b.down_s, a_s_, x, lc.gate_mlp, cls, b.down_b);
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
struct Layout {
    int text_len, latent_t, lat_h, lat_w, audio_t; size_t audio_rows, video_rows, seq_len;
    std::vector<double> pos;                 // [seq][3] (t, h, w)
    std::vector<int32_t> adaln_rows, tclass; // per row: class*3 + modality; 0 video timestep, 1 audio
    static std::vector<double> axis(int dim, double sqrt_area) {
        const double ratio = dim / sqrt_area; const int n = dim / 2; std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = (i * (ratio / n) + (1.0 - ratio) / 2.0) * SPATIAL_SCALE;
        return v;
    }
    Layout(int text_len_, int latent_t_, int lat_h_, int lat_w_, int audio_t_) : text_len(text_len_), latent_t(latent_t_), lat_h(lat_h_), lat_w(lat_w_), audio_t(audio_t_) {
        const double area = std::sqrt(double(lat_h) * lat_w);
        const std::vector<double> ah = axis(lat_h, area), aw = axis(lat_w, area);
        audio_rows = size_t(audio_t) * 2; video_rows = size_t(latent_t) * ah.size() * aw.size(); seq_len = text_len + audio_rows + video_rows;
        pos.assign(seq_len * 3, 0.0); adaln_rows.assign(seq_len, 0); tclass.assign(seq_len, 0);
        size_t r = 0;
        for (int i = 0; i < text_len; ++i, ++r) { pos[3 * r] = i; adaln_rows[r] = 1; tclass[r] = 0; }
        const double cursor = text_len;
        for (int c = 0; c < 2; ++c) for (int t = 0; t < audio_t; ++t, ++r) { pos[3 * r] = cursor + t; pos[3 * r + 2] = c == 0 ? aw.front() : aw.back(); adaln_rows[r] = 1 * MODALITIES + 2; tclass[r] = 1; }
        std::vector<double> tg(latent_t); double acc = cursor;
        for (int k = 0; k < latent_t; ++k) { tg[k] = acc; acc += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5]; }
        for (int k = 0; k < latent_t; ++k) for (double hv : ah) for (double wv : aw) { pos[3 * r] = tg[k]; pos[3 * r + 1] = hv; pos[3 * r + 2] = wv; adaln_rows[r] = 0; tclass[r] = 0; ++r; }
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
public:
    explicit Pipe(const h3pipe_config &cfg) {
        if (!cfg.glue_dir || !cfg.blocks_dir || !cfg.te_dir || !cfg.kernel_sources || !cfg.cache_dir || !cfg.loom_compile) throw std::invalid_argument("every directory and the loom-compile path are required");
        HIP_CHECK(hipInit(0));
        comp_.exe = cfg.loom_compile; comp_.sources = cfg.kernel_sources; comp_.cache = cfg.cache_dir;
        vae_dir_ = cfg.vae_dir ? cfg.vae_dir : ""; vae_bits_ = cfg.vae_bits ? cfg.vae_bits : 8;
        glue_.open(cfg.glue_dir, [](const std::string &n) { return n == "te.embed"; });
        embed_ = glue_.span("te.embed");
        blocks_.open(cfg.blocks_dir);
        te_dir_ = cfg.te_dir;
        // host-side conditioning tables
        curve_ = glue_.host_f32("h3.adaln_t_table", 1025 * 8);
        inv_freq_ = glue_.host_f32("h3.rope_inv_freq", 16);
        for (int i = 0; i < 50; ++i) { adaln_w_.push_back(glue_.host_f32(fmt("h3.blocks.%d.adaln.w", i), size_t(3 * 6 * HID) * 8)); adaln_b_.push_back(glue_.host_f32(fmt("h3.blocks.%d.adaln.b", i), size_t(3 * 6 * HID))); }
        final_w_ = glue_.host_f32("h3.final.adaln.w", size_t(2 * HID) * 8); final_b_ = glue_.host_f32("h3.final.adaln.b", size_t(2 * HID));
        HIP_CHECK(hipMalloc(&ones_, size_t(HID) * 4)); { std::vector<float> o(HID, 1.0f); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)ones_, o.data(), o.size() * 4)); }
        HIP_CHECK(hipMalloc(&zeros_, size_t(2 * TE_FFN) * 4)); HIP_CHECK(hipMemset(zeros_, 0, size_t(2 * TE_FFN) * 4));
        HIP_CHECK(hipMalloc(&mods_, size_t(50) * MODS_ROWS * HID * 4));
        HIP_CHECK(hipMalloc(&final_table_, size_t(4) * HID * 4));
    }
    ~Pipe() { for (void *p : {ones_, zeros_, mods_, final_table_, x_, cls_, cls0_, tcls_, cos_, sin_, in16_, a_q_, a_s_, out16_, text_copy_, ref_cos_, ref_sin_, te_x_, te_cos_, te_sin_, te_cls_, vx_, vcos_, vsin_, vin16_, va_q_, va_s_, vout16_, vcls_, vnorm_table_, ah_, aacc_, ahj_, ar_, ar2_, atmp_, ain_}) if (p) (void)hipFree(p); }

    Profile prof;

    // --- the prompt: embedding lookup on the host, the encoder, condition_proj, the refiner ---
    void ensure_seq(size_t seq) {
        if (seq <= seq_cap_) return;
        for (void *p : {x_, cls_, cls0_, tcls_, cos_, sin_, in16_, a_q_, a_s_, out16_}) if (p) (void)hipFree(p);
        seq_cap_ = (seq + 255) / 256 * 256 + 32;
        const size_t T = seq_cap_;
        HIP_CHECK(hipMalloc(&x_, T * HID * 4)); HIP_CHECK(hipMemset(x_, 0, T * HID * 4));
        HIP_CHECK(hipMalloc(&cls_, T * 4)); HIP_CHECK(hipMemset(cls_, 0, T * 4));
        HIP_CHECK(hipMalloc(&cls0_, T * 4)); HIP_CHECK(hipMemset(cls0_, 0, T * 4));       // the single-class GEMMs (embedders, condition proj) index their gate table with this
        HIP_CHECK(hipMalloc(&tcls_, T * 4)); HIP_CHECK(hipMemset(tcls_, 0, T * 4));
        HIP_CHECK(hipMalloc(&cos_, T * ROPE_HALF * 4)); HIP_CHECK(hipMalloc(&sin_, T * ROPE_HALF * 4));
        HIP_CHECK(hipMalloc(&in16_, T * TEXT_DIM * 2)); HIP_CHECK(hipMemset(in16_, 0, T * TEXT_DIM * 2));
        HIP_CHECK(hipMalloc(&a_q_, T * TEXT_DIM)); HIP_CHECK(hipMalloc(&a_s_, T * 4));
        HIP_CHECK(hipMalloc(&out16_, T * FINAL_N * 2));
    }

    // x rows [row0, row0 + rows) = W · in + b through the int8 family: in f16 [rows][k] on in16_, rows zeroed first
    void embed_linear(const char *stage, size_t row0, size_t rows, int k, const std::string &wname) {
        Prepare &prep = prepares_[fmt("plain_%d_%zu", k, rows)]; if (!prep.k) prep.build(comp_, "plain", 8, k);
        Gemm &g = gemms_[fmt("resid_%d_%zu", k, rows)]; if (!g.k) g.build(comp_, "resid", 8, true, true, k, HID, rows, 1);
        prep.run(&prof, stage, unsigned(rows), in16_, nullptr, nullptr, nullptr, a_q_, a_s_);
        float *xr = (float *)x_ + row0 * HID;
        HIP_CHECK(hipMemset(xr, 0, rows * HID * 4));
        g.run(&prof, stage, unsigned(rows), a_q_, glue_.at(wname + ".q", size_t(HID) * k), glue_.at(wname + ".s", size_t(HID) * 4), a_s_, xr, ones_, cls0_, glue_.at(wname + ".b", size_t(HID) * 4));
    }

    void text_in(const int32_t *ids, int n) {
        // embedding rows from the file (bf16) -> f32
        std::vector<float> emb(size_t(n) * TEXT_DIM);
        { std::ifstream f(glue_.path, std::ios::binary); std::vector<uint16_t> row(TEXT_DIM);
          for (int i = 0; i < n; ++i) { if (ids[i] < 0 || size_t(ids[i]) * TEXT_DIM * 2 >= embed_.bytes) throw std::invalid_argument("token id out of range: " + std::to_string(ids[i]));
              f.seekg(std::streamoff(embed_.offset + size_t(ids[i]) * TEXT_DIM * 2)); f.read((char *)row.data(), TEXT_DIM * 2); if (!f) throw std::runtime_error("short read of the embedding table");
              for (int j = 0; j < TEXT_DIM; ++j) emb[size_t(i) * TEXT_DIM + j] = bf16_to_f32(row[j]); } }
        // the encoder's 50 layers
        if (!te_ || te_->tokens() != size_t(n)) {
            if (!te_blob_.dev) te_blob_.open(te_dir_);
            te_.reset(); te_ = std::make_unique<Stack>(comp_, StackDims{TE_HID, TE_HEADS, TE_KV, HEAD_DIM, TE_FFN, 128, 1, 8, 1e-6f, false, true, true}, size_t(n), 50, te_blob_, "blocks.%d.", true, nullptr);
            for (void *p : {te_x_, te_cos_, te_sin_, te_cls_}) if (p) (void)hipFree(p);
            const size_t T = te_->capacity();
            HIP_CHECK(hipMalloc(&te_x_, T * TE_HID * 4)); HIP_CHECK(hipMemset(te_x_, 0, T * TE_HID * 4));
            HIP_CHECK(hipMalloc(&te_cos_, T * TE_ROPE_HALF * 4)); HIP_CHECK(hipMalloc(&te_sin_, T * TE_ROPE_HALF * 4));
            HIP_CHECK(hipMalloc(&te_cls_, T * 4)); HIP_CHECK(hipMemset(te_cls_, 0, T * 4));
            std::vector<float> c(size_t(n) * TE_ROPE_HALF), s(size_t(n) * TE_ROPE_HALF);
            for (int t = 0; t < n; ++t) for (int j = 0; j < TE_ROPE_HALF; ++j) { const double inv = std::pow(5000000.0, -double(2 * j) / HEAD_DIM), ang = t * inv; c[size_t(t) * TE_ROPE_HALF + j] = float(std::cos(ang)); s[size_t(t) * TE_ROPE_HALF + j] = float(std::sin(ang)); }
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)te_cos_, c.data(), c.size() * 4)); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)te_sin_, s.data(), s.size() * 4));
        }
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)te_x_, emb.data(), emb.size() * 4));
        te_->forward(&prof, te_x_, te_cls_, te_cos_, te_sin_, [&](int) { return LayerCond{(const float *)zeros_, (const float *)ones_, (const float *)zeros_, (const float *)ones_}; });
        // hidden -> f16 -> condition_proj into x rows [0, n)
        std::vector<float> hid(size_t(n) * TE_HID); HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipMemcpyDtoH(hid.data(), (hipDeviceptr_t)te_x_, hid.size() * 4));
        std::vector<uint16_t> h16(hid.size()); for (size_t i = 0; i < hid.size(); ++i) h16[i] = f32_to_f16(hid[i]);
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)in16_, h16.data(), h16.size() * 2));
        embed_linear("condition proj", 0, size_t(n), TEXT_DIM, "h3.cond");
        // the token refiner: two H3-shaped blocks without rope (identity tables), then its final norm
        if (!refiner_ || refiner_->tokens() != size_t(n)) {
            refiner_.reset(); refiner_ = std::make_unique<Stack>(comp_, StackDims{HID, HEADS, HEADS, HEAD_DIM, FFN, ROPE_DIM, 1, 8, 1e-5f, false, true, false}, size_t(n), 2, glue_, "h3.refiner.%d.", true, nullptr);
            std::vector<float> c(size_t(n) * ROPE_HALF, 1.0f), s(size_t(n) * ROPE_HALF, 0.0f);
            for (void *q : {ref_cos_, ref_sin_}) if (q) (void)hipFree(q);
            HIP_CHECK(hipMalloc(&ref_cos_, c.size() * 4)); HIP_CHECK(hipMalloc(&ref_sin_, s.size() * 4));
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)ref_cos_, c.data(), c.size() * 4)); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)ref_sin_, s.data(), s.size() * 4));
            const std::string ns = "h3.norm_mod_f32.";
            norm_f32_ = comp_.get("norm_mod_f32", "h3_norm_mod_f32", {{ns + "width", std::to_string(HID)}, {ns + "lanes", std::to_string(lanes_for(HID))}, {ns + "eps", num(1e-5)}, {ns + "classes", "1"}});
        }
        refiner_->forward(&prof, x_, cls0_, ref_cos_, ref_sin_, [&](int) { return LayerCond{(const float *)zeros_, (const float *)ones_, (const float *)zeros_, (const float *)ones_}; });
        { KernArgs a; a.i32(n).ptr(x_).ptr(glue_.at("h3.refiner.final_norm", size_t(HID) * 4)).ptr(zeros_).ptr(cls0_); launch(*norm_f32_, &prof, "refiner final norm", unsigned(n), 1, unsigned(lanes_for(HID)), a); }
    }

    void get_text_in(const int32_t *ids, int n, float *out) {
        ensure_seq(size_t(n));
        text_in(ids, n);
        HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipMemcpyDtoH(out, (hipDeviceptr_t)x_, size_t(n) * HID * 4));
    }

    // --- conditioning per step ---
    void temb(float t, float *out8) const {
        const double pos = std::min(std::max(double(t), 0.0), 1.0) * 1024.0; const int i0 = std::min(int(std::floor(pos)), 1023); const double f = pos - i0;
        for (int j = 0; j < 8; ++j) out8[j] = float(curve_[size_t(i0) * 8 + j] * (1.0 - f) + curve_[size_t(i0 + 1) * 8 + j] * f);
    }
    // the per-layer tables in the runtime's layout: rows [0,2C) (scale_msa, shift_msa) per class, [2C,3C) gate_msa, [3C,5C) (scale_mlp, shift_mlp), [5C,6C) gate_mlp;
    // class = timestep index * 3 + modality; the projection's chunks per (modality d, j): shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    void upload_mods(const float tv[8], const float ta[8]) {
        std::vector<float> table(size_t(50) * MODS_ROWS * HID, 0.0f), proj(size_t(3 * 6 * HID));
        for (int i = 0; i < 50; ++i) {
            const float *W = adaln_w_[i].data(), *B = adaln_b_[i].data();
            for (int m = 0; m < 2; ++m) {
                const float *te = m == 0 ? tv : ta;
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
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)mods_, table.data(), table.size() * 4));
        // the final layer's (scale, shift) per timestep class: rows 2c = scale, 2c + 1 = shift; the projection gives (shift | scale)
        std::vector<float> ft(size_t(4) * HID);
        for (int m = 0; m < 2; ++m) {
            const float *te = m == 0 ? tv : ta;
            for (int r = 0; r < 2 * HID; ++r) { float acc = final_b_[r]; for (int j = 0; j < 8; ++j) acc += final_w_[size_t(r) * 8 + j] * te[j]; (r < HID ? ft[size_t(2 * m + 1) * HID + r] : ft[size_t(2 * m) * HID + r - HID]) = acc; }
        }
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)final_table_, ft.data(), ft.size() * 4));
    }
    LayerCond cond(int i) const {
        const float *mod = (const float *)mods_ + size_t(i) * MODS_ROWS * HID;
        return LayerCond{mod, mod + size_t(CLASSES) * 2 * HID, mod + size_t(CLASSES) * 3 * HID, mod + size_t(CLASSES) * 5 * HID};
    }

    // --- denoising ---
    void denoise(const int32_t *ids, int n, const h3pipe_params &p, const float *noise_video, const float *noise_audio, float *video_out, float *audio_out, h3pipe_progress progress, void *user) {
        if (p.height % 32 || p.width % 32 || p.height < 64 || p.width < 64) throw std::invalid_argument("height and width must be multiples of 32");
        if (p.steps < 2 || p.steps > 1000) throw std::invalid_argument("steps must be 2..1000");
        h3pipe_shape sh; h3pipe_shape_for(&p, &sh);
        Layout lay(n, sh.latent_t, sh.lat_h, sh.lat_w, sh.audio_t);
        const size_t L = size_t(n), Na = lay.audio_rows, Nv = lay.video_rows, S = lay.seq_len;
        ensure_seq(S);
        text_in(ids, n);
        // the packed layout's tables
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)cls_, lay.adaln_rows.data(), S * 4)); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)tcls_, lay.tclass.data(), S * 4));
        { std::vector<float> c(S * ROPE_HALF), s(S * ROPE_HALF);
          for (size_t r = 0; r < S; ++r) for (int ax = 0; ax < 3; ++ax) for (int j = 0; j < 16; ++j) { const float ang = float(lay.pos[3 * r + ax]) * inv_freq_[j]; c[r * ROPE_HALF + ax * 16 + j] = std::cos(ang); s[r * ROPE_HALF + ax * 16 + j] = std::sin(ang); }
          HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)cos_, c.data(), c.size() * 4)); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)sin_, s.data(), s.size() * 4)); }
        if (!dit_ || dit_->tokens() != S) {
            const char *qk = std::getenv("H3_ATTN_QK"); StackDims dd{HID, HEADS, HEADS, HEAD_DIM, FFN, ROPE_DIM, CLASSES, 4, 1e-5f, false, true, false}; dd.attn_i4 = !(qk && std::string(qk) == "f16");   // int4 QK^T by default
            dit_.reset(); dit_ = std::make_unique<Stack>(comp_, dd, S, 50, blocks_, "blocks.%d.", true, nullptr); }
        if (!final_prep_.k) { final_prep_.build(comp_, "norm", 8, HID, 1e-5f, 2); }
        Gemm &final_gemm = gemms_[fmt("final_%zu", S)]; if (!final_gemm.k) final_gemm.build(comp_, "plain", 8, true, true, HID, FINAL_N, S);
        // latents as rows: video [Nv][96] (row = (t*(H/2) + hh)*(W/2) + ww, column = c*4 + dy*2 + dx), audio [2*audio_t][32] (row = c*audio_t + t)
        std::vector<float> vrows(Nv * VIDEO_PATCH), arows(Na * AUDIO_CH);
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
        Schedule sv(p.steps, p.video_shift > 0 ? p.video_shift : 12.0), sa(p.steps, p.audio_shift > 0 ? p.audio_shift : 3.0);
        if (sv.timesteps.size() != sa.timesteps.size()) throw std::runtime_error("the two schedules differ in length");
        std::vector<uint16_t> in16(std::max(Na, Nv) * KPAD); std::vector<uint16_t> out16(S * FINAL_N);
        const auto t_start = std::chrono::steady_clock::now();
        for (size_t step = 0; step < sv.timesteps.size(); ++step) {
            float tv[8], ta[8]; temb(sv.timesteps[step], tv); temb(sa.timesteps[step], ta); upload_mods(tv, ta);
            // audio rows -> x[L, L+Na), video rows -> x[L+Na, S), each through prepare(256) + the padded-K int8 GEMM
            std::fill(in16.begin(), in16.end(), 0);
            for (size_t r = 0; r < Na; ++r) for (int k = 0; k < AUDIO_CH; ++k) in16[r * KPAD + k] = f32_to_f16(arows[r * AUDIO_CH + k]);
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)in16_, in16.data(), Na * KPAD * 2));
            embed_linear("audio in", L, Na, KPAD, "h3.audio_in");
            std::fill(in16.begin(), in16.end(), 0);
            for (size_t r = 0; r < Nv; ++r) for (int k = 0; k < VIDEO_PATCH; ++k) in16[r * KPAD + k] = f32_to_f16(vrows[r * VIDEO_PATCH + k]);
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)in16_, in16.data(), Nv * KPAD * 2));
            embed_linear("video in", L + Na, Nv, KPAD, "h3.video_in");
            // the text rows are refreshed from a copy each step (the blocks update x in place)
            if (step == 0) { if (!text_copy_) HIP_CHECK(hipMalloc(&text_copy_, seq_cap_ * HID * 4)); HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)text_copy_, (hipDeviceptr_t)x_, L * HID * 4)); }
            else HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)x_, (hipDeviceptr_t)text_copy_, L * HID * 4));
            dit_->forward(&prof, x_, cls_, cos_, sin_, [&](int i) { return cond(i); });
            final_prep_.run(&prof, "final norm", unsigned(S), x_, glue_.at("h3.final.norm", size_t(HID) * 4), final_table_, tcls_, a_q_, a_s_);
            final_gemm.run(&prof, "final out", unsigned(S), a_q_, glue_.at("h3.final.out.q", size_t(FINAL_N) * HID), glue_.at("h3.final.out.s", size_t(FINAL_N) * 4), a_s_, out16_, nullptr, nullptr, glue_.at("h3.final.out.b", size_t(FINAL_N) * 4));
            HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipMemcpyDtoH(out16.data(), (hipDeviceptr_t)out16_, S * FINAL_N * 2));
            // Euler step per schedule: x0 = x + sigma * v, x' = r x + (1 - r) x0
            const float sg_v = sv.sigmas[step], r_v = sv.sigmas[step + 1] / sg_v, sg_a = sa.sigmas[step], r_a = sa.sigmas[step + 1] / sg_a;
            for (size_t r = 0; r < Nv; ++r) for (int k = 0; k < VIDEO_PATCH; ++k) { float &x = vrows[r * VIDEO_PATCH + k]; const float v = f16_to_f32(out16[(L + Na + r) * FINAL_N + k]); x = r_v * x + (1.0f - r_v) * (x + sg_v * v); }
            for (size_t r = 0; r < Na; ++r) for (int k = 0; k < AUDIO_CH; ++k) { float &x = arows[r * AUDIO_CH + k]; const float v = f16_to_f32(out16[(L + r) * FINAL_N + VIDEO_PATCH + k]); x = r_a * x + (1.0f - r_a) * (x + sg_a * v); }
            if (progress && progress(user, int(step + 1), int(sv.timesteps.size()), std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count())) throw Cancelled();
        }
        rows_to_tensor(video_out);
        for (int c = 0; c < 2; ++c) for (int t = 0; t < A; ++t) for (int k = 0; k < AUDIO_CH; ++k) audio_out[(size_t(c) * AUDIO_CH + k) * A + t] = arows[(size_t(c) * A + t) * AUDIO_CH + k];
    }
    struct Cancelled {};

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
            for (void *q : {ah_, aacc_, ahj_, ar_, ar2_, atmp_, ain_}) if (q) (void)hipFree(q);
            HIP_CHECK(hipMalloc(&ah_, cap * 4)); HIP_CHECK(hipMalloc(&aacc_, cap * 4)); HIP_CHECK(hipMalloc(&ahj_, cap * 4)); HIP_CHECK(hipMalloc(&ar_, cap * 4)); HIP_CHECK(hipMalloc(&ar2_, cap * 4));
            HIP_CHECK(hipMalloc(&atmp_, cap * 8)); HIP_CHECK(hipMalloc(&ain_, size_t(32) * T * 4 + 4096)); audio_cap_ = cap;
        }
        std::vector<float> lmean = glue_.host_f32("audio.latents_mean", 32), lstd = glue_.host_f32("audio.latents_std", 32);
        std::vector<float> in(size_t(32) * T), out(L_out);
        for (int ch = 0; ch < 2; ++ch) {
            for (int c = 0; c < 32; ++c) for (int t = 0; t < T; ++t) in[size_t(c) * T + t] = latents[(size_t(ch) * 32 + c) * T + t] * lstd[c] + lmean[c];
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)ain_, in.data(), in.size() * 4));
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
                HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)ah_, (hipDeviceptr_t)ar_, plane * 4)); HIP_CHECK(hipMemset(aacc_, 0, plane * 4));
                for (int j = 0; j < 3; ++j) {                                   // the three AMP blocks, averaged
                    const int r = i * 3 + j, kk = RESK[j];
                    HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)ahj_, (hipDeviceptr_t)ah_, plane * 4));
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
            HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipMemcpyDtoH(out.data(), (hipDeviceptr_t)ar2_, L_out * 4));
            for (size_t i = 0; i < L_out; ++i) samples[size_t(ch) * L_out + i] = std::min(std::max(out[i], -1.0f), 1.0f);
        }
    }

    // --- the video decoder: diffusers' chunking and heads around the 36 blocks in Loom ---
    // one clip: model-space latents z [24][ft][h][w] (already * std + mean) -> ImageNet-space frames [3][ft*4][h*16][w*16]
    void decode_clip(const float *z, int ft, int h, int w, std::vector<float> &frames) {
        const size_t N = size_t(ft) * h * w, NT = N + VAE_REG + 1;
        if (!vae_blob_.dev) { if (vae_dir_.empty()) throw std::invalid_argument("vae_dir is required for video decoding"); vae_blob_.open(vae_dir_); }
        if (!vae_ || vae_->tokens() != NT) {
            vae_.reset(); vae_ = std::make_unique<Stack>(comp_, StackDims{VAE_HID, VAE_HEADS, VAE_HEADS, VAE_D, VAE_FFN, 48, 1, vae_bits_, 1e-5f, true, false, false}, NT, 36, vae_blob_, "blocks.%d.", false, (const float *)ones_);
            for (void *q : {vx_, vcos_, vsin_, vin16_, va_q_, va_s_, vout16_, vcls_}) if (q) (void)hipFree(q);
            const size_t T = vae_->capacity();
            HIP_CHECK(hipMalloc(&vx_, T * VAE_HID * 4)); HIP_CHECK(hipMemset(vx_, 0, T * VAE_HID * 4));
            HIP_CHECK(hipMalloc(&vcos_, T * VAE_ROPE_HALF * 4)); HIP_CHECK(hipMalloc(&vsin_, T * VAE_ROPE_HALF * 4));
            HIP_CHECK(hipMalloc(&vin16_, T * KPAD * 2)); HIP_CHECK(hipMemset(vin16_, 0, T * KPAD * 2));
            HIP_CHECK(hipMalloc(&va_q_, T * VAE_HID)); HIP_CHECK(hipMalloc(&va_s_, T * 4));
            HIP_CHECK(hipMalloc(&vout16_, T * VAE_OUT * 2)); HIP_CHECK(hipMalloc(&vcls_, T * 4)); HIP_CHECK(hipMemset(vcls_, 0, T * 4));
            // the rotary tables: coordinates 2 * ((i + 0.5) / size) - 1 per axis, angles 2 pi * pos * inv_freq (8 frequencies per axis), zero for the register / cls rows
            std::vector<float> c(T * VAE_ROPE_HALF, 1.0f), s(T * VAE_ROPE_HALF, 0.0f);
            const int sizes[3] = {ft, h, w};
            for (int t = 0; t < ft; ++t) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                const size_t r = (size_t(t) * h + y) * w + x; const int idx[3] = {t, y, x};
                for (int ax = 0; ax < 3; ++ax) for (int j = 0; j < 8; ++j) {
                    const double pos = 2.0 * ((idx[ax] + 0.5) / sizes[ax]) - 1.0, inv = std::pow(100.0, -double(j) * 6.0 / 48.0), ang = 2.0 * M_PI * pos * inv;
                    c[r * VAE_ROPE_HALF + ax * 8 + j] = float(std::cos(ang)); s[r * VAE_ROPE_HALF + ax * 8 + j] = float(std::sin(ang)); } }
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)vcos_, c.data(), c.size() * 4)); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)vsin_, s.data(), s.size() * 4));
            vproj_in_prep_.build(comp_, "plain", 8, KPAD); vproj_in_.build(comp_, "resid", 8, true, true, KPAD, VAE_HID, N, 1);
            vnorm_out_.build(comp_, "lnorm", 8, VAE_HID, 1e-5f, 1); vproj_out_.build(comp_, "plain", 8, true, true, VAE_HID, VAE_OUT, NT);
            if (!vnorm_table_) { HIP_CHECK(hipMalloc(&vnorm_table_, size_t(2) * VAE_HID * 4)); HIP_CHECK(hipMemset(vnorm_table_, 0, size_t(VAE_HID) * 4));
                HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)((float *)vnorm_table_ + VAE_HID), (hipDeviceptr_t)glue_.at("vae.norm_out.b", size_t(VAE_HID) * 4), size_t(VAE_HID) * 4)); }
        }
        // post_quant_conv (a 24x24 matrix per voxel) on the host, then the tokens padded to K = 256 in f16
        static std::vector<float> pq_w, pq_b; if (pq_w.empty()) { pq_w = glue_.host_f32("vae.post_quant_conv.w", 24 * 24); pq_b = glue_.host_f32("vae.post_quant_conv.b", 24); }
        std::vector<uint16_t> in16(N * KPAD, 0);
        for (size_t v = 0; v < N; ++v) for (int o = 0; o < LATENT_CH; ++o) { float acc = pq_b[o]; for (int i = 0; i < LATENT_CH; ++i) acc += pq_w[o * 24 + i] * z[size_t(i) * N + v]; in16[v * KPAD + o] = f32_to_f16(acc); }
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)vin16_, in16.data(), in16.size() * 2));
        vproj_in_prep_.run(&prof, "vae proj_in", unsigned(N), vin16_, nullptr, nullptr, nullptr, va_q_, va_s_);
        HIP_CHECK(hipMemset(vx_, 0, NT * VAE_HID * 4));
        vproj_in_.run(&prof, "vae proj_in", unsigned(N), va_q_, glue_.at("vae.proj_in.q", size_t(VAE_HID) * KPAD), glue_.at("vae.proj_in.s", size_t(VAE_HID) * 4), va_s_, vx_, ones_, vcls_, glue_.at("vae.proj_in.b", size_t(VAE_HID) * 4));
        HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)((float *)vx_ + N * VAE_HID), (hipDeviceptr_t)glue_.at("vae.register_tokens", size_t(VAE_REG) * VAE_HID * 4), size_t(VAE_REG) * VAE_HID * 4));   // then a zero cls row
        vae_->forward(&prof, vx_, vcls_, vcos_, vsin_, [&](int i) { const Stack::Block &b = vae_->block(i); return LayerCond{(const float *)zeros_, (const float *)b.scale1, (const float *)zeros_, (const float *)b.scale2}; });
        vnorm_out_.run(&prof, "vae norm_out", unsigned(NT), vx_, glue_.at("vae.norm_out.w", size_t(VAE_HID) * 4), vnorm_table_, vcls_, va_q_, va_s_);
        vproj_out_.run(&prof, "vae proj_out", unsigned(NT), va_q_, glue_.at("vae.proj_out.q", size_t(VAE_OUT) * VAE_HID), glue_.at("vae.proj_out.s", size_t(VAE_OUT) * 4), va_s_, vout16_, nullptr, nullptr, glue_.at("vae.proj_out.b", size_t(VAE_OUT) * 4));
        std::vector<uint16_t> out16(N * VAE_OUT); HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipMemcpyDtoH(out16.data(), (hipDeviceptr_t)vout16_, out16.size() * 2));
        // unpatchify: token (t, y, x) holds [3][4][16][16] -> frames [3][ft*4][h*16][w*16]
        const size_t FH = size_t(h) * VAE_PS, FW = size_t(w) * VAE_PS, FT = size_t(ft) * VAE_PT;
        frames.assign(size_t(3) * FT * FH * FW, 0.0f);
        for (int t = 0; t < ft; ++t) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const uint16_t *tok = out16.data() + ((size_t(t) * h + y) * w + x) * VAE_OUT;
            for (int c = 0; c < 3; ++c) for (int pt = 0; pt < VAE_PT; ++pt) for (int py = 0; py < VAE_PS; ++py) for (int px = 0; px < VAE_PS; ++px)
                frames[((size_t(c) * FT + size_t(t) * VAE_PT + pt) * FH + size_t(y) * VAE_PS + py) * FW + size_t(x) * VAE_PS + px] = f16_to_f32(tok[((size_t(c) * VAE_PT + pt) * VAE_PS + py) * VAE_PS + px]);
        }
    }

    // the clip loop of diffusers' _decode: 5-token chunks with a 2-token overlap, 17 kept frames per chunk (3 dropped in front), 5-frame cross-fades
    void decode_video(const h3pipe_params &p, const float *latents, uint8_t *out) {
        h3pipe_shape sh; h3pipe_shape_for(&p, &sh);
        const int T = sh.latent_t, H = sh.lat_h, W = sh.lat_w, F = sh.frames;
        const size_t FH = size_t(H) * VAE_PS, FW = size_t(W) * VAE_PS, plane = FH * FW;
        std::vector<float> lmean = glue_.host_f32("vae.latents_mean", 24), lstd = glue_.host_f32("vae.latents_std", 24);
        const int num_tokens = T + VAE_TOKEN_DROP, pad_tokens = ((-num_tokens) % VAE_CHUNK + VAE_CHUNK) % VAE_CHUNK, num_chunks = (num_tokens + pad_tokens) / VAE_CHUNK - 1;
        const int Tp = T + pad_tokens;
        std::vector<float> zp(size_t(LATENT_CH) * Tp * H * W);
        for (int c = 0; c < LATENT_CH; ++c) for (int t = 0; t < Tp; ++t) for (size_t i = 0; i < size_t(H) * W; ++i)
            zp[(size_t(c) * Tp + t) * H * W + i] = latents[(size_t(c) * T + std::min(t, T - 1)) * H * W + i] * lstd[c] + lmean[c];
        const int chunk_frames = VAE_CHUNK * VAE_TRATIO, pre = ((-VAE_CLIP) % VAE_TRATIO + VAE_TRATIO) % VAE_TRATIO, overlap_frames = std::max(VAE_OVERLAP * VAE_TRATIO - pre, 0);
        std::vector<float> dec; std::vector<float> overlap; bool have_overlap = false;   // dec: [3][frames][H][W] accumulated
        size_t dec_frames = 0;
        auto append = [&](const std::vector<float> &chunk, size_t nf, size_t src_ft) {
            const size_t old = dec_frames; dec.resize(size_t(3) * (old + nf) * plane);
            std::vector<float> tmp(size_t(3) * (old + nf) * plane);
            for (int c = 0; c < 3; ++c) { memcpy(tmp.data() + size_t(c) * (old + nf) * plane, dec.data() + size_t(c) * old * plane, old * plane * 4); memcpy(tmp.data() + (size_t(c) * (old + nf) + old) * plane, chunk.data() + size_t(c) * src_ft * plane, nf * plane * 4); }
            dec.swap(tmp); dec_frames = old + nf;
        };
        std::vector<float> clip, chunk;
        for (int i = 0; i < num_chunks; ++i) {
            const int start = i * VAE_CHUNK, ft = std::min(VAE_CHUNK + VAE_OVERLAP, Tp - start);
            std::vector<float> z(size_t(LATENT_CH) * ft * H * W);
            for (int c = 0; c < LATENT_CH; ++c) memcpy(z.data() + size_t(c) * ft * H * W, zp.data() + (size_t(c) * Tp + start) * H * W, size_t(ft) * H * W * 4);
            decode_clip(z.data(), ft, H, W, clip);
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
        for (size_t f = 0; f < keep; ++f) for (size_t q = 0; q < plane; ++q) for (int c = 0; c < 3; ++c) {
            const float v = dec[(size_t(c) * dec_frames + f) * plane + q] * IMAGENET_STD[c] + IMAGENET_MEAN[c];
            out[(f * plane + q) * 3 + c] = uint8_t(std::lround(std::min(std::max(v, 0.0f), 1.0f) * 255.0f));
        }
    }

private:
    Compiler comp_;
    Blob glue_, blocks_, te_blob_; Span embed_; std::string te_dir_, vae_dir_; int vae_bits_ = 8;
    std::vector<float> curve_, inv_freq_, final_w_, final_b_; std::vector<std::vector<float>> adaln_w_, adaln_b_;
    std::unique_ptr<Stack> te_, refiner_, dit_, vae_; Blob vae_blob_;
    Prepare vproj_in_prep_, vnorm_out_; Gemm vproj_in_, vproj_out_;
    size_t audio_cap_ = 0; void *ah_ = nullptr, *aacc_ = nullptr, *ahj_ = nullptr, *ar_ = nullptr, *ar2_ = nullptr, *atmp_ = nullptr, *ain_ = nullptr;
    void *vx_ = nullptr, *vcos_ = nullptr, *vsin_ = nullptr, *vin16_ = nullptr, *va_q_ = nullptr, *va_s_ = nullptr, *vout16_ = nullptr, *vcls_ = nullptr, *vnorm_table_ = nullptr;
    std::map<std::string, Prepare> prepares_; std::map<std::string, Gemm> gemms_; Prepare final_prep_;
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
extern "C" int h3pipe_decode_audio(h3pipe_session *s, const float *latents, size_t audio_elements, int audio_t, float *samples, size_t sample_elements, char *error, size_t cap) {
    GUARD({
        if (!s || !latents || !samples || audio_t < 1) throw std::invalid_argument("session, latents (audio_t >= 1) and samples are required");
        if (audio_elements != size_t(2) * AUDIO_CH * audio_t) throw std::invalid_argument("audio_latents must hold 2 * 32 * audio_t floats");
        if (sample_elements != size_t(2) * audio_t * 800) throw std::invalid_argument("samples must hold 2 * audio_t * 800 floats");
        std::lock_guard<std::mutex> lock(s->mutex); s->value.decode_audio(latents, audio_t, samples);
    })
}
