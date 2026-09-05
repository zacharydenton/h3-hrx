// The text encoder's 50 Qwen3 layers as a resident Loom session: hidden 5120, 64 query heads
// and 8 key/value heads of 128 (fused qkv 10240), SwiGLU 25600 (gate first), no biases, RMSNorm
// eps 1e-6 with plain weights (the AdaLN table is zero, one class; the residual gates are ones),
// causal attention with eight query heads per key/value head. Ten launches per layer, W8A8 ConvRot.
//
// Build: ./scripts/build_host.sh
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3te.h"

namespace {

constexpr int HIDDEN = 5120, HEADS = 64, KV_HEADS = 8, HEAD_DIM = 128, FFN = 25600;
constexpr int INNER = HEADS * HEAD_DIM, KV_INNER = KV_HEADS * HEAD_DIM;   // 8192, 1024
constexpr int QKV = INNER + 2 * KV_INNER;                                  // 10240
constexpr int ROPE_HALF = HEAD_DIM / 2;
constexpr int THREADS = 256, NORM_LANES = 160, ATTN_LANES = 256, DOWN_LANES = 320;   // prepare workgroups: width / (8 * lanes) chunks

#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e_)); } while (0)

struct Span { size_t offset, bytes; };

std::map<std::string, Span> read_manifest(const std::string &path) {
    std::map<std::string, Span> spans;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string name, dtype, shape; size_t offset, bytes;
        if (ls >> name >> offset >> bytes >> dtype >> shape) spans[name] = {offset, bytes};
    }
    return spans;
}

// Streams weights.bin into device memory in slices (the file is 24 GB; no host copy of the whole).
void *upload_file(const std::string &path, size_t *size_out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path);
    const size_t size = size_t(in.tellg());
    in.seekg(0);
    void *device = nullptr;
    HIP_CHECK(hipMalloc(&device, size));
    std::vector<char> chunk(size_t(256) << 20);
    for (size_t done = 0; done < size;) {
        const size_t n = std::min(chunk.size(), size - done);
        in.read(chunk.data(), std::streamsize(n));
        if (!in) { (void)hipFree(device); throw std::runtime_error("short read of " + path); }
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)((char *)device + done), chunk.data(), n));
        done += n;
    }
    *size_out = size;
    return device;
}

struct Kernel {
    hipModule_t module = nullptr; hipFunction_t function = nullptr;
    void load(const std::string &path, const char *symbol) {
        HIP_CHECK(hipModuleLoad(&module, path.c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module, symbol));
    }
};

struct KernArgs {
    alignas(16) unsigned char bytes[160]; size_t size = 0;
    void scalar_i32(int v) { size = (size + 3) & ~size_t(3); memcpy(bytes + size, &v, 4); size += 4; }
    void pointer(const void *p) { size = (size + 7) & ~size_t(7); memcpy(bytes + size, &p, 8); size += 8; }
};

class Session {
public:
    Session(const std::string &weights_dir, const std::string &kernels_dir, int tokens, int layers)
        : tokens_(tokens), layers_(layers) {
        if (tokens < 1 || tokens > 65536) throw std::invalid_argument("tokens must be 1..65536");
        if (layers < 1 || layers > 50) throw std::invalid_argument("layers must be 1..50");
        HIP_CHECK(hipInit(0));
        capacity_ = std::max<size_t>((tokens + 16 + 31) / 32 * 32, (tokens + 127) / 128 * 128);
        auto spans = read_manifest(weights_dir + "/manifest.txt");
        size_t blob_size = 0;
        weights_ = upload_file(weights_dir + "/weights.bin", &blob_size);
        for (const auto &e : spans)
            if (e.second.offset + e.second.bytes > blob_size)
                throw std::runtime_error("manifest span '" + e.first + "' runs past weights.bin");
        auto need = [&](const std::string &name, size_t bytes) {
            auto it = spans.find(name);
            if (it == spans.end()) throw std::runtime_error("missing tensor " + name);
            if (it->second.bytes != bytes)
                throw std::runtime_error("tensor " + name + " has " + std::to_string(it->second.bytes) + " bytes, expected " + std::to_string(bytes));
            return (char *)weights_ + it->second.offset;
        };
        for (int i = 0; i < layers; ++i) {
            std::string p = "blocks." + std::to_string(i);
            Block b;
            b.qkv_q = need(p + ".qkv.q", size_t(QKV) * HIDDEN);   b.qkv_s = need(p + ".qkv.s", size_t(QKV) * 4);
            b.out_q = need(p + ".out.q", size_t(HIDDEN) * INNER);  b.out_s = need(p + ".out.s", size_t(HIDDEN) * 4);
            b.gu_q = need(p + ".gu.q", size_t(2 * FFN) * HIDDEN);  b.gu_s = need(p + ".gu.s", size_t(2 * FFN) * 4);
            b.down_q = need(p + ".down.q", size_t(HIDDEN) * FFN);  b.down_s = need(p + ".down.s", size_t(HIDDEN) * 4);
            b.norm1 = need(p + ".norm1", HIDDEN * 4); b.norm2 = need(p + ".norm2", HIDDEN * 4);
            b.qnorm = need(p + ".qnorm", HEAD_DIM * 4); b.knorm = need(p + ".knorm", HEAD_DIM * 4);
            blocks_.push_back(b);
        }
        auto load = [&](Kernel &k, const char *stem, const char *symbol) { k.load(kernels_dir + "/" + stem + ".hsaco", symbol); };
        load(k_prep_norm_, "prepare_norm", "h3_prepare_norm_i8");
        load(k_prep_attn_, "prepare_attn", "h3_prepare_plain_i8");
        load(k_prep_down_, "prepare_down", "h3_prepare_plain16_i8");   // f16 LDS: the 25600-wide row
        load(k_gemm_qkv_, "gemm_qkv", "h3_gemm_i8_256");
        load(k_gemm_gu_, "gemm_gu", "h3_gemm_i8_swiglu_256");
        load(k_gemm_out_, "gemm_out", "h3_gemm_i8_resid_256");
        load(k_gemm_down_, "gemm_down", "h3_gemm_i8_resid_256");
        load(k_rope_, "rope_qknorm", "h3_rope128_qknorm_f16");
        load(k_attention_, "attention", "h3_attention_gqa8c_lds_f16_wmma");
        const size_t T = capacity_;
        HIP_CHECK(hipMalloc(&x_, T * HIDDEN * 4));              // the residual stream, f32
        HIP_CHECK(hipMalloc(&a_q_, T * FFN));                   // the widest prepared operand, int8
        HIP_CHECK(hipMalloc(&a_s_, T * 4));
        HIP_CHECK(hipMalloc(&fused_, T * QKV * 2));
        HIP_CHECK(hipMalloc(&q_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&k_, T * KV_INNER * 2));
        HIP_CHECK(hipMalloc(&v_, T * KV_INNER * 2));
        HIP_CHECK(hipMalloc(&attn_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&gu_, T * FFN * 2));                // silu(gate) * up, fused into the GEMM epilogue
        HIP_CHECK(hipMalloc(&mods_, size_t(2) * HIDDEN * 4));   // the zero (scale, shift) table of the one class
        HIP_CHECK(hipMemset(mods_, 0, size_t(2) * HIDDEN * 4));
        HIP_CHECK(hipMalloc(&cls_, T * 4));
        HIP_CHECK(hipMalloc(&ones_, HIDDEN * 4));               // the residual gate table: plain adds
        { std::vector<float> ones(HIDDEN, 1.0f); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)ones_, ones.data(), HIDDEN * 4)); }
        HIP_CHECK(hipMalloc(&cos_, T * ROPE_HALF * 4));
        HIP_CHECK(hipMalloc(&sin_, T * ROPE_HALF * 4));
        HIP_CHECK(hipMemset(x_, 0, T * HIDDEN * 4)); HIP_CHECK(hipMemset(fused_, 0, T * QKV * 2));   // headroom rows stay zero
        HIP_CHECK(hipMemset(q_, 0, T * INNER * 2)); HIP_CHECK(hipMemset(k_, 0, T * KV_INNER * 2)); HIP_CHECK(hipMemset(v_, 0, T * KV_INNER * 2));
        HIP_CHECK(hipMemset(cls_, 0, T * 4));
    }
    ~Session() {
        for (void *p : {x_, a_q_, a_s_, fused_, q_, k_, v_, attn_, gu_, mods_, cls_, cos_, sin_, ones_, weights_}) if (p) (void)hipFree(p);
        for (Kernel *k : {&k_prep_norm_, &k_prep_attn_, &k_prep_down_, &k_gemm_qkv_, &k_gemm_gu_, &k_gemm_out_, &k_gemm_down_, &k_rope_, &k_attention_})
            if (k->module) (void)hipModuleUnload(k->module);
    }

    void run(float *x, size_t x_elements, const float *cos, const float *sin, size_t rope_elements) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t T = tokens_;
        if (x_elements != T * HIDDEN) throw std::invalid_argument("x has " + std::to_string(x_elements) + " elements, expected " + std::to_string(T * HIDDEN));
        if (rope_elements != T * ROPE_HALF) throw std::invalid_argument("cos/sin have the wrong element count");
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)x_, x, T * HIDDEN * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)cos_, cos, rope_elements * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)sin_, sin, rope_elements * 4));
        for (int i = 0; i < layers_; ++i) block(i);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpyDtoH(x, (hipDeviceptr_t)x_, T * HIDDEN * 4));
        if (profile) {
            double total = 0; for (auto &e : stage_us) total += e.second;
            std::vector<std::pair<double, std::string>> rows;
            for (auto &e : stage_us) rows.push_back({e.second, e.first});
            std::sort(rows.rbegin(), rows.rend());
            fprintf(stderr, "stage profile over %d layer(s), %zu tokens:\n", layers_, T);
            for (auto &r : rows) fprintf(stderr, "  %-24s %9.3f ms  %5.1f%%\n", r.second.c_str(), r.first / 1000.0, 100.0 * r.first / total);
            fprintf(stderr, "  %-24s %9.3f ms\n", "total", total / 1000.0);
            stage_us.clear();
        }
    }

    bool profile = false;
    std::map<std::string, double> stage_us;

private:
    struct Block { char *qkv_q, *qkv_s, *out_q, *out_s, *gu_q, *gu_s, *down_q, *down_s, *norm1, *norm2, *qnorm, *knorm; };

    void launch(Kernel &k, const char *stage, unsigned gx, unsigned gy, unsigned bx, KernArgs &args) {
        std::chrono::steady_clock::time_point t0;
        if (profile) { HIP_CHECK(hipDeviceSynchronize()); t0 = std::chrono::steady_clock::now(); }
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size, HIP_LAUNCH_PARAM_END};
        HIP_CHECK(hipModuleLaunchKernel(k.function, gx, gy, 1, bx, 1, 1, 0, nullptr, nullptr, config));
        if (profile) {
            HIP_CHECK(hipDeviceSynchronize());
            stage_us[stage] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    // m-tiles per raster group, the same rule as scripts/build_kernels_te.py.
    unsigned gemm_m_group(size_t m) const {
        const size_t tiles = (m + GEMM_TILE - 1) / GEMM_TILE;
        unsigned best = 4; size_t best_pad = (tiles + 3) / 4 * 4;
        for (unsigned g : {3u, 2u}) { const size_t pad = (tiles + g - 1) / g * g; if (pad < best_pad) { best = g; best_pad = pad; } }
        return best;
    }
    unsigned gemm_grid_y(size_t m) const { const unsigned g = gemm_m_group(m); return unsigned(((m + GEMM_TILE - 1) / GEMM_TILE + g - 1) / g * g); }

    void gemm(Kernel &k, const char *stage, char *w_q, char *w_s, int n, void *out, const float *gate) {
        KernArgs a;
        a.scalar_i32(int(tokens_));
        a.pointer(a_q_); a.pointer(w_q); a.pointer(w_s); a.pointer(a_s_); a.pointer(out);
        if (gate) { a.pointer(gate); a.pointer(cls_); }
        launch(k, stage, unsigned(n / 128), gemm_grid_y(tokens_), THREADS, a);
    }

    // H3TE_DEBUG=1: a few values of each stage's output for the first layer
    void dump(const char *what, const void *p, size_t bytes_per, size_t count, bool f16) {
        if (!debug_) return;
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned char> h(bytes_per * count);
        HIP_CHECK(hipMemcpyDtoH(h.data(), (hipDeviceptr_t)p, h.size()));
        fprintf(stderr, "  %-20s", what);
        for (size_t i = 0; i < count; ++i) {
            if (bytes_per == 4 && !f16) { float v; memcpy(&v, h.data() + 4 * i, 4); fprintf(stderr, " %10.4g", v); }
            else if (f16) { uint16_t u; memcpy(&u, h.data() + 2 * i, 2); const int e = (u >> 10) & 31, m = u & 1023; const float v = (e == 0 ? std::ldexp(float(m), -24) : std::ldexp(1.0f + m / 1024.0f, e - 15)) * ((u & 0x8000) ? -1 : 1); fprintf(stderr, " %10.4g", v); }
            else fprintf(stderr, " %4d", int((signed char)h[i]));
        }
        fprintf(stderr, "\n");
    }

    void block(int i) {
        const Block &b = blocks_[i];
        const unsigned T = unsigned(tokens_);
        debug_ = i == 0 && std::getenv("H3TE_DEBUG");
        dump("x in", x_, 4, 6, false);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.norm1); a.pointer(mods_); a.pointer(cls_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, NORM_LANES, a); }
        dump("norm1 a_q", a_q_, 1, 8, false); dump("norm1 a_s", a_s_, 4, 4, false);
        gemm(k_gemm_qkv_, "gemm qkv", b.qkv_q, b.qkv_s, QKV, fused_, nullptr);
        dump("qkv", fused_, 2, 6, true);
        { KernArgs a; a.scalar_i32(T); a.pointer(fused_); a.pointer(b.qnorm); a.pointer(b.knorm); a.pointer(cos_); a.pointer(sin_); a.pointer(q_); a.pointer(k_); a.pointer(v_);
          launch(k_rope_, "qk norm + rope", T, 1, THREADS, a); }
        dump("q", q_, 2, 6, true); dump("k", k_, 2, 6, true); dump("v", v_, 2, 6, true);
        { KernArgs a; a.scalar_i32(T); a.scalar_i32(KV_HEADS); a.pointer(q_); a.pointer(k_); a.pointer(v_); a.pointer(attn_);
          launch(k_attention_, "attention", (T + 15) / 16, KV_HEADS, THREADS, a); }
        dump("attn", attn_, 2, 6, true);
        { KernArgs a; a.scalar_i32(T); a.pointer(attn_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_attn_, "prepare out input", T, 1, ATTN_LANES, a); }
        dump("attn a_q", a_q_, 1, 8, false); dump("attn a_s", a_s_, 4, 4, false);
        gemm(k_gemm_out_, "gemm out + residual", b.out_q, b.out_s, HIDDEN, x_, (const float *)ones_);
        dump("x after out", x_, 4, 6, false);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.norm2); a.pointer(mods_); a.pointer(cls_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, NORM_LANES, a); }
        gemm(k_gemm_gu_, "gemm ff + swiglu", b.gu_q, b.gu_s, 2 * FFN, gu_, nullptr);
        dump("gu", gu_, 2, 6, true);
        { KernArgs a; a.scalar_i32(T); a.pointer(gu_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_down_, "prepare down input", T, 1, DOWN_LANES, a); }
        dump("down a_q", a_q_, 1, 8, false); dump("down a_s", a_s_, 4, 4, false);
        gemm(k_gemm_down_, "gemm down + residual", b.down_q, b.down_s, HIDDEN, x_, (const float *)ones_);
        dump("x after down", x_, 4, 6, false);
    }

    static constexpr size_t GEMM_TILE = 256;
    bool debug_ = false;
    int tokens_, layers_;
    size_t capacity_ = 0;
    std::mutex mutex_;
    std::vector<Block> blocks_;
    void *weights_ = nullptr, *x_ = nullptr, *a_q_ = nullptr, *a_s_ = nullptr, *fused_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr,
         *attn_ = nullptr, *gu_ = nullptr, *mods_ = nullptr, *cls_ = nullptr, *cos_ = nullptr, *sin_ = nullptr, *ones_ = nullptr;
    Kernel k_prep_norm_, k_prep_attn_, k_prep_down_, k_gemm_qkv_, k_gemm_gu_, k_gemm_out_, k_gemm_down_, k_rope_, k_attention_;
};

void write_error(char *error, size_t capacity, const char *message) noexcept {
    if (error && capacity) std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

}  // namespace

struct h3te_session { Session value; h3te_session(const char *w, const char *k, int t, int l) : value(w, k, t, l) {} };

extern "C" uint32_t h3te_abi_version(void) { return H3TE_ABI_VERSION; }
extern "C" void h3te_destroy(h3te_session *s) { delete s; }
extern "C" int h3te_profile(h3te_session *s, int enable) { if (!s) return H3TE_INVALID_ARGUMENT; s->value.profile = enable != 0; return H3TE_OK; }

extern "C" int h3te_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers, h3te_session **out, char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    if (!out) { write_error(error, cap, "out_session must not be null"); return H3TE_INVALID_ARGUMENT; }
    *out = nullptr;
    try {
        if (!weights_dir || !kernels_dir) throw std::invalid_argument("weights_dir and kernels_dir are required");
        *out = new h3te_session(weights_dir, kernels_dir, tokens, layers);
        return H3TE_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return H3TE_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return H3TE_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return H3TE_ERROR; }
}

extern "C" int h3te_run(h3te_session *s, float *x, size_t x_elements, const float *cos, const float *sin, size_t rope_elements,
                        char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    try {
        if (!s || !x || !cos || !sin) throw std::invalid_argument("null argument");
        s->value.run(x, x_elements, cos, sin, rope_elements);
        return H3TE_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return H3TE_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return H3TE_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return H3TE_ERROR; }
}
