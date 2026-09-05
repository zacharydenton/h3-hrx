// The video VAE's 36 ViT decoder blocks as a resident Loom session, the H3 block session
// with the decoder's shapes: hidden 2048, 32 heads of 64, SwiGLU 8192, biases on every
// linear, RMSNorm with a plain weight (the AdaLN table is zero, one class) and layer scales
// as the residual gate tables. Ten launches per block, W4A4 ConvRot throughout.
//
// Build: ./scripts/build_host.sh
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
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

#include "h3vae.h"

namespace {

constexpr int HIDDEN = 2048, HEADS = 32, HEAD_DIM = 64, INNER = HEADS * HEAD_DIM, FFN = 8192;
constexpr int QKV = 3 * INNER;                                  // 6144
constexpr int ROPE_HALF = 24;
constexpr int THREADS = 256, NORM_LANES = 256, ATTN_LANES = 256, DOWN_LANES = 256;   // prepare workgroups: width / (8 * lanes) chunks

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

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<char> buffer(in.tellg());
    in.seekg(0);
    in.read(buffer.data(), buffer.size());
    return buffer;
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
        if (tokens < 16 || tokens > 65536) throw std::invalid_argument("tokens must be 16..65536");
        if (layers < 1 || layers > 36) throw std::invalid_argument("layers must be 1..36");
        HIP_CHECK(hipInit(0));
        capacity_ = std::max<size_t>((tokens + 16 + 31) / 32 * 32, (tokens + 127) / 128 * 128);   // tokens+16 headroom, whole query blocks of up to 128 rows
        auto spans = read_manifest(weights_dir + "/manifest.txt");
        auto blob = read_file(weights_dir + "/weights.bin");
        for (const auto &e : spans)
            if (e.second.offset + e.second.bytes > blob.size())
                throw std::runtime_error("manifest span '" + e.first + "' runs past weights.bin");
        HIP_CHECK(hipMalloc(&weights_, blob.size()));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights_, blob.data(), blob.size()));
        { std::ifstream bf(kernels_dir + "/bits.txt"); if (bf) bf >> bits_; if (bits_ != 4 && bits_ != 8) throw std::runtime_error("bits.txt must say 4 or 8"); }
        const size_t per = bits_ == 4 ? 2 : 1;                       // K elements per weight byte
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
            b.qkv_q = need(p + ".qkv.q", size_t(QKV) * HIDDEN / per);   b.qkv_s = need(p + ".qkv.s", size_t(QKV) * 4);   b.qkv_b = need(p + ".qkv.b", size_t(QKV) * 4);
            b.out_q = need(p + ".out.q", size_t(HIDDEN) * INNER / per);  b.out_s = need(p + ".out.s", size_t(HIDDEN) * 4);  b.out_b = need(p + ".out.b", size_t(HIDDEN) * 4);
            b.gu_q = need(p + ".gu.q", size_t(2 * FFN) * HIDDEN / per);  b.gu_s = need(p + ".gu.s", size_t(2 * FFN) * 4);   b.gu_b = need(p + ".gu.b", size_t(2 * FFN) * 4);
            b.down_q = need(p + ".down.q", size_t(HIDDEN) * FFN / per);  b.down_s = need(p + ".down.s", size_t(HIDDEN) * 4); b.down_b = need(p + ".down.b", size_t(HIDDEN) * 4);
            b.norm1 = need(p + ".norm1", HIDDEN * 4); b.norm2 = need(p + ".norm2", HIDDEN * 4);
            b.scale1 = need(p + ".scale1", HIDDEN * 4); b.scale2 = need(p + ".scale2", HIDDEN * 4);
            blocks_.push_back(b);
        }
        auto load = [&](Kernel &k, const char *stem, const char *symbol) { k.load(kernels_dir + "/" + stem + ".hsaco", symbol); };
        const std::string ib = bits_ == 4 ? "i4" : "i8";
        load(k_prep_norm_, "prepare_norm_i4", ("h3_prepare_norm_" + ib).c_str());
        load(k_prep_attn_, "prepare_attn_i4", ("h3_prepare_plain_" + ib).c_str());
        load(k_prep_down_, "prepare_down_i4", ("h3_prepare_plain_" + ib).c_str());
        load(k_gemm_qkv_, "gemm_qkv", ("h3_gemm_" + ib + "_256b").c_str());
        load(k_gemm_gu_, "gemm_gu", ("h3_gemm_" + ib + "_swiglu_256b_gs").c_str());
        load(k_gemm_out_, "gemm_out", ("h3_gemm_" + ib + "_resid_256b").c_str());
        load(k_gemm_down_, "gemm_down", ("h3_gemm_" + ib + "_resid_256b").c_str());
        load(k_rope_, "rope_qknorm", "h3_rope64_qknorm_f16");
        { std::ifstream wf(kernels_dir + "/attention_waves.txt"); if (wf) wf >> attn_waves_; if (attn_waves_ != 4 && attn_waves_ != 8) throw std::runtime_error("attention_waves.txt must say 4 or 8"); }
        load(k_attention_, "attention", attn_waves_ == 8 ? "h3_attention_mha648_lds_f16_wmma" : "h3_attention_mha64_lds_f16_wmma");
        gemm_tile_ = 256;
        const size_t T = capacity_;
        HIP_CHECK(hipMalloc(&x_, T * HIDDEN * 4));            // the residual stream, f32
        HIP_CHECK(hipMalloc(&a_q_, T * FFN));                   // the widest prepared operand, sized for int8
        HIP_CHECK(hipMalloc(&a_s_, T * 4));
        HIP_CHECK(hipMalloc(&fused_, T * QKV * 2));
        HIP_CHECK(hipMalloc(&q_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&k_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&v_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&attn_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&gu_, T * FFN * 2));                // silu(gate) * up, fused into the GEMM epilogue
        HIP_CHECK(hipMalloc(&mods_, size_t(2) * HIDDEN * 4));           // the zero (scale, shift) table of the one class
        HIP_CHECK(hipMemset(mods_, 0, size_t(2) * HIDDEN * 4));
        HIP_CHECK(hipMalloc(&cls_, T * 4));
        HIP_CHECK(hipMalloc(&ones_, HEAD_DIM * 4));
        { std::vector<float> ones(HEAD_DIM, 1.0f); HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)ones_, ones.data(), HEAD_DIM * 4)); }
        HIP_CHECK(hipMalloc(&cos_, T * ROPE_HALF * 4));
        HIP_CHECK(hipMalloc(&sin_, T * ROPE_HALF * 4));
        for (void *p : {x_, fused_, q_, k_, v_, cls_}) HIP_CHECK(hipMemset(p, 0, p == cls_ ? T * 4 : (p == x_ ? T * HIDDEN * 4 : (p == fused_ ? T * QKV * 2 : T * INNER * 2))));   // headroom rows stay zero
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
            fprintf(stderr, "stage profile over %d block(s), %zu tokens:\n", layers_, T);
            for (auto &r : rows) fprintf(stderr, "  %-24s %9.3f ms  %5.1f%%\n", r.second.c_str(), r.first / 1000.0, 100.0 * r.first / total);
            fprintf(stderr, "  %-24s %9.3f ms\n", "total", total / 1000.0);
            stage_us.clear();
        }
    }

    bool profile = false;
    std::map<std::string, double> stage_us;

private:
    struct Block { char *qkv_q, *qkv_s, *qkv_b, *out_q, *out_s, *out_b, *gu_q, *gu_s, *gu_b, *down_q, *down_s, *down_b, *norm1, *norm2, *scale1, *scale2; };

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
    // m-tiles per raster group: of 4, 3, 2 the one that pads the tile rows least (ties to the
    // larger); scripts/build_kernels.py compiles the GEMMs with the same rule for this token count.
    unsigned gemm_m_group(size_t m) const {
        if (const char* e = std::getenv("H3_M_GROUP")) return unsigned(std::atoi(e));   // A/B override, mirrored in the builder
        const size_t tiles = (m + gemm_tile_ - 1) / gemm_tile_;
        unsigned best = 4; size_t best_pad = (tiles + 3) / 4 * 4;
        for (unsigned g : {3u, 2u}) { const size_t pad = (tiles + g - 1) / g * g; if (pad < best_pad) { best = g; best_pad = pad; } }
        return best;
    }
    unsigned gemm_grid_y(size_t m) const { const unsigned g = gemm_m_group(m); return unsigned(((m + gemm_tile_ - 1) / gemm_tile_ + g - 1) / g * g); }

    void gemm(Kernel &k, const char *stage, char *w_q, char *w_s, int n, void *out, const float *gate, const char *bias) {
        KernArgs a;
        a.scalar_i32(int(tokens_));
        a.pointer(a_q_); a.pointer(w_q); a.pointer(w_s); a.pointer(a_s_); a.pointer(out);
        if (gate) { a.pointer(gate); a.pointer(cls_); }
        a.pointer(bias);
        launch(k, stage, unsigned(n / 128), gemm_grid_y(tokens_), THREADS, a);
    }

    void block(int i) {
        const Block &b = blocks_[i];
        const unsigned T = unsigned(tokens_);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.norm1); a.pointer(mods_); a.pointer(cls_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, NORM_LANES, a); }
        gemm(k_gemm_qkv_, "gemm qkv", b.qkv_q, b.qkv_s, QKV, fused_, nullptr, b.qkv_b);
        { KernArgs a; a.scalar_i32(T); a.pointer(fused_); a.pointer(ones_); a.pointer(ones_); a.pointer(cos_); a.pointer(sin_); a.pointer(q_); a.pointer(k_); a.pointer(v_);
          launch(k_rope_, "qk norm + rope", T, 1, THREADS, a); }
        { KernArgs a; a.scalar_i32(T); a.scalar_i32(HEADS); a.pointer(q_); a.pointer(k_); a.pointer(v_); a.pointer(attn_);
          const unsigned qblock = 16 * unsigned(attn_waves_);
          launch(k_attention_, "attention", (T + qblock - 1) / qblock, HEADS, 32 * unsigned(attn_waves_), a); }
        { KernArgs a; a.scalar_i32(T); a.pointer(attn_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_attn_, "prepare out input", T, 1, ATTN_LANES, a); }
        gemm(k_gemm_out_, "gemm out + residual", b.out_q, b.out_s, HIDDEN, x_, (const float *)b.scale1, b.out_b);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.norm2); a.pointer(mods_); a.pointer(cls_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, NORM_LANES, a); }
        gemm(k_gemm_gu_, "gemm ff + swiglu", b.gu_q, b.gu_s, 2 * FFN, gu_, nullptr, b.gu_b);
        { KernArgs a; a.scalar_i32(T); a.pointer(gu_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_down_, "prepare down input", T, 1, DOWN_LANES, a); }
        gemm(k_gemm_down_, "gemm down + residual", b.down_q, b.down_s, HIDDEN, x_, (const float *)b.scale2, b.down_b);
    }

    int tokens_, layers_;
    size_t capacity_ = 0, gemm_tile_ = 128, attn_waves_ = 4, bits_ = 4;
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

struct h3vae_session { Session value; h3vae_session(const char *w, const char *k, int t, int l) : value(w, k, t, l) {} };

extern "C" uint32_t h3vae_abi_version(void) { return H3VAE_ABI_VERSION; }
extern "C" void h3vae_destroy(h3vae_session *s) { delete s; }
extern "C" int h3vae_profile(h3vae_session *s, int enable) { if (!s) return H3VAE_INVALID_ARGUMENT; s->value.profile = enable != 0; return H3VAE_OK; }

extern "C" int h3vae_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers, h3vae_session **out, char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    if (!out) { write_error(error, cap, "out_session must not be null"); return H3VAE_INVALID_ARGUMENT; }
    *out = nullptr;
    try {
        if (!weights_dir || !kernels_dir) throw std::invalid_argument("weights_dir and kernels_dir are required");
        *out = new h3vae_session(weights_dir, kernels_dir, tokens, layers);
        return H3VAE_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return H3VAE_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return H3VAE_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return H3VAE_ERROR; }
}

extern "C" int h3vae_run(h3vae_session *s, float *x, size_t x_elements, const float *cos, const float *sin, size_t rope_elements,
                         char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    try {
        if (!s || !x || !cos || !sin) throw std::invalid_argument("null argument");
        s->value.run(x, x_elements, cos, sin, rope_elements);
        return H3VAE_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return H3VAE_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return H3VAE_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return H3VAE_ERROR; }
}
