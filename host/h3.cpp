// The 50 MiniMax H3 transformer blocks as a resident Loom session: per block ten launches
// (AdaLN prepare -> fused qkv GEMM -> QK-norm+RoPE (+ contiguous q/k/v) -> attention ->
// prepare -> out GEMM with the class-gated residual -> AdaLN prepare -> fused gate|up GEMM
// with the SwiGLU product -> prepare -> down GEMM with the class-gated residual), W4A4
// ConvRot throughout. Modulation comes from per-class tables (class = timestep class * 3 +
// modality of the row).
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

#include "h3.h"

namespace {

constexpr int HIDDEN = 5376, HEADS = 56, HEAD_DIM = 128, INNER = HEADS * HEAD_DIM, FFN = 14336;
constexpr int QKV = 3 * INNER;                                  // 21504
constexpr int CLASSES = int(H3_CLASSES), ROPE_HALF = 48;
constexpr int THREADS = 256, NORM_LANES = 96, ATTN_LANES = 128, DOWN_LANES = 256;   // prepare workgroups: width / (8 * lanes) chunks
constexpr size_t MODS_PER_LAYER = size_t(6) * CLASSES * HIDDEN;

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
        if (layers < 1 || layers > 50) throw std::invalid_argument("layers must be 1..50");
        HIP_CHECK(hipInit(0));
        capacity_ = std::max<size_t>((tokens + 16 + 31) / 32 * 32, (tokens + 63) / 64 * 64);   // tokens+16 headroom, whole 64-key blocks
        auto spans = read_manifest(weights_dir + "/manifest.txt");
        auto blob = read_file(weights_dir + "/weights.bin");
        for (const auto &e : spans)
            if (e.second.offset + e.second.bytes > blob.size())
                throw std::runtime_error("manifest span '" + e.first + "' runs past weights.bin");
        HIP_CHECK(hipMalloc(&weights_, blob.size()));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights_, blob.data(), blob.size()));
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
            b.qkv_q = need(p + ".qkv.q", size_t(QKV) * HIDDEN / 2);   b.qkv_s = need(p + ".qkv.s", size_t(QKV) * 4);
            b.out_q = need(p + ".out.q", size_t(HIDDEN) * INNER / 2);  b.out_s = need(p + ".out.s", size_t(HIDDEN) * 4);
            b.gu_q = need(p + ".gu.q", size_t(2 * FFN) * HIDDEN / 2);  b.gu_s = need(p + ".gu.s", size_t(2 * FFN) * 4);
            b.down_q = need(p + ".down.q", size_t(HIDDEN) * FFN / 2);  b.down_s = need(p + ".down.s", size_t(HIDDEN) * 4);
            b.norm1 = need(p + ".norm1", HIDDEN * 4); b.norm2 = need(p + ".norm2", HIDDEN * 4);
            b.qnorm = need(p + ".qnorm", HEAD_DIM * 4); b.knorm = need(p + ".knorm", HEAD_DIM * 4);
            blocks_.push_back(b);
        }
        auto load = [&](Kernel &k, const char *stem, const char *symbol) { k.load(kernels_dir + "/" + stem + ".hsaco", symbol); };
        load(k_prep_norm_, "prepare_norm_i4", "h3_prepare_norm_i4");
        load(k_prep_attn_, "prepare_attn_i4", "h3_prepare_plain_i4");
        load(k_prep_down_, "prepare_down_i4", "h3_prepare_plain_i4");
        load(k_gemm_qkv_, "gemm_qkv", "h3_gemm_i4");
        load(k_gemm_gu_, "gemm_gu", "h3_gemm_i4_swiglu");
        load(k_gemm_out_, "gemm_out", "h3_gemm_i4_resid");
        load(k_gemm_down_, "gemm_down", "h3_gemm_i4_resid");
        load(k_rope_, "rope_qknorm", "h3_rope_qknorm_f16");
        load(k_attention_, "attention", "h3_attention_mha_lds_f16_wmma");
        const size_t T = capacity_;
        HIP_CHECK(hipMalloc(&x_, T * HIDDEN * 2));
        HIP_CHECK(hipMalloc(&a_q_, T * FFN / 2));               // the widest prepared operand (down's K = 14336)
        HIP_CHECK(hipMalloc(&a_s_, T * 4));
        HIP_CHECK(hipMalloc(&fused_, T * QKV * 2));
        HIP_CHECK(hipMalloc(&q_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&k_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&v_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&attn_, T * INNER * 2));
        HIP_CHECK(hipMalloc(&gu_, T * FFN * 2));                // silu(gate) * up, fused into the GEMM epilogue
        HIP_CHECK(hipMalloc(&mods_, size_t(layers) * MODS_PER_LAYER * 4));
        HIP_CHECK(hipMalloc(&cls_, T * 4));
        HIP_CHECK(hipMalloc(&cos_, T * ROPE_HALF * 4));
        HIP_CHECK(hipMalloc(&sin_, T * ROPE_HALF * 4));
        for (void *p : {x_, fused_, q_, k_, v_, cls_}) HIP_CHECK(hipMemset(p, 0, p == cls_ ? T * 4 : (p == x_ ? T * HIDDEN * 2 : (p == fused_ ? T * QKV * 2 : T * INNER * 2))));   // headroom rows stay zero
    }
    ~Session() {
        for (void *p : {x_, a_q_, a_s_, fused_, q_, k_, v_, attn_, gu_, mods_, cls_, cos_, sin_, weights_}) if (p) (void)hipFree(p);
        for (Kernel *k : {&k_prep_norm_, &k_prep_attn_, &k_prep_down_, &k_gemm_qkv_, &k_gemm_gu_, &k_gemm_out_, &k_gemm_down_, &k_rope_, &k_attention_})
            if (k->module) (void)hipModuleUnload(k->module);
    }

    void run(uint16_t *x, size_t x_elements, const int32_t *cls, size_t cls_elements, const float *mods, size_t mods_elements,
             const float *cos, const float *sin, size_t rope_elements) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t T = tokens_;
        if (x_elements != T * HIDDEN) throw std::invalid_argument("x has " + std::to_string(x_elements) + " elements, expected " + std::to_string(T * HIDDEN));
        if (cls_elements != T) throw std::invalid_argument("cls has the wrong element count");
        for (size_t i = 0; i < T; ++i) if (cls[i] < 0 || cls[i] >= CLASSES) throw std::invalid_argument("cls value out of range");
        if (mods_elements != size_t(layers_) * MODS_PER_LAYER) throw std::invalid_argument("mods has the wrong element count");
        if (rope_elements != T * ROPE_HALF) throw std::invalid_argument("cos/sin have the wrong element count");
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)x_, x, T * HIDDEN * 2));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)cls_, cls, T * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)mods_, mods, mods_elements * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)cos_, cos, rope_elements * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)sin_, sin, rope_elements * 4));
        for (int i = 0; i < layers_; ++i) block(i);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpyDtoH(x, (hipDeviceptr_t)x_, T * HIDDEN * 2));
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
    // m-tiles per raster group: of 4, 3, 2 the one that pads the tile rows least (ties to the
    // larger); scripts/build_kernels.py compiles the GEMMs with the same rule for this token count.
    static unsigned gemm_m_group(size_t m) {
        if (const char* e = std::getenv("H3_M_GROUP")) return unsigned(std::atoi(e));   // A/B override, mirrored in the builder
        const size_t tiles = (m + 127) / 128;
        unsigned best = 4; size_t best_pad = (tiles + 3) / 4 * 4;
        for (unsigned g : {3u, 2u}) { const size_t pad = (tiles + g - 1) / g * g; if (pad < best_pad) { best = g; best_pad = pad; } }
        return best;
    }
    static unsigned gemm_grid_y(size_t m) { const unsigned g = gemm_m_group(m); return unsigned(((m + 127) / 128 + g - 1) / g * g); }

    void gemm(Kernel &k, const char *stage, char *w_q, char *w_s, int n, void *out, const float *gate) {
        KernArgs a;
        a.scalar_i32(int(tokens_));
        a.pointer(a_q_); a.pointer(w_q); a.pointer(w_s); a.pointer(a_s_); a.pointer(out);
        if (gate) { a.pointer(gate); a.pointer(cls_); }
        launch(k, stage, unsigned(n / 128), gemm_grid_y(tokens_), THREADS, a);
    }

    void block(int i) {
        const Block &b = blocks_[i];
        const float *mod = (const float *)mods_ + size_t(i) * MODS_PER_LAYER;
        const float *table_msa = mod, *gate_msa = mod + size_t(CLASSES) * 2 * HIDDEN;
        const float *table_mlp = gate_msa + size_t(CLASSES) * HIDDEN, *gate_mlp = table_mlp + size_t(CLASSES) * 2 * HIDDEN;
        const unsigned T = unsigned(tokens_);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.norm1); a.pointer(table_msa); a.pointer(cls_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, NORM_LANES, a); }
        gemm(k_gemm_qkv_, "gemm qkv", b.qkv_q, b.qkv_s, QKV, fused_, nullptr);
        { KernArgs a; a.scalar_i32(T); a.pointer(fused_); a.pointer(b.qnorm); a.pointer(b.knorm); a.pointer(cos_); a.pointer(sin_); a.pointer(q_); a.pointer(k_); a.pointer(v_);
          launch(k_rope_, "qk norm + rope", T, 1, THREADS, a); }
        { KernArgs a; a.scalar_i32(T); a.scalar_i32(HEADS); a.pointer(q_); a.pointer(k_); a.pointer(v_); a.pointer(attn_);
          launch(k_attention_, "attention", unsigned((T + 63) / 64), HEADS, 128, a); }
        { KernArgs a; a.scalar_i32(T); a.pointer(attn_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_attn_, "prepare out input", T, 1, ATTN_LANES, a); }
        gemm(k_gemm_out_, "gemm out + residual", b.out_q, b.out_s, HIDDEN, x_, gate_msa);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.norm2); a.pointer(table_mlp); a.pointer(cls_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, NORM_LANES, a); }
        gemm(k_gemm_gu_, "gemm gate|up + swiglu", b.gu_q, b.gu_s, 2 * FFN, gu_, nullptr);
        { KernArgs a; a.scalar_i32(T); a.pointer(gu_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_down_, "prepare down input", T, 1, DOWN_LANES, a); }
        gemm(k_gemm_down_, "gemm down + residual", b.down_q, b.down_s, HIDDEN, x_, gate_mlp);
    }

    int tokens_, layers_;
    size_t capacity_ = 0;
    std::mutex mutex_;
    std::vector<Block> blocks_;
    void *weights_ = nullptr, *x_ = nullptr, *a_q_ = nullptr, *a_s_ = nullptr, *fused_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr,
         *attn_ = nullptr, *gu_ = nullptr, *mods_ = nullptr, *cls_ = nullptr, *cos_ = nullptr, *sin_ = nullptr;
    Kernel k_prep_norm_, k_prep_attn_, k_prep_down_, k_gemm_qkv_, k_gemm_gu_, k_gemm_out_, k_gemm_down_, k_rope_, k_attention_;
};

void write_error(char *error, size_t capacity, const char *message) noexcept {
    if (error && capacity) std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

}  // namespace

struct h3_session { Session value; h3_session(const char *w, const char *k, int t, int l) : value(w, k, t, l) {} };

extern "C" uint32_t h3_abi_version(void) { return H3_ABI_VERSION; }
extern "C" void h3_destroy(h3_session *s) { delete s; }
extern "C" int h3_profile(h3_session *s, int enable) { if (!s) return H3_INVALID_ARGUMENT; s->value.profile = enable != 0; return H3_OK; }

extern "C" int h3_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers, h3_session **out, char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    if (!out) { write_error(error, cap, "out_session must not be null"); return H3_INVALID_ARGUMENT; }
    *out = nullptr;
    try {
        if (!weights_dir || !kernels_dir) throw std::invalid_argument("weights_dir and kernels_dir are required");
        *out = new h3_session(weights_dir, kernels_dir, tokens, layers);
        return H3_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return H3_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return H3_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return H3_ERROR; }
}

extern "C" int h3_run(h3_session *s, uint16_t *x, size_t x_elements, const int32_t *cls, size_t cls_elements,
                      const float *mods, size_t mods_elements, const float *cos, const float *sin, size_t rope_elements,
                      char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    try {
        if (!s || !x || !cls || !mods || !cos || !sin) throw std::invalid_argument("null argument");
        s->value.run(x, x_elements, cls, cls_elements, mods, mods_elements, cos, sin, rope_elements);
        return H3_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return H3_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return H3_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return H3_ERROR; }
}
