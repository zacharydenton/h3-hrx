// CPU integration gate for the experimental decoder stack. The fake compiler
// records configs; the fake runtime checks dispatch geometry and moves row tags
// through SwiGLU -> optional prepare -> down. No GPU or model weights are used.
#include "../host/h3pipe.cpp"
#include <cassert>
#include <filesystem>

struct RtKernel {
    std::string symbol;
    std::map<std::string, int> cfg;
};
struct FakeRt : Rt {
    std::map<void *, size_t> allocations;
    bool expect_wide = false, expect_fast = false, expect_fused = false;
    int prepares = 0, gemms = 0, downs = 0, ropes = 0, fused_attentions = 0;
    const void *gu = nullptr, *qkv = nullptr;
    size_t qkv_capacity = 0;
    const char *name() const override { return "fake"; }
    void *alloc(size_t n) override {
        assert(n <= 128 * 1024 * 1024);
        void *p = std::calloc(1, std::max(n, size_t(1))); assert(p);
        allocations[p] = n; return p;
    }
    void free(void *p) override { assert(allocations.erase(p)); std::free(p); }
    void check(const void *p, size_t bytes) const {
        const auto it = allocations.find(const_cast<void *>(p));
        assert(it != allocations.end() && it->second >= bytes);
    }
    void memset(void *p, int value, size_t n) override { check(p, n); std::memset(p, value, n); }
    void h2d(void *d, const void *s, size_t n) override { std::memcpy(d, s, n); }
    void d2h(void *d, const void *s, size_t n) override { std::memcpy(d, s, n); }
    void d2d(void *d, const void *s, size_t n) override { std::memmove(d, s, n); }
    void sync() override {}
    RtKernel *load(const std::string &path, const std::string &symbol) override {
        auto *k = new RtKernel{symbol, {}};
        std::ifstream f(path); std::string line;
        while (std::getline(f, line)) if (line.rfind("--config=", 0) == 0) {
            const size_t eq = line.find('=', 9), dot = line.rfind('.', eq);
            k->cfg[line.substr(dot + 1, eq - dot - 1)] = std::atoi(line.c_str() + eq + 1);
        }
        return k;
    }
    void unload(RtKernel *k) override { delete k; }
    void launch(RtKernel *k, unsigned gx, unsigned gy, unsigned bx, const KernArgs &a) override {
        const size_t tokens = a.scalars[0];
        if (k->symbol.find("gemm_f16") != std::string::npos) {
            ++gemms;
            const bool wide = k->symbol.find("_wide_") != std::string::npos;
            const bool fused = k->symbol == "h3_gemm_f16_qkvropehm_256b";
            const bool fast = fused || k->symbol.find("_fast_") != std::string::npos;
            const size_t tm = fast ? 128 : 256;
            assert(fast == expect_fast);
            assert(wide == expect_wide && bx == (wide ? 512u : 256u));
            assert(gx == unsigned(k->cfg.at("n_size") / ((wide || fast) ? 256 : 128)));
            const unsigned group = expect_fused && tokens == 1797
                ? (k->cfg.at("k_size") == 8192 ? 1u : 15u) : m_group_for(tokens, tm);
            assert(gy == gemm_grid_y(tokens, group, tm));
            assert(k->cfg.at("m_group") == int(group));
            check(a.ptrs[0], tokens * size_t(k->cfg.at("k_stride")) * 2);
            check(a.ptrs[1], size_t(k->cfg.at("n_size")) * k->cfg.at("k_stride") * 2);
            if (fused) {
                assert(expect_fused && k->cfg.at("n_size") == 6144 && k->cfg.at("k_size") == 2048);
                assert(a.nptrs == 8);
                qkv = a.ptrs[2]; qkv_capacity = k->cfg.at("token_capacity");
                check(qkv, qkv_capacity * 6144 * 2);
                assert(qkv_capacity >= tokens + 16);
            } else if (k->symbol.find("swiglu") != std::string::npos) {
                const size_t stride = (wide || fast) ? k->cfg.at("out_stride") : 512;
                assert(stride == size_t(k->cfg.at("n_size") / 2 + ((wide || fast) ? 128 : 0)));
                check(a.ptrs[2], tokens * stride * 2); gu = a.ptrs[2];
                auto *out = static_cast<uint16_t *>(const_cast<void *>(gu));
                for (size_t row = 0; row < tokens; ++row) out[row * stride] = f32_to_f16(float(row + 1));
            } else if (k->symbol.find("resid") != std::string::npos && gu) {
                ++downs;
                assert((a.ptrs[0] == gu) == (wide || fast));
                const auto *in = static_cast<const uint16_t *>(a.ptrs[0]);
                for (size_t row = 0; row < tokens; ++row) assert(in[row * size_t(k->cfg.at("k_stride"))] == f32_to_f16(float(row + 1)));
                gu = nullptr;
            }
        } else if (k->symbol == "h3_attention_mha64hm32_lds_f16_wmma") {
            ++fused_attentions;
            assert(expect_fused && qkv && a.ptrs[0] == qkv);
            const size_t component_bytes = qkv_capacity * 2048 * 2;
            assert(a.ptrs[1] == static_cast<const char *>(qkv) + component_bytes);
            assert(a.ptrs[2] == static_cast<const char *>(qkv) + 2 * component_bytes);
            assert(bx == 128 && gx == (tokens + 63) / 64 && gy == 32);
            assert(size_t(k->cfg.at("token_capacity")) == qkv_capacity);
            assert(k->cfg.at("q_stride") == 2048 && k->cfg.at("kv_stride") == 2048);
            assert(k->cfg.at("out_stride") == 2176);
            check(a.ptrs[3], qkv_capacity * 2176 * 2);
        } else if (k->symbol.rfind("h3_rope", 0) == 0) {
            ++ropes;
            assert(!expect_fused);
        } else if (k->symbol == "h3_prepare_plain_f16") {
            ++prepares;
            assert(!expect_wide && !expect_fast && a.ptrs[0] == gu && k->cfg.at("out_stride") == 640);
            check(a.ptrs[0], tokens * 512 * 2); check(a.ptrs[1], tokens * 640 * 2);
            const auto *in = static_cast<const uint16_t *>(a.ptrs[0]);
            auto *out = static_cast<uint16_t *>(const_cast<void *>(a.ptrs[1]));
            for (size_t row = 0; row < tokens; ++row) std::memcpy(out + row * 640, in + row * 512, 512 * 2);
        }
    }
} fake;
Rt &rt() { return fake; }

static void plan(const Checkpoint &ck, Weights &w, int hidden = 512, int ffn = 512) {
    const char *src = ck.data(ck.at("bytes"));
    auto add = [&](const std::string &name, size_t rows, size_t bytes, size_t pitch) {
        Recipe r; r.rows = rows; r.row_bytes = bytes; r.pitch_bytes = pitch;
        r.segments.push_back({src, rows}); w.add("blocks.0." + name, std::move(r));
    };
    for (const auto &part : {std::make_pair("qkv", 3 * hidden), {"out", hidden}, {"gu", 2 * ffn}, {"down", hidden}}) {
        const size_t k = std::string(part.first) == "down" ? ffn : hidden;
        add(std::string(part.first) + ".q", part.second, k * 2, (k + 128) * 2);
        add(std::string(part.first) + ".b", 1, part.second * 4, part.second * 4);
    }
    for (const char *name : {"norm1", "norm2"}) add(name, 1, hidden * 4, hidden * 4);
}

int main(int argc, char **argv) {
    assert(argc == 2); const std::string dir = argv[1], file = dir + "/wide.safetensors";
    const size_t bytes = 128 * 1024 * 1024;
    const std::string header = "{\"bytes\":{\"dtype\":\"U8\",\"shape\":[134217728],\"data_offsets\":[0,134217728]}}";
    { std::ofstream f(file, std::ios::binary); const uint64_t n = header.size();
      f.write(reinterpret_cast<const char *>(&n), 8); f << header; }
    std::filesystem::resize_file(file, 8 + header.size() + bytes);
    const std::string compiler = dir + "/record-compiler";
    { std::ofstream f(compiler); f << "#!/bin/sh\nfor arg do\ncase $arg in --output=*) out=${arg#--output=};; esac\ndone\nprintf '%s\\n' \"$@\" > \"$out\"\n"; }
    std::filesystem::permissions(compiler, std::filesystem::perms::owner_all);
    for (int mode : {0, 1, 2, 3}) for (size_t tokens : {size_t(1), size_t(250), size_t(517), size_t(1797)}) {
        const bool wide = mode == 1, fast = mode >= 2, fused = mode == 3;
        const int hidden = fused ? 2048 : 512, ffn = fused ? 8192 : 512;
        setenv("H3_VAE_WIDE", wide ? "1" : "0", 1);
        setenv("H3_VAE_FAST", fast ? "1" : "0", 1);
        fake.expect_wide = wide; fake.expect_fast = fast; fake.expect_fused = fused;
        fake.prepares = fake.gemms = fake.downs = fake.ropes = fake.fused_attentions = 0; fake.qkv = nullptr;
        {
            Compiler c; c.exe = compiler; c.sources = "kernels"; c.cache = dir + "/wide-cache";
            Weights w; w.open(file, fused
                ? +[](const Checkpoint &ck, Weights &weights) { plan(ck, weights, 2048, 8192); }
                : +[](const Checkpoint &ck, Weights &weights) { plan(ck, weights); });
            Stack stack(c, StackDims{hidden, hidden / 64, hidden / 64, 64, ffn, 48, 1, 16, 1e-5f, true, false, false},
                        tokens, 1, w, "blocks.%d.", false, nullptr, "vae");
            DeviceBuffers memory;
            void *x = memory.alloc(stack.capacity() * hidden * 4);
            stack.forward(nullptr, x, nullptr, nullptr, nullptr, [](int) { return LayerCond{nullptr, nullptr, nullptr, nullptr}; });
            assert(fake.gemms == 4 && fake.downs == 1 && fake.prepares == ((wide || fast) ? 0 : 1));
            assert(fake.ropes == (fused ? 0 : 1) && fake.fused_attentions == (fused ? 1 : 0));
        }
        assert(fake.allocations.empty());
    }
    puts("PASS CPU decoder dispatch, padded allocations, SwiGLU routing and fused head-major QKV routing");
}
