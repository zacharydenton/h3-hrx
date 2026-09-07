// The checkpoint loader on a synthetic safetensors file, without HIP, a compiler or model weights: the header reader's
// errors, the recipes' layouts (concatenation, the 16-row gate/up interleave, the zero-padded pitch, widening), and the
// laziness and cleanup of the device uploads.
#include "../host/h3pipe.cpp"
#include <cassert>
#include <filesystem>

struct RtKernel {};
struct FakeRt : Rt {
    std::map<void *, size_t> allocations; size_t bytes = 0; bool fail_copy = false;
    const char *name() const override { return "fake"; }
    void *alloc(size_t n) override { void *p = std::malloc(std::max(n, size_t(1))); assert(p); allocations[p] = n; bytes += n; return p; }
    void free(void *p) override { assert(allocations.count(p)); bytes -= allocations.at(p); allocations.erase(p); std::free(p); }
    void memset(void *p, int v, size_t n) override { std::memset(p, v, n); }
    void h2d(void *d, const void *s, size_t n) override { if (fail_copy) throw std::runtime_error("injected upload failure"); std::memcpy(d, s, n); }
    void d2h(void *d, const void *s, size_t n) override { std::memcpy(d, s, n); }
    void d2d(void *d, const void *s, size_t n) override { std::memmove(d, s, n); }
    void sync() override {}
    RtKernel *load(const std::string &, const std::string &) override { throw std::runtime_error("unexpected kernel load"); }
    void unload(RtKernel *) override {}
    void launch(RtKernel *, unsigned, unsigned, unsigned, const KernArgs &) override { assert(false); }
} fake;
Rt &rt() { return fake; }

// name -> (dtype, shape, bytes) written as a safetensors file; the data is each tensor's index repeated, so rows are identifiable.
struct Tensor { std::string dtype; std::vector<int64_t> shape; std::vector<char> data; };
static void write_file(const std::string &path, const std::vector<std::pair<std::string, Tensor>> &tensors, long header_override = -1) {
    std::string header = "{"; size_t offset = 0; std::string blob;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const Tensor &t = tensors[i].second;
        header += (i ? "," : "") + std::string("\"") + tensors[i].first + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t d = 0; d < t.shape.size(); ++d) header += (d ? "," : "") + std::to_string(t.shape[d]);
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + t.data.size()) + "]}";
        blob.append(t.data.begin(), t.data.end()); offset += t.data.size();
    }
    header += "}";
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = header_override >= 0 ? uint64_t(header_override) : header.size();
    f.write((const char *)&n, 8); f.write(header.data(), std::streamsize(header.size())); f.write(blob.data(), std::streamsize(blob.size()));
}
static Tensor i8_rows(int rows, int cols, int base) { Tensor t{"I8", {rows, cols}, {}}; t.data.resize(size_t(rows) * cols); for (int r = 0; r < rows; ++r) std::memset(t.data.data() + size_t(r) * cols, base + r, cols); return t; }
static Tensor f32_vec(std::vector<float> v) { Tensor t{"F32", {int64_t(v.size())}, {}}; t.data.resize(v.size() * 4); memcpy(t.data.data(), v.data(), t.data.size()); return t; }
static Tensor f16_vec(std::vector<float> v) { Tensor t{"F16", {int64_t(v.size())}, {}}; t.data.resize(v.size() * 2); for (size_t i = 0; i < v.size(); ++i) { const uint16_t h = f32_to_f16(v[i]); memcpy(t.data.data() + i * 2, &h, 2); } return t; }
static Tensor bf16_vec(std::vector<float> v) { Tensor t{"BF16", {int64_t(v.size())}, {}}; t.data.resize(v.size() * 2); for (size_t i = 0; i < v.size(); ++i) { uint32_t x; memcpy(&x, &v[i], 4); const uint16_t b = uint16_t(x >> 16); memcpy(t.data.data() + i * 2, &b, 2); } return t; }

int main(int argc, char **argv) {
    assert(argc == 2); const std::string dir = argv[1], path = dir + "/tiny.safetensors";
    auto fails = [](const std::function<void()> &f, const char *needle) {
        try { f(); } catch (const std::exception &e) { assert(std::string(e.what()).find(needle) != std::string::npos); return; }
        assert(false);
    };
    // --- the header reader ---
    { std::ofstream(dir + "/short.bin", std::ios::binary) << "abc"; fails([&] { Checkpoint c(dir + "/short.bin"); }, "not a safetensors file"); }
    fails([&] { Checkpoint c(dir + "/absent.safetensors"); }, "cannot open");
    write_file(path, {{"w", i8_rows(4, 8, 1)}}, 1 << 20);
    fails([&] { Checkpoint c(path); }, "corrupt safetensors header length");
    { std::ofstream f(dir + "/past.safetensors", std::ios::binary); const std::string h = "{\"w\":{\"dtype\":\"I8\",\"shape\":[4,8],\"data_offsets\":[0,999]}}"; const uint64_t n = h.size();
      f.write((const char *)&n, 8); f.write(h.data(), std::streamsize(h.size())); f.write("0123", 4); }
    fails([&] { Checkpoint c(dir + "/past.safetensors"); }, "tensor span past the data");

    auto raw = [&](const std::string &header, const std::string &data = "") {
        std::ofstream f(path, std::ios::binary); const uint64_t n = header.size();
        f.write((const char *)&n, 8); f.write(header.data(), std::streamsize(header.size())); f.write(data.data(), std::streamsize(data.size()));
    };
    for (const std::string &shape : {"[4096]", "[]", "[1,4]"}) {
        raw("{\"w\":{\"dtype\":\"F32\",\"shape\":" + shape + ",\"data_offsets\":[0,0]}}");
        fails([&] { Checkpoint c(path); }, "byte count does not match");
    }
    raw("{\"w\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,8]}}", "12345678");
    fails([&] { Checkpoint c(path); }, "byte count does not match");
    for (const std::string &dim : {"-1", "1.5", "true", "null", "\"2\"", "1e999", "NaN", "9223372036854775808", "18446744073709551616"}) {
        raw("{\"w\":{\"dtype\":\"F32\",\"shape\":[" + dim + "],\"data_offsets\":[0,0]}}");
        fails([&] { Checkpoint c(path); }, "invalid integer dimension");
    }
    for (const std::string &offset : {"-1", "0.5", "false", "null", "\"0\"", "1e999", "NaN", "18446744073709551616"}) {
        raw("{\"w\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[" + offset + ",0]}}");
        fails([&] { Checkpoint c(path); }, "invalid integer offset");
    }
    raw("{\"w\":{\"dtype\":\"F32\",\"shape\":[4611686018427387904,4],\"data_offsets\":[0,0]}}");
    fails([&] { Checkpoint c(path); }, "tensor size overflow");
    raw("{\"w\":{\"dtype\":\"unknown\",\"shape\":[0],\"data_offsets\":[0,0]}}");
    fails([&] { Checkpoint c(path); }, "unsupported checkpoint dtype");
    // Scalars, empty tensors, and quantisation metadata remain valid.
    write_file(path, {{"scalar", {"F32", {}, std::vector<char>(4)}}, {"empty", {"BF16", {2, 0, 3}, {}}}, {"metadata", {"U8", {3}, {'a', 'b', 'c'}}}});
    { Checkpoint c(path); assert(c.at("scalar").elements() == 1 && c.at("empty").elements() == 0 && c.at("metadata").bytes == 3); }

    // An invalid embedding must be rejected on every retry, before committing the text checkpoint.
    const std::string te_path = dir + "/te.safetensors";
    h3pipe_config cfg{}; cfg.te_file = te_path.c_str(); cfg.kernel_sources = "unused"; cfg.cache_dir = "unused"; cfg.loom_compile = "unused";
    for (bool missing : {true, false}) {
        if (missing) write_file(te_path, {});
        else write_file(te_path, {{"model.embed_tokens.weight", {"F16", {1, TEXT_DIM}, std::vector<char>(TEXT_DIM * 2)}}});
        Pipe pipe(cfg);
        for (int retry = 0; retry < 2; ++retry) fails([&] { pipe.ensure_te(); }, missing ? "missing tensor model.embed_tokens.weight" : "expected BF16");
    }
    assert(fake.bytes == 0);

    write_file(path, {{"gate", i8_rows(32, 8, 0)}, {"up", i8_rows(32, 8, 100)}, {"gs", f32_vec(std::vector<float>(32, 1.5f))}, {"us", f32_vec(std::vector<float>(32, 2.5f))},
                      {"q", i8_rows(2, 8, 10)}, {"k", i8_rows(3, 8, 20)}, {"half", f16_vec({1.0f, -2.5f, 0.125f})}, {"brain", bf16_vec({3.0f, -0.5f})}, {"metadata_free", f32_vec({7.0f})}});
    Checkpoint ck(path);
    assert(ck.has("gate") && !ck.has("absent"));
    fails([&] { ck.at("absent"); }, "missing tensor absent");
    fails([&] { ck.at("gate", "F16", {32, 8}); }, "gate is I8 [32, 8], expected F16 [32, 8]");
    fails([&] { ck.at("gate", "I8", {16, 8}); }, "expected I8 [16, 8]");
    ck.at("gate", "I8", {-1, 8});   // a free dimension

    Weights w;
    struct Plan { static void run(const Checkpoint &c, Weights &out) {
        out.add("cat", rows_of(c, {&c.at("q"), &c.at("k")}));                                  // 5 rows of 8
        out.add("cat_pitch", rows_of(c, {&c.at("q"), &c.at("k")}, 12));                        // the same at a 12-byte pitch
        out.add("gu", interleave16(c, c.at("gate"), 0, c.at("up"), 0, 32));                    // 64 rows: 16 gate, 16 up, 16 gate, 16 up
        out.add("gu.s", scales_interleave16(c, c.at("gs"), 0, c.at("us"), 0, 32));
        out.add("widened", widen_f32(c, {&c.at("half"), &c.at("brain")}));
        out.add("scales", scales_rows(c, {&c.at("gs")}));
    } };
    w.open(path, Plan::run);
    fails([&] { w.recipe("nothing"); }, "no recipe for tensor nothing");
    assert(fake.bytes == 0);   // the plan uploads nothing

    { const char *p = w.at("cat", 5 * 8); assert(fake.bytes == 40);
      for (int r = 0; r < 5; ++r) assert(p[r * 8] == char(r < 2 ? 10 + r : 20 + (r - 2)));   // q's rows, then k's
      assert(w.at("cat", 40) == p && fake.bytes == 40); }                                     // memoised, uploaded once
    { const char *p = w.at("cat_pitch", 5 * 12);
      for (int r = 0; r < 5; ++r) { assert(p[r * 12] == char(r < 2 ? 10 + r : 20 + (r - 2))); for (int c = 8; c < 12; ++c) assert(p[r * 12 + c] == 0); } }   // the pad is zero
    { const char *p = w.at("gu", 64 * 8);
      for (int r = 0; r < 64; ++r) { const int run = r / 16, within = r % 16; const int want = (run % 2 == 0) ? (run / 2) * 16 + within : 100 + (run / 2) * 16 + within; assert(p[r * 8] == char(want)); } }
    { const float *s = (const float *)w.at("gu.s", 64 * 4);
      for (int r = 0; r < 64; ++r) assert(s[r] == ((r / 16) % 2 == 0 ? 1.5f : 2.5f)); }
    { const std::vector<float> v = w.host_f32("widened", 5); assert(v[0] == 1.0f && v[1] == -2.5f && v[2] == 0.125f && v[3] == 3.0f && v[4] == -0.5f); }
    fails([&] { w.at("cat", 41); }, "expected 41");
    fails([&] { w.rows("cat", 5, 8, 12); }, "is not 5 rows of 8 bytes at pitch 12");
    assert(w.rows("cat_pitch", 5, 8, 12) == w.at("cat_pitch", 60));

    // a failed upload leaves neither an allocation nor a memo
    const size_t before = fake.bytes; fake.fail_copy = true;
    fails([&] { w.at("scales", 32 * 4); }, "injected upload failure");
    assert(fake.bytes == before); fake.fail_copy = false;
    assert(w.at("scales", 32 * 4) != nullptr && fake.bytes == before + 128);

    // a plan that rejects the file keeps the previous recipes and mapping
    struct Bad { static void run(const Checkpoint &c, Weights &out) { out.add("x", rows_of(c, {&c.at("q")})); c.at("gate", "F32", {32, 8}); } };
    fails([&] { w.open(path, Bad::run); }, "gate is I8");
    assert(w.has("cat") && !w.has("x") && w.at("cat", 40) != nullptr);

    // the DiT plan names the first tensor it cannot find in a file that is not the checkpoint
    fails([&] { Weights d; d.open(path, dit_plan); }, "missing tensor blocks.0.attn.qkv_proj.weight");
    puts("PASS checkpoint header errors, recipe layouts (concat, interleave, pitch, widen), lazy memoised uploads and failure cleanup");
}
