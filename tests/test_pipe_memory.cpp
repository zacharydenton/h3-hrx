// Failure injection without HIP, real weights, or a compiler.
#include "../host/h3pipe.cpp"
#include <cassert>
#include <filesystem>
#include <set>

struct RtKernel {};
struct FakeRt : Rt {
    std::map<void *, size_t> allocations;
    int fail_after = -1;
    bool fail_copy = false;
    size_t bytes = 0;
    const char *name() const override { return "fake"; }
    void *alloc(size_t n) override {
        if (fail_after == 0) throw std::runtime_error("injected allocation failure");
        if (fail_after > 0) --fail_after;
        void *p = std::malloc(std::max(n, size_t(1))); assert(p);
        allocations[p] = n; bytes += n; return p;
    }
    void free(void *p) override {
        assert(allocations.count(p)); bytes -= allocations.at(p); allocations.erase(p); std::free(p);
    }
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

int main(int argc, char **argv) {
    assert(argc == 2); const std::string dir = argv[1];
    const float values[] = {1, 2, 3, 4};
    std::ofstream(dir + "/weights.bin", std::ios::binary).write((const char *)values, sizeof values);
    std::ofstream(dir + "/manifest.txt") << "te.embed 0 8 torch.float32 2\nsmall 8 8 torch.float32 2\n";
    {
        Blob b; b.open(dir); assert(fake.bytes == 0); // metadata only
        assert(b.host_f32("small", 2)[0] == 3 && fake.bytes == 0);
        fake.fail_copy = true;
        try { b.at("small", 8); assert(false); } catch (const std::runtime_error &) {}
        assert(fake.bytes == 0 && b.tensors.empty());
        fake.fail_copy = false;
        const void *p = b.at("small", 8); assert(fake.bytes == 8);
        assert(b.at("small", 8) == p && fake.bytes == 8); // never allocates te.embed
        assert(((float *)p)[1] == 4);
        std::filesystem::resize_file(dir + "/weights.bin", 8);
        try { b.open(dir); assert(false); } catch (const std::runtime_error &) {}
        assert(b.at("small", 8) == p); // failed reopen keeps the valid state
    }
    assert(fake.allocations.empty());
    std::ofstream(dir + "/weights.bin", std::ios::binary).write((const char *)values, sizeof values);
    h3pipe_config cfg{}; cfg.glue_dir = dir.c_str(); cfg.blocks_dir = "/missing/dit"; cfg.te_dir = "/missing/te";
    cfg.kernel_sources = "/missing/kernels"; cfg.cache_dir = dir.c_str(); cfg.loom_compile = "/missing/compiler";
    for (int failure = 0; failure < 2; ++failure) {
        fake.fail_after = failure;
        try { Pipe pipe(cfg); assert(false); } catch (const std::runtime_error &) {}
        assert(fake.allocations.empty());
    }
    fake.fail_after = -1; fake.fail_copy = true;
    try { Pipe pipe(cfg); assert(false); } catch (const std::runtime_error &) {}
    assert(fake.allocations.empty()); fake.fail_copy = false;
    for (int failure = 0; failure < 10; ++failure) {
        { Pipe pipe(cfg); assert(fake.bytes < 256 * 1024); // opens without DiT or text weights
          fake.fail_after = failure;
          try { pipe.ensure_seq(16); assert(false); } catch (const std::runtime_error &) {}
          fake.fail_after = -1; pipe.ensure_seq(16); // retry after partially allocated buffers
        }
        assert(fake.allocations.empty());
    }
    try { DeviceBuffers stage; stage.alloc(64); stage.alloc(128); throw std::runtime_error("stage failed"); }
    catch (const std::runtime_error &) {}
    assert(fake.allocations.empty());
    puts("PASS lazy tensor loading, failed uploads/constructors/resizes, retry, scoped cleanup");
}
