// Session failure injection without HIP, checkpoints, or a compiler (the loader's own cases are tests/test_weights.cpp).
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
    h3pipe_config cfg{}; cfg.dit_file = "/missing/dit.safetensors"; cfg.te_file = "/missing/te.safetensors";
    cfg.kernel_sources = "/missing/kernels"; cfg.cache_dir = dir.c_str(); cfg.loom_compile = "/missing/compiler";
    for (int failure = 0; failure < 2; ++failure) {
        fake.fail_after = failure;
        try { Pipe pipe(cfg); assert(false); } catch (const std::runtime_error &) {}
        assert(fake.allocations.empty());
    }
    fake.fail_copy = true; fake.fail_after = -1;
    try { Pipe pipe(cfg); assert(false); } catch (const std::runtime_error &) {}
    assert(fake.allocations.empty()); fake.fail_copy = false;
    { Pipe pipe(cfg); bool failed = false;   // a missing checkpoint is named, not a crash
      try { pipe.ensure_dit(); } catch (const std::exception &e) { failed = std::string(e.what()).find("/missing/dit.safetensors") != std::string::npos; }
      assert(failed); }
    assert(fake.allocations.empty());
    int buffers = 0;   // how many allocations one ensure_seq makes: every one of them is failure-injected below
    { Pipe pipe(cfg); const size_t before = fake.allocations.size(); pipe.ensure_seq(16); buffers = int(fake.allocations.size() - before); }
    assert(buffers > 0 && fake.allocations.empty());
    for (int failure = 0; failure < buffers; ++failure) {
        { Pipe pipe(cfg); assert(fake.bytes < 256 * 1024); // opens without the checkpoints
          fake.fail_after = failure;
          try { pipe.ensure_seq(16); assert(false); } catch (const std::runtime_error &) {}
          fake.fail_after = -1; pipe.ensure_seq(16); // retry after partially allocated buffers
        }
        assert(fake.allocations.empty());
    }
    try { DeviceBuffers stage; stage.alloc(64); stage.alloc(128); throw std::runtime_error("stage failed"); }
    catch (const std::runtime_error &) {}
    assert(fake.allocations.empty());
    puts("PASS failed session constructors and resizes, retry, scoped cleanup");
}
