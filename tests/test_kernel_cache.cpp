// The native kernel cache without HIP or a compiler: the cache identity covers the source, symbol, config and the compiler
// binary; a failed compile leaves nothing behind; publication goes through a lock and a unique temporary.
#include "../host/h3pipe.cpp"
#include <cassert>
#include <filesystem>
#include <thread>

struct RtKernel { std::string path; };
struct FakeRt : Rt {
    const char *name() const override { return "fake"; }
    void *alloc(size_t n) override { return std::malloc(std::max(n, size_t(1))); }
    void free(void *p) override { std::free(p); }
    void memset(void *p, int v, size_t n) override { std::memset(p, v, n); }
    void h2d(void *d, const void *s, size_t n) override { std::memcpy(d, s, n); }
    void d2h(void *d, const void *s, size_t n) override { std::memcpy(d, s, n); }
    void d2d(void *d, const void *s, size_t n) override { std::memmove(d, s, n); }
    void sync() override {}
    RtKernel *load(const std::string &path, const std::string &) override { return new RtKernel{path}; }
    void unload(RtKernel *k) override { delete k; }
    void launch(RtKernel *, unsigned, unsigned, unsigned, const KernArgs &) override { assert(false); }
} fake;
Rt &rt() { return fake; }

static std::string slurp(const std::string &path) { std::ifstream f(path); return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }

int main(int argc, char **argv) {
    assert(argc == 2); const std::string dir = argv[1], cache = dir + "/cache";
    std::ofstream(dir + "/k.loom") << "source v1";
    auto compiler = [&](const char *version, bool fail = false) {   // a stand-in loom-compile: writes its version to --output=
        std::ofstream out(dir + "/compiler");
        out << "#!/bin/sh\nfor arg in \"$@\"; do case \"$arg\" in --output=*) output=${arg#--output=};; esac; done\n";
        if (fail) out << "printf 'partial' > \"$output\"; exit 1\n"; else out << "sleep 0.2; printf '" << version << "' > \"$output\"\n";
        out.close(); chmod((dir + "/compiler").c_str(), 0755);
        struct timespec ts[2] = {{0, UTIME_NOW}, {0, UTIME_NOW}}; utimensat(AT_FDCWD, (dir + "/compiler").c_str(), ts, 0);
    };
    auto binary = [&](const Cfg &cfg = {}, const char *symbol = "entry") { Compiler c; c.exe = dir + "/compiler"; c.sources = dir; c.cache = cache; return slurp(c.get("k", symbol, cfg)->k->path); };
    auto count = [&] { size_t n = 0; for (auto &e : std::filesystem::directory_iterator(cache)) if (e.path().extension() == ".hsaco") ++n; return n; };
    compiler("v1");
    assert(binary() == "v1" && count() == 1);
    assert(binary() == "v1" && count() == 1);                              // the same request loads the cached binary
    assert(binary({{"h3.k.width", "64"}}) == "v1" && count() == 2);        // a different config is another kernel
    assert(binary({{"h3.k.width", "64"}}, "other") == "v1" && count() == 3);   // so is another entry symbol
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    compiler("v2");
    assert(binary() == "v2" && count() == 4);                              // a replaced compiler never reuses its predecessor's binary
    std::ofstream(dir + "/k.loom") << "source v2";
    assert(binary() == "v2" && count() == 5);                              // nor does an edited source
    // a failed compile publishes nothing and leaves no temporary behind
    std::ofstream(dir + "/k.loom") << "source v3"; compiler("v3", true);
    bool failed = false; try { binary(); } catch (const std::runtime_error &e) { failed = std::string(e.what()).find("loom-compile failed") != std::string::npos; }
    assert(failed && count() == 5);
    for (auto &e : std::filesystem::directory_iterator(cache)) assert(e.path().string().find(".tmp.") == std::string::npos);
    // two processes asking for the same kernel at once: one compiles, the other waits on the lock and loads the same binary
    compiler("v3");
    std::vector<std::string> got(2); std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) threads.emplace_back([&, i] { got[i] = binary(); });
    for (auto &t : threads) t.join();
    assert(got[0] == "v3" && got[1] == "v3" && count() == 6);
    puts("PASS kernel cache identity (source, symbol, config, compiler), failure cleanup and locked publication");
}
