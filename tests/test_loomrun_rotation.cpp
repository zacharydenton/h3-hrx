// No HIP runtime is linked: verify pointer rotation and invalid CLI handling on CPU.
#define main loomrun_main
#include "../host/loomrun.cpp"
#undef main
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

static std::vector<void *> seen;
static hipError_t capture(void **extra) {
    auto *args = static_cast<unsigned char *>(extra[1]);
    void *weight = nullptr;
    std::memcpy(&weight, args + 16, 8);  // i32, A pointer, W pointer
    assert(*static_cast<float *>(weight) == 7.0f);
    seen.push_back(weight);
    return hipSuccess;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string input = std::string(argv[1]) + "/rotate.bin";
    const float value = 7.0f;
    std::ofstream(input, std::ios::binary).write(reinterpret_cast<const char *>(&value), sizeof(value));
    auto run = [&](const std::string &rotation, const std::string &repeat = "7") {
        std::vector<std::string> words = {"loomrun", "--hsaco", "fake", "--kernel", "fake", "--repeat", repeat,
            "--i32", "1", "--in", input, "--in", input, "--rotate-input", rotation};
        std::vector<char *> args;
        for (auto &word : words) args.push_back(word.data());
        return loomrun_main(int(args.size()), args.data());
    };
    for (const std::string invalid : {"-1:3", "2:1", "2:65", "3:3", "0:3", "2:3junk"}) {
        assert(run(invalid) == 64);
        assert(fake_init_count == 0);
    }
    assert(run("2:3", "0") == 64 && fake_init_count == 0);
    fake_module_load_failure = fake_symbol_failure = false;
    fake_launch_hook = capture;
    assert(run("2:3") == 0);
    assert(seen.size() == 8);  // one warmup, seven timed launches
    assert(seen[0] != seen[1] && seen[1] != seen[2] && seen[0] != seen[2]);
    for (size_t i = 0; i < seen.size(); ++i) assert(seen[i] == seen[i % 3]);
    assert(fake_allocations.empty() && fake_modules.empty());
    seen.clear();
    assert(run("2:3", "1") == 0 && seen.size() == 1);  // no hidden warmup
    assert(fake_allocations.empty() && fake_modules.empty());
    puts("PASS CPU-only streaming benchmark pointer rotation, cleanup and CLI validation");
}
