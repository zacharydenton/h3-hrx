#if defined(TEST_TE)
#include "../host/h3te.cpp"
#elif defined(TEST_VAE)
#include "../host/h3vae.cpp"
#else
#include "../host/h3.cpp"
#endif
#include <cassert>

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string dir = argv[1];
    std::ofstream(dir + "/manifest.txt").close();
    std::ofstream(dir + "/weights.bin", std::ios::binary).write("tiny blob", 9);
    for (bool copy_failure : {false, true}) {
        fake_copy_failure = copy_failure;
        const int before = fake_allocation_count;
        bool failed = false;
        try { Session session(dir, dir, 16, 1); }
        catch (const std::runtime_error &e) {
            failed = true;
            assert(std::string(e.what()).find(copy_failure ? "hipMemcpyHtoD" : "missing tensor") != std::string::npos);
        }
        assert(failed && fake_allocation_count == before + 1);
        assert(fake_allocations.empty());
    }
    puts("PASS failed session creation releases weights, including failed upload");
}
