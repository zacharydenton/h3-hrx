#include "../host/rt_hip.cpp"
#include <cassert>
#include <cstdio>

int main() {
    for (bool fail_load : {true, false}) {
        fake_module_load_failure = fail_load;
        try { rt().load("missing", "missing"); assert(false); }
        catch (const std::runtime_error &) {}
        assert(fake_modules.empty());
    }
    fake_symbol_failure = false;
    RtKernel *k = rt().load("valid", "valid");
    assert(fake_modules.size() == 1);
    rt().unload(k); assert(fake_modules.empty());
    puts("PASS runtime module cleanup after load/symbol failure and success");
}
