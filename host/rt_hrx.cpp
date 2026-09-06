// rt.h over hrx-system's libhrx (IREE's AMDGPU HAL over the HSA runtime; no HIP anywhere in the process).
// Device pointers stay the currency of the pipeline: every allocation is remembered so a pointer into it
// becomes the (buffer, offset, length) binding a dispatch needs.
#include "rt.h"
#include "hrx_runtime.h"
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
struct RtKernel { hrx_executable_t exe = nullptr; uint32_t ordinal = 0; hrx_executable_export_info_t info{}; std::string symbol; };
namespace {
void check(hrx_status_t s, const char *what) {
    if (hrx_status_is_ok(s)) return;
    char *msg = nullptr; size_t len = 0; hrx_status_to_string(s, &msg, &len);
    std::string text = std::string(what) + ": " + (msg ? msg : "?"); hrx_status_free_message(msg); hrx_status_ignore(s);
    throw std::runtime_error(text);
}
struct HrxRt : Rt {
    hrx_device_t dev = nullptr; hrx_stream_t stream = nullptr;
    std::map<uintptr_t, std::pair<hrx_buffer_t, size_t>> allocs;   // synthetic base -> (buffer, bytes)
    uintptr_t next_base = uintptr_t(1) << 44;
    HrxRt() {
        check(hrx_gpu_initialize(0), "hrx_gpu_initialize");
        int n = 0; check(hrx_gpu_device_count(&n), "hrx_gpu_device_count");
        if (n < 1) throw std::runtime_error("libhrx sees no GPU (is the HSA runtime it needs on LD_LIBRARY_PATH?)");
        check(hrx_gpu_device_get(0, &dev), "hrx_gpu_device_get");
        check(hrx_stream_create(dev, 0, &stream), "hrx_stream_create");
    }
    const char *name() const override { return "hrx"; }
    struct Ref { hrx_buffer_t buf; size_t offset, length; };
    Ref ref(const void *p) const {
        auto it = allocs.upper_bound(uintptr_t(p)); if (it == allocs.begin()) throw std::runtime_error("device pointer outside every allocation");
        --it; const uintptr_t off = uintptr_t(p) - it->first;
        if (off > it->second.second) throw std::runtime_error("device pointer outside every allocation");
        return {it->second.first, size_t(off), it->second.second - off};
    }
    void *alloc(size_t bytes) override {
        hrx_buffer_t buf = nullptr; check(hrx_buffer_allocate(stream, bytes ? bytes : 1, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buf), "hrx_buffer_allocate");
        // device-local buffers expose no device pointer, and nothing on the host dereferences one: hand out a synthetic
        // address per allocation (4 KB aligned, never reused) that ref() maps back to (buffer, offset)
        const size_t size = bytes ? bytes : 1; void *p = reinterpret_cast<void *>(next_base); next_base += (size + 4095) & ~size_t(4095);
        try { allocs[uintptr_t(p)] = {buf, size}; } catch (...) { hrx_buffer_release(buf); throw; }
        return p;
    }
    void free(void *p) override {
        auto it = allocs.find(uintptr_t(p)); if (it == allocs.end()) return;
        (void)hrx_stream_synchronize(stream); hrx_buffer_release(it->second.first); allocs.erase(it);
    }
    void memset(void *p, int value, size_t bytes) override {
        if (!bytes) return; const Ref r = ref(p); const uint8_t pattern = uint8_t(value);
        check(hrx_stream_fill_buffer(stream, r.buf, r.offset, bytes, &pattern, 1), "hrx_stream_fill_buffer");
    }
    void h2d(void *dst, const void *src, size_t bytes) override {
        if (!bytes) return; const Ref r = ref(dst); sync();
        check(hrx_synchronous_h2d(dev, src, r.buf, r.offset, bytes), "hrx_synchronous_h2d");
    }
    void d2h(void *dst, const void *src, size_t bytes) override {
        if (!bytes) return; const Ref r = ref(src); sync();
        check(hrx_synchronous_d2h(dev, r.buf, r.offset, dst, bytes), "hrx_synchronous_d2h");
    }
    void d2d(void *dst, const void *src, size_t bytes) override {
        if (!bytes) return; const Ref s = ref(src), d = ref(dst);
        check(hrx_stream_copy_buffer(stream, s.buf, s.offset, d.buf, d.offset, bytes), "hrx_stream_copy_buffer");
    }
    void sync() override { check(hrx_stream_synchronize(stream), "hrx_stream_synchronize"); }
    RtKernel *load(const std::string &path, const std::string &symbol) override {
        RtKernel *k = new RtKernel;
        try {
            k->symbol = symbol;
            check(hrx_executable_load_file(dev, path.c_str(), "amdgpu", "gfx1151", &k->exe), ("hrx_executable_load_file " + path).c_str());
            check(hrx_executable_lookup_export_by_name(k->exe, symbol.c_str(), &k->ordinal), ("export " + symbol).c_str());
            check(hrx_executable_export_info(k->exe, k->ordinal, &k->info), "hrx_executable_export_info");
            return k;
        } catch (...) { unload(k); throw; }
    }
    void unload(RtKernel *k) override { if (k) { if (k->exe) hrx_executable_release(k->exe); delete k; } }
    void launch(RtKernel *k, unsigned gx, unsigned gy, unsigned bx, const KernArgs &a) override {
        // the constants block is the by-value arguments packed as the export declares them: 4-byte i32 or 8-byte index slots
        const unsigned width = a.nscalars ? k->info.constant_byte_length / unsigned(a.nscalars) : 0;
        if ((a.nscalars && (width != 4 && width != 8)) || k->info.constant_byte_length != width * unsigned(a.nscalars) || k->info.binding_count != unsigned(a.nptrs))
            throw std::runtime_error(k->symbol + ": launch passes " + std::to_string(a.nscalars) + " scalars and " + std::to_string(a.nptrs) + " buffers; the export wants " +
                                     std::to_string(k->info.constant_byte_length) + " constant bytes and " + std::to_string(k->info.binding_count) + " bindings");
        unsigned char constants[64]; for (int i = 0; i < a.nscalars; ++i) memcpy(constants + width * i, &a.scalars[i], width);   // little-endian: the low bytes are the i32
        hrx_buffer_ref_t refs[16];
        for (int i = 0; i < a.nptrs; ++i) { const Ref r = ref(a.ptrs[i]); refs[i] = {r.buf, r.offset, r.length}; }
        const hrx_dispatch_config_t cfg = {{gx, gy, 1u}, {bx, 1u, 1u}, 32u};
        check(hrx_stream_dispatch(stream, k->exe, k->ordinal, &cfg, constants, k->info.constant_byte_length, refs, size_t(a.nptrs), 0), (k->symbol + " dispatch").c_str());
    }
};
}
Rt &rt() { static HrxRt r; return r; }
