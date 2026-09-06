// rt.h over the HIP runtime API (module load + launch, no device code).
#include "rt.h"
#include <hip/hip_runtime.h>
#include <cstring>
#include <stdexcept>
#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e_)); } while (0)
struct RtKernel { hipModule_t module = nullptr; hipFunction_t function = nullptr; };
namespace {
struct HipRt : Rt {
    HipRt() { HIP_CHECK(hipInit(0)); }
    const char *name() const override { return "hip"; }
    void *alloc(size_t bytes) override { void *p = nullptr; HIP_CHECK(hipMalloc(&p, bytes)); return p; }
    void free(void *p) override { (void)hipFree(p); }
    void memset(void *p, int value, size_t bytes) override { hipError_t e = hipMemset(p, value, bytes); if (e != hipSuccess) throw std::runtime_error("hipMemset(" + std::to_string(uintptr_t(p)) + ", " + std::to_string(bytes) + "): " + hipGetErrorString(e)); }
    void h2d(void *dst, const void *src, size_t bytes) override { HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)dst, const_cast<void *>(src), bytes)); }
    void d2h(void *dst, const void *src, size_t bytes) override { HIP_CHECK(hipMemcpyDtoH(dst, (hipDeviceptr_t)src, bytes)); }
    void d2d(void *dst, const void *src, size_t bytes) override { HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)dst, (hipDeviceptr_t)src, bytes)); }
    void sync() override { HIP_CHECK(hipDeviceSynchronize()); }
    RtKernel *load(const std::string &path, const std::string &symbol) override {
        RtKernel *k = new RtKernel;
        HIP_CHECK(hipModuleLoad(&k->module, path.c_str()));
        HIP_CHECK(hipModuleGetFunction(&k->function, k->module, symbol.c_str()));
        return k;
    }
    void unload(RtKernel *k) override { if (k) { if (k->module) (void)hipModuleUnload(k->module); delete k; } }
    void launch(RtKernel *k, unsigned gx, unsigned gy, unsigned bx, const KernArgs &a) override {
        // the kernarg layout as Loom declares it: by-value scalars as contiguous 4-byte slots (an index that the
        // kernel keeps as i64 occupies 8 bytes; the high half is zero here), then the buffer pointers 8-byte aligned
        alignas(16) unsigned char bytes[8 * 24]; ::memset(bytes, 0, sizeof bytes); size_t size = 0;
        for (int i = 0; i < a.nscalars; ++i) { memcpy(bytes + size, &a.scalars[i], 4); size += 4; }
        size = (size + 7) & ~size_t(7);
        for (int i = 0; i < a.nptrs; ++i) { memcpy(bytes + size, &a.ptrs[i], 8); size += 8; }
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, bytes, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
        HIP_CHECK(hipModuleLaunchKernel(k->function, gx, gy, 1, bx, 1, 1, 0, nullptr, nullptr, config));
    }
};
}
Rt &rt() { static HipRt r; return r; }
