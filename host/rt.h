// The device runtime behind libh3pipe: allocation, copies, kernel load and dispatch. Two implementations,
// chosen at build time: rt_hip.cpp (the HIP runtime API) and rt_hrx.cpp (hrx-system's libhrx: IREE's AMDGPU
// HAL over the HSA runtime, no HIP). Kernels are Loom-compiled HSACOs whose kernarg layout is the launch
// signature: each `index` argument an 8-byte by-value slot, then one 8-byte global pointer per buffer.
#ifndef H3_RT_H
#define H3_RT_H
#include <cstddef>
#include <cstdint>
#include <string>
struct RtKernel;
struct KernArgs {
    uint64_t scalars[8]; int nscalars = 0; const void *ptrs[16]; int nptrs = 0;
    KernArgs &i32(int v) { scalars[nscalars++] = uint64_t(uint32_t(v)); return *this; }
    KernArgs &ptr(const void *p) { ptrs[nptrs++] = p; return *this; }
};
struct Rt {
    virtual ~Rt() {}
    virtual const char *name() const = 0;
    virtual void *alloc(size_t bytes) = 0;
    virtual void free(void *p) = 0;
    virtual void memset(void *p, int value, size_t bytes) = 0;
    virtual void h2d(void *dst, const void *src, size_t bytes) = 0;
    virtual void d2h(void *dst, const void *src, size_t bytes) = 0;
    virtual void d2d(void *dst, const void *src, size_t bytes) = 0;
    virtual void sync() = 0;
    virtual RtKernel *load(const std::string &path, const std::string &symbol) = 0;
    virtual void unload(RtKernel *k) = 0;
    virtual void launch(RtKernel *k, unsigned gx, unsigned gy, unsigned bx, const KernArgs &a) = 0;
};
Rt &rt();   // the process's runtime, created on first use
#endif
