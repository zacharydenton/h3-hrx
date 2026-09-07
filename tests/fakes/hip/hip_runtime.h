// Host-session error tests use this allocator instead of creating a GPU context.
#pragma once
#include <cstdlib>
#include <cstring>
#include <unordered_set>

using hipError_t = int;
using hipDeviceptr_t = void *;
using hipModule_t = void *;
using hipFunction_t = void *;
using hipEvent_t = void *;
constexpr hipError_t hipSuccess = 0;
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)1)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void *)2)
#define HIP_LAUNCH_PARAM_END ((void *)3)
inline std::unordered_set<void *> fake_allocations;
inline bool fake_copy_failure = false;
inline int fake_allocation_count = 0;
inline bool fake_module_load_failure = true, fake_symbol_failure = true;
inline std::unordered_set<void *> fake_modules;
inline int fake_init_count = 0;
inline hipError_t (*fake_launch_hook)(void **) = nullptr;
inline const char *hipGetErrorString(hipError_t) { return "injected HIP failure"; }
inline hipError_t hipInit(unsigned) { ++fake_init_count; return hipSuccess; }
inline hipError_t hipMalloc(void **p, size_t n) {
    if (n > 1024 * 1024) return 1;       // tests cannot accidentally allocate a model
    *p = std::malloc(n ? n : 1);
    if (!*p) return 1;
    fake_allocations.insert(*p); ++fake_allocation_count;
    return hipSuccess;
}
inline hipError_t hipFree(void *p) {
    if (!fake_allocations.erase(p)) std::abort();  // catches double free too
    std::free(p); return hipSuccess;
}
inline hipError_t hipMemcpyHtoD(void *dst, const void *src, size_t n) {
    if (fake_copy_failure) return 1;
    std::memcpy(dst, src, n); return hipSuccess;
}
inline hipError_t hipMemcpyDtoH(void *dst, void *src, size_t n) { std::memcpy(dst, src, n); return hipSuccess; }
inline hipError_t hipMemcpyDtoD(void *dst, void *src, size_t n) { std::memmove(dst, src, n); return hipSuccess; }
inline hipError_t hipMemset(void *p, int v, size_t n) { std::memset(p, v, n); return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipModuleLoad(hipModule_t *out, const char *) {
    if (fake_module_load_failure) return 1;
    *out = std::malloc(1); fake_modules.insert(*out); return hipSuccess;
}
inline hipError_t hipModuleGetFunction(hipFunction_t *out, hipModule_t module, const char *) {
    if (fake_symbol_failure) return 1;
    *out = module; return hipSuccess;
}
inline hipError_t hipModuleUnload(hipModule_t module) {
    if (!fake_modules.erase(module)) std::abort();
    std::free(module); return hipSuccess;
}
inline hipError_t hipModuleLaunchKernel(hipFunction_t, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned, unsigned,
                                       void *, void **, void **extra) {
    return fake_launch_hook ? fake_launch_hook(extra) : 1;
}
inline hipError_t hipEventCreate(hipEvent_t *event) { *event = std::malloc(1); return *event ? hipSuccess : 1; }
inline hipError_t hipEventDestroy(hipEvent_t event) { std::free(event); return hipSuccess; }
inline hipError_t hipEventRecord(hipEvent_t, void *) { return hipSuccess; }
inline hipError_t hipEventElapsedTime(float *ms, hipEvent_t, hipEvent_t) { *ms = 1.0f; return hipSuccess; }
