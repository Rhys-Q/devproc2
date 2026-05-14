#ifdef DEVPROC2_WITH_CUDA

#include <cuda.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>

namespace devproc2 {

#define CU_CHECK(expr)                                                           \
    do {                                                                         \
        CUresult _r = (expr);                                                    \
        if (_r != CUDA_SUCCESS) {                                                \
            const char* _msg = nullptr;                                          \
            cuGetErrorString(_r, &_msg);                                         \
            throw std::runtime_error(                                            \
                std::string("CUDA driver error in " #expr ": ") +               \
                (_msg ? _msg : "unknown"));                                      \
        }                                                                        \
    } while (0)

// CUDAModule wraps a loaded cubin (or PTX) via the CUDA Driver API.
// Used internally by M11/X3 kernel launch infrastructure.
class CUDAModule {
public:
    explicit CUDAModule(CUmodule mod) : module_(mod) {}

    ~CUDAModule() {
        if (module_) {
            cuModuleUnload(module_);
            module_ = nullptr;
        }
    }

    // Non-copyable
    CUDAModule(const CUDAModule&)            = delete;
    CUDAModule& operator=(const CUDAModule&) = delete;

    CUDAModule(CUDAModule&& o) noexcept : module_(o.module_) {
        o.module_ = nullptr;
    }

    static CUDAModule FromData(const void* image) {
        CUmodule mod;
        CU_CHECK(cuModuleLoadData(&mod, image));
        return CUDAModule(mod);
    }

    CUfunction GetFunction(const std::string& name) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = func_cache_.find(name);
            if (it != func_cache_.end()) return it->second;
        }
        CUfunction fn;
        CU_CHECK(cuModuleGetFunction(&fn, module_, name.c_str()));
        std::lock_guard<std::mutex> lock(mu_);
        func_cache_[name] = fn;
        return fn;
    }

private:
    CUmodule module_{nullptr};
    std::mutex mu_;
    std::unordered_map<std::string, CUfunction> func_cache_;
};

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
