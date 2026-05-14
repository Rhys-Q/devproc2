#pragma once

#ifdef DEVPROC2_WITH_CUDA

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernel.h"

namespace devproc2 {

// Thread-safe global registry mapping kernel name → KernelObj.
// Kernels are registered once at artifact load time (Executable::Load),
// then looked up at dispatch time in VMState::DispatchExternal.
class CUDAKernelRegistry {
public:
    static CUDAKernelRegistry& Global() {
        static CUDAKernelRegistry instance;
        return instance;
    }

    // Register a kernel. Overwrites any existing registration with the same name.
    void Register(
        const std::string&          name,
        const std::vector<uint8_t>& cubin_data,
        const std::string&          func_name,
        std::array<int32_t, 3>      block_dims = {128, 1, 1}
    );

    // Returns nullptr if not found.
    KernelObj* Get(const std::string& name) const;

    bool Has(const std::string& name) const;

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, std::unique_ptr<KernelObj>> kernels_;
};

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
