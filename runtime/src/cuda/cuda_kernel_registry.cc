#ifdef DEVPROC2_WITH_CUDA

#include "devproc2/runtime/cuda_kernel_registry.h"

namespace devproc2 {

void CUDAKernelRegistry::Register(
    const std::string&          name,
    const std::vector<uint8_t>& cubin_data,
    const std::string&          func_name,
    std::array<int32_t, 3>      grid_dims,
    std::array<int32_t, 3>      block_dims,
    int32_t                     shared_memory_bytes
) {
    std::lock_guard<std::mutex> lock(mu_);
    auto obj = std::make_unique<KernelObj>();
    obj->name       = name;
    obj->func_name  = func_name;
    obj->cubin_data = cubin_data;
    obj->grid_dims  = grid_dims;
    obj->block_dims = block_dims;
    obj->shared_memory_bytes = shared_memory_bytes;
    kernels_[name]  = std::move(obj);
}

KernelObj* CUDAKernelRegistry::Get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = kernels_.find(name);
    return (it != kernels_.end()) ? it->second.get() : nullptr;
}

bool CUDAKernelRegistry::Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    return kernels_.count(name) > 0;
}

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
