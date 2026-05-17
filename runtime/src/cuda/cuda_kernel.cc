#ifdef DEVPROC2_WITH_CUDA

#include <cuda.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "devproc2/runtime/cuda_kernel_registry.h"
#include "devproc2/runtime/kernel.h"
#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/vm.h"
#include "devproc2/runtime/vm_value.h"

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

namespace {

// Module cache: kernel_name → CUmodule + function cache.
// Kernels are loaded once and cached for the lifetime of the process.
struct ModuleEntry {
    CUmodule module{nullptr};
    std::unordered_map<std::string, CUfunction> func_cache;
    std::mutex mu;
};

std::mutex g_module_cache_mu;
std::unordered_map<std::string, std::unique_ptr<ModuleEntry>> g_module_cache;

CUfunction get_or_load_function(const KernelObj* kernel) {
    std::unique_lock<std::mutex> cache_lock(g_module_cache_mu);
    auto& entry = g_module_cache[kernel->name];
    if (!entry) {
        entry = std::make_unique<ModuleEntry>();
        CU_CHECK(cuModuleLoadData(&entry->module, kernel->cubin_data.data()));
    }
    cache_lock.unlock();

    std::lock_guard<std::mutex> fn_lock(entry->mu);
    auto it = entry->func_cache.find(kernel->func_name);
    if (it != entry->func_cache.end()) return it->second;
    CUfunction fn;
    CU_CHECK(cuModuleGetFunction(&fn, entry->module, kernel->func_name.c_str()));
    entry->func_cache[kernel->func_name] = fn;
    return fn;
}

}  // namespace

// CUDAKernelLauncher::Launch
//
// Args convention:
//   args:         explicit kernel ABI params only.
//   launch_args:  optional grid_x/y/z, block_x/y/z, shared_memory_bytes.
//
// Tensor args are passed as void* pointers to the kernel (device data ptrs).
// Int args are passed as int64_t by value.
void CUDAKernelLauncher_Launch(
    const KernelObj*       kernel,
    std::vector<VMValue>&  args,
    const std::vector<int64_t>& launch_args,
    void*                  stream
) {
    uint32_t grid_x = static_cast<uint32_t>(kernel->grid_dims[0]);
    uint32_t grid_y = static_cast<uint32_t>(kernel->grid_dims[1]);
    uint32_t grid_z = static_cast<uint32_t>(kernel->grid_dims[2]);
    uint32_t block_x = static_cast<uint32_t>(kernel->block_dims[0]);
    uint32_t block_y = static_cast<uint32_t>(kernel->block_dims[1]);
    uint32_t block_z = static_cast<uint32_t>(kernel->block_dims[2]);
    uint32_t shared_memory_bytes =
        static_cast<uint32_t>(kernel->shared_memory_bytes);

    if (!launch_args.empty()) {
        if (launch_args.size() != 7) {
            throw std::runtime_error(
                "CUDAKernelLauncher: launch metadata must have 7 values "
                "(grid3, block3, shared_memory_bytes)");
        }
        grid_x = static_cast<uint32_t>(launch_args[0]);
        grid_y = static_cast<uint32_t>(launch_args[1]);
        grid_z = static_cast<uint32_t>(launch_args[2]);
        block_x = static_cast<uint32_t>(launch_args[3]);
        block_y = static_cast<uint32_t>(launch_args[4]);
        block_z = static_cast<uint32_t>(launch_args[5]);
        shared_memory_bytes = static_cast<uint32_t>(launch_args[6]);
    }

    // Collect kernel argument pointers.
    // Tensor args → data pointer; Int args → int64_t value.
    std::vector<void*> raw_args;
    std::vector<void*> ptr_storage;    // stable storage for tensor data pointers
    std::vector<int64_t> int_storage;  // stable storage for int args
    ptr_storage.reserve(args.size());
    int_storage.reserve(args.size());
    raw_args.reserve(args.size());

    for (int i = 0; i < static_cast<int>(args.size()); ++i) {
        const VMValue& v = args[i];
        if (v.IsObjectRef()) {
            auto* tobj = v.AsObjectAs<TensorObj>();
            if (!tobj) {
                throw std::runtime_error(
                    "CUDAKernelLauncher: expected TensorObj arg at index " +
                    std::to_string(i));
            }
            ptr_storage.push_back(tobj->data());
            raw_args.push_back(&ptr_storage.back());
        } else if (v.IsInt()) {
            int_storage.push_back(v.AsInt());
            raw_args.push_back(&int_storage.back());
        } else {
            throw std::runtime_error(
                "CUDAKernelLauncher: unsupported VMValue type at index " +
                std::to_string(i));
        }
    }

    // Load (or retrieve from cache) the CUfunction.
    CUfunction fn = get_or_load_function(kernel);

    CUstream cu_stream = static_cast<CUstream>(stream);

    CU_CHECK(cuLaunchKernel(
        fn,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        static_cast<unsigned int>(shared_memory_bytes),
        cu_stream,
        raw_args.empty() ? nullptr : raw_args.data(),
        nullptr      // extra
    ));
}

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
