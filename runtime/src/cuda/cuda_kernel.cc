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
// Args convention (matches VMCodegenPass kernel output):
//   args[0..n-1]     : input tensors + output tensor (as ObjectRef VMValues)
//   args[n..n+2]     : grid_x, grid_y, grid_z (Int VMValues), when grid_fn was set
//
// Tensor args are passed as void* pointers to the kernel (device data ptrs).
// Int args are passed as int64_t by value.
//
// If grid dims are not provided in args (< block_dims + 3), falls back to (1,1,1).
void CUDAKernelLauncher_Launch(
    const KernelObj*       kernel,
    std::vector<VMValue>&  args,
    void*                  stream
) {
    // Detect whether grid dims were appended as last 3 Int args.
    // Heuristic: count how many trailing args are Int (up to 3).
    uint32_t grid_x = 1, grid_y = 1, grid_z = 1;
    int tensor_count = static_cast<int>(args.size());

    if (args.size() >= 3) {
        int tail = static_cast<int>(args.size()) - 1;
        if (args[tail].IsInt() && args[tail-1].IsInt() && args[tail-2].IsInt()) {
            grid_z = static_cast<uint32_t>(args[tail].AsInt());
            grid_y = static_cast<uint32_t>(args[tail-1].AsInt());
            grid_x = static_cast<uint32_t>(args[tail-2].AsInt());
            tensor_count -= 3;
        }
    }

    // Collect kernel argument pointers.
    // Tensor args → data pointer; Int args → int64_t value.
    std::vector<void*> raw_args;
    std::vector<int64_t> int_storage;  // stable storage for int args

    for (int i = 0; i < tensor_count; ++i) {
        const VMValue& v = args[i];
        if (v.IsObjectRef()) {
            auto* tobj = v.AsObjectAs<TensorObj>();
            if (!tobj) {
                throw std::runtime_error(
                    "CUDAKernelLauncher: expected TensorObj arg at index " +
                    std::to_string(i));
            }
            raw_args.push_back(tobj->dl().data);
        } else if (v.IsInt()) {
            int_storage.push_back(v.AsInt());
            raw_args.push_back(&int_storage.back());
        } else {
            throw std::runtime_error(
                "CUDAKernelLauncher: unsupported VMValue type at index " +
                std::to_string(i));
        }
    }

    // Build params array (array of pointers to args).
    std::vector<void*> params(raw_args.size());
    for (size_t i = 0; i < raw_args.size(); ++i) {
        params[i] = raw_args[i];
    }

    // Load (or retrieve from cache) the CUfunction.
    CUfunction fn = get_or_load_function(kernel);

    uint32_t block_x = static_cast<uint32_t>(kernel->block_dims[0]);
    uint32_t block_y = static_cast<uint32_t>(kernel->block_dims[1]);
    uint32_t block_z = static_cast<uint32_t>(kernel->block_dims[2]);

    CUstream cu_stream = static_cast<CUstream>(stream);

    CU_CHECK(cuLaunchKernel(
        fn,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        0,           // sharedMemBytes
        cu_stream,
        params.empty() ? nullptr : params.data(),
        nullptr      // extra
    ));
}

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
