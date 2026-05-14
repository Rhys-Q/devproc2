#ifdef DEVPROC2_WITH_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "devproc2/runtime/device_api.h"

namespace devproc2 {

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t _e = (expr);                                                 \
        if (_e != cudaSuccess) {                                                 \
            throw std::runtime_error(                                            \
                std::string("CUDA error in " #expr ": ") +                      \
                cudaGetErrorString(_e));                                         \
        }                                                                        \
    } while (0)

static cudaMemcpyKind GetCopyKind(const DLTensor* from, const DLTensor* to) {
    const int ft = from->device.device_type;
    const int tt = to->device.device_type;
    if (ft == kDLCPU  && tt == kDLCPU)  return cudaMemcpyHostToHost;
    if (ft == kDLCPU  && tt == kDLCUDA) return cudaMemcpyHostToDevice;
    if (ft == kDLCUDA && tt == kDLCPU)  return cudaMemcpyDeviceToHost;
    if (ft == kDLCUDA && tt == kDLCUDA) return cudaMemcpyDeviceToDevice;
    // kDLCUDAHost, kDLCUDAManaged, or other types: let CUDA determine direction.
    return cudaMemcpyDefault;
}

static size_t BytesOf(const DLTensor* t) {
    size_t n = 1;
    for (int i = 0; i < t->ndim; ++i) n *= static_cast<size_t>(t->shape[i]);
    return (n * t->dtype.bits * t->dtype.lanes + 7) / 8;
}

class CUDADeviceAPI : public DeviceAPI {
public:
    void* Alloc(Device dev, size_t nbytes, size_t alignment) override {
        // cudaMalloc guarantees 256-byte alignment; larger values are unsupported.
        if (alignment > 256) {
            throw std::runtime_error(
                "CUDADeviceAPI::Alloc: alignment " + std::to_string(alignment) +
                " exceeds cudaMalloc guarantee of 256 bytes");
        }
        CUDA_CHECK(cudaSetDevice(dev.device_id));
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, nbytes));
        return ptr;
    }

    void Free(Device /*dev*/, void* ptr) override {
        // cudaFree is safe to call without setting the device
        CUDA_CHECK(cudaFree(ptr));
    }

    void CopyDataFromTo(DLTensor* from, DLTensor* to, void* stream) override {
        size_t nbytes = BytesOf(from);
        cudaMemcpyKind kind = GetCopyKind(from, to);
        auto s = static_cast<cudaStream_t>(stream);
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(to->data)         + to->byte_offset,
            static_cast<const char*>(from->data) + from->byte_offset,
            nbytes, kind, s));
    }

    void StreamSync(Device /*dev*/, void* stream) override {
        CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    }

    void DeviceSync(Device dev) override {
        CUDA_CHECK(cudaSetDevice(dev.device_id));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void* CreateStream(Device dev) override {
        CUDA_CHECK(cudaSetDevice(dev.device_id));
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        return static_cast<void*>(s);
    }

    void FreeStream(Device /*dev*/, void* stream) override {
        CUDA_CHECK(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
    }

    void SetDevice(Device dev) override {
        CUDA_CHECK(cudaSetDevice(dev.device_id));
    }
};

// ── Registration ──────────────────────────────────────────────────────────────

namespace {
CUDADeviceAPI g_cuda_device_api;
}  // namespace

// Called from device_api.cc's CPUDeviceAPIRegistrar (which is in a TU that
// is always linked) to force this translation unit to be included.
void RegisterCUDADeviceAPI() {
    DeviceAPIRegistry::Register(kDLCUDA, &g_cuda_device_api);
}

}  // namespace devproc2

#endif  // DEVPROC2_WITH_CUDA
