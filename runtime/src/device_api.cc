#include "devproc2/runtime/device_api.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace devproc2 {

// ── DeviceAPIRegistry ─────────────────────────────────────────────────────────

std::unordered_map<int, DeviceAPI*>& DeviceAPIRegistry::Registry() {
    static std::unordered_map<int, DeviceAPI*> reg;
    return reg;
}

DeviceAPI* DeviceAPIRegistry::Get(int device_type) {
    auto& m = Registry();
    auto it = m.find(device_type);
    if (it == m.end()) {
        throw std::runtime_error(
            "DeviceAPIRegistry: no DeviceAPI registered for device_type=" +
            std::to_string(device_type));
    }
    return it->second;
}

void DeviceAPIRegistry::Register(int device_type, DeviceAPI* api) {
    Registry()[device_type] = api;
}

// ── CPUDeviceAPI ──────────────────────────────────────────────────────────────

class CPUDeviceAPI : public DeviceAPI {
public:
    void* Alloc(Device /*dev*/, size_t nbytes, size_t alignment) override {
        // aligned_alloc requires size to be a multiple of alignment
        size_t aligned_size = (nbytes + alignment - 1) / alignment * alignment;
        void* ptr = std::aligned_alloc(alignment, aligned_size);
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    void Free(Device /*dev*/, void* ptr) override {
        std::free(ptr);
    }

    void CopyDataFromTo(DLTensor* from, DLTensor* to, void* /*stream*/) override {
        size_t nbytes = 1;
        for (int i = 0; i < from->ndim; ++i) nbytes *= static_cast<size_t>(from->shape[i]);
        nbytes = (nbytes * from->dtype.bits * from->dtype.lanes + 7) / 8;
        std::memcpy(
            static_cast<char*>(to->data)         + to->byte_offset,
            static_cast<const char*>(from->data) + from->byte_offset,
            nbytes);
    }

    void StreamSync(Device, void*)  override {}
    void DeviceSync(Device)         override {}
    void* CreateStream(Device)      override { return nullptr; }
    void  FreeStream(Device, void*) override {}
    void  SetDevice(Device)         override {}
};

// ── Auto-registration ─────────────────────────────────────────────────────────

namespace {
CPUDeviceAPI g_cpu_device_api;

struct CPUDeviceAPIRegistrar {
    CPUDeviceAPIRegistrar() {
        DeviceAPIRegistry::Register(kDLCPU, &g_cpu_device_api);
#ifdef DEVPROC2_WITH_CUDA
        // Register CUDA backend; call is in device_api.cc (always linked)
        // so the linker includes cuda_device_api.cc's TU.
        RegisterCUDADeviceAPI();
#endif
    }
} g_cpu_registrar;
}  // namespace

}  // namespace devproc2
