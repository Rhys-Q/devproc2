#pragma once

#include <dlpack/dlpack.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace devproc2 {

using Device = DLDevice;

struct DeviceHash {
    size_t operator()(const DLDevice& d) const noexcept {
        return std::hash<int>()(d.device_type) ^ (std::hash<int>()(d.device_id) << 16);
    }
};

struct DeviceEqual {
    bool operator()(const DLDevice& a, const DLDevice& b) const noexcept {
        return a.device_type == b.device_type && a.device_id == b.device_id;
    }
};

class DeviceAPI {
public:
    virtual ~DeviceAPI() = default;

    virtual void* Alloc(Device dev, size_t nbytes, size_t alignment) = 0;
    virtual void  Free(Device dev, void* ptr) = 0;

    virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, void* stream) = 0;

    virtual void StreamSync(Device dev, void* stream) = 0;
    virtual void DeviceSync(Device dev) = 0;
    virtual void* CreateStream(Device dev) = 0;
    virtual void  FreeStream(Device dev, void* stream) = 0;
    virtual void  SetDevice(Device dev) = 0;
};

class DeviceAPIRegistry {
public:
    static DeviceAPI* Get(int device_type);
    static void Register(int device_type, DeviceAPI* api);

private:
    static std::unordered_map<int, DeviceAPI*>& Registry();
};

#ifdef DEVPROC2_WITH_CUDA
// Defined in cuda_device_api.cc; call once to register CUDADeviceAPI.
void RegisterCUDADeviceAPI();
#endif

}  // namespace devproc2
