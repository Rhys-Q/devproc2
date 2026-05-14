#include "devproc2/runtime/stream.h"
#include "devproc2/runtime/device_api.h"

namespace devproc2 {

StreamObj::~StreamObj() {
    if (handle) {
        DeviceAPI* api = DeviceAPIRegistry::Get(device.device_type);
        api->FreeStream(device, handle);
        handle = nullptr;
    }
}

}  // namespace devproc2
