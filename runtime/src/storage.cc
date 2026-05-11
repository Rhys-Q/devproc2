#include "devproc2/runtime/storage.h"
#include "devproc2/runtime/device_api.h"

namespace devproc2 {

StorageObj::~StorageObj() {
    if (owns_data && data) {
        DeviceAPIRegistry::Get(device.device_type)->Free(device, data);
    } else if (!owns_data && deleter) {
        deleter();
    }
}

}  // namespace devproc2
