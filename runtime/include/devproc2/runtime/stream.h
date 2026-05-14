#pragma once

#include "device_api.h"
#include "object.h"
#include "object_ref.h"

namespace devproc2 {

class StreamObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Stream";
    const char* type_key() const override { return _type_key; }

    ~StreamObj();  // calls DeviceAPI::FreeStream

    Device device{};
    void*  handle{nullptr};
};

class Stream : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(Stream, StreamObj)
};

}  // namespace devproc2
