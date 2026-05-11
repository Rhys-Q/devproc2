#pragma once

#include <cstddef>
#include <functional>
#include <dlpack/dlpack.h>
#include "object.h"
#include "object_ref.h"

namespace devproc2 {

class StorageObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Storage";
    const char* type_key() const override { return _type_key; }

    ~StorageObj();

    DLDevice device{};
    void*    data{nullptr};
    size_t   nbytes{0};
    size_t   alignment{256};
    bool     owns_data{true};
    std::function<void()> deleter;
};

class Storage : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(Storage, StorageObj)
};

}  // namespace devproc2
