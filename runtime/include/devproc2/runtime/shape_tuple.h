#pragma once

#include <cstdint>
#include <vector>
#include "object.h"
#include "object_ref.h"

namespace devproc2 {

class ShapeTupleObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.ShapeTuple";
    const char* type_key() const override { return _type_key; }

    std::vector<int64_t> dims;

    int64_t ndim() const { return static_cast<int64_t>(dims.size()); }
    int64_t operator[](int i) const { return dims[i]; }
};

class ShapeTuple : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(ShapeTuple, ShapeTupleObj)

    static ShapeTuple Make(std::vector<int64_t> dims) {
        auto* obj = new ShapeTupleObj();
        obj->dims = std::move(dims);
        return ShapeTuple(obj);
    }
};

}  // namespace devproc2
