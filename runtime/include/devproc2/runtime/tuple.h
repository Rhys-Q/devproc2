#pragma once

#include <vector>
#include "object.h"
#include "object_ref.h"

namespace devproc2 {

class TupleObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Tuple";
    const char* type_key() const override { return _type_key; }

    std::vector<ObjectRef> fields;

    int size() const { return static_cast<int>(fields.size()); }
    const ObjectRef& operator[](int i) const { return fields[i]; }
};

class Tuple : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(Tuple, TupleObj)

    static Tuple Make(std::vector<ObjectRef> fields) {
        auto* obj = new TupleObj();
        obj->fields = std::move(fields);
        return Tuple(obj);
    }
};

}  // namespace devproc2
