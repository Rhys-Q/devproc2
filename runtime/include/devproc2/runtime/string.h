#pragma once

#include <string>
#include "object.h"
#include "object_ref.h"

namespace devproc2 {

class StringObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.String";
    const char* type_key() const override { return _type_key; }

    std::string data;
};

class String : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(String, StringObj)

    static String Make(std::string s) {
        auto* obj = new StringObj();
        obj->data = std::move(s);
        return String(obj);
    }
};

}  // namespace devproc2
