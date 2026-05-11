#pragma once

#include <functional>
#include <vector>
#include "object.h"
#include "object_ref.h"
#include "vm_value.h"

namespace devproc2 {

class PackedArgs {
public:
    explicit PackedArgs(std::vector<VMValue>& args) : args_(args) {}

    int size() const { return static_cast<int>(args_.size()); }
    VMValue&       operator[](int i)       { return args_[i]; }
    const VMValue& operator[](int i) const { return args_[i]; }

private:
    std::vector<VMValue>& args_;
};

class PackedFuncObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.PackedFunc";
    const char* type_key() const override { return _type_key; }

    std::function<void(PackedArgs)> body;

    void Call(PackedArgs args) {
        if (body) body(args);
    }
};

class PackedFunc : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(PackedFunc, PackedFuncObj)

    void Call(PackedArgs args) { (*this)->Call(args); }
};

}  // namespace devproc2
