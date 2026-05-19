#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
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

// ── PackedFuncRegistry ────────────────────────────────────────────────────────

class PackedFuncRegistry {
public:
    static PackedFuncRegistry& Global();

    void Register(const std::string& name, PackedFunc func);
    void RegisterWithDevice(const std::string& name, PackedFunc func, std::string device);
    void SetDevice(const std::string& name, std::string device);
    PackedFunc Get(const std::string& name) const;
    bool Has(const std::string& name) const;
    std::string Device(const std::string& name) const;

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, PackedFunc> registry_;
    std::unordered_map<std::string, std::string> devices_;
};

// Helper for static-initializer registration
struct PackedFuncRegistrar {
    explicit PackedFuncRegistrar(const char* name) : name_(name) {}

    PackedFuncRegistrar& set_body(std::function<void(PackedArgs)> f) {
        auto* obj = new PackedFuncObj();
        obj->body = std::move(f);
        PackedFuncRegistry::Global().Register(name_, PackedFunc(obj));
        return *this;
    }

    std::string name_;
};

#define DEVPROC2_REGISTER_PACKED_FUNC(name)                                    \
    static ::devproc2::PackedFuncRegistrar _pfreg_##__LINE__(name)

}  // namespace devproc2
