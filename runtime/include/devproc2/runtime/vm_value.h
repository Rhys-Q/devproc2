#pragma once

#include <cstdint>
#include "object_ref.h"

namespace devproc2 {

class VMValue {
public:
    enum class Tag { kNull, kInt, kFloat, kBool, kObjectRef };

    VMValue() : tag_(Tag::kNull) {}

    static VMValue Int(int64_t v) {
        VMValue r;
        r.tag_ = Tag::kInt;
        r.data_.i = v;
        return r;
    }

    static VMValue Float(double v) {
        VMValue r;
        r.tag_ = Tag::kFloat;
        r.data_.f = v;
        return r;
    }

    static VMValue Bool(bool v) {
        VMValue r;
        r.tag_ = Tag::kBool;
        r.data_.b = v;
        return r;
    }

    static VMValue ObjRef(ObjectRef o) {
        VMValue r;
        r.tag_ = Tag::kObjectRef;
        r.obj_ = std::move(o);
        return r;
    }

    Tag tag() const { return tag_; }
    bool IsNull()      const { return tag_ == Tag::kNull; }
    bool IsInt()       const { return tag_ == Tag::kInt; }
    bool IsFloat()     const { return tag_ == Tag::kFloat; }
    bool IsBool()      const { return tag_ == Tag::kBool; }
    bool IsObjectRef() const { return tag_ == Tag::kObjectRef; }

    int64_t AsInt() const {
        DEVPROC2_DCHECK(IsInt());
        return data_.i;
    }

    double AsFloat() const {
        DEVPROC2_DCHECK(IsFloat());
        return data_.f;
    }

    bool AsBool() const {
        DEVPROC2_DCHECK(IsBool());
        return data_.b;
    }

    ObjectRef AsObjectRef() const {
        DEVPROC2_DCHECK(IsObjectRef());
        return obj_;
    }

    template <typename T>
    T* AsObjectAs() const {
        return obj_.as<T>();
    }

private:
    Tag tag_{Tag::kNull};
    union {
        int64_t i;
        double  f;
        bool    b;
    } data_{};
    ObjectRef obj_;
};

}  // namespace devproc2
