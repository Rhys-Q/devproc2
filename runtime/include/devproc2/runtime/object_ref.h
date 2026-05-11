#pragma once

#include <string>
#include "object.h"

namespace devproc2 {

class ObjectRef {
public:
    ObjectRef() = default;

    explicit ObjectRef(Object* ptr) : ptr_(ptr) {
        if (ptr_) ptr_->IncRef();
    }

    ObjectRef(const ObjectRef& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->IncRef();
    }

    ObjectRef(ObjectRef&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    ~ObjectRef() {
        if (ptr_) ptr_->DecRef();
    }

    ObjectRef& operator=(const ObjectRef& other);
    ObjectRef& operator=(ObjectRef&& other) noexcept;

    Object* get() const { return ptr_; }
    bool defined() const { return ptr_ != nullptr; }

    template <typename T>
    T* as() const {
        if (ptr_ && std::string(ptr_->type_key()) == T::_type_key) {
            return static_cast<T*>(ptr_);
        }
        return nullptr;
    }

protected:
    Object* ptr_{nullptr};
};

// Generates the boilerplate for a typed ref class (mirrors TVM_DEFINE_OBJECT_REF_METHODS).
// RefName  : the ref class being defined (e.g. Tensor)
// ObjName  : the corresponding Object subclass (e.g. TensorObj)
//
// Provides:
//   - default constructor
//   - explicit constructor from Object*
//   - copy-construct / copy-assign from ObjectRef (upcast-safe)
//   - operator->()  returning ObjName*  (both const and non-const)
//   - ContainerType typedef
#define DEVPROC2_DEFINE_OBJECT_REF_METHODS(RefName, ObjName)                   \
    RefName() = default;                                                        \
    explicit RefName(Object* p) : ::devproc2::ObjectRef(p) {}                  \
    RefName(const ::devproc2::ObjectRef& ref)  /* NOLINT */                    \
        : ::devproc2::ObjectRef(ref) {}                                         \
    RefName(::devproc2::ObjectRef&& ref) noexcept                              \
        : ::devproc2::ObjectRef(std::move(ref)) {}                              \
    const ObjName* operator->() const {                                        \
        return static_cast<const ObjName*>(ptr_);                              \
    }                                                                           \
    ObjName* operator->() {                                                     \
        return static_cast<ObjName*>(ptr_);                                    \
    }                                                                           \
    using ContainerType = ObjName;

}  // namespace devproc2
