#include "devproc2/runtime/object_ref.h"

namespace devproc2 {

ObjectRef& ObjectRef::operator=(const ObjectRef& other) {
    if (this != &other) {
        Object* old = ptr_;
        ptr_ = other.ptr_;
        if (ptr_) ptr_->IncRef();
        if (old)  old->DecRef();
    }
    return *this;
}

ObjectRef& ObjectRef::operator=(ObjectRef&& other) noexcept {
    if (this != &other) {
        Object* old = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
        if (old) old->DecRef();
    }
    return *this;
}

}  // namespace devproc2
