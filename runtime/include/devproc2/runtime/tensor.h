#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include <dlpack/dlpack.h>
#include "object.h"
#include "object_ref.h"
#include "storage.h"

namespace devproc2 {

class TensorObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Tensor";
    const char* type_key() const override { return _type_key; }

    ~TensorObj();

    // DLTensor MUST be first field: TensorObj* == DLTensor* cast works
    DLTensor dl_tensor{};

    Storage storage;
    void*   manager_ctx{nullptr};

    DLTensor*       dl()       { return &dl_tensor; }
    const DLTensor* dl() const { return &dl_tensor; }

    void*       data()    const { return dl_tensor.data; }
    DLDevice    device()  const { return dl_tensor.device; }
    int         ndim()    const { return dl_tensor.ndim; }
    DLDataType  dtype()   const { return dl_tensor.dtype; }
    int64_t*    shape()   const { return dl_tensor.shape; }
    int64_t*    strides() const { return dl_tensor.strides; }

    DLManagedTensor* ToDLPack() const;

private:
    std::vector<int64_t> shape_storage_;
    std::vector<int64_t> strides_storage_;

    friend class Tensor;
};

class Tensor : public ObjectRef {
public:
    DEVPROC2_DEFINE_OBJECT_REF_METHODS(Tensor, TensorObj)

    static Tensor Empty(const std::vector<int64_t>& shape,
                        DLDataType dtype, DLDevice device);

    static Tensor FromDLPack(DLManagedTensor* managed);

    static Tensor FromExternalBuffer(
        void* data, DLDevice device,
        const std::vector<int64_t>& shape, DLDataType dtype,
        std::vector<int64_t> strides = {},
        std::function<void()> deleter = nullptr);
};

}  // namespace devproc2
