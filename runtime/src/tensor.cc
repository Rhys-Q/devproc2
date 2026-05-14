#include "devproc2/runtime/tensor.h"
#include "devproc2/runtime/device_api.h"
#include "devproc2/runtime/storage.h"

#include <cstdlib>
#include <stdexcept>

namespace devproc2 {

TensorObj::~TensorObj() {
    if (manager_ctx) {
        auto* managed = static_cast<DLManagedTensor*>(manager_ctx);
        if (managed->deleter) managed->deleter(managed);
        manager_ctx = nullptr;
    }
    // storage destructor handles owned memory via StorageObj
}

Tensor Tensor::Empty(const std::vector<int64_t>& shape,
                     DLDataType dtype, DLDevice device) {
    size_t nbytes = 1;
    for (auto d : shape) nbytes *= static_cast<size_t>(d);
    nbytes = (nbytes * dtype.bits * dtype.lanes + 7) / 8;

    auto* api = DeviceAPIRegistry::Get(device.device_type);
    void* data = api->Alloc(device, nbytes, 256);

    auto* sobj = new StorageObj();
    sobj->device    = device;
    sobj->data      = data;
    sobj->nbytes    = nbytes;
    sobj->alignment = 256;
    sobj->owns_data = true;

    auto* tobj = new TensorObj();
    tobj->storage               = Storage(sobj);
    tobj->shape_storage_        = shape;
    tobj->dl_tensor.data        = data;
    tobj->dl_tensor.device      = device;
    tobj->dl_tensor.ndim        = static_cast<int>(shape.size());
    tobj->dl_tensor.dtype       = dtype;
    tobj->dl_tensor.shape       = tobj->shape_storage_.data();
    tobj->dl_tensor.strides     = nullptr;
    tobj->dl_tensor.byte_offset = 0;

    return Tensor(tobj);
}

Tensor Tensor::FromStorage(ObjectRef storage,
                           int64_t byte_offset,
                           const std::vector<int64_t>& shape,
                           DLDataType dtype) {
    auto* sobj = storage.as<StorageObj>();
    if (!sobj) throw std::runtime_error("Tensor::FromStorage: invalid storage");

    auto* tobj = new TensorObj();
    tobj->storage               = storage;
    tobj->shape_storage_        = shape;
    tobj->dl_tensor.data        =
        static_cast<char*>(sobj->data) + byte_offset;
    tobj->dl_tensor.device      = sobj->device;
    tobj->dl_tensor.ndim        = static_cast<int>(shape.size());
    tobj->dl_tensor.dtype       = dtype;
    tobj->dl_tensor.shape       = tobj->shape_storage_.data();
    tobj->dl_tensor.strides     = nullptr;
    tobj->dl_tensor.byte_offset = 0;
    return Tensor(tobj);
}

Tensor Tensor::FromDLPack(DLManagedTensor* managed) {
    auto* obj = new TensorObj();
    obj->dl_tensor = managed->dl_tensor;

    obj->shape_storage_.assign(
        managed->dl_tensor.shape,
        managed->dl_tensor.shape + managed->dl_tensor.ndim);
    obj->dl_tensor.shape = obj->shape_storage_.data();

    if (managed->dl_tensor.strides) {
        obj->strides_storage_.assign(
            managed->dl_tensor.strides,
            managed->dl_tensor.strides + managed->dl_tensor.ndim);
        obj->dl_tensor.strides = obj->strides_storage_.data();
    }

    obj->manager_ctx = managed;
    // storage left empty: memory is externally owned
    return Tensor(obj);
}

DLManagedTensor* TensorObj::ToDLPack() const {
    auto* managed = new DLManagedTensor();
    managed->dl_tensor   = dl_tensor;
    managed->manager_ctx = const_cast<TensorObj*>(this);
    const_cast<TensorObj*>(this)->IncRef();
    managed->deleter = [](DLManagedTensor* self) {
        auto* obj = static_cast<TensorObj*>(self->manager_ctx);
        obj->DecRef();
        delete self;
    };
    return managed;
}

Tensor Tensor::FromExternalBuffer(
    void* data, DLDevice device,
    const std::vector<int64_t>& shape, DLDataType dtype,
    std::vector<int64_t> strides,
    std::function<void()> deleter) {

    auto* sobj = new StorageObj();
    sobj->device    = device;
    sobj->data      = data;
    sobj->nbytes    = 0;
    sobj->owns_data = false;
    sobj->deleter   = std::move(deleter);

    auto* tobj = new TensorObj();
    tobj->storage               = Storage(sobj);
    tobj->shape_storage_        = shape;
    tobj->dl_tensor.data        = data;
    tobj->dl_tensor.device      = device;
    tobj->dl_tensor.ndim        = static_cast<int>(shape.size());
    tobj->dl_tensor.dtype       = dtype;
    tobj->dl_tensor.shape       = tobj->shape_storage_.data();
    tobj->dl_tensor.byte_offset = 0;

    if (!strides.empty()) {
        tobj->strides_storage_  = std::move(strides);
        tobj->dl_tensor.strides = tobj->strides_storage_.data();
    } else {
        tobj->dl_tensor.strides = nullptr;
    }

    return Tensor(tobj);
}

}  // namespace devproc2
