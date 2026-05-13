#include <devproc2/runtime/vm.h>
#include <devproc2/runtime/device_api.h>
#include <devproc2/runtime/storage.h>
#include <devproc2/runtime/tensor.h>
#include <devproc2/runtime/shape_tuple.h>
#include <devproc2/runtime/tuple.h>
#include <stdexcept>
#include <string>
#include <atomic>

namespace devproc2 {

namespace {

// Guard against double-registration
std::atomic<bool> g_builtins_registered{false};

}  // namespace

void RegisterVMBuiltins() {
    if (g_builtins_registered.exchange(true)) return;  // idempotent

    auto& reg = BuiltinRegistry::Global();

    // ── vm.builtin.alloc_storage ───────────────────────────────────────────
    // args: [size_bytes: Int, alignment: Int, device_type: Int, device_id: Int]
    // → Storage
    reg.Register("vm.builtin.alloc_storage", [](std::vector<VMValue>& args) -> VMValue {
        auto nbytes    = static_cast<size_t>(args[0].AsInt());
        auto alignment = static_cast<size_t>(args[1].AsInt());
        auto dev_type  = static_cast<int>(args[2].AsInt());
        auto dev_id    = static_cast<int>(args[3].AsInt());
        DLDevice dev{static_cast<DLDeviceType>(dev_type), dev_id};

        auto* api = DeviceAPIRegistry::Get(dev_type);
        if (!api) {
            throw std::runtime_error(
                "vm.builtin.alloc_storage: no DeviceAPI for device_type=" +
                std::to_string(dev_type));
        }
        void* data = api->Alloc(dev, nbytes, alignment);

        auto* obj = new StorageObj();
        obj->device    = dev;
        obj->data      = data;
        obj->nbytes    = nbytes;
        obj->alignment = alignment;
        obj->owns_data = true;
        return VMValue::ObjRef(Storage(obj));
    });

    // ── vm.builtin.alloc_tensor ────────────────────────────────────────────
    // args: [storage: Storage, offset: Int, shape: ShapeTuple,
    //        dtype_code: Int, dtype_bits: Int, dtype_lanes: Int]
    // → Tensor
    reg.Register("vm.builtin.alloc_tensor", [](std::vector<VMValue>& args) -> VMValue {
        auto storage     = args[0].AsObjectRef();
        auto offset      = args[1].AsInt();
        auto shape_tuple = args[2].AsObjectRef();
        DLDataType dtype;
        dtype.code  = static_cast<uint8_t>(args[3].AsInt());
        dtype.bits  = static_cast<uint8_t>(args[4].AsInt());
        dtype.lanes = static_cast<uint16_t>(args[5].AsInt());

        auto* shobj = shape_tuple.as<ShapeTupleObj>();
        DEVPROC2_DCHECK(shobj);

        return VMValue::ObjRef(
            Tensor::FromStorage(storage, offset, shobj->dims, dtype));
    });

    // ── vm.builtin.make_shape ─────────────────────────────────────────────
    // args: [d0: Int, d1: Int, ...] → ShapeTuple
    reg.Register("vm.builtin.make_shape", [](std::vector<VMValue>& args) -> VMValue {
        std::vector<int64_t> dims;
        dims.reserve(args.size());
        for (auto& a : args) dims.push_back(a.AsInt());
        return VMValue::ObjRef(ShapeTuple::Make(std::move(dims)));
    });

    // ── vm.builtin.make_tuple ─────────────────────────────────────────────
    // args: [f0, f1, ...] → Tuple
    reg.Register("vm.builtin.make_tuple", [](std::vector<VMValue>& args) -> VMValue {
        std::vector<ObjectRef> fields;
        fields.reserve(args.size());
        for (auto& a : args) fields.push_back(a.AsObjectRef());
        return VMValue::ObjRef(Tuple::Make(std::move(fields)));
    });

    // ── vm.builtin.tuple_get_item ─────────────────────────────────────────
    // args: [tuple: Tuple, idx: Int] → ObjectRef
    reg.Register("vm.builtin.tuple_get_item", [](std::vector<VMValue>& args) -> VMValue {
        auto* tobj = args[0].AsObjectAs<TupleObj>();
        DEVPROC2_DCHECK(tobj);
        auto idx = static_cast<int>(args[1].AsInt());
        DEVPROC2_DCHECK(idx >= 0 && idx < static_cast<int>(tobj->size()));
        return VMValue::ObjRef((*tobj)[idx]);
    });

    // ── vm.builtin.identity ───────────────────────────────────────────────
    // args: [x] → x
    reg.Register("vm.builtin.identity", [](std::vector<VMValue>& args) -> VMValue {
        return args[0];
    });

    // ── vm.builtin.lt_i64 ────────────────────────────────────────────────
    // args: [a: Int, b: Int] → Bool
    reg.Register("vm.builtin.lt_i64", [](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Bool(args[0].AsInt() < args[1].AsInt());
    });

    // ── vm.builtin.add_i64 ────────────────────────────────────────────────
    // args: [a: Int, b: Int] → Int
    reg.Register("vm.builtin.add_i64", [](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Int(args[0].AsInt() + args[1].AsInt());
    });

    // ── vm.builtin.shape_assert ───────────────────────────────────────────
    // args: [tensor: Tensor, dim_idx: Int, upper: Int]
    reg.Register("vm.builtin.shape_assert", [](std::vector<VMValue>& args) -> VMValue {
        auto* tobj = args[0].AsObjectAs<TensorObj>();
        DEVPROC2_DCHECK(tobj);
        auto dim   = static_cast<int>(args[1].AsInt());
        auto upper = args[2].AsInt();
        DEVPROC2_DCHECK(dim >= 0 && dim < tobj->dl_tensor.ndim);
        auto actual = tobj->dl_tensor.shape[dim];
        if (actual > upper) {
            throw std::runtime_error(
                "RuntimeShapeError: dim " + std::to_string(dim) +
                " = " + std::to_string(actual) +
                " exceeds upper bound " + std::to_string(upper));
        }
        return VMValue{};
    });
}

}  // namespace devproc2
