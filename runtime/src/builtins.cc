#include <devproc2/runtime/vm.h>
#include <devproc2/runtime/device_api.h>
#include <devproc2/runtime/storage.h>
#include <devproc2/runtime/tensor.h>
#include <devproc2/runtime/shape_tuple.h>
#include <devproc2/runtime/tuple.h>
#include <devproc2/runtime/string.h>
#include <stdexcept>
#include <string>

namespace devproc2 {

// ── vm.builtin.alloc_storage ──────────────────────────────────────────────────
// args: [size_bytes: Int, alignment: Int, device_type: Int, device_id: Int]
// → Storage
DEVPROC2_REGISTER_BUILTIN("vm.builtin.alloc_storage")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
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

// ── vm.builtin.alloc_tensor ───────────────────────────────────────────────────
// args: [storage: Storage, offset: Int, shape: ShapeTuple,
//        dtype_code: Int, dtype_bits: Int, dtype_lanes: Int]
// → Tensor
DEVPROC2_REGISTER_BUILTIN("vm.builtin.alloc_tensor")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
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

// ── vm.builtin.make_shape ─────────────────────────────────────────────────────
// args: [d0: Int, d1: Int, ...] → ShapeTuple
DEVPROC2_REGISTER_BUILTIN("vm.builtin.make_shape")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        std::vector<int64_t> dims;
        dims.reserve(args.size());
        for (auto& a : args) dims.push_back(a.AsInt());
        return VMValue::ObjRef(ShapeTuple::Make(std::move(dims)));
    });

// ── vm.builtin.make_tuple ─────────────────────────────────────────────────────
// args: [f0, f1, ...] → Tuple
DEVPROC2_REGISTER_BUILTIN("vm.builtin.make_tuple")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        std::vector<ObjectRef> fields;
        fields.reserve(args.size());
        for (auto& a : args) fields.push_back(a.AsObjectRef());
        return VMValue::ObjRef(Tuple::Make(std::move(fields)));
    });

// ── vm.builtin.tuple_get_item ─────────────────────────────────────────────────
// args: [tuple: Tuple, idx: Int] → ObjectRef
DEVPROC2_REGISTER_BUILTIN("vm.builtin.tuple_get_item")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto* tobj = args[0].AsObjectAs<TupleObj>();
        DEVPROC2_DCHECK(tobj);
        auto idx = static_cast<int>(args[1].AsInt());
        DEVPROC2_DCHECK(idx >= 0 && idx < static_cast<int>(tobj->size()));
        return VMValue::ObjRef((*tobj)[idx]);
    });

// ── vm.builtin.identity ───────────────────────────────────────────────────────
// args: [x] → x
DEVPROC2_REGISTER_BUILTIN("vm.builtin.identity")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return args[0];
    });

// ── vm.builtin.lt_i64 ────────────────────────────────────────────────────────
// args: [a: Int, b: Int] → Bool
DEVPROC2_REGISTER_BUILTIN("vm.builtin.lt_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Bool(args[0].AsInt() < args[1].AsInt());
    });

// ── vm.builtin.add_i64 ────────────────────────────────────────────────────────
// args: [a: Int, b: Int] → Int
DEVPROC2_REGISTER_BUILTIN("vm.builtin.add_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Int(args[0].AsInt() + args[1].AsInt());
    });

// ── vm.builtin.shape_assert ───────────────────────────────────────────────────
// args: [tensor: Tensor, dim_idx: Int, upper: Int]
DEVPROC2_REGISTER_BUILTIN("vm.builtin.shape_assert")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
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

// ── vm.builtin.shape_of ───────────────────────────────────────────────────────
// args: [tensor: Tensor] → ShapeTuple
DEVPROC2_REGISTER_BUILTIN("vm.builtin.shape_of")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto* tobj = args[0].AsObjectAs<TensorObj>();
        DEVPROC2_DCHECK(tobj);
        int ndim = tobj->dl_tensor.ndim;
        std::vector<int64_t> dims(tobj->dl_tensor.shape,
                                  tobj->dl_tensor.shape + ndim);
        return VMValue::ObjRef(ShapeTuple::Make(std::move(dims)));
    });

// ── vm.builtin.get_shape_dim ──────────────────────────────────────────────────
// args: [shape: ShapeTuple, idx: Int] → Int
DEVPROC2_REGISTER_BUILTIN("vm.builtin.get_shape_dim")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto* shobj = args[0].AsObjectAs<ShapeTupleObj>();
        DEVPROC2_DCHECK(shobj);
        auto idx = static_cast<int>(args[1].AsInt());
        DEVPROC2_DCHECK(idx >= 0 && idx < static_cast<int>(shobj->dims.size()));
        return VMValue::Int(shobj->dims[static_cast<size_t>(idx)]);
    });

// ── Integer arithmetic builtins ───────────────────────────────────────────────

DEVPROC2_REGISTER_BUILTIN("vm.builtin.sub_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Int(args[0].AsInt() - args[1].AsInt());
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.mul_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Int(args[0].AsInt() * args[1].AsInt());
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.floordiv_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto a = args[0].AsInt();
        auto b = args[1].AsInt();
        DEVPROC2_DCHECK(b != 0);
        return VMValue::Int(a / b);
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.ceildiv_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto a = args[0].AsInt();
        auto b = args[1].AsInt();
        DEVPROC2_DCHECK(b > 0);
        return VMValue::Int((a + b - 1) / b);
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.min_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto a = args[0].AsInt(), b = args[1].AsInt();
        return VMValue::Int(a < b ? a : b);
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.max_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto a = args[0].AsInt(), b = args[1].AsInt();
        return VMValue::Int(a > b ? a : b);
    });

// ── Integer comparison builtins ───────────────────────────────────────────────

DEVPROC2_REGISTER_BUILTIN("vm.builtin.eq_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Bool(args[0].AsInt() == args[1].AsInt());
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.le_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Bool(args[0].AsInt() <= args[1].AsInt());
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.gt_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Bool(args[0].AsInt() > args[1].AsInt());
    });

DEVPROC2_REGISTER_BUILTIN("vm.builtin.ge_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        return VMValue::Bool(args[0].AsInt() >= args[1].AsInt());
    });

// ── vm.builtin.assert_le_i64 ──────────────────────────────────────────────────
// args: [val: Int, bound: Int, msg: String] → Null
DEVPROC2_REGISTER_BUILTIN("vm.builtin.assert_le_i64")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto val   = args[0].AsInt();
        auto bound = args[1].AsInt();
        if (val > bound) {
            std::string msg = "upper bound exceeded";
            if (args.size() >= 3) {
                auto* sobj = args[2].AsObjectAs<StringObj>();
                if (sobj) msg = sobj->data;
            }
            throw std::runtime_error(
                "RuntimeShapeError: " + msg +
                " (value=" + std::to_string(val) +
                ", bound=" + std::to_string(bound) + ")");
        }
        return VMValue{};
    });

}  // namespace devproc2
