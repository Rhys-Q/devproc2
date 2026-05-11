# devproc2 C++ Runtime 详细设计

---

## 1. Object / ObjectRef 动态类型系统

### 设计原则

MVP runtime 需要统一表示以下类型：Tensor / Storage / ShapeTuple / String / Tuple / PackedFunc / Kernel / Executable / VMState。

采用类 TVM 的 `Object / ObjectRef` 模式：
- `Object`：引用计数基类，有 `type_key()` 纯虚函数
- `ObjectRef`：持有 `Object*` 裸指针的 RAII 包装，**不暴露 `ObjectPtr`**
- 所有具体类型以 `XxxObj / Xxx` 命名对（如 `TensorObj / Tensor`）

### Object 基类

```cpp
// runtime/include/devproc2/runtime/object.h
class Object {
public:
    virtual ~Object() = default;
    virtual const char* type_key() const = 0;

    void IncRef() { ref_count_.fetch_add(1, std::memory_order_relaxed); }
    void DecRef() {
        if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }
    int32_t use_count() const { return ref_count_.load(); }

private:
    std::atomic<int32_t> ref_count_{0};
};
```

### ObjectRef

```cpp
// runtime/include/devproc2/runtime/object_ref.h
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
    ~ObjectRef() { if (ptr_) ptr_->DecRef(); }

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

private:
    Object* ptr_{nullptr};
};
```

### 核心对象类型

#### TensorObj / Tensor：DLPack 兼容设计

devproc2 的 Tensor 设计参考 TVM NDArray，**以 `DLTensor` 作为第一个字段**，使 `TensorObj*` 可以直接 reinterpret 为 `DLTensor*` 使用，实现零成本的 DLPack FFI。

```cpp
#include <dlpack/dlpack.h>  // https://github.com/dmlc/dlpack

// TensorObj / Tensor
class TensorObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Tensor";
    const char* type_key() const override { return _type_key; }

    // ──────────────────────────────────────────────────────────────
    // DLTensor 必须是第一个字段，保证 TensorObj* == DLTensor* 成立
    // （同 TVM NDArray::ContainerBase 的设计）
    // ──────────────────────────────────────────────────────────────
    DLTensor dl_tensor;  // data + device + ndim + dtype + shape + strides + byte_offset

    // 管理层：storage 负责内存生命周期
    Storage storage;         // 底层 StorageObj（引用计数共享）
    void* manager_ctx{nullptr};  // 外部 DLPack 来源时指向原始 DLManagedTensor

    // 取出 DLTensor* 用于底层 API（device copy 等）
    DLTensor* dl() { return &dl_tensor; }
    const DLTensor* dl() const { return &dl_tensor; }

    // 便捷访问
    void*    data()     const { return dl_tensor.data; }
    DLDevice device()   const { return dl_tensor.device; }
    int      ndim()     const { return dl_tensor.ndim; }
    DLDataType dtype()  const { return dl_tensor.dtype; }
    int64_t* shape()    const { return dl_tensor.shape; }
    int64_t* strides()  const { return dl_tensor.strides; }

    // 分配（通过 DeviceAPI）
    static Tensor Empty(ShapeTuple shape, DLDataType dtype, DLDevice device);

    // DLPack 互操作
    static Tensor FromDLPack(DLManagedTensor* managed);  // zero-copy，接管外部 tensor
    DLManagedTensor* ToDLPack() const;                   // 导出给 torch / numpy / JAX

    // 外部 buffer 包装（用户 buffer，不负责释放）
    static Tensor FromExternalBuffer(
        void* data, DLDevice device,
        ShapeTuple shape, DLDataType dtype,
        std::vector<int64_t> strides = {},
        std::function<void()> deleter = nullptr);

private:
    std::vector<int64_t> shape_storage_;    // shape 数组的实际内存
    std::vector<int64_t> strides_storage_;  // strides 数组的实际内存（可选）
};
using Tensor = ObjectRef;  // holds TensorObj
```

**DLPack 互操作实现**：

```cpp
// FromDLPack：将 PyTorch / NumPy / JAX 的 DLManagedTensor 包装为 devproc2 Tensor
// 不拷贝数据，通过 manager_ctx 持有外部引用，析构时调用外部 deleter
Tensor TensorObj::FromDLPack(DLManagedTensor* managed) {
    auto* obj = new TensorObj();
    obj->dl_tensor = managed->dl_tensor;  // 浅拷贝 DLTensor header（data ptr 共享）
    // 将 shape/strides 拷贝到自有 storage（避免悬空指针）
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
    obj->manager_ctx = managed;  // 持有引用，防止提前释放
    // storage 为空（外部内存，不由 devproc2 释放）
    return ObjectRef(obj);
}

// TensorObj 析构时释放外部 DLManagedTensor
TensorObj::~TensorObj() {
    if (manager_ctx) {
        auto* managed = static_cast<DLManagedTensor*>(manager_ctx);
        if (managed->deleter) managed->deleter(managed);
    }
}

// ToDLPack：导出给外部框架（PyTorch 通过 from_dlpack 接收）
DLManagedTensor* TensorObj::ToDLPack() const {
    auto* managed = new DLManagedTensor();
    managed->dl_tensor = dl_tensor;
    managed->manager_ctx = const_cast<TensorObj*>(this);
    this->IncRef();  // 让 TensorObj 在外部框架持有期间不被释放
    managed->deleter = [](DLManagedTensor* self) {
        auto* obj = static_cast<TensorObj*>(self->manager_ctx);
        obj->DecRef();
        delete self;
    };
    return managed;
}
```

**与 PyTorch 互操作示例**：

```python
# Python 侧（binding.py）
import torch
from devproc2.runtime import binding as rt

# torch → devproc2（zero-copy）
torch_tensor = torch.randn(4, 512, 4096, dtype=torch.float16, device="cuda")
dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
devproc_tensor = rt.from_dlpack(dlpack_capsule)

# devproc2 → torch（zero-copy）
result_tensor = vm.invoke("main", [devproc_tensor])
torch_result = torch.utils.dlpack.from_dlpack(result_tensor.to_dlpack())
```

#### StorageObj / Storage

```cpp
class StorageObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Storage";
    const char* type_key() const override { return _type_key; }

    DLDevice device;
    void* data{nullptr};
    size_t nbytes{0};
    size_t alignment{256};
    bool owns_data{true};  // false = external buffer，不负责释放
    std::function<void()> deleter;  // external buffer 自定义释放

    ~StorageObj() {
        if (owns_data && data) {
            DeviceAPIRegistry::Get((DeviceType)device.device_type)->Free(device, data);
        } else if (!owns_data && deleter) {
            deleter();
        }
    }
};
using Storage = ObjectRef;

// ShapeTupleObj / ShapeTuple
class ShapeTupleObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.ShapeTuple";
    const char* type_key() const override { return _type_key; }

    std::vector<int64_t> dims;
    int64_t ndim() const { return dims.size(); }
    int64_t operator[](int i) const { return dims[i]; }
};
using ShapeTuple = ObjectRef;

// PackedFuncObj / PackedFunc
class PackedArgs {
public:
    explicit PackedArgs(std::vector<VMValue>& args) : args_(args) {}
    int size() const { return args_.size(); }
    VMValue& operator[](int i) { return args_[i]; }
    const VMValue& operator[](int i) const { return args_[i]; }
private:
    std::vector<VMValue>& args_;
};

class PackedFuncObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.PackedFunc";
    const char* type_key() const override { return _type_key; }

    std::function<void(PackedArgs)> body;
    void Call(PackedArgs args) { body(args); }
};
using PackedFunc = ObjectRef;
```

---

## 2. VMValue

VM register 存储的 tagged union：

```cpp
// runtime/include/devproc2/runtime/vm_value.h
class VMValue {
public:
    enum class Tag { kNull, kInt, kFloat, kBool, kObjectRef };

    VMValue() : tag_(Tag::kNull) {}
    static VMValue Int(int64_t v)      { VMValue r; r.tag_=Tag::kInt;   r.data_.i=v; return r; }
    static VMValue Float(double v)     { VMValue r; r.tag_=Tag::kFloat; r.data_.f=v; return r; }
    static VMValue Bool(bool v)        { VMValue r; r.tag_=Tag::kBool;  r.data_.b=v; return r; }
    static VMValue ObjRef(ObjectRef o) { VMValue r; r.tag_=Tag::kObjectRef; r.obj_=std::move(o); return r; }

    Tag tag() const { return tag_; }
    bool IsNull()      const { return tag_ == Tag::kNull; }
    bool IsInt()       const { return tag_ == Tag::kInt; }
    bool IsFloat()     const { return tag_ == Tag::kFloat; }
    bool IsBool()      const { return tag_ == Tag::kBool; }
    bool IsObjectRef() const { return tag_ == Tag::kObjectRef; }

    int64_t AsInt()        const { DCHECK(IsInt());       return data_.i; }
    double  AsFloat()      const { DCHECK(IsFloat());     return data_.f; }
    bool    AsBool()       const { DCHECK(IsBool());      return data_.b; }
    ObjectRef AsObjectRef() const { DCHECK(IsObjectRef()); return obj_; }

    template <typename T>
    T* AsObjectAs() const { return obj_.as<T>(); }

private:
    Tag tag_{Tag::kNull};
    union { int64_t i; double f; bool b; } data_{};
    ObjectRef obj_;  // 仅 kObjectRef 时有效
};
```

---

## 3. VM 执行模型

### 数据结构

```cpp
// runtime/include/devproc2/runtime/vm.h

// VM 指令（4 种 opcode）
enum class Opcode : uint8_t { CALL = 0, RET = 1, IF = 2, GOTO = 3 };

struct Instruction {
    Opcode opcode;
    union {
        struct { int32_t dst_reg; int32_t func_idx; int32_t num_args; } call;
        struct { int32_t src_reg; } ret;
        struct { int32_t cond_reg; int32_t true_offset; int32_t false_offset; } if_;
        struct { int32_t offset; } goto_;
    };
    std::vector<int32_t> arg_regs;  // 仅 CALL 使用
};

// callee 类型
enum class CalleeKind : uint8_t {
    kVMFunc    = 0,  // IR Function，压栈递归执行
    kBuiltin   = 1,  // vm.builtin.* 内置函数
    kPackedFunc = 2, // PackedFunc registry 中的函数
    kKernel    = 3,  // KernelRegistry 中的 CUDA kernel
};

struct FunctionEntry {
    std::string name;
    CalleeKind  kind;
    int32_t     instr_offset;  // 在全局指令数组中的起始位置
    int32_t     instr_count;
    int32_t     num_regs;      // 此函数需要的 register 数
    int32_t     num_args;
};

// Executable：只读，多个 VMState 可共享
class Executable {
public:
    std::vector<FunctionEntry>  function_table;
    std::vector<Instruction>    instructions;
    std::vector<VMValue>        constants;  // 编译时常量

    int32_t GetFuncIndex(const std::string& name) const;
    static Executable Load(const std::string& path);
    void Save(const std::string& path) const;
};

// VM 调用帧
struct VMFrame {
    int32_t func_idx;
    int32_t pc;        // 当前指令在 function 内的偏移
    int32_t reg_base;  // 在全局 register file 中的起始位置
};

// VM 状态（每次 invoke 一个独立 VMState）
class VMState {
public:
    explicit VMState(std::shared_ptr<Executable> exec);

    // 执行入口
    VMValue Invoke(const std::string& func_name, std::vector<VMValue> args);

    // session 级别资源
    std::unordered_map<Device, Stream, DeviceHash> default_streams;

private:
    std::shared_ptr<Executable> exec_;
    std::vector<VMFrame>        frames_;
    std::vector<VMValue>        regs_;   // 全局 register file（按需扩展）

    void ExecuteLoop();
    void DispatchCall(const Instruction& instr);
    VMValue CallBuiltin(int32_t func_idx, std::vector<VMValue>& args);
    VMValue CallPackedFunc(int32_t func_idx, std::vector<VMValue>& args);
    VMValue CallKernel(int32_t func_idx, std::vector<VMValue>& args);
};
```

### 执行循环

```cpp
void VMState::ExecuteLoop() {
    while (!frames_.empty()) {
        VMFrame& frame = frames_.back();
        const FunctionEntry& func = exec_->function_table[frame.func_idx];
        const Instruction& instr = exec_->instructions[func.instr_offset + frame.pc];

        switch (instr.opcode) {
        case Opcode::CALL: {
            DispatchCall(instr);
            break;
        }
        case Opcode::RET: {
            VMValue result = (instr.ret.src_reg >= 0)
                ? regs_[frame.reg_base + instr.ret.src_reg]
                : VMValue{};
            frames_.pop_back();
            // 将 result 写回 caller 的 dst_reg
            if (!frames_.empty()) {
                // (由 DispatchCall 中 vm_func 分支负责写回)
            }
            // ...（return value 传递逻辑）
            break;
        }
        case Opcode::IF: {
            bool cond = regs_[frame.reg_base + instr.if_.cond_reg].AsBool();
            frame.pc += (cond ? instr.if_.true_offset : instr.if_.false_offset);
            continue;  // 不 ++pc
        }
        case Opcode::GOTO: {
            frame.pc += instr.goto_.offset;
            continue;
        }
        }
        ++frame.pc;
    }
}
```

---

## 4. DeviceAPI 抽象层

### 接口定义

```cpp
// runtime/include/devproc2/runtime/device_api.h
#include <dlpack/dlpack.h>

enum class DeviceType : int { kCPU = 0, kCUDA = 1, kNPU = 2, kMetal = 3, kVulkan = 4 };

// 复用 DLPack 的 DLDevice，不再自定义 Device struct
// DLDevice = { device_type: DLDeviceType, device_id: int }
using Device = DLDevice;

struct DeviceHash {
    size_t operator()(const Device& d) const {
        return std::hash<int>()(d.device_type * 100 + d.device_id);
    }
};

class DeviceAPI {
public:
    virtual ~DeviceAPI() = default;

    virtual void* Alloc(Device dev, size_t nbytes, size_t alignment) = 0;
    virtual void  Free(Device dev, void* ptr) = 0;

    // 接受 DLTensor*，与 TVM DeviceAPI::CopyDataFromTo 接口对齐
    // 框架可直接传 TensorObj::dl() 指针，无需手动提取 data/shape
    virtual void CopyDataFromTo(
        DLTensor* from,
        DLTensor* to,
        TVMStreamHandle stream) = 0;

    virtual void StreamSync(Device dev, void* stream) = 0;
    virtual void DeviceSync(Device dev) = 0;

    virtual void* CreateStream(Device dev) = 0;
    virtual void  FreeStream(Device dev, void* stream) = 0;
    virtual void  SetDevice(Device dev) = 0;
};

class DeviceAPIRegistry {
public:
    static DeviceAPI* Get(int device_type);
    static void Register(int device_type, DeviceAPI* api);
private:
    static std::unordered_map<int, DeviceAPI*>& Registry();
};
```

### CPUDeviceAPI

```cpp
// runtime/src/device_api.cc（CPU 部分）
class CPUDeviceAPI : public DeviceAPI {
public:
    void* Alloc(Device, size_t nbytes, size_t alignment) override {
        return std::aligned_alloc(alignment, (nbytes + alignment - 1) / alignment * alignment);
    }
    void Free(Device, void* ptr) override { std::free(ptr); }
    void CopyDataFromTo(const void* from, void* to, size_t nbytes,
                        Device, Device, void*) override {
        std::memcpy(to, from, nbytes);
    }
    void StreamSync(Device, void*) override {}
    void DeviceSync(Device) override {}
    void* CreateStream(Device) override { return nullptr; }
    void FreeStream(Device, void*) override {}
    void SetDevice(Device) override {}
};
```

### CUDADeviceAPI（简要）

```cpp
// runtime/src/cuda/cuda_device_api.cc
#ifdef DEVPROC2_WITH_CUDA
class CUDADeviceAPI : public DeviceAPI {
public:
    void* Alloc(Device dev, size_t nbytes, size_t) override {
        SetDevice(dev);
        void* ptr;
        CUDA_CALL(cudaMalloc(&ptr, nbytes));
        return ptr;
    }
    void Free(Device, void* ptr) override {
        CUDA_CALL(cudaFree(ptr));
    }
    void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) override {
        // 根据 from/to 的 device_type 选择 cudaMemcpyKind
        cudaMemcpyKind kind = cudaMemcpyDefault;
        size_t nbytes = 1;
        for (int i = 0; i < from->ndim; ++i) nbytes *= from->shape[i];
        nbytes *= (from->dtype.bits * from->dtype.lanes + 7) / 8;
        CUDA_CALL(cudaMemcpyAsync(
            static_cast<char*>(to->data) + to->byte_offset,
            static_cast<const char*>(from->data) + from->byte_offset,
            nbytes, kind, static_cast<cudaStream_t>(stream)));
    }
    void StreamSync(Device, void* stream) override {
        CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    }
    void DeviceSync(Device) override { CUDA_CALL(cudaDeviceSynchronize()); }
    void* CreateStream(Device dev) override {
        SetDevice(dev);
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        return static_cast<void*>(stream);
    }
    void FreeStream(Device, void* stream) override {
        CUDA_CALL(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
    }
    void SetDevice(Device dev) override {
        CUDA_CALL(cudaSetDevice(dev.device_id));
    }
};
#endif
```

### Zero-copy External Buffer

`TensorObj::FromExternalBuffer` 和 `TensorObj::FromDLPack` 已在第 1 节定义，此处说明 Memory Planner 侧的配合规则：

- `StorageObj::owns_data = false` 的 storage → MemoryPlanner 不为其分配内存，不参与 storage reuse
- 识别方式：编译器对函数参数中的 input tensor 和 output tensor 标记为 `external=true`
- `effect=write(k_cache, v_cache)` 的 mutable external buffer 也标记为 `external=true`

```cpp
// DeviceAPI::CopyDataFromTo 接受 DLTensor*
// 因此 Tensor 之间的 H2D / D2H / D2D copy 可以直接写：
auto* api = DeviceAPIRegistry::Get(src_tensor->dl_tensor.device.device_type);
api->CopyDataFromTo(src_tensor->dl(), dst_tensor->dl(), stream_handle);
```

---

## 5. MemoryPool

MemoryPool 管理 session 级别的 storage 分配。VM 在函数开头执行 `alloc_storage` builtin，通过 MemoryPool 分配。

```cpp
// runtime/include/devproc2/runtime/memory_pool.h
class MemoryPool {
public:
    Storage Alloc(Device device, size_t nbytes, size_t alignment) {
        auto* api = DeviceAPIRegistry::Get(device.type);
        void* ptr = api->Alloc(device, nbytes, alignment);

        auto* obj = new StorageObj();
        obj->device = device;
        obj->data = ptr;
        obj->nbytes = nbytes;
        obj->owns_data = true;

        return ObjectRef(obj);
    }

    // session 级别：提前分配 storage plan 中所有 storage
    void PreAllocate(const StoragePlan& plan) {
        for (const auto& entry : plan.entries) {
            storages_[entry.id] = Alloc(entry.device, entry.size_bytes, 256);
        }
    }

    Storage GetStorage(int32_t id) { return storages_.at(id); }

private:
    std::unordered_map<int32_t, Storage> storages_;
};
```

---

## 6. PackedFunc Registry

```cpp
// runtime/src/packed_func.cc

class PackedFuncRegistry {
public:
    static PackedFuncRegistry& Global() {
        static PackedFuncRegistry instance;
        return instance;
    }

    void Register(const std::string& name, PackedFunc func) {
        std::lock_guard<std::mutex> lock(mu_);
        registry_[name] = std::move(func);
    }

    PackedFunc Get(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = registry_.find(name);
        if (it == registry_.end()) return PackedFunc{};
        return it->second;
    }

    bool Has(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mu_);
        return registry_.count(name) > 0;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, PackedFunc> registry_;
};

// 注册辅助类（用于静态初始化）
struct PackedFuncRegistrar {
    explicit PackedFuncRegistrar(const char* name) : name_(name) {}
    PackedFuncRegistrar& set_body(std::function<void(PackedArgs)> f) {
        auto* obj = new PackedFuncObj();
        obj->body = std::move(f);
        PackedFuncRegistry::Global().Register(name_, ObjectRef(obj));
        return *this;
    }
    std::string name_;
};

#define DEVPROC_REGISTER_PACKED_FUNC(name) \
    static ::devproc2::PackedFuncRegistrar _reg_##__LINE__(name)
```

**使用示例**：

```cpp
DEVPROC_REGISTER_PACKED_FUNC("runtime.tokenizer.encode")
    .set_body([](PackedArgs args) {
        // args[0]: String（输入文本）
        // args[1]: Tensor（output token ids，由 caller 分配）
        String text = args[0].AsObjectAs<StringObj>()->data;
        Tensor output = args[1].AsObjectRef();
        // ... 写入 token ids
    });
```

---

## 7. Kernel Registry + CUDA Launch

### Kernel 注册

```cpp
// KernelObj：持有 cubin binary + metadata
class KernelObj : public Object {
public:
    static constexpr const char* _type_key = "runtime.Kernel";
    const char* type_key() const override { return _type_key; }

    std::string name;
    std::vector<uint8_t> cubin_data;   // cubin binary
    std::vector<uint8_t> ptx_data;     // 可选 ptx（调试用）
    std::string func_name;             // CUDA function 名
    // grid/block 表达式已在 VM 层计算为 int64 参数，此处只存 block 常量
    std::array<int32_t, 3> block_dims; // {block_x, block_y, block_z}
};

// KernelRegistry
class KernelRegistry {
public:
    static KernelRegistry& Global();

    void Register(const std::string& name, Kernel kernel);
    Kernel Lookup(const std::string& name) const;
};
```

### CUDA Kernel Launch

```cpp
// runtime/src/cuda/cuda_module.cc

class CUDAModule {
public:
    explicit CUDAModule(const std::vector<uint8_t>& cubin) {
        CUDA_DRIVER_CALL(cuModuleLoadData(&module_, cubin.data()));
    }
    ~CUDAModule() { cuModuleUnload(module_); }

    CUfunction GetFunction(const std::string& name) {
        CUfunction func;
        CUDA_DRIVER_CALL(cuModuleGetFunction(&func, module_, name.c_str()));
        return func;
    }

private:
    CUmodule module_;
};

// Kernel Launch：grid 由 VM 层在调用前计算好，作为参数传入
void LaunchKernel(
    CUfunction func,
    int32_t grid_x, int32_t grid_y, int32_t grid_z,
    int32_t block_x, int32_t block_y, int32_t block_z,
    std::vector<void*>& kernel_params,  // CUDA kernel 参数指针数组
    CUstream stream)
{
    CUDA_DRIVER_CALL(cuLaunchKernel(
        func,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        0,        // shared memory bytes
        stream,
        kernel_params.data(),
        nullptr
    ));
}
```

---

## 8. VM Builtins

所有 shape 计算和内存分配通过 builtin 函数完成，不新增 VM opcode。

```cpp
// runtime/src/builtins.cc

// 注册所有 vm.builtin.* 函数
void RegisterVMBuiltins() {
    // 内存分配
    DEVPROC_REGISTER_PACKED_FUNC("vm.builtin.alloc_storage")
        .set_body([](PackedArgs args) {
            // args: [size_bytes: Int, alignment: Int, device_type: Int, device_id: Int]
            // → Storage
        });

    DEVPROC_REGISTER_PACKED_FUNC("vm.builtin.alloc_tensor")
        .set_body([](PackedArgs args) {
            // args: [storage: Storage, offset: Int, shape: ShapeTuple, dtype: Int]
            // → Tensor
        });

    // Shape 操作
    DEVPROC_REGISTER_PACKED_FUNC("vm.builtin.shape_of")
        .set_body([](PackedArgs args) {
            // args: [tensor: Tensor] → ShapeTuple
            auto* t = args[0].AsObjectAs<TensorObj>();
            args[0] = VMValue::ObjRef(t->shape);
        });

    DEVPROC_REGISTER_PACKED_FUNC("vm.builtin.get_shape_dim")
        .set_body([](PackedArgs args) {
            // args: [shape: ShapeTuple, idx: Int] → Int
            auto* s = args[0].AsObjectAs<ShapeTupleObj>();
            int64_t idx = args[1].AsInt();
            args[0] = VMValue::Int((*s)[idx]);
        });

    DEVPROC_REGISTER_PACKED_FUNC("vm.builtin.ceildiv_i64")
        .set_body([](PackedArgs args) {
            // args: [a: Int, b: Int] → Int
            int64_t a = args[0].AsInt(), b = args[1].AsInt();
            args[0] = VMValue::Int((a + b - 1) / b);
        });

    DEVPROC_REGISTER_PACKED_FUNC("vm.builtin.assert_le_i64")
        .set_body([](PackedArgs args) {
            // args: [val: Int, bound: Int, msg: String]
            int64_t val = args[0].AsInt(), bound = args[1].AsInt();
            if (val > bound) {
                std::string msg = args[2].AsObjectAs<StringObj>()->data;
                throw std::runtime_error("RuntimeShapeError: " + msg +
                    " actual=" + std::to_string(val) +
                    " bound=" + std::to_string(bound));
            }
        });

    // 算术 builtins（add/sub/mul/floordiv/min/max/compare）
    // ... 类似模式注册
}
```

---

## 9. ABI 格式规范

### manifest.json

```json
{
  "name": "kvcache_demo",
  "devproc2_version": "0.1.0",
  "build_time": "2026-05-11T00:00:00Z",
  "target": "cuda",
  "target_arch": "sm_80",
  "python_version": "3.11"
}
```

### abi.json

```json
{
  "devproc_abi_version": "0.1",
  "vm_bytecode_version": "0.1",
  "kernel_calling_convention": "dps_kernel_v1",
  "packed_func_calling_convention": "dps_packed_v1",
  "inputs": [
    {"name": "x",       "dtype": "float16", "shape": ["B", "S", 4096]},
    {"name": "k_cache", "dtype": "float16", "shape": [8, 2048, 32, 128], "mutable": true},
    {"name": "v_cache", "dtype": "float16", "shape": [8, 2048, 32, 128], "mutable": true},
    {"name": "pos",     "dtype": "int32",   "shape": []}
  ],
  "outputs": [
    {"name": "out", "dtype": "float16", "shape": ["B", "S", 4096]}
  ],
  "shape_constraints": {
    "B": {"lower": 1, "upper": 8},
    "S": {"lower": 1, "upper": 2048}
  },
  "required_packed_funcs": [
    "runtime.tokenizer.encode"
  ]
}
```

### ABI 版本兼容性规则

- `devproc_abi_version` 主版本（0.x → 1.x）不兼容，加载时报错
- `vm_bytecode_version` 主版本不兼容，加载时报错
- 次版本（x.0 → x.1）向后兼容，只警告不报错

### Executable 加载验证

```cpp
// runtime/src/executable.cc
Executable Executable::Load(const std::string& artifact_dir) {
    // 1. 读 abi.json，检查版本
    auto abi = LoadJson(artifact_dir + "/abi.json");
    CheckABIVersion(abi["devproc_abi_version"]);

    // 2. 加载 executable.vm
    auto exec = DeserializeVM(artifact_dir + "/executable.vm");

    // 3. 检查所需 packed_funcs 均已注册
    for (const auto& name : abi["required_packed_funcs"]) {
        if (!PackedFuncRegistry::Global().Has(name)) {
            throw std::runtime_error(
                "PackedFunc '" + name + "' is required but not registered.");
        }
    }

    // 4. 加载 cubin
    auto kernel_table = LoadJson(artifact_dir + "/metadata/kernel_table.json");
    for (const auto& entry : kernel_table) {
        LoadCUBIN(entry["name"], artifact_dir + "/kernels/" + entry["cubin"]);
    }

    return exec;
}
```

---

## 10. Python Binding

通过 pybind11 将 C++ runtime 暴露给 Python：

```cpp
// runtime/binding/python_binding.cc
#include <pybind11/pybind11.h>
#include <devproc2/runtime/vm.h>
#include <devproc2/runtime/tensor.h>

namespace py = pybind11;

PYBIND11_MODULE(devproc2_cpp, m) {
    m.doc() = "devproc2 C++ runtime binding";

    // Executable
    py::class_<Executable>(m, "Executable")
        .def_static("load", &Executable::Load)
        .def("save", &Executable::Save);

    // VMState
    py::class_<VMState>(m, "VMState")
        .def(py::init<std::shared_ptr<Executable>>())
        .def("invoke", [](VMState& vm, const std::string& name,
                          std::vector<VMValue> args) {
            return vm.Invoke(name, std::move(args));
        });

    // PackedFuncRegistry
    m.def("register_packed_func", [](const std::string& name, py::function fn) {
        auto* obj = new PackedFuncObj();
        obj->body = [fn](PackedArgs args) {
            // 将 PackedArgs 转为 Python 对象列表后调用 fn
            // ...
        };
        PackedFuncRegistry::Global().Register(name, ObjectRef(obj));
    });

    // Tensor
    py::class_<TensorObj>(m, "Tensor")
        .def_property_readonly("shape", [](const TensorObj& t) {
            return t.shape.as<ShapeTupleObj>()->dims;
        })
        .def_property_readonly("dtype", &TensorObj::dtype)
        .def_property_readonly("device", [](const TensorObj& t) {
            return std::make_pair((int)t.device.type, t.device.device_id);
        });
}
```

---

## 11. 重要设计约束总结

| 约束 | 原因 |
|---|---|
| `TensorObj` 的第一个字段是 `DLTensor` | TensorObj* 可直接当 DLTensor* 用，零成本 DLPack FFI；与 TVM NDArray 设计对齐 |
| `Device` 复用 DLPack 的 `DLDevice` | 统一使用 `device_type + device_id` 语义，避免自定义类型与 DLPack 做转换 |
| `DeviceAPI::CopyDataFromTo` 接受 `DLTensor*` | 可直接传 TensorObj::dl()，无需手动提取 data/nbytes；与 TVM DeviceAPI 接口对齐 |
| `FromDLPack / ToDLPack` 零拷贝互操作 | torch/numpy/JAX 均支持 DLPack；`__dlpack__` 协议已是 Python 生态标准 |
| VM 不直接调用 `cudaMalloc/cudaFree/cudaMemcpy` | 所有 device 操作经 DeviceAPI，后续扩展 NPU/Metal 不需改 VM |
| `alloc_storage/alloc_tensor` 只在 Pass 12 之后出现 | 前端 DSL 自然表达，内存规划由中端自动完成 |
| `CallDPS output=None` 不可 DCE | no-output stateful call（update_kvcache）是一等公民 |
| kernel ABI 显式传 shape scalar | kernel 内部不解析 Tensor metadata，ABI 边界清晰 |
| VM 只有 4 条指令 | 复杂性放在 function table 和 callee dispatch，不膨胀 opcode |
| 不暴露 `ObjectPtr` | API 层只有 `ObjectRef`，简化内存管理心智模型 |
| storage reuse 不跨 device | 跨 device copy 语义复杂，MVP 保守处理 |
| Triton 只做 AOT 编译 | Runtime 不编译 Triton，只加载 cubin，startup latency 可控 |
