# devproc2 MVP 里程碑规划

共 15 个里程碑：M1-M12（核心功能）+ X1-X3（Runtime 扩展）。

严格按下方顺序推进。每个里程碑完成后需通过其验收标准，再进入下一个。

---

## M1：C++ Object / ObjectRef 动态类型系统

**目标**：建立 runtime 类型系统底座，所有 C++ 对象统一继承 Object，统一通过 ObjectRef 引用计数管理。

### 任务清单

**核心基础**
- [ ] `Object`：引用计数基类（`std::atomic<int32_t>`），`type_key()` 纯虚函数，`IncRef/DecRef`
- [ ] `ObjectRef`：持有 `Object*` 裸指针（不暴露 `ObjectPtr`），模板 `as<T>()`，析构时自动 DecRef
- [ ] `VMValue`：tagged union（Null / Int / Float / Bool / ObjectRef），用作 VM register 值

**核心对象类型**
- [ ] `TensorObj / Tensor`：参考 TVM NDArray，**`DLTensor` 作为第一字段**（`TensorObj*` 可直接作 `DLTensor*` 用），携带 `Storage` 引用管理内存生命周期；实现 `FromDLPack / ToDLPack`（zero-copy torch 互操作）和 `FromExternalBuffer`
- [ ] `StorageObj / Storage`：`DLDevice` + data + nbytes + owns_data 标志；析构时通过 DeviceAPI 释放内存
- [ ] `ShapeTupleObj / ShapeTuple`：int64 数组
- [ ] `TupleObj / Tuple`：ObjectRef 数组
- [ ] `StringObj / String`：std::string 包装
- [ ] `PackedFuncObj / PackedFunc`：`std::function<void(PackedArgs)>` 包装
- [ ] `KernelObj / Kernel`：cubin binary + metadata（暂时 stub）

**头文件**（对应 `runtime/include/devproc2/runtime/`）

```
object.h / object_ref.h / vm_value.h
tensor.h / storage.h / shape_tuple.h / tuple.h / string.h
packed_func.h / kernel.h
```

注：`tensor.h` 需要 `#include <dlpack/dlpack.h>`；CMakeLists.txt 需要通过 FetchContent 或系统路径引入 dlpack 头文件。

### 验收标准

- C++ 单测能创建 Tensor/Storage/ShapeTuple/Tuple/String 并互相嵌套
- 引用计数：最后一个 ObjectRef 析构后，Object 被正确 delete（valgrind 无泄漏）
- `VMValue` 能保存 Int / Float / Bool / ObjectRef，tag 查询正确
- `ObjectRef::as<T>()` 类型不匹配时返回 nullptr 而不是崩溃
- `TensorObj::FromDLPack` 能从 torch tensor 的 DLPack capsule 创建 Tensor，数据地址一致（零拷贝验证）

---

## M2：High-level IR MVP

**目标**：实现前端高层 IR 的 Python 数据结构，能构造并打印出规范文本形式。这一层 IR 中**不允许出现** `alloc_storage` / `alloc_tensor`。

### 任务清单

**IR 节点（`python/devproc2/ir/`）**
- [ ] `IRModule`：函数字典 `{name: Function}`
- [ ] `Function`：参数列表 + Block + 返回类型
- [ ] `Block`：`(bindings: List[Binding], body: Expr)`
- [ ] `Var`：SSA 变量（name + struct_info）
- [ ] `Constant`：标量/张量常量
- [ ] `Call`：普通函数式调用，最多一个返回值
- [ ] `CallDPS`：DPS 调用，含 inputs + output（可 None）+ effect + callee_kind
- [ ] `Tuple`：值元组（`(a, b, c)`）
- [ ] `TupleGetItem`：`tuple[i]`
- [ ] `TensorCreateOp`：`dp.empty / dp.zeros / dp.full / dp.empty_like`
- [ ] `Return`：函数返回
- [ ] `TensorStructInfo`：shape（ShapeExpr）+ dtype + device
- [ ] `ShapeExpr`：常量、SymbolicDim 引用、算术组合
- [ ] `SymbolicDim`：名字 + UpperBound
- [ ] `EffectInfo`：`pure | read_only | write(vars) | opaque`

**工具（`python/devproc2/ir/`）**
- [ ] `printer.py`：将 IRModule 转为可读文本（类似 TVM Relax 打印格式）
- [ ] `verifier.py`：检查 IR 不变量（SSA 定义前使用、callee_kind 合法、alloc_* 不出现等）

### 验收标准

打印结果如下形式的 IR：

```
@main(%x: Tensor[(B, S, 4096), float16], %w: Tensor[(4096, 4096), float16]) -> Tensor[(B, S, 4096), float16] {
  %y = call @matmul(%x, %w)
  %z = call @silu(%y)
  return %z
}
```

以及含 no-output CallDPS 的：

```
call_dps @kernel.update_kvcache(
  inputs=[%k_cache, %v_cache, %k, %v, %pos],
  output=None,
  callee_kind=kernel,
  effect=write(%k_cache, %v_cache)
)
```

verifier 能检测到 `alloc_storage` 出现时报错。

---

## M3：Control Flow MVP

**目标**：支持 Python `if/elif/else` 和 `for/dp.range`，IR 采用结构化表达（不引入 CFG / φ 节点）。

### 任务清单

**IR 节点（`python/devproc2/ir/control_flow.py`）**
- [ ] `If`：condition + true_branch(Block) + false_branch(Block)
- [ ] `For`：loop_var + range + iter_args（loop-carried variables）+ body(Block) + yield

**前端 DSL（`python/devproc2/frontend/dsl.py`）**
- [ ] 通过 `ast` 模块捕获 Python `if/elif/else`
- [ ] 通过 `ast` 模块捕获 `for i in dp.range(start, end, step)`
- [ ] loop-carried variable 检测（循环体内被重新赋值的变量）

**Pass（`python/devproc2/compiler/passes/control_flow_normalize.py`）**
- [ ] `elif` → nested `if` 展平
- [ ] `for` body 内对外部变量的读写转为 iter_args

### 验收标准

以下代码通过编译并输出正确 IR：

```python
@dp.function
def decode_step(x, flag, n):
    if flag:
        y = dp.ops.relu(x)
    elif flag > 0:
        y = dp.ops.silu(x)
    else:
        y = dp.ops.gelu(x)

    for i in dp.range(0, n):
        y = dp.ops.layernorm(y)

    return y
```

生成的 IR 中 elif 被展平为嵌套 if，loop-carried `y` 通过 iter_args 正确传递。

---

## M4：Dynamic Shape MVP

**目标**：完整支持 `SymbolicDim + UpperBound`，能参与 StructInfo 类型推导，并在编译期验证约束一致性。

### 任务清单

**IR 扩展（`python/devproc2/ir/shape_expr.py`）**
- [ ] `SymbolicDim(name, upper: Optional[int])`
- [ ] `ShapeConstraint`：不等式约束集合
- [ ] `ShapeExpr`：支持 `const / dim / add / sub / mul / floordiv / ceildiv / min / max`
- [ ] `UpperBound`：编译时可知的最大值

**Pass（`python/devproc2/compiler/passes/`）**
- [ ] `infer_struct_info.py`：从函数参数 StructInfo 向下传播，推导所有中间值的 shape/dtype/device
- [ ] `dynamic_shape_analyze.py`：建立 symbolic dim 约束图，区分编译时可静态化和必须 runtime 计算的 dim
- [ ] `shape_constraint_verify.py`：验证约束不矛盾，upper bound 合法（如 `B <= 8` 且 `B >= 0`）

**Runtime 插桩**
- [ ] 编译器自动在函数入口插入 shape assert：`assert_le_i64(%B, 8)`

### 验收标准

```python
B = dp.symbolic_dim("B", upper=8)
S = dp.symbolic_dim("S", upper=2048)

@dp.function
def main(x: dp.Tensor[(B, S, 4096), "float16"]):
    y = dp.ops.layernorm(x)
    return y
```

- `y` 的 StructInfo 被正确推导为 `Tensor[(B, S, 4096), float16]`
- 编译产物中包含 `assert S <= 2048` 和 `assert B <= 8`
- 运行时输入 `S=4096` 时抛出 `RuntimeShapeError`

---

## M5：Effect System MVP

**目标**：effect 信息驱动正确的 DCE、内存规划、调度决策，no-output stateful call 成为一等公民。

### 任务清单

**Pass（`python/devproc2/compiler/passes/effect_analyze.py`）**
- [ ] 标注每个 `Call` / `CallDPS` 的 `EffectInfo`
- [ ] 推导规则：
  - `pure`：Call 调用纯函数，无状态读写
  - `read_only`：读取外部状态（如 tokenizer vocab_size）
  - `write(vars)`：明确写入某些 tensor/state
  - `opaque`：有副作用但编译器不理解范围

**DCE（在 `normalize.py` 或单独 pass）**
- [ ] `pure` 的 Call 若结果未被使用 → 可删除
- [ ] `write / opaque` 的 Call → 不可删除（即使结果未使用）
- [ ] `output=None` 的 CallDPS → 不可删除

**Memory Planner 配合（M7 实现，但接口在此定义）**
- [ ] `write(k_cache, v_cache)` → 延长 k_cache/v_cache 的 live range 到当前指令之后
- [ ] `opaque` → 周围所有 tensor live range 保守延长

### 验收标准

```python
@dp.function
def decode(x, k_cache, v_cache, pos):
    q, k, v = dp.ops.qkv_proj(x)
    dp.ops.update_kvcache(k_cache, v_cache, k, v, pos)  # no-output, effect=write
    out = dp.ops.attention_with_cache(q, k_cache, v_cache, pos)
    return out
```

- `update_kvcache` 调用不被 DCE 删除
- `k_cache / v_cache` 的 live range 包含 `attention_with_cache` 调用
- `attention_with_cache` 不被调度到 `update_kvcache` 之前

---

## M6：DPS Lowering MVP

**目标**：将高层函数式 `Call` lower 成 `CallDPS`（绑定具体 kernel 实现），`call_dps_packed` 保持原样。

### 任务清单

**Kernel Registry（`python/devproc2/kernel/register.py`）**
- [ ] `KernelMatchKey`：`op_name + device + dtype + layout + rank`
- [ ] `KernelRegistry.register(key, kernel_spec, priority)`
- [ ] `KernelRegistry.lookup(key)` → 按优先级返回最佳 KernelSpec

**Pass（`python/devproc2/compiler/passes/`）**
- [ ] `kernel_select.py`：遍历所有 `Call`，根据 StructInfo（dtype/device/shape）从 KernelRegistry 选择 kernel
- [ ] `dps_lowering.py`：
  - 对每个 `Call @op(%a, %b)`：
    1. 根据 StructInfo 计算 output shape/dtype/device
    2. 插入 `TensorCreateOp`（`%y = dp.empty(...)`）
    3. 将 `Call` 替换为 `CallDPS @kernel.xxx(inputs=[...], output=%y, effect=write(%y))`
  - `call_dps_packed` 保持原样（callee_kind=packed_func）
  - no-output `CallDPS` 保持原样（effect=write/opaque，output=None）

### 验收标准

高层 IR：
```
%y = call @matmul(%a, %b)
```

DPS lowering 后：
```
%y = dp.empty(shape=[%M, %N], dtype=float16, device=cuda)
call_dps @kernel.matmul(inputs=[%a, %b, %M, %N, %K], output=%y, callee_kind=kernel, effect=write(%y))
```

no-output call 原样保留，`call_dps_packed` 原样保留。

---

## M7：Memory Planning MVP

**目标**：自动插入 `alloc_storage` / `alloc_tensor`，支持 storage reuse（生命区间不重叠的 tensor 共享同一块内存）。

### 任务清单

**单一 Pass（`python/devproc2/compiler/passes/memory_planning.py`）**

`MemoryPlanningPass` 是一个 IR-in-IR-out 的 pass，**内部按固定顺序串行执行四个分析阶段**，不对外暴露中间数据结构。分析结果通过 `PassContext` 传递给后续的 `LowerTensorCreateToAllocPass`。

- [ ] **阶段 A — TensorCreateAnalyze**：收集所有 `TensorCreateOp`（empty/zeros/full/empty_like），记录 shape/dtype/device/消费者 CallDPS；标记 input/output tensor 和 effectful mutable state（k_cache/v_cache）为不可 reuse
- [ ] **阶段 B — LifetimeAnalyze**：线性化 IR，计算每个 tensor 的 `LiveInterval(first_def, last_use)`；`effect=write(v)` 延长 v 的 last_use；`effect=opaque` 保守延长周围所有活跃 tensor
- [ ] **阶段 C — StorageSizeAnalyze**：静态 shape → `size = prod(shape) * dtype.itemsize`；动态 shape → 用 UpperBound 替代各 SymbolicDim 计算 `max_bytes`；向上对齐到 256 bytes
- [ ] **阶段 D — StoragePlan（贪心）**：按 `first_def` 排序，找同 device + size 足够 + live range 不重叠的已有 storage 复用；否则分配新 storage
- [ ] 结果写入 `PassContext("storage_plan", StoragePlan(...))`，**IRModule 本身不被修改**

**相关 pass（`lower_tensor_create_to_alloc.py`）**

- [ ] `requires = ["MemoryPlanningPass"]`
- [ ] 从 `PassContext` 读取 `storage_plan`，将 `TensorCreateOp` 替换为 `alloc_storage + alloc_tensor`
- [ ] `alloc_storage` 节点提升到函数顶部（session 级别，只分配一次）

**不做（MVP 限制）**：复杂 alias、view mutation、inplace op、跨 device reuse、storage escape 后的激进复用。

### 验收标准

`PassContext` 中 `storage_plan` 输出如下形式：

```json
{
  "storage_plan": [
    {"id": 0, "device": "cuda:0", "size_bytes": 67108864, "reused_by": ["tmp0", "tmp3", "tmp7"]},
    {"id": 1, "device": "cuda:0", "size_bytes": 16777216,  "reused_by": ["tmp1", "tmp5"]}
  ]
}
```

`LowerTensorCreateToAllocPass` 执行后 Memory-explicit IR 中出现：

```
%s0 = alloc_storage(size=67108864, alignment=256, device=cuda:0)
%tmp0 = alloc_tensor(%s0, offset=0, shape=[%B, %S, 4096], dtype=float16)
```

至少 2 个 tensor 共享同一 storage；`MemoryPlanningPass` 运行前后 IRModule 内容不变（verifier 可验证）。

---

## M8：VM MVP

**目标**：实现极简 4 指令 VM（call / ret / if / goto），支持 builtin / packed_func / kernel dispatch。

### 任务清单

**VM 格式（`python/devproc2/vm/`）**
- [ ] `Opcode` 枚举：`CALL / RET / IF / GOTO`（仅 4 个）
- [ ] `Instruction` dataclass：`opcode + operands`
  - CALL: `dst_reg, func_idx, arg_regs`（dst_reg=-1 表示无返回值）
  - RET: `src_reg`（-1 表示无返回值）
  - IF: `cond_reg, true_offset, false_offset`
  - GOTO: `offset`
- [ ] `FunctionEntry`：`name + kind (vm_func/builtin/packed_func/kernel) + instructions`
- [ ] `Executable`：`function_table + instructions + constants`
- [ ] `serializer.py`：Executable ↔ 二进制字节流

**C++ VM 执行引擎（`runtime/src/vm.cc`）**
- [ ] `VMFrame`：func_idx + pc + reg_base
- [ ] `VMState`：executable + frames stack + register file（`std::vector<VMValue>`）+ default_streams
- [ ] 执行循环：根据 opcode dispatch
- [ ] Callee dispatch：
  - `vm_func` → 压栈新 VMFrame，跳转
  - `builtin` → 查全局 builtin 函数表，调用
  - `packed_func` → 查 PackedFuncRegistry，调用
  - `kernel` → 查 KernelRegistry，launch（M11 实现，此处预留接口）
- [ ] `VMCodegenPass`（`python/devproc2/compiler/passes/vm_codegen.py`）：Memory-explicit IR → Executable

### 验收标准

VM 能执行如下 bytecode 序列：

```
call -1, @vm.builtin.alloc_storage, [...]      # 分配 storage
call %r0, @vm.builtin.alloc_tensor, [...]       # 创建 tensor
call -1, @kernel.relu, [%input, %r0]            # launch kernel（M11 前用 mock）
ret %r0
```

`if / goto` 控制流分支正确执行（真/假分支各执行其 block）。

---

## X1：Runtime Device API MVP

**目标**：建立 CPU / CUDA device 统一抽象，VM 不直接调用任何 CUDA API，所有设备操作经过 DeviceAPI 接口。

### 任务清单

**接口定义（`runtime/include/devproc2/runtime/`）**

```cpp
// device_api.h — 复用 DLPack 的 DLDevice，不再自定义 Device struct
#include <dlpack/dlpack.h>

using Device = DLDevice;  // { device_type: DLDeviceType, device_id: int }

class DeviceAPI {
public:
    virtual void* Alloc(Device dev, size_t nbytes, size_t alignment) = 0;
    virtual void  Free(Device dev, void* ptr) = 0;

    // 接受 DLTensor*，可直接传 TensorObj::dl()，与 TVM DeviceAPI 对齐
    virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, void* stream) = 0;

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
};

// stream.h
class StreamObj : public Object {
public:
    Device device;
    void* handle;
};
using Stream = ObjectRef;  // holds StreamObj
```

**实现**
- [ ] `CPUDeviceAPI`：Alloc=`aligned_alloc`，Free=`free`，CopyDataFromTo=`memcpy`，Stream=noop
- [ ] `CUDADeviceAPI`：Alloc=`cudaMalloc`，Free=`cudaFree`，CopyDataFromTo=`cudaMemcpyAsync`，Stream=`cudaStream_t`
- [ ] `MemoryPool` 通过 DeviceAPI 分配：`api->Alloc(device, nbytes, alignment)`
- [ ] `Tensor::FromExternalBuffer` 和 `Tensor::FromDLPack`（已在 M1 TensorObj 中定义，此处确保 StoragePlan 不为其分配 storage）
- [ ] VMState 维护 `default_streams: unordered_map<Device, Stream, DeviceHash>`

### 验收标准

- VM 源码中搜索不到 `cudaMalloc / cudaFree / cudaMemcpy`（只出现在 `cuda_device_api.cc`）
- CUDA tensor 分配 / H2D 拷贝 / D2H 拷贝 / stream sync 全部通过 DeviceAPI 接口正常工作
- `Tensor::FromDLPack` 从 torch DLPack capsule 创建的 tensor 可以传给 VM；StoragePlan 不为其分配 storage（`owns_data=false` 标志正确）
- `DeviceAPI::CopyDataFromTo(from->dl(), to->dl(), stream)` 调用能正确完成 GPU 间内存拷贝

---

## X2：Runtime Shape Builtin MVP

**目标**：支持动态 shape 在 runtime 通过 builtin 函数计算（grid 计算、output shape 计算、upper bound 检查）。

### 任务清单

**Builtin 函数（`runtime/src/builtins.cc`）**
- [ ] `vm.builtin.shape_of(tensor)` → ShapeTuple
- [ ] `vm.builtin.get_shape_dim(shape_tuple, idx)` → Int
- [ ] `vm.builtin.make_shape(d0, d1, ...)` → ShapeTuple
- [ ] `vm.builtin.add_i64 / sub_i64 / mul_i64`
- [ ] `vm.builtin.floordiv_i64 / ceildiv_i64`
- [ ] `vm.builtin.min_i64 / max_i64`
- [ ] `vm.builtin.eq_i64 / lt_i64 / le_i64 / gt_i64 / ge_i64`
- [ ] `vm.builtin.assert_le_i64(val, bound, msg)`（失败时抛 RuntimeShapeError）

**Pass（`python/devproc2/compiler/passes/shape_expr_lowering.py`）**
- [ ] 将 IR 中的 `ShapeExpr` 节点 lower 为 VM builtin call 序列
- [ ] 编译器在函数入口插入 `assert_le_i64` 检查 upper bound

### 验收标准

动态输入 `Tensor[(B, S, 4096), float16]`（B=2, S=512）：

```
%shape = call @vm.builtin.shape_of(%x)         → [2, 512, 4096]
%B = call @vm.builtin.get_shape_dim(%shape, 0)  → 2
%S = call @vm.builtin.get_shape_dim(%shape, 1)  → 512
%grid_x = call @vm.builtin.ceildiv_i64(%S, 16)  → 32
call @vm.builtin.assert_le_i64(%S, 2048, "S exceeds upper bound")  → ok
```

`S=4096` 时：`assert_le_i64` 抛出 `RuntimeShapeError: S=4096 exceeds upper bound 2048`。

---

## X3：Kernel Launch Expression MVP

**目标**：支持动态 shape kernel launch grid 表达式（`grid = (ceildiv(M, BM), ceildiv(N, BN), B)`），同一 cubin 在不同 shape 下正确 launch。

### 任务清单

**Kernel Metadata**
- [ ] `KernelSpec` 中增加 `grid_expr: List[ShapeExpr]`（每个维度一个表达式）
- [ ] 示例：`grid_expr = [ceildiv(M, 16), ceildiv(N, 16), B]`
- [ ] ABI JSON 中记录 grid expression

**Pass（`python/devproc2/compiler/passes/kernel_launch_expr_lowering.py`）**
- [ ] 将 `grid_expr` lower 为 VM builtin call 序列
- [ ] kernel ABI 中显式传 shape scalar 参数（M / N / K / B / S 等）

**C++ Launcher（`runtime/src/cuda/cuda_module.cc`）**
- [ ] `CUDAKernelLauncher.Launch(kernel_obj, args, grid, block, stream)`
- [ ] grid / block 从 VMValue Int 中读取
- [ ] `cuLaunchKernel(func, grid_x, grid_y, grid_z, block_x, block_y, block_z, 0, stream, params, 0)`

### 验收标准

同一 `matmul.cubin`：
- `M=128, N=128, K=256` → `grid=(8, 8, 1)`，结果正确
- `M=256, N=512, K=128` → `grid=(16, 32, 1)`，结果正确

`ceildiv(S, BLOCK_M)` 在 runtime 动态计算，不同 S 下结果均正确。

---

## M9：ABI + Artifact MVP

**目标**：生成稳定、自描述、可独立加载的编译产物包（`.devproc2_module`）。

### 任务清单

**Artifact 结构**

```
build/<model_name>/
  manifest.json           # 包元数据（名称、版本、构建时间、target arch）
  abi.json                # ABI 描述（版本、input/output contract、shape 约束）
  executable.vm           # VM bytecode 二进制
  constants/
    const_0.bin           # 权重/常量 blobs
  kernels/
    relu_fp16.cubin
    matmul_fp16.cubin
    relu_fp16.ptx         # 可选，用于调试
  metadata/
    function_table.json
    kernel_table.json
    packed_func_table.json
    storage_plan.json
    shape_constraints.json
```

**ABI JSON 示例**

```json
{
  "devproc_abi_version": "0.1",
  "vm_bytecode_version": "0.1",
  "kernel_calling_convention": "dps_kernel_v1",
  "packed_func_calling_convention": "dps_packed_v1",
  "target": "cuda",
  "target_arch": "sm_80",
  "inputs": [
    {"name": "x", "dtype": "float16", "shape": ["B", "S", 4096]}
  ],
  "outputs": [
    {"name": "out", "dtype": "float16", "shape": ["B", "S", 4096]}
  ],
  "shape_constraints": {
    "B": {"upper": 8},
    "S": {"upper": 2048}
  },
  "required_packed_funcs": ["runtime.tokenizer.encode"]
}
```

**Pass（`python/devproc2/compiler/passes/`）**
- [ ] `emit_executable.py`：序列化 VM bytecode + function_table → `executable.vm`
- [ ] `emit_abi.py`：生成 abi.json + manifest.json + metadata/*.json
- [ ] `executable.py` 中实现 `serialize() → bytes` 和 `deserialize(bytes) → Executable`

**C++ 加载器（`runtime/src/executable.cc`）**
- [ ] `Executable::Load(path)` → 加载 executable.vm + 所有 metadata
- [ ] ABI 版本检查（主版本不兼容时报错）
- [ ] 缺失 packed_func 时明确报错

### 验收标准

- `Executable::Load` 加载产物时：ABI 版本不匹配 → 明确错误信息
- 产物中 `required_packed_funcs` 包含未注册的函数时报错：`PackedFunc 'runtime.tokenizer.encode' is required but not registered.`
- 同一产物在不同进程中加载，执行结果一致

---

## M10：PackedFunc + call_dps_packed MVP

**目标**：支持 tokenizer 等 runtime C++ 函数通过 PackedFunc registry 从 VM 调用。

### 任务清单

**C++ PackedFunc（`runtime/include/devproc2/runtime/packed_func.h`）**

```cpp
class PackedArgs {
public:
    int size() const;
    VMValue operator[](int i) const;
    VMValue& operator[](int i);
};

class PackedFuncObj : public Object {
public:
    std::function<void(PackedArgs)> body;
    void Call(PackedArgs args) { body(args); }
};

// 注册宏
#define DEVPROC_REGISTER_PACKED_FUNC(name) \
    static PackedFuncRegistrar _reg_##__LINE__(name)
```

**PackedFunc Registry（`runtime/src/packed_func.cc`）**
- [ ] 全局 `std::unordered_map<std::string, PackedFunc>` + mutex
- [ ] `PackedFuncRegistry::Register(name, func)`
- [ ] `PackedFuncRegistry::Get(name)` → PackedFunc（找不到返回 nullptr）

**Python DSL（`python/devproc2/frontend/dsl.py`）**
- [ ] `dp.call_dps_packed(name, inputs=[...], output=..., effect="opaque")` → CallDPS(callee_kind=packed_func)

**Tokenizer Mock（`tests/runtime/mock_tokenizer.cc`）**
- [ ] 注册 `runtime.tokenizer.encode`：接收 String text + Tensor output，写入 token ids

### 验收标准

```python
@dp.function
def tokenize(text, max_len):
    tokens = dp.empty((max_len,), dtype="int32", device="cpu")
    dp.call_dps_packed(
        "runtime.tokenizer.encode",
        inputs=[text],
        output=tokens,
        effect="opaque",
    )
    return tokens
```

- tokens 由 caller 创建传入，tokenizer 写入正确 token ids
- `output=None` 的 `call_dps_packed` 不被 DCE 删除

---

## M11：@devproc.kernel + Triton Cubin MVP

**目标**：支持 `@dp.kernel` 注册 Triton kernel，AOT 编译 cubin，VM 调用 `cuLaunchKernel`。

### 任务清单

**DSL（`python/devproc2/frontend/dsl.py`）**
- [ ] `@dp.kernel(op, backend, device, dtype, ...)` 装饰器
- [ ] 解析 kernel 函数签名为 KernelSpec（DPS：最后一个参数为 output，无 output 则 effect 必须声明）

**KernelSpec（`python/devproc2/kernel/kernel_spec.py`）**
- [ ] `KernelSpec`：op_name + backend + device + dtype + layout + rank + grid_expr + abi + triton_fn
- [ ] `KernelABI`：inputs（含 shape scalar 参数）+ output（可 None）+ effect

**KernelRegistry（`python/devproc2/kernel/register.py`）**
- [ ] 按优先级排序的匹配链：用户注册 > devproc2 内置 fused > Triton generated > cuBLAS/cuDNN > 默认 CUDA > 默认 CPU

**Triton AOT Compile（`python/devproc2/compiler/passes/triton_aot_compile.py`）**
- [ ] 调用 `triton.compile(fn, signature, ...)` 生成 cubin
- [ ] 可选保存 ptx（用于调试/inspection）
- [ ] 写入 `build/kernels/<name>.cubin`

**CUDA Loader + Launcher（`runtime/src/cuda/`）**
- [ ] `cuda_module.cc`：`cuModuleLoadData(cubin_data)` + `cuModuleGetFunction`
- [ ] `cuda_kernel.cc`：`CUDAKernelLauncher.Launch(func, args, grid, block, stream)`

**VM Dispatch**
- [ ] KernelRegistry 在 C++ 侧注册（通过 Python binding 初始化）
- [ ] VM `call @kernel.xxx` → 查 KernelRegistry → launch cubin

### 验收标准

```python
@dp.kernel(op="relu", backend="triton", device="cuda", dtype="float16")
def relu_kernel(x, out):
    # triton kernel implementation
    ...

@dp.function
def main(x):
    return dp.ops.relu(x)
```

端到端流程：Python DSL → cubin 编译 → Artifact 打包 → VM 加载 → `cuLaunchKernel` → 输出数值与 PyTorch `torch.relu` 对比误差 < 1e-3。

---

## M12：End-to-End Demo

**目标**：跑通完整的 LLM decode step，覆盖所有 MVP 核心特性。

### Demo 流程

```
text (str)
  ─→ call_dps_packed("runtime.tokenizer.encode")      # M10: PackedFunc
  ─→ dp.ops.embedding(tokens, embed_weight)           # M2: 普通 Call
  ─→ dp.ops.layernorm(x)                              # M4: dynamic shape
  ─→ dp.ops.qkv_proj(x) → (q, k, v)                  # M2: Tuple 多输出
  ─→ update_kvcache(k_cache, v_cache, k, v, pos)      # M5: no-output stateful
  ─→ dp.ops.attention_with_cache(q, k_cache, v_cache) # M8: VM execute
  ─→ matmul_add_silu(attn_out, w, bias)               # M11: Triton cubin
  ─→ output tensor
```

### 覆盖特性清单

- [x] 普通函数式 `Call`（embedding / layernorm）
- [x] `Tuple` 多逻辑输出（qkv_proj）
- [x] `CallDPS`（所有 lowered kernel call）
- [x] `call_dps_packed`（tokenizer）
- [x] no-output stateful call（update_kvcache, output=None, effect=write）
- [x] `EffectInfo` 保护 k_cache/v_cache
- [x] 动态 shape（B / S）
- [x] UpperBound（B <= 8, S <= 2048）
- [x] Memory planning（中间 tensor 共享 storage）
- [x] Storage reuse（至少 3 个 tensor 复用）
- [x] VM 4 指令执行（call / ret / if / goto）
- [x] 稳定 ABI artifact 产物（manifest + abi.json + executable.vm + .cubin）
- [x] C++ Object/ObjectRef 动态类型系统
- [x] DeviceAPI 抽象（无直接 CUDA API 调用）
- [x] Triton cubin AOT 编译与 dynamic grid launch

### 验收标准

- Demo 脚本 `examples/kv_cache_mvp/run.py` 完整执行无报错
- 输出结果与 PyTorch 参考实现误差 < 1e-3（float16 精度）
- `devproc_cli.py inspect build/kvcache_demo/` 能显示 ABI / function table / storage plan
