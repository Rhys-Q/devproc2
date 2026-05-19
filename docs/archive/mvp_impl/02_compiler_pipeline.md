# devproc2 编译 Pass 流水线

共 18 步，从 Python DSL 到 ABI-stable 编译产物（原 21 步，合并了 memory planning 相关的 4 个分析 pass）。

**核心不变量**：`alloc_storage` / `alloc_tensor` 节点**只在第 12 步之后**出现。第 1-11 步的 IR 中禁止出现这两个节点，verifier 应检测并报错。

---

## Pass 设计原则

### 原则一：IR Module in → IR Module out

**所有 Pass 的签名均为**：

```python
class Pass(ABC):
    @abstractmethod
    def run(self, module: IRModule) -> IRModule:
        ...
```

Pass 不能返回纯分析数据结构。分析结果必须以**注解（annotation）**的形式附加到 IRModule 上，或通过 **PassContext** 向后续 pass 传递。

### 原则二：显式依赖声明

Pass 通过 `requires` 字段声明前置依赖。PassManager 在调度时自动检查依赖链，缺少前置 pass 时报错：

```python
class Pass(ABC):
    requires: ClassVar[list[str]] = []       # 前置 pass 名称列表
    invalidates: ClassVar[list[str]] = []    # 运行后使哪些缓存分析结果失效

class StructInfoInferPass(Pass):
    requires = ["NormalizeIRPass", "ControlFlowNormalizePass"]

class DPSLoweringPass(Pass):
    requires = ["KernelSelectPass", "StructInfoInferPass"]

class MemoryPlanningPass(Pass):
    requires = ["DPSLoweringPass", "EffectAnalyzePass"]
```

### 原则三：紧耦合分析 pass 合并

原 [11]-[14]（TensorCreateAnalyze → LifetimeAnalyze → StorageSizeAnalyze → StoragePlan）存在严格的线性依赖：每一步输入是上一步的输出，无法独立运行，也没有被其他 pass 单独使用。**合并为一个 `MemoryPlanningPass`**，内部顺序执行四个阶段，对外暴露统一接口（IR Module in → annotated IR Module out）。

### PassContext 设计

```python
@dataclass
class PassContext:
    kernel_registry: KernelRegistry
    build_config: BuildConfig
    # 各 pass 可以在这里存放跨 pass 共享的分析结果
    _cache: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        self._cache[key] = value
```

Pass 通过 `PassContext` 读写分析结果；IRModule 自身只携带 IR 节点和注解。

---

## 总览

```
Python DSL
  [1] DSL Capture
  [2] High-level IR Build
  [3] NormalizeIRPass
  [4] ControlFlowNormalizePass
  [5] StructInfoInferPass
  [6] DynamicShapeAnalyzePass
  [7] ShapeConstraintVerifyPass
  [8] EffectAnalyzePass
  [9] KernelSelectPass
  [10] DPSLoweringPass
─────────────── 分界线：引入 TensorCreateOp ───────────────
  [11] MemoryPlanningPass   ← 合并原 [11]-[14]，内部 4 阶段串行
─────────────── 分界线：引入 alloc_storage/alloc_tensor ───────────────
  [12] LowerTensorCreateToAllocPass
  [13] ShapeExprLoweringPass
  [14] KernelLaunchExprLoweringPass
  [15] VMCodegenPass
  [16] TritonAOTCompilePass
  [17] ExecutableEmitPass
  [18] ABIEmitPass
Artifact
```

---

## Pass 详细说明

### [1] Python DSL Capture

**文件**：`python/devproc2/frontend/dsl.py`, `frontend/builder.py`

**输入**：带 `@dp.function` 装饰器的 Python 函数对象

**输出**：IRModule（高层 IR，SSA 形式）

**变换内容**：
- 通过 Python `ast` 模块（或 `inspect.getsource`）解析函数体
- `a = dp.ops.matmul(x, w)` → `%a = call @matmul(%x, %w)`
- `if flag:` → 结构化 `If` 节点（此时还未 normalize）
- `for i in dp.range(0, n):` → 结构化 `For` 节点
- `dp.call_dps_packed(name, ...)` → `CallDPS(callee_kind=packed_func)`
- 函数参数 → `Var`（附带用户声明的 `TensorStructInfo`，dynamic dim 用 `SymbolicDim`）

**关键约束**：
- 此阶段不插入 `alloc_storage` / `alloc_tensor`
- 用户写 `y = dp.ops.matmul(a, b)` → IR 是 `%y = call @matmul(%a, %b)`，不是 DPS 形式

---

### [2] High-level IR Build

**文件**：`python/devproc2/frontend/builder.py`

**输入**：Python AST + 函数参数 annotation（`dp.Tensor[(B, S, H), "float16"]`）

**输出**：规范的 SSA IRModule，所有节点类型已确定

**变换内容**：
- 解析参数 annotation 为 `TensorStructInfo`
- 将 Python 字面量转为 `Constant` 节点
- `q, k, v = dp.ops.qkv_proj(x)` → `%qkv = call @qkv_proj(%x); %q = tuple_get_item(%qkv, 0); ...`
- `@dp.kernel` 装饰的函数不进入 IRModule，转为 `KernelSpec` 注册到 `KernelRegistry`

---

### [3] NormalizeIRPass

**文件**：`python/devproc2/compiler/passes/normalize.py`

**输入**：初始 IRModule

**输出**：规范化的 IRModule

**变换内容**：
- 规范化 SSA 变量命名（`%0, %1, %2, ...`）
- 确保每个 `Var` 只被定义一次
- 将 Python `None` 返回值统一为 `Return(Constant(None))`
- 常量折叠（编译时可求值的算术表达式）
- 死代码预扫描（纯函数结果未使用的 `Call` 标记为候选删除，effect 系统在 M8 再确认）

---

### [4] ControlFlowNormalizePass

**文件**：`python/devproc2/compiler/passes/control_flow_normalize.py`

**输入**：可能含 `elif` 的 IRModule

**输出**：`elif` 全部展平为嵌套 `if` 的 IRModule

**变换内容**：

```python
# 输入
if a:   ...
elif b: ...
else:   ...

# 输出
if a:
    ...
else:
    if b: ...
    else: ...
```

- `For` body 内对外部变量的重新赋值 → 转为 `iter_args`（loop-carried variable）
- 检测循环不变量（可提升到 loop 外）

---

### [5] StructInfoInferPass

**文件**：`python/devproc2/compiler/passes/infer_struct_info.py`

**输入**：IRModule（部分 Var 无 StructInfo）

**输出**：所有 Var 都有完整 `TensorStructInfo`（shape + dtype + device）的 IRModule

**变换内容**：
- 从函数入参 `TensorStructInfo` 向下传播
- 每个 op 注册输出 shape 推导规则，例如：
  - `matmul(A:[M,K], B:[K,N])` → `output:[M,N]`
  - `concat([A:[M,H], B:[M,H]], axis=1)` → `output:[M,2H]`
  - `layernorm(x:[B,S,H])` → `output:[B,S,H]`
- 动态 dim 保持符号形式（`SymbolicDim`），不强制为常量
- Tuple 的 StructInfo = `TupleStructInfo([elem_struct_infos...])`

**失败情形**：shape 推导失败（如 matmul 维度不匹配）→ 编译期报错。

---

### [6] DynamicShapeAnalyzePass

**文件**：`python/devproc2/compiler/passes/dynamic_shape_analyze.py`

**输入**：含 `SymbolicDim` 的 IRModule

**输出**：标注了 shape 类别（compile-time-static / runtime-dynamic）的 IRModule

**变换内容**：
- 建立 symbolic dim 依赖图
- 标注哪些 dim 可在编译时静态化（如 `H=4096` 是常量 dim）
- 标注哪些 dim 必须 runtime 计算（如 `B=shape_of(x)[0]`）
- 对每个运算（add/mul/ceildiv）推导是否可 compile-time 常量折叠

---

### [7] ShapeConstraintVerifyPass

**文件**：`python/devproc2/compiler/passes/shape_constraint_verify.py`

**输入**：含 `SymbolicDim + UpperBound + ShapeConstraint` 的 IRModule

**输出**：验证通过的 IRModule（或报错）

**变换内容**：
- 检查 upper bound 合法性：`upper >= 1`，`upper >= 任何已知 lower bound`
- 检查约束不矛盾（如不能同时有 `B <= 4` 和 `B >= 8`）
- 检查 shape 算术不溢出（`max_bytes < 2^63`）
- 在函数入口插入 shape assert 插桩节点（`RuntimeShapeAssert`），在 X2 步 lower 为 `assert_le_i64`

---

### [8] EffectAnalyzePass

**文件**：`python/devproc2/compiler/passes/effect_analyze.py`

**输入**：IRModule（Call/CallDPS 无 effect 标注）

**输出**：每个 Call/CallDPS 都有 `EffectInfo` 标注的 IRModule

**变换内容**：

| 类型 | 规则 |
|---|---|
| `pure` | 调用纯函数（matmul/relu/add 等）|
| `read_only` | 读取外部状态但不修改（tokenizer.vocab_size）|
| `write(vars)` | 明确写入 vars（update_kvcache → write(k_cache, v_cache)）|
| `opaque` | 用户显式标注 opaque 的 call_dps_packed |

**副作用**：
- `write/opaque` 的 Call → 不可 DCE（即使结果未使用）
- `write(vars)` → 延长 vars 的 live range（M7 使用）
- `output=None` 的 CallDPS → 强制保留

---

### [9] KernelSelectPass

**文件**：`python/devproc2/compiler/passes/kernel_select.py`

**输入**：有完整 StructInfo 的 IRModule

**输出**：每个 `Call @op(...)` 绑定了 `KernelSpec` 的 IRModule

**变换内容**：
- 对每个 `Call`，构造 `KernelMatchKey`（op_name + device + dtype + layout + rank）
- 查询 `KernelRegistry.lookup(key)`
- 将选中的 `KernelSpec` 附加到 `Call` 节点的 attrs 中
- 找不到 kernel 时编译报错：`No kernel found for op 'matmul' on device cuda with dtype float16`

**匹配优先级**：
1. 用户注册的 shape-specialized kernel
2. devproc2 内置 fused kernel
3. Triton generated kernel
4. cuBLAS / cuDNN wrapper
5. 默认 CUDA kernel
6. 默认 CPU kernel

---

### [10] DPSLoweringPass

**文件**：`python/devproc2/compiler/passes/dps_lowering.py`

**输入**：绑定了 KernelSpec 的 IRModule

**输出**：所有普通 `Call @op(...)` 被替换为 `TensorCreateOp + CallDPS` 的 IRModule

**变换内容**：

```
# 输入
%y = call @matmul(%a, %b)   # StructInfo: Tensor[(M,N), float16]

# 输出
%M = shape_of(%a)[0]
%K = shape_of(%a)[1]
%N = shape_of(%b)[1]
%y = dp.empty(shape=[%M, %N], dtype=float16, device=cuda)
call_dps @kernel.matmul(
    inputs=[%a, %b, %M, %N, %K],
    output=%y,
    callee_kind=kernel,
    effect=write(%y)
)
```

- **不改动** `call_dps_packed`（callee_kind=packed_func）
- **不改动** no-output CallDPS（output=None）
- kernel ABI：inputs 中显式传 shape scalar 参数（M/N/K/B/S 等），而不是让 kernel 自己解析 Tensor shape metadata

---

### [11] MemoryPlanningPass

**文件**：`python/devproc2/compiler/passes/memory_planning.py`

**requires**：`["DPSLoweringPass", "EffectAnalyzePass"]`

**输入**：含 `TensorCreateOp` 且每个 CallDPS 已有 EffectInfo 的 IRModule

**输出**：IRModule，并在 `PassContext` 中写入 `storage_plan: StoragePlan`

**合并原因**：内部四个阶段（TensorCreateAnalyze → LifetimeAnalyze → StorageSizeAnalyze → StoragePlan）存在严格线性依赖，没有被其他 pass 单独引用，不需要对外暴露中间数据结构。合并后对外只有一个 IR-in-IR-out 接口，内部实现细节隐藏。

**内部阶段**（顺序不可打乱）：

**阶段 A — TensorCreateAnalyze**：
- 收集所有 `TensorCreateOp`（empty / zeros / full / empty_like）
- 记录每个 TensorCreateOp 的：shape（ShapeExpr）、dtype、device、消费它的 CallDPS
- 标记 input tensor、output tensor、effectful mutable state（k_cache / v_cache）→ 不参与 reuse

**阶段 B — LifetimeAnalyze**：
- 将 IR 线性化为指令序列（控制流：if/for 取最大 live range）
- 计算每个 tensor 的 `LiveInterval(first_def, last_use)`
- `effect=write(k_cache, v_cache)` → 延长 k_cache/v_cache 的 `last_use` 到该 call 之后
- `effect=opaque` → 保守延长周围所有活跃 tensor 的 `last_use`

**阶段 C — StorageSizeAnalyze**：
- 静态 shape：`size = shape.prod() * dtype.itemsize`
- 动态 shape：使用 UpperBound 替代各 SymbolicDim：`max_bytes = upper(B) * upper(S) * H * sizeof(dtype)`
- 向上对齐到 256 bytes

**阶段 D — StoragePlan（贪心）**：
- 将 tensor 按 `first_def` 排序
- 对每个 tensor，找已有 storage 满足：同 device + size 足够 + live range 不重叠 → 复用
- 否则分配新 storage
- 不复用：input tensor、output tensor、effectful mutable state、external buffer（`owns_data=false`）

**输出（写入 PassContext）**：

```python
@dataclass
class StoragePlan:
    entries: list[StorageEntry]

@dataclass
class StorageEntry:
    id: int
    device: Device
    size_bytes: int       # 已对齐的 max_bytes
    alignment: int
    reused_by: list[str]  # TensorCreateOp 的 var 名称
```

**IRModule 不变**：此 pass 不修改 IR 节点，只在 PassContext 中写入 `storage_plan`。

---

### [12] LowerTensorCreateToAllocPass

**文件**：`python/devproc2/compiler/passes/lower_tensor_create_to_alloc.py`

**requires**：`["MemoryPlanningPass"]`

**输入**：含 `TensorCreateOp` 的 IRModule（PassContext 中已有 `storage_plan`）

**输出**：所有 `TensorCreateOp` 替换为 `alloc_storage + alloc_tensor` 的 Memory-explicit IRModule

**变换内容**：

```
# 输入
%y = dp.empty(shape=[%M, %N], dtype=float16, device=cuda)

# 输出
# storage 节点提升到函数顶部（session 级别，只分配一次）
%s0 = alloc_storage(size=67108864, alignment=256, device=cuda:0)
...
# tensor view 在原位置
%y = alloc_tensor(%s0, offset=0, shape=[%M, %N], dtype=float16)
```

规则：
- `alloc_storage` 节点提升到函数顶部
- 共享同一 storage 的多个 tensor 引用同一 `alloc_storage` 结果
- **这是第一步允许 `alloc_storage` / `alloc_tensor` 出现的 pass**

---

### [13] ShapeExprLoweringPass

**文件**：`python/devproc2/compiler/passes/shape_expr_lowering.py`

**输入**：Memory-explicit IR（含 ShapeExpr 节点）

**输出**：所有 ShapeExpr 被替换为 `vm.builtin.*` call 序列的 IR

**变换内容**：

```
# 输入（ShapeExpr 形式）
alloc_tensor(%s0, offset=0, shape=[B, S, 4096], dtype=float16)

# 输出（builtin call 展开）
%shape_x = call @vm.builtin.shape_of(%x)
%B = call @vm.builtin.get_shape_dim(%shape_x, 0)
%S = call @vm.builtin.get_shape_dim(%shape_x, 1)
call @vm.builtin.assert_le_i64(%B, 8)
call @vm.builtin.assert_le_i64(%S, 2048)
%shape_y = call @vm.builtin.make_shape(%B, %S, 4096)
alloc_tensor(%s0, offset=0, shape=%shape_y, dtype=float16)
```

- UpperBound assert 在此步插入
- 常量 dim 直接内联（不生成 builtin call）

---

### [14] KernelLaunchExprLoweringPass

**文件**：`python/devproc2/compiler/passes/kernel_launch_expr_lowering.py`

**输入**：Memory-explicit IR（CallDPS kernel 含 grid_expr 元数据）

**输出**：每个 kernel launch 前插入了 grid 计算序列的 IR

**变换内容**：

```
# 输入（grid_expr = [ceildiv(S, 16), B, 1]）
call_dps @kernel.matmul(inputs=[%a, %b, %M, %N, %K], output=%y, ...)

# 输出
%grid_x = call @vm.builtin.ceildiv_i64(%S, 16)
%grid_y = %B  # 直接引用
call_dps @kernel.matmul(
    inputs=[%a, %b, %M, %N, %K, %grid_x, %grid_y, 1],  # grid 作为 hidden args
    output=%y, ...
)
```

kernel ABI 中 grid/block 作为末尾的隐藏参数传入 launcher。

---

### [15] VMCodegenPass

**文件**：`python/devproc2/compiler/passes/vm_codegen.py`

**输入**：Memory-explicit IR（所有 ShapeExpr 和 grid expr 已展开）

**输出**：`Executable`（VM function_table + 4 指令字节码序列 + constants）

**变换内容**：
- 为每个 IR Function 生成 VM function entry
- 每条 IR binding → 一条或多条 VM 指令
- `Call` → `CALL dst_reg, func_idx, arg_regs`
- `alloc_storage / alloc_tensor` → `CALL _ / CALL reg, @vm.builtin.alloc_storage/alloc_tensor, [...]`
- `If` → `IF cond_reg, true_offset, false_offset` + 分支指令序列 + `GOTO end`
- `For` → 循环计数 + `GOTO` 回跳
- `Return` → `RET src_reg`
- 构建 function_table（函数名 → index，包含 callee_kind）

---

### [16] TritonAOTCompilePass

**文件**：`python/devproc2/compiler/passes/triton_aot_compile.py`

**输入**：KernelRegistry 中注册的所有 `KernelSpec(backend=triton)`

**输出**：`.cubin` 二进制 + 可选 `.ptx`，写入 `build/kernels/`

**变换内容**：
- 对每个 Triton kernel spec，调用 `triton.compile(fn, signature, constants, num_warps, ...)`
- 生成 `<kernel_name>.cubin`（执行主产物）
- 生成 `<kernel_name>.ptx`（可选，用于调试/inspection）
- 记录 kernel metadata（name + grid_expr + ABI + cubin path）

**注意**：runtime 不编译 Triton。Runtime 只负责加载 cubin 和 launch。

---

### [17] ExecutableEmitPass

**文件**：`python/devproc2/compiler/passes/emit_executable.py`

**输入**：`Executable` Python 对象

**输出**：`executable.vm` 二进制文件

**变换内容**：
- 序列化 function_table（每个函数的名称、kind、指令起始/长度）
- 序列化指令序列（定长编码）
- 序列化 constants（packed binary blobs）
- 写入版本 header（`vm_bytecode_version=0.1`）

---

### [18] ABIEmitPass

**文件**：`python/devproc2/compiler/passes/emit_abi.py`

**输入**：IRModule + StoragePlan + KernelMetadata + ShapeConstraints

**输出**：artifact 目录下所有 JSON 元数据文件

**生成文件**：

```
manifest.json          # 包名、版本、构建时间、target arch
abi.json               # input/output ABI、shape 约束、所需 packed_funcs
metadata/
  function_table.json  # 函数名 → index → kind
  kernel_table.json    # kernel 名 → cubin 文件 → grid/block 信息
  packed_func_table.json  # 所有被调用的 packed_func 名称
  storage_plan.json    # storage 分配计划
  shape_constraints.json  # symbolic dim 约束
```

---

## Pass Pipeline 调用示例

```python
from devproc2.compiler.pipeline import PassPipeline, PassContext
from devproc2.compiler import passes

ctx = PassContext(
    kernel_registry=KernelRegistry.global_(),
    build_config=BuildConfig(target="cuda", target_arch="sm_80"),
)

pipeline = PassPipeline(ctx, [
    passes.NormalizeIRPass(),
    passes.ControlFlowNormalizePass(),
    passes.StructInfoInferPass(),
    passes.DynamicShapeAnalyzePass(),
    passes.ShapeConstraintVerifyPass(),
    passes.EffectAnalyzePass(),
    passes.KernelSelectPass(),
    passes.DPSLoweringPass(),
    passes.MemoryPlanningPass(),           # 内含 4 个分析阶段
    passes.LowerTensorCreateToAllocPass(),
    passes.ShapeExprLoweringPass(),
    passes.KernelLaunchExprLoweringPass(),
    passes.VMCodegenPass(),
    passes.TritonAOTCompilePass(output_dir="build/kernels"),
    passes.ExecutableEmitPass(output_path="build/executable.vm"),
    passes.ABIEmitPass(output_dir="build"),
])

result_module = pipeline.run(ir_module)
# storage_plan 可从 ctx 取出
storage_plan = ctx.get("storage_plan")
```

PassManager 在运行前自动拓扑排序 `requires` 依赖图；循环依赖时报错。

---

## 关键不变量检查点

在以下 pass 之后运行 `verifier.verify(module)` 断言：

| 检查点 | 断言 |
|---|---|
| 第 3 步后 | 所有 Var 只被定义一次（SSA 正确）|
| 第 5 步后 | 所有 Var 都有 StructInfo（shape/dtype/device）|
| 第 10 步后 | IR 中无 `Call @op(...)` 形式（已全部 lower）；无 `alloc_storage/alloc_tensor` |
| 第 11 步后 | IR 不变（MemoryPlanningPass 不修改 IR）；PassContext 中 `storage_plan` 已写入 |
| 第 12 步后 | 无 `TensorCreateOp`；所有 storage 引用合法；alloc_storage 在函数顶部 |
| 第 13 步后 | 无 `ShapeExpr` 算术节点；只有 Int 常量或 `vm.builtin.*` 调用结果 |
| 第 15 步后 | IR 只包含 4 种 VM 指令（CALL / RET / IF / GOTO）|
