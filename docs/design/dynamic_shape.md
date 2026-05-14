# 动态 Shape 设计文档

本文梳理 devproc2 中动态 shape 的完整处理链路：从编译期的符号表达到运行时的 VM 执行，重点回答以下问题：

1. VMCodegenPass 如何支持动态 shape？
2. Shape 如何从编译期符号映射到 VM 寄存器（所谓"shape 寄存器"）？
3. 内核匹配（kernel match）是否基于 shape？
4. `alloc_storage` / `alloc_tensor` 如何处理动态 shape？

---

## 1. 核心数据结构

### 1.1 PrimExpr — 编译期符号表达式

`python/devproc2/ir/prim_expr.py`

```
PrimExpr
├── IntImm(value: int)                 — 编译期整数常量
├── PrimVar(name, upper=None, sym_id)  — 符号变量，对象同一性决定唯一性
└── Binary(lhs, rhs):
    Add / Sub / Mul / FloorDiv / CeilDiv / Min / Max
    EQ / LT / LE / GT / GE
```

关键设计点：

- **`PrimVar` 用对象同一性（`eq=False`）区分不同的符号**。两个 `PrimVar("B")` 是不同符号，即使名字相同。
- `upper: Optional[int]` 是编译期可知的上界，用于内存规划中保守分配，以及运行时的 assert 插桩。
- `prim_expr_structural_eq()` 用"按 (name, upper) 比较"的结构相等，仅用于跨 pass 重建后的相等性判断（如 storage reuse 判断两个动态 tensor 是否同形）。

### 1.2 TensorStructInfo — 类型标注

```python
@dataclass
class TensorStructInfo(StructInfo):
    shape:  tuple[PrimExpr, ...]  # 每个维度可以是 IntImm 或 PrimVar
    dtype:  str
    device: str
```

`shape` 是 `PrimExpr` 的元组：

- 静态维度：`IntImm(4096)`
- 动态维度：`PrimVar("S", upper=2048)`
- 编译期可约简的组合表达式：`CeilDiv(PrimVar("S"), IntImm(16))`

TensorStructInfo 附着在 `Var`（函数参数）和 `OpResult` 上，随 `InferStructInfoPass` 在 IR 中传播。

---

## 2. 编译 Pass 流水线（动态 shape 视角）

完整的 18 步流水线中，与动态 shape 相关的关键步骤：

```
[1]  DSL Capture             — @dp.function 捕获带 PrimVar 标注的参数
[5]  InferStructInfoPass     — 将 TensorStructInfo（含 PrimVar）向下传播到中间结果
[9]  KernelSelectPass        — kernel 匹配基于 dtype/device，不感知具体 shape 值
[10] DPSLoweringPass         — 用 struct_info.shape（含 PrimVar）创建 TensorCreateOp
[11] MemoryPlanningPass      — 用 PrimVar.upper 保守估算 storage 大小；动态 size 用 PrimExpr
[12] LowerTensorCreateToAllocPass — 产出 AllocStorageOp(size_bytes=PrimExpr) + AllocTensorOp(shape=PrimExpr tuple)
[15] VMCodegenPass           — 内嵌 ShapeExprLoweringPass，将 PrimExpr 物化到 VM 寄存器
```

### 2.1 Pass [1] — DSL Capture

用户通过 `dp.symbolic_dim` 创建 `PrimVar` 并用在函数参数的类型标注上：

```python
B = dp.symbolic_dim("B", upper=8)
S = dp.symbolic_dim("S", upper=2048)

@dp.function
def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
    ...
```

参数 `x` 被创建为 `Var("x", struct_info=TensorStructInfo(shape=(B, S, IntImm(4096)), ...))`.  
`B`、`S` 是具体的 `PrimVar` 对象，其 Python 对象 id 贯穿整个编译流水线。

### 2.2 Pass [5] — InferStructInfoPass

`python/devproc2/compiler/passes/infer_struct_info.py`

维护 `_type_env: dict[Value, StructInfo]`。对 `CallOp`：

1. 如果 result 已有 struct_info（来自标注），直接记录。
2. 否则从第一个参数传播（MVP：element-wise op 结果 shape 等于 arg[0] 的 shape）。

此步骤将 `PrimVar` 组成的 shape 从参数传播到中间计算结果，使后续 pass 能获得完整类型信息。

**当前局限**：binary op（如 matmul）的输出 shape 与 arg[0] 不同，暂不支持自动推导；需要调用者在 CallOp 上显式标注 result_struct_info。

### 2.3 Pass [9/10] — KernelSelectPass / DPSLoweringPass

`python/devproc2/compiler/passes/dps_lowering.py`  
`python/devproc2/kernel/registry.py`

**kernel 匹配不感知具体 shape 值**，只使用 `(op_name, device, input_dtypes)` 三元组作为字典 key。

```python
key = KernelMatchKey(
    op_name=op.callee.lstrip("@"),
    device=si.device,
    input_dtypes=build_input_dtypes(op.args),   # 只提取 dtype，不管 shape
)
spec = registry.lookup(key, sm_arch, call_op)
```

**两级 dispatch**：
- Level 1：O(1) 精确字典查找 `(op_name, device, input_dtypes)`
- Level 2：线性扫描，过滤 SM 算力 + 可选 `match` 谓词（用于 shape/attr 精细匹配）

```python
@dataclass(frozen=True)
class KernelSpec:
    op_name:      str
    device:       str
    input_dtypes: tuple[str, ...]
    kernel_name:  str
    sm_arches:    tuple[int, ...] = ()     # 空 = 任意 SM
    priority:     int = 0
    match:        Optional[Callable[[CallOp], bool]] = None  # 二阶谓词
```

`match` 谓词接收完整的 `CallOp`（含 args 的 struct_info），可以基于 shape 范围等条件进一步筛选，但这是可选的精细化，不是基础 dispatch 机制。

**DPS 展开**：匹配后，对每个 `CallOp`:

1. 插入 `TensorCreateOp(kind=empty, shape=si.shape, dtype=si.dtype, device=si.device)` — `shape` 此时包含 `PrimVar`
2. 将 `CallOp` 替换为 `CallDPSOp(callee=kernel_name, inputs=op.args, output=create_op.result)`

`TensorCreateOp.shape` 直接继承自 `struct_info.shape`，包含原始 `PrimVar` 对象。

### 2.4 Pass [11] — MemoryPlanningPass

`python/devproc2/compiler/passes/memory_planning.py`

**分析阶段，不修改 IR**，只向 PassContext 写入 `StoragePlan`。

动态 shape 的处理分两条路：

#### 2.4.1 StorageSizeAnalyze — 静态估算 vs 动态表达式

```python
def _compute_size_bytes(shape, dtype) -> Optional[int]:
    # 用 PrimVar.upper 替换所有符号维度，计算保守上界（字节）
    # 若任一 PrimVar 无 upper，返回 None（无法静态估算）
    try:
        nbytes = prod(_eval_upper(d) for d in shape) * dtype_itemsize(dtype)
        return _align256(max(nbytes, 1))
    except _NoBound:
        return None

def _compute_size_expr(shape, dtype) -> PrimExpr:
    # 保留符号，构建 PrimExpr 乘积，供运行时动态求值
    # 不做 upper 替换，返回 Mul(d0, Mul(d1, ... IntImm(itemsize)))
    itemsize = dtype_itemsize(dtype)
    result: PrimExpr = IntImm(itemsize)
    for dim in shape:
        result = Mul(dim, result)
    return result
```

两个值分别用于不同目的：
- `size_bytes`（Optional[int]）：用于 storage reuse 的尺寸比较（静态 >= 所需即可复用）
- `size_expr`（PrimExpr）：传入 `AllocStorageOp.size_bytes`，运行时由 VM 动态求值

#### 2.4.2 StoragePlan — storage reuse 对动态 shape 的处理

```python
def accepts(self, ti: TensorInfo) -> bool:
    if ti.device != self.device:
        return False
    if self.size_bytes is not None and ti.size_bytes is not None:
        # 双方均静态：storage 尺寸足够即可
        if self.size_bytes < ti.size_bytes:
            return False
    elif self.size_bytes is None and ti.size_bytes is None:
        # 双方均动态：要求 size_expr 结构相等（PrimVar 按 name+upper 比较）
        if not prim_expr_structural_eq(self.size_expr, ti.size_expr):
            return False
    else:
        # 一静一动：不兼容，不复用
        return False
    # 再检查生命周期不重叠
    return all(not ti.interval.overlaps(iv) for iv in self._intervals)
```

两个动态 tensor 可以共享 storage 的条件：**size_expr 结构相等**（即形状表达式相同）且**生命周期不重叠**。

### 2.5 Pass [12] — LowerTensorCreateToAllocPass

`python/devproc2/compiler/passes/lower_tensor_create_to_alloc.py`

读取 `StoragePlan`，将 `TensorCreateOp` 替换为 `AllocStorageOp + AllocTensorOp`：

```python
AllocStorageOp(
    result_name=f"s{entry.id}",
    size_bytes=entry.size_expr,   # PrimExpr，可能是 Mul(S, Mul(4096, IntImm(2)))
    alignment=entry.alignment,
    device=entry.device,
)

AllocTensorOp(
    result_name=name,
    storage=storage_op.results[0],
    offset=0,
    shape=si.shape,               # tuple[PrimExpr]，包含原始 PrimVar 对象
    dtype=si.dtype,
)
```

`AllocStorageOp.size_bytes` 和 `AllocTensorOp.shape` 都携带 `PrimVar`。这些表达式需要在 VMCodegenPass 阶段物化为具体寄存器操作。

---

## 3. VMCodegenPass 的动态 shape 处理

`python/devproc2/compiler/passes/vm_codegen.py`  
`python/devproc2/compiler/passes/shape_expr_lowering.py`

VMCodegenPass 是整个动态 shape 处理链路的终点，它通过内嵌的 `ShapeExprLoweringPass` 将所有 `PrimExpr` 物化为 VM 寄存器上的指令序列。

### 3.1 _FnCtx — 函数级代码生成上下文

```python
class _FnCtx:
    _value_reg:   dict[int, int]    # id(Value) → 寄存器编号
    instrs:       list[Instruction]
    next_reg:     int               # 寄存器分配计数器
    const_inits:  list[ConstInit]   # 函数入口时预置到寄存器的常量
    prim_lowerer: _PrimExprLowerer  # 由 ShapeExprLoweringPass 设置
```

`_value_reg` 的 key 是 `id(Value)`，包含 `id(PrimVar)`。`ShapeExprLoweringPass.setup_fn` 会把 `PrimVar` 的物化寄存器写入 `_value_reg`，使后续逻辑无需区分对待。

### 3.2 ShapeExprLoweringPass — 函数入口 prologue

`python/devproc2/compiler/passes/shape_expr_lowering.py`

在 `VMCodegenPass._codegen_fn` 中，参数绑定完成后、函数体代码生成开始前调用：

```python
ctx.prim_lowerer = ShapeExprLoweringPass.setup_fn(fn, ctx)
```

`setup_fn` 执行以下操作：

```
对每个带 TensorStructInfo 的 tensor 参数 param：
  如果该参数 shape 中有未处理的 PrimVar：
    r_shape = CALL vm.builtin.shape_of(r_param)         — 提取运行时 ShapeTuple
    对每个未见过的 PrimVar dim (idx, pvar)：
      r_idx   = const(idx)
      r_dim   = CALL vm.builtin.get_shape_dim(r_shape, r_idx)  — 提取单个维度
      bind pvar → r_dim（写入 lowerer._var_reg 和 ctx._value_reg）
      if pvar.upper is not None：
        r_bound = const(pvar.upper)
        r_msg   = const(f"{pvar.name} exceeds upper bound {pvar.upper}")
        CALL vm.builtin.assert_le_i64(r_dim, r_bound, r_msg)   — 运行时断言
返回 _PrimExprLowerer 对象
```

执行后，所有 `PrimVar` 都在 `ctx._value_reg` 中有对应的寄存器，后续 `ctx.reg_of(pvar)` 能正常工作。

生成的字节码序列（对应 `f(x: Tensor[(B, S, 4096), float16])`）：

```
; prologue（函数入口自动插入）
r2 = CALL vm.builtin.shape_of(r0)         ; r0 = param x
r3 = const 0
r4 = CALL vm.builtin.get_shape_dim(r2, r3)   ; r4 = B
r5 = const 8
r6 = const "B exceeds upper bound 8"
     CALL vm.builtin.assert_le_i64(r4, r5, r6)
r7 = const 1
r8 = CALL vm.builtin.get_shape_dim(r2, r7)   ; r8 = S
r9 = const 2048
r10 = const "S exceeds upper bound 2048"
      CALL vm.builtin.assert_le_i64(r8, r9, r10)
; 以下为函数体指令 ...
```

### 3.3 _PrimExprLowerer — 递归 PrimExpr 物化

`setup_fn` 返回的 `_PrimExprLowerer` 用于在需要时将任意 `PrimExpr` 物化为寄存器：

```python
def materialize(expr: PrimExpr) -> int:  # 返回存有该值的寄存器编号
    IntImm(v)   → ctx.reg_for_int(v)         # 常量放入 const_inits
    PrimVar(v)  → _var_reg[id(v)]            # 查已绑定的寄存器
    Add(a, b)   → materialize(a), materialize(b), CALL vm.builtin.add_i64
    Sub(a, b)   → ... CALL vm.builtin.sub_i64
    Mul(a, b)   → ... CALL vm.builtin.mul_i64
    FloorDiv    → ... CALL vm.builtin.floordiv_i64
    CeilDiv     → ... CALL vm.builtin.ceildiv_i64
    Min / Max   → ... CALL vm.builtin.min_i64 / max_i64
```

例：`AllocStorageOp.size_bytes = Mul(PrimVar("S"), Mul(IntImm(4096), IntImm(2)))` 的物化：

```
r_4096 = const 4096
r_2    = const 2
r_tmp1 = CALL vm.builtin.mul_i64(r_4096, r_2)    ; 4096 * 2 = 8192
r_size = CALL vm.builtin.mul_i64(r8, r_tmp1)      ; S * 8192
```

> **注意**：`const_inits` 机制会在帧建立时把常量预置到寄存器，因此 `IntImm` 叶节点不产生额外指令，只产生寄存器槽位。

### 3.4 AllocStorageOp 的动态 size_bytes 处理

```python
def _lower_alloc_storage(self, op: AllocStorageOp, ctx: _FnCtx) -> None:
    if isinstance(op.size_bytes, IntImm):
        size_reg = ctx.reg_for_int(op.size_bytes.value)   # 静态：直接常量
    else:
        size_reg = ctx.prim_lowerer.materialize(op.size_bytes)  # 动态：递归物化

    align_reg = ctx.reg_for_int(op.alignment)
    dev_type, dev_id = parse_device(op.device)
    dtype_reg  = ctx.reg_for_int(dev_type)
    devid_reg  = ctx.reg_for_int(dev_id)

    result_reg = ctx.alloc_reg()
    ctx.bind(op.results[0], result_reg)
    ctx.emit(CALL vm.builtin.alloc_storage(size_reg, align_reg, dtype_reg, devid_reg))
```

C++ 侧的 `vm.builtin.alloc_storage`：

```cpp
// runtime/src/builtins.cc
DEVPROC2_REGISTER_BUILTIN("vm.builtin.alloc_storage")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto nbytes    = static_cast<size_t>(args[0].AsInt());  // 运行时实际值
        auto alignment = static_cast<size_t>(args[1].AsInt());
        DLDevice dev{...};
        void* data = DeviceAPIRegistry::Get(dev_type)->Alloc(dev, nbytes, alignment);
        auto* obj = new StorageObj{...};
        return VMValue::ObjRef(Storage(obj));
    });
```

**动态 storage 的实际大小由运行时寄存器值决定**，编译器只生成计算该值的指令序列。

### 3.5 AllocTensorOp 的动态 shape 处理

```python
def _lower_alloc_tensor(self, op: AllocTensorOp, ctx: _FnCtx) -> None:
    storage_reg = ctx.reg_of(op.storage)
    offset_reg  = ctx.reg_for_int(op.offset)

    shape_regs = []
    for dim in op.shape:
        if isinstance(dim, IntImm):
            shape_regs.append(ctx.reg_for_int(dim.value))
        elif isinstance(dim, PrimVar):
            shape_regs.append(ctx._value_reg[id(dim)])   # setup_fn 已绑定
        else:
            shape_regs.append(ctx.prim_lowerer.materialize(dim))  # 复合表达式

    shape_reg = ctx.alloc_reg()
    ctx.emit(CALL vm.builtin.make_shape(*shape_regs))       # 构建 ShapeTuple

    ctx.emit(CALL vm.builtin.alloc_tensor(
        storage_reg, offset_reg, shape_reg,
        code_reg, bits_reg, lanes_reg
    ))
```

`vm.builtin.make_shape` 在运行时收集每个维度的整数值，构造 `ShapeTuple` 对象。`vm.builtin.alloc_tensor` 用这个 ShapeTuple + Storage 创建 `TensorObj`：

```cpp
DEVPROC2_REGISTER_BUILTIN("vm.builtin.alloc_tensor")
    .set_body([](std::vector<VMValue>& args) -> VMValue {
        auto* shobj = shape_tuple.as<ShapeTupleObj>();
        return VMValue::ObjRef(
            Tensor::FromStorage(storage, offset, shobj->dims, dtype));
    });
```

TensorObj 的 `dl_tensor.shape` 指针直接指向运行时从 shape 寄存器传来的维度值，**alloc_tensor 不做 shape 的编译期解析**。

---

## 4. 运行时 Shape Builtin 一览

`runtime/src/builtins.cc`

| Builtin | 参数 | 返回 | 用途 |
|---------|------|------|------|
| `vm.builtin.shape_of` | Tensor | ShapeTuple | 提取 tensor 的运行时 shape |
| `vm.builtin.get_shape_dim` | ShapeTuple, idx: Int | Int | 提取单个维度 |
| `vm.builtin.make_shape` | d0, d1, ... Int | ShapeTuple | 从寄存器值构建 ShapeTuple |
| `vm.builtin.assert_le_i64` | val, bound: Int, msg: String | Null | shape 上界检查，违反时抛 RuntimeShapeError |
| `vm.builtin.add_i64` | a, b: Int | Int | PrimExpr Add 物化 |
| `vm.builtin.sub_i64` | a, b: Int | Int | PrimExpr Sub 物化 |
| `vm.builtin.mul_i64` | a, b: Int | Int | PrimExpr Mul 物化 |
| `vm.builtin.floordiv_i64` | a, b: Int | Int | PrimExpr FloorDiv 物化 |
| `vm.builtin.ceildiv_i64` | a, b: Int | Int | PrimExpr CeilDiv 物化，`(a+b-1)/b` |
| `vm.builtin.min_i64` | a, b: Int | Int | PrimExpr Min 物化 |
| `vm.builtin.max_i64` | a, b: Int | Int | PrimExpr Max 物化 |
| `vm.builtin.eq_i64` | a, b: Int | Bool | 比较 |
| `vm.builtin.le_i64` | a, b: Int | Bool | 比较（ForOp 循环条件）|
| `vm.builtin.lt_i64` | a, b: Int | Bool | 比较 |
| `vm.builtin.gt_i64` | a, b: Int | Bool | 比较 |
| `vm.builtin.ge_i64` | a, b: Int | Bool | 比较 |

---

## 5. 完整示例：端到端追踪

以下面函数为例追踪每个 pass 的变化：

```python
B = dp.symbolic_dim("B", upper=8)
S = dp.symbolic_dim("S", upper=2048)

@dp.function
def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
    y = dp.ops.layernorm(x)
    return y
```

### Pass [1] — 捕获后 IR

```
%main(%x: Tensor[(B, S, 4096), float16, cuda]) {
  %y = call @layernorm(%x)
  return %y
}
```

`B`, `S` 是具体 Python 对象（PrimVar），携带 upper=8 / upper=2048。

### Pass [5] — InferStructInfo 后

```
%y = call @layernorm(%x)  ; result_struct_info = TensorStructInfo((B, S, 4096), float16, cuda)
```

### Pass [10] — DPS Lowering 后

```
%y   = TensorCreateOp(empty, shape=(B, S, 4096), dtype=float16, device=cuda)
       call_dps @kernel.layernorm_fp16(inputs=[%x], output=%y, effect=opaque)
return %y
```

### Pass [11] — MemoryPlanningPass（passContext 内容）

```json
{
  "storage_plan": [
    {
      "id": 0,
      "device": "cuda",
      "size_bytes": null,
      "size_expr": Mul(B, Mul(S, Mul(4096, 2))),
      "reused_by": ["y"]
    }
  ]
}
```

`size_bytes = null`（B 或 S 无上界则无法静态化；若有上界则 `= _align256(8 * 2048 * 4096 * 2)`）。

### Pass [12] — LowerTensorCreateToAlloc 后

```
%s0  = alloc_storage(size_bytes=Mul(B, Mul(S, Mul(4096, 2))), alignment=256, device=cuda)
%y   = alloc_tensor(%s0, offset=0, shape=(B, S, 4096), dtype=float16)
       call_dps @kernel.layernorm_fp16(inputs=[%x], output=%y, effect=opaque)
return %y
```

此时 IR 中的 `B`, `S` 仍是编译期 `PrimVar` 对象。

### Pass [15] — VMCodegenPass 生成的字节码

```
; === prologue（ShapeExprLoweringPass 插入）===
r1 = const 0
r2 = const 1
r3 = const 8
r4 = const "B exceeds upper bound 8"
r5 = const 2048
r6 = const "S exceeds upper bound 2048"

r7  = CALL vm.builtin.shape_of(r0)            ; r0 = param %x
r8  = CALL vm.builtin.get_shape_dim(r7, r1)   ; r8 = B（运行时值）
      CALL vm.builtin.assert_le_i64(r8, r3, r4)
r9  = CALL vm.builtin.get_shape_dim(r7, r2)   ; r9 = S（运行时值）
      CALL vm.builtin.assert_le_i64(r9, r5, r6)

; === alloc_storage（动态 size_bytes 物化）===
r10 = const 4096
r11 = const 2
r12 = CALL vm.builtin.mul_i64(r10, r11)       ; 4096 * 2 = 8192
r13 = CALL vm.builtin.mul_i64(r9, r12)        ; S * 8192
r14 = CALL vm.builtin.mul_i64(r8, r13)        ; B * S * 8192
r15 = const 256                               ; alignment
r16 = const 2                                 ; kDLCUDA
r17 = const 0                                 ; device_id
r18 = CALL vm.builtin.alloc_storage(r14, r15, r16, r17)  ; → Storage

; === alloc_tensor（动态 shape 物化）===
r19 = CALL vm.builtin.make_shape(r8, r9, r10)  ; make_shape(B, S, 4096)
r20 = const 2                                  ; dtype_code = kDLFloat
r21 = const 16                                 ; dtype_bits
r22 = const 1                                  ; dtype_lanes
r23 = const 0                                  ; offset
r24 = CALL vm.builtin.alloc_tensor(r18, r23, r19, r20, r21, r22)  ; → Tensor %y

; === kernel call ===
      CALL kernel.layernorm_fp16(r0, r24)      ; DPS: x, y

; === return ===
      RET r24
```

---

## 6. "Shape Heap" 的问题

**devproc2 没有独立的 shape heap**。动态 shape 的传递方式是：

1. **编译期**：`PrimVar` 对象通过 Python 对象同一性（`id(pvar)`）贯穿所有 pass，充当符号标识符。
2. **运行时**：`PrimVar` 对应的值存储在普通的 **VM 寄存器**中（`ctx._value_reg[id(pvar)] = reg`）。

寄存器文件就是 shape 的"存储空间"。所有 shape 运算（提取、算术、断言）都是普通的 `CALL` 指令，寄存器是传递媒介。

这与 TVM 中的 shape heap（一段专用内存区域）不同。devproc2 的方式更简单：shape 值和 tensor 值用同一套寄存器机制，没有特殊处理路径。

---

## 7. 复合 Shape 表达式：以 `[B * S, 2]` 为例

前面各节以单个 `PrimVar` 维度（如 `[B, S, 4096]`）为例。本节追踪**复合表达式维度**（如 `B * S`）的完整路径。

### 7.1 IR 表示

```python
B  = PrimVar("B", upper=8)
S  = PrimVar("S", upper=2048)
BS = Mul(B, S)   # 复合表达式，不再是叶节点 PrimVar
```

`AllocTensorOp.shape = (Mul(B, S), IntImm(2))`

`AllocStorageOp.size_bytes` 由 `_compute_size_expr` 构建：

```python
def _compute_size_expr(shape, dtype):
    itemsize = dtype_itemsize(dtype)   # float16 → 2
    result = IntImm(itemsize)
    for dim in shape:
        result = Mul(dim, result)
    return result
# 对 shape=(Mul(B,S), IntImm(2)), dtype=float16：
# = Mul(IntImm(2), IntImm(2))         → step 1: Mul(2, 2) = inner 4
# = Mul(Mul(B,S), Mul(IntImm(2), IntImm(2)))  → step 2
```

最终 `size_expr = Mul(Mul(B, S), IntImm(4))`，这是一棵 PrimExpr 树，根节点是 `Mul`，叶节点含 `PrimVar`。

### 7.2 MemoryPlanningPass 处理

**静态上界估算**（`_compute_size_bytes`）：

```python
_eval_upper(Mul(Mul(B,S), IntImm(4)))
= _eval_upper(Mul(B,S)) * _eval_upper(IntImm(4))
= (B.upper * S.upper) * 4
= 8 * 2048 * 4 = 65536   →  _align256(65536) = 65536
```

`size_bytes = 65536`（静态）。

**StoragePlan**：

```json
{
  "id": 0,
  "device": "cpu",
  "size_bytes": 65536,
  "size_expr": Mul(Mul(B, S), IntImm(4)),
  "reused_by": ["t0"]
}
```

### 7.3 VMCodegenPass 字节码生成

调用路径：

```
_lower_alloc_storage:
  size_bytes = Mul(Mul(B,S), IntImm(4))   ← 非 IntImm
  → ctx.prim_lowerer.materialize(Mul(Mul(B,S), IntImm(4)))
      materialize(Mul(B,S)):
          materialize(B) → r3   ← 已在 prologue 中绑定
          materialize(S) → r7   ← 已在 prologue 中绑定
          emit: r10 = CALL mul_i64(r3, r7)
          return r10
      materialize(IntImm(4)) → r11   ← const_init
      emit: r12 = CALL mul_i64(r10, r11)
      return r12
  → size_reg = r12

_lower_alloc_tensor:
  for dim in (Mul(B,S), IntImm(2)):
    Mul(B,S) → ctx.prim_lowerer.materialize(Mul(B,S))
        materialize(B) → r3
        materialize(S) → r7
        emit: r18 = CALL mul_i64(r3, r7)   ← 再次计算！
        return r18
    IntImm(2) → ctx.reg_for_int(2) = r19
  emit: r20 = CALL make_shape(r18, r19)
```

完整生成的指令序列（`B=2, S=4` 时运行时值如注释）：

```
; === prologue ===
[ 0] r1  = shape_of(r0)                     ; r0 = param x
[ 1] r3  = get_shape_dim(r1, r2=0)          ; r3 = B = 2
[ 2]       assert_le_i64(r3, r4=8, r5=msg)
[ 3] r7  = get_shape_dim(r1, r6=1)          ; r7 = S = 4
[ 4]       assert_le_i64(r7, r8=2048, r9=msg)

; === alloc_storage: size = B*S*4 ===
[ 5] r10 = mul_i64(r3, r7)                  ; r10 = B*S = 8
[ 6] r12 = mul_i64(r10, r11=4)              ; r12 = B*S*4 = 32
[ 7] r16 = alloc_storage(r12, r13=256, r14=1, r15=0)

; === alloc_tensor: shape = (B*S, 2) ===
[ 8] r18 = mul_i64(r3, r7)                  ; r18 = B*S = 8  ← 重复计算
[ 9] r20 = make_shape(r18, r19=2)           ; ShapeTuple(8, 2)
[10] r24 = alloc_tensor(r16, r17=0, r20, ...)

[11] RET r24
```

运行时 `result.shape = (8, 2)`，符合预期（`B*S = 2*4 = 8`）。

### 7.4 重复物化问题

指令 [5] 和 [8] 都计算了 `B*S`，因为 `_PrimExprLowerer.materialize` **没有缓存**：

```python
def materialize(self, expr: PrimExpr) -> int:
    # 每次调用都可能 emit 新指令，没有 memoization
    ...
    lhs_reg = self.materialize(expr.lhs)
    rhs_reg = self.materialize(expr.rhs)
    result_reg = ctx.alloc_reg()
    ctx.emit(CALL builtin(lhs_reg, rhs_reg))
    return result_reg   # 每次返回新寄存器
```

对于 MVP，这是可接受的（多余的 `mul_i64` 代价极小）。若需要消除冗余，可在 `_PrimExprLowerer` 中维护 `expr_id → reg` 缓存：

```python
# 扩展方案（未实现）
def materialize(self, expr: PrimExpr) -> int:
    eid = id(expr)   # 对 frozen dataclass，相同结构对象可能 id 不同
    # 更健壮：用 prim_expr_structural_eq 建立缓存 key
    if eid in self._expr_cache:
        return self._expr_cache[eid]
    reg = self._materialize_impl(expr)
    self._expr_cache[eid] = reg
    return reg
```

但注意：`frozen=True` 的 `PrimExpr` 节点在不同 pass 中可能被重建为不同 Python 对象（相同结构、不同 `id()`），所以精确缓存需要用 `prim_expr_structural_eq` 作为相等判断，实现比较复杂，暂未优先。

### 7.5 任意嵌套深度的支持

`materialize` 是递归的，支持任意嵌套深度：

```python
# 例：ceildiv(B*S, 16) 用于计算 grid size
expr = CeilDiv(Mul(B, S), IntImm(16))

materialize(CeilDiv(...)):
    lhs = materialize(Mul(B, S))
        materialize(B) → r_b
        materialize(S) → r_s
        emit: r_bs = mul_i64(r_b, r_s)
    rhs = materialize(IntImm(16)) → r_16
    emit: r_grid = ceildiv_i64(r_bs, r_16)
    return r_grid
```

生成：

```
r_bs   = mul_i64(r_b, r_s)
r_grid = ceildiv_i64(r_bs, r_16)
```

X3（Kernel Launch Grid Expression）正是复用这套机制将 `grid_expr = [ceildiv(M, 16), ceildiv(N, 16), B]` 物化为 kernel launch 参数。

---

## 8. 设计约束与扩展点

### 当前 MVP 约束

| 约束 | 原因 |
|------|------|
| `InferStructInfoPass` 只从 arg[0] 传播（element-wise）| 多输入 shape 推导未实现（matmul 等需要显式标注） |
| 动态 storage reuse 要求 size_expr 结构相等 | 无法比较不同符号表达式的值域 |
| `PrimVar` 必须出现在函数参数的 tensor shape 中 | 其他来源的动态值（如 scalar 参数）暂不支持 |
| `assert_le_i64` 依赖 `upper` 字段存在 | 无 upper 的 PrimVar 不会生成 assert |

### 扩展点

- **X3**：`grid_expr` 中的 `PrimExpr`（如 `ceildiv(S, BLOCK_M)`）通过同一套 `_PrimExprLowerer` 物化，机制复用。
- **M11**：kernel 的 shape scalar 参数（M, N, K）也用 `materialize()` 物化后作为 `CallDPSOp.inputs` 的一部分传入。
- **M4 扩展**：`DynamicShapeAnalyzePass` 可进一步区分哪些维度在 kernel launch 时是静态已知的，减少运行时 shape 提取开销。
