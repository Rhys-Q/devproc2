# devproc2 MVP 实施计划文档

## 1. MVP 定位

devproc2 MVP 的目标不是做一个简单 DSL demo，而是做出一套可持续演进的端侧推理编译系统骨架。

MVP 需要跑通：

```text
Python DSL
  -> High-level IR
  -> StructInfo / Dynamic Shape Inference
  -> Control Flow Normalize
  -> Kernel Selection
  -> DPS Lowering
  -> Tensor Create Analyze
  -> Memory Planning
  -> Explicit Storage IR
  -> VM Codegen
  -> ABI-stable Executable
  -> C++ VM Runtime
```

MVP 的核心目标是验证：

```text
1. 前端 DSL 能自然表达端到端推理逻辑；
2. IR 能表达普通 op、控制流、动态 shape、runtime packed call、stateful call；
3. 中端能自动做 DPS lowering 和 memory planning；
4. 编译产物有稳定 ABI；
5. Runtime 是类似 TVM Relax VM 的 C++ VM；
6. Kernel 能通过 @devproc.kernel 以 DPS 形式注册；
7. Triton kernel 能 AOT 编译成 cubin，并由 VM 加载执行；
8. tokenizer / runtime 函数能通过 call_dps_packed 接入；
9. 动态 shape upper bound 能参与 memory plan；
10. alloc_storage / alloc_tensor 只在中端 pass 后自动插入。
```

---

## 2. 设计原则

### 2.1 前端 IR 保持高层语义

前端 DSL 和前端生成的 IR 不应该出现：

```text
alloc_storage
alloc_tensor
```

这些属于中端 memory planning 之后的低层显式内存 IR。

前端应该出现的是：

```text
Call
CallDPS
If
For
Range
Tuple
TupleGetItem
TensorCreateOp
Return
```

例如用户写：

```python
y = dp.ops.matmul(a, b)
z = dp.ops.silu(y)
return z
```

前端 IR 应该是：

```text
%y = call @matmul(%a, %b)
%z = call @silu(%y)
return %z
```

而不是：

```text
%s0 = alloc_storage(...)
%y = alloc_tensor(%s0, ...)
call @matmul(%a, %b, %y)
```

---

### 2.2 普通 op 保持函数式写法

普通 op 应该自然表达：

```python
y = dp.ops.matmul(a, b)
z = dp.ops.add(y, bias)
out = dp.ops.silu(z)
```

不要强迫用户手写：

```python
y = dp.empty(...)
dp.ops.matmul(a, b, y)
```

原因是：

```text
1. 普通 tensor op 的 shape / dtype / device 应由编译器推导；
2. 用户不应该提前关心 output buffer；
3. 函数式写法更适合图优化；
4. memory planner 的价值不应该泄露到 DSL 层；
5. 代码更接近 PyTorch / Relax / NumPy / JAX 的使用体验。
```

---

### 2.3 `@devproc.kernel` 使用 DPS 签名

`@devproc.kernel` 是实现层，不是高层数学表达。

Kernel 写法应该采用 DPS：

```python
@dp.kernel
def matmul_add_silu(a, b, bias, out):
    ...
```

含义是：

```text
输入：a, b, bias
输出：out，由 caller 提供
kernel 只负责写 out
```

这是合理的，因为 kernel 层需要明确：

```text
1. 输入 buffer；
2. 输出 buffer；
3. memory layout；
4. shape 参数；
5. 是否有 side effect；
6. runtime ABI。
```

---

### 2.4 `call_dps_packed` 显式 DPS

`call_dps_packed` 用于调用 runtime 注册函数。

典型场景：

```text
tokenizer
image decode
image resize
runtime helper
debug/profile function
CPU fallback function
opaque C++ function
```

它必须显式指定 output，因为 runtime 函数不应该偷偷分配 tensor。

示例：

```python
tokens = dp.empty((max_len,), dtype="int32", device="cpu")

dp.call_dps_packed(
    "runtime.tokenizer.encode",
    inputs=[text, tokenizer],
    output=tokens,
)
```

---

### 2.5 `CallDPS` 支持无 output

真实推理系统里存在大量没有返回值、但有副作用的调用。

例如 KV cache update：

```python
dp.ops.update_kvcache(k_cache, v_cache, new_k, new_v, pos)
```

它没有返回值，但会修改 `k_cache` / `v_cache`。

IR 必须支持：

```text
call_dps @update_kvcache(
  inputs=[%k_cache, %v_cache, %new_k, %new_v, %pos],
  output=None,
  effect=write(%k_cache, %v_cache)
)
```

这类 call 不能被 DCE 删除，也不能被随意重排。

---

### 2.6 MVP 不支持原生多输出

MVP 不需要在 IR 层面支持 multiple return values。

统一规则：

```text
1. Call 最多返回一个 value；
2. 多个逻辑结果用 Tuple；
3. CallDPS 最多一个 output；
4. CallDPS 可以没有 output；
5. @devproc.kernel 最多一个 output，或者无 output；
6. call_dps_packed 最多一个 output，或者无 output。
```

例如：

```python
q, k, v = dp.ops.qkv_proj(x)
```

前端可以有语法糖，但 IR 应该是：

```text
%qkv = call @qkv_proj(%x)
%q = tuple_get_item(%qkv, 0)
%k = tuple_get_item(%qkv, 1)
%v = tuple_get_item(%qkv, 2)
```

而不是：

```text
%q, %k, %v = call @qkv_proj(%x)
```

---

### 2.7 VM 指令集保持极简

VM 指令集参考 TVM Relax VM，MVP 只保留四类：

```text
call
ret
if
goto
```

不要把这些都做成独立 opcode：

```text
alloc_storage
alloc_tensor
call_kernel
call_packed
shape_of
assert_shape
```

它们都可以通过 VM `call` 调用 builtin function / packed function / kernel function 完成。

例如：

```text
call @vm.builtin.alloc_storage(...)
call @vm.builtin.alloc_tensor(...)
call @runtime.tokenizer.encode(...)
call @kernel.matmul_add_silu(...)
call @vm.builtin.shape_of(...)
```

VM 的复杂性应该放在：

```text
function table
callee kind
runtime registry
ABI metadata
Object / ObjectRef 动态类型系统
```

而不是 opcode 膨胀。

---

### 2.8 Runtime 需要 C++ 动态类型系统

Runtime 需要支持：

```text
Tensor
Storage
ShapeTuple
String
Int
Float
Bool
Tuple
PackedFunc
Kernel
Object
```

所以必须做类似 TVM 的动态类型系统：

```text
Object
ObjectRef
```

但 MVP 不需要暴露 `ObjectPtr` 概念。

对外只保留：

```cpp
Object
ObjectRef
Tensor
Storage
ShapeTuple
String
Tuple
PackedFunc
Kernel
```

内部可以用 intrusive ref count 或 `shared_ptr` 管理，但 API 层不暴露 `ObjectPtr`。

---

## 3. 总体架构

```text
┌────────────────────────────────────┐
│ Python DSL                          │
│ @dp.function                        │
│ @dp.kernel                          │
│ dp.ops.matmul(a, b)                 │
│ dp.call_dps_packed(..., output=out) │
│ if / elif / else / for / range      │
└──────────────────┬─────────────────┘
                   │
                   v
┌────────────────────────────────────┐
│ High-level IR                       │
│ Call                                │
│ CallDPS                             │
│ If / For / Range                    │
│ Tuple / TupleGetItem                │
│ TensorCreateOp                      │
│ Dynamic Shape / Upper Bound         │
└──────────────────┬─────────────────┘
                   │
                   v
┌────────────────────────────────────┐
│ Middle-end                          │
│ Normalize                           │
│ StructInfo Inference                │
│ Dynamic Shape Analysis              │
│ Effect Analysis                     │
│ Kernel Selection                    │
│ DPS Lowering                        │
│ Tensor Create Analyze               │
│ Lifetime Analysis                   │
│ Storage Planning                    │
└──────────────────┬─────────────────┘
                   │
                   v
┌────────────────────────────────────┐
│ Memory-explicit IR                  │
│ alloc_storage                       │
│ alloc_tensor                        │
│ storage reuse annotation            │
│ explicit destination tensor         │
└──────────────────┬─────────────────┘
                   │
                   v
┌────────────────────────────────────┐
│ VM Lowering                         │
│ call / ret / if / goto              │
│ function table                      │
│ executable.vm                       │
│ ABI metadata                        │
└──────────────────┬─────────────────┘
                   │
                   v
┌────────────────────────────────────┐
│ C++ VM Runtime                      │
│ Object / ObjectRef                  │
│ VMValue                             │
│ PackedFunc Registry                 │
│ Kernel Registry                     │
│ Memory Pool                         │
│ CUDA Cubin Loader                   │
│ Stateful Invoke                     │
└────────────────────────────────────┘
```

---

## 4. IR 设计

### 4.1 IR 分层

devproc2 IR 建议分成三层：

```text
High-level IR
Middle IR
Memory-explicit IR
```

### High-level IR

来自 Python DSL。

这一层包括：

```text
IRModule
Function
Block
Var
Call
CallDPS
If
For
Range
Tuple
TupleGetItem
TensorCreateOp
Return
TensorStructInfo
ShapeExpr
SymbolicDim
UpperBound
EffectInfo
```

这一层不包括：

```text
alloc_storage
alloc_tensor
```

---

### Middle IR

中端优化过程中的 IR。

这一层会完成：

```text
1. control flow normalize；
2. struct info inference；
3. dynamic shape constraint propagation；
4. effect analysis；
5. kernel selection；
6. DPS lowering；
7. tensor create analyze；
8. lifetime analysis；
9. storage planning。
```

---

### Memory-explicit IR

Memory planning 后的 IR。

这一层才允许出现：

```text
alloc_storage
alloc_tensor
```

示例：

```text
%s0 = alloc_storage(size=..., alignment=256, device=cuda)
%y = alloc_tensor(storage=%s0, offset=0, shape=[B, H], dtype=float16)
call_dps @kernel.relu(inputs=[%x], output=%y)
```

---

### 4.2 `Call`

`Call` 表示普通函数式调用。

用于：

```text
1. 调用 IR Function；
2. 调用 high-level op；
3. 调用 builtin op；
4. 返回 Tensor / Scalar / Shape / Tuple / ObjectRef。
```

示例：

```text
%y = call @matmul(%a, %b)
```

Call 规则：

```text
1. Call 最多返回一个 value；
2. 如果逻辑上有多个结果，返回 Tuple；
3. Call 默认是表达式；
4. 如果需要无返回值且有副作用，应该使用 CallDPS。
```

---

### 4.3 `CallDPS`

`CallDPS` 表示 destination-passing style 调用。

用于：

```text
1. call_dps_packed；
2. lowering 后的 kernel call；
3. effectful runtime call；
4. no-output stateful call；
5. 显式 output 的底层调用。
```

结构：

```text
CallDPS {
  callee: Symbol
  callee_kind: kernel | packed_func | builtin | vm_func
  inputs: List[Value]
  output: Optional[Value]
  effect: EffectInfo
  attrs: Dict
}
```

有 output：

```text
call_dps @kernel.relu(
  inputs=[%x],
  output=%y,
  effect=write(%y)
)
```

无 output：

```text
call_dps @kernel.update_kvcache(
  inputs=[%k_cache, %v_cache, %new_k, %new_v, %pos],
  output=None,
  effect=write(%k_cache, %v_cache)
)
```

---

### 4.4 Tuple

MVP 不支持原生多输出。

多输出统一用 Tuple 表达。

DSL：

```python
q, k, v = dp.ops.qkv_proj(x)
```

IR：

```text
%qkv = call @qkv_proj(%x)
%q = tuple_get_item(%qkv, 0)
%k = tuple_get_item(%qkv, 1)
%v = tuple_get_item(%qkv, 2)
```

Tuple 是一个 value，而不是多个 return value。

---

### 4.5 Tensor Create Op

前端创建 tensor 使用：

```python
dp.empty(...)
dp.zeros(...)
dp.full(...)
dp.empty_like(...)
dp.zeros_like(...)
```

前端 IR 示例：

```text
%y = dp.empty(shape=[B, H], dtype=float16, device=cuda)
%z = dp.zeros(shape=[B, H], dtype=float16, device=cuda)
```

这些只是高层 tensor create 语义。

中端之后才 lowering 成：

```text
%s = alloc_storage(...)
%y = alloc_tensor(%s, ...)
call @runtime.fill_zero(%z)
```

---

### 4.6 EffectInfo

IR 必须支持 effect 信息。

否则 no-output call 会被错误删除，stateful call 会被错误重排，memory reuse 也可能不安全。

MVP 推荐四级 effect：

```text
pure
read_only
write
opaque
```

#### pure

无副作用。

```text
%y = call @add(%a, %b)
```

可删除、可重排、可融合。

#### read_only

读取外部状态，但不修改。

```text
%vocab_size = call @tokenizer.vocab_size(%tokenizer)
```

#### write

明确写某些 tensor / state。

```text
call_dps @update_kvcache(
  inputs=[%k_cache, %v_cache, %new_k, %new_v, %pos],
  output=None,
  effect=write(%k_cache, %v_cache)
)
```

#### opaque

有副作用，但编译器不理解具体读写范围。

```text
call_dps @runtime.custom_func(
  inputs=[...],
  output=%out,
  effect=opaque
)
```

MVP 对 opaque 保守处理：

```text
1. 不删除；
2. 不跨越其他 effectful call 重排；
3. 不做激进 memory reuse；
4. 不假设它不会修改输入。
```

---

## 5. Control Flow 设计

> 详细设计见 [docs/design/control_flow.md](control_flow.md)。本节给出核心要点。

### 5.1 设计原则

MVP 采用 **Structured Control Flow + Region + Yield** 方案，不引入 CFG / BasicBlock / φ 节点。

IR 保留嵌套结构：

```text
Function
  Block
    If
      then_region (Block)
      else_region (Block)
    For
      body_region (Block)
```

控制流有两种模式：

**有 SSA result（纯函数式）**：

```text
%y = if %flag {
    %v0 = call @relu(%x)
    yield %v0
} else {
    %v1 = call @silu(%x)
    yield %v1
}
```

**Effect-only（无 result）**：

```text
if %cond {
    call_dps @kernel.update_kvcache(...)
    yield
} else {
    call_dps @kernel.noop(...)
    yield
}
```

这是 devproc2 DPS + stateful call 场景的必需支持。`If.result_structs == []` 时为 effect-only 分支。

---

### 5.2 IR 节点

| 节点 | 说明 |
|---|---|
| `If` | `cond + true_branch(Block) + false_branch(Block) + result_structs` |
| `For` | `loop_var + range(Range) + iter_args + body(Block) + result_structs` |
| `Range` | `start + end + step`（均为 Value） |
| `IterArg` | `var + init`，表达 loop-carried variable |
| `Yield` | Region terminator，`values` 可为空 |

`Yield` 是每个 control-flow region 的终结语句，不等同于 `Return`。

---

### 5.3 典型示例

**If / Elif / Else**（`elif` → nested If）：

```python
if flag:
    y = dp.ops.relu(x)
elif x.shape[0] > 1:
    y = dp.ops.silu(x)
else:
    y = dp.ops.gelu(x)
```

IR（`ControlFlowNormalizePass` 展平后）：

```text
%y = if %flag {
    %v0 = call @relu(%x)
    yield %v0
} else {
    %y2 = if %cond {
        %v1 = call @silu(%x)
        yield %v1
    } else {
        %v2 = call @gelu(%x)
        yield %v2
    }
    yield %y2
}
```

**Loop-carried For**：

```python
acc = init
for i in dp.range(0, n):
    acc = acc + x
```

IR：

```text
%acc_out = for %i in range(0, %n, 1)
           iter_args(%acc_iter = %init) {
    %next = call @add(%acc_iter, %x)
    yield %next
}
```

**Effect-only For**（无 iter_args，无 result）：

```python
for i in dp.range(0, n):
    dp.ops.update_kvcache(k_cache, v_cache, k[i], v[i], i)
```

IR：

```text
for %i in range(0, %n, 1) {
    call_dps @kernel.update_kvcache(
        inputs=[%k_cache, %v_cache, ...],
        output=None,
        effect=write(%k_cache, %v_cache)
    )
    yield
}
```

---

### 5.4 MVP 支持范围

支持：

```text
if / elif / else
for i in dp.range(start, end, step)
loop-carried variable（iter_args）
effect-only if / for
nested if/for
```

暂不支持：

```text
while / break / continue
return inside if/for
for x in iterable（只支持 dp.range）
Python object truthiness
```

---

### 5.5 Lowering 到 VM

结构化控制流 lower 到 VM 4 指令中的 `if / goto`，不引入新 opcode：

```text
If  → IF cond_reg, true_offset, false_offset + GOTO end
For → 循环计数 CALL + IF + GOTO 回跳
```

VM 层没有独立的 `for` 指令。

---

## 6. Dynamic Shape 设计

### 6.1 必须支持的能力

MVP 必须支持完整动态 shape：

```text
SymbolicDim
ShapeExpr
UpperBound
ShapeConstraint
RuntimeShapeValue
RuntimeShapeAssert
```

示例：

```python
B = dp.symbolic_dim("B", upper=8)
S = dp.symbolic_dim("S", upper=2048)

@dp.function
def main(x: dp.Tensor[(B, S, 4096), "float16"]):
    y = dp.ops.layernorm(x)
    return y
```

IR 保留：

```text
Tensor[(B, S, 4096), float16]
where B <= 8, S <= 2048
```

---

### 6.2 Upper Bound 参与 Memory Planning

如果 shape 是：

```text
[B, S, H]
where B <= 8, S <= 2048
```

memory planner 可以计算：

```text
max_bytes = 8 * 2048 * H * sizeof(dtype)
```

实际运行时：

```text
B = 1
S = 512
```

tensor view 使用实际 shape，但 storage 按 upper bound 分配和复用。

---

### 6.3 Runtime Shape Assert

编译器必须插入 shape assert：

```text
assert B <= 8
assert S <= 2048
```

如果实际输入超过 upper bound，runtime 必须报错。

---

## 7. Memory Planning 设计

### 7.1 Memory Planning 的位置

Memory planning 不属于前端。

前端：

```text
%y = call @matmul(%a, %b)
```

中端 DPS lowering：

```text
%y = dp.empty(shape=[M, N], dtype=float16, device=cuda)
call_dps @kernel.matmul(inputs=[%a, %b], output=%y)
```

memory planning 后：

```text
%s0 = alloc_storage(size=..., device=cuda)
%y = alloc_tensor(storage=%s0, shape=[M, N], dtype=float16)
call_dps @kernel.matmul(inputs=[%a, %b], output=%y)
```

---

### 7.2 Pass 流程

推荐流程：

```text
1. DPSLoweringPass
2. TensorCreateAnalyzePass
3. LifetimeAnalyzePass
4. StorageSizeAnalyzePass
5. UpperBoundSizeAnalyzePass
6. StoragePlanPass
7. LowerTensorCreateToAllocPass
```

---

### 7.3 Storage Reuse 规则

MVP 支持 storage reuse。

基本规则：

```text
1. 生命周期不重叠的 tensor 可以复用 storage；
2. 输入 tensor 默认不复用；
3. 输出 tensor 默认不复用；
4. 不跨 device 复用；
5. alignment 必须满足；
6. storage size 必须足够；
7. effectful call 的 read/write 对象必须延长 live range；
8. opaque call 周围保守处理。
```

---

### 7.4 MVP 限制

为了避免 MVP 过度复杂，先限制：

```text
1. 只支持 dense contiguous tensor；
2. 不支持复杂 alias；
3. 不支持 view mutation；
4. 不支持 inplace op 的复杂分析；
5. 不支持跨 device storage reuse；
6. 不支持 storage escape 后的激进复用。
```

KV cache 这种显式 mutable tensor 需要通过 EffectInfo 保护。

---

## 8. `@devproc.kernel` 设计

### 8.1 Kernel 是 implementation

`@devproc.kernel` 注册的是具体实现，不等价于高层 op 本身。

示例：

```python
@dp.kernel(
    op="matmul_add_silu",
    backend="triton",
    device="cuda",
    dtype="float16",
)
def matmul_add_silu_kernel(a, b, bias, out):
    ...
```

普通用户可以继续写：

```python
y = dp.ops.matmul_add_silu(a, b, bias)
```

前端 IR：

```text
%y = call @matmul_add_silu(%a, %b, %bias)
```

中端 kernel selection 后：

```text
%y = dp.empty(shape=[...], dtype=float16, device=cuda)
call_dps @kernel.matmul_add_silu_kernel(
  inputs=[%a, %b, %bias],
  output=%y
)
```

---

### 8.2 Kernel 签名规则

MVP 约束：

```text
1. @devproc.kernel 采用 DPS 签名；
2. 最多一个 output；
3. 可以没有 output；
4. 无 output kernel 必须声明 effect；
5. 不支持多 output kernel；
6. 多逻辑输出使用 Tuple 或 packed tensor。
```

有 output kernel：

```python
@dp.kernel
def relu(x, out):
    ...
```

无 output kernel：

```python
@dp.kernel(effect="write")
def update_kvcache(k_cache, v_cache, new_k, new_v, pos):
    ...
```

---

### 8.3 Kernel Registry

Kernel matching key：

```text
op_name
device
dtype
layout
rank
shape_constraints
attrs
target_arch
priority
```

匹配优先级：

```text
1. 用户注册 shape-specialized kernel
2. devproc2 内置 fused kernel
3. Triton generated kernel
4. cuBLAS / cuDNN wrapper
5. 默认 CUDA kernel
6. 默认 CPU kernel
```

---

## 9. `call_dps_packed` 设计

### 9.1 用途

`call_dps_packed` 用于 runtime 注册函数。

它不是专门调用 kernel。

典型用途：

```text
tokenizer
image decode
image resize
runtime helper
debug print
profiling marker
opaque C++ function
```

---

### 9.2 签名规则

MVP 约束：

```text
1. 最多一个 output；
2. 可以没有 output；
3. 必须显式声明 effect；
4. output 由 caller 创建；
5. runtime function 不负责分配 tensor。
```

示例：

```python
tokens = dp.empty((max_len,), dtype="int32", device="cpu")

dp.call_dps_packed(
    "runtime.tokenizer.encode",
    inputs=[text, tokenizer],
    output=tokens,
    effect="opaque",
)
```

无 output：

```python
dp.call_dps_packed(
    "runtime.profile.mark",
    inputs=["decode_start"],
    output=None,
    effect="opaque",
)
```

---

### 9.3 Runtime 注册

C++：

```cpp
DEVPROC_REGISTER_PACKED_FUNC("runtime.tokenizer.encode")
    .set_body([](PackedArgs args) {
        String text = args[0].AsString();
        Tokenizer tokenizer = args[1].As<Tokenizer>();
        Tensor output = args[2].As<Tensor>();

        tokenizer.EncodeTo(text, output);
    });
```

---

## 10. KV Cache 建模

### 10.1 MVP 建议：显式 mutable tensor

MVP 不要一开始引入复杂 `KVCacheState` object。

先把 KV cache 当成显式 mutable tensor。

DSL：

```python
@dp.function
def decode_step(x, k_cache, v_cache, pos):
    qkv = dp.ops.qkv_proj(x)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    dp.ops.update_kvcache(k_cache, v_cache, k, v, pos)

    out = dp.ops.attention_with_cache(q, k_cache, v_cache, pos)
    return out
```

IR：

```text
%qkv = call @qkv_proj(%x)
%q = tuple_get_item(%qkv, 0)
%k = tuple_get_item(%qkv, 1)
%v = tuple_get_item(%qkv, 2)

call_dps @kernel.update_kvcache(
  inputs=[%k_cache, %v_cache, %k, %v, %pos],
  output=None,
  effect=write(%k_cache, %v_cache)
)

%out = call @attention_with_cache(%q, %k_cache, %v_cache, %pos)
return %out
```

---

### 10.2 Memory planner 处理规则

对于：

```text
effect=write(%k_cache, %v_cache)
```

memory planner 必须保证：

```text
1. k_cache / v_cache 的 storage 不能被中间 tensor 复用；
2. update_kvcache 不能被 DCE 删除；
3. update_kvcache 不能跨 attention_with_cache 错误重排；
4. read/write 会延长 live range；
5. opaque call 前后保守处理。
```

---

## 11. VM 设计

### 11.1 VM 指令集

MVP VM 只保留：

```text
call
ret
if
goto
```

---

### 11.2 VM Function Table

通过 function table 区分 callee 类型：

```text
FunctionTable:
  0: @main                         kind=vm_func
  1: @vm.builtin.alloc_storage      kind=builtin
  2: @vm.builtin.alloc_tensor       kind=builtin
  3: @runtime.tokenizer.encode      kind=packed_func
  4: @kernel.matmul_add_silu        kind=kernel
  5: @vm.builtin.shape_of           kind=builtin
  6: @vm.builtin.assert_shape       kind=builtin
```

VM 执行 `call` 时，根据 callee kind dispatch。

---

### 11.3 VM 指令语义

#### call

```text
call dst_reg, func_index, arg_regs
```

用于：

```text
1. VM function call；
2. builtin call；
3. packed function call；
4. kernel launch；
5. alloc_storage；
6. alloc_tensor；
7. shape helper。
```

如果没有返回值：

```text
call _, func_index, arg_regs
```

或者在 bytecode 中 `dst_reg = -1`。

#### ret

```text
ret reg
```

返回当前函数结果。

无返回函数可以：

```text
ret
```

#### if

```text
if cond_reg, true_offset, false_offset
```

#### goto

```text
goto offset
```

---

## 12. C++ Runtime 动态类型系统

### 12.1 Object / ObjectRef

MVP 实现：

```cpp
class Object {
public:
    virtual ~Object() = default;
    virtual const char* type_key() const = 0;

    void IncRef();
    void DecRef();

private:
    std::atomic<int32_t> ref_count_{0};
};

class ObjectRef {
public:
    ObjectRef() = default;
    explicit ObjectRef(Object* ptr);

    Object* get() const;
    bool defined() const;

    template <typename T>
    T* as() const;

private:
    Object* ptr_{nullptr};
};
```

不对外暴露 `ObjectPtr`。

---

### 12.2 核心对象

```text
TensorObj / Tensor
StorageObj / Storage
ShapeTupleObj / ShapeTuple
StringObj / String
TupleObj / Tuple
PackedFuncObj / PackedFunc
KernelObj / Kernel
ExecutableObj / Executable
VMStateObj / VMState
```

---

### 12.3 VMValue

VM register 需要承载：

```text
Null
Int
Float
Bool
ObjectRef
```

建议：

```cpp
class VMValue {
public:
    enum class Tag {
        kNull,
        kInt,
        kFloat,
        kBool,
        kObjectRef,
    };

private:
    Tag tag_;
    // implementation detail
};
```

---



---

## 12A. Runtime Device API 抽象

前面 MVP 文档只提到了 `Tensor / Storage / MemoryPool / CUDA cubin load`，但这还不够。  
Runtime 必须有一层统一的 **Device API**，否则 CPU / CUDA / 后续 NPU / Metal / Vulkan / vendor backend 都会直接侵入 VM。

devproc2 runtime 不应该在 VM 执行逻辑里直接写 CUDA API。VM 应该只知道：

```text
我要在某个 device 上分配 storage；
我要在某个 device 上拷贝数据；
我要在某个 stream 上 launch kernel；
我要做 device sync / stream sync。
```

具体怎么实现，由 DeviceAPI 负责。

### 12A.1 Device 抽象

Runtime 需要定义统一 device 标识：

```cpp
enum class DeviceType {
    kCPU,
    kCUDA,
    kNPU,
    kMetal,
    kVulkan,
};

struct Device {
    DeviceType type;
    int device_id;
};
```

示例：

```text
cpu(0)
cuda(0)
cuda(1)
npu(0)
```

Tensor / Storage 都必须携带 device：

```cpp
class StorageObj : public Object {
public:
    Device device;
    void* data;
    size_t nbytes;
};

class TensorObj : public Object {
public:
    Storage storage;
    int64_t offset;
    ShapeTuple shape;
    DataType dtype;
    Device device;
};
```

---

### 12A.2 DeviceAPI 接口

MVP 至少需要：

```cpp
class DeviceAPI {
public:
    virtual ~DeviceAPI() = default;

    virtual void* Alloc(Device dev, size_t nbytes, size_t alignment) = 0;
    virtual void Free(Device dev, void* ptr) = 0;

    // 接受 DLTensor*，可直接传 TensorObj::dl()，与 TVM DeviceAPI 接口对齐
    virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, void* stream) = 0;

    virtual void StreamSync(Device dev, void* stream) = 0;
    virtual void DeviceSync(Device dev) = 0;

    virtual void* CreateStream(Device dev) = 0;
    virtual void FreeStream(Device dev, void* stream) = 0;

    virtual void SetDevice(Device dev) = 0;
};
```

MVP 可以先实现：

```text
CPUDeviceAPI
CUDADeviceAPI
```

后续再扩展：

```text
NPUDeviceAPI
MetalDeviceAPI
VulkanDeviceAPI
VendorDeviceAPI
```

---

### 12A.3 DeviceAPI Registry

Runtime 通过 registry 查找 device API：

```cpp
class DeviceAPIRegistry {
public:
    static DeviceAPI* Get(DeviceType type);
    static void Register(DeviceType type, DeviceAPI* api);
};
```

使用方式：

```cpp
auto* api = DeviceAPIRegistry::Get(DeviceType::kCUDA);
void* ptr = api->Alloc(Device{kCUDA, 0}, nbytes, 256);
```

VM / MemoryPool / Tensor copy 不直接依赖 CUDA。

---

### 12A.4 MemoryPool 与 DeviceAPI 的关系

MemoryPool 不应该直接调用 `cudaMalloc`。

正确关系是：

```text
VM
  -> StoragePool / MemoryPool
      -> DeviceAPI
          -> cudaMalloc / cudaFree / cudaMemcpyAsync
```

StoragePlan 决定需要哪些 storage：

```text
storage_0: device=cuda(0), size=64MB
storage_1: device=cpu(0), size=4MB
```

Runtime 执行时：

```cpp
Storage storage = memory_pool.Alloc(device, nbytes, alignment);
```

MemoryPool 内部再调用对应 DeviceAPI。

---

### 12A.5 Stream 抽象

CUDA runtime 必须支持 stream。  
但 VM 不应该直接知道 `cudaStream_t`。

建议 runtime 定义：

```cpp
class StreamObj : public Object {
public:
    Device device;
    void* handle;
};
```

VMState 中保存默认 stream：

```cpp
class VMStateObj : public Object {
public:
    std::unordered_map<Device, Stream> default_streams;
};
```

Kernel launch 时从 VMState 取 stream：

```text
Call kernel.matmul_add_silu
  -> resolve CUDA kernel
  -> get cuda stream from VMState
  -> cuLaunchKernel(..., stream)
```

MVP 至少支持：

```text
1. 每个 device 一个 default stream；
2. 用户可选传入外部 stream；
3. stateful invoke 复用 stream；
4. benchmark 时支持 stream sync。
```

---

### 12A.6 Zero-copy 与 External Buffer

Zero-copy input/output 本质上是用户把外部 buffer 包装成 Tensor。

需要 API：

```cpp
Tensor Tensor::FromExternalBuffer(
    void* data,
    Device device,
    ShapeTuple shape,
    DataType dtype,
    std::vector<int64_t> strides,
    Deleter deleter = nullptr
);
```

如果是外部 buffer：

```text
StorageObj owns_data = false
```

Runtime 不负责释放。

Memory planner 也必须知道：

```text
1. input external storage 不参与 reuse；
2. output external storage 不参与 reuse；
3. external mutable state，比如 k_cache/v_cache，不参与临时 storage reuse。
```

---

## 12B. Runtime Shape / PrimExpr 计算

MVP 文档里提到了 dynamic shape 和 upper bound，但还缺少一个关键点：

> 动态 shape 相关的表达式，最终必须在 runtime 里计算。

例如：

```text
B = shape_of(x)[0]
S = shape_of(x)[1]
out_shape = [B, S, 4096]
storage_size = upper(B) * upper(S) * 4096 * sizeof(float16)
grid_x = ceildiv(S, BLOCK_M)
grid_y = B
```

其中一部分可以 compile-time 静态化，一部分必须 runtime 计算。

---

### 12B.1 ShapeExpr / PrimExpr 分类

建议把 shape 表达式分成两类：

```text
1. Compile-time shape expr
2. Runtime shape expr
```

例如：

```text
H = 4096
BLOCK_M = 16
```

是 compile-time constant。

而：

```text
B = shape_of(x)[0]
S = shape_of(x)[1]
ceildiv(S, 16)
```

需要 runtime 计算。

---

### 12B.2 Runtime 需要支持的 PrimExpr

MVP 不需要完整符号代数系统，但至少需要支持：

```text
const
dim
add
sub
mul
floordiv
ceildiv
mod
min
max
eq
lt
le
gt
ge
and
or
```

其中最关键的是：

```text
add / mul / ceildiv / min / max / le
```

这些会用于：

```text
1. output shape 计算；
2. storage size 计算；
3. runtime shape assert；
4. kernel launch grid 计算；
5. for/range loop bound 计算。
```

---

### 12B.3 ShapeExpr Lowering 到 VM

不要为每个 shape op 增加 VM opcode。

仍然保持 VM 四指令：

```text
call
ret
if
goto
```

shape 计算通过 builtin function table 完成：

```text
@vm.builtin.shape_of
@vm.builtin.get_shape_dim
@vm.builtin.add_i64
@vm.builtin.mul_i64
@vm.builtin.ceildiv_i64
@vm.builtin.min_i64
@vm.builtin.max_i64
@vm.builtin.assert_le_i64
```

例如：

```text
%shape = call @vm.builtin.shape_of(%x)
%B = call @vm.builtin.get_shape_dim(%shape, 0)
%S = call @vm.builtin.get_shape_dim(%shape, 1)
%grid_x = call @vm.builtin.ceildiv_i64(%S, 16)
call @vm.builtin.assert_le_i64(%S, 2048)
```

VM opcode 仍然只有 `call`，但 function table 里有 shape builtin。

---

### 12B.4 Runtime Shape Heap / Shape Register

VM register 可以直接保存 int64 shape value，也可以保存 ShapeTuple。

建议：

```text
单个 dim: VMValue::Int
shape tuple: ShapeTuple ObjectRef
```

例如：

```text
B -> Int
S -> Int
[B, S, H] -> ShapeTuple
```

这样 `alloc_tensor` 可以接受：

```text
storage, offset, shape_tuple, dtype, device
```

---

### 12B.5 Upper Bound 的 Runtime 检查

对于：

```text
S <= 2048
```

VM codegen 插入：

```text
call @vm.builtin.assert_le_i64(%S, 2048)
```

如果失败，runtime 报错：

```text
RuntimeShapeError: dimension S actual value 4096 exceeds upper bound 2048.
```

这是 memory planning 正确性的底线。

---

## 12C. `@devproc.kernel` 中的 div / mul / grid 计算怎么实现

`@devproc.kernel` 中会出现两类 div / mul：

```text
1. kernel 内部的 div/mul
2. kernel launch 参数中的 div/mul
```

它们处理方式不同。

---

### 12C.1 Kernel 内部 div / mul

例如 Triton kernel 里：

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < N
```

或者：

```python
row = idx // N
col = idx % N
```

这些属于 kernel 内部计算。

处理方式：

```text
@devproc.kernel backend=triton
  -> 保留在 Triton source / Triton IR 中
  -> Triton 编译成 cubin
  -> runtime 不理解这些 div / mul
```

也就是说，kernel 内部 arithmetic 不由 devproc2 runtime 执行。

Runtime 只负责 launch cubin。

---

### 12C.2 Kernel launch grid 中的 div / mul

例如：

```text
grid = (ceildiv(M, BLOCK_M), ceildiv(N, BLOCK_N), B)
```

这里的 `ceildiv / mul / add` 是 runtime 需要计算的。

因为 M / N / B 可能是动态 shape。

解决方案：

```text
1. kernel metadata 记录 grid expression；
2. VM codegen 将 grid expression lower 成 shape builtin call；
3. runtime 算出 grid_x / grid_y / grid_z；
4. kernel launcher 使用计算后的 grid launch cubin。
```

示例 kernel metadata：

```json
{
  "name": "matmul_kernel",
  "grid": [
    {"op": "ceildiv", "args": ["M", "BLOCK_M"]},
    {"op": "ceildiv", "args": ["N", "BLOCK_N"]},
    1
  ],
  "block": [256, 1, 1]
}
```

VM lowering 后：

```text
%grid_x = call @vm.builtin.ceildiv_i64(%M, %BLOCK_M)
%grid_y = call @vm.builtin.ceildiv_i64(%N, %BLOCK_N)
call @kernel.matmul(%a, %b, %out, %M, %N, %K, %grid_x, %grid_y)
```

---

### 12C.3 Kernel ABI 中的 shape 参数

DPS kernel ABI 应该明确 shape 参数如何传递。

例如：

```python
@dp.kernel
def matmul(a, b, out, M: int, N: int, K: int):
    ...
```

高层 op：

```python
y = dp.ops.matmul(a, b)
```

中端 lowering：

```text
%M = shape_of(a)[0]
%K = shape_of(a)[1]
%N = shape_of(b)[1]
%y = dp.empty([M, N], dtype=float16, device=cuda)
call_dps @kernel.matmul(inputs=[%a, %b, %M, %N, %K], output=%y)
```

这样 kernel 内部不需要自己解析 Tensor shape metadata。  
它直接拿到展开后的 int64 shape 参数。

MVP 推荐这样做，ABI 更明确。

---

### 12C.4 Kernel launch builtin

VM 可以把 kernel launch 统一封装成 builtin / kernel function：

```text
call @kernel.matmul(args...)
```

底层 runtime 做：

```text
1. 解析 Tensor 参数；
2. 解析 scalar shape 参数；
3. 根据 kernel metadata 计算或读取 grid/block；
4. 准备 CUDA kernel 参数；
5. 使用 DeviceAPI 获取 stream；
6. cuLaunchKernel。
```

如果 grid 已经由 VM shape builtin 计算好，也可以作为 hidden args 传给 launcher。

---

### 12C.5 MVP 推荐规则

MVP 里建议明确规定：

```text
1. kernel 内部 arithmetic 由 backend 编译器处理；
2. kernel launch arithmetic 由 devproc2 runtime shape builtin 处理；
3. dynamic dim 在 VM 中表现为 int64；
4. kernel ABI 显式传 shape scalar；
5. grid expression 进入 kernel metadata；
6. runtime 只支持有限 PrimExpr：add/mul/ceildiv/min/max/compare。
```

这样边界清晰：

```text
Triton 负责 kernel 内部计算；
VM/runtime 负责 shape 和 launch 参数计算；
DeviceAPI 负责实际 device 操作。
```

---

## 12D. Device API / Shape Runtime 相关新增里程碑

### Milestone X1：Runtime Device API MVP

目标：建立跨设备执行抽象。

任务：

```text
- [ ] Device / DeviceType
- [ ] DeviceAPI interface
- [ ] DeviceAPIRegistry
- [ ] CPUDeviceAPI
- [ ] CUDADeviceAPI
- [ ] Stream abstraction
- [ ] MemoryPool 通过 DeviceAPI 分配内存
- [ ] Tensor external buffer / zero-copy
- [ ] device copy API
```

验收：

```text
VM 不直接调用 cudaMalloc / cudaMemcpy / cudaStreamSynchronize；
所有设备操作都经过 DeviceAPI；
CUDA tensor allocation / copy / stream sync 可运行。
```

---

### Milestone X2：Runtime Shape Builtin MVP

目标：支持 dynamic shape runtime 计算。

任务：

```text
- [ ] ShapeTuple Object
- [ ] shape_of builtin
- [ ] get_shape_dim builtin
- [ ] add_i64 / sub_i64 / mul_i64 builtin
- [ ] floordiv_i64 / ceildiv_i64 builtin
- [ ] min_i64 / max_i64 builtin
- [ ] compare builtin
- [ ] assert_le_i64 builtin
- [ ] ShapeExpr lowering to VM call
```

验收：

```text
动态输入 Tensor[(B, S, H)] 能在 runtime 计算 B/S；
能检查 S <= upper_bound；
能用 ceildiv(S, BLOCK) 计算 kernel grid；
能用动态 shape 创建 output tensor view。
```

---

### Milestone X3：Kernel Launch Expression MVP

目标：支持动态 shape kernel launch。

任务：

```text
- [ ] kernel metadata 记录 grid expression
- [ ] VM codegen lower grid expression
- [ ] kernel ABI 支持 scalar shape args
- [ ] CUDA launcher 接收 runtime grid/block
- [ ] Triton cubin launch 支持 dynamic grid
```

验收：

```text
同一个 cubin 能在不同 S 下 launch；
grid_x = ceildiv(S, BLOCK) 在 runtime 正确计算；
kernel 输出正确。
```



## 13. ABI 与编译产物

### 13.1 Artifact 结构

```text
build/devproc2_module/
  manifest.json
  abi.json
  executable.vm
  constants/
    const_0.bin
    const_1.bin
  kernels/
    kernel_0.cubin
    kernel_0.ptx        # optional debug artifact
  metadata/
    function_table.json
    kernel_table.json
    packed_func_table.json
    storage_plan.json
    shape_constraints.json
```

---

### 13.2 ABI 必须描述

```text
devproc_abi_version
vm_bytecode_version
target
target_arch
input ABI
output ABI
function ABI
packed function ABI
kernel ABI
shape constraint ABI
storage plan ABI
effect ABI
```

---

### 13.3 ABI Version

MVP 固定：

```text
devproc_abi_version = 0.1
vm_bytecode_version = 0.1
kernel_calling_convention = dps_kernel_v1
packed_func_calling_convention = dps_packed_v1
```

---

## 14. Triton AOT Cubin 设计

### 14.1 编译路径

```text
@devproc.kernel
  -> Triton source
  -> specialization
  -> cubin
  -> optional ptx
  -> kernel metadata
  -> artifact packaging
  -> runtime load cubin
  -> cuLaunchKernel
```

Runtime 不编译 Triton。

---

### 14.2 为什么保留 PTX

执行主产物是 cubin。

PTX 用于：

```text
1. debug；
2. kernel inspection；
3. agent 优化；
4. fallback investigation；
5. 编译报告分析。
```

---

## 15. Compiler Pipeline

MVP pipeline（共 18 步，详见 [docs/mvp_impl/02_compiler_pipeline.md](../mvp_impl/02_compiler_pipeline.md)）：

```text
[1]  Python DSL Capture
[2]  High-level IR Build
[3]  NormalizeIRPass
[4]  ControlFlowNormalizePass       ← elif→nested if，loop-carried variable→iter_args
[5]  StructInfoInferPass
[6]  DynamicShapeAnalyzePass
[7]  ShapeConstraintVerifyPass
[8]  EffectAnalyzePass
[9]  KernelSelectPass
[10] DPSLoweringPass
──────────── 分界线：引入 TensorCreateOp ────────────
[11] MemoryPlanningPass             ← 合并原 4 个分析阶段（TCA→LA→SSA→SP）
──────────── 分界线：引入 alloc_storage/alloc_tensor ────────────
[12] LowerTensorCreateToAllocPass
[13] ShapeExprLoweringPass
[14] KernelLaunchExprLoweringPass
[15] VMCodegenPass
[16] TritonAOTCompilePass
[17] ExecutableEmitPass
[18] ABIEmitPass
```

关键点：

```text
alloc_storage / alloc_tensor 只在第 12 步之后出现。
ControlFlowNormalizePass（第 4 步）将 elif 展平、loop-carried variable 转 iter_args。
VMCodegenPass（第 15 步）将 If→IF+GOTO、For→循环计数+GOTO，VM 只有 4 条指令。
```

---

## 16. 推荐目录结构

```text
devproc2/
  python/
    devproc2/
      __init__.py

      ir/
        module.py
        function.py
        block.py
        expr.py
        call.py
        control_flow.py
        tuple.py
        tensor_create.py
        struct_info.py
        shape_expr.py
        effect.py
        printer.py
        verifier.py

      frontend/
        dsl.py
        builder.py

      ops/
        tensor.py
        nn.py
        fused.py
        stateful.py

      kernel/
        register.py
        kernel_spec.py
        triton_kernel.py

      compiler/
        build.py
        pipeline.py
        passes/
          normalize.py
          control_flow_normalize.py
          infer_struct_info.py
          dynamic_shape_analyze.py
          shape_constraint_verify.py
          effect_analyze.py
          kernel_select.py
          dps_lowering.py
          memory_planning.py
          lower_tensor_create_to_alloc.py
          vm_codegen.py
          triton_aot_compile.py
          emit_executable.py
          emit_abi.py

      vm/
        bytecode.py
        instruction.py
        executable.py
        serializer.py

      runtime/
        binding.py

      testing/
        verify.py
        benchmark.py

  runtime/
    include/devproc2/runtime/
      object.h
      object_ref.h
      vm_value.h
      tensor.h
      storage.h
      shape_tuple.h
      tuple.h
      string.h
      packed_func.h
      kernel.h
      executable.h
      vm.h
      state.h
      memory_pool.h

    src/
      object.cc
      object_ref.cc
      vm_value.cc
      tensor.cc
      storage.cc
      shape_tuple.cc
      tuple.cc
      string.cc
      packed_func.cc
      kernel.cc
      executable.cc
      vm.cc
      state.cc
      memory_pool.cc

      cuda/
        cuda_module.cc
        cuda_kernel.cc
        cuda_memory_pool.cc

  kernels/
    triton/
      add.py
      relu.py
      matmul.py
      layernorm.py
      matmul_add_silu.py
      update_kvcache.py

  examples/
    static_graph_mvp/
    control_flow_mvp/
    dynamic_shape_mvp/
    tokenizer_mvp/
    kv_cache_mvp/
    fused_kernel_mvp/

  tests/
    ir/
    compiler/
    runtime/
    integration/

  docs/
    mvp_plan.md
    ir.md
    vm.md
    abi.md
    memory_planning.md
    dynamic_shape.md
    effect.md
```

---

## 17. MVP 里程碑

### Milestone 1：C++ Object / ObjectRef 动态类型系统

目标：搭建 runtime 类型系统底座。

任务：

```text
- [ ] Object
- [ ] ObjectRef
- [ ] runtime type key
- [ ] reference counting
- [ ] TensorObj / Tensor
- [ ] StorageObj / Storage
- [ ] ShapeTupleObj / ShapeTuple
- [ ] TupleObj / Tuple
- [ ] StringObj / String
- [ ] PackedFuncObj / PackedFunc
- [ ] KernelObj / Kernel
- [ ] VMValue
```

验收：C++ runtime 可以统一保存和传递：

```text
Tensor / Storage / ShapeTuple / Tuple / String / Int / Float / Bool
```

---

### Milestone 2：High-level IR MVP

目标：实现前端高层 IR。

任务：

```text
- [ ] IRModule
- [ ] Function
- [ ] Block
- [ ] Var
- [ ] Call
- [ ] CallDPS
- [ ] If
- [ ] For
- [ ] Range
- [ ] Tuple
- [ ] TupleGetItem
- [ ] TensorCreateOp
- [ ] Return
- [ ] TensorStructInfo
- [ ] ShapeExpr
- [ ] SymbolicDim
- [ ] UpperBound
- [ ] EffectInfo
- [ ] IR printer
- [ ] IR verifier
```

验收：能打印：

```text
def @main(%x: Tensor[(B, S), float16]) {
  %y = call @matmul(%x, %w)
  %z = call @silu(%y)
  return %z
}
```

以及：

```text
call_dps @update_kvcache(
  inputs=[%k_cache, %v_cache, %k, %v, %pos],
  output=None,
  effect=write(%k_cache, %v_cache)
)
```

---

### Milestone 3：Control Flow MVP

目标：支持常见 Python 控制流。

任务：

```text
- [ ] DSL 支持 if
- [ ] DSL 支持 elif
- [ ] DSL 支持 else
- [ ] DSL 支持 for
- [ ] DSL 支持 dp.range
- [ ] IR structured if
- [ ] IR structured for
- [ ] loop-carried variable
- [ ] ControlFlowNormalizePass
```

验收：支持：

```python
for i in dp.range(0, n):
    if i < threshold:
        x = dp.ops.relu(x)
    else:
        x = dp.ops.silu(x)
```

---

### Milestone 4：Dynamic Shape MVP

目标：完整支持 symbolic shape 和 upper bound。

任务：

```text
- [ ] SymbolicDim
- [ ] ShapeExpr
- [ ] ShapeConstraint
- [ ] UpperBound
- [ ] RuntimeShapeValue
- [ ] ShapeConstraintVerifier
- [ ] runtime shape assert
- [ ] upper bound size inference
```

验收：支持：

```text
Tensor[(B, S, H), float16]
where B <= 8, S <= 2048
```

输入超过 upper bound 时，runtime 必须报错。

---

### Milestone 5：Effect System MVP

目标：支持 no-output call、stateful call、opaque runtime call。

任务：

```text
- [ ] EffectInfo
- [ ] pure / read_only / write / opaque
- [ ] effect verifier
- [ ] DCE 尊重 effect
- [ ] memory planner 尊重 effect
- [ ] scheduler 不跨 effectful call 错误重排
```

验收：无 output 的：

```text
call_dps @update_kvcache(..., output=None, effect=write(...))
```

不能被删除，不能被错误重排。

---

### Milestone 6：DPS Lowering MVP

目标：将普通函数式 op lower 成 DPS 调用。

任务：

```text
- [ ] 根据 StructInfo 推导 output shape / dtype / device
- [ ] 插入 TensorCreateOp
- [ ] 将 Call lower 为 CallDPS
- [ ] 绑定 kernel implementation
- [ ] 处理 no-output CallDPS
- [ ] 保留 call_dps_packed 显式 DPS
```

验收：高层 IR：

```text
%y = call @matmul(%a, %b)
```

lower 成：

```text
%y = dp.empty(shape=[M, N], dtype=float16, device=cuda)
call_dps @kernel.matmul(inputs=[%a, %b], output=%y)
```

---

### Milestone 7：Memory Planning MVP

目标：自动插入 `alloc_storage` / `alloc_tensor`，并支持 storage reuse。

任务：

```text
- [ ] TensorCreateAnalyzePass
- [ ] LifetimeAnalyzePass
- [ ] StorageSizeAnalyzePass
- [ ] UpperBoundSizeAnalyzePass
- [ ] StoragePlanPass
- [ ] LowerTensorCreateToAllocPass
- [ ] storage reuse
- [ ] memory plan dump
```

验收：输出 memory plan：

```text
Storage 0:
  device = cuda
  size = 64MB
  reused_by = [%tmp0, %tmp3, %tmp7]

Storage 1:
  device = cuda
  size = 16MB
  reused_by = [%tmp1, %tmp5]
```

并且 memory-explicit IR 中出现：

```text
%storage0 = alloc_storage(...)
%tmp0 = alloc_tensor(%storage0, ...)
```

---

### Milestone 8：VM MVP

目标：实现极简 VM。

任务：

```text
- [ ] VM executable format
- [ ] VM function table
- [ ] VM register file
- [ ] VM frame
- [ ] instruction dispatch
- [ ] call
- [ ] ret
- [ ] if
- [ ] goto
- [ ] builtin dispatch
- [ ] packed function dispatch
- [ ] kernel dispatch
- [ ] stateful invoke
```

验收：VM 能执行：

```text
call @vm.builtin.alloc_storage
call @vm.builtin.alloc_tensor
call @kernel.relu
ret
```

也能执行由 `if/goto` 表达的控制流。

---

### Milestone 9：ABI + Artifact MVP

目标：生成稳定可加载的编译产物。

任务：

```text
- [ ] manifest.json
- [ ] abi.json
- [ ] executable.vm
- [ ] function table
- [ ] kernel table
- [ ] packed function table
- [ ] shape constraint table
- [ ] storage plan table
- [ ] effect metadata
- [ ] executable serializer
- [ ] executable loader
- [ ] ABI version check
```

验收：Runtime 加载 artifact 时检查：

```text
ABI version
VM bytecode version
required packed funcs
required kernels
target arch
input/output contract
shape constraints
```

缺失 runtime function 时明确报错：

```text
Packed function runtime.tokenizer.encode is required but not registered.
```

---

### Milestone 10：PackedFunc + call_dps_packed MVP

目标：支持 tokenizer 等 runtime 函数。

任务：

```text
- [ ] C++ PackedFunc
- [ ] PackedFunc registry
- [ ] DEVPROC_REGISTER_PACKED_FUNC
- [ ] call_dps_packed DSL API
- [ ] CallDPS callee_kind=packed_func
- [ ] DPS packed ABI
- [ ] tokenizer mock
- [ ] tokenizer real implementation hook
```

验收：可以运行：

```text
text -> runtime.tokenizer.encode -> tokens
```

其中 tokens 由 caller 创建并传入。

---

### Milestone 11：`@devproc.kernel` + Triton Cubin MVP

目标：支持 DPS kernel 注册和 Triton AOT cubin 编译。

任务：

```text
- [ ] @devproc.kernel
- [ ] kernel spec
- [ ] kernel registry
- [ ] DPS kernel ABI
- [ ] Triton AOT compile cubin
- [ ] optional PTX dump
- [ ] VM kernel call dispatch
- [ ] CUDA module loader
- [ ] CUDA kernel launcher
```

验收：可以跑通：

```text
@devproc.kernel backend="triton"
  -> cubin
  -> artifact
  -> VM call
  -> cuLaunchKernel
```

---

### Milestone 12：End-to-End Demo

推荐 demo：

```text
text
  -> call_dps_packed(runtime.tokenizer.encode)
  -> embedding
  -> dynamic shape layernorm
  -> matmul_add_silu fused kernel
  -> update_kvcache no-output stateful kernel
  -> attention_with_cache
  -> output
```

这个 demo 覆盖：

```text
1. 普通函数式 Call；
2. Tuple 多逻辑输出；
3. CallDPS；
4. call_dps_packed；
5. no-output stateful call；
6. EffectInfo；
7. dynamic shape；
8. upper bound；
9. memory planning；
10. storage reuse；
11. VM；
12. ABI；
13. C++ Object / ObjectRef；
14. Triton cubin。
```

---

## 18. 测试计划

### 18.1 IR 测试

```text
- [ ] Call 测试
- [ ] CallDPS 测试
- [ ] no-output CallDPS 测试
- [ ] Tuple / TupleGetItem 测试
- [ ] If / For / Range 测试
- [ ] TensorCreateOp 测试
- [ ] EffectInfo 测试
- [ ] dynamic shape 测试
```

---

### 18.2 Compiler 测试

```text
- [ ] StructInfoInferPass
- [ ] DynamicShapeAnalyzePass
- [ ] ShapeConstraintVerifyPass
- [ ] EffectAnalyzePass
- [ ] KernelSelectPass
- [ ] DPSLoweringPass
- [ ] TensorCreateAnalyzePass
- [ ] LifetimeAnalyzePass
- [ ] StoragePlanPass
- [ ] LowerTensorCreateToAllocPass
- [ ] VMCodegenPass
```

---

### 18.3 Runtime 测试

```text
- [ ] Object / ObjectRef
- [ ] VMValue
- [ ] Executable loader
- [ ] Function table
- [ ] call / ret / if / goto
- [ ] PackedFunc registry
- [ ] Kernel registry
- [ ] CUDA cubin load
- [ ] memory pool
- [ ] stateful invoke
- [ ] zero-copy input/output
```

---

### 18.4 Integration 测试

```text
- [ ] static graph demo
- [ ] control flow demo
- [ ] dynamic shape demo
- [ ] memory reuse demo
- [ ] tokenizer demo
- [ ] kv cache update demo
- [ ] fused kernel demo
- [ ] end-to-end VM demo
```

---

## 19. 推荐开发顺序

实际开发建议顺序：

```text
1. C++ Object / ObjectRef / VMValue
2. High-level IR
3. Call / CallDPS / Tuple / EffectInfo
4. Control Flow IR
5. Dynamic Shape / StructInfo
6. VM bytecode format
7. VM runtime skeleton
8. ABI / executable loader
9. DPSLoweringPass
10. TensorCreateOp
11. Memory planning
12. LowerTensorCreateToAllocPass
13. PackedFunc registry
14. call_dps_packed
15. @devproc.kernel
16. Triton cubin AOT
17. KV cache no-output stateful kernel demo
18. End-to-end demo
```

核心原则：

```text
先定 IR / VM / ABI / memory model；
再接 PackedFunc 和 Kernel；
最后做 Triton 和性能 demo。
```

不要先做 Triton。否则 IR、ABI、memory plan 一变，Triton 接入一定返工。

---

## 20. MVP 成功标准

### IR 层

```text
- [ ] 支持普通 Call
- [ ] 支持 CallDPS
- [ ] Call 最多一个返回值
- [ ] 多逻辑输出用 Tuple
- [ ] CallDPS 最多一个 output
- [ ] CallDPS 支持 output=None
- [ ] 支持 EffectInfo
- [ ] 支持 if / elif / else
- [ ] 支持 for / range
- [ ] 支持 TensorCreateOp
- [ ] 支持 dynamic shape
- [ ] 支持 upper bound
```

---

### 中端

```text
- [ ] 支持 DPS lowering
- [ ] 支持 effect analysis
- [ ] 支持 dynamic shape analysis
- [ ] 支持 memory planning
- [ ] 支持 storage reuse
- [ ] 自动插入 alloc_storage
- [ ] 自动插入 alloc_tensor
- [ ] upper bound 参与 storage size 计算
```

---

### VM

```text
- [ ] 只有 call / ret / if / goto 四类核心指令
- [ ] 支持 function table
- [ ] 支持 builtin call
- [ ] 支持 packed func call
- [ ] 支持 kernel call
- [ ] 支持 stateful invoke
```

---

### Runtime

```text
- [ ] C++ Object / ObjectRef
- [ ] Tensor / Storage / ShapeTuple / Tuple / String
- [ ] PackedFunc registry
- [ ] Kernel registry
- [ ] CUDA cubin load
- [ ] zero-copy input/output
- [ ] memory pool
- [ ] DeviceAPI / DeviceAPIRegistry
- [ ] CPUDeviceAPI / CUDADeviceAPI
- [ ] Stream abstraction
- [ ] runtime shape builtin
- [ ] dynamic kernel launch grid expression
```

---

### Demo

```text
- [ ] static tensor graph demo
- [ ] control flow demo
- [ ] dynamic shape + memory reuse demo
- [ ] tokenizer + call_dps_packed demo
- [ ] no-output update_kvcache demo
- [ ] Triton fused kernel demo
- [ ] end-to-end VM demo
```

---

## 21. 最终设计结论

devproc2 MVP 最终可以概括为：

```text
前端 DSL：
  普通 op 保持函数式写法；
  @devproc.kernel 使用 DPS 签名；
  call_dps_packed 显式 DPS；
  no-output stateful call 合法。

IR：
  Call 最多一个返回值；
  多逻辑输出用 Tuple；
  CallDPS 最多一个 output 或 output=None；
  EffectInfo 保护 stateful / opaque call；
  支持 if / for / range / dynamic shape。

中端：
  将普通 Call lower 成 CallDPS；
  自动插入 TensorCreateOp；
  memory planning 后插入 alloc_storage / alloc_tensor；
  支持 storage reuse 和 upper bound 优化。

VM：
  指令集只保留 call / ret / if / goto；
  通过 function table 区分 builtin / packed_func / kernel / vm_func。

Runtime：
  C++ Object / ObjectRef 动态类型系统；
  PackedFunc registry；
  Kernel registry；
  CUDA cubin loader；
  stateful invoke；
  zero-copy input/output。
```

一句话：

> devproc2 MVP 应该保持前端自然、IR 语义清晰、中端负责 DPS 和内存规划、VM 极简、Runtime 稳定。普通 op 用 `y = matmul(a, b)`，kernel 和 runtime escape hatch 用 DPS，no-output stateful call 通过 `CallDPS(output=None, effect=...)` 成为一等公民。