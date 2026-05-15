# M10：PackedFunc + call_dps_packed 设计文档

> 适合读者：了解基本 Python / C++ 概念，没有编译器背景的初学者。
>
> 读完本文，你将理解：devproc2 如何让 C++ 运行时函数（如分词器）从 VM 中被调用，整个调用链是什么样的。

---

## 1. 问题背景：VM 为什么需要调用"外部函数"？

devproc2 的 VM 很纯粹：它按照编译好的字节码执行张量操作和 kernel launch。但实际的 LLM inference 需要更多东西，比如：

| 操作 | 谁来负责 |
|---|---|
| 文本 tokenization（把句子拆成 token id） | C++ 分词库（BPE / tiktoken） |
| KV cache 更新（写入循环 buffer） | 自定义 C++ 算子 |
| 统计 profiling | 外部 profiler callback |
| 动态 sampling（top-k, nucleus） | Python / C++ 实现 |

这些操作的特点是：**它们不是 tensor 计算，不适合用 Triton kernel 表达，但又必须在 VM 执行流中被调用。**

devproc2 的解决方案叫做 **PackedFunc（打包函数）**。

---

## 2. PackedFunc 是什么？

PackedFunc 是 devproc2 对"任意 C++ 函数"的统一封装。

思路很简单：

```
任意 C++ 函数
   ↓ 包装成
PackedFuncObj { std::function<void(PackedArgs)> body }
   ↓ 注册到
PackedFuncRegistry { std::unordered_map<string, PackedFunc> }
   ↓ VM 在执行时通过名字查找并调用
```

类比理解：就像操作系统的"动态链接库（.so/.dll）"。你写好一个函数，注册进去，VM 只知道函数名字，运行时再找到它并调用。

### 2.1 核心 C++ 类

```cpp
// PackedArgs：传给 PackedFunc 的参数包
// 本质上是 std::vector<VMValue>& 的包装，提供下标访问
class PackedArgs {
    int size() const;
    VMValue& operator[](int i);
};

// PackedFuncObj：持有一个 std::function 的 Object
class PackedFuncObj : public Object {
    std::function<void(PackedArgs)> body;
    void Call(PackedArgs args) { body(args); }
};

// PackedFunc：ObjectRef 包装（智能引用计数）
class PackedFunc : public ObjectRef { ... };
```

### 2.2 全局注册表

```cpp
class PackedFuncRegistry {
    static PackedFuncRegistry& Global();  // 单例

    void Register(const std::string& name, PackedFunc func);
    PackedFunc Get(const std::string& name) const;  // 不存在返回 undefined
    bool Has(const std::string& name) const;
};
```

**线程安全**：内部用 `std::mutex` 保护，多线程注册/查找不会出错。

### 2.3 注册宏

```cpp
// 静态初始化注册（.cc 文件顶层使用）
DEVPROC2_REGISTER_PACKED_FUNC("runtime.tokenizer.encode")
    .set_body([](PackedArgs args) {
        // args[0] = String 文本
        // args[1] = Tensor 输出 buffer
        // ... 写入 token ids
    });
```

这个宏利用了 C++ **静态初始化顺序**：`main()` 运行前，所有注册都已完成。

---

## 3. call_dps_packed：在 IR 里调用 PackedFunc

PackedFunc 注册好了，但怎么在 devproc2 的编译流程里使用它？

答案是通过 **`CallDPSOp`（Destination Passing Style Call）**，并设置 `callee_kind=packed_func`。

### 3.1 DPS（Destination Passing Style）是什么？

普通函数调用会**返回**一个新张量：
```python
out = relu(x)   # 分配新内存，填充结果
```

DPS 调用要求调用者**提前分配好输出内存，传给被调函数写入**：
```python
out = dp.empty(shape, dtype)   # 调用者预分配
relu_dps(x, out)               # 被调函数写入 out
```

好处：
- **零额外分配**：内存由 Memory Planner 统一规划，被调函数只负责写
- **可复用 storage**：同一块内存可以被多个 DPS 调用复用（生命周期不重叠时）
- **PackedFunc 自然适配**：tokenizer 这类函数本来就是"你给我个 buffer 我往里写"

### 3.2 IR 节点：CallDPSOp

```python
@dataclass
class CallDPSOp(Op):
    callee:      str          # 函数名，如 "runtime.tokenizer.encode"
    callee_kind: CalleeKind   # packed_func / kernel / builtin
    inputs:      tuple[Value, ...]   # 输入（只读）
    output:      Optional[Value]     # 输出 buffer（可为 None：effect-only）
    effect:      EffectInfo          # OpaqueEffect() 等
```

**`output=None` 的含义**：有些 packed_func 只有副作用，不写任何 tensor（例如更新 KV cache 内部的循环指针）。`output=None` 表示"我只是调用它，什么都不用接收"。

---

## 4. 从 Python DSL 到 IR：两个新语法

M10 在 Python DSL 层面增加了两个功能：

### 4.1 `dp.empty()`：在 DSL 中预分配 buffer

```python
@dp.function
def tokenize(text, max_len):
    tokens = dp.empty((max_len,), dtype="int32", device="cpu")  # ← 新增
    dp.call_dps_packed(
        "runtime.tokenizer.encode",
        inputs=[text],
        output=tokens,
    )
    return tokens
```

`dp.empty(shape, dtype, device)` 在 IR 中生成一个 `TensorCreateOp`：

```
# IR 表示
tokens = TensorCreateOp(kind=empty, shape=(max_len,), dtype="int32", device="cpu")
```

Memory Planner 会给它分配实际的存储，最终替换为：

```
s0     = alloc_storage(size=..., alignment=256, device=cpu)
tokens = alloc_tensor(s0, offset=0, shape=(max_len,), dtype=int32)
```

### 4.2 `dp.call_dps_packed()`：调用 PackedFunc

```python
dp.call_dps_packed(
    "runtime.tokenizer.encode",  # ← 函数名（字符串）
    inputs=[text],               # ← 只读输入列表
    output=tokens,               # ← 输出 buffer（或 None）
    effect="opaque",             # ← 副作用类型（默认 opaque）
)
```

DSL 底层通过 AST 解析将这段代码转成 `CallDPSOp`。关键点：**整个 `@dp.function` 装饰器不执行函数体，只解析 AST**。所以 `dp.empty()` 和 `dp.call_dps_packed()` 在函数体里"调用"时，实际上只是告诉编译器"我要做这件事"，不会真的运行。

---

## 5. 编译 Pipeline 中每一步发生了什么

以下面这个函数为例，逐步追踪：

```python
@dp.function
def tokenize(text: dp.Tensor[(1,), "int32", "cpu"]):
    tokens = dp.empty((8,), dtype="int32", device="cpu")
    dp.call_dps_packed("runtime.tokenizer.encode", inputs=[text], output=tokens)
    return tokens
```

### Step 1：DSL → 高层 IR

`@dp.function` 装饰器触发 AST 解析，产生：

```
Function "tokenize"
  params: [text: Var]
  body: Block
    TensorCreateOp(result="tokens", kind=empty, shape=(8,), dtype="int32", device="cpu")
    CallDPSOp(callee="runtime.tokenizer.encode",
              callee_kind=packed_func,
              inputs=(text,), output=tokens, effect=OpaqueEffect)
    ReturnOp(values=(tokens,))
```

### Step 2：InferStructInfoPass

`tokens` 是 `TensorCreateOp` 的结果，它的 `struct_info` 被推导为：

```
TensorStructInfo(shape=(IntImm(8),), dtype="int32", device="cpu")
```

这一步确保下游 pass 知道每个值的形状和类型。

### Step 3：DPSLoweringPass

这个 pass 会把 `CallOp`（普通函数调用）lower 成 DPS 形式。但 `CallDPSOp` 已经是 DPS 了，所以这里**保持不变**，只处理 `relu` 这类普通 `CallOp`。

### Step 4：MemoryPlanningPass

分析所有 tensor 的生命周期，给 `tokens` 分配 storage：

```json
{
  "id": 0,
  "device": "cpu",
  "size_bytes": 32,        // 8 × 4字节
  "reused_by": ["tokens"]
}
```

### Step 5：LowerTensorCreateToAllocPass

把 `TensorCreateOp` 替换成真正的分配操作：

```
s0     = alloc_storage(size=32, alignment=256, device=cpu)
tokens = alloc_tensor(s0, offset=0, shape=(8,), dtype=int32)
CallDPSOp("runtime.tokenizer.encode", inputs=(text,), output=tokens)
ReturnOp(tokens)
```

### Step 6：VMCodegenPass

把 IR 翻译成 VM 字节码：

```
# 函数 "tokenize" 的字节码
CALL dst=r2, @vm.builtin.alloc_storage,  [r_32, r_256, r_1, r_0]  // s0 = alloc_storage(32, 256, cpu:0)
CALL dst=r3, @vm.builtin.make_shape,     [r_8]                     // shape = (8,)
CALL dst=r4, @vm.builtin.alloc_tensor,   [r2, r_0, r3, r_0, r_32, r_1] // tokens
CALL dst=-1, @runtime.tokenizer.encode,  [r0, r4]                  // packed_func 调用
RET  src=r4                                                         // return tokens
```

注意 `dst=-1`：DPS 调用不产生 SSA result，函数直接写入 `r4`（tokens 的寄存器）。

### Step 7：VM 执行时的 PackedFunc 调用链

```
VMState::ExecuteLoop()
    遇到 CALL @runtime.tokenizer.encode (kind=packed_func)
        ↓
    VMState::DispatchExternal(callee, args)
        ↓
    PackedFuncRegistry::Global().Get("runtime.tokenizer.encode")
        ↓
    PackedFunc.Call(PackedArgs([text_tensor, tokens_tensor]))
        ↓
    用户注册的 C++ 函数体执行，写入 tokens_tensor
```

如果 `runtime.tokenizer.encode` 没有注册，VM 会立刻抛出：
```
RuntimeError: PackedFunc 'runtime.tokenizer.encode' not registered
```

---

## 6. output=None 的特殊情况：effect-only 调用

有些 packed_func 没有输出 tensor，只有副作用：

```python
@dp.function
def update_cache(k_cache, v_cache, k_new, v_new, pos):
    dp.call_dps_packed(
        "runtime.update_kvcache",
        inputs=[k_cache, v_cache, k_new, v_new, pos],
        output=None,    # ← 没有输出 buffer
    )
    return k_cache
```

IR 中：
```
CallDPSOp(callee="runtime.update_kvcache",
          callee_kind=packed_func,
          inputs=(...), output=None, effect=OpaqueEffect)
```

字节码：
```
CALL dst=-1, @runtime.update_kvcache, [r_k_cache, r_v_cache, r_k_new, r_v_new, r_pos]
```

**DCE 保护**：因为 `effect=OpaqueEffect`，Dead Code Elimination pass 不会删除这条指令，即使其结果没人使用。

---

## 7. ABI Artifact：required_packed_funcs

编译产物的 `abi.json` 中会列出所有必须注册的 packed_func：

```json
{
  "devproc_abi_version": "0.1",
  "required_packed_funcs": [
    "runtime.tokenizer.encode",
    "runtime.update_kvcache"
  ]
}
```

`Executable::Load()` 在加载时会检查这些函数是否都已注册，如果缺失立刻报错：

```
RuntimeError: PackedFunc 'runtime.tokenizer.encode' is required but not registered.
```

这是一个 **fail-fast 设计**：比运行时崩溃更友好，错误信息明确指出缺什么。

---

## 8. 完整 Demo：分词 + 推理

下面是一个完整的端到端例子，把上述所有知识串起来。

### 8.1 C++ 侧：注册 mock 分词器

```cpp
// mock_tokenizer.cc
#include "devproc2/runtime/packed_func.h"
#include "devproc2/runtime/tensor.h"

DEVPROC2_REGISTER_PACKED_FUNC("runtime.tokenizer.encode")
    .set_body([](devproc2::PackedArgs args) {
        // args[0] = 输入文本 Tensor（int32，存 ASCII 码）
        // args[1] = 输出 tokens Tensor（int32）

        auto* in  = args[0].AsObjectAs<devproc2::TensorObj>();
        auto* out = args[1].AsObjectAs<devproc2::TensorObj>();

        int n = static_cast<int>(out->shape()[0]);
        int32_t* src = static_cast<int32_t*>(in->dl().data);
        int32_t* dst = static_cast<int32_t*>(out->dl().data);

        for (int i = 0; i < n; ++i) {
            dst[i] = src[0] + i;  // 简单 mock：token = first_char + position
        }
    });
```

### 8.2 Python 侧：DSL 定义 + 执行

```python
import struct
import devproc2.frontend.dsl as dp
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.compiler.pass_context import PassContext
from devproc2.kernel.registry import KernelRegistry
from devproc2.vm.interpreter import VMInterpreter, _Storage, _Tensor

# ① 定义 DSL 函数
@dp.function
def tokenize(text: dp.Tensor[(1,), "int32", "cpu"]):
    tokens = dp.empty((8,), dtype="int32", device="cpu")
    dp.call_dps_packed("runtime.tokenizer.encode", inputs=[text], output=tokens)
    return tokens

# ② 编译 pipeline
module = dp.get_module()
module = InferStructInfoPass().run(module)
module = DPSLoweringPass(KernelRegistry()).run(module)
ctx = PassContext()
MemoryPlanningPass().run(module, ctx)
module = LowerTensorCreateToAllocPass(ctx).run(module)
exe = VMCodegenPass().run(module)

# ③ 注册 mock packed_func（Python 版本）
vm = VMInterpreter(exe)

def mock_encode(args):
    text_t, tokens_t = args[0], args[1]
    first = struct.unpack_from("<i", text_t.storage.data, text_t.offset)[0]
    for i in range(8):
        struct.pack_into("<i", tokens_t.storage.data, tokens_t.offset + i * 4, first + i)

vm.register_packed_func("runtime.tokenizer.encode", mock_encode)

# ④ 执行
in_storage = _Storage(bytearray(4), 1, 0)
struct.pack_into("<i", in_storage.data, 0, 65)  # ASCII 'A' = 65
in_tensor = _Tensor(in_storage, 0, (1,), 0, 32, 1)

result = vm.invoke("tokenize", [in_tensor])

# ⑤ 验证
for i in range(8):
    val = struct.unpack_from("<i", result.storage.data, result.offset + i * 4)[0]
    print(f"tokens[{i}] = {val}")
    # 期望：65, 66, 67, 68, 69, 70, 71, 72
```

运行结果：
```
tokens[0] = 65
tokens[1] = 66
tokens[2] = 67
...
tokens[7] = 72
```

---

## 9. 设计要点总结

| 设计决策 | 原因 |
|---|---|
| DPS 调用约定（调用者分配输出） | 内存由 Memory Planner 统一管理，避免运行时碎片化分配 |
| PackedFunc 用名字注册/查找 | 解耦编译时与运行时，允许 C++/Python 双侧注册 |
| output=None 不被 DCE | `effect=OpaqueEffect` 保证 side-effecting 调用不被删除 |
| ABI 中声明 required_packed_funcs | Fail-fast：加载时检查依赖，避免运行时神秘崩溃 |
| `dst_reg=-1` 的 CALL 指令 | DPS 调用的结果已经在 output 寄存器里，无需额外寄存器 |

---

## 10. 与其他组件的关系

```
PackedFunc
   ├── 被 VM（vm.cc DispatchExternal）调用
   ├── 在 ABI（abi.json）中声明依赖
   ├── 在 IR（CallDPSOp callee_kind=packed_func）中表示
   └── 在 DSL（dp.call_dps_packed + dp.empty）中编写

dp.empty()
   ├── 生成 TensorCreateOp（高层 IR）
   ├── 被 InferStructInfoPass 赋予 TensorStructInfo
   └── 被 LowerTensorCreateToAllocPass 替换为 alloc_storage + alloc_tensor
```
