# M12：端到端 Demo 设计文档

> 适合读者：想从整体视角看懂 devproc2 工作原理的初学者。
>
> 读完本文，你将看到：一段 Python DSL 函数是如何一步步被编译、打包、执行，
> 最终产出正确的张量结果。全程用 CPU + numpy mock，不需要 GPU。

---

## 1. 目标：一个 Mini LLM Decode Step

M12 的目标是跑通一个"迷你版 LLM decode 步骤"，覆盖 M1-M11 所有核心特性。

真实的 LLM decode 步骤大概是：

```
输入 token id
   → embedding 查表（PackedFunc）
   → layernorm / attention / feedforward（各种 kernel）
   → 输出 logits
```

M12 demo 用一个简化版：

```
token_id (int32, shape=[1])
   → embedding 查表（PackedFunc: runtime.embed）
   → relu 激活（kernel: kernel.relu_fp32）
   → 线性投影（PackedFunc: runtime.linear）
   → output (float32, shape=[8])
```

这三步虽然简单，但覆盖了所有关键组件：

| 组件 | 覆盖的特性 |
|---|---|
| `runtime.embed` PackedFunc | M10: call_dps_packed + dp.empty() |
| `kernel.relu_fp32` kernel | M11: @dp.kernel + DPS kernel dispatch |
| `runtime.linear` PackedFunc | M10: 无输出 SSA result 的调用 |
| 完整 pipeline | M6-M9: DPS lowering + Memory Planning + VMCodegen |
| Artifact 产物 | M9: abi.json + executable.vm |

---

## 2. 仓库结构

```
examples/
  kv_cache_mvp/
    __init__.py
    run.py         ← 主入口，完整 demo + emit_artifact 函数
    ref_impl.py    ← numpy 参考实现（用于验证精度）
devproc_cli.py     ← 命令行工具，inspect artifact 目录
```

---

## 3. 第一步：用 DSL 写模型

```python
# examples/kv_cache_mvp/run.py

import devproc2.frontend.dsl as dp
from ref_impl import EMBED_DIM, OUTPUT_DIM

@dp.function
def decode_step(token_id: dp.Tensor[(1,), "int32", "cpu"]):
    # ① embedding 查表（PackedFunc 写入预分配的 buffer）
    embedded = dp.empty((EMBED_DIM,), dtype="float32", device="cpu")
    dp.call_dps_packed("runtime.embed", inputs=[token_id], output=embedded)

    # ② relu 激活（kernel 调用）
    relu_out = dp.ops.relu(embedded)

    # ③ 线性投影（PackedFunc 写入预分配的 buffer）
    output = dp.empty((OUTPUT_DIM,), dtype="float32", device="cpu")
    dp.call_dps_packed("runtime.linear", inputs=[relu_out], output=output)

    return output
```

**关键点**：
- `@dp.function` 不会**执行**这段代码，而是**解析 AST**，把每一行转成 IR 节点
- `EMBED_DIM = 8` 是 Python 整数常量，DSL 会通过 `fn.__globals__` 查到它并转成 `IntImm(8)`
- `dp.empty(...)` → `TensorCreateOp`，`dp.call_dps_packed(...)` → `CallDPSOp`，`dp.ops.relu(...)` → `CallOp`

---

## 4. 第二步：编译 Pipeline（7 道工序）

```python
def compile_model():
    module = dp.get_module()  # 取出 @dp.function 注册的 IRModule

    # 注册 relu kernel（DPS lowering 时查找用）
    relu_spec = KernelSpec(
        op_name="relu", device="cpu",
        input_dtypes=("float32",),
        kernel_name="kernel.relu_fp32",
    )
    kernel_registry = KernelRegistry()
    kernel_registry.register(relu_spec)

    # Pass 1：推导 struct_info
    module = InferStructInfoPass().run(module)

    # Pass 2：DPS lowering（CallOp → TensorCreateOp + CallDPSOp）
    module = DPSLoweringPass(kernel_registry).run(module)

    # Pass 3+4：内存规划 + lower alloc
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)

    # Pass 5：生成 VM 字节码
    exe = VMCodegenPass().run(module)

    return exe, ctx, module
```

下面逐 Pass 看 IR 变化。

### Pass 1：InferStructInfoPass

**变化**：给每个值打上 TensorStructInfo（shape + dtype + device）。

关键新增：`TensorCreateOp` 的结果也被赋予 struct_info，这样后续 DPS lowering 才知道 `embedded` 的 dtype 是 `float32`，能正确查到 `relu_fp32` kernel。

```
Before:
  TensorCreateOp("embedded", shape=(8,), dtype="float32", device="cpu")
    → result.struct_info = None  ← 没有类型信息

After:
  TensorCreateOp("embedded", ...)
    → result.struct_info = TensorStructInfo(shape=(IntImm(8),), dtype="float32", device="cpu")
```

### Pass 2：DPSLoweringPass

**变化**：`CallOp @relu` → `TensorCreateOp("relu_out") + CallDPSOp("kernel.relu_fp32")`。

```
Before:
  %relu_out = CallOp(@relu, args=(%embedded,))

After:
  %relu_out = TensorCreateOp(shape=(8,), dtype="float32", device="cpu")
  CallDPSOp("kernel.relu_fp32",
            callee_kind=kernel,
            inputs=(%embedded,), output=%relu_out,
            effect=OpaqueEffect)
```

这时 IR 中完全没有普通 `CallOp` 了，全部都是 `TensorCreateOp` 或 `CallDPSOp`。

### Pass 3：MemoryPlanningPass（不修改 IR）

**分析**生命周期，产生 `StoragePlan`，存入 `PassContext`：

```
%embedded   live: [instr 0, instr 2]   size=32 bytes  device=cpu
%relu_out   live: [instr 2, instr 4]   size=32 bytes  device=cpu
%output     live: [instr 3, instr 5]   size=32 bytes  device=cpu
```

贪心分配结果：
```json
{
  "storage_plan": [
    {"id": 0, "size_bytes": 32, "reused_by": ["embedded", "output"]},
    {"id": 1, "size_bytes": 32, "reused_by": ["relu_out"]}
  ]
}
```

`embedded` 和 `output` **共享同一块内存**！因为它们的生命周期不重叠（`embedded` 用完时 `output` 还没开始）。

### Pass 4：LowerTensorCreateToAllocPass

**变化**：`TensorCreateOp` → `AllocStorageOp + AllocTensorOp`。

```
Before:
  %embedded = TensorCreateOp(shape=(8,), dtype="float32", device="cpu")
  %relu_out = TensorCreateOp(shape=(8,), dtype="float32", device="cpu")
  %output   = TensorCreateOp(shape=(8,), dtype="float32", device="cpu")

After（根据 StoragePlan）：
  %s0       = AllocStorageOp(size=32, alignment=256, device="cpu")  # embedded + output 共享
  %s1       = AllocStorageOp(size=32, alignment=256, device="cpu")  # relu_out 专用
  %embedded = AllocTensorOp(%s0, offset=0, shape=(8,), dtype="float32")
  %relu_out = AllocTensorOp(%s1, offset=0, shape=(8,), dtype="float32")
  %output   = AllocTensorOp(%s0, offset=0, shape=(8,), dtype="float32")  # 复用 s0！
```

两个 `alloc_storage` 提升到函数顶部（只分配一次），对应 VM 的 session-level 内存。

### Pass 5：VMCodegenPass

**变化**：全部翻译为 VM 字节码。

```
# decode_step 函数字节码（伪代码）
r0 = param[0]                                      # token_id

# alloc_storage & alloc_tensor for embedded
CALL r2, @vm.builtin.alloc_storage, [32, 256, 1, 0]     # s0 = cpu:0
CALL r4, @vm.builtin.make_shape,    [8]
CALL r5, @vm.builtin.alloc_tensor,  [r2, 0, r4, 0, 32, 1]  # embedded

# alloc_storage & alloc_tensor for relu_out
CALL r6, @vm.builtin.alloc_storage, [32, 256, 1, 0]     # s1 = cpu:0
CALL r8, @vm.builtin.alloc_tensor,  [r6, 0, r4, 0, 32, 1]  # relu_out

# alloc_tensor for output（复用 s0）
CALL r9, @vm.builtin.alloc_tensor,  [r2, 0, r4, 0, 32, 1]  # output = s0

# Step 1: embed
CALL -1, @runtime.embed,        [r0, r5]            # packed_func

# Step 2: relu
CALL -1, @kernel.relu_fp32,     [r5, r8]            # kernel

# Step 3: linear
CALL -1, @runtime.linear,       [r8, r9]            # packed_func

RET r9
```

---

## 5. 第三步：实现 mock 函数

由于没有 GPU 和真实分词器，我们用 Python/numpy mock 代替。

```python
# examples/kv_cache_mvp/run.py

def register_mock_packed_funcs(vm):
    def embed_fn(args):
        """从 EMBED_WEIGHT 查表，写入 embedded buffer。"""
        tok_t, out_t = args[0], args[1]
        token_id = struct.unpack_from("<i", tok_t.storage.data, tok_t.offset)[0]
        row = EMBED_WEIGHT[token_id % VOCAB_SIZE]   # shape: (EMBED_DIM,)
        out_data = pack_f32(row)
        out_t.storage.data[out_t.offset : out_t.offset + len(out_data)] = out_data

    def linear_fn(args):
        """矩阵乘法，写入 output buffer。"""
        h_t, out_t = args[0], args[1]
        hidden = unpack_f32(h_t.storage.data, HIDDEN_DIM, h_t.offset)
        result = hidden @ LINEAR_WEIGHT   # (HIDDEN_DIM,) × (HIDDEN_DIM, OUTPUT_DIM)
        out_t.storage.data[out_t.offset : out_t.offset + len(pack_f32(result))] = pack_f32(result)

    vm.register_packed_func("runtime.embed",  embed_fn)
    vm.register_packed_func("runtime.linear", linear_fn)


def register_mock_kernel(vm):
    def relu_fn(args):
        """element-wise relu，写入 out buffer。"""
        in_t, out_t = args[0], args[1]
        vals   = unpack_f32(in_t.storage.data, EMBED_DIM, in_t.offset)
        result = vals.clip(min=0.0)
        out_t.storage.data[out_t.offset : out_t.offset + len(pack_f32(result))] = pack_f32(result)

    vm.register_kernel("kernel.relu_fp32", relu_fn)
```

---

## 6. 第四步：执行 + 验证精度

```python
def run_demo(token_id: int = 5) -> float:
    dp.reset_module()
    build_model()                               # 注册 @dp.function
    exe, ctx, inferred_module = compile_model() # 编译

    vm = VMInterpreter(exe)
    register_mock_packed_funcs(vm)
    register_mock_kernel(vm)

    # 构造输入 tensor
    in_storage = _Storage(bytearray(4), 1, 0)
    struct.pack_into("<i", in_storage.data, 0, token_id)
    in_tensor = _Tensor(in_storage, 0, (1,), 0, 32, 1)

    # 执行
    result = vm.invoke("decode_step", [in_tensor])
    vm_output = unpack_f32(result.storage.data, OUTPUT_DIM, result.offset)

    # 与 numpy 参考实现对比
    ref_output = reference_decode_step(token_id)
    max_err = float(abs(vm_output - ref_output).max())
    return max_err

if __name__ == "__main__":
    max_err = run_demo(token_id=5)
    if max_err < 1e-3:
        print(f"PASS: max error = {max_err:.2e}")
    else:
        print(f"FAIL: max error = {max_err:.2e}")
```

运行：
```
$ python examples/kv_cache_mvp/run.py
PASS: max error = 0.00e+00
```

误差为 0，因为 VM 和 numpy 用的完全是同一个计算图（都是整数查表 + float32 矩阵乘），没有浮点精度差异。

---

## 7. 第五步：生成 Artifact

```python
def emit_artifact(output_dir: str) -> None:
    dp.reset_module()
    build_model()
    exe, ctx, inferred_module = compile_model()

    # 写 executable.vm
    EmitExecutablePass().run(exe, output_dir)

    # 写 abi.json + manifest.json + metadata/*.json
    EmitABIPass().run(inferred_module, exe, ctx, output_dir,
                      model_name="kv_cache_demo", target="cpu")
```

产物目录结构：

```
output_dir/
  executable.vm              ← VM 字节码（二进制，magic="DV2E"）
  abi.json                   ← ABI 描述（输入/输出/依赖）
  manifest.json              ← 版本、构建时间、target
  metadata/
    function_table.json      ← 所有函数条目
    kernel_table.json        ← kernel 函数条目
    packed_func_table.json   ← packed_func 函数条目
    storage_plan.json        ← 内存复用方案
    shape_constraints.json   ← shape 约束（如 B <= 8）
```

`abi.json` 示例：

```json
{
  "devproc_abi_version": "0.1",
  "target": "cpu",
  "inputs": [
    {"name": "token_id", "dtype": "int32", "shape": [1]}
  ],
  "outputs": [
    {"dtype": "float32", "shape": [8]}
  ],
  "shape_constraints": {},
  "required_packed_funcs": [
    "runtime.embed",
    "runtime.linear"
  ]
}
```

---

## 8. 第六步：用 CLI 检查 Artifact

```bash
$ python devproc_cli.py inspect /tmp/kv_cache_demo/
```

输出：

```
devproc2 artifact: /tmp/kv_cache_demo

────────────────────────────────────────────────────────────
  Manifest
────────────────────────────────────────────────────────────
{
  "name": "kv_cache_demo",
  "version": "0.1.0",
  "build_time": "2026-05-15T10:23:45Z",
  "target": "cpu",
  "target_arch": ""
}

────────────────────────────────────────────────────────────
  ABI
────────────────────────────────────────────────────────────
{
  "devproc_abi_version": "0.1",
  "target": "cpu",
  "inputs":  [{"name": "token_id", "dtype": "int32", "shape": [1]}],
  "outputs": [{"dtype": "float32", "shape": [8]}],
  "required_packed_funcs": ["runtime.embed", "runtime.linear"]
}

────────────────────────────────────────────────────────────
  Function Table
────────────────────────────────────────────────────────────
[
  {"name": "decode_step",     "kind": "vm_func",     "num_regs": 14},
  {"name": "runtime.embed",   "kind": "packed_func", "num_regs": 0},
  {"name": "kernel.relu_fp32","kind": "kernel",      "num_regs": 0},
  {"name": "runtime.linear",  "kind": "packed_func", "num_regs": 0}
]

executable.vm: 312 bytes
```

### CLI 的实现原理

```python
# devproc_cli.py

def cmd_inspect(artifact_dir: str) -> int:
    # 读 manifest.json、abi.json、metadata/function_table.json
    # 用 json.dumps(..., indent=2) 格式化打印
    # 最后报告 executable.vm 的大小
    ...
```

CLI 是纯文件读取，不需要加载 VM，任何机器都可以运行。

---

## 9. 整体数据流图

```
Python 代码
──────────────────────────────────────────────────────────
@dp.function def decode_step(token_id):
    embedded = dp.empty((8,), dtype="float32")
    dp.call_dps_packed("runtime.embed", ...)
    relu_out = dp.ops.relu(embedded)
    output   = dp.empty((8,), dtype="float32")
    dp.call_dps_packed("runtime.linear", ...)
    return output
──────────────────────────────────────────────────────────
        │ AST 解析（不执行函数体）
        ▼
高层 IR（M2-M4）
  TensorCreateOp("embedded")
  CallDPSOp("runtime.embed", packed_func)
  CallOp("@relu")                    ← 还是高层形式
  TensorCreateOp("relu_out")
  TensorCreateOp("output")
  CallDPSOp("runtime.linear", packed_func)
  ReturnOp(output)
──────────────────────────────────────────────────────────
        │ InferStructInfoPass
        ▼  embedded.struct_info = TensorStructInfo(shape=(8,), float32, cpu)
        │ DPSLoweringPass
        ▼  CallOp("@relu") → TensorCreateOp("relu_out") + CallDPSOp("kernel.relu_fp32")
内存前 IR
  TensorCreateOp("embedded")
  CallDPSOp("runtime.embed")
  TensorCreateOp("relu_out")
  CallDPSOp("kernel.relu_fp32")
  TensorCreateOp("output")
  CallDPSOp("runtime.linear")
──────────────────────────────────────────────────────────
        │ MemoryPlanningPass（不修改 IR，产生 StoragePlan）
        │   s0 (32B, cpu) → embedded + output（复用）
        │   s1 (32B, cpu) → relu_out
        │ LowerTensorCreateToAllocPass
        ▼  TensorCreateOp → AllocStorageOp + AllocTensorOp
内存显式 IR
  AllocStorageOp(s0, 32, cpu)
  AllocStorageOp(s1, 32, cpu)
  AllocTensorOp(embedded, s0)
  CallDPSOp("runtime.embed")
  AllocTensorOp(relu_out, s1)
  CallDPSOp("kernel.relu_fp32")
  AllocTensorOp(output, s0)    ← 复用 s0！
  CallDPSOp("runtime.linear")
──────────────────────────────────────────────────────────
        │ VMCodegenPass
        ▼
VM 字节码（Executable）
  CALL r2 @alloc_storage [32, 256, 1, 0]     → s0
  CALL r6 @alloc_storage [32, 256, 1, 0]     → s1
  CALL r5 @alloc_tensor  [r2, ...]           → embedded
  CALL -1 @runtime.embed [r0, r5]            → embed
  CALL r8 @alloc_tensor  [r6, ...]           → relu_out
  CALL -1 @kernel.relu_fp32 [r5, r8]         → relu
  CALL r9 @alloc_tensor  [r2, ...]           → output (s0!)
  CALL -1 @runtime.linear [r8, r9]           → linear
  RET r9
──────────────────────────────────────────────────────────
        │ VMInterpreter.invoke("decode_step", [token_id_tensor])
        ▼
寄存器文件执行（Python list[Any]）
  r2 = _Storage(32 bytes, cpu)        # s0
  r6 = _Storage(32 bytes, cpu)        # s1
  r5 = _Tensor(r2, offset=0, (8,))   # embedded
  mock_embed(r0, r5)                  # 写 embedded
  r8 = _Tensor(r6, offset=0, (8,))   # relu_out
  mock_relu(r5, r8)                   # relu
  r9 = _Tensor(r2, offset=0, (8,))   # output（同一个 _Storage r2！）
  mock_linear(r8, r9)                 # linear
  return r9
──────────────────────────────────────────────────────────
最终结果：output tensor（shape=(8,), float32）
         与 numpy 参考实现误差 = 0.00e+00 ✓
```

---

## 10. 参考实现（ref_impl.py）

参考实现用来验证 VM 输出是否正确。它完全用 numpy 实现相同的计算：

```python
# examples/kv_cache_mvp/ref_impl.py

VOCAB_SIZE  = 16
EMBED_DIM   = 8
HIDDEN_DIM  = 8
OUTPUT_DIM  = 8

# 固定权重（用固定随机种子，保证每次一样）
EMBED_WEIGHT  = np.random.default_rng(42).normal(0, 0.1, (16, 8)).astype(np.float32)
LINEAR_WEIGHT = np.random.default_rng(7 ).normal(0, 0.1, (8, 8)).astype(np.float32)

def reference_decode_step(token_id: int) -> np.ndarray:
    embedded = EMBED_WEIGHT[token_id % VOCAB_SIZE]   # lookup
    hidden   = np.maximum(embedded, 0.0)             # relu
    output   = hidden @ LINEAR_WEIGHT                # linear
    return output.astype(np.float32)
```

VM mock 函数使用**完全相同的权重**，所以两边计算结果一致，误差为 0。

---

## 11. 怎么跑

### 直接运行 demo

```bash
cd /path/to/devproc2
python examples/kv_cache_mvp/run.py
# PASS: max error = 0.00e+00
```

### 生成 Artifact 并 inspect

```python
from examples.kv_cache_mvp.run import emit_artifact
emit_artifact("/tmp/kv_cache_demo")
```

```bash
python devproc_cli.py inspect /tmp/kv_cache_demo
```

### 跑测试

```bash
pytest tests/compiler/test_m12_demo.py -v
```

---

## 12. 常见问题

**Q：为什么 `embedded` 和 `output` 能共享内存（storage reuse）？**

A：Memory Planner 做了生命周期分析：
- `embedded` 的 last_use 是 `runtime.embed` 调用时（之后不再用）
- `output` 的 first_def 是在 `embedded` 已经死了之后

两者的 live interval 不重叠，且 size 一样（都是 32 bytes），设备一样（cpu），所以 Memory Planner 让它们共享同一个 `AllocStorageOp`。这节省了一次内存分配。

---

**Q：`dp.ops.relu(embedded)` 是怎么变成 `kernel.relu_fp32` 的？**

A：`DPSLoweringPass` 在 `KernelRegistry` 里查找：
- `op_name = "relu"`（去掉 `@` 后）
- `device = "cpu"`（从 `embedded.struct_info.device` 得到）
- `input_dtypes = ("float32",)`（从 `embedded.struct_info.dtype` 得到）

能找到我们注册的 `relu_spec`，`kernel_name = "kernel.relu_fp32"`，于是替换。

---

**Q：如果忘记注册 `runtime.embed`，会发生什么？**

A：两个层面的保护：
1. `Executable::Load()` 读 `abi.json` 时发现 `runtime.embed` 在 `required_packed_funcs` 里但未注册，立刻报错（C++ 侧）
2. `VMInterpreter._dispatch_external` 找不到 `runtime.embed` 时，抛 `RuntimeError: PackedFunc 'runtime.embed' not registered`（Python 侧）

---

**Q：为什么选 CPU + numpy mock 而不是真 GPU？**

A：M12 的目标是验证**整个编译 + 执行流程的正确性**，而不是 GPU 性能。使用 numpy mock 有三个好处：
1. 无需 GPU 环境，任何机器都能跑
2. float32 计算精确，误差为 0，比 float16 GPU 计算更容易验证
3. 可以单步调试每一个 mock 函数，比真 GPU kernel 好 trace

真 GPU 路径（M11 `CUDAKernelLauncher_Launch`）已经实现，等有 GPU 环境时可以直接切换。
