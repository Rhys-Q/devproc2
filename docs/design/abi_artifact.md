# ABI + Artifact MVP 设计文档

## 1. 为什么需要 Artifact 打包？

### 1.1 问题背景

经过 M1–M8 的编译 pipeline，devproc2 已经能从 Python DSL 生成一个内存中的 `Executable` 对象，并在 Python `VMInterpreter` 上执行。但这只解决了"能跑"的问题，还远没有解决"能交付"的问题。

现实的部署场景要求：

- **编译与执行分离**：编译发生在开发机（Python 环境），执行发生在推理服务（C++ 进程，无 Python 依赖）
- **跨进程加载**：同一份编译产物要能在不同进程、不同时间反复加载
- **依赖自描述**：C++ runtime 在加载产物时，需要知道这份产物需要哪些外部函数（tokenizer、packed_func），并在启动时验证它们是否已注册
- **版本兼容性检查**：编译器版本和 runtime 版本可能不同，需要在加载时尽早发现不兼容并给出明确错误

没有 Artifact 层，每次部署都要重新从 Python 编译，无法做到"编译一次，到处运行"。

### 1.2 设计目标

| 目标 | 说明 |
|------|------|
| **自描述** | 产物目录本身包含运行所需的全部元数据，不依赖外部数据库 |
| **可独立加载** | C++ `Executable::Load(dir)` 一个调用完成所有验证和加载 |
| **版本安全** | 主版本不兼容时立即报错，不会静默出错 |
| **依赖前置检查** | 加载时检查所有 required_packed_funcs，而非等到运行时才崩溃 |
| **人类可读** | JSON 格式的元数据方便调试工具直接查看 |
| **可扩展** | M11 可以直接在 `kernels/` 目录下放 `.cubin` 文件，M10 的 packed_func 已经有占位符 |

---

## 2. 在编译 Pipeline 中的位置

```
Python DSL
   │
   ▼  @dp.function 装饰器捕获
High-level IR（CallOp）
   │
   ▼  InferStructInfoPass      — 推导 shape/dtype/device，填充 TensorStructInfo
   ▼  DPSLoweringPass          — CallOp → TensorCreateOp + CallDPSOp
   │
   ▼  MemoryPlanningPass       — 生命周期分析，生成 StoragePlan → PassContext
   ▼  LowerTensorCreateToAllocPass  — TensorCreateOp → AllocStorageOp + AllocTensorOp
   │
   ▼  VMCodegenPass            — Memory-explicit IR → Executable（bytecode）
   │
   ▼  【M9: ABI + Artifact 层】   ← 本文重点
   │     EmitExecutablePass       — 序列化 Executable → executable.vm
   │     EmitABIPass              — 从 IRModule + Executable + PassContext
   │                                提取元数据 → abi.json / manifest.json / metadata/
   │
   ▼  build/<model_name>/         ← 落盘的 Artifact 目录
         │
         ▼  C++ Executable::Load(artifact_dir)
               ├── Deserialize(executable.vm)
               ├── 读 abi.json → 验证版本 + 检查 packed_func 注册
               └── 返回 shared_ptr<Executable>，可直接传入 VMState
```

M9 是**编译器的最后一步**，也是 C++ runtime 的**入口第一步**。它们通过磁盘上的 Artifact 目录握手。

---

## 3. Artifact 目录结构

```
build/<model_name>/
  manifest.json             # 包元数据（名称、版本、构建时间、target arch）
  abi.json                  # ABI 契约（版本、input/output 类型、shape 约束、依赖的 packed_func）
  executable.vm             # VM bytecode 二进制（DV2E 格式）
  constants/                # 权重/常量 blobs（M11 填充；MVP 为空目录占位）
  kernels/                  # cubin 文件（M11 填充；MVP 为空目录占位）
  metadata/
    function_table.json     # Executable.function_table 的 JSON 镜像
    kernel_table.json       # function_table 中 kind==kernel 的子集
    packed_func_table.json  # function_table 中 kind==packed_func 的子集
    storage_plan.json       # M7 MemoryPlanningPass 的 StoragePlan 结果
    shape_constraints.json  # 所有 SymbolicDim 的 upper bound 约束
```

**设计原则：**
- `executable.vm` 是 runtime **必须**读取的文件，包含完整可执行信息
- `abi.json` 是 runtime 在执行前做**验证**的文件，只读取两个字段
- `metadata/` 下的文件是**可选的**调试辅助信息，runtime 不依赖它们
- `constants/` 和 `kernels/` 是**占位目录**，M11 会填充 cubin 和权重

---

## 4. 二进制格式：executable.vm

### 4.1 设计动机

`executable.vm` 是跨语言边界的核心媒介：Python 编译器写入，C++ runtime 读取。需要一个紧凑、可校验、向前兼容的格式。

选择**自定义二进制格式**而不是 JSON 或 protobuf 的原因：
- **紧凑**：bytecode 指令密集，JSON 编码会放大 5-10×
- **版本校验**：magic bytes 在最开头，corrupt 文件立即被检测
- **零依赖**：不引入任何序列化库

### 4.2 整体布局

```
┌─────────────────────────────────────────────────────┐
│  Magic (4 bytes)    b"DV2E"                         │
│  Version (4 bytes)  uint32 LE = 1                   │
│  num_funcs (4 bytes) uint32 LE                      │
│  num_instrs (4 bytes) uint32 LE                     │
│  num_consts (4 bytes) uint32 LE                     │
├─────────────────────────────────────────────────────┤
│  Function Table                                      │
│    FunctionEntry × num_funcs                        │
├─────────────────────────────────────────────────────┤
│  Instructions                                        │
│    Instruction × num_instrs                         │
├─────────────────────────────────────────────────────┤
│  Constants                                           │
│    ConstantRecord × num_consts                      │
└─────────────────────────────────────────────────────┘
```

所有多字节整数均为**小端序（little-endian）**。

### 4.3 FunctionEntry 编码

每个 `FunctionEntry` 编码如下：

```
name_len  : uint32          函数名字节数
name      : bytes           UTF-8 编码的函数名
kind      : uint8           CalleeKind（0=vm_func, 1=builtin, 2=packed_func, 3=kernel）
instr_offset : int32        在 instructions 数组中的起始偏移；外部函数为 -1
instr_count  : int32        指令数量；外部函数为 0
num_regs  : int32           该函数的寄存器总数
num_args  : int32           调用参数数量（前 num_args 个寄存器存放参数）
n_ci      : int32           const_inits 数量
ConstInit × n_ci:
    reg_idx   : int32       目标寄存器编号
    const_idx : int32       常量池索引
```

**注意**：`kind != vm_func` 的外部函数（builtin / packed_func / kernel）也存在函数表中，`instr_offset = -1`，`instr_count = 0`。这样函数索引统一，CALL 指令只需要一个 `func_idx` 就能找到被调函数的所有元数据。

### 4.4 Instruction 编码

每条 `Instruction` 是固定头部加可变长 `arg_regs`：

```
opcode       : uint8        Opcode（0=CALL, 1=RET, 2=IF, 3=GOTO）
dst_reg      : int32        CALL 结果寄存器（-1=无返回值）
func_idx     : int32        CALL 被调函数在函数表中的索引
src_reg      : int32        RET 返回值寄存器（-1=void return）
cond_reg     : int32        IF 条件寄存器
true_offset  : int32        IF 条件为 True 时的 pc 相对偏移
false_offset : int32        IF 条件为 False 时的 pc 相对偏移
offset       : int32        GOTO 的 pc 相对偏移（可以为负，用于循环回跳）
nargs        : uint32       arg_regs 数组长度
arg_regs     : int32 × nargs  CALL 参数的寄存器编号列表
```

所有指令字段都存在，即使当前 opcode 不使用某些字段——这简化了序列化逻辑，代价是每条指令多占几个字节。

**固定头部大小**：`1 + 7×4 + 4 = 33 bytes`，加上 `nargs × 4` 字节的 arg_regs。

### 4.5 常量池编码

每个常量是 **tag-length-value** 格式。目前支持 5 种类型：

| tag | 类型 | 编码 | 说明 |
|-----|------|------|------|
| 0 (`NULL`) | None / null | `1 byte tag + 8 bytes 填充` | 固定 9 字节 |
| 1 (`INT`) | int64 | `1 byte tag + 8 bytes int64 LE` | 固定 9 字节 |
| 2 (`FLOAT`) | float64 | `1 byte tag + 8 bytes double LE` | 固定 9 字节 |
| 3 (`BOOL`) | bool | `1 byte tag + 8 bytes int64 LE`（0 或 1）| 固定 9 字节 |
| 4 (`STR`) | string | `1 byte tag + 4 bytes uint32 长度 + N bytes UTF-8` | 可变长 |

NULL/INT/FLOAT/BOOL 统一为 9 字节，方便快速随机访问（如有需要）。STR 是可变长的，用于 `assert_le_i64` 等携带错误消息的 builtin。

**C++ 侧的 STR 处理**：`VMValue` 没有字符串类型，反序列化时 STR 被存为 `VMValue{}（null）`。这是有意为之——`assert_le_i64` 的消息已在 Python 解释器中用于调试，C++ 侧的 builtin 若检测到 `args[2]` 不是 `StringObj` 则降级使用通用错误消息。

### 4.6 Python 序列化示例

```python
from devproc2.vm import serializer

data = serializer.serialize(exe)   # Executable → bytes
exe2 = serializer.deserialize(data)  # bytes → Executable（完全等价）
```

Python 使用 `struct` 模块以小端序直接打包，无填充。关键格式字符串：

```python
# 文件头
struct.pack("<III", VERSION, num_funcs, num_instrs)
struct.pack("<I",   num_consts)

# FunctionEntry
struct.pack("<I",      len(name_bytes)) + name_bytes
struct.pack("<Biiiii", kind, instr_offset, instr_count, num_regs, num_args, n_ci)
struct.pack("<ii",     ci.reg_idx, ci.const_idx)   # 每个 ConstInit

# Instruction
struct.pack("<BiiiiiiiI",
            opcode, dst_reg, func_idx, src_reg,
            cond_reg, true_offset, false_offset, offset,
            len(arg_regs))
struct.pack("<i", r)   # 每个 arg_reg

# 常量（NULL 示例）
struct.pack("<B8x", TAG_NULL)
# 常量（INT 示例）
struct.pack("<Bq", TAG_INT, value)
# 常量（STR 示例）
s_b = msg.encode()
struct.pack("<BI", TAG_STR, len(s_b)) + s_b
```

---

## 5. EmitExecutablePass：序列化 Pass

### 5.1 职责

`EmitExecutablePass` 是编译 pipeline 中最后一个 pass，职责极为单一：

1. 调用 `serializer.serialize(exe)` 将内存中的 `Executable` 序列化为二进制
2. 将二进制写入 `<output_dir>/executable.vm`
3. 返回 `bytes`，方便测试做内存中的 round-trip 验证

```python
class EmitExecutablePass:
    def run(self, exe: Executable, output_dir: str) -> bytes:
        data = serializer.serialize(exe)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "executable.vm"), "wb") as f:
            f.write(data)
        return data
```

### 5.2 为什么单独成 Pass？

`serializer.serialize()` 已经完成了所有实质工作。之所以单独封装成一个 Pass，是为了：

- 和 `EmitABIPass` 保持接口一致（都接受 output_dir）
- 便于在 pipeline 中插拔：调试时可以只跑到 VMCodegen 不写磁盘
- 返回 `bytes` 使单元测试不依赖磁盘，直接做 round-trip 验证

---

## 6. ABI JSON 设计

### 6.1 abi.json 完整结构

```json
{
  "devproc_abi_version": "0.1",
  "vm_bytecode_version": "0.1",
  "kernel_calling_convention": "dps_kernel_v1",
  "packed_func_calling_convention": "dps_packed_v1",
  "target": "cuda",
  "target_arch": "sm_80",
  "inputs": [
    {
      "name": "x",
      "dtype": "float16",
      "shape": ["B", "S", 4096],
      "device": "cuda"
    }
  ],
  "outputs": [
    {
      "dtype": "float16",
      "shape": ["B", "S", 4096],
      "device": "cuda"
    }
  ],
  "shape_constraints": {
    "B": {"upper": 8},
    "S": {"upper": 2048}
  },
  "required_packed_funcs": [
    "runtime.tokenizer.encode"
  ]
}
```

### 6.2 各字段说明

| 字段 | 来源 | 含义 |
|------|------|------|
| `devproc_abi_version` | 硬编码 `"0.1"` | C++ 加载器版本兼容性检查的依据（见 §9） |
| `vm_bytecode_version` | 同上 | executable.vm 格式版本（当前与 ABI 版本同步） |
| `kernel_calling_convention` | 硬编码 | kernel DPS 调用约定标识符，M11 会使用 |
| `packed_func_calling_convention` | 硬编码 | packed_func DPS 调用约定标识符，M10 会使用 |
| `target` | `EmitABIPass.run()` 参数 | 编译目标平台（`"cuda"` 或 `"cpu"`） |
| `target_arch` | `EmitABIPass.run()` 参数 | 具体架构，如 `"sm_80"`（M9 MVP 可为空字符串） |
| `inputs` | `IRModule["main"].params` | 函数入参的类型信息 |
| `outputs` | `IRModule["main"].ret_struct_info` | 函数返回值的类型信息 |
| `shape_constraints` | 遍历 inputs/outputs 的 `TensorStructInfo.shape` | 所有 `PrimVar` 的 upper bound 约束 |
| `required_packed_funcs` | `Executable.function_table` 中 `kind==packed_func` 的项 | C++ 加载器必须验证这些函数已注册 |

### 6.3 输入输出类型信息的提取

`inputs` 和 `outputs` 来自编译后的 IRModule，而不是原始 DSL 的函数签名。这保证了类型信息是经过 `InferStructInfoPass` 推导和填充之后的精确信息。

```
Python DSL 函数签名
    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cpu"]):
        ...
         │
         ▼  InferStructInfoPass 处理后
IRModule["main"].params[0]
    → Var("x", TensorStructInfo(
          shape=(PrimVar("B", upper=8),
                 PrimVar("S", upper=2048),
                 IntImm(4096)),
          dtype="float16",
          device="cpu"
      ))
         │
         ▼  EmitABIPass._extract_abi_from_fn() 提取
abi["inputs"][0]
    → {"name": "x", "dtype": "float16",
       "shape": ["B", "S", 4096], "device": "cpu"}
```

**shape 维度的 JSON 转换规则**：

```python
def _shape_dim_to_json(dim):
    if isinstance(dim, IntImm):  # 静态维度 → int
        return dim.value         # e.g., 4096
    if isinstance(dim, PrimVar): # 符号维度 → str（变量名）
        return dim.name          # e.g., "B"
    return str(dim)              # 算术表达式 → 字符串表示
```

### 6.4 shape_constraints 的提取

`shape_constraints` 由递归遍历所有 inputs/outputs 的 `TensorStructInfo.shape` 中的 `PrimVar` 得到。只有携带 `upper` 的 `PrimVar` 才会出现在约束中（无界符号维度不写入）。

```python
def _collect_prim_vars_from_expr(expr, out: dict[str, Optional[int]]) -> None:
    if isinstance(expr, PrimVar):
        if expr.name not in out:      # 去重：同名 PrimVar 只记录一次
            out[expr.name] = expr.upper
        return
    # 递归处理复合表达式（Add / Sub / Mul / FloorDiv / CeilDiv / Min / Max）
    for cls in (Add, Sub, Mul, FloorDiv, CeilDiv, Min, Max):
        if isinstance(expr, cls):
            _collect_prim_vars_from_expr(expr.lhs, out)
            _collect_prim_vars_from_expr(expr.rhs, out)
            return
```

### 6.5 required_packed_funcs 的来源

`required_packed_funcs` 直接从 `Executable.function_table` 提取：

```python
def _extract_required_packed_funcs(self, exe: Executable) -> list[str]:
    return [
        fe.name
        for fe in exe.function_table
        if fe.kind == CalleeKind.packed_func
    ]
```

这里使用 **Executable 而不是 IRModule** 作为信息来源，原因是：

- `VMCodegenPass` 在生成 bytecode 时已经把所有 packed_func 调用汇总进了函数表
- 函数表中的 `packed_func` 条目正是 VM 运行时实际会调用的函数，没有遗漏
- 从 IRModule 中提取需要遍历所有 `CallDPSOp(callee_kind=packed_func)`，逻辑更复杂

---

## 7. manifest.json 设计

### 7.1 结构

```json
{
  "name": "kvcache_demo",
  "version": "0.1.0",
  "build_time": "2025-09-01T14:32:10Z",
  "target": "cuda",
  "target_arch": "sm_80"
}
```

### 7.2 字段说明

| 字段 | 说明 |
|------|------|
| `name` | 模型/产物名称，由调用方传入 `EmitABIPass.run(..., model_name=...)` |
| `version` | 产物版本，当前固定为 `"0.1.0"` |
| `build_time` | ISO 8601 UTC 时间戳，格式 `YYYY-MM-DDTHH:MM:SSZ` |
| `target` | 同 abi.json 的 `target` |
| `target_arch` | 同 abi.json 的 `target_arch` |

manifest.json 是人类可读的"标签"，C++ 加载器**不读取**它。它的价值在于：调试工具、日志、模型管理系统可以通过这个文件快速了解一份产物的基本信息。

---

## 8. metadata/ 文件族

metadata/ 下的所有文件都是**调试辅助信息**，C++ runtime 加载时不依赖它们。它们的存在意义是：

- 允许工具（如 `devproc_cli.py inspect`）直接读取人类可读的元数据，无需反解析二进制
- 帮助开发者验证编译器各阶段的输出是否符合预期

### 8.1 function_table.json

`Executable.function_table` 的完整 JSON 镜像。

```json
[
  {
    "name": "main",
    "kind": "vm_func",
    "instr_offset": 0,
    "instr_count": 12,
    "num_regs": 8,
    "num_args": 1
  },
  {
    "name": "vm.builtin.alloc_storage",
    "kind": "builtin",
    "instr_offset": -1,
    "instr_count": 0,
    "num_regs": 0,
    "num_args": 0
  },
  {
    "name": "kernel.relu_fp16",
    "kind": "kernel",
    "instr_offset": -1,
    "instr_count": 0,
    "num_regs": 0,
    "num_args": 0
  }
]
```

通过这份文件可以直接看到编译后的完整函数表，包括所有 builtin 和外部函数。

### 8.2 kernel_table.json

`function_table.json` 中 `kind == "kernel"` 的子集，方便快速了解产物依赖了哪些 CUDA kernel。M11 实现后，每个 kernel 在 `kernels/` 目录下对应一个 `.cubin` 文件。

### 8.3 packed_func_table.json

`function_table.json` 中 `kind == "packed_func"` 的子集。与 `abi.json` 中的 `required_packed_funcs` 对应，但包含更多细节（参数数量等）。

### 8.4 storage_plan.json

M7 `MemoryPlanningPass` 的 `StoragePlan` 结果，序列化后的形式：

```json
[
  {
    "id": 0,
    "device": "cuda",
    "size_bytes": 67108864,
    "alignment": 256,
    "reused_by": ["tmp0", "tmp3", "tmp7"]
  },
  {
    "id": 1,
    "device": "cuda",
    "size_bytes": 16777216,
    "alignment": 256,
    "reused_by": ["tmp1", "tmp5"]
  }
]
```

`reused_by` 列出了共享这块 storage 的所有张量名，顺序反映了时间上的复用顺序。`size_bytes` 是静态形状下的精确字节数（已按 256 字节对齐），动态形状（`size_bytes = null`）时表示运行时才能确定大小。

**注意**：`StorageEntry.size_expr`（PrimExpr 对象）不序列化到 JSON，因为它只在 LowerTensorCreateToAllocPass 内部使用，到 M9 阶段已经完成使命。

### 8.5 shape_constraints.json

与 `abi.json` 中的 `shape_constraints` 字段内容完全相同，单独存放方便直接引用：

```json
{
  "B": {"upper": 8},
  "S": {"upper": 2048}
}
```

---

## 9. EmitABIPass：元数据发射 Pass

### 9.1 接口

```python
class EmitABIPass:
    def run(
        self,
        module: IRModule,          # 经过 InferStructInfoPass 的 IRModule（含完整类型信息）
        exe: Executable,           # VMCodegenPass 生成的 Executable
        ctx: PassContext,          # 含 "storage_plan" 键
        output_dir: str,           # 输出目录
        model_name: str = "model", # manifest.json 中的名称
        target: str = "cpu",       # "cpu" 或 "cuda"
        target_arch: str = "",     # 如 "sm_80"
    ) -> None:
```

### 9.2 信息来源汇总

```
IRModule["main"].params
    → inputs（名称 + TensorStructInfo）
    → shape_constraints（递归收集 PrimVar）

IRModule["main"].ret_struct_info
    → outputs（TensorStructInfo）
    → shape_constraints（递归收集 PrimVar）

Executable.function_table
    → function_table.json（完整函数表）
    → kernel_table.json（kind==kernel 的子集）
    → packed_func_table.json（kind==packed_func 的子集）
    → required_packed_funcs（packed_func 的名称列表）

PassContext["storage_plan"]
    → storage_plan.json
```

### 9.3 目录创建策略

`EmitABIPass` 在写入前先创建所有必要目录：

```python
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "kernels"), exist_ok=True)    # M11 占位
os.makedirs(os.path.join(output_dir, "constants"), exist_ok=True)  # M11 占位
```

`kernels/` 和 `constants/` 在 MVP 阶段是空目录，M11（Triton cubin）和未来的常量张量功能会往里填充文件。

---

## 10. C++ 加载器：Executable::Load

### 10.1 接口

```cpp
// vm.h
class Executable {
public:
    // 从原始字节流反序列化（供测试和内存加载使用）
    static std::shared_ptr<Executable> Deserialize(const uint8_t* data, size_t size);

    // 从 artifact 目录加载，含 ABI 验证
    static std::shared_ptr<Executable> Load(const std::string& artifact_dir);
};
```

### 10.2 Load() 执行流程

```
Executable::Load(artifact_dir)
    │
    ├─ 1. 读取二进制
    │      read_file_binary(artifact_dir + "/executable.vm")
    │      → vector<uint8_t> vm_bytes
    │
    ├─ 2. 反序列化
    │      Deserialize(vm_bytes.data(), vm_bytes.size())
    │      → shared_ptr<Executable> exe
    │
    ├─ 3. 读取 ABI JSON
    │      read_file_text(artifact_dir + "/abi.json")
    │      → string abi_json
    │
    ├─ 4. 版本检查
    │      json_extract_string(abi_json, "devproc_abi_version")
    │      → "0.1"
    │      major = abi_version.substr(0, abi_version.find('.'))  →  "0"
    │      if (major != "0")
    │          throw "ABI version mismatch: expected major 0, got X"
    │
    ├─ 5. packed_func 依赖检查
    │      json_extract_string_array(abi_json, "required_packed_funcs")
    │      → ["runtime.tokenizer.encode", ...]
    │      for each name:
    │          if !PackedFuncRegistry::Global().Has(name)
    │              throw "PackedFunc 'name' is required but not registered."
    │
    └─ 6. 返回 exe
```

### 10.3 Deserialize() 实现

`Deserialize` 使用一个轻量的 `ByteReader` helper，通过 `memcpy` 逐字段读取（避免 unaligned access）：

```cpp
struct ByteReader {
    const uint8_t* data;
    size_t size, pos = 0;

    template <typename T>
    T read() {
        // 边界检查 + memcpy（解决对齐问题）
        T val{};
        std::memcpy(&val, data + pos, sizeof(T));
        pos += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint32_t len = read<uint32_t>();
        std::string s(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return s;
    }
};
```

关键设计选择：**使用 `memcpy` 而不是直接强转指针**，这避免了 UB（未定义行为）中的 strict aliasing 和对齐问题，在任何平台上都是安全的。

常量反序列化时各类型的处理：

```cpp
uint8_t tag = r.read<uint8_t>();
switch (tag):
    TAG_NULL  → skip(8); constants[i] = VMValue{};
    TAG_INT   → constants[i] = VMValue::Int(r.read<int64_t>());
    TAG_FLOAT → constants[i] = VMValue::Float(r.read<double>());
    TAG_BOOL  → constants[i] = VMValue::Bool((bool)r.read<int64_t>());
    TAG_STR   → slen = r.read<uint32_t>(); skip(slen);
                constants[i] = VMValue{};   // C++ 无 string VMValue，存为 null
    default   → throw "unknown constant tag"
```

### 10.4 JSON 解析：nlohmann/json

C++ 侧使用 [nlohmann/json](https://github.com/nlohmann/json)（`3rdparty/json`，git submodule，header-only），与 dlpack 引入方式一致。

```cpp
#include <nlohmann/json.hpp>

// Load() 中：
auto abi = nlohmann::json::parse(read_file_text(abi_path));

// 版本检查
std::string abi_version = abi.value("devproc_abi_version", std::string{});

// packed_func 依赖检查
for (const auto& name : abi.value("required_packed_funcs",
                                   nlohmann::json::array())) {
    std::string fn = name.get<std::string>();
    if (!PackedFuncRegistry::Global().Has(fn)) {
        throw std::runtime_error(
            "PackedFunc '" + fn + "' is required but not registered.");
    }
}
```

`abi.value("key", default)` 在字段不存在时返回默认值，解析出错时 nlohmann 直接抛出 `nlohmann::json::exception`，不需要手动处理边界情况。

选择 nlohmann/json 而非手写解析器，是因为后续 M11（cubin metadata）、M12 以及更多需要读写 JSON 的场景都会用到它，统一一个库比维护多份临时代码更合理。

**CMakeLists.txt 引入方式**（`CMakeLists.txt` 根目录）：

```cmake
# ── nlohmann/json (git submodule, header-only) ────────────────────────────────
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/single_include/nlohmann/json.hpp")
  message(FATAL_ERROR
    "nlohmann/json submodule not found. Run:\n"
    "  git submodule update --init --recursive")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/single_include)
```

`runtime/CMakeLists.txt` 链接：

```cmake
target_link_libraries(devproc2_runtime PUBLIC dlpack nlohmann_json)
```

### 10.5 ABI 版本兼容性设计

版本号格式为 `major.minor`：

```
"0.1"  →  major = "0", minor = "1"
```

**兼容性规则**：
- `major` 相同 → 兼容（minor 只增加新字段，不修改已有字段的语义）
- `major` 不同 → **不兼容**，Load() 立即抛出异常

```cpp
std::string actual_major = abi_version.substr(0, abi_version.find('.'));
if (actual_major != expected_major /* "0" */) {
    throw std::runtime_error(
        "ABI version mismatch: expected major " + expected_major
        + ", got " + actual_major + " (full version: " + abi_version + ")");
}
```

当 ABI 发生不兼容变更时（如指令格式改变、调用约定变更），递增 major 版本。minor 版本用于向后兼容的扩展（如新增 JSON 字段）。

### 10.6 PackedFunc 依赖检查

```cpp
auto required = json_extract_string_array(abi_json, "required_packed_funcs");
for (const auto& name : required) {
    if (!PackedFuncRegistry::Global().Has(name)) {
        throw std::runtime_error(
            "PackedFunc '" + name + "' is required but not registered.");
    }
}
```

这个检查在 `Load()` 时（即进程启动阶段）执行，而不是在第一次调用时执行。好处是：
- 问题在最早的时机暴露，错误信息明确（而不是在推理时某个神秘的 nullptr dereference）
- 应用开发者可以在服务启动时捕获这个错误，避免带着未满足依赖上线

---

## 11. 端到端示例

以一个含动态 shape 和 packed_func 的模型为例，演示完整的编译→打包→加载流程。

### 11.1 Python 编译端

```python
import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_abi import EmitABIPass
from devproc2.compiler.passes.emit_executable import EmitExecutablePass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.kernel.registry import KernelRegistry, KernelSpec

# 1. 用 DSL 定义模型
B = dp.symbolic_dim("B", upper=8)
S = dp.symbolic_dim("S", upper=2048)

@dp.function
def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
    y = dp.ops.layernorm(x)
    return y

module = dp.get_module()

# 2. 运行编译 pipeline
reg = KernelRegistry()
reg.register(KernelSpec(
    op_name="layernorm", device="cuda",
    input_dtypes=("float16",),
    kernel_name="kernel.layernorm_fp16",
))

module = InferStructInfoPass().run(module)
inferred_module = module                    # 保存给 EmitABIPass 用

module = DPSLoweringPass(reg).run(module)
ctx = PassContext()
MemoryPlanningPass().run(module, ctx)
module = LowerTensorCreateToAllocPass(ctx).run(module)
exe = VMCodegenPass().run(module)

# 3. 发射 Artifact
output_dir = "build/my_model"
EmitExecutablePass().run(exe, output_dir)
EmitABIPass().run(
    inferred_module, exe, ctx,
    output_dir,
    model_name="my_model",
    target="cuda",
    target_arch="sm_80",
)
```

### 11.2 生成的 Artifact 内容

```
build/my_model/
  manifest.json
  abi.json
  executable.vm      ← 二进制，首 4 字节是 b"DV2E"
  constants/         ← 空目录（占位）
  kernels/           ← 空目录（M11 会放 layernorm_fp16.cubin）
  metadata/
    function_table.json
    kernel_table.json
    packed_func_table.json
    storage_plan.json
    shape_constraints.json
```

**abi.json 内容**：

```json
{
  "devproc_abi_version": "0.1",
  "vm_bytecode_version": "0.1",
  "kernel_calling_convention": "dps_kernel_v1",
  "packed_func_calling_convention": "dps_packed_v1",
  "target": "cuda",
  "target_arch": "sm_80",
  "inputs": [
    {"name": "x", "dtype": "float16", "shape": ["B", "S", 4096], "device": "cuda"}
  ],
  "outputs": [
    {"dtype": "float16", "shape": ["B", "S", 4096], "device": "cuda"}
  ],
  "shape_constraints": {
    "B": {"upper": 8},
    "S": {"upper": 2048}
  },
  "required_packed_funcs": []
}
```

### 11.3 C++ 加载端

```cpp
#include "devproc2/runtime/vm.h"
#include "devproc2/runtime/packed_func.h"

int main() {
    // 注册 packed_func（如果 abi.json 中 required_packed_funcs 非空，
    // 必须在 Load() 之前注册完毕）
    // PackedFuncRegistry::Global().Register("runtime.tokenizer.encode", ...);

    // Load() = 反序列化 + ABI 版本检查 + packed_func 依赖检查
    auto exe = devproc2::Executable::Load("build/my_model");

    // 构建 VMState，传入 Executable
    devproc2::VMState vm(exe);

    // 准备输入 tensor（此处省略构建细节）
    devproc2::VMValue input_tensor = ...;

    // 执行推理
    devproc2::VMValue result = vm.Invoke("main", {input_tensor});

    return 0;
}
```

### 11.4 版本不匹配时的错误

假设用旧版产物（`devproc_abi_version: "1.0"`）在新 runtime 上加载：

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  ABI version mismatch: expected major 0, got 1 (full version: 1.0)
```

假设产物声明了 `required_packed_funcs: ["runtime.tokenizer.encode"]` 但 C++ 端未注册：

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  PackedFunc 'runtime.tokenizer.encode' is required but not registered.
```

两种错误都在 `Load()` 时立即抛出，不会等到实际执行时才崩溃。

---

## 12. 关键设计决策

### 12.1 为什么不用 protobuf / flatbuffers？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 自定义二进制 | 零依赖，格式完全可控 | 需要自己维护序列化代码 |
| protobuf | 成熟，有版本演化支持 | 引入重量级依赖，pb 文件需要代码生成步骤 |
| flatbuffers | 零拷贝访问 | 同上，而且对小文件收益不明显 |

devproc2 的 `Executable` 结构非常简单（函数表 + 指令数组 + 常量池），自定义格式 100 行以内搞定，维护成本低。等 schema 变复杂时再引入 protobuf 是合理的演化路径。

### 12.2 为什么 C++ 用 nlohmann/json 而不是手写解析器？

C++ 加载器需要从 abi.json 中读取：
1. `devproc_abi_version`：版本检查
2. `required_packed_funcs`：依赖检查

最初考虑过用 `std::string::find` 写 30 行临时解析器，但后续 M11（cubin metadata 读取）、M12 以及更多配置/元数据场景都需要 JSON 支持，统一使用 nlohmann/json 更合适。

nlohmann/json 是 header-only 库，以 git submodule 形式引入（`3rdparty/json`），与 dlpack 的引入方式完全一致，不增加构建复杂度。

### 12.3 为什么 EmitABIPass 读取 inferred_module 而不是最终 module？

```
inferred_module  ← InferStructInfoPass 之后，DPSLoweringPass 之前
                    params 有完整 TensorStructInfo，没有 alloc_* 干扰
```

DPSLoweringPass 之后的 module 引入了 `TensorCreateOp`、`CallDPSOp` 等；LowerTensorCreateToAllocPass 之后又变成了 `AllocStorageOp`、`AllocTensorOp`。这些变化不影响函数的 input/output 类型，但 IR 结构更复杂，遍历时更容易出错。

选择 `inferred_module` 的好处：
- params 依然是原始的 `Var(name, TensorStructInfo)`，直接读取，无需额外处理
- `ret_struct_info` 经过 `InferStructInfoPass` 已经填充完毕
- 完全不受后续 lowering 的影响

### 12.4 StoragePlan 的 size_expr 为何不序列化？

`StorageEntry.size_expr` 是一个 `PrimExpr` 对象，它在 `LowerTensorCreateToAllocPass` 中被写入 `AllocStorageOp`，进而被 `VMCodegenPass` 编译为 builtin 调用序列。到 M9 阶段，这个表达式已经"变身"为 bytecode——它的使命已经完成。

序列化 `PrimExpr` 需要一套完整的表达式 AST 序列化方案，复杂度不小，而且 runtime 加载时完全用不到（alloc_storage 的大小参数已经以寄存器值的形式出现在 bytecode 里了）。因此 storage_plan.json 只保存数值信息（`size_bytes`）。

### 12.5 STR 常量在 Python 和 C++ 的不对称处理

Python `VMInterpreter` 完整支持字符串常量：assert 消息是真实的字符串，失败时给出具体的错误信息（如 `"S exceeds upper bound 2048"`）。

C++ `VMState` 目前没有 `VMValue::String`。字符串在反序列化时被丢弃（存为 `VMValue{}`），assert_le_i64 在触发时给出通用消息 `"upper bound exceeded"`。

这是有意的**渐进实现**：Python 端用于开发调试时需要详细错误信息，C++ 端的生产部署中触发 shape assert 本身已经是严重错误，消息精确度是次要问题。如需改进，可以在 `VMValue` 中增加 `kString` tag，并在 `StringObj` 中存储内容。

---

## 13. 文件结构

### 13.1 Python 编译器侧

```
python/devproc2/
  vm/
    executable.py      # Executable / FunctionEntry / Instruction / CalleeKind / ConstInit
    serializer.py      # serialize(Executable) → bytes；deserialize(bytes) → Executable
    interpreter.py     # Python VMInterpreter（含所有 builtin 的 Python 实现，用于测试）
  compiler/
    passes/
      emit_executable.py  # EmitExecutablePass
      emit_abi.py         # EmitABIPass
      vm_codegen.py       # VMCodegenPass（生成 Executable 的上游）
    pass_context.py     # PassContext（key-value 存储，携带 storage_plan）
```

### 13.2 C++ Runtime 侧

```
runtime/
  include/devproc2/runtime/
    vm.h               # Executable（含 Load/Deserialize）、VMState、BuiltinRegistry
    packed_func.h      # PackedFuncRegistry（Load() 时验证依赖）
  src/
    vm.cc              # Executable::Deserialize、Executable::Load、VMState 执行循环
  tests/
    test_m9_artifact.cc  # C++ 单元测试：Load ABI 版本校验、packed_func 缺失校验
```

### 13.3 测试

```
tests/compiler/
  test_m9_artifact.py   # Python 单元测试（26 个）
    ├── EmitExecutablePass：写文件、magic 校验、round-trip
    ├── EmitABIPass：目录结构、abi.json 字段、shape_constraints 提取
    ├── manifest.json：字段完整性、build_time 格式
    ├── metadata/*.json：function_table、kernel_table、storage_plan
    └── 端到端：full_artifact_structure、cross_process_round_trip
```

---

## 14. 扩展路线

| 里程碑 | 对 Artifact 层的影响 |
|--------|----------------------|
| **M10 PackedFunc** | `required_packed_funcs` 开始非空；`packed_func_table.json` 有实际内容 |
| **M11 Triton cubin** | `kernels/<name>.cubin` 开始写入；`kernel_table.json` 有对应条目；C++ `Executable::Load` 需要加载 cubin |
| **M12 End-to-End** | 完整的 kvcache demo 产物，覆盖所有字段 |
| **未来：常量张量** | `constants/const_N.bin` 写入权重 blob；abi.json 增加 `constants` 字段描述每个 blob 的 dtype/shape |
| **未来：版本升级** | 若 bytecode 格式不兼容变更，递增 `devproc_abi_version` major；`Executable::Deserialize` 的 `_VERSION` 同步更新 |
