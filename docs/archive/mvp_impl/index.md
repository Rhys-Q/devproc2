# devproc2 MVP 实施文档索引

## 文档目录

| 文档 | 内容 |
|---|---|
| [00_infra.md](00_infra.md) | 项目基建：目录结构、Python uv 环境、CMakeLists.txt、工具链配置 |
| [01_milestones.md](01_milestones.md) | 15 个里程碑（M1-M12 + X1-X3）：任务清单 + 验收标准 |
| [02_compiler_pipeline.md](02_compiler_pipeline.md) | 18 步编译 Pass 流水线：每步输入/输出/关键不变量 |
| [03_runtime_design.md](03_runtime_design.md) | C++ Runtime 详细设计：Object 系统、VM、DeviceAPI、PackedFunc、ABI |

## 设计文档（上游参考）

| 文档 | 内容 |
|---|---|
| [docs/design/overview.md](../../design/overview.md) | 系统整体架构、模块分解、长期路线图 |
| [docs/design/mvp.md](../../design/mvp.md) | IR 设计、控制流、内存规划、VM、里程碑 1–12 + X1–X3 |
| [docs/design/control_flow.md](../../design/control_flow.md) | M3 控制流详细设计：Structured CF + Region + Yield + IterArgs + Effect-aware IR |
| [docs/design/从llama.cpp中学到了什么.md](../../design/从llama.cpp中学到了什么.md) | 从 llama.cpp 学到的关键设计经验和四阶段演进路线 |

## 推荐开发顺序

```
M1: C++ Object/ObjectRef/VMValue
   ↓
M2: High-level IR
M3: Control Flow IR          ← 可与 M2 并行
M4: Dynamic Shape            ← 依赖 M2
   ↓
M5: Effect System
   ↓
M6: DPS Lowering
M7: Memory Planning          ← 依赖 M6
   ↓
X1: Runtime Device API       ← 依赖 M1
M8: VM MVP                   ← 依赖 M7 + X1
   ↓
X2: Runtime Shape Builtins   ← 依赖 M8
X3: Kernel Launch Expr       ← 依赖 X2
   ↓
M9: ABI + Artifact
M10: PackedFunc              ← 可与 M9 并行
   ↓
M11: @devproc.kernel + Triton
   ↓
M12: End-to-End Demo
```

**关键约束**：不要在 VM（M8）和 ABI（M9）稳定前接入 Triton（M11），否则 IR/ABI/memory plan 一旦变动，Triton 接入必须返工。

## 术语表

| 术语 | 含义 |
|---|---|
| **DPS** | Destination-Passing Style：output buffer 由 caller 创建并传入，callee 只负责写入 |
| **SSA** | Static Single Assignment：每个变量只被赋值一次，贯穿整个 IR |
| **StructInfo** | 附在每个 IR value 上的 type + shape + device 元数据 |
| **UpperBound** | SymbolicDim 的编译时上界，用于静态分配 storage |
| **VMValue** | VM register 中的 tagged union：Null / Int / Float / Bool / ObjectRef |
| **CallDPS** | DPS 形式的 IR 节点：显式指定 inputs + output（可为 None）+ effect |
| **callee_kind** | CallDPS/Call 中区分被调用者类型：vm_func / builtin / packed_func / kernel |
| **EffectInfo** | 副作用描述：pure / read_only / write / opaque |
| **alloc_storage** | 中端 Memory Planning 之后才出现的低层 IR 节点，分配设备内存块 |
| **alloc_tensor** | 在 alloc_storage 上创建 tensor view，携带 shape/dtype/offset |

## MVP 成功标志

M12 端到端 demo 完整跑通，覆盖：

- 普通函数式 `Call`（`y = matmul(a, b)`）
- Tuple 多逻辑输出（`q, k, v = qkv_proj(x)`）
- `CallDPS`（lowering 后的 kernel call）
- `call_dps_packed`（tokenizer 等 runtime 函数）
- no-output stateful call（`update_kvcache(output=None, effect=write(...))`）
- 动态 shape + upper bound（`Tensor[(B, S, H)] where B<=8, S<=2048`）
- Memory planning + storage reuse
- VM 4 指令执行（call / ret / if / goto）
- 稳定 ABI artifact 产物
- C++ Object/ObjectRef 动态类型系统
- Triton cubin AOT 编译与 launch
