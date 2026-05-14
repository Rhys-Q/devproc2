# devproc2 VM 设计与实现

## 1. 为什么需要 VM？

### 1.1 问题背景

完成 Memory Planning 之后，IR 已经是"内存显式"的了：每个中间张量有明确的 `alloc_storage` 和 `alloc_tensor` 操作，storage 复用关系也已确定。但这个 IR 仍然是静态的数据结构，无法直接执行。

需要一个**执行引擎**把这段 IR 跑起来。

devproc2 选择 **字节码 VM（Virtual Machine）** 而不是直接代码生成（codegen to C++/LLVM）的原因：

- **可移植性**：VM bytecode 不依赖宿主机 ISA，可以在 CPU/GPU 混合环境运行
- **动态调度**：builtin 函数、kernel、packed_func 的调用通过函数表在 runtime 动态决定
- **调试友好**：bytecode 可以打印、检查、单步执行
- **极简 opcode**：4 条指令足以表达所有控制流，复杂功能通过函数调用扩展

### 1.2 4 指令哲学

devproc2 VM 只有 4 条指令：

```
CALL   调用函数（vm_func / builtin / packed_func / kernel）
RET    从当前函数返回
IF     条件分支（pc 相对跳转）
GOTO   无条件跳转（pc 相对跳转）
```

这不是偶然。指令集设计遵循一个原则：**复杂性留在函数调用层，不进入指令集**。

- 内存分配？→ `CALL @vm.builtin.alloc_storage`
- shape 计算？→ `CALL @vm.builtin.make_shape`
- 算术运算？→ `CALL @vm.builtin.add_i64`
- Kernel launch？→ `CALL @kernel.xxx`

这样 VM 本身的执行循环非常简单，容易验证正确性，扩展能力也不受 opcode 数量限制。

---

## 2. 编译 Pipeline 中的位置

```
Python DSL
   │
   ▼  @dp.function 装饰器捕获
High-level IR（CallOp）
   │
   ▼  InferStructInfoPass   — 推导 shape/dtype/device
   ▼  DPSLoweringPass       — CallOp → TensorCreateOp + CallDPSOp
   │
   ▼  MemoryPlanningPass    — 分析生命周期，生成 StoragePlan
   ▼  LowerTensorCreateToAllocPass  — TensorCreateOp → alloc_storage + alloc_tensor
   │
   ▼  Memory-explicit IR（AllocStorageOp / AllocTensorOp / CallDPSOp）
   │
   ▼  【VMCodegenPass】     ← 本文重点
   │
   ▼  Executable（bytecode + 函数表 + 常量池）
   │
   ▼  VMInterpreter（Python）/ VMState（C++）执行
```

---

## 3. 数据结构

### 3.1 Instruction

```python
@dataclass
class Instruction:
    opcode: Opcode       # CALL / RET / IF / GOTO

    # CALL 字段
    dst_reg:  int = -1   # 目标寄存器（-1 = 无返回值）
    func_idx: int = 0    # 函数表中的索引
    arg_regs: list[int]  # 参数寄存器列表

    # RET 字段
    src_reg: int = -1    # 返回值寄存器（-1 = void return）

    # IF 字段（pc 相对偏移，执行后不再 ++pc）
    cond_reg:     int    # 条件寄存器
    true_offset:  int    # cond 为 True 时：pc += true_offset
    false_offset: int    # cond 为 False 时：pc += false_offset

    # GOTO 字段（pc 相对偏移，执行后不再 ++pc）
    offset: int          # pc += offset（可以为负数，用于循环回跳）
```

**偏移量语义**：IF 和 GOTO 执行时 `pc += offset`，然后 `continue`（不再 `++pc`）。其他指令执行后 `++pc`。这意味着偏移量是相对于当前指令自身位置的。

### 3.2 FunctionEntry 与 ConstInit

```python
@dataclass
class ConstInit:
    reg_idx:   int   # 目标寄存器
    const_idx: int   # Executable.constants 中的索引

@dataclass
class FunctionEntry:
    name:         str
    kind:         CalleeKind  # vm_func / builtin / packed_func / kernel
    instr_offset: int         # 在全局指令数组中的起始偏移（外部函数为 -1）
    instr_count:  int
    num_regs:     int         # 此函数需要的寄存器总数
    num_args:     int         # 参数个数（占用 reg[0..num_args-1]）
    const_inits:  list[ConstInit]  # frame 建立时预填充的常量寄存器
```

`const_inits` 是解决"只有 4 条指令却没有 LOAD_CONST"这个问题的关键设计：

> 每次调用一个 `vm_func` 时，在建立新调用帧之后、执行第一条指令之前，将 `constants[const_idx]` 预写入 `regs[reg_base + reg_idx]`。

这样 `AllocStorageOp` 的 `size_bytes=IntImm(1024)` 就能出现在寄存器里，无需新增指令。

### 3.3 Executable

```python
@dataclass
class Executable:
    function_table: list[FunctionEntry]  # 所有函数（vm_func + 外部）
    instructions:   list[Instruction]   # 全局指令数组（所有 vm_func 共用）
    constants:      list[Any]           # 常量池（int/float/bool/None）
```

`function_table[0..n_vm]` 是真正的 vm_func，`function_table[n_vm..]` 是外部函数（builtin/packed_func/kernel），它们的 `instr_offset = -1`，由 VM 动态分派。

### 3.4 寄存器文件布局

VM 使用一个**扁平的全局寄存器文件**（`list[Any]` 或 `vector<VMValue>`），每个调用帧占用一段连续区间：

```
┌─────────────────────────────────────────────────────────────────┐
│ caller frame regs         │ callee frame regs                    │
│ [0 .. num_regs_caller-1]  │ [0 .. num_regs_callee-1]            │
└──────────────────┬────────┴──────────────────────────────────────┘
                   │                    │
                   reg_base(caller)=0   reg_base(callee)=num_regs_caller
```

`VMFrame` 记录 `reg_base`：访问寄存器 `r` 时实际访问 `regs[frame.reg_base + r]`。

---

## 4. VMCodegenPass：从 IR 到 Bytecode

本节是全文最核心的部分。我们先用一个**极小的例子**把所有概念讲清楚，再给出完整的规则表。

---

### 4.0 用最小的例子开始

假设有这样一个 memory-explicit IR：

```
@add_one(%x: Tensor[(4,), float32, cpu]) {
  %s0 = alloc_storage(size=16, alignment=64, device=cpu)
  %y   = alloc_tensor(%s0, offset=0, shape=[4], dtype=float32)
  call_dps kernel.add_one_fp32(inputs=[%x], output=%y)
  return %y
}
```

这个函数做的事情很简单：
1. 申请 16 字节的 CPU 内存（`alloc_storage`）
2. 在这块内存上创建一个 shape=(4,)、float32 的 tensor（`alloc_tensor`）
3. 调用 kernel 往 `%y` 里写入结果
4. 返回 `%y`

VMCodegenPass 要把这 4 行 IR 翻译成 VM 可以执行的字节码。整个翻译过程分三件事：

1. **寄存器（Register）是什么，怎么分配**
2. **常量（Constant）如何放进寄存器**
3. **每条 IR Op 翻译成哪几条 VM 指令**

---

### 4.1 寄存器是什么

VM 是一台"寄存器机"。你可以把寄存器理解成一排**带编号的格子**，每个格子能放一个值（整数、Tensor、Storage……任何东西）：

```
寄存器文件（针对函数 add_one 的一次调用）

  r0  │ <x tensor>   ← 函数参数，外部传入
  r1  │  ???         ← 待分配
  r2  │  ???
  r3  │  ???
  ...
```

**寄存器编号就是一个整数下标**，没有"类型"，装什么都行。

#### 寄存器的分配规则

VMCodegenPass 使用"顺序分配、不复用"策略：

- 参数从 `r0` 开始，依次占用 `r0, r1, ..., r(n_params-1)`
- 之后每次需要一个新寄存器，就取当前的 `next_reg`，然后 `next_reg += 1`
- **永远不回收，永远不复用**

这样每个 IR 中的 SSA 值（`%x`、`%s0`、`%y`……）都对应唯一一个寄存器编号，1:1 映射，永不冲突。

对应的 Python 代码（`_FnCtx` 类）：

```python
def alloc_reg(self) -> int:
    r = self.next_reg      # 取当前编号
    self.next_reg += 1     # 编号+1，下次再取就是下一个
    return r

def bind(self, value: Value, reg: int) -> None:
    self._value_reg[id(value)] = reg   # 记住"这个IR值对应哪个寄存器"

def reg_of(self, value: Value) -> int:
    return self._value_reg[id(value)]  # 查询"这个IR值在哪个寄存器里"
```

对我们的例子，参数 `%x` 只有一个，所以：

```
参数分配完毕：
  r0 → %x       (next_reg = 1)
```

---

### 4.2 常量放进寄存器：const_init 机制

现在遇到第一条 IR：

```
%s0 = alloc_storage(size=16, alignment=64, device=cpu)
```

`alloc_storage` 是一个函数调用，它需要 4 个参数：

| 参数 | 值 | 来源 |
|---|---|---|
| `size_bytes` | `16` | IntImm 常量 |
| `alignment` | `64` | IntImm 常量 |
| `device_type` | `1` (kDLCPU) | 从字符串 "cpu" 查表得到 |
| `device_id` | `0` | 默认值 |

问题来了：VM 的 CALL 指令只接受**寄存器编号**作为参数，但 `16`、`64`、`1`、`0` 这些值是字面量，不在任何寄存器里。怎么办？

#### 方案：const_init + 常量池

做法分两步：

**步骤 1：把常量值存进"常量池"（`Executable.constants`）**

`Executable.constants` 就是一个列表，按顺序存放所有常量值：

```python
exe.constants = []
```

每次遇到一个新常量，就 append 进去（相同值只存一次，去重）：

```python
def _intern_const(self, val) -> int:
    for i, c in enumerate(self._exec.constants):
        if c == val and type(c) is type(val):
            return i          # 已有，返回已有索引
    self._exec.constants.append(val)
    return len(self._exec.constants) - 1
```

处理完 `size=16, align=64, dev_type=1, dev_id=0` 之后，常量池变成：

```
exe.constants = [16, 64, 1, 0]
                  ↑   ↑  ↑  ↑
                 idx0 1  2  3
```

**步骤 2：为每个常量分配一个寄存器，记录"帧建立时把 constants[idx] 写入 reg[r]"**

```python
def _reg_for_const(self, val) -> int:
    const_idx = self._intern_const(val)   # 找到/创建常量池索引
    reg = self.alloc_reg()                # 分配一个新寄存器
    self.const_inits.append(
        ConstInit(reg_idx=reg, const_idx=const_idx)
    )
    return reg
```

这个 `ConstInit(reg_idx=r, const_idx=i)` 的意思是：
> 每次调用这个函数建立新帧时，在执行任何指令之前，先把 `constants[i]` 的值预填充到寄存器 `r` 里。

所以处理 `alloc_storage(size=16, ...)` 时，依次调用 `reg_for_int`：

```
ctx.reg_for_int(16) → r1,  const_inits += [ConstInit(reg=1, const_idx=0)]
ctx.reg_for_int(64) → r2,  const_inits += [ConstInit(reg=2, const_idx=1)]
ctx.reg_for_int(1)  → r3,  const_inits += [ConstInit(reg=3, const_idx=2)]
ctx.reg_for_int(0)  → r4,  const_inits += [ConstInit(reg=4, const_idx=3)]
```

寄存器布局更新：

```
r0  → %x          (函数参数，调用时传入)
r1  → 常量 16     (帧建立时由 const_init 预填)
r2  → 常量 64
r3  → 常量 1
r4  → 常量 0
r5  → 待分配（%s0 的结果寄存器）
```

然后分配 `%s0` 的结果寄存器，并生成指令：

```python
result_reg = ctx.alloc_reg()          # r5
ctx.bind(op.results[0], result_reg)   # %s0 → r5
ctx.emit(Instruction(
    opcode=Opcode.CALL,
    dst_reg=result_reg,               # 结果写进 r5
    func_idx=builtin("vm.builtin.alloc_storage"),
    arg_regs=[r1, r2, r3, r4],        # 参数：[size, align, dev_type, dev_id]
))
```

生成的指令：

```
PC=0: CALL r5, @vm.builtin.alloc_storage, [r1, r2, r3, r4]
```

---

### 4.3 逐 Op 翻译：完整 demo 走一遍

继续翻译剩余的 IR。

#### alloc_tensor

```
%y = alloc_tensor(%s0, offset=0, shape=[4], dtype=float32)
```

`alloc_tensor` 的签名是：
```
alloc_tensor(storage, offset, shape_tuple, dtype_code, dtype_bits, dtype_lanes)
```

需要依次处理：
- `storage` → 已有寄存器 `r5`（`%s0`）
- `offset=0` → 常量 0，查常量池：已有 idx=3，分配新寄存器 `r6`
  - `const_inits += [ConstInit(reg=6, const_idx=3)]`
- `shape=[4]` → 先要调用 `make_shape(4)` 生成 ShapeTuple：
  - 常量 `4` → 新常量 idx=4，分配 `r7`
  - `const_inits += [ConstInit(reg=7, const_idx=4)]`
  - 分配 shape 结果寄存器 `r8`，生成：`CALL r8, @vm.builtin.make_shape, [r7]`
- `dtype=float32` → DLPack 编码 `(code=2, bits=32, lanes=1)`：
  - 常量 2 → 新常量 idx=5，分配 `r9`
  - 常量 32 → 新常量 idx=6，分配 `r10`
  - 常量 1 → 已有 idx=2，分配 `r11`（注意：常量池去重，但每次都分配新寄存器）

最终分配 `%y` 的结果寄存器 `r12`，生成：

```
PC=1: CALL r8,  @vm.builtin.make_shape,   [r7]
PC=2: CALL r12, @vm.builtin.alloc_tensor, [r5, r6, r8, r9, r10, r11]
```

此时寄存器布局：

```
r0   → %x          (参数)
r1   → 16          (const_init: size)
r2   → 64          (const_init: alignment)
r3   → 1           (const_init: kDLCPU)
r4   → 0           (const_init: device_id)
r5   → %s0         (alloc_storage 结果)
r6   → 0           (const_init: offset=0)
r7   → 4           (const_init: shape dim)
r8   → ShapeTuple  (make_shape 结果)
r9   → 2           (const_init: dtype_code=kDLFloat)
r10  → 32          (const_init: dtype_bits)
r11  → 1           (const_init: dtype_lanes)
r12  → %y          (alloc_tensor 结果)
```

常量池：

```
exe.constants = [16, 64, 1, 0, 4, 2, 32]
                 0    1  2  3  4  5   6
```

#### call_dps

```
call_dps kernel.add_one_fp32(inputs=[%x], output=%y)
```

DPS 调用没有 SSA 结果（kernel 直接写进 output buffer），所以 `dst_reg = -1`：

```
PC=3: CALL -1, @kernel.add_one_fp32, [r0, r12]
```

#### return

```
return %y
```

```
PC=4: RET r12
```

---

### 4.4 最终 Executable

整合以上所有信息，生成的 `Executable` 如下：

```
Executable {

  constants = [16, 64, 1, 0, 4, 2, 32]
                0   1  2  3  4  5   6

  function_table = [
    # 先出现的 builtin 按调用顺序填入
    FunctionEntry("vm.builtin.alloc_storage", kind=builtin,  instr_offset=-1, ...)
    FunctionEntry("vm.builtin.make_shape",    kind=builtin,  instr_offset=-1, ...)
    FunctionEntry("vm.builtin.alloc_tensor",  kind=builtin,  instr_offset=-1, ...)
    FunctionEntry("kernel.add_one_fp32",      kind=kernel,   instr_offset=-1, ...)
    FunctionEntry(
      name         = "add_one",
      kind         = vm_func,
      instr_offset = 0,         # 从全局指令数组第 0 条开始
      instr_count  = 5,
      num_regs     = 13,        # 需要 r0~r12，共 13 个寄存器
      num_args     = 1,         # r0 接收 %x
      const_inits  = [
        ConstInit(reg=1,  const_idx=0),   # r1  ← constants[0] = 16
        ConstInit(reg=2,  const_idx=1),   # r2  ← constants[1] = 64
        ConstInit(reg=3,  const_idx=2),   # r3  ← constants[2] = 1
        ConstInit(reg=4,  const_idx=3),   # r4  ← constants[3] = 0
        ConstInit(reg=6,  const_idx=3),   # r6  ← constants[3] = 0  (offset=0 复用)
        ConstInit(reg=7,  const_idx=4),   # r7  ← constants[4] = 4
        ConstInit(reg=9,  const_idx=5),   # r9  ← constants[5] = 2
        ConstInit(reg=10, const_idx=6),   # r10 ← constants[6] = 32
        ConstInit(reg=11, const_idx=2),   # r11 ← constants[2] = 1  (lanes 复用)
      ]
    )
  ]

  instructions = [
    PC=0: CALL r5,  @[0]vm.builtin.alloc_storage, arg=[r1, r2, r3, r4]
    PC=1: CALL r8,  @[1]vm.builtin.make_shape,    arg=[r7]
    PC=2: CALL r12, @[2]vm.builtin.alloc_tensor,  arg=[r5, r6, r8, r9, r10, r11]
    PC=3: CALL -1,  @[3]kernel.add_one_fp32,      arg=[r0, r12]
    PC=4: RET r12
  ]
}
```

**关键点总结：**

| 概念 | 含义 |
|---|---|
| `num_regs = 13` | 这次函数调用需要开辟 13 个格子（r0~r12） |
| `num_args = 1` | 调用方把参数写进 r0，其余寄存器由 VM 自行初始化 |
| `const_inits` | 帧建立后、第一条指令执行前，把常量批量写入对应寄存器 |
| `instr_offset = 0` | 这个函数的指令从全局指令数组下标 0 开始 |
| `dst_reg = -1` | CALL 没有返回值（DPS kernel / void builtin） |

---

### 4.5 帧建立过程（const_init 何时执行）

每次调用一个 `vm_func` 时，VM 按以下顺序初始化新帧：

```
1. 在全局寄存器文件末尾扩展 num_regs 个格子（全部初始化为 None）
2. 把调用方传来的 args 依次写入 r0, r1, ..., r(num_args-1)
3. 遍历 const_inits，把 constants[const_idx] 写入 r[reg_idx]
4. 开始执行第一条指令（PC = instr_offset）
```

对 `add_one` 的一次调用，帧建立后寄存器文件状态（假设 `reg_base = 0`）：

```
r0  = <x tensor>   ← 调用方传入
r1  = 16           ← const_init
r2  = 64           ← const_init
r3  = 1            ← const_init
r4  = 0            ← const_init
r5  = None         ← 等 PC=0 执行后写入
r6  = 0            ← const_init
r7  = 4            ← const_init
r8  = None         ← 等 PC=1 执行后写入
r9  = 2            ← const_init
r10 = 32           ← const_init
r11 = 1            ← const_init
r12 = None         ← 等 PC=2 执行后写入
```

---

### 4.6 各 Op 的翻译规则（完整表）

| IR Op | 生成的 VM 指令 | 说明 |
|---|---|---|
| `AllocStorageOp` | `CALL r_dst, @vm.builtin.alloc_storage, [r_size, r_align, r_devtype, r_devid]` | 4 个参数均来自 const_init |
| `AllocTensorOp` | `CALL r_shape, @vm.builtin.make_shape, [r_d0, ...]`<br>`CALL r_dst, @vm.builtin.alloc_tensor, [r_storage, r_offset, r_shape, r_code, r_bits, r_lanes]` | 两条指令，先建 ShapeTuple 再建 Tensor |
| `CallDPSOp` | `CALL -1, @callee, [r_in0, ..., r_out]` | `dst_reg=-1`：kernel 写 output buffer，无 SSA 结果 |
| `TupleOp` | `CALL r_dst, @vm.builtin.make_tuple, [r_e0, r_e1, ...]` | |
| `TupleGetItemOp` | `CALL r_dst, @vm.builtin.tuple_get_item, [r_tuple, r_idx]` | `r_idx` 来自 const_init |
| `ShapeAssertOp` | `CALL -1, @vm.builtin.shape_assert, [r_tensor, r_dim, r_upper]` | 运行时 shape 检查，违反时 throw |
| `ReturnOp` | `RET r_val`（void 返回用 `RET -1`） | |
| `IfOp` | `IF + GOTO + identity copies`（见 §4.7） | |
| `ForOp` | `identity init + lt_i64 + IF + body + add_i64 + GOTO`（见 §4.8） | |
| `YieldOp` | 不生成指令 | 由父 IfOp/ForOp 读取 `yield.values` 后生成 identity CALL |

---

### 4.7 IfOp 的 backpatching（跳转目标回填）

IfOp 是控制流，翻译时需要知道"跳到哪里"，但在生成 IF 指令时，then-branch 和 else-branch 的代码还没生成，长度未知。解决方法是**先占位，后回填**（backpatching）。

以下面这个 IR 为例：

```
%z = if %cond {
    yield %a
} else {
    yield %b
}
```

**生成过程分 5 步：**

**Step 1：预分配结果寄存器**

IfOp 有 SSA result `%z`，在生成任何指令之前先分配：

```
ctx.alloc_reg() → r_z
ctx.bind(%z, r_z)
```

这样 then-branch 和 else-branch 都知道应该把 yield 的值写进 `r_z`。

**Step 2：emit IF 占位指令**

```python
if_pc = ctx.emit_placeholder(Opcode.IF)  # if_pc = 当前指令数组长度
ctx.instrs[if_pc].cond_reg = r_cond
ctx.instrs[if_pc].true_offset = 1        # true 时 PC += 1（= 跳到 IF 的下一条）
# false_offset 暂时不知道，留空
```

此时指令数组：

```
[if_pc]   IF  cond=r_cond, true=+1, false=???
```

**Step 3：生成 then-branch**

遍历 then-block 的所有 op。遇到 `YieldOp` 时，把 yield 值用 identity 复制到 `r_z`：

```
CALL r_z, @vm.builtin.identity, [r_a]
```

then-branch 结束后，emit GOTO 占位（用于跳过 else-branch）：

```python
goto_pc = ctx.emit_placeholder(Opcode.GOTO)
```

此时：

```
[if_pc]      IF   cond=r_cond, true=+1, false=???
[if_pc+1]    CALL r_z, @identity, [r_a]     ← then-branch
[goto_pc]    GOTO offset=???                ← 待填
```

**Step 4：回填 IF.false_offset，生成 else-branch**

```python
else_pc = ctx.pc()    # 当前长度 = else-branch 的起始 PC
ctx.instrs[if_pc].false_offset = else_pc - if_pc
```

然后生成 else-branch，同样在 `YieldOp` 处写 `r_z`：

```
CALL r_z, @vm.builtin.identity, [r_b]
```

**Step 5：回填 GOTO.offset**

```python
after_pc = ctx.pc()   # else-branch 结束后的 PC
ctx.instrs[goto_pc].offset = after_pc - goto_pc
```

最终指令序列：

```
[0]  IF    cond=r_cond, true_offset=+1, false_offset=+3
[1]  CALL  r_z, @identity, [r_a]     ← then: yield %a
[2]  GOTO  offset=+2                 ← 跳过 else
[3]  CALL  r_z, @identity, [r_b]     ← else: yield %b
[4]  ...（后续指令）
```

执行语义：
- `cond=True`：PC += 1 → 执行 [1]（identity copy a→z），然后 [2]（GOTO +2 → PC=4），跳过 else
- `cond=False`：PC += 3 → 执行 [3]（identity copy b→z），然后继续 [4]

> **为什么用相对偏移而不是绝对地址？**
> 相对偏移让 bytecode 位置无关。当多个函数共享同一个全局指令数组时，拼接后不需要重新计算地址。

---

### 4.8 ForOp 的 loop codegen

以下面的 IR 为例（把 0~N-1 累加）：

```
%result = for %i in range(0, %N, 1), iter_args=[acc=0] {
    %new_acc = add(%acc, 1)
    yield %new_acc
}
```

**生成过程：**

```
Step 1: 初始化循环变量和 iter_arg

CALL r_i,   @identity, [r_start]   # r_i   = 0   （循环变量初值）
CALL r_acc, @identity, [r_zero]    # r_acc = 0   （iter_arg 初值）

Step 2: 循环头（loop header）——条件检查

[loop_header_pc]
CALL r_cond, @lt_i64, [r_i, r_N]   # r_cond = (i < N)
IF   r_cond, true_offset=+1, false_offset=???   ← 待填

Step 3: 循环体

... 循环体指令 ...
CALL r_acc, @identity, [r_new_acc]  # 更新 iter_arg（来自 YieldOp）
CALL r_i,   @add_i64, [r_i, r_step] # i = i + 1（step=1）

Step 4: 跳回循环头

[goto_pc]
GOTO (loop_header_pc - goto_pc)     # 负偏移，回跳

Step 5: 循环结束

[after_loop_pc]
# 回填 IF.false_offset = after_loop_pc - if_pc
CALL r_result, @identity, [r_acc]   # 导出 ForOp SSA result
```

最终指令结构（以 `loop_header_pc=2` 为例）：

```
PC=0: CALL r_i,    @identity, [r_start]
PC=1: CALL r_acc,  @identity, [r_zero]
PC=2: CALL r_cond, @lt_i64,   [r_i, r_N]     ← loop_header_pc
PC=3: IF   r_cond, true=+1, false=+5          ← false 时跳到 PC=8
PC=4: ... 循环体指令（计算 %new_acc）...
PC=5: CALL r_acc, @identity, [r_new_acc]      ← 更新 iter_arg
PC=6: CALL r_i,   @add_i64,  [r_i, r_step]   ← i++
PC=7: GOTO -5                                  ← 跳回 PC=7-5=2
PC=8: CALL r_result, @identity, [r_acc]        ← ForOp SSA result
```

每次循环迭代的 PC 轨迹：`2 → 3(true) → 4 → 5 → 6 → 7 → 2 → 3 → ...`，直到条件为 false 时跳到 `PC=8`。

---

## 5. 执行引擎：Python Interpreter

`VMInterpreter`（Python 实现，用于测试和快速验证）维护：

```python
regs:          list[Any]              # 扁平寄存器文件，动态增长
frames:        list[(func_idx, pc, reg_base)]  # 调用栈
return_slots:  list[(dst_reg, caller_reg_base)] # 与 frames 平行，记录返回值写目标
```

主循环：

```python
while frames:
    fidx, pc, base = frames[-1]
    fe = exec_.function_table[fidx]
    instr = exec_.instructions[fe.instr_offset + pc]

    if instr.opcode == CALL:
        callee = exec_.function_table[instr.func_idx]
        args = [regs[base + r] for r in instr.arg_regs]
        if callee.kind == vm_func:
            frames[-1] = (fidx, pc + 1, base)  # 保存 caller pc（已前进）
            return_slots.append((instr.dst_reg, base))
            push_frame(instr.func_idx, args)   # 不再 ++pc
            continue
        else:
            result = dispatch_external(callee, args)
            if instr.dst_reg >= 0:
                regs[base + instr.dst_reg] = result

    elif instr.opcode == RET:
        result = regs[base + instr.src_reg] if instr.src_reg >= 0 else None
        del regs[base:]          # 释放当前帧的寄存器
        frames.pop()
        if frames:
            dst_reg, caller_base = return_slots.pop()
            if dst_reg >= 0:
                regs[caller_base + dst_reg] = result
            continue             # 不再 ++pc（caller pc 已在 CALL 时前进）
        else:
            return result        # 顶层返回

    elif instr.opcode == IF:
        cond = bool(regs[base + instr.cond_reg])
        frames[-1] = (fidx, pc + (instr.true_offset if cond else instr.false_offset), base)
        continue                 # 不再 ++pc

    elif instr.opcode == GOTO:
        frames[-1] = (fidx, pc + instr.offset, base)
        continue                 # 不再 ++pc

    frames[-1] = (fidx, pc + 1, base)  # 普通指令 ++pc
```

### 5.1 Builtin 函数

Python interpreter 内置以下 builtin：

| 函数名 | 参数 | 返回 |
|---|---|---|
| `vm.builtin.alloc_storage` | `(size, align, dev_type, dev_id)` | `_Storage` 对象 |
| `vm.builtin.alloc_tensor` | `(storage, offset, shape, code, bits, lanes)` | `_Tensor` 对象 |
| `vm.builtin.make_shape` | `(*dims)` | `tuple[int, ...]` |
| `vm.builtin.make_tuple` | `(*fields)` | `tuple` |
| `vm.builtin.tuple_get_item` | `(tup, idx)` | `tup[idx]` |
| `vm.builtin.identity` | `(x,)` | `x`（寄存器间"移动"） |
| `vm.builtin.lt_i64` | `(a, b)` | `bool(a < b)` |
| `vm.builtin.add_i64` | `(a, b)` | `a + b` |
| `vm.builtin.shape_assert` | `(tensor, dim, upper)` | 超界时 raise |

---

## 6. C++ 执行引擎

`VMState`（C++ 实现，用于生产运行时）与 Python interpreter 镜像对应。

```cpp
class VMState {
    std::shared_ptr<Executable> exec_;
    std::vector<VMFrame>        frames_;
    std::vector<VMValue>        regs_;

    VMValue Invoke(const std::string& func_name, std::vector<VMValue> args);
    VMValue ExecuteLoop();
    VMValue DispatchExternal(const FunctionEntry&, std::vector<VMValue>&);
};
```

`VMFrame` 比 Python interpreter 多两个字段，省去了平行的 `return_slots` 栈：

```cpp
struct VMFrame {
    int32_t func_idx;
    int32_t pc;
    int32_t reg_base;
    int32_t caller_dst_reg;   // 返回值写哪里（-1 = 顶层调用）
    int32_t caller_reg_base;  // caller 的 reg_base，写返回值时用
};
```

C++ 侧的 `BuiltinRegistry`（全局单例）通过 `RegisterVMBuiltins()` 一次性注册所有 `vm.builtin.*`，`PackedFuncRegistry` 管理用户注册的 packed_func。

---

## 7. 端到端 Demo：从 DSL 到 VM 指令

下面用一个最简单的例子，完整展示每一个变换步骤。

### 7.1 模型：两层 relu + 条件分支

```python
import devproc2.frontend.dsl as dp

@dp.function
def demo(x: dp.Tensor[(512,), "float16", "cpu"],
         flag: dp.Tensor[(1,),  "bool",    "cpu"]):
    y = dp.ops.relu(x)        # 第一层 relu
    if flag:
        z = dp.ops.relu(y)    # flag=True：再过一层
    else:
        z = y                 # flag=False：原样返回
    return z
```

这个模型涉及：
- DPS 算子调用（relu）
- 内存规划（中间 tensor）
- 控制流（if/else 分支）
- SSA result（z 由 if 产生）

---

### 7.2 Step 1：DSL 捕获 → High-level IR

`@dp.function` 解析函数 AST，生成高层 IR：

```
@demo(%x: Tensor[(512), float16, cpu],
      %flag: Tensor[(1), bool, cpu]) {
  %y    = @relu(%x)
  %z    = if %flag {
              %v0 = @relu(%y)
              yield %v0
          } else {
              yield %y
          }
  return %z
}
```

**特点**：
- `CallOp`（`%y = @relu(%x)`）是高层函数式调用，没有显式 output buffer
- `IfOp` 是结构化控制流，两个 branch 各有 `YieldOp`
- `%z` 是 IfOp 的 SSA result

---

### 7.3 Step 2：InferStructInfoPass → 类型传播

推导每个 SSA value 的 `TensorStructInfo`（shape + dtype + device）：

```
%x    : Tensor[(512), float16, cpu]   ← 参数注解
%flag : Tensor[(1), bool, cpu]        ← 参数注解
%y    : Tensor[(512), float16, cpu]   ← relu 输出与输入同 shape/dtype
%v0   : Tensor[(512), float16, cpu]   ← relu(y) 同理
%z    : Tensor[(512), float16, cpu]   ← IfOp result 从 branch yield 推断
```

---

### 7.4 Step 3：DPSLoweringPass → 插入 output buffer

`CallOp @relu(%x)` 变成：分配一块 output buffer + DPS 调用：

```
@demo(%x: Tensor[(512), float16, cpu],
      %flag: Tensor[(1), bool, cpu]) {
  %y_buf = dp.empty(shape=[512], dtype=float16, device=cpu)
  call_dps @kernel.relu_fp16(inputs=[%x], output=%y_buf)
  %z    = if %flag {
              %v0_buf = dp.empty(shape=[512], dtype=float16, device=cpu)
              call_dps @kernel.relu_fp16(inputs=[%y_buf], output=%v0_buf)
              yield %v0_buf
          } else {
              yield %y_buf
          }
  return %z
}
```

**注意**：IfOp 内部的 `dp.empty` 也被插入 then-branch。`%z` 是 IfOp 的 SSA result，指向分支的 yield value。

---

### 7.5 Step 4：MemoryPlanningPass + LowerTensorCreateToAllocPass → 内存显式 IR

MemoryPlanningPass 分析 live interval：

| Tensor | first_def | last_use | reusable? |
|---|---|---|---|
| `y_buf` | 0 | 3（if 中作为 relu 输入）| yes |
| `v0_buf`| 2 | 4（yield 后 = IfOp result）| no（是 result）|

`y_buf` 和 `v0_buf` live interval 不重叠，分配到同一个 storage `s0`。`v0_buf` 是返回路径上的 tensor，必须用独立 storage `s1`。

LowerTensorCreateToAllocPass 替换后得到**内存显式 IR**：

```
@demo(%x: Tensor[(512), float16, cpu],
      %flag: Tensor[(1), bool, cpu]) {
  %s0 = alloc_storage(size=1024, alignment=256, device=cpu)   ← y_buf 的 storage
  %s1 = alloc_storage(size=1024, alignment=256, device=cpu)   ← v0_buf 的 storage
  %y_buf  = alloc_tensor(%s0, offset=0, shape=[512], dtype=float16)
  call_dps @kernel.relu_fp16(inputs=[%x], output=%y_buf)
  %z    = if %flag {
              %v0_buf = alloc_tensor(%s1, offset=0, shape=[512], dtype=float16)
              call_dps @kernel.relu_fp16(inputs=[%y_buf], output=%v0_buf)
              yield %v0_buf
          } else {
              yield %y_buf
          }
  return %z
}
```

此时 IR 中已没有隐式内存分配，一切 tensor 都有明确 storage 来源。

---

### 7.6 Step 5：VMCodegenPass → Executable

VMCodegenPass 遍历内存显式 IR，生成 Executable。

#### 寄存器分配结果

| 值 | 寄存器 | 来源 |
|---|---|---|
| `%x` | r0 | 参数 |
| `%flag` | r1 | 参数 |
| `%s0_size` | r2 | const_init: 1024 |
| `%s0_align` | r3 | const_init: 256 |
| `%s0_devtype` | r4 | const_init: 1 (kDLCPU) |
| `%s0_devid` | r5 | const_init: 0 |
| `%s0` | r6 | alloc_storage 结果 |
| `%s1_size` | r7 | const_init: 1024 |
| `%s1_align` | r8 | const_init: 256 |
| `%s1_devtype` | r9 | const_init: 1 |
| `%s1_devid` | r10 | const_init: 0 |
| `%s1` | r11 | alloc_storage 结果 |
| `%shape_dims` | r12 | const_init: 512 |
| `%shape_y` | r13 | make_shape 结果 |
| `%dtype_code` | r14 | const_init: 2 (kDLFloat) |
| `%dtype_bits` | r15 | const_init: 16 |
| `%dtype_lanes`| r16 | const_init: 1 |
| `%offset_0` | r17 | const_init: 0 |
| `%y_buf` | r18 | alloc_tensor 结果 |
| `%z` | r19 | IfOp SSA result（预分配） |
| `%cond_val` | r20 | 用于 IF 指令（从 flag tensor 中提取） |
| `%shape_v0` | r21 | make_shape 结果（then-branch）|
| `%v0_buf` | r22 | alloc_tensor 结果（then-branch）|

> 实际实现中常量寄存器由 `const_inits` 机制在 frame 建立时预填充，不占用显式指令。

#### 生成的指令序列

```
函数: demo  (func_idx=0, num_args=2, instr_offset=0)
   const_inits: [r2←1024, r3←256, r4←1, r5←0,
                  r7←1024, r8←256, r9←1, r10←0,
                  r12←512, r14←2, r15←16, r16←1, r17←0]

PC=0:  CALL r6,  @vm.builtin.alloc_storage, [r2, r3, r4, r5]
       # r6 = alloc_storage(1024, 256, 1, 0)  → Storage(s0, cpu)

PC=1:  CALL r11, @vm.builtin.alloc_storage, [r7, r8, r9, r10]
       # r11 = alloc_storage(1024, 256, 1, 0) → Storage(s1, cpu)

PC=2:  CALL r13, @vm.builtin.make_shape,    [r12]
       # r13 = make_shape(512) → ShapeTuple([512])

PC=3:  CALL r18, @vm.builtin.alloc_tensor,  [r6, r17, r13, r14, r15, r16]
       # r18 = alloc_tensor(s0, 0, [512], float16) → Tensor(y_buf)

PC=4:  CALL -1, @kernel.relu_fp16, [r0, r18]
       # call_dps relu(x → y_buf)，无 SSA result

PC=5:  IF   r1, true_offset=1, false_offset=7
       # flag(r1): True → PC=6(then), False → PC=12(else)

       ── then-branch (PC=6..10) ──

PC=6:  CALL r21, @vm.builtin.make_shape,    [r12]
       # r21 = make_shape(512)

PC=7:  CALL r22, @vm.builtin.alloc_tensor,  [r11, r17, r21, r14, r15, r16]
       # r22 = alloc_tensor(s1, 0, [512], float16) → Tensor(v0_buf)

PC=8:  CALL -1, @kernel.relu_fp16, [r18, r22]
       # call_dps relu(y_buf → v0_buf)

PC=9:  CALL r19, @vm.builtin.identity, [r22]
       # z = v0_buf  （YieldOp → 写入 IfOp result 寄存器）

PC=10: GOTO +2
       # 跳过 else-branch → PC=12

       ── else-branch (PC=11) ──

PC=11: CALL r19, @vm.builtin.identity, [r18]
       # z = y_buf  （YieldOp → 写入 IfOp result 寄存器）

       ── after if (PC=12) ──

PC=12: RET r19
       # return z
```

#### 函数表

```
func_idx=0  "demo"           kind=vm_func    instr_offset=0,  instr_count=13
func_idx=1  "alloc_storage"  kind=builtin    instr_offset=-1
func_idx=2  "alloc_tensor"   kind=builtin    instr_offset=-1
func_idx=3  "make_shape"     kind=builtin    instr_offset=-1
func_idx=4  "kernel.relu_fp16" kind=kernel   instr_offset=-1
func_idx=5  "identity"       kind=builtin    instr_offset=-1
```

#### 常量池

```
constants = [1024, 256, 1, 0, 512, 2, 16]  （去重后）
```

---

### 7.7 Step 6：执行跟踪

假设输入 `flag=True`，逐指令跟踪 VM 执行过程：

```
== invoke("demo", [x_tensor, True]) ==

[frame 建立] reg_base=0, num_regs=23
  const_inits 预填充: r2=1024, r3=256, r4=1, r5=0, r7=1024, ...
  regs = [x_tensor, True, 1024, 256, 1, 0, None, 1024, ...]
           r0       r1   r2   r3   r4  r5  r6    r7

PC=0: CALL alloc_storage([r2,r3,r4,r5]) → r6 = Storage(size=1024, cpu)
PC=1: CALL alloc_storage([r7,r8,r9,r10]) → r11 = Storage(size=1024, cpu)
PC=2: CALL make_shape([r12=512]) → r13 = (512,)
PC=3: CALL alloc_tensor([r6,r17=0,r13,r14=2,r15=16,r16=1]) → r18 = Tensor(shape=(512,), float16)
PC=4: CALL kernel.relu_fp16([r0=x, r18=y_buf]) → 无返回值  [kernel mock: no-op]

PC=5: IF r1=True, → true_offset=1 → PC=5+1=6

PC=6: CALL make_shape([512]) → r21 = (512,)
PC=7: CALL alloc_tensor([r11, 0, r21, 2, 16, 1]) → r22 = Tensor(shape=(512,), float16)
PC=8: CALL kernel.relu_fp16([r18=y_buf, r22=v0_buf]) → no-op
PC=9: CALL identity([r22]) → r19 = v0_buf   (z = v0_buf)
PC=10: GOTO +2 → PC=10+2=12

PC=12: RET r19 → return r19 = v0_buf
```

**结果**：返回 `v0_buf`（一个 shape=(512,), dtype=float16 的 Tensor）。

假设输入 `flag=False`：

```
PC=5: IF r1=False, → false_offset=7 → PC=5+7=12
  ✗ 跳过了 then-branch 的 PC=6~10

PC=11: CALL identity([r18]) → r19 = y_buf   (z = y_buf)

PC=12: RET r19 → return y_buf
```

---

## 8. 编写与运行

### 8.1 完整可运行代码

以下代码可以直接复制运行（不需要 GPU，使用 CPU mock kernel）：

```python
import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.kernel.registry import KernelRegistry, KernelSpec
from devproc2.vm import VMInterpreter
from devproc2.ir import print_module

# ── Step 1: 定义 DSL 函数 ──────────────────────────────────────────────

dp.reset_module()

@dp.function
def demo(x: dp.Tensor[(512,), "float16", "cpu"]):
    y = dp.ops.relu(x)
    z = dp.ops.relu(y)
    return z

# ── Step 2: 配置 Kernel Registry ──────────────────────────────────────

reg = KernelRegistry()
reg.register(KernelSpec(
    op_name="relu",
    device="cpu",
    input_dtypes=("float16",),
    kernel_name="kernel.relu_fp16",
))

# ── Step 3: 运行编译 pipeline ─────────────────────────────────────────

module = dp.get_module()

print("=== High-level IR ===")
print(print_module(module))

module = InferStructInfoPass().run(module)
module = DPSLoweringPass(reg).run(module)

print("=== After DPS Lowering ===")
print(print_module(module))

ctx = PassContext()
MemoryPlanningPass().run(module, ctx)
module = LowerTensorCreateToAllocPass(ctx).run(module)

print("=== Memory-explicit IR ===")
print(print_module(module))

exe = VMCodegenPass().run(module)

# ── Step 4: 打印 Executable ───────────────────────────────────────────

print("=== Executable ===")
print(f"Constants: {exe.constants}")
print(f"Function table:")
for i, fe in enumerate(exe.function_table):
    print(f"  [{i}] {fe.name!r}  kind={fe.kind.name}  "
          f"instr_offset={fe.instr_offset}  num_regs={fe.num_regs}  "
          f"num_args={fe.num_args}  const_inits=[{len(fe.const_inits)} entries]")

print(f"\nInstructions ({len(exe.instructions)} total):")
fn_idx = exe.get_func_index("demo")
fe = exe.function_table[fn_idx]
for i, instr in enumerate(
    exe.instructions[fe.instr_offset : fe.instr_offset + fe.instr_count]
):
    callee_name = ""
    if instr.opcode.name == "CALL":
        callee_name = f" @{exe.function_table[instr.func_idx].name}"
    print(f"  PC={i:2d}  {instr.opcode.name:<6}"
          f"  dst={instr.dst_reg:3d}  src={instr.src_reg:3d}"
          f"  cond={instr.cond_reg:2d}  to={instr.true_offset:3d}  fo={instr.false_offset:3d}"
          f"  offset={instr.offset:4d}  args={instr.arg_regs}"
          f"{callee_name}")

# ── Step 5: 执行 ──────────────────────────────────────────────────────

vm = VMInterpreter(exe)
result = vm.invoke("demo", [None])  # x=None，kernel 是 no-op mock

from devproc2.vm.interpreter import _Tensor
assert isinstance(result, _Tensor)
print(f"\n=== Execution Result ===")
print(f"shape: {result.shape}, dtype: ({result.dtype_code}, {result.dtype_bits}, {result.dtype_lanes})")
```

### 8.2 预期输出（节选）

```
=== High-level IR ===
@demo(%x: Tensor[(512), float16, cpu]) {
  %y = @relu(%x)
  %z = @relu(%y)
  return %z
}

=== After DPS Lowering ===
@demo(%x: Tensor[(512), float16, cpu]) {
  %y = dp.empty(shape=(512), dtype=float16, device=cpu)
  call_dps kernel.relu_fp16(inputs=[%x], output=%y, ...)
  %z = dp.empty(shape=(512), dtype=float16, device=cpu)
  call_dps kernel.relu_fp16(inputs=[%y], output=%z, ...)
  return %z
}

=== Memory-explicit IR ===
@demo(%x: Tensor[(512), float16, cpu]) {
  %s0 = alloc_storage(size=1024, alignment=256, device=cpu)
  %s1 = alloc_storage(size=1024, alignment=256, device=cpu)
  %y  = alloc_tensor(%s0, offset=0, shape=(512), dtype=float16)
  call_dps kernel.relu_fp16(inputs=[%x], output=%y, ...)
  %z  = alloc_tensor(%s1, offset=0, shape=(512), dtype=float16)
  call_dps kernel.relu_fp16(inputs=[%y], output=%z, ...)
  return %z
}
# 注意：此例中 z 是返回值，内存规划器标记其为不可复用，因此 y 和 z 各有独立 storage。
# 若链路更长（≥4 层），非返回的中间 tensor 会自动复用 storage。

=== Executable ===
Constants: [1024, 256, 1, 0, 512, 2, 16]

Function table:
  [0] 'vm.builtin.alloc_storage' kind=builtin  instr_offset=-1
  [1] 'vm.builtin.make_shape'    kind=builtin  instr_offset=-1
  [2] 'vm.builtin.alloc_tensor'  kind=builtin  instr_offset=-1
  [3] 'kernel.relu_fp16'         kind=kernel   instr_offset=-1
  [4] 'demo'            kind=vm_func    instr_offset=0  num_regs=25  num_args=1

Instructions (9 total):
  PC= 0  CALL   dst= 5  args=[1,2,3,4]         @vm.builtin.alloc_storage   # s0
  PC= 1  CALL   dst=10  args=[6,7,8,9]          @vm.builtin.alloc_storage   # s1
  PC= 2  CALL   dst=13  args=[12]               @vm.builtin.make_shape      # shape y
  PC= 3  CALL   dst=17  args=[5,11,13,14,15,16] @vm.builtin.alloc_tensor    # y
  PC= 4  CALL   dst=-1  args=[0,17]             @kernel.relu_fp16           # relu(x→y)
  PC= 5  CALL   dst=20  args=[19]               @vm.builtin.make_shape      # shape z
  PC= 6  CALL   dst=24  args=[10,18,20,21,22,23]@vm.builtin.alloc_tensor    # z
  PC= 7  CALL   dst=-1  args=[17,24]            @kernel.relu_fp16           # relu(y→z)
  PC= 8  RET    src=24

=== Execution Result ===
shape: (512,), dtype: (2, 16, 1)
```

---

## 9. 关键设计决策与权衡

### 9.1 4 条指令而不是更多

**为什么不加 LOAD_CONST 指令？**

加了 LOAD_CONST 反而要为每个常量多一条指令。`const_inits` 机制是一次性批量预填充，对于参数众多的 alloc_storage（需要 4 个常量参数）更加高效。

**为什么不加 MOVE 指令？**

`vm.builtin.identity` 已经满足需求，且语义更清晰：identity 是一个 builtin 函数调用，与其他 CALL 完全对称，执行引擎不需要特殊处理。

### 9.2 扁平寄存器文件 vs 栈式寄存器

devproc2 使用**扁平寄存器文件**（类似 TVM RelaxVM），而不是 JVM/Python 的操作数栈：

- 每个 SSA value 对应唯一寄存器编号，不需要 dup/pop 等栈操作
- 函数调用时直接传寄存器编号列表，不需要 push/pop 序列
- 寄存器文件随调用栈增长，函数返回时收缩

### 9.3 IF/GOTO 使用 pc 相对偏移而不是绝对地址

这是一个权衡：绝对地址更容易计算，但相对偏移让 bytecode 更加位置无关，适合后续的 serialization 和 partial linking。

### 9.4 YieldOp 不生成指令

IfOp/ForOp 的 YieldOp 不生成 YIELD 指令，而是由 codegen 读取 `YieldOp.values` 并生成 `identity` 调用写入 result 寄存器。这避免了需要一条专用 YIELD 指令，同时确保两个 branch 收敛到同一个寄存器（IfOp 的 SSA result）。

### 9.5 测试文件禁止 `from __future__ import annotations`

Python 的 `from __future__ import annotations` 会将所有类型注解字符串化（PEP 563）。这会导致 `fn.__annotations__` 返回 `"dp.Tensor[(512,), 'float16', 'cpu']"` 字符串而非 `TensorStructInfo` 对象，从而使 `DPSLoweringPass._lookup` 的 `isinstance(si, TensorStructInfo)` 检查失败，CallOp 无法被转换。在测试文件开头**绝对不要加这行导入**。

---

## 10. 文件结构

```
python/devproc2/
├── vm/
│   ├── __init__.py          # 公开 Opcode, Instruction, FunctionEntry, Executable, VMInterpreter
│   ├── executable.py        # 数据结构（Opcode / Instruction / FunctionEntry / Executable）
│   ├── interpreter.py       # 纯 Python VMInterpreter + builtin 实现
│   └── serializer.py        # Executable ↔ 二进制序列化（M9 使用）
└── compiler/passes/
    └── vm_codegen.py        # VMCodegenPass：memory-explicit IR → Executable

runtime/
├── include/devproc2/runtime/
│   ├── vm.h                 # Opcode/Instruction/FunctionEntry/Executable/VMFrame/VMState/BuiltinRegistry
│   ├── packed_func.h        # PackedFuncObj/PackedFunc/PackedFuncRegistry（M8 新增 Registry 部分）
│   └── tensor.h             # 新增 Tensor::FromStorage 工厂方法
└── src/
    ├── vm.cc                # VMState::Invoke / ExecuteLoop / DispatchExternal
    ├── builtins.cc          # RegisterVMBuiltins()：所有 vm.builtin.* C++ 实现
    └── packed_func.cc       # PackedFuncRegistry 全局单例实现
```

---

## 11. 扩展路径

M8 是 VM 的最小可用版本。后续里程碑将在此基础上扩展：

| 里程碑 | 扩展内容 |
|---|---|
| **M9** | ABI + Artifact：`serialize/deserialize`，Executable::Load，abi.json |
| **M10** | PackedFunc 调用：`call_dps_packed`，tokenizer 注册调用 |
| **M11** | Kernel launch：C++ KernelRegistry，`cuLaunchKernel`，Triton cubin 加载 |
| **X1** | CUDADeviceAPI：`cudaMalloc/cudaFree/cudaMemcpyAsync` 替换 CPU stub |
| **X2** | Shape builtin：`shape_of/get_shape_dim/ceildiv/assert_le`，动态 shape 支持 |
| **X3** | Kernel grid expression：`grid_expr` lower 到 VM builtin 序列 |

VM 的 4 条指令不会增加，所有新能力都通过扩展 builtin/packed_func/kernel 函数表实现。
