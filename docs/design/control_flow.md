# devproc2 Control Flow MVP 设计文档（Effect-aware Structured Control Flow）

## 1. 背景

devproc2 需要支持控制流，以表达：

- 动态 shape 分支
- runtime dispatch
- kernel orchestration
- memory scheduling
- DPS（Destination Passing Style）调用
- effectful runtime op
- device/runtime control logic

但 devproc2 的目标不是成为完整 Python 编译器。

因此，本阶段只支持：

```python
if cond:
    ...
else:
    ...

for i in dp.range(...):
    ...
```

并采用：

```text
Structured Control Flow + Region + Yield
```

方案。

该方案对齐：

- MLIR SCF (`scf.if` / `scf.for`)
- TVM Relax 高层控制流
- region-based structured IR

MVP 不引入：

```text
CFG
BasicBlock jump
phi node
dominance frontier
SSA construction
```

控制流中的值合流通过：

```text
region result + yield
```

表达。

---

# 2. 核心设计目标

M3 的核心目标不是“支持几个 Python 语法”，而是建立 devproc2 后续控制流能力的正确骨架。

正确骨架是：

```text
Structured Control Flow
+ Region
+ Terminator
+ Yield
+ Optional SSA Result
+ Effect-aware IR
+ IterArgs
+ Scope-aware Frontend
+ Strict Verifier
```

MVP 可以小，但这些核心抽象不能省。

---

# 3. 设计原则

## 3.1 采用结构化控制流 IR

IR 保留源码中的嵌套结构：

```text
Function
  Block
    If
      then_region
      else_region
    For
      body_region
```

而不是：

```text
CFG
BasicBlock
jump edge
phi node
```

---

## 3.2 If / For 既可以产生 result，也可以只有 effect

这是 devproc2 和纯函数式 Tensor IR 最大的区别。

因为 devproc2 已经支持：

```text
DPS kernel
runtime call
memory write
device operation
external packed func
```

这些天然是：

```text
effectful op
```

因此：

```text
If / For 不应该强制必须有 SSA result。
```

控制流有两种模式：

---

### 纯函数式控制流

Python：

```python
if cond:
    y = a + b
else:
    y = a - b
```

IR：

```text
%y = if %cond {
    %v0 = add %a, %b
    yield %v0
} else {
    %v1 = sub %a, %b
    yield %v1
}
```

这里：

```text
If 有 result
Yield 返回 values
```

---

### Effectful 控制流

Python：

```python
if cond:
    dps_add(a, b, out)
else:
    dps_sub(a, b, out)
```

IR：

```text
if %cond {
    call_dps_add(...)
    yield
} else {
    call_dps_sub(...)
    yield
}
```

这里：

```text
If 没有 result
Yield 不返回 values
结果通过 effect 写入已有 Value / Buffer
```

这是 devproc2 必须支持的核心场景。

---

## 3.3 Yield 是 Region Terminator

每个 control-flow region 必须以：

```text
Yield
```

结束。

Yield：

```text
可以返回 values
也可以为空
```

即：

```text
yield %x
```

和：

```text
yield
```

都合法。

---

## 3.4 Python DSL 是受限 Python

devproc2 的 Python DSL 不是完整 Python。

MVP 只支持：

```python
if / elif / else
for i in dp.range(...)
```

暂不支持：

```python
while
break
continue
return inside if/for
for x in iterable
Python object truthiness
Python list/dict mutation
```

目标是：

```text
生成可验证、可 lower、可执行的 IR
```

而不是支持完整 Python 语义。

---

# 4. IR 设计

建议新增文件：

```text
python/devproc2/ir/control_flow.py
```

新增核心节点：

```text
Range
Yield
If
For
IterArg
```

同时扩展：

```text
Block
Terminator
```

---

# 4.1 Block / Region

## 目标

用于承载结构化控制流内部代码。

---

## 建议定义

```python
@dataclass
class Block:
    params: list[Var]
    stmts: list[Stmt]
    terminator: Terminator | None
```

---

## 字段说明

```text
params:
  block 参数。
  MVP 可以先为空。
  后续 For body 可以用 block args 表达 iter args。

stmts:
  block 内部普通语句。

terminator:
  block 终结语句。
  control-flow region 通常以 Yield 结束。
```

---

## 设计要求

不允许：

```text
yield
stmt_after_yield
```

Yield 必须是最后一个节点。

---

# 4.2 Yield

## 目标

从 control-flow region 中返回结果。

---

## 建议定义

```python
@dataclass
class Yield(Terminator):
    values: list[Value]
```

---

## 语义

```text
Yield 只能作为 region terminator；
Yield 不等同于 Return；
Yield 可以返回 values；
Yield 也可以为空；
```

---

## 示例

### 有 values

```text
yield %x
```

### 无 values

```text
yield
```

---

# 4.3 If

## 目标

支持结构化条件分支。

---

## 建议定义

```python
@dataclass
class If(StmtExpr):
    cond: Value
    true_branch: Block
    false_branch: Block | None
    result_structs: list[StructInfo]
```

---

## 字段说明

```text
cond:
  条件值。
  MVP 限制为 bool scalar 或 0-d bool tensor。

true_branch:
  then region。

false_branch:
  else region。

result_structs:
  If 的 SSA result 类型。
  可以为空。
```

---

## 有 result 的 If

Python：

```python
if cond:
    y = a + b
else:
    y = a - b
```

IR：

```text
%y = if %cond {
    %v0 = add %a, %b
    yield %v0
} else {
    %v1 = sub %a, %b
    yield %v1
}
```

---

## Effect-only If

Python：

```python
if cond:
    dps_add(a, b, out)
else:
    dps_sub(a, b, out)
```

IR：

```text
if %cond {
    call_dps_add(...)
    yield
} else {
    call_dps_sub(...)
    yield
}
```

这里：

```text
If.result_structs == []
```

---

# 4.4 Range

## 目标

显式表达：

```python
dp.range(start, end, step)
```

---

## 建议定义

```python
@dataclass
class Range:
    start: Value
    end: Value
    step: Value
```

---

## MVP 规则

```text
start/end/step 必须是 int scalar 或 symbolic int；
step != 0；
MVP 可以先只支持正向 range；
默认 step = 1。
```

---

# 4.5 IterArg

## 目标

表达 loop-carried variable。

---

## 建议定义

```python
@dataclass
class IterArg:
    var: Var
    init: Value
```

---

## 语义

```text
init:
  初始值。

var:
  body 内部当前迭代值。

body yield:
  下一轮迭代值。
```

---

# 4.6 For

## 目标

支持结构化 loop。

---

## 建议定义

```python
@dataclass
class For(StmtExpr):
    loop_var: Var
    range: Range
    iter_args: list[IterArg]
    body: Block
    result_structs: list[StructInfo]
```

---

## 字段说明

```text
loop_var:
  induction variable。
  只在 body 内可见。

range:
  dp.range 的 IR 表达。

iter_args:
  loop-carried variables。
  可以为空。

body:
  loop body region。

result_structs:
  loop SSA result。
  可以为空。
```

---

## Effect-only For

Python：

```python
for i in dp.range(0, n):
    dps_kernel(...)
```

IR：

```text
for %i in range(0, %n, 1) {
    call_dps_kernel(...)
    yield
}
```

这里：

```text
没有 iter_args
没有 SSA result
只有 effect
```

---

## Loop-carried For

Python：

```python
acc = init
for i in dp.range(0, n):
    acc = acc + x
```

IR：

```text
%acc_out = for %i in range(0, %n, 1)
           iter_args(%acc_iter = %init) {
    %next = add %acc_iter, %x
    yield %next
}
```

---

# 5. Effect-aware IR

devproc2 不是纯函数式 Tensor IR。

因为它已经支持：

```text
runtime
memory
DPS
kernel launch
device op
external call
```

因此 control flow 必须兼容 effect。

---

## 5.1 建议引入 Effect 信息

建议预留：

```python
class EffectKind(Enum):
    PURE
    READ
    WRITE
    READ_WRITE
    OPAQUE
```

或者：

```python
class Op:
    has_side_effect: bool
```

即使 MVP 不实现完整 effect system，也必须让 IR 对 effect 友好。

---

## 5.2 为什么重要

未来这些优化都会依赖 effect 信息：

```text
DCE
CSE
loop optimization
control-flow simplification
hoist
memory scheduling
async scheduling
```

例如：

```python
if cond:
    pure_add(...)
else:
    pure_add(...)
```

未来可能可以 hoist。

但：

```python
if cond:
    dps_write(...)
else:
    dps_write(...)
```

不能乱动。

---

# 6. Frontend Lowering

主要文件：

```text
python/devproc2/frontend/dsl.py
python/devproc2/frontend/scope.py
```

---

# 6.1 Scope 管理

建议引入：

```python
class ScopeFrame:
    bindings: dict[str, Value]

class ScopeStack:
    ...
```

需要支持：

```text
function scope
if scope
for scope
loop_var scope
iter_arg rebinding
If/For result 回写
```

---

# 6.2 If Lowering

## 支持语法

```python
if cond:
    ...
elif cond2:
    ...
else:
    ...
```

---

## Lowering 规则

```text
1. lower cond；
2. cond 必须是 bool scalar；
3. then/else 分别进入子 scope；
4. 分析变量合流；
5. 必要时生成 SSA result；
6. effect-only if 不生成 result；
7. 每个 region 末尾插入 Yield；
8. elif lower 成 nested If。
```

---

## 变量合流

Python：

```python
if cond:
    y = a + b
else:
    y = a - b
```

IR：

```text
%y = if %cond {
    yield %v0
} else {
    yield %v1
}
```

---

## Effect-only 分支

Python：

```python
if cond:
    dps_write_a(out)
else:
    dps_write_b(out)
```

IR：

```text
if %cond {
    dps_write_a(out)
    yield
} else {
    dps_write_b(out)
    yield
}
```

---

# 6.3 For Lowering

## 支持语法

```python
for i in dp.range(...):
    ...
```

---

## 不支持

```python
for i in range(...)
for x in iterable
```

---

## Lowering 规则

```text
1. 捕获 ast.For；
2. iter 必须是 dp.range；
3. lower start/end/step；
4. 创建 Range；
5. 检测 loop-carried variable；
6. 如果存在 loop-carried variable，则生成 iter_args；
7. 如果只有 effect，则不生成 iter_args；
8. body 末尾插入 Yield；
9. 创建 For。
```

---

## Loop-carried variable 检测

MVP 使用保守规则：

```text
变量来自 loop 外部作用域；
并且在 body 内被重新赋值；
则成为 iter_arg。
```

---

# 7. Normalize / Verify

建议文件：

```text
python/devproc2/compiler/passes/control_flow_normalize.py
python/devproc2/compiler/passes/control_flow_verify.py
```

---

# 7.1 Normalize

主要任务：

```text
elif -> nested If；
dp.range 默认参数补齐；
规范化 Yield；
规范化 iter_args 顺序；
推导 result_structs。
```

注意：

```text
Normalize 不应该 lower 成 CFG。
```

---

# 7.2 Verify

## If 校验

### 有 result

```text
result_structs 非空：
  then/else 必须 yield values；
  yield values 数量必须一致；
  StructInfo 必须一致。
```

### 无 result

```text
result_structs 为空：
  then/else 必须 yield 空；
  或允许 yield values=[]。
```

---

## For 校验

### 有 iter_args

```text
iter_args 数量
==
yield values 数量
==
result_structs 数量
```

### 无 iter_args

允许：

```text
iter_args=[]
yield=[]
result_structs=[]
```

---

## Scope 校验

```text
禁止未定义变量；
禁止 loop_var 泄漏；
禁止 Yield 后还有语句；
禁止 unsupported control flow。
```

---

# 8. MVP 支持范围

## 支持

### 有 result 的 If

```python
if cond:
    y = a + b
else:
    y = a - b
```

---

### Effect-only If

```python
if cond:
    dps_add(...)
else:
    dps_sub(...)
```

---

### Loop-carried For

```python
acc = init
for i in dp.range(0, n):
    acc = acc + x
```

---

### Effect-only For

```python
for i in dp.range(0, n):
    dps_kernel(...)
```

---

### Nested Control Flow

```python
for i in dp.range(0, n):
    if cond:
        dps_write_a(...)
    else:
        dps_write_b(...)
```

---

# 9. MVP 暂不支持

```python
while
break
continue
return inside if/for
for x in iterable
Python truthiness
list/dict mutation
```

---

# 10. 推荐文件结构

```text
python/devproc2/ir/control_flow.py
  - Range
  - Yield
  - If
  - For
  - IterArg

python/devproc2/ir/block.py
  - Block
  - Terminator

python/devproc2/frontend/scope.py
  - ScopeFrame
  - ScopeStack

python/devproc2/frontend/dsl.py
  - visit_If
  - visit_For
  - lower_dp_range
  - merge_binding
  - loop_carried_analysis

python/devproc2/compiler/passes/control_flow_normalize.py

python/devproc2/compiler/passes/control_flow_verify.py
```

---

# 11. 实施拆分

# 11.1 M3.1：IR Skeleton

目标：

```text
建立结构化控制流 IR 骨架。
```

任务：

```text
新增 If / For / Yield / Range / IterArg；
扩展 Block terminator；
支持 pretty print；
支持 effect-only control flow；
支持 verifier。
```

---

# 11.2 M3.2：If Frontend

目标：

```text
支持 Python if/elif/else lowering。
```

任务：

```text
AST 捕获；
分支 scope；
变量合流；
result If；
effect-only If；
nested If。
```

---

# 11.3 M3.3：For Frontend

目标：

```text
支持 dp.range for-loop lowering。
```

任务：

```text
Range lowering；
iter_args 检测；
effect-only loop；
Yield 插入；
scope 回写。
```

---

# 11.4 M3.4：Nested Control Flow

目标：

```text
支持 if/for 嵌套。
```

---

# 11.5 M3.5：Normalize + Verify

目标：

```text
保证 IR 合法性。
```

任务：

```text
检查 Yield；
检查 iter_args；
检查 cond 类型；
检查 scope；
拒绝 unsupported control flow。
```

---

# 12. 后续扩展预留

当前设计必须允许未来扩展：

```text
while
break/continue
early return
shape-dependent control flow
loop optimization
memory effect analysis
async scheduling
CFG lowering
```

而不需要推翻现有 IR。

---

# 13. 最终结论

devproc2 的 control flow 不应该走：

```text
纯函数式 Tensor IR
```

因为 devproc2 已经明确支持：

```text
runtime
memory
DPS
kernel launch
device op
```

因此正确路线是：

```text
Structured Control Flow
+
Effect-aware IR
+
Optional SSA Result
```

也就是：

```text
If / For 可以返回 SSA result；
也可以只表达 effect；
Yield 可以返回 values；
也可以为空；
For 可以有 iter_args；
也可以是纯 effect loop。
```

这个方向会比纯 Relax 风格更适合 devproc2 的 runtime 和端侧场景。

