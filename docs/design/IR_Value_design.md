# IR Value 系统设计文档

## 1. 设计结论

本 IR 的 Value 系统采用如下核心划分：

```text
ValueLike
  ├─ SSAValue
  ├─ SymbolicExpr
  └─ Const

SymbolicExpr
  ├─ Symbol
  ├─ Const
  └─ Expr(op, operands...)
```

其中：

- `SSAValue` 表示真正进入运行时数据流的值
- `Const` 表示编译期内嵌常量
- `SymbolicExpr` 表示符号标量表达式
- `Symbol` 是 `SymbolicExpr` 的叶子节点，用来表达动态 shape、动态 launch 参数、index 变量等符号量

动态 shape 的维度不需要单独设计 `DimExpr`。它本质上就是一个 `Symbol`，例如 `$N`、`$M`、`$seq_len`。

## 2. 为什么不能只用 `SSAValue`

`SSAValue` 很适合表达普通计算图里的数据流，例如 tensor、scalar、buffer、token 等运行时值。但有几类对象不适合直接塞进普通 SSA use-def：

- 编译期常量，例如 `42`、`true`、`"NCHW"`
- 动态 shape 的符号维度，例如 `$N`
- shape/index/bound 表达式，例如 `ceildiv($N, 256)`
- kernel launch 的动态 grid/block 参数，例如 `grid.x = ceildiv($N, 256)`

这些对象有些最终会 lower 成运行时 scalar SSA 值，但在 high-level IR 中，它们更适合作为元层标量表达存在。

## 3. 总体模型

推荐对象关系如下：

```text
Type
  ├─ TensorType(shape: Shape, elem: Type)
  ├─ ScalarType(dtype)
  ├─ IndexType
  ├─ BufferType
  └─ TokenType

Shape
  └─ dims: [SymbolicExpr]

ValueLike
  ├─ SSAValue
  │   ├─ OpResult
  │   └─ BlockArg
  ├─ SymbolicExpr
  │   ├─ Symbol
  │   ├─ ConstExpr
  │   └─ OpExpr
  └─ Const

Operation
  ├─ operands: [SSAValue]
  ├─ results: [SSAValue]
  ├─ attrs: OpAttrDict
  └─ regions: [Region]
```

核心原则：

- `tensor`、`scalar`、`index` 是 `Type`，不是 `Value` 子类
- 普通计算 op 的输入输出统一是 `SSAValue`
- 动态 shape 维度统一是 `Symbol`
- shape/index/launch 参数表达式统一是 `SymbolicExpr`
- 编译期字面量统一是 `Const`
- op 配置项使用 `OpAttr`，不要和 `Const` 混在一起

## 4. `SSAValue`

`SSAValue` 表示 IR 中真正参与运行时数据流的值。

典型对象：

- tensor 输入、输出、中间结果
- scalar 输入、输出、中间结果
- buffer / pointer / memref
- block argument
- function argument
- `constant op` 的结果
- materialize 后的 shape/index/launch 标量值

建议接口：

```cpp
struct SSAValue {
  Type type;
  DefRef def;
  UseList uses;
};

struct OpResult : SSAValue {
  Operation* owner;
  int result_index;
};

struct BlockArg : SSAValue {
  Block* owner;
  int arg_index;
};
```

普通计算 op 只吃 `SSAValue`，也只产生 `SSAValue`。

示例：

```text
%0 = constant 42 : i32
%1 = add %arg0, %0 : i32
%2 = matmul %A, %B : tensor<64x64xf32>
```

这里 `%0`、`%1`、`%2` 都是 `SSAValue`。

## 5. `Const`

`Const` 表示编译期内嵌常量。它不是 SSA 值，不参与 use-def。

典型对象：

- `42`
- `3.14`
- `true`
- `"NCHW"`
- `[1, 2, 3]`
- 静态 shape 维度

建议接口：

```cpp
struct Const {
  Type type;
  ConstKind kind;
};

struct IntConst : Const {
  int64_t value;
};

struct FloatConst : Const {
  double value;
};

struct StringConst : Const {
  std::string value;
};

struct ArrayConst : Const {
  std::vector<Const> values;
};
```

关键约束：

- `Const(42)` 是编译期常量对象
- `%0 = constant 42 : i32` 的 `%0` 是 `SSAValue`
- `constant op` 可以持有一个 `Const` 作为 payload，并产生一个 `SSAValue`

示例：

```text
%0 = constant 42 : i32
```

可以理解为：

```text
ConstantOp(value = Const<i32>(42)) -> SSAValue<i32>
```

## 6. `SymbolicExpr`

`SymbolicExpr` 是符号标量表达式系统。它用于表达 shape、index、bound、offset、kernel launch 参数等元层标量表达。

它不是只表达 `add`、`mul` 这类算术节点。它的核心是：

```text
SymbolicExpr = Symbol | Const | Expr(op, operands...)
```

也就是说，动态 shape 的维度本身就是一个 `SymbolicExpr`。

典型例子：

- `$N`
- `$M`
- `$seq_len`
- `128`
- `$N + 1`
- `ceildiv($N, 256)`
- `$block_id * 256 + $thread_id`

建议接口：

```cpp
struct SymbolicExpr {
  Type type;  // usually index, i64, i32, bool
  SymbolicExprKind kind;
};

struct Symbol : SymbolicExpr {
  SymbolId id;
  std::string name;
  SymbolOrigin origin;
  ConstraintSet constraints;
};

struct ConstExpr : SymbolicExpr {
  Const value;
};

struct OpExpr : SymbolicExpr {
  SymbolicOp op;
  std::vector<SymbolicExpr> operands;
};
```

建议支持的 `SymbolicOp`：

```text
add, sub, mul
floordiv, ceildiv, mod
min, max
eq, ne, lt, le, gt, ge
and, or, not
select
cast
```

如果后续需要表达更复杂的 index map，可以继续扩展 `SymbolicOp`，但第一版不需要把它做成通用编程语言。

## 7. `Symbol`

`Symbol` 是 `SymbolicExpr` 中最重要的叶子节点。它表示一个当前无法静态确定、但可以被追踪和约束的标量量。

动态 shape 的 dim、kernel launch 的动态参数、loop index、外部传入的 runtime scalar，都可以通过 `Symbol` 表达。

推荐字段：

```cpp
enum class SymbolOriginKind {
  InputShapeDim,
  OpResultShapeDim,
  RuntimeScalar,
  LoopIndex,
  KernelParam,
  Unknown
};

struct SymbolOrigin {
  SymbolOriginKind kind;
  void* owner;
  int index;
};

struct ConstraintSet {
  bool non_negative;
  std::optional<int64_t> min_value;
  std::optional<int64_t> max_value;
  std::optional<int64_t> divisible_by;
};
```

例子：

```text
$N : index, origin = InputShapeDim(%x, 0)
$M : index, origin = InputShapeDim(%x, 1)
$B : index, origin = RuntimeScalar(%batch)
```

这里 `$N`、`$M`、`$B` 都是 `Symbol`，不是 `SSAValue`。

## 8. 动态 Shape 表达

Tensor 的 shape 建议直接使用 `SymbolicExpr` 数组：

```cpp
struct TensorType {
  std::vector<SymbolicExpr> shape;
  Type element_type;
};
```

示例：

```text
tensor<$N x 128 x f32>
tensor<$B x $S x 768 x f16>
tensor<ceildiv($N, 16) x 16 x f32>
```

解释：

- 静态维度 `128` 是 `ConstExpr`
- 动态维度 `$N` 是 `Symbol`
- 派生维度 `ceildiv($N, 16)` 是 `OpExpr`

这样可以自然支持动态 shape 推导。

示例：

```text
%y = reshape %x : tensor<$N x $M x f32> -> tensor<$N x (2 * $K) x f32>
```

其中 `$N`、`$M`、`$K` 都是 symbol，`2 * $K` 是 `SymbolicExpr`。

## 9. Scalar 的表达

`scalar` 不应该成为单独的 `Value` 子类，而应该通过 `Type` 区分。

运行时 scalar：

```text
%s = add %a, %b : i32
```

这里 `%s` 是 `SSAValue<i32>`。

编译期 scalar：

```text
Const<i32>(42)
```

这里 `42` 是 `Const`。

符号 scalar：

```text
$N + 1
```

这里 `$N + 1` 是 `SymbolicExpr<index>`。

因此 scalar 在系统里有三种存在形态：

```text
SSAValue<ScalarType>     // 运行时数据流 scalar
Const<ScalarType>        // 编译期常量 scalar
SymbolicExpr<IndexType>  // 元层符号 scalar
```

## 10. Kernel Launch 参数

kernel launch 的 grid/block 参数建议使用 `SymbolicExpr` 表达，lowering 时再 materialize 成 `SSAValue<index>`。

高层表示：

```text
gpu.launch @kernel
  grid  = (ceildiv($N, 256), $M, 1)
  block = (256, 1, 1)
  args  = (%x, %y)
```

这里：

- `$N`、`$M` 是 `Symbol`
- `ceildiv($N, 256)` 是 `SymbolicExpr`
- `256`、`1` 是 `ConstExpr`
- `%x`、`%y` 是普通 `SSAValue`

建议接口：

```cpp
struct LaunchConfig {
  SymbolicExpr grid_x;
  SymbolicExpr grid_y;
  SymbolicExpr grid_z;
  SymbolicExpr block_x;
  SymbolicExpr block_y;
  SymbolicExpr block_z;
};

struct KernelLaunchOp {
  LaunchConfig config;
  std::vector<SSAValue> kernel_args;
};
```

lowering 后表示：

```text
%n = shape.dim %x, 0 : index
%m = shape.dim %x, 1 : index
%c256 = constant 256 : index
%c1 = constant 1 : index
%gx = ceildiv %n, %c256 : index

gpu.launch @kernel
  grid(%gx, %m, %c1)
  block(%c256, %c1, %c1)
  args(%x, %y)
```

lowering 后，launch op 的 grid/block 参数都变成普通 `SSAValue<index>`。

这个过程可以称为：

```text
materialize(SymbolicExpr) -> SSAValue<index>
```

## 11. Materialization

`SymbolicExpr` 只负责表达，不负责直接执行。需要进入真正 runtime 数据流时，通过 materialization 生成 `SSAValue`。

典型触发场景：

- kernel launch 需要 runtime grid/block 参数
- buffer allocation 需要 runtime size
- loop lowering 需要 runtime bound
- shape runtime check 需要实际标量值

建议接口：

```cpp
SSAValue materialize(SymbolicExpr expr, InsertionPoint ip);
```

materialization 规则：

- `ConstExpr(42)` 生成 `constant 42 : index`
- `Symbol($N)` 根据 origin 生成或复用对应的 runtime scalar
- `OpExpr(add, a, b)` 递归 materialize operands，再生成 scalar add op
- `OpExpr(ceildiv, a, b)` 递归 materialize operands，再生成 scalar ceildiv op

示例：

```text
ceildiv($N, 256)
```

materialize 成：

```text
%n = shape.dim %x, 0 : index
%c256 = constant 256 : index
%gx = ceildiv %n, %c256 : index
```

## 12. `OpAttr` 的边界

`OpAttr` 表示操作配置项，不是 Value。

典型对象：

- `layout = "NCHW"`
- `axis = 1`
- `padding = [1, 1]`
- `strides = [2, 2]`
- `dtype = "float32"`

这些配置项可以在实现上复用 `Const` 作为 payload，但语义上它们属于 `OpAttr`。

建议接口：

```cpp
struct OpAttr {
  std::string name;
  AttrValue value;
};

using AttrValue = std::variant<Const, Type, SymbolicExpr, std::string>;
```

注意：

- 静态配置项可以是 `Const`
- 动态 shape / launch 参数不建议藏在普通 `OpAttr` 里
- 会影响数据依赖或 runtime 执行的动态参数，应该显式建模为 `SymbolicExpr` 或 materialized `SSAValue`

## 13. 推荐 IR 写法

### 13.1 普通计算 op

```text
%z = add %x, %y : tensor<$N x f32>
```

解释：

- `%x`、`%y`、`%z` 是 `SSAValue`
- `$N` 是 shape 中的 `Symbol`

### 13.2 常量

```text
%c = constant 128 : index
```

解释：

- `128` 是 `Const`
- `%c` 是 `SSAValue<index>`

### 13.3 动态 shape

```text
%y = reshape %x : tensor<$B x $S x f32> -> tensor<($B * $S) x f32>
```

解释：

- `$B`、`$S` 是 `Symbol`
- `$B * $S` 是 `SymbolicExpr`

### 13.4 动态 kernel launch

```text
gpu.launch @vec_add
  grid  = (ceildiv($N, 256), 1, 1)
  block = (256, 1, 1)
  args  = (%a, %b, %out)
```

解释：

- launch config 是 `SymbolicExpr`
- kernel args 是 `SSAValue`
- lowering 后 launch config 会变成 `SSAValue<index>`

## 14. Pass 设计建议

围绕这套 Value 系统，建议至少准备以下工具：

- `SymbolTable`：维护 `SymbolId`、名称、origin、约束
- `ExprSimplifier`：简化 `SymbolicExpr`
- `ExprSubstitution`：替换 symbol 或表达式
- `ShapeInference`：为 tensor type 生成和传播 shape expr
- `Materializer`：把 `SymbolicExpr` 转成 `SSAValue`
- `LaunchLowering`：把 launch config materialize 成 runtime scalar

`SymbolicExpr` 相关 pass 应尽量保持纯函数风格，因为它们处理的是表达式，不是普通 SSA 图。

## 15. 最终推荐

最终推荐采用这套最小但完整的模型：

```text
SSAValue
  - 表示普通运行时数据流
  - 支持 tensor、scalar、buffer、index
  - 用 Type 区分具体种类

Const
  - 表示编译期内嵌常量
  - 可作为 constant op payload
  - 可作为 SymbolicExpr 叶子

SymbolicExpr
  - 表示 shape/index/bound/launch 参数等符号标量表达
  - 由 Symbol、Const、Expr(op, args...) 组成
  - 需要运行时值时 materialize 成 SSAValue

Symbol
  - 是 SymbolicExpr 的叶子
  - 表示动态 shape 维度、动态 launch 参数、loop index、runtime scalar 绑定等

OpAttr
  - 表示操作配置项
  - 不参与普通 Value 分类
```

这套设计能覆盖：

- 正常计算 op
- `const` 和 `constant op`
- 运行时 scalar
- 动态 shape
- 派生 shape 表达式
- kernel launch 的动态 grid/block 参数
- 后续 lowering 到纯 SSA scalar 计算

