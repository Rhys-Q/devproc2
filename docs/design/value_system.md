# Value 系统设计

devproc2 IR 中有两类独立的"值"对象，它们服务于不同的层次：

```
Value          — 可作为 Op 操作数的运行时值对象
PrimExpr       — 符号标量表达式，用于 shape / index / launch 参数
```

两者不混用：Op 的 `args`/`inputs`/`output` 字段只接受 `Value`；
`TensorStructInfo.shape` 只包含 `PrimExpr`。

---

## 1. Value — 可作为 Op 操作数的运行时值对象

```
Value
  ├─ Var          # SSA def（block 参数：函数参数、循环迭代变量）
  ├─ OpResult     # SSA def（Op 产生的结果）
  └─ Constant     # inline operand，不是 SSA def，不占用名字
```

`Var` 和 `OpResult` 是真正的 SSA def-use 节点，参与 use-def 链；
`Constant` 是 inline operand，直接嵌入 Op，没有 def 点，也不加入 use-def 链。

定义在 `python/devproc2/ir/nodes.py`。

### 身份语义（Identity Equality）

**所有 IR graph node（Var、OpResult、Op、Block、Region）必须使用身份语义，不使用结构相等。**
因为这些节点的字段可能形成循环引用（OpResult → Op → results → OpResult），
dataclass 默认的结构比较在此场景下会无限递归或产生错误的相等结论。

正确做法：

```python
@dataclass(frozen=True, eq=False)
class OpResult(Value):
    ...

@dataclass(frozen=True, eq=False)
class Var(Value):
    ...
```

`eq=False` 使 `==` 退回 Python 默认的 `is`（对象身份），`hash` 退回 `id(self)`。
这样 `OpResult` / `Var` 可以直接放进 set / dict，键是对象身份，而不是字段结构。

> 需要结构比较时，单独写工具函数，不要依赖 `==`。

### Var

```python
@dataclass(frozen=True, eq=False)
class Var(Value):
    name: str
    struct_info: Optional[StructInfo] = None
```

`Var` 表示 block entry 处定义的 SSA 值——函数参数或 `ForOp`/`IfOp` 的迭代参数。
它不由任何 Op 产生，而是在 `Block.args` 中声明。

### OpResult

```python
@dataclass(frozen=True, eq=False)
class OpResult(Value):
    op:    Op
    index: int
    struct_info: Optional[StructInfo] = None
```

由 Op 在 `__post_init__` 中生成，存入 `op.results[index]`。
打印名（`%y` 等）存在 Op 上（`result_name` / `result_names`），不在 OpResult 本身。

使用对象身份：`OpResult` 直接作为 dict/set 键，`IRRewriter._sub` 用 `id(result)` 或
直接以 `result` 对象为键均可（`eq=False` 下两者等价）。

### Constant

```python
@dataclass(frozen=True)
class Constant(Value):
    value: Union[int, float, bool, None]
```

编译期**标量**常量，直接嵌入 Op 的操作数，不占用 SSA 名字，不参与 def-use 链。
典型用途：loop bound 字面量、scalar fill value 等。

```python
Range(Constant(0), n, Constant(1))       # loop bound
CallOp(args=(x, Constant(0.5)), ...)     # scalar fill value
```

**`Constant` 不支持 tensor / ndarray**。原因：
- 没有 SSA 名字，无法在多处引用，也无法参与 effect 分析；
- 把大型权重矩阵塞进 inline operand，语义错误；
- `np.ndarray` 的 `==` 返回数组而非 `bool`，frozen dataclass 的 hash/equality 行为异常；
- ndarray 本身可变，放进 frozen dataclass 不代表不可变。

**模型权重的正确表示方式**见下一节。

---

## 2. 模型权重的表示

权重是推理系统中最大的数据源，有三种场景，各自的正确表示方式不同：

| 场景 | 表示方式 | 说明 |
|---|---|---|
| 运行时传入的权重（标准推理路径） | `Var`（函数参数） | 运行时由调用方绑定，IR 只持有形参 |
| 静态嵌入的小型 tensor 常量（lookup table、量化 scale 等） | `ConstantTensorOp`（待实现） | 产生具名 `OpResult`，参与 def-use |
| 标量字面量（0、1、0.5…） | `Constant` | 当前已有，仅限 scalar |

### 运行时权重（Var）

推理时权重从外部传入：

```python
@dp.function
def forward(x: Tensor[...], weight: Tensor[...], bias: Tensor[...]):
    y = linear(x, weight)
    return y + bias
```

`weight`、`bias` 都是 `Var`，运行时绑定具体数据。
这也是 TVM Relax、ONNX Runtime 等主流框架的做法。

### 静态嵌入权重（ConstantTensorOp，未来）

当权重需要编译期内联（例如小型 embedding table、per-layer scale）时，
应引入 `ConstantTensorOp`，而不是扩展 `Constant`：

```python
@dataclass(frozen=True, eq=False)
class ConstantTensorOp(Op):
    """Embeds a compile-time tensor as an SSA value."""
    result_name: str
    data:  NDArrayConst         # immutable wrapper: copies array, marks read-only
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))
```

```
%scale = constant_tensor [1.0, 0.5, ...] : tensor<128xf16>
%y     = mul %x, %scale
```

`ConstantTensorOp` 产生的 `OpResult` 有 SSA 名字，可在多处引用，
可做 CSE，也可参与 effect 分析（read-only）。

---

## 3. StructInfo — 类型与 shape 元信息

```python
class StructInfo: ...

@dataclass(frozen=True)
class TensorStructInfo(StructInfo):
    shape:  tuple[PrimExpr, ...]
    dtype:  str     # "float16", "float32", ...
    device: str     # "cuda", "cpu", ...
```

`StructInfo` 通过 `struct_info` 字段附加到 `Var` 或 `OpResult` 上，
是可选的——未推导时为 `None`。

`InferStructInfoPass` 负责将 DSL 层的类型标注传播到 IR，
填充所有 `CallOp` result 的 `struct_info`。

---

## 4. PrimExpr — 符号标量表达式

```
PrimExpr
  ├─ IntImm(value: int)              # 整型常量
  ├─ PrimVar(name, id, upper)        # 符号变量（动态 shape 维度、loop bound 等）
  └─ 二元节点
       Add / Sub / Mul / FloorDiv / CeilDiv / Min / Max
       EQ / LT / LE / GT / GE
```

定义在 `python/devproc2/ir/prim_expr.py`，设计参考 TVM `tir.PrimExpr`。

### 身份语义

**PrimExpr 节点同样使用身份语义（`eq=False`）。**

```python
@dataclass(frozen=True, eq=False)
class PrimVar(PrimExpr):
    ...

@dataclass(frozen=True, eq=False)
class Add(PrimExpr):
    ...
```

原因：
- 若使用结构相等，两个 `PrimVar("B")` 实例会被认为是同一个符号，
  但它们可能来自不同 scope 或不同 pass 的不同引入点。
- `PrimExpr` 节点可能出现在 shape、断言、launch config 等多处，
  直接作 set/dict 键时应以对象身份区分。

符号条件表达式用 `.eq()` / `.lt()` 等方法显式构造 `EQ`/`LT` 节点：

```python
B.eq(IntImm(8))   # 符号等价，返回 EQ 节点
B < IntImm(8)     # 运算符重载，同上
```

### PrimVar — 符号变量

```python
@dataclass(frozen=True, eq=False)
class PrimVar(PrimExpr):
    name:  str
    id:    int               # 唯一身份标识，由 DSL context 分配
    upper: Optional[int] = None
```

- `name`：仅用于打印，不代表身份。
- `id`：真实身份标识符，由 DSL 层的 `SymbolContext` 在创建时分配（自增整数）。
  两个 `PrimVar("B")` 若 `id` 不同，则是不同的符号变量。
- `upper`：devproc2 扩展字段（TVM 没有），供 `MemoryPlanningPass` 估算最大 buffer size，
  以及 `ShapeAssertOp` 在运行时验证 shape 不超界。

DSL 层通过 `dp.symbolic_dim("B", upper=8)` 创建 `PrimVar`，
`SymbolContext` 保证同一 `dp.symbolic_dim` 调用返回同一个 `PrimVar` 对象（interning）。

### 运算符重载

```python
B = dp.symbolic_dim("B", upper=8)
S = dp.symbolic_dim("S", upper=2048)
expr = ceildiv(B * S, 256)
# → CeilDiv(Mul(PrimVar("B", id=0), PrimVar("S", id=1)), IntImm(256))
```

### 使用场景

| 场景 | 示例 |
|---|---|
| Tensor shape 维度 | `TensorStructInfo(shape=(B, S, IntImm(4096)), ...)` |
| TensorCreateOp shape | `TensorCreateOp(shape=(B, S, IntImm(512)), ...)` |
| 运行时 shape 断言 | `ShapeAssertOp(tensor=x, dim_idx=0, upper=B.upper)` |
| Kernel launch grid（M11） | `ceildiv(S, IntImm(128))` → 物化为 SSA scalar 后传给 GPU launch Op |

---

## 5. 两套系统的分工

```
┌──────────────────────────────────────────────────────────┐
│  Op.args / inputs / output  ──→  Value（运行时数据流）   │
│  TensorStructInfo.shape     ──→  PrimExpr（符号维度）    │
└──────────────────────────────────────────────────────────┘
```

一个典型的 `CallDPSOp` 示例：

```
%buf = dp.empty(%B, %S, 512) : tensor<B×S×512xf16>   # TensorCreateOp
call_dps @kernel.layernorm_fp16 [%x] → %buf            # CallDPSOp
```

- `%x`, `%buf` 是 `OpResult`（Value）；
- `B`, `S`, `IntImm(512)` 是 `PrimExpr`，存在 `%buf` 的 `struct_info.shape` 里，
  不直接出现在 Op 的操作数中。

M7 的 `ShapeExprLoweringPass` 将把 shape 中的 `PrimExpr` 物化为运行时 SSA scalar，
注入 `CallDPSOp.inputs`，供 kernel 取得实际维度值。

### Kernel launch 的两个阶段

高层（M6 之前）的 launch config 用 `PrimExpr` 表示：

```python
grid = (ceildiv(S, IntImm(128)), B, IntImm(1))   # PrimExpr tuple
```

低层（M11 GPU launch Op）的 launch config 用 `Value`（已物化的 SSA scalar）：

```
%gx = ceildiv %s_val, %c128    # %s_val, %c128 是 OpResult（Value）
gpu.launch grid(%gx, %b_val, %c1) block(...) args(%x, %buf)
```

**同一字段不能一会儿是 `PrimExpr`，一会儿是 `Value`**：
高层 launch Op 和低层 GPU launch Op 应该是不同的 Op 类型，
由 `ShapeExprLoweringPass` 在两者之间做转换。

---

## 6. EffectInfo — 副作用标注

`CallDPSOp` 带有一个 `effect` 字段，描述内核对内存的访问模式：

```
EffectInfo
  ├─ PureEffect        # 无副作用（纯函数）
  ├─ ReadOnlyEffect    # 只读
  ├─ WriteEffect(vars) # 写指定变量
  └─ OpaqueEffect      # 未知（当前 M6 默认）
```

M5 的 effect 分析 pass 将基于此推导 DCE 和 alias 信息。

---

## 7. 不变量与约束

- **Value 子类语义**：`Var` / `OpResult` 是 SSA def-use 节点；`Constant` 是 inline operand，不存在 def 点。
- **IR graph node 使用身份语义**：`Var`、`OpResult`、`Op`、`Block`、`Region` 均应 `eq=False`，`==` 等价于 `is`；需要结构比较时单独写工具函数。
- **PrimExpr 使用身份语义**：`PrimVar` 及所有 `PrimExpr` 节点 `eq=False`；符号条件表达用 `.eq()` / `.lt()` 等方法构造节点，不用 `==`。
- **PrimVar 的 `name` 只用于打印**：真实身份由 `id`（或 object identity）决定；两个 `PrimVar("B")` 若对象不同，则是不同符号。
- **单赋值**：每个 `OpResult` 恰好由一个 Op 的 `__post_init__` 创建，不可重赋值（frozen）。
- **`Var` 只在 block entry 定义**：不能由 Op 产生，只出现在 `Block.args` 中。
- **结果名存在 Op 上**：`result_name: str`（单结果 Op）或 `result_names: tuple[str, ...]`（多结果 Op）。
- **CallOp 最多一个结果**：多输出通过 `TupleOp` + `TupleGetItemOp` 组合。
- **PrimExpr 不是 Value，物化前不出现在 Op.args 中**：shape 维度在 `ShapeExprLoweringPass` 之前只存在于 `StructInfo.shape`；在 kernel launch lowering、codegen 之前必须完成物化。

---

## 8. 演化路径

| 里程碑 | 扩展点 |
|---|---|
| M7 ShapeExprLoweringPass | 将 `TensorStructInfo.shape` 中的 `PrimVar`/`PrimExpr` 物化为 SSA scalar，注入 `CallDPSOp.inputs` |
| M5 Effect 分析 | 填充 `WriteEffect(vars)` 替换当前的 `OpaqueEffect` |
| ConstantTensorOp（按需） | 静态嵌入小型 tensor 常量；引入 `NDArrayConst` 不可变包装；`ConstantTensorOp` 产生 `OpResult`，effect 标注为 `ReadOnlyEffect` |
| M11 Kernel launch config | 高层 launch Op（`PrimExpr` grid/block）→ 低层 GPU launch Op（`Value` grid/block），两个 Op 类型由 `ShapeExprLoweringPass` 转换 |
