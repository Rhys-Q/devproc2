# devproc2 优雅 IR 设计：从 MVP IR 到长期可优化 IR

## 1. 目标

这份文档回答一个问题：

```text
什么样的 IR 才算我认可的优雅 IR？
```

这里的“优雅”不是打印出来好看，也不是抽象层级多，而是长期工程上能扛住复杂度：

- 新 op 能自然接入，不需要每个 pass 都补特殊分支。
- 每个 stage 都有硬 contract，不只是一个名字。
- type、shape、device、effect、alias、memory 都有一等语义。
- pass 读的是 IR 语义，不是字符串约定和历史习惯。
- 高层表达、优化表达、低层执行表达边界清楚。
- verifier 能发现不合法 IR，而不是让后端在奇怪位置炸掉。
- textual IR 和序列化只是 IR 的展示方式，不是 IR 的语义来源。

一句话目标：

```text
devproc2 IR 应该保持 Python 实现的轻量感，
但语义边界要接近 MLIR / TVM Relax 这类优化 IR 的严谨度。
```

## 2. 设计立场

devproc2 不应该走两个极端。

第一个极端是“全泛化 MLIR clone”。所有 op 都变成一个巨大的 generic operation，然后靠 dialect registry 和 verifier 完成一切。这很强，但对 devproc2 当前体量太重，会把实现成本提前拉满。

第二个极端是继续维持“能跑的 dataclass IR”。每个 op 一个 Python dataclass，每个 pass 知道一堆具体类，遇到新语义就补 `isinstance`。这很快，但随着 dynamic shape、object、KVCache、external call、memory reuse、kernel selection 增多，会快速变成隐式规则堆叠。

我认可的路线是中间层：

```text
实现形态保持轻量 dataclass，
语义模型向 MLIR / Relax 靠拢，
stage contract 由 verifier 强制，
pass 之间通过一等语义对象交接。
```

换句话说，IR core 可以是 Pythonic 的，但 IR 语义不能是随意的。

## 3. 优雅 IR 的判断标准

### 3.1 语义局部性

读一个 op，就应该知道它的基本语义：

- 它属于哪个 dialect。
- 它读哪些 operand。
- 它产生哪些 result。
- 它有没有 region。
- 它有没有 side effect。
- 它的 attrs 是什么类型。
- 它的结果 struct info 是否已经推导。

不应该出现这种情况：

```text
这个 CallOp 到底是 high-level tensor op、runtime builtin、external call，
还是已经 lowered 的 kernel call，需要去看 callee 字符串和某个 pass 的习惯。
```

### 3.2 Stage 是 contract，不是标签

每个 stage 必须回答三个问题：

- 允许出现什么 op。
- 每个 value 必须携带什么 metadata。
- 哪些高层语义已经被 lowering 掉。

如果一个 IR 标成 `DPSIR`，它还残留需要 kernel lowering 的 high-level tensor op，这就不是 DPSIR。verifier 应该拒绝它。

### 3.3 Value 是 SSA 语义，不是名字游戏

`Value` 的身份应该来自 IR graph 中的定义点：

- `BlockArg` 表示 region/block 入口定义。
- `OpResult` 表示 operation 结果。
- `Constant` 表示 inline 标量常量，不是 SSA def。

名字只用于打印和调试。pass 不应该通过名字判断 use-def 关系。序列化时可以给 value 分配稳定 id，但内存中的 IR 仍然应该以定义点为准。

### 3.4 OpRef 是调用语义入口

调用类 op 不应该靠裸字符串区分种类。IR 内部应该明确区分：

- standard op
- builtin op
- external function
- kernel
- packed function

printer 可以打印 `@matmul`，但 pass 不应该通过 `@`、前缀、后缀、字符串表来猜语义。

### 3.5 Effect 和 alias 是优化基础设施

只要 IR 里有 runtime call、DPS kernel、memory planning、in-place write，就必须认真建模 effect 和 alias。

否则 memory planning、CSE、fusion、dead code elimination、in-place lowering 都会各自维护一套局部判断，最后语义不一致。

### 3.6 抽象必须服务 pass

优雅 IR 不是“看起来抽象”，而是能让 pass 写得更少、更稳。

一个判断方法：

```text
新增一个 op 后，需要改多少 unrelated pass？
```

如果每个 pass 都要手写一个 `elif isinstance(NewOp)`，说明 IR 缺少公共接口或者公共语义服务。

## 4. 目标 IR 总体模型

目标结构：

```text
IRModule
  └─ Function
      ├─ params: BlockArg[]
      ├─ ret_struct_info
      ├─ attrs
      └─ body: Region
          └─ Block
              ├─ args: BlockArg[]
              └─ ops: Operation[]

Value
  ├─ BlockArg
  ├─ OpResult
  └─ Constant

Operation
  ├─ op_ref
  ├─ operands
  ├─ results
  ├─ attrs
  ├─ regions
  ├─ effects
  └─ source_span
```

这里有两个层次：

- 通用层：`Operation` 的公共语义接口。
- 便利层：`IfOp`、`ForOp`、`TensorCreateOp` 等 Python dataclass 仍然可以存在。

也就是说，devproc2 不一定要立刻把所有 op 改成一个 generic `Operation` 类，但每个 concrete op 必须能投影到同一套公共接口：

```python
class Op:
    @property
    def op_ref(self) -> OpRef: ...

    @property
    def operands(self) -> tuple[Value, ...]: ...

    @property
    def regions(self) -> tuple[Region, ...]: ...

    @property
    def effects(self) -> EffectSummary: ...
```

这样 rewriter、verifier、use-def 分析、alias 分析可以优先消费公共接口，而不是到处写具体 op 分支。

## 5. 核心类型设计

### 5.1 Value

目标 value 层级：

```text
Value
  ├─ BlockArg
  ├─ OpResult
  └─ Constant
```

`BlockArg` 替代当前语义较宽的 `Var` 名称。它明确表达“这是 block 入口定义的 SSA value”，可用于函数参数、loop iter arg、region 参数。

推荐结构：

```python
@dataclass(frozen=True, eq=False)
class BlockArg(Value):
    name: str
    struct_info: StructInfo | None = None
    owner: Block | None = None
    index: int = 0

@dataclass(frozen=True, eq=False)
class OpResult(Value):
    op: Op
    index: int
    struct_info: StructInfo | None = None

@dataclass(frozen=True)
class Constant(Value):
    value: int | float | bool | None
```

规则：

- `BlockArg` 和 `OpResult` 使用 identity equality。
- `Constant` 只表示小型标量字面量。
- tensor 常量必须是 `ConstantTensorOp`，不能塞进 inline `Constant`。
- value 名字只用于打印，不参与语义判断。

### 5.2 OpRef

目标 op ref 层级：

```text
OpRef
  ├─ StandardOpRef
  ├─ BuiltinOpRef
  ├─ ExternalFuncRef
  ├─ KernelRef
  └─ PackedFuncRef
```

语义：

- `StandardOpRef`：高层 tensor / shape / object 标准 op，必须能从 registry 解析到 schema。
- `BuiltinOpRef`：VM/runtime 内建 op，语义由 runtime 固定实现。
- `ExternalFuncRef`：用户或系统 external call，必须有 effect summary。
- `KernelRef`：已选择的设备 kernel，指向 kernel registry spec。
- `PackedFuncRef`：runtime packed function，面向 VM ABI。

规则：

- high-level tensor op 必须使用 `StandardOpRef`。
- lowered DPS kernel call 必须使用 `KernelRef` 或 `PackedFuncRef`。
- external call 不允许伪装成 unknown standard op。
- verifier 负责拒绝 stage 中不合法的 ref kind。

### 5.3 Operation

长期应该把“调用类 op”拆清楚：

```text
StandardCallOp
BuiltinCallOp
ExternalCallOp
CallDPSOp
```

它们不一定必须立刻变成四个 Python 类，但 IR 语义上必须能区分。

推荐模型：

```python
@dataclass(frozen=True, eq=False)
class StandardCallOp(Op):
    op_ref: StandardOpRef
    operands: tuple[Value, ...]
    attrs: AttrDict
    result_names: tuple[str, ...]

@dataclass(frozen=True, eq=False)
class CallDPSOp(Op):
    target_ref: KernelRef | PackedFuncRef | BuiltinOpRef
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    attrs: AttrDict
    effects: EffectSummary
```

关键区别：

- `StandardCallOp` 是高层表达，产生 SSA result。
- `CallDPSOp` 是 destination-passing 表达，不产生 tensor result，通过 output buffer 写入。
- `ExternalCallOp` 必须显式携带 effect，不允许默认 pure。
- `BuiltinCallOp` 只用于 runtime builtin，不和 high-level tensor op 混用。

### 5.4 AttrValue

attrs 必须是编译期常量，不能引用 SSA value。

目标层级：

```text
AttrValue
  ├─ IntAttr
  ├─ FloatAttr
  ├─ BoolAttr
  ├─ StringAttr
  ├─ DTypeAttr
  ├─ DeviceAttr
  ├─ PrimExprAttr
  ├─ ShapeAttr
  ├─ ArrayAttr
  ├─ DictAttr
  └─ NoneAttr
```

规则：

- op schema normalize 后，IR 中只保留 `AttrValue`。
- printer、JSON serializer、schema validator 共享同一套 attr 表示。
- runtime-varying 数据必须作为 operand，不能伪装成 attr。
- axis、dtype、device、layout 这类语义值要有明确 attr type。

### 5.5 StructInfo

目标 struct info：

```text
StructInfo
  ├─ TensorStructInfo(shape, dtype, device)
  ├─ ScalarStructInfo(dtype)
  ├─ ShapeStructInfo(ndim, values?)
  ├─ TupleStructInfo(fields)
  ├─ ObjectStructInfo(type_key, role)
  └─ FuncStructInfo(params, ret, effects)
```

规则：

- `InferredIR` 以后，所有 op result 必须有 `struct_info`。
- tuple result 不应该靠 pass 临时展开，应该由 `TupleStructInfo` 表达。
- shape value 和 tensor value 要分开，不要把 shape 当普通 tensor。
- object value 必须有 `type_key`，例如 tokenizer、KVCache、runtime handle。
- function value 如果出现，必须表达参数、返回值和 effect。

### 5.6 EffectSummary

目标 effect 模型：

```text
EffectSummary
  ├─ reads: Value[]
  ├─ writes: Value[]
  ├─ allocates: bool
  ├─ frees: bool
  ├─ opaque: bool
  ├─ external_state: str | None
  └─ alias: AliasSummary
```

语义：

- `pure`：不读写外部 state，只由 operands 决定结果。
- `readonly`：读取某些 value 或 external state，但不写。
- `write`：写入明确 value。
- `opaque`：无法精确建模，优化必须保守。
- `external_state`：例如 filesystem、network、runtime session、device stream。

规则：

- DPS kernel 至少要声明 outputs 为 writes。
- external call 默认不应该是 pure。
- memory planning 必须消费 effect，而不是只看 operands。
- CSE 和 DCE 必须尊重 effect。

### 5.7 AliasInfo

alias 不应该只是一个备用 dataclass，而应该成为公共分析结果。

目标模型：

```text
AliasInfo
  ├─ no_alias
  ├─ may_alias
  ├─ must_alias
  └─ view_of(source, offset?, shape?, strides?)
```

用途：

- 判断 return value 是否把内部 tensor 逃逸到 caller。
- 判断两个 tensor 是否可以共享 storage。
- 判断 view op 是否只是别名，不是真分配。
- 支撑 in-place lowering。
- 支撑 external call 的 conservative barrier。

规则：

- Tuple、TupleGetItem、If result、For iter result 这类 forwarding 语义应由公共 alias 分析处理。
- memory planning 不应该自己维护一套局部 alias graph。
- lowering pass 生成新 value 时必须维护 alias 信息或使其可重新推导。

## 6. Dialect 设计

推荐 dialect：

```text
tensor   : high-level tensor compute
shape    : shape value, symbolic shape op, assertion
object   : runtime object, tokenizer, KVCache, model handle
memory   : storage allocation, tensor view, buffer view
runtime  : packed call, builtin call, external call, VM primitive
control  : if, for, yield, return
```

边界：

- `tensor` 不直接表达 raw storage。
- `memory` 不表达 high-level compute。
- `runtime` 不假装是 pure tensor op。
- `control` 只表达结构化控制流，不藏设备执行语义。
- `object` 表达 runtime resource，不混进普通 tensor attrs。

这种分层的目的不是追求分类漂亮，而是让 verifier 和 pass pipeline 能用 dialect 做第一层合法性判断。

## 7. Stage Contract

### 7.1 RawIR

来源：Python frontend。

允许：

- frontend 直接捕获的 high-level op。
- 未完全规范化的 attrs。
- 结构化控制流。
- 部分缺失的 struct info。

禁止：

- 低层 VM-only op。
- 已经绑定具体 storage 的 memory op，除非是 frontend 显式资源。

目标：

```text
RawIR 保留用户程序结构，不负责优化友好。
```

### 7.2 NormalizedIR

来源：control-flow normalize、attr normalize、op ref resolve。

要求：

- 所有 standard op 都有可解析 `StandardOpRef`。
- attrs 已规范化成 `AttrDict`。
- control flow region 形态合法。
- effect-only control flow 和 result-yielding control flow 区分明确。

禁止：

- unknown standard op。
- 字符串 callee 推断语义。
- 不合法 terminator。

目标：

```text
NormalizedIR 是后续分析可以信任的高层 IR。
```

### 7.3 InferredIR

来源：struct info inference。

要求：

- 所有 op result 都有 `struct_info`。
- function return struct info 已知或可由 return values 推导。
- Tuple、shape、object、scalar 都有正规 struct info。
- standard op schema validate 通过。

禁止：

- 产生 result 但没有 struct info 的 op。
- schema 输入输出数量不匹配。
- dtype/device/shape 缺失但后续 lowering 需要它们的 tensor op。

目标：

```text
InferredIR 是合法化和 kernel selection 的输入。
```

### 7.4 DPSIR

来源：DPS lowering。

要求：

- 需要 kernel lowering 的 high-level tensor op 已经消失。
- tensor 结果通过 `TensorCreateOp + CallDPSOp(outputs=...)` 表达。
- `CallDPSOp` 的 target 是 `KernelRef`、`PackedFuncRef` 或合法 builtin。
- output buffer 的 write effect 明确。

允许：

- shape op。
- control op。
- object/runtime op。
- 不需要 kernel lowering 的 builtin call。

禁止：

- 残留 lowering kind 为 kernel 的 `StandardCallOp`。
- 已经降到 raw storage 的 `AllocStorageOp`，这属于 MemoryIR。

目标：

```text
DPSIR 是 memory planning 的输入。
```

### 7.5 MemoryIR

来源：memory planning 和 tensor create lowering。

要求：

- `TensorCreateOp` 已降为 `AllocStorageOp + AllocTensorOp` 或等价 memory op。
- storage reuse plan 已体现在 allocation 结构中。
- tensor view 和 storage 的关系可验证。
- return-escaping tensor 不和临时 tensor 错误复用 storage。

禁止：

- high-level tensor create。
- high-level kernel-lowering call。
- 无法解释的 storage alias。

目标：

```text
MemoryIR 是 VM codegen 可以消费的低层内存 IR。
```

### 7.6 VMIR

来源：VM codegen。

要求：

- 只保留 VM 可执行 op、runtime call、control op、constant、shape scalar。
- 所有 kernel、packed func、external func 都有 ABI 信息。
- memory allocation、tensor view、device movement 已可映射到 VM instruction。

禁止：

- high-level standard tensor op。
- 未选择 target 的 call。
- 需要编译期推导但仍未推导的信息。

目标：

```text
VMIR 是 executable emitter 的直接输入。
```

## 8. Region 和 Control Flow

devproc2 应继续采用 structured control flow。

目标：

```text
IfOp / ForOp 通过 region + yield 表达值合流，
不在高层 IR 过早引入 CFG、jump、phi。
```

v1 规则：

- 一个 region 只包含一个 block。
- region 必须以 `YieldOp` 结束。
- function body 必须以 `ReturnOp` 结束。
- `IfOp` 有 result 时必须有 `else_region`。
- `IfOp` result 数量必须等于 then/else yield 数量。
- `ForOp` result 数量必须等于 iter args 数量。
- effect-only control flow 的 yield 必须为空。

未来扩展：

- 如果需要异常控制流、early return、复杂 runtime branch，再引入 CFG region。
- CFG region 不能偷偷混进当前 structured region，必须是新的 region kind。

这能避免一个常见问题：类型上允许 multi-block region，但 verifier 和 pass 实际只按 single-block 处理。

## 9. Pass 如何消费 IR 语义

### 9.1 Rewriter

rewriter 应该依赖公共接口：

- `op.operands`
- `op.regions`
- `op.results`
- `op.effects`
- `op.replace_operands(...)`

而不是每个 pass 都维护一份所有 op 类型的字段列表。

短期可以保留 concrete op 分支，但要把它视为过渡实现。新增 op 时，首要任务是补公共接口，而不是让每个 pass 知道它。

### 9.2 InferStructInfo

type inference 应该只依赖：

- op schema 的 infer 函数。
- operand struct info。
- attr dict。
- region yield struct info。

不应该依赖 printer name、callee string 或 pass 顺序里的隐式约定。

### 9.3 DPS Lowering

DPS lowering 的输入是 `InferredIR`，它应该能相信：

- result struct info 已完整。
- op schema 已 validate。
- lowering policy 已知。

输出是 `DPSIR`，它必须保证：

- 被 lowering policy 标记为 kernel 的 high-level op 不再残留。
- output buffer 和 write effect 对齐。
- 未能 lowering 的 op 要么合法保留，要么明确报错。

### 9.4 Memory Planning

memory planning 应该消费公共语义：

- def-use。
- effect reads/writes。
- alias analysis。
- return escape analysis。
- tensor size expression。

它不应该自己特殊处理 Tuple、If、For 的 forwarding 细节。那些应该属于 alias analysis。

### 9.5 VM Codegen

VM codegen 的输入必须足够低层。它不应该再负责：

- 解析 standard op。
- 推导 shape/dtype/device。
- 选择 kernel。
- 决定 tensor 是否能复用 storage。

VM codegen 只做映射：

```text
MemoryIR / RuntimeIR -> VM instruction / executable artifact
```

## 10. Verifier 设计

优雅 IR 必须有强 verifier。

verifier 至少分三层：

### 10.1 Structural verifier

检查：

- block 非空。
- terminator 位置正确。
- use before def。
- result owner/index 正确。
- region parent/owner 合法。
- block arg 不重复。

### 10.2 Semantic verifier

检查：

- op schema。
- attr 类型。
- operand struct info。
- result struct info。
- effect summary 合法。
- control flow yield/result arity。
- alias source 合法。

### 10.3 Stage verifier

检查：

- 当前 stage 允许哪些 dialect。
- 当前 stage 禁止哪些 op/ref kind。
- 当前 stage 必须具备哪些 metadata。
- 当前 stage 不允许残留哪些高层语义。

每个 pass 的基本契约：

```text
verify(input, pass.input_stage)
run pass
verify(output, pass.output_stage)
```

debug 模式下应该默认开启，release 模式可以按配置关闭。

## 11. Textual IR 和 Serialization

textual IR 的目标是可读、可 diff、可 roundtrip，但不是语义源。

规则：

- printer 可以使用 `%x`、`@matmul`、`tensor<...>` 这类友好形式。
- parser 必须把字符串恢复成 `OpRef`、`AttrValue`、`StructInfo`。
- 序列化格式必须保留 value def-use、op ref kind、attrs 类型、effects、regions。
- roundtrip 后 verifier 必须通过。

推荐优先级：

```text
in-memory IR correctness
-> stable JSON serialization
-> textual IR roundtrip
-> pretty printer
```

不要反过来为了打印方便牺牲 IR 内部语义。

## 12. 和当前 IR 的关系

当前 devproc2 IR 已经有一个不错的骨架：

- `Function / Region / Block / Op / Value` 层级合理。
- `IfOp / ForOp / YieldOp` 的 structured control flow 方向正确。
- `StructInfo` 已经能支撑 tensor shape/dtype/device。
- `OpRef` 和 op registry 已经开始把裸字符串语义收回来。
- DPS lowering、memory planning、VM codegen pipeline 已经跑通。

但它还不够优雅，主要问题是：

- stage dialect 集合过宽，stage contract 主要靠 verifier 特判。
- `CallOp` 同时承担 standard、builtin、external 多种语义。
- alias/effect 虽然有类型，但还不是所有 pass 共用的语义基础。
- memory planning 里仍然有局部 alias forwarding 逻辑。
- concrete op 缺少统一 operand/region/effect 接口，rewriter 需要手写字段分支。

这份文档不是否定当前实现，而是给出一个长期目标：

```text
保留当前能跑的主干，
逐步把隐式约定提升成显式 IR contract。
```

## 13. 迁移路线

### Phase 1：收紧 verifier

目标：

- 把 stage contract 写实。
- 拒绝不合法 stage 残留。
- 检查 result owner/index。
- 检查 control flow result/yield arity。
- 检查 InferredIR result struct info。

收益：

- 不改变大量 API。
- 立刻提高 pass 边界可信度。
- 让错误更早暴露。

### Phase 2：统一 op 公共接口

目标：

- 给 concrete op 增加公共 `operands`、`regions`、`effects` 访问。
- rewriter 和 analysis 优先消费公共接口。
- 新增 op 时只要实现公共接口，不需要修改每个 pass。

收益：

- 降低 pass 与具体 op class 的耦合。
- 为后续拆 `CallOp` 铺路。

### Phase 3：拆清 CallOp 语义

目标：

- 引入或显式区分 `StandardCallOp`、`BuiltinCallOp`、`ExternalCallOp`。
- `CallDPSOp` 只保留 destination-passing runtime/kernel 语义。
- verifier 按 ref kind 和 stage 做合法性检查。

收益：

- high-level op、runtime builtin、external call 边界变硬。
- lowering pass 不再靠 `op_ref` 类型和 lowering kind 特判所有场景。

### Phase 4：公共 alias/effect 分析

目标：

- 把 Tuple、TupleGetItem、If result、For result 的 forwarding 统一进 alias analysis。
- memory planning、DCE、CSE、in-place lowering 共享 alias/effect 结果。
- external call 和 opaque effect 形成统一 barrier 语义。

收益：

- memory reuse 更安全。
- 优化 pass 更容易组合。
- 控制流和 tuple 不再是每个 pass 的局部问题。

### Phase 5：稳定序列化和 textual IR

目标：

- JSON roundtrip 保留 op ref kind、attrs type、effects、struct info、regions。
- textual IR 可读、可 diff。
- parser 恢复后 verifier 通过。

收益：

- 支撑测试 fixture。
- 支撑 artifact inspection。
- 支撑未来跨进程编译和缓存。

## 14. 非目标

这份设计不主张立即做这些事：

- 不立即实现完整 MLIR-style generic operation system。
- 不立即支持 arbitrary CFG。
- 不把所有 pass 一次性重写。
- 不把 Python dataclass IR 全部替换掉。
- 不为了形式统一牺牲当前 MVP pipeline。

正确节奏是：

```text
先让 contract 变硬，
再让公共接口变稳，
最后把已有 pass 迁移到公共语义服务。
```

## 15. 最终形态

我认可的 devproc2 IR 最终应该有这样的气质：

```text
看起来仍然轻量，
但每个边界都有 verifier 守住。

写 pass 时像在消费语义模型，
而不是在猜历史约定。

新增 op 时主要改 registry 和 schema，
而不是到处补 if/elif。

从 high-level program 到 VM executable 的每一步，
都有明确输入、明确输出、明确禁止项。
```

这种 IR 不一定最学院派，但它会非常适合 devproc2：小系统能快速推进，大系统又不会在隐式规则里失控。
