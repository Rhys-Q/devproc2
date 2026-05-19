# devproc2 IR 重构计划：从 MVP IR 到可优化 IR

## 1. 目标

当前 devproc2 IR 已经能支撑 Python DSL、结构化控制流、动态 shape、DPS lowering、kernel selection、memory planning、VM codegen 和 artifact emit。它是一个可工作的 MVP IR，但还不是一个足够优雅、稳定、适合长期优化的 IR。

本轮 IR 重构的目标是在保留当前能力的基础上，把 devproc2 IR 重构成一个更像 TVM Relax / MLIR 风格的优化 IR：

- 新增标准 op 只需要注册 schema、attrs、infer、validate、lowering policy。
- pass 不依赖裸字符串和 ad hoc 分支。
- type / shape / device / effect / memory contract 在 IR 层可验证。
- high-level tensor op、builtin op、memory op、runtime external call 边界清楚。
- frontend API 可以演进，但 IR 本身保持规范、可序列化、可优化。

一句话目标：

```text
devproc2 IR 应该从“能跑的 Python dataclass IR”
演进为“op/schema/type/effect/memory 都有一等建模的优化 IR”。
```

## 2. 当前状态判断

### 2.1 已经做对的部分

- `Function / Region / Block / Op / Value` 的基本层级是合理的。
- `IfOp / ForOp / YieldOp` 的结构化控制流适合 Python DSL，不需要过早降成 CFG。
- `TensorStructInfo(shape, dtype, device)` 已经支撑了当前 shape infer、DPS lowering、memory planning。
- `PrimExpr` 支持动态 shape，并且已经贯穿到 allocation 和 VM shape lowering。
- `CallOp -> TensorCreateOp + CallDPSOp` 这条高层 op 到 destination-passing kernel 的 lowering 线清楚。
- `compiler/op` registry 已经开始吸收 TVM Relax 的设计，把标准 op 的 schema、attrs、infer 集中管理。

### 2.2 当前主要问题

#### 裸字符串过多

当前 IR 里很多关键语义仍然靠字符串表达：

```python
CallOp(callee="@matmul", ...)
CallDPSOp(callee="kernel.relu_fp16", ...)
callee_kind=CalleeKind.kernel
```

这会导致：

- pass 容易用 `lstrip("@")` 这种脆弱逻辑；
- 标准 op、builtin、external、kernel call 边界不够硬；
- typo 只能到 verifier 或更晚才暴露；
- 序列化和反序列化后语义恢复不够稳。

#### `CallOp` 半对象半字符串

当前 `CallOp` 同时持有：

- `callee: str`
- `op: OpDef | None`
- `call_kind`

这是从字符串 IR 迁移到 registry IR 的过渡形态。短期可用，长期不够干净。

#### attrs 仍然是 `Mapping[str, object]`

虽然 schema 已经能做类型校验，但 attrs 本身没有强类型：

```python
attrs: Mapping[str, object]
```

问题：

- JSON roundtrip 不够严格；
- `tuple/list/int/float/None` 的稳定打印和规范化容易散落；
- dtype、shape、axis、layout、device 这类语义值没有专门类型；
- attr 和 runtime `Constant` 的边界不够清楚。

#### StructInfo 表达力不够

当前主要有：

- `TensorStructInfo`
- `ScalarStructInfo`
- `ObjectStructInfo`

缺少：

- `TupleStructInfo`
- `ShapeStructInfo`
- `FuncStructInfo`
- unknown rank / unknown dtype / symbolic shape handle 的正规表达

这会限制 tuple-output op、shape builtin、runtime packed function、KVCache/object API 的类型推导能力。

#### Memory / builtin / standard op 不是同一套 dialect

当前：

- 标准 tensor op 在 `compiler/op` registry。
- `TensorCreateOp / AllocStorageOp / AllocTensorOp` 是独立 dataclass。
- `ShapeAssertOp` 是独立 op。
- runtime packed call 走 `CallDPSOp`。

这些可以工作，但边界还不清晰。长期应该明确分成几个 IR dialect：

```text
tensor dialect    : matmul, add, permute_dims, layer_norm
shape dialect     : shape_of, get_shape_dim, assert_shape
memory dialect    : alloc_storage, alloc_tensor, view
runtime dialect   : call_packed, call_builtin, external_object
control dialect   : if, for, yield, return
```

#### Effect / alias 系统太粗

当前有：

- `PureEffect`
- `ReadOnlyEffect`
- `WriteEffect`
- `OpaqueEffect`

但缺少：

- alias set；
- in-place buffer contract；
- noalias / readonly / writeonly；
- effect 对 memory planning 的形式化约束；
- external call 的 side-effect summary。

这会限制未来的 buffer reuse、in-place lowering、CSE、fusion 和 scheduling。

## 3. 目标 IR 形态

### 3.1 Value 与 Op 的目标模型

长期目标：

```text
IRModule
  └─ Function
      └─ Region
          └─ Block(args: BlockArg[])
              └─ Operation[]

Value
  ├─ BlockArg
  └─ OpResult

Operation
  ├─ op_ref: OpRef
  ├─ operands: tuple[Value, ...]
  ├─ results: tuple[OpResult, ...]
  ├─ attrs: AttrDict
  ├─ regions: tuple[Region, ...]
  └─ effects: EffectSummary
```

`CallOp` 不应该长期承担所有调用形态。建议逐步收敛成：

```text
StandardCallOp(op_ref=StandardOpRef, operands, attrs)
BuiltinCallOp(op_ref=BuiltinOpRef, operands, attrs)
ExternalCallOp(func_ref=ExternalFuncRef, operands, attrs/effects)
CallDPSOp(kernel_ref=KernelRef | PackedFuncRef, inputs, outputs, attrs/effects)
```

短期可以不立刻拆类，但必须引入一等 `OpRef`。

### 3.2 OpRef

新增引用类型：

```python
class OpRef: ...

@dataclass(frozen=True)
class StandardOpRef(OpRef):
    name: str
    op_def: OpDef

@dataclass(frozen=True)
class BuiltinOpRef(OpRef):
    name: str
    op_def: OpDef

@dataclass(frozen=True)
class ExternalFuncRef(OpRef):
    name: str
    kind: ExternalKind

@dataclass(frozen=True)
class KernelRef(OpRef):
    name: str
    spec: KernelSpec | None
```

规则：

- High-level tensor op 必须是 `StandardOpRef`。
- Runtime builtin 必须是 `BuiltinOpRef`。
- 用户 external call 必须显式是 `ExternalFuncRef`。
- Lowered kernel call 必须是 `KernelRef` 或 `PackedFuncRef`。
- printer 可以继续打印 `@matmul`，但 IR 内部不要靠字符串判断语义。

### 3.3 AttrValue

新增强类型 attr 系统：

```text
AttrValue
  ├─ IntAttr
  ├─ FloatAttr
  ├─ BoolAttr
  ├─ StringAttr
  ├─ DTypeAttr
  ├─ DeviceAttr
  ├─ ShapeAttr
  ├─ PrimExprAttr
  ├─ ArrayAttr
  ├─ DictAttr
  └─ NoneAttr
```

约束：

- attr 必须是编译期常量。
- attr 不允许引用 SSA `Value`。
- runtime-varying 数据必须作为 operand。
- printer、JSON、schema normalize 必须共用同一套 AttrValue。

目标接口：

```python
@dataclass(frozen=True)
class AttrDef:
    name: str
    type: AttrType
    default: AttrValue | None
    required: bool = False

@dataclass(frozen=True)
class AttrDict:
    values: Mapping[str, AttrValue]
```

### 3.4 StructInfo

扩展为：

```text
StructInfo
  ├─ TensorStructInfo(shape, dtype, device)
  ├─ ScalarStructInfo(dtype)
  ├─ ShapeStructInfo(ndim | values)
  ├─ TupleStructInfo(fields)
  ├─ ObjectStructInfo(type_key, role)
  └─ FuncStructInfo(params, ret, purity/effects)
```

Tensor shape 支持：

```text
KnownShape(values: tuple[PrimExpr, ...])
UnknownShape(ndim: int | None)
```

不要用 `None` 混用未知、缺失、推导失败三种状态。

### 3.5 OpDef

标准 op 定义应包含：

```python
@dataclass(frozen=True)
class OpDef:
    name: str
    inputs: tuple[InputDef, ...]
    attrs: tuple[AttrDef, ...]
    outputs: tuple[OutputDef, ...]
    infer: InferStructInfoFn
    normalize: NormalizeFn | None
    validate: ValidateFn | None
    purity: PurityKind
    pattern: OpPatternKind
    dialect: DialectKind
    lowering: LoweringPolicy
```

新增 op 的验收标准：

- schema 明确；
- attrs 强类型；
- infer 不 silent fallback；
- validate 覆盖 shape / dtype / device 约束；
- pattern 正确；
- lowering policy 明确；
- 至少有合法和非法用例测试。

## 4. 重构阶段

### Phase 0: 冻结当前行为

目标：确保重构过程中不丢当前能力。

任务：

- 保持现有全量测试 green。
- 增加 IR golden/snapshot 测试，覆盖：
  - standard call；
  - external call；
  - control flow；
  - dynamic shape；
  - DPS lowering；
  - memory planning；
  - VM codegen。
- 明确当前公开 API：
  - `dp.function`
  - `dp.Tensor`
  - `dp.empty`
  - `dp.call_dps_packed`
  - `dp.relu/add/matmul/permute_dims/layer_norm/rms_norm`
  - `nn.Module / nn.Linear / nn.LayerNorm / nn.RMSNorm`

验收：

- 当前 tests 全部通过；
- 文档列出哪些 API 是稳定的，哪些是兼容层。

### Phase 1: 引入 OpRef，消除 pass 中裸字符串依赖

目标：让 IR 内部按对象语义区分 standard/builtin/external/kernel。

任务：

- 新增 `OpRef` 类型。
- `CallOp` 构造时将 `callee` 解析为 `op_ref`。
- `KernelSelectPass` 使用 `call.op_ref.name`。
- `DPSLoweringPass` 使用 `op_ref` 和 lowering policy。
- `VMCodegenPass` / `EmitABIPass` 不再靠字符串推断 callee kind。
- printer 继续输出当前文本格式，避免大规模 snapshot 抖动。

迁移策略：

- 保留 `callee` 作为 property 或打印辅助字段。
- 禁止新 pass 读取 `callee.lstrip("@")`。

验收：

- `rg "lstrip\\(\"@\"\\)" python/devproc2` 不应出现在 pass 逻辑里。
- unknown standard op 在 verifier 早期报错。
- external call 必须显式标记。

### Phase 2: 强类型 AttrValue

目标：让 attr 稳定、可序列化、可校验。

任务：

- 新增 `python/devproc2/ir/attrs.py`。
- 实现 `AttrValue` 层级。
- `OpDef.normalize_attrs()` 返回 `AttrDict`，而不是普通 dict。
- printer 和 JSON serializer 使用统一 attr formatter。
- schema type 从字符串演进为 `AttrType`。

迁移策略：

- 短期允许 Python API 传普通 `int/float/tuple/None`。
- 在 op emit 边界 normalize 成 `AttrValue`。
- IR 内部只保存 `AttrValue`。

验收：

- attr JSON roundtrip 测试。
- attr printer snapshot 稳定。
- unknown attr / wrong type / missing required attr 报错清晰。

### Phase 3: StructInfo 扩展

目标：支持 tuple/shape/function/object 的系统化推导。

任务：

- 新增 `TupleStructInfo`。
- 新增 `ShapeStructInfo`。
- 新增 `FuncStructInfo`。
- Tensor shape 引入 `KnownShape / UnknownShape`。
- 更新 `InferStructInfoPass`，去掉 `None` 表示 unknown 的混乱用法。
- Tuple op、TupleGetItem op 使用 `TupleStructInfo`。
- shape builtin 使用 `ShapeStructInfo`。

验收：

- tuple-producing op 可以正确 infer。
- shape_of / get_shape_dim 等 builtin 不再用 ad hoc object/scalar。
- infer 失败和 unknown 可以区分。

### Phase 4: Dialect 化 builtin / memory / runtime op

目标：清楚区分 high-level tensor IR 和 lowered/runtime IR。

任务：

- 定义 dialect：
  - `tensor`
  - `shape`
  - `memory`
  - `runtime`
  - `control`
- `TensorCreateOp` 移入 memory/tensor-create dialect。
- `AllocStorageOp / AllocTensorOp` 移入 memory dialect。
- `ShapeAssertOp` 移入 shape dialect。
- `call_dps_packed` 变成明确 runtime op，不和 standard tensor op 混用。
- verifier 按 pipeline stage 检查允许出现的 dialect。

示例：

```text
HighLevelIR:
  allowed = tensor + shape + control + external

DPSIR:
  allowed = tensor_create + call_dps + shape + control

MemoryIR:
  allowed = alloc_storage + alloc_tensor + call_dps + shape + control

VMIR:
  high-level tensor op should not exist
```

验收：

- 每个 pass 声明输入/输出 IR stage。
- verifier 可以按 stage 检查非法 op。

### Phase 5: Effect 与 alias 系统

目标：为 memory reuse、in-place lowering、external runtime call 提供硬约束。

任务：

- 定义 `EffectSummary`：
  - read values；
  - write values；
  - allocate；
  - free；
  - opaque；
  - external state token。
- 定义 alias set：
  - `NoAlias`
  - `MayAlias`
  - `MustAlias`
  - `ViewOf`
- `CallDPSOp` outputs 从单个 `output` 扩展为 tuple。
- external packed call 必须声明 effect。
- MemoryPlanningPass 读 effect/alias，不再只做保守 opaque。

验收：

- pure op 可 CSE。
- write effect 会延长 live range。
- in-place lowering 必须通过 alias/effect verifier。

### Phase 6: 标准 op 库清理

目标：op 集合对齐 TVM Relax / PyTorch 常见语义。

当前规范：

- `permute_dims(x, axes=None)`，不要 `transpose(dim0, dim1)`。
- `matmul(a, b, out_dtype=None)`。
- `layer_norm(x, weight, bias, axes=(-1,), epsilon=1e-6, center=True, scale=True)`。
- `rms_norm(x, weight, axes=(-1,), epsilon=1e-6)`。
- `adarms_norm(x, weight, cond, axes=(-1,), epsilon=1e-6)`。
- `embedding(indices, weight, padding_idx=None)`，不要 `scale` attr；scale 用 `multiply`。
- `gelu(x, approximate="none")`，需要 tanh 时显式 `"tanh"`。
- binary op 使用 `add/multiply/subtract/divide/...`，支持 broadcast infer。

后续新增：

- `reshape`
- `expand_dims`
- `squeeze`
- `concat`
- `split`
- `take/gather`
- `where`
- `softmax`
- `astype`
- `full/zeros/ones`

验收：

- 每个标准 op 有 schema、infer、validate、测试。
- 不再引入组合型 convenience op，例如 `linear`、`mlp`、`attention_block`。
- 融合只在 lowering/fusion 层表达，不污染 high-level IR。

### Phase 7: Pass pipeline 正规化

目标：每个 pass 输入输出清楚、可组合、可调试。

任务：

- 定义 IR stage：
  - `RawIR`
  - `NormalizedIR`
  - `InferredIR`
  - `DPSIR`
  - `MemoryIR`
  - `VMIR`
- 每个 pass 声明：
  - input stage；
  - output stage；
  - required analysis；
  - preserved analysis。
- `PassContext` 里 analysis key 规范化。
- pass 后自动 verifier 可选开启。

验收：

- 错误 pass 顺序能早期报错。
- pipeline 可以打印 stage-by-stage IR。

## 5. 兼容策略

### 5.1 不破坏当前能力

必须保留：

- Python DSL function capture；
- structured control flow；
- dynamic shape；
- kernel registry；
- DPS lowering；
- memory planning；
- VM codegen；
- artifact emit；
- C++ runtime tests。

### 5.2 允许清理历史兼容 API

允许删除：

- 非标准 op alias，例如旧 `layernorm`。
- 只为早期 MVP 服务的 op 名。
- `dim0/dim1` 这种不标准 attr。

但删除前必须：

- 替换所有 tests；
- 确认 public API 文档；
- 保留必要 migration note。

### 5.3 Python API 与 IR API 分离

Python frontend 可以提供 convenience API，但 IR op 必须标准。

例如：

```python
nn.Linear.forward(x)
```

可以展开成：

```text
permute_dims(weight, axes=(1, 0))
matmul(x, weight_t)
add(y, bias)
```

但 IR 不应该有 `linear` op。

## 6. 近期落地顺序

建议按以下顺序执行：

1. Phase 1：引入 `OpRef`，清掉 pass 中裸字符串判断。
2. Phase 2：强类型 attrs。
3. Phase 3：补 `TupleStructInfo / ShapeStructInfo`。
4. Phase 4：按 stage verifier 区分 high-level / lowered / memory IR。
5. Phase 6：继续扩标准 op 库。
6. Phase 5：effect/alias 系统，支撑更激进 memory 和 in-place 优化。
7. Phase 7：正规 pass pipeline。

原因：

- `OpRef` 是后续所有清理的地基。
- Attr 和 StructInfo 是 op infer/validate 的地基。
- Dialect/stage verifier 能让后续 lowering 更安全。
- Effect/alias 改动最大，应该等 memory/lowering 边界更稳后做。

## 7. 验收标准

最终 devproc2 IR 达到以下状态，才算完成本轮重构：

- 新增标准 op 不需要修改 pass，只需要注册 op schema/infer/lowering policy。
- pass 不直接依赖 callee 字符串。
- attrs 是强类型、可 JSON roundtrip、可稳定打印。
- StructInfo 能表达 tensor、tuple、shape、object、function。
- verifier 能按 IR stage 检查非法 op。
- DPS lowering 和 memory planning 依赖 effect/alias contract，而不是 ad hoc 保守逻辑。
- high-level IR 中没有 convenience/composite op。
- 全量测试保持通过。

## 8. 非目标

本轮 IR 重构不直接实现：

- 自动融合算法；
- 具体高性能 kernel；
- CUDA Graph capture；
- distributed/sharding IR；
- complete ONNX/PyTorch import；
- 完整量化类型系统。

这些能力应该建立在重构后的 IR 基础上，而不是在当前 MVP IR 上继续堆 ad hoc 逻辑。

