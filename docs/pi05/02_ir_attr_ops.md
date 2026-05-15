# IR Attr 与 Op 设计

## 目标

当前 `CallOp` 只有 `callee`、`args` 和 `result_struct_info`。openpi0.5 所需 op 必须携带语义参数，例如：

- `layer_norm.eps`
- `gelu.approximate`
- `cat.axis`
- `transpose.dim0/dim1`
- `matmul.transpose_a/transpose_b`
- `rope.rotary_dim/base`、`softmax.axis` 等标准 op attrs
- `rope.base/rotary_dim`

因此需要在 IR 层引入稳定、可打印、可序列化、可校验的 attr 系统。

## AttrValue 数据模型

建议最小集合：

```python
AttrValue =
    IntAttr
  | FloatAttr
  | BoolAttr
  | StringAttr
  | DTypeAttr
  | ShapeAttr
  | ArrayAttr
  | DictAttr
```

原则：

- attr 必须是编译期常量。
- attr 不引用 SSA `Value`。运行时变化的数据必须作为 op input。
- attr 可序列化为 JSON，用于 metadata 和调试。
- printer 输出必须稳定，便于 snapshot test。

示例 IR 打印：

```text
%wt = call @transpose(%w) {dim0=0, dim1=1}
%y0 = call @matmul(%x, %wt) {out_dtype="float16"}
%y = call @add(%y0, %b)
%z = call @layer_norm(%y, %gamma, %beta) {eps=1e-6}
%h = call @gelu(%z) {approximate="tanh"}
```

## IR 节点变更

建议扩展：

```python
@dataclass(frozen=True, eq=False)
class CallOp(Op):
    callee: str
    args: tuple[Value, ...]
    result_name: str = ""
    result_struct_info: Optional[StructInfo] = None
    attrs: Mapping[str, AttrValue] = field(default_factory=dict)

@dataclass(frozen=True, eq=False)
class CallDPSOp(Op):
    callee: str
    callee_kind: CalleeKind
    inputs: tuple[Value, ...]
    output: Optional[Value]
    effect: EffectInfo
    attrs: Mapping[str, AttrValue] = field(default_factory=dict)
```

lowering 规则：

- high-level `CallOp.attrs` 必须原样传递到 `CallDPSOp.attrs`。
- kernel selection 可以读取 attrs。
- VM codegen 不直接解释 attrs；需要 kernel table 或 metadata 把 attrs 编译进 kernel launch/config，或作为额外常量参数传给 kernel。

## op 命名约定

- 使用小写 snake_case：`layer_norm`、`rms_norm`、`gelu`。
- 语义 op 不包含 dtype 后缀；dtype 由 input/output struct info 决定。
- kernel 名可以包含 dtype/layout 后缀：`kernel.matmul_fp16_rowmajor`。

## 标准 op 原则

IR 中的 tensor op 必须是标准、原子的张量算子，不能为了模型前端方便把组合型算子包装成新 op。否则会造成 op 集膨胀、shape infer 分裂、kernel selection 混乱，并削弱后续 pattern fusion 的可控性。

明确规则：

- `Linear` 不是 IR op，只是 nn 前端模块。`Linear.forward` 必须展开为 `transpose + matmul + add(optional)`。
- MLP、attention block、QKV projection、gated residual 都不是 IR op，只能由标准 op 组合表达。
- 允许在 lowering/kernel 层识别标准 op pattern 并选择 fused kernel，但 high-level IR 仍保留标准 op 语义。
- 新增 IR op 前必须说明它无法由已有标准 op 清晰表达，并提供 op schema、attrs、infer shape 和 lowering 规则。

## Infer Shape 规则

每个 tensor-producing op 必须注册 infer shape/struct info 函数。当前 devproc2 的 `InferStructInfoPass` 仍是 MVP：`TensorCreateOp` 可直接生成 `TensorStructInfo`，普通 `CallOp` 主要从第一个参数传播，无法正确覆盖 `matmul`、`cat`、`reshape`、`embedding` 等结果 shape 变化的 op。

后续要求：

- 每个标准 tensor op 都必须有 op schema 和 infer 函数。
- infer 输入只能是 input `StructInfo` 与 attrs，不能读取 runtime tensor data。
- infer 输出必须包含 shape、dtype、device。
- infer 失败必须在编译期报错，例如 matmul K 维不匹配、cat 非 axis 维不一致、reshape 元素数量不一致。
- 没有 infer 函数的 tensor op 不允许进入 DPS lowering 和 kernel selection。

## 首批基础 op

### embedding

输入：

- `indices`: int32/int64 tensor，shape `[...,]`
- `weight`: weight tensor，shape `[vocab, hidden]`

attrs：

- `padding_idx: int | None`
- `scale: float`，openpi 语言 embedding 后会乘 `sqrt(hidden_dim)`，可单独作为 `mul`，也可融合进 embedding kernel

输出 shape：`indices.shape + (hidden,)`

### matmul

输入：

- `a`: `[..., M, K]` 或 `[M, K]`
- `b`: `[..., K, N]` 或 `[K, N]`

attrs：

- `out_dtype: dtype | None`
- `transpose_a: bool = false`
- `transpose_b: bool = false`

输出：broadcast batch dims 后的 `[..., M, N]`。

PyTorch `nn.Linear(x, weight, bias)` 必须由前端展开为：

1. `transpose(weight)`，把 `[out_features, in_features]` 变为 `[in_features, out_features]`。
2. `matmul(x, weight_t)`。
3. 如果存在 bias，执行 `add(matmul_out, bias)`，由 add 的 broadcast infer 处理。

IR op 集中不引入 `linear`。

### layer_norm

输入：

- `x`
- `weight`
- `bias`

attrs：

- `eps: float`
- `normalized_shape: list[int]`

输出 shape 与 input 相同。

### rms_norm

输入：

- `x`
- `weight`
- 可选 `cond`，用于 adaRMS

attrs：

- `eps: float`
- `use_adarms: bool`

输出：

- 普通 RMSNorm：`y`
- adaRMS：建议第一阶段拆成 `adarms_norm` op，输出 `y, gate`；或者用两个 op `rms_norm` 与 `adarms_gate`，避免 tuple output 在 lowering 中复杂化。

### gelu

attrs：

- `approximate: "none" | "tanh"`

openpi0.5 使用 transformers Gemma 的 `gelu_pytorch_tanh`，应固定为 `approximate="tanh"`。

### silu

无必需 attrs。

### shape/layout ops

需要：

- `reshape`
- `transpose`
- `permute`
- `slice`
- `gather`
- `cat`
- `expand`
- `where`
- `mask_fill`
- `cumsum`

这些 op 可以先以 kernel 或 builtin 实现，后续再融合。

## attention 相关 op

attention block 不能作为一个高层 IR op。它必须由标准 op 组合表达，后续可以在 lowering 阶段识别 pattern 并选择 fused attention kernel。

1. `matmul + add` 得到 q/k/v。
2. `reshape + transpose` 到 `[B, H, S, D]`。
3. `rope(q, k, position_ids)`。
4. `matmul(q, transpose(k))` 得到 attention scores。
5. `mul(scores, scaling)`。
6. `add/mask_fill(scores, mask)`。
7. `softmax(scores)`。
8. `matmul(prob, v)`。
9. `transpose + reshape`。
10. `matmul + add` output projection。

KV cache：

- prefix 阶段生成 `past_key_values`。
- denoise 阶段读取 prefix KV，并拼接 suffix KV 做 attention。
- IR 中推荐用 tuple/list-like value 表达 `past_key_values`，lowering 前可展开为每层每个 key/value tensor。

## verifier 规则

- attr key 必须属于 op schema。
- attr 类型必须匹配 schema。
- required attrs 必须存在。
- shape infer 只能依赖 input struct info 和 attrs，不能依赖 runtime data。
- 每个 tensor op 必须有 infer shape/struct info 函数；缺失 infer 的 op 在 verifier 中报错。
- weight input 必须是 `Parameter` 或已绑定 weight tensor，不允许误用普通 activation。

## 测试策略

- AttrValue JSON round-trip。
- printer snapshot。
- 每个 op schema 的合法/非法 attr 测试。
- `CallOp -> CallDPSOp` lowering 后 attrs 不丢失。
- `matmul/add/layer_norm/cat/reshape/embedding` 的 shape infer 测试。
