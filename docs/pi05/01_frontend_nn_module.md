# nn.Module 前端设计

## 目标

devproc2 当前 DSL 更接近函数式写法：

```python
@dp.function
def f(x):
    return dp.ops.relu(x)
```

openpi0.5 的 PyTorch 实现是典型 `torch.nn.Module` 结构，包含嵌套子模块、参数、少量持久常量 tensor、`forward`、模块复用和权重命名路径。需要新增一层 `devproc2.nn` 前端，使模型可以用接近 torch 的方式描述，同时底层仍生成 devproc2 IR。

## 非目标

- 不执行 PyTorch eager 计算。
- 不直接复用 `torch.nn.Module` 对象作为 runtime。
- 不在本阶段实现 autograd、training、optimizer。
- 不在前端处理量化 kernel；只允许参数 metadata 表达量化后的权重形态。

## API 草案

```python
import devproc2 as dp
import devproc2.nn as nn

class Pi05DenoiseStep(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_in_proj = nn.Linear(config.action_dim, config.width, bias=True)
        self.time_mlp_in = nn.Linear(config.width, config.width, bias=True)
        self.time_mlp_out = nn.Linear(config.width, config.width, bias=True)
        self.action_out_proj = nn.Linear(config.width, config.action_dim, bias=True)

    def forward(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_masks, suffix_att, cond = self.embed_suffix(state, x_t, timestep)
        hidden = self.expert(..., suffix_embs, cond)
        return self.action_out_proj(hidden)
```

`Module.__call__` 负责调用 `forward`，并在 tracing/build 阶段把每个 nn module call 转成 IR op 或 IR function call。

## 核心对象

### Module

职责：

- 拦截 `__setattr__`，注册子模块和参数。
- 提供 `named_modules()`、`named_parameters()`、`state_dict()`。
- 维护当前模块路径，例如 `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj`。
- 在 build 阶段提供 stable parameter path。

约束：

- 子模块顺序按赋值顺序稳定。
- list/tuple 子模块需要通过 `ModuleList` 表达。
- 不支持运行时动态新增 parameter。

### Parameter

职责：

- 表示一个待绑定权重的符号参数。
- 记录 shape、dtype、device、layout、role。
- 在 IR 中表现为 weight handle，而不是普通 SSA 中间值。

草案：

```python
@dataclass(frozen=True)
class Parameter:
    shape: tuple[PrimExpr, ...]
    dtype: str
    device: str = "cuda"
    layout: str = "row_major"
    role: Literal["weight", "constant_tensor"] = "weight"
    name: str | None = None
```

推理框架中不单独引入 PyTorch 式 Buffer。所有需要进入 artifact、由 C++ runtime 持久加载的 tensor 都用 `Parameter` 表达，通过 `role` 或 weight metadata 区分：

- `role="weight"`：传统权重和 bias。
- `role="constant_tensor"`：rotary `inv_freq`、position embedding、预计算表等持久 tensor。

真正的小常量不应该成为 `Parameter`，而应使用 IR attr 或 VM const，例如 `eps`、`axis`、`num_heads`、`sqrt(hidden_dim)`、固定 shape/list/string enum。

## 基础 nn modules

首批需要覆盖：

- `Embedding(num_embeddings, embedding_dim, dtype, device)`
- `Linear(in_features, out_features, bias=True, dtype, device)`
- `LayerNorm(normalized_shape, eps=1e-6, elementwise_affine=True)`
- `RMSNorm(hidden_size, eps=1e-6, use_adarms=False)`
- `GELU(approximate="tanh")`
- `SiLU()`
- `ModuleList`
- `Sequential`，仅作为简单容器，不作为 openpi0.5 必需接口

模块 forward 发出的 IR：

| Module | IR op |
|---|---|
| `Embedding` | `embedding(input, weight, attrs={padding_idx, scale})` |
| `Linear` | `matmul(input, transpose(weight))`，如果有 bias 再接 `add` |
| `LayerNorm` | `layer_norm(input, weight, bias, attrs={eps})` |
| `RMSNorm` | `rms_norm(input, weight, cond?, attrs={eps, use_adarms})` |
| `GELU` | `gelu(input, attrs={approximate: "tanh"})` |
| `SiLU` | `silu(input)` |

`Linear` 是前端组合模块，不是 IR op。它的 forward 必须展开为标准 op：

```text
%wt = call @transpose(%weight) {dim0=0, dim1=1}
%y0 = call @matmul(%x, %wt)
%y = call @add(%y0, %bias)
```

如果 kernel 层后续要做 fused linear，也只能作为 `matmul(+add)` pattern 的 lowering 优化，不能把 `linear` 引入高层 IR op 集。

## openpi0.5 模型映射

openpi0.5 PyTorch 结构中需要表达：

- PaliGemma vision tower 和 language model。
- Gemma expert model。
- action projection：`action_in_proj`、`action_out_proj`。
- pi0.5 专用 time MLP：`time_mlp_in`、`time_mlp_out`。
- adaRMS 条件路径：`use_adarms=[False, True]`。

实施时优先级：

1. 先表达 Gemma expert denoise path，不包含 vision tower。
2. 再表达 PaliGemma language prefix path 和 KV cache。
3. 最后接入 SigLIP/PaliGemma vision tower。

## tracing/build 规则

模型 build 入口：

```python
model = Pi05Model(config)
builder = dp.nn.GraphBuilder()
ir_mod = builder.build(model.sample_actions, input_specs)
```

规则：

- `TensorSpec` 描述输入 shape/dtype/device。
- `Parameter` 作为特殊 IR value 进入 op inputs。
- Python 常量、module attrs 和 op attrs 分离：模型结构常量进入 attrs，运行时 tensor 仍作为 SSA value。
- `for layer in self.layers` 允许静态展开；`while time >= ...` 对固定 `num_steps=10` 先展开为静态 10 次，后续再考虑动态 loop。

## 与现有 DSL 的关系

`devproc2.nn` 不替代现有 `@dp.function`。推荐分层：

- `@dp.function` 继续作为低层函数式 DSL。
- `devproc2.nn` 作为模型结构 DSL，最终调用现有 builder 能力生成 `IRModule`。
- 基础 nn module 的 forward 只发出 `dp.ops.*`，不绕过 IR。

## 验收标准

- 能构建包含 `Linear -> SiLU -> Linear` 的嵌套 Module，并生成稳定参数名。
- `named_parameters()` 与权重映射文档中的 path 规则一致。
- 同一个模块多次调用不会重复创建 parameter。
- `ModuleList` 中 layer path 使用稳定数字索引，例如 `layers.0.self_attn.q_proj.weight`。
