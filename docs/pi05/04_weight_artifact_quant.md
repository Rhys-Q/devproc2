# 权重映射、Artifact 与量化预留

## 目标

openpi0.5 的原始权重来自 HuggingFace/PyTorch safetensors，但 devproc2 的编译和运行阶段不应该直接消费 HuggingFace 权重。需要在编译前增加独立的 `convert_weight` 环节，将外部 checkpoint 转换为 devproc2 自有权重格式并保存到本地。后续 compile、artifact emit、C++ runtime 都只依赖 devproc2 权重包。

同时，本设计需要为后续外部量化框架输出的权重预留 metadata。量化本身不在本次任务范围内。

## 权重生命周期

整体流程分为两个明确阶段：

```text
HuggingFace / PyTorch checkpoint
  └── convert_weight
        └── devproc2 weight package
              ├── weights.bin
              ├── weights.index.json
              └── weight_map.json

devproc2 model source + devproc2 weight package
  └── compile
        └── devproc2 artifact
              ├── executable.vm
              ├── abi.json
              ├── kernels/
              ├── weights/              # copied or linked from devproc2 weight package
              └── metadata/
```

原则：

- `convert_weight` 是编译前的显式步骤，不是 compile pass 的隐式副作用。
- compile 只验证和绑定 devproc2 weight package，不读取原始 safetensors。
- runtime 只加载 artifact 中的 devproc2 权重，不知道 HuggingFace key、checkpoint 路径或 PyTorch 命名细节。
- HuggingFace source metadata 只保留在转换报告中用于审计和 debug，不参与 runtime ABI。

## WeightSpec

建议定义：

```python
@dataclass(frozen=True)
class WeightSpec:
    name: str
    source_key: str | None
    kind: Literal["weight", "constant_tensor"]
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    transform: str | None = None
    tied_to: str | None = None
    quant: QuantSpec | None = None
```

字段说明：

- `name`：devproc2 模型内稳定路径，例如 `action_out_proj.weight`。
- `source_key`：原始 safetensors 中的 key，仅在 `convert_weight` 输出的转换报告中使用；compile/runtime 不能依赖该字段。
- `kind`：`weight` 表示传统权重/bias；`constant_tensor` 表示推理需要持久加载的 tensor 常量，例如 rotary `inv_freq` 或 position embedding。
- `shape/dtype/layout`：devproc2 runtime 看到的最终形态。
- `transform`：加载时的布局转换，例如 `transpose`、`permute_qkv`。
- `tied_to`：共享权重，例如 PaliGemma language embedding 与 lm head。
- `quant`：量化 metadata，首版为 None。

## devproc2 权重包格式

`convert_weight` 的输出目录建议为：

```text
build/pi05_fp16.weights/
  manifest.json
  weights.bin
  weights.index.json
  weight_map.json
  convert_report.json
```

`manifest.json`：

```json
{
  "format": "devproc2.weights",
  "format_version": 1,
  "model": "openpi0.5",
  "precision": "float16",
  "data_file": "weights.bin",
  "index_file": "weights.index.json",
  "weight_map_file": "weight_map.json"
}
```

`weight_map.json` 是 devproc2 内部命名到本地权重 entry 的映射：

```json
{
  "format_version": 1,
  "weights": [
    {
      "name": "action_in_proj.weight",
      "kind": "weight",
      "shape": [1024, 32],
      "dtype": "float16",
      "layout": "row_major",
      "transform": null,
      "tied_to": null,
      "quant": null
    }
  ]
}
```

`convert_report.json` 才记录外部来源：

```json
{
  "source": {
    "type": "safetensors",
    "path": "/root/autodl-tmp/tools/pi05-pytorch-base/model.safetensors"
  },
  "ruleset": "openpi05_hf_to_devproc2_v1",
  "entries": [
    {
      "source_key": "action_in_proj.weight",
      "target_name": "action_in_proj.weight",
      "transform": null,
      "status": "converted"
    }
  ]
}
```

`convert_report.json` 不进入 runtime 依赖链。它用于排查转换规则、确认外部 checkpoint 来源和做完整性审计。

## Artifact 中的权重

artifact 扩展：

```text
artifact/
  executable.vm
  abi.json
  manifest.json
  weights/
    weights.bin
    weights.index.json
  kernels/
    *.cubin
  metadata/
    weight_map.json
    kernel_table.json
    function_table.json
```

artifact 中的 `weights/` 可以由 compile 阶段从 devproc2 weight package 复制，也可以在开发阶段用 manifest 记录本地引用。但面向部署的 artifact 必须自包含，不能依赖原始 checkpoint。

`weights.index.json`：

```json
{
  "format_version": 1,
  "data_file": "weights.bin",
  "entries": [
    {
      "name": "action_in_proj.weight",
      "offset": 0,
      "nbytes": 65536,
      "shape": [1024, 32],
      "dtype": "float16",
      "alignment": 256
    }
  ]
}
```

规则：

- 所有 weight blob 按 256-byte alignment 写入 `weights.bin`。
- C++ runtime 读取 index 后创建 tensor view。
- GPU weight 第一阶段可加载到 host 后复制到 device；后续可优化为 pinned/mmap staged copy。
- tied weight 不重复存储，index entry 可引用同一 offset。

## HF/safetensors 到 devproc2 的映射流程

该流程属于 `convert_weight`，在 compile 前独立执行：

1. 加载 openpi PyTorch 模型或 checkpoint metadata，读取外部 `named_parameters()`；如遇推理所需持久 tensor 常量，再从 PyTorch buffer 或 config 中显式加入转换规则。
2. 加载 devproc2 nn 模型定义，生成目标 `named_parameters()`，其中传统权重和持久 tensor 常量都以 Parameter/Weight 形式出现。
3. 通过规则表建立 source key 到 target name 的映射。
4. 校验 shape、dtype、layout 和 tied weight。
5. 应用必要 transform，例如 transpose、permute、dtype cast、layout pack。
6. 写入 devproc2 `weights.bin`。
7. 生成 devproc2 `weights.index.json`、`weight_map.json` 和 `convert_report.json`。

compile 阶段只做：

1. 读取 devproc2 weight package manifest。
2. 校验模型 `Parameter` 集合与 `weight_map.json` 完全匹配。
3. 校验 shape/dtype/layout/quant metadata 与 IR 期望一致。
4. 将权重包复制或链接进 artifact。
5. 在 artifact metadata 中记录 devproc2 weight package 的 format version 和校验和。

映射规则优先级：

- 完全同名优先。
- 已知 tied weight 使用 `tied_to`。
- 已知 transformers 命名差异使用显式规则表。
- 无法匹配必须报错，不能静默跳过。

转换规则必须版本化，例如 `openpi05_hf_to_devproc2_v1`。当 openpi 或 transformers checkpoint 命名变化时，需要新增 ruleset 版本，而不是在 compile 阶段做临时兼容。

## 与 IR 的关系

IR 中 `Parameter` 应作为特殊 value 或 module-level symbol 存在。建议打印形式：

```text
%wt = call @transpose(@weight("action_out_proj.weight")) {dim0=0, dim1=1}
%y0 = call @matmul(%x, %wt)
%y = call @add(%y0, @weight("action_out_proj.bias"))
```

lowering 后 VM codegen 需要把 weight handle 变成 runtime register 中的 Tensor。权重 tensor 的生命周期是 executable/session 级，不参与普通 activation memory planning。

## 量化预留

`QuantSpec` 草案：

```python
@dataclass(frozen=True)
class QuantSpec:
    scheme: str
    storage_dtype: str
    compute_dtype: str
    scale_name: str | None
    zero_point_name: str | None
    group_size: int | None
    axis: int | None
    packed_layout: str | None
```

示例：

```json
{
  "scheme": "int4_weight_only",
  "storage_dtype": "uint8",
  "compute_dtype": "float16",
  "scale_name": "layers.0.mlp.down_proj.weight.scale",
  "zero_point_name": null,
  "group_size": 128,
  "axis": 0,
  "packed_layout": "int4x2_row_major"
}
```

原则：

- 量化由外部框架完成，例如 torchao，然后通过 `convert_weight` 转成 devproc2 quantized weight package。
- devproc2 不假设原始 fp16 权重一定存在。
- kernel selection 必须能读取 `quant.scheme`，选择对应 quantized kernel。
- scale/zero_point 也是 weight entries，有独立 offset 和 dtype。

## Runtime 加载错误

C++ runtime 必须清晰报错：

- `weight_map.json` 缺失。
- `weights.index.json` 缺失。
- `weights.bin` 缺失。
- offset/nbytes 越界。
- dtype 不受支持。
- shape 与 ABI/IR 期望不一致。
- quant scheme 未被 runtime/kernel 支持。

## 测试策略

- `convert_weight` 对 safetensors key 到 devproc2 target name 做完整性检查。
- compile 拒绝直接读取 safetensors，只接受 devproc2 weight package。
- shape/dtype mismatch 报错。
- tied weight offset 复用。
- `weights.index.json` round-trip。
- C++ runtime 加载最小权重 artifact，并创建 Tensor view。
- 量化 metadata 能被解析，但不要求执行 quantized kernel。
