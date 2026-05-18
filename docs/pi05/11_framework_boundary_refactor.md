# Pi0.5 编译部署边界锐评与重构方案

## 结论

当前 Pi0.5 编译部署方案已经能跑通完整链路，也有明确的性能和精度验证资产；从工程交付角度看，它是有效的。但从 devproc2 框架设计角度看，它还不优雅。

核心问题不是 Pi0.5 模型目录本身存在，而是 Pi0.5 业务模型名已经进入了框架层命名空间：

- `python/devproc2/export/pi05.py`
- `python/devproc2/artifact/pi05.py`
- `python/devproc2/integrations/pi05/weights.py`

这些目录名表达的是 devproc2 的通用能力：导出、artifact 打包、外部 producer 集成。这里长期出现 `pi05.py`，说明通用框架和具体业务模型的边界仍然没守住。它比之前的兼容层清理进步了，但还没有达到“简洁与优雅”。

优雅的目标应该是：devproc2 框架层只提供通用编译、导出、artifact、权重包和资源打包机制；Pi0.5 只以模型 recipe 的形式挂进框架。框架可以运行 Pi0.5 recipe，但框架目录不应该拥有 Pi0.5 文件。

## 当前链路

现在的 Pi0.5 编译部署链路大致是：

```text
OpenPI / safetensors checkpoint
  -> devproc2.integrations.pi05.weights.convert_pi05_weights(...)
  -> devproc2 weight package

devproc2.export.pi05
  -> import devproc2.models.pi05.model
  -> 构造 Pi0.5 Module
  -> GraphBuilder 构图
  -> InferStructInfo / DPSLowering / MemoryPlanning / VMCodegen
  -> EmitExecutablePass / EmitABIPass

devproc2.artifact.pi05.prepare_pi05_artifact(...)
  -> 拷贝 weights / metadata / tokenizer
  -> 根据 emitted kernel_table 编译 cubin
  -> 写 metadata/pi05_artifact.json

C++ runtime
  -> LoadArtifact
  -> WeightStore 自动绑定权重
  -> CUDA kernel registry / packed func / tokenizer packed func
  -> VM Invoke
```

这条链路的能力是完整的，问题在于责任分布：

- `export/pi05.py` 同时知道 Pi0.5 entrypoint、shape 默认值、input ABI、Module 构造、compiler pass 顺序、executable emission、artifact resource packaging 和 CLI 参数。
- `artifact/pi05.py` 位于通用 artifact 包下，但知道 Pi0.5 tokenizer 默认路径、Pi0.5 manifest 文件名、Pi0.5 model id、Pi0.5 kernel catalog 迁移细节。
- `integrations/pi05/weights.py` 位于通用 integrations 包下，但实际承载 OpenPI/PaliGemma checkpoint 命名、Pi0.5 fusion 规则、FP8 layout、style table 预计算和权重包写入。
- C++ runtime 里也有类似问题：`runtime.tokenizer.paligemma_pi05_encode` 和 `runtime.cuda.pi05_fa2_bf16` 是业务模型级 runtime 能力，但目前注册在通用 runtime 路径里。

## 做得好的地方

这轮实现不是一无是处，几个方向是对的：

- `python/devproc2/models/pi05/model.py` 已经成为很薄的公开导出面，旧 `modules.py` 已删除。
- Pi0.5 模型构图实现已经收进 `python/devproc2/models/pi05/graph/`，并按业务域拆分为 `layers.py`、`ffn.py`、`vision.py`、`prefix.py`、`decoder.py`、`denoise.py`、`sample.py` 和 `_helpers.py`，避免再让 `pi05` 根目录或单个 `model.py` 聚合所有模型结构。
- 模型 fragment 文件已经避免直接出现 `dp.cuda_call`、`dp.call_dps_packed`、`dp.tensor_view` 这类后端细节，CUDA/HPC 调用集中到了 `models/pi05/ops.py`。
- `forward()` / `forward_fast()` 的双路径接口是合理的：normal path 表达标准语义，fast path 表达同一语义的高性能实现。
- `python/devproc2/quantization/` 已经出现通用 manifest 和 FP8 helper，不再把所有量化概念都塞进 Pi0.5 模型文件。
- artifact 已经自包含 weights、metadata、kernel table、tokenizer resource，部署形态是正确方向。
- 测试已经对旧兼容路径建立了硬删除约束，说明代码风格开始从“兼容优先”转向“边界优先”。

这些优点应该保留。重构目标不是推翻现有性能路径，而是把业务 recipe 从框架层剥离出去。

## 不优雅的地方

### 1. 框架目录出现业务模型文件

`devproc2.export`、`devproc2.artifact`、`devproc2.integrations` 应该是通用框架层。它们可以提供 generic API：

- `compile_entrypoint(...)`
- `emit_executable(...)`
- `export_artifact(...)`
- `ArtifactBuilder`
- `WeightPackageWriter`
- `ResourceSpec`

但它们不应该出现 `pi05.py`。一旦框架层接受 `pi05.py`，后面就会出现 `gemma.py`、`qwen.py`、`llama.py`，框架目录会变成业务模型集合，devproc2 的核心边界会被稀释。

### 2. `export/pi05.py` 既是 recipe，又是 compiler pipeline

`export/pi05.py` 当前超过一千行，里面混在一起的是两类完全不同的东西：

- 通用：构图、pass pipeline、VM codegen、ABI/executable emission。
- 业务：Pi0.5 entrypoint、input specs、默认 shape、模型类选择、artifact model name、CLI entry kind。

优雅设计里，通用 pipeline 应该不知道 Pi0.5；Pi0.5 只提供 `CompileRecipe`。框架拿 recipe 做事，而不是框架自己 import `PI05DenoiseLoop`、`PI05VisionEncoder`、`PI05SampleActionsFromTokens`。

### 3. `artifact/pi05.py` 把 resource packager 写成了模型专用

artifact builder 的职责应该是：

- 创建目录结构；
- 写通用 manifest；
- 拷贝权重包；
- 拷贝 tokenizer/resource；
- 根据 kernel table 编译或安装 cubin；
- 校验引用是否完整。

Pi0.5 应该只提供资源声明：

```text
model_id = "openpi0.5"
weights = weight package
resources = tokenizer.model
kernels = emitted kernel_table
metadata = pi05-specific recipe metadata
```

现在 `artifact/pi05.py` 反过来把这些都写死了，甚至 manifest format 是 `devproc2.artifact.pi05`。这说明 artifact schema 还没有真正通用化。

### 4. producer integration 和 deploy weight spec 混在一起

`models/pi05/weights.py` 和 `integrations/pi05/weights.py` 现在有明显重复：`QuantSpec`、`WeightEntry`、`WeightPackageWriter`、FP8 常量和 Pi0.5 shape 常量都有重复影子。

更干净的边界应该是：

- 通用权重包格式放在 `devproc2.weights` 或 `devproc2.artifact.weights`。
- Pi0.5 deploy 权重命名、fusion manifest、logical weight spec 放在 `models/pi05/weights.py`。
- OpenPI safetensors 到 devproc2 权重包的转换放在 `tools/pi05/convert_weights.py`，或者一个明确标注为 producer-side 的模型私有模块里。

`convert_pi05_weights(...)` 依赖 PyTorch、safetensors、OpenPI checkpoint 命名和本地预计算策略，它不是 devproc2 框架 API。

### 5. 默认路径和业务假设进入库代码

`artifact/pi05.py` 里有本机 tokenizer 默认路径：

```text
/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model
```

库代码不应该带这样的环境路径。路径可以出现在 docs、tool 默认配置、测试 fixture 或本机脚本里，但不应该作为 devproc2 package 的默认行为。

### 6. runtime 也有业务痕迹

Python 层不是唯一问题。C++ runtime 当前也存在 Pi0.5 业务能力直接注册到通用 runtime 的情况：

- `runtime.tokenizer.paligemma_pi05_encode`
- `runtime.cuda.pi05_fa2_bf16`
- 多个 runtime tests 和 benchmark 直接以 Pi0.5 artifact 为中心。

这些不一定要马上删除，但长期形态应该是 runtime extension 或 model-owned backend registration。通用 runtime 可以有 tokenizer 和 CUDA packed func registry，但不应该把 Pi0.5 prompt/state 规则当成通用 runtime 规则。

## 优雅设计应该长什么样

优雅不是把文件搬个位置，而是依赖方向变干净：

```text
devproc2.export        -> 只认识 CompileRecipe / ArtifactRecipe
devproc2.artifact      -> 只认识通用 ArtifactManifest / ResourceSpec / KernelSpec
devproc2.weights       -> 只认识通用 weight package schema
devproc2.quantization  -> 只认识通用 quantization manifest 和 requant helper

devproc2.models.pi05   -> Pi0.5 config / model / ops / weights / recipe
tools.pi05             -> OpenPI checkpoint conversion / oracle dump / 本机生产工具
runtime extensions     -> 可选 Pi0.5 tokenizer / FA2 / kernel registration
```

目标目录建议：

```text
python/devproc2/export/
  __init__.py
  pipeline.py          # generic compile/emit/export pipeline
  cli.py               # generic CLI: --recipe import.path:object

python/devproc2/artifact/
  __init__.py
  builder.py           # generic ArtifactBuilder
  manifest.py          # generic ArtifactManifest / ResourceSpec

python/devproc2/weights/
  __init__.py
  package.py           # generic WeightPackageWriter / WeightEntry / QuantSpec

python/devproc2/models/pi05/
  __init__.py
  config.py            # PI05Config and defaults
  model.py             # public re-export facade only
  graph/
    __init__.py        # graph fragment exports
    layers.py          # reusable primitive layers: Linear / Embedding / Attention
    ffn.py             # feed-forward block and FP8 FFN fast variants
    vision.py          # SigLIP patch embedding, encoder layer, encoder
    prefix.py          # PaliGemma prefix encoder layer and KV materializer
    decoder.py         # action decoder layer
    denoise.py         # denoise step and fixed-step loop
    sample.py          # sample_actions composite entry modules
    _helpers.py        # private shape/grid/reference helpers
  ops.py               # Pi0.5 CUDA/HPC facade
  weights.py           # Pi0.5 logical weight names and deploy manifest
  recipe.py            # Pi0.5 CompileRecipe / entrypoints / artifact resources
  cuda/

tools/pi05/
  convert_weights.py   # OpenPI/HF safetensors -> devproc2 weight package
  dump_torch_oracle.py
```

这样之后，框架层的导出调用应该像这样：

```python
from devproc2.export import export_artifact
from devproc2.models.pi05.recipe import pi05_recipe

summary = export_artifact(
    recipe=pi05_recipe.entrypoint("sample_tokens"),
    artifact_dir="build/pi05_fp8_sample_tokens_artifact",
    options={"sm_arch": 89, "compile_mode": "fast"},
    resources={"weight_package_dir": "build/pi05_fp8.weights"},
)
```

CLI 也应该是通用 CLI 加 recipe，而不是 `python -m devproc2.export.pi05`：

```bash
PYTHONPATH=python python -m devproc2.export.cli \
  --recipe devproc2.models.pi05.recipe:sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_artifact \
  --weight-package-dir build/pi05_fp8.weights \
  --sm-arch 89 \
  --compile-mode fast
```

如果希望有业务友好的短命令，可以放在模型命名空间或 tools 里：

```bash
PYTHONPATH=python python -m devproc2.models.pi05.recipe export \
  --entry sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_artifact
```

但 `devproc2.export` 自身不再出现 `pi05.py`。

### 模型包内部边界

Pi0.5 是业务模型，模型实现可以留在 `devproc2.models.pi05`；但模型包内部也不能退回到一个巨型文件。当前 `model.py` 已经只保留公开 re-export，实际构图文件统一放在 `devproc2.models.pi05.graph`，具体职责如下：

- `graph/layers.py`：最小可复用层，包含 `PI05Linear`、`PI05LanguageEmbedding`、`PI05Attention`。
- `graph/ffn.py`：`PI05FFN` 及其静态/动态 FP8 fast helper。
- `graph/vision.py`：SigLIP vision path，包括 patch embedding、vision block 和 vision encoder。
- `graph/prefix.py`：PaliGemma prefix encoder 和 prefix KV materialization。
- `graph/decoder.py`：action expert decoder layer。
- `graph/denoise.py`：`PI05DenoiseStep` 和 `PI05DenoiseLoop`。
- `graph/sample.py`：`sample_actions` 组合入口，把 vision/text/prefix/denoise 串起来。
- `graph/_helpers.py`：仅限私有 shape/grid/reference helper，不承载模型结构。
- `ops.py`：唯一允许封装 Pi0.5 CUDA/HPC primitive 的边界。

这个拆分的原则是：按业务语义拆，不按 normal/fast 拆。`forward()` 和 `forward_fast()` 仍然留在同一个 Module 内，表示同一个语义节点的 reference graph 和 high-performance graph；不能把 fast path 拆成另一套平行模型。

模型层的长期门禁应该是：

```bash
rg -n "dp\\.cuda_call|dp\\.call_dps_packed|dp\\.tensor_view|tensor_view\\(|runtime\\.cuda\\." \
  python/devproc2/models/pi05/model.py \
  python/devproc2/models/pi05/graph/layers.py \
  python/devproc2/models/pi05/graph/ffn.py \
  python/devproc2/models/pi05/graph/vision.py \
  python/devproc2/models/pi05/graph/prefix.py \
  python/devproc2/models/pi05/graph/decoder.py \
  python/devproc2/models/pi05/graph/denoise.py \
  python/devproc2/models/pi05/graph/sample.py \
  python/devproc2/models/pi05/graph/_helpers.py
```

该命令应无输出。`ops.py` 是受控例外，因为它就是 Pi0.5 CUDA/HPC facade。

## 核心抽象

### CompileRecipe

Pi0.5 需要告诉框架“怎么构图”，但不应该把 compiler pipeline 复制一份。

建议通用 recipe 形态：

```python
@dataclass(frozen=True)
class EntrypointRecipe:
    name: str
    model_id: str
    build_module: Callable[[dict[str, object]], Module]
    input_specs: Callable[[dict[str, object]], dict[str, object]]
    function_name: str = "main"
    normal_method: str = "forward"
    fast_method: str = "forward_fast"

@dataclass(frozen=True)
class CompileRecipe:
    model_id: str
    entrypoints: dict[str, EntrypointRecipe]
```

`devproc2.export.pipeline` 负责：

- reset DSL；
- 选择 `forward` 或 `forward_fast`；
- GraphBuilder 构图；
- 校验 normal path 不出现 backend op；
- 执行 pass pipeline；
- emit `executable.vm` 和 `abi.json`。

Pi0.5 recipe 只负责：

- 构造 `PI05DenoiseLoop`、`PI05VisionEncoder`、`PI05SampleActionsFromTokens`；
- 给出 input specs；
- 给出默认 config；
- 声明 entrypoint 名称和 ABI。

### ArtifactRecipe

artifact builder 应该通用，Pi0.5 只传资源声明：

```python
@dataclass(frozen=True)
class ArtifactRecipe:
    model_id: str
    resources: tuple[ResourceSpec, ...]
    metadata: dict[str, object]
```

通用 artifact manifest 应该像：

```json
{
  "format": "devproc2.artifact",
  "format_version": 1,
  "model_id": "openpi0.5",
  "entrypoint": "sample_tokens",
  "target": {"kind": "cuda", "arch": "sm89"},
  "executable": "executable.vm",
  "abi": "abi.json",
  "weights": {"path": "weights", "index": "weights/weights.index.json"},
  "resources": [{"name": "tokenizer", "path": "resources/tokenizer.model"}],
  "kernels": {"table": "metadata/kernel_table.json", "compiled": true}
}
```

不再写 `metadata/pi05_artifact.json`，也不再把 format 写成 `devproc2.artifact.pi05`。Pi0.5 专有信息可以放进：

```text
metadata/model.json
metadata/pi05_recipe.json
```

但通用 loader 只依赖 `metadata/artifact.json`。

### WeightPackage

`WeightPackageWriter`、`WeightEntry`、`QuantSpec` 是通用格式，不应该在 Pi0.5 deploy spec 和 Pi0.5 producer integration 里各自定义一遍。

目标边界：

```text
devproc2.weights.package
  - WeightPackageWriter
  - WeightEntry
  - QuantSpec
  - read_manifest(...)
  - validate_package(...)

devproc2.models.pi05.weights
  - pi05_fp8_weight_name(...)
  - pi05_fp8_scale_name(...)
  - pi05_deploy_quantization_manifest(...)
  - Pi0.5 logical weight groups

tools.pi05.convert_weights
  - convert_pi05_weights(...)
  - _convert_openpi_safetensors(...)
  - _precompute_decoder_styles(...)
```

这样 PyTorch/safetensors/OpenPI 命名不会进入框架层。

## 迁移计划

### Phase 0：保持模型包内部小文件边界

这一步已经完成初始拆分：

- `model.py` 仅作为公开 re-export facade。
- 基础层、FFN、vision、prefix、decoder、denoise、sample 入口各自落到 `models/pi05/graph/` 下的独立文件。
- `tests/compiler/test_pi05_fast_modules.py` 扫描所有模型 fragment，确保裸 backend API 不回流到模型层。

后续重构如果继续增加 Pi0.5 能力，应优先放入 `graph/` 下已有业务域文件；当单个文件继续膨胀时，再按同一原则细分，例如把 `graph/vision.py` 拆成 `graph/vision_patch.py`、`graph/vision_layer.py`、`graph/vision_encoder.py`。不要重新把实现塞回 `model.py` 或 `pi05` 根目录。

### Phase 1：先抽通用 pipeline，不移动行为

新增：

- `python/devproc2/export/pipeline.py`
- `python/devproc2/export/recipe.py`

把 `export/pi05.py` 里的通用部分抽出来：

- `_select_method`
- `_build_graph`
- normal path backend op 校验
- `_compile_ir_module`
- `_emit_compile_result`

Pi0.5 旧导出函数先调用 generic pipeline，确保行为不变。

验收：

```bash
PYTHONPATH=python pytest tests/compiler/test_pi05_fast_modules.py
```

### Phase 2：建立 Pi0.5 recipe

新增：

- `python/devproc2/models/pi05/recipe.py`

把以下内容从 `export/pi05.py` 移入 Pi0.5 recipe：

- Pi0.5 entrypoint 列表；
- input spec factory；
- Pi0.5 Module 构造；
- default model names；
- `step`、`loop`、`sample_precomputed_prefix`、`sample_precomputed_prefix_embs`、`sample_tokens`、`vision_encoder`、`paligemma_prefix_encoder`、`paligemma_prefix_kv_encoder`。

`devproc2.export` 只暴露：

- `compile_entrypoint(recipe, ...)`
- `emit_entrypoint(recipe, ...)`
- `export_artifact(recipe, ...)`

验收：

```bash
rg -n "PI05|pi05|openpi|paligemma" python/devproc2/export
```

应无匹配，或只允许文档字符串中的通用示例。更严格的最终门禁应该是零匹配。

### Phase 3：抽通用 artifact builder

新增：

- `python/devproc2/artifact/builder.py`
- `python/devproc2/artifact/manifest.py`

把 `artifact/pi05.py` 拆成：

- 通用目录创建；
- 通用权重包安装；
- 通用 resource copy；
- 通用 kernel table 到 cubin packaging；
- 通用 `metadata/artifact.json` 写入。

Pi0.5 的 tokenizer、model id、resource policy 移到 `models/pi05/recipe.py`。

最终删除：

```text
python/devproc2/artifact/pi05.py
```

验收：

```bash
find python/devproc2/artifact -maxdepth 1 -name '*pi05*' -print
rg -n "openpi|paligemma|pi05" python/devproc2/artifact
```

两个命令都应该无输出。

### Phase 4：拆 producer conversion

新增：

- `python/devproc2/weights/package.py`
- `tools/pi05/convert_weights.py`

移动职责：

- 通用 `WeightPackageWriter` 等 schema 进入 `devproc2.weights.package`。
- Pi0.5 logical weight naming 保留在 `models/pi05/weights.py`。
- `convert_pi05_weights(...)` 从 `devproc2.integrations.pi05.weights` 移到 `tools/pi05/convert_weights.py`。

最终删除：

```text
python/devproc2/integrations/pi05/
```

如果未来确实需要 producer integration package，应命名为具体 producer，而不是业务模型：

```text
python/devproc2/integrations/modelopt/
python/devproc2/integrations/safetensors/
```

验收：

```bash
find python/devproc2/integrations -maxdepth 2 -name '*pi05*' -print
rg -n "convert_pi05_weights|openpi|paligemma" python/devproc2/integrations
```

应无输出。

### Phase 5：收 runtime 业务注册

把 runtime 中的 Pi0.5 业务注册收进扩展边界：

- tokenizer 通用化为 `runtime.tokenizer.sentencepiece_encode`，Pi0.5 prompt/state formatting 放在模型 recipe 或 runtime model extension。
- `runtime.cuda.pi05_fa2_bf16` 要么变成通用 attention packed func，要么进入 Pi0.5-owned runtime extension。
- Pi0.5 benchmark 和 oracle tests 保留在测试目录可以接受，但构建注册路径应明确它们是 model tests，不是 runtime core contract。

这一步可以晚于 Python 层目录清理，但应该进入重构计划，否则 devproc2 runtime 会继续累积业务模型痕迹。

### Phase 6：删除旧路径，不做长期兼容

这轮代码风格已经选择直接删除兼容层，这个方向是对的。完成 recipe 化之后，应直接删除：

```text
python/devproc2/export/pi05.py
python/devproc2/artifact/pi05.py
python/devproc2/integrations/pi05/
```

测试改为断言旧路径不可导入：

```python
import importlib
import pytest

for name in (
    "devproc2.export.pi05",
    "devproc2.artifact.pi05",
    "devproc2.integrations.pi05",
):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)
```

同时新增边界门禁：

```bash
find python/devproc2/export python/devproc2/artifact python/devproc2/integrations \
  -name '*pi05*' -print

rg -n "devproc2\\.(export|artifact|integrations)\\.pi05" \
  python tests docs tools
```

最终应无输出。

## 推荐的新用户入口

### 权重转换

权重转换是 producer-side tool：

```bash
PYTHONPATH=python python tools/pi05/convert_weights.py \
  --checkpoint-dir /root/autodl-tmp/tools/pi05-pytorch-base \
  --output-dir build/pi05_fp8.weights \
  --hardware rtx_sm89
```

### artifact 导出

框架通用 CLI：

```bash
PYTHONPATH=python python -m devproc2.export.cli \
  --recipe devproc2.models.pi05.recipe:sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_artifact \
  --weight-package-dir build/pi05_fp8.weights \
  --resource tokenizer=/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model \
  --sm-arch 89 \
  --compile-mode fast
```

### Python API

```python
from devproc2.export import export_artifact
from devproc2.models.pi05.recipe import pi05_recipe

summary = export_artifact(
    recipe=pi05_recipe.entrypoint("sample_tokens"),
    artifact_dir="build/pi05_fp8_sample_tokens_artifact",
    resources={
        "weight_package_dir": "build/pi05_fp8.weights",
        "tokenizer": "/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model",
    },
    options={
        "sm_arch": 89,
        "compile_mode": "fast",
    },
)
```

## 最终验收标准

结构验收：

- `python/devproc2/export/` 下没有 Pi0.5 文件，也不 import Pi0.5 模型。
- `python/devproc2/artifact/` 下没有 Pi0.5 文件，也没有 OpenPI/tokenizer 默认路径。
- `python/devproc2/integrations/` 下没有 Pi0.5 业务转换器。
- `python/devproc2/models/pi05/` 是 Pi0.5 业务模型、ops、weights、recipe 的唯一 Python package 归属。
- `python/devproc2/models/pi05/model.py` 只做 re-export，不承载实际模型实现。
- Pi0.5 构图实现集中在 `python/devproc2/models/pi05/graph/`，按业务域拆分，裸 backend API 只能出现在 `models/pi05/ops.py`。
- `tools/pi05/` 是 OpenPI producer、oracle、一次性转换脚本的唯一归属。
- artifact manifest format 是 `devproc2.artifact`，Pi0.5 只作为 `model_id` 或 recipe metadata 出现。

行为验收：

- 现有 Pi0.5 编译入口全部能通过 recipe 导出同等 artifact。
- `PYTHONPATH=python pytest` 全量通过。
- C++ runtime oracle、tokenizer、kernel launch、benchmark 继续可跑。
- 新旧 artifact 的 `abi.json`、权重参数名、kernel table 在迁移阶段可比对，性能不因目录重构回退。

边界验收：

```bash
find python/devproc2/export python/devproc2/artifact python/devproc2/integrations \
  -name '*pi05*' -print

rg -n "openpi|paligemma|pi05" \
  python/devproc2/export python/devproc2/artifact python/devproc2/integrations
```

最终两个命令都应该无输出。

## 一句话评价

当前方案是“功能优先的有效实现”，不是“框架边界优雅的实现”。真正优雅的 devproc2 应该让 Pi0.5 作为 recipe 被框架消费，而不是让框架目录长出 Pi0.5 业务文件。下一轮重构的主线应当是 recipe 化、artifact 通用化、权重包通用化、producer 工具外移，然后直接删除 `export/pi05.py`、`artifact/pi05.py` 和 `integrations/pi05/`。
