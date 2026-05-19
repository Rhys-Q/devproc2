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
  -> tools/pi05/convert_weights.py.convert_pi05_weights(...)
  -> devproc2 weight package

python/devproc2/export/pi05.py
  -> import devproc2.models.pi05.model
  -> 构造 Pi0.5 Module
  -> GraphBuilder 构图
  -> InferStructInfo / DPSLowering / MemoryPlanning / VMCodegen
  -> EmitExecutablePass / EmitABIPass

python/devproc2/artifact/pi05.py.prepare_pi05_artifact(...)
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

现在 `artifact/pi05.py` 反过来把这些都写死了，甚至 manifest format 是 `python/devproc2/artifact/pi05.py`。这说明 artifact schema 还没有真正通用化。

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

### 7. Pi0.5 前端 DSL 的 CUDA 接入仍然只是 facade

当前 Pi0.5 graph 文件已经不再直接写 `dp.cuda_call(...)`、`dp.call_dps_packed(...)` 和 `runtime.cuda.*`，这是明显进步。但这还不等于前端 DSL 很优雅。

现在的真实形态是：`graph/` 调用 `pi05_ops.call_cuda(...)`，`ops.py` 再把它转成 `dp.cuda_call(source::symbol, metadata=...)`。这只是把裸 backend API 包了一层 facade。模型 fragment 里仍然能看到大量 backend 负担：

- 手写具体 CUDA symbol，例如 `pi05_qkv_split_rope_concat_bf16`、`pi05_gate_residual_ada_norm_to_fp8_bf16`；
- 手写 kernel 参数顺序；
- 手写输出 buffer；
- 手写 launch grid/block/shared memory；
- 手写 packed func 名称和 ABI 顺序。

这说明当前实现已经“把脏东西集中到一扇门后面”，但还没有真正做到“前端表达模型语义”。优雅的 graph 层应该写：

```python
q, full_k, full_v = pi05_ops.qkv_split_rope_concat(...)
attn = pi05_ops.prefix_attention(...)
hidden = pi05_ops.gated_residual_ada_norm_to_fp8(...)
```

而不是写：

```python
pi05_ops.call_cuda(
    "pi05_qkv_split_rope_concat_bf16",
    qkv,
    rope,
    prefix_k,
    prefix_v,
    prefix_rows,
    rows,
    q_dim,
    k_dim,
    v_dim,
    head_dim,
    q,
    full_k,
    full_v,
    launch=...,
)
```

也就是说，`ops.py` 的长期职责不应该是一个通用 `call_cuda(name, *args, launch=...)` 入口，而应该是 Pi0.5 语义级 primitive 集合。kernel symbol、launch policy、参数 ABI、output layout、effect summary 都应进入 `ops.py` 内部的 typed kernel catalog 或 backend registry。

### 8. `runtime.cuda.*` packed func 路径不是优雅解

`runtime/src/cuda/cuda_gemm.cc` 及对应 header 当前承担了 Pi0.5 fast path 的核心性能能力：FP8/BF16 GEMM 的 cuBLASLt host runner、CUTLASS FP8 特化、FA2 attention host runner、scratch buffer 管理、stream 继承和 packed func 注册。

从性能角度看，这些能力不能被朴素 CUDA kernel 替代。任何“为了目录干净，把 cuBLASLt/CUTLASS/FA2 换成简单 `__global__` kernel”的做法都不可接受，属于性能回退。

但从框架边界看，它们也不应该作为通用 runtime packed func 注册：

- `runtime.cuda.fp8_nt_bf16` 当前服务的是 Pi0.5 FP8 artifact layout、scale ABI 和固定 shape 性能路径，不是标准 `matmul` lowering 自动选择的通用 kernel。
- `runtime.cuda.fp8_nt_bf16_accum` 暴露的是 Pi0.5 residual accumulate fusion 约定，不是通用 runtime contract。
- `runtime.cuda.pi05_fa2_bf16` 和 `runtime.cuda.pi05_fa2_bf16_batched` 名字上已经承认是 Pi0.5 业务 backend，却注册在 runtime core。
- VM 通过 `callee.name.startswith("runtime.cuda.")` 注入默认 CUDA stream，这把一个命名约定变成了执行语义。

标准 op 的优雅路径应该是：frontend 产生 `dp.matmul`、`dp.attention` 等标准 op，compiler lowering 根据 target、dtype、layout 和 shape 选择提前注册好的 `KernelSpec` 或 backend provider。它不应该依赖 runtime 启动时全局注册一个 `"runtime.cuda.fp8_nt_bf16"` packed func。

显式 CUDA kernel 的优雅路径应该是：前端 DSL 或 `ops.py` 插入 `CudaCallOp`，DPS lowering 生成 `KernelRef`，artifact 写出 kernel table，runtime 只按 artifact 加载 cubin 并 launch。它也不应该走 runtime packed func 注册。

因此，`runtime.cuda.*` packed func 路径应该从 Pi0.5 fast path 中删除。但删除的含义不是降级性能，也不是删除 `call_dps_packed` 这类 DPS host-call ABI，而是把“编译 host backend 并注册 packed funcs”做成 devproc2 的标准通用路径。Pi0.5 只是通过这条标准路径把高性能 host runner 收回到 Pi0.5-owned backend 边界：

```text
models/pi05/cuda/
  kernels/              # AOT source-symbol CUDA kernels
  backends/
    fp8_gemm/           # cuBLASLt/CUTLASS host runner, Pi0.5-owned
    fa2/                # FA2 wrapper and scratch policy, Pi0.5-owned
```

调用边界应该是：

- `pi05_ops` 发出 Pi0.5 semantic op，内部用 `call_dps_packed("pi05.cuda.*", ...)` 表达 host backend 调用；VM 在 artifact 声明驱动下注册 Pi0.5-owned packed funcs。
- `pi05_ops` 发出显式 source-symbol `CudaCallOp`，仅用于纯 device kernel；cuBLASLt/CUTLASS/FA2 这类 host runner 不伪装成 cubin kernel，而由 Pi0.5 backend provider 负责。

更长期可以新增 `BackendCallRef` / `CallBackendOp`，让 IR 显式区分 packed func 和 host backend；但第一阶段不必阻塞在新 IR。`call_dps_packed` 本身可以成为 host backend 的标准表达，只要注册来源和命名空间是 artifact/recipe 驱动，而不是 runtime core 硬编码。

通用 runtime 不应该在启动时无条件注册 Pi0.5 CUDA packed func。runtime core 只保留通用机制：加载 artifact、加载 cubin、加载 packed backend、调用 backend 注册符号、分发 packed func、传入 stream、管理 tensor/storage。Pi0.5 backend 是否存在、如何初始化、需要哪些 CUDA/C++ source，应由 Pi0.5 配置、recipe 和 artifact manifest 声明。

## 优雅设计应该长什么样

优雅不是把文件搬个位置，而是依赖方向变干净：

```text
devproc2.export        -> 只认识 CompileRecipe / ArtifactRecipe
devproc2.artifact      -> 只认识通用 ArtifactManifest / ResourceSpec / KernelSpec / PackedBackendRecipe
devproc2.weights       -> 只认识通用 weight package schema
devproc2.quantization  -> 只认识通用 quantization manifest 和 requant helper

devproc2.models.pi05   -> Pi0.5 config / model / ops / weights / recipe / CUDA backend
tools.pi05             -> OpenPI checkpoint conversion / oracle dump / 本机生产工具
runtime core           -> 通用 VM / artifact loader / CUDA kernel launch / backend dispatch
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
  manifest.py          # generic ArtifactManifest / ResourceSpec / PackedBackendRecipe

python/devproc2/weights/
  __init__.py
  package.py           # generic WeightPackageWriter / WeightEntry / QuantSpec

python/devproc2/models/pi05/
  __init__.py
  config.py            # PI05Config and defaults
  model.py             # public re-export facade + product export declaration
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
  diagnostic_export_spec.py  # oracle / benchmark entrypoints
  cuda/
    kernels/           # source-symbol CUDA kernels
    backends/          # Pi0.5-owned cuBLASLt/CUTLASS/FA2 host runners

tools/pi05/
  convert_weights.py   # OpenPI/HF safetensors -> devproc2 weight package
  dump_torch_oracle.py
```

这样之后，框架层的导出调用应该像这样：

```python
from devproc2.export import export_artifact
from devproc2.models.pi05.model import pi05_recipe

summary = export_artifact(
    recipe=pi05_recipe.entrypoint("sample_tokens"),
    artifact_dir="build/pi05_fp8_sample_tokens_artifact",
    options={"sm_arch": 89, "compile_mode": "fast"},
    resources={"weight_package_dir": "build/pi05_fp8.weights"},
)
```

CLI 也应该是通用 CLI 加 recipe，而不是 `python -m python/devproc2/export/pi05.py`：

```bash
PYTHONPATH=python python -m devproc2.export.cli \
  --recipe devproc2.models.pi05.model:sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_artifact \
  --weight-package-dir build/pi05_fp8.weights \
  --sm-arch 89 \
  --compile-mode fast
```

如果希望有业务友好的短命令，应使用通用产品入口 `python -m devproc2.build --model pi05 --entry sample_tokens ...`。`devproc2.export` 自身不再出现 `pi05.py`。

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

不再写 `metadata/pi05_artifact.json`，也不再把 format 写成 `python/devproc2/artifact/pi05.py`。Pi0.5 专有信息可以放进：

```text
metadata/model.json
metadata/pi05_recipe.json
```

但通用 loader 只依赖 `metadata/artifact.json`。

### PackedBackendRecipe

CUDA backend 需要区分两类东西：

- device kernel：可以由 `CudaCallOp` 指向 `source::symbol`，AOT 编译成 cubin，进入 `metadata/kernel_table.json`；
- host backend：需要 cuBLASLt、CUTLASS device adapter、FA2 wrapper、scratch buffer、tuning cache 和 CUDA stream，不应该伪装成一个普通 `__global__` kernel。

host backend 应该走一条通用标准路径：模型 recipe 根据配置声明 backend source，artifact builder 编译这些 source，runtime loader 按 manifest 加载编译产物并调用注册函数，注册函数把 packed funcs 安装到 `PackedFuncRegistry`。这样 `call_dps_packed(...)` 在运行时能正常解析，同时 backend ownership 仍然属于模型。

通用 recipe 形态：

```python
@dataclass(frozen=True)
class PackedFuncSpec:
    name: str                  # "pi05.cuda.fp8_nt_bf16"
    device: str | None = None  # "cuda" means VM passes default CUDA stream
    effect: str = "opaque"

@dataclass(frozen=True)
class PackedBackendRecipe:
    name: str                         # "pi05.cuda"
    kind: str                         # "compiled_packed_backend"
    sources: tuple[str, ...]          # model-owned .cc/.cu sources
    include_dirs: tuple[str, ...] = ()
    compile_definitions: tuple[str, ...] = ()
    compile_options: tuple[str, ...] = ()
    link_libraries: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()     # ("sm89",)
    register_symbol: str = "devproc2_register_packed_backend"
    packed_funcs: tuple[PackedFuncSpec, ...] = ()
```

Pi0.5 可以通过 `PI05Config` 或 `model.py` 产生这个通用 recipe：

```python
PackedBackendRecipe(
    name="pi05.cuda",
    kind="compiled_packed_backend",
    sources=(
        "python/devproc2/models/pi05/cuda/backends/cuda_gemm.cc",
        "python/devproc2/models/pi05/cuda/backends/cutlass_fp8_gemm_sm89.cu",
        "python/devproc2/models/pi05/cuda/backends/fa2/fa2_wrapper.cu",
    ),
    include_dirs=(...),
    compile_definitions=("DEVPROC2_WITH_PI05_FA2",),
    link_libraries=("CUDA::cudart", "CUDA::cublasLt"),
    targets=("sm89",),
    register_symbol="devproc2_register_pi05_cuda_backend",
    packed_funcs=(
        PackedFuncSpec("pi05.cuda.fp8_nt_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.fp8_nt_bf16_accum", device="cuda"),
        PackedFuncSpec("pi05.cuda.bf16_nn_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.fa2_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.fa2_bf16_batched", device="cuda"),
    ),
)
```

artifact builder 负责编译 backend 产物，例如：

```text
artifact/
  backends/
    pi05_cuda.so
  metadata/
    packed_backend_table.json
```

对应 artifact manifest 声明：

```json
{
  "packed_backends": [
    {
      "name": "pi05.cuda",
      "kind": "compiled_packed_backend",
      "library": "backends/pi05_cuda.so",
      "register_symbol": "devproc2_register_pi05_cuda_backend",
      "target_arch": "sm89",
      "packed_funcs": [
        {"name": "pi05.cuda.fp8_nt_bf16", "device": "cuda"},
        {"name": "pi05.cuda.fp8_nt_bf16_accum", "device": "cuda"},
        {"name": "pi05.cuda.bf16_nn_bf16", "device": "cuda"},
        {"name": "pi05.cuda.fa2_bf16", "device": "cuda"},
        {"name": "pi05.cuda.fa2_bf16_batched", "device": "cuda"}
      ]
    }
  ]
}
```

runtime loader 的通用流程：

```text
Executable::Load(artifact)
  -> read metadata/packed_backend_table.json
  -> dlopen artifact/backends/pi05_cuda.so
  -> dlsym devproc2_register_pi05_cuda_backend
  -> call register function with PackedFuncRegistry / backend context
  -> verify abi.required_packed_funcs are registered
```

Pi0.5 `ops.py` 使用这条通用路径：

```python
dp.call_dps_packed(
    "pi05.cuda.fp8_nt_bf16",
    inputs=[x_fp8, w_fp8, rows, out_features, in_features, x_scale, w_scale, out],
)
```

这里 `call_dps_packed` 是标准路径，不是脏路径。脏的是 runtime core 无条件注册业务 packed funcs，以及用 `runtime.cuda.` 名字前缀隐式决定 stream 语义。

VM 仍然可以给 CUDA packed backend 传入 default stream，也仍然可以参与 CUDA graph capture；但 stream 注入必须来自 `PackedFuncSpec(device="cuda")` 或 registry traits，而不是靠字符串前缀判断。

这条路径要作为 devproc2 标准通用能力，而不是 Pi0.5 私有特判。任何模型只要有 host-side 高性能实现，都可以声明自己的 `PackedBackendRecipe`：

```text
model recipe
  -> declares PackedBackendRecipe
  -> export/artifact builder compiles backend sources
  -> artifact records packed_backend_table.json
  -> runtime loader dlopen/dlsym/register
  -> call_dps_packed resolves through PackedFuncRegistry
```

Pi0.5 的 `cuda_gemm.cc` 迁移后只是这条路径的首个使用者：

```text
Pi05Config(enable_cuda_backend=True, target="rtx4090_sm89")
  -> recipe emits PackedBackendRecipe("pi05.cuda")
  -> artifact builder compiles pi05-owned cuda_gemm.cc / cutlass_fp8_gemm_sm89.cu / fa2 sources
  -> runtime loads backends/pi05_cuda.so
  -> devproc2_register_pi05_cuda_backend registers pi05.cuda.* packed funcs
  -> ops.py call_dps_packed("pi05.cuda.fp8_nt_bf16", ...)
```

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

### Phase 2：建立 Pi0.5 export spec

新增：

- `python/devproc2/models/pi05/diagnostic_export_spec.py`

把以下内容从 `export/pi05.py` 移入 Pi0.5 model-owned export spec：

- Pi0.5 entrypoint 列表；
- input spec factory；
- Pi0.5 Module 构造；
- default model names；
- 产品 recipe 只暴露 `sample_tokens`。`step`、`loop`、`sample_precomputed_prefix`、`sample_precomputed_prefix_embs`、`vision_encoder`、`paligemma_prefix_encoder`、`paligemma_prefix_kv_encoder` 归入 diagnostic recipe，用于 oracle、benchmark 和性能归因。

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
- 通用 packed backend source 编译、动态库安装和 `packed_backend_table.json` 写入；
- 通用 `metadata/artifact.json` 写入。

Pi0.5 的 tokenizer、model id、resource policy、packed backend recipe 移到 `models/pi05/model.py`。

最终删除：

```text
python/devproc2/artifact/pi05.py
```

验收：

```bash
find python/devproc2/artifact -maxdepth 1 -name '*pi05*' -print
rg -n "openpi|paligemma|pi05" python/devproc2/artifact
```

两个命令都应该无输出。`artifact` 可以包含通用 `PackedBackendRecipe` / `packed_backend_table.json` 处理逻辑，但不能包含 Pi0.5 专用 source 列表、packed func 名称或默认路径。

### Phase 4：拆 producer conversion

新增：

- `python/devproc2/weights/package.py`
- `tools/pi05/convert_weights.py`

移动职责：

- 通用 `WeightPackageWriter` 等 schema 进入 `devproc2.weights.package`。
- Pi0.5 logical weight naming 保留在 `models/pi05/weights.py`。
- `convert_pi05_weights(...)` 从 `tools/pi05/convert_weights.py` 移到 `tools/pi05/convert_weights.py`。

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

### Phase 5：收回 Pi0.5 CUDA host backend，不允许性能回退

这一阶段的目标不是把高性能路径改成普通 CUDA kernel，而是把 ownership 和调用路径从 runtime core 收回到 Pi0.5。

当前 `runtime/src/cuda/cuda_gemm.cc` 和 `runtime/include/devproc2/runtime/cuda_gemm.h` 不应该继续作为 devproc2 标准 runtime API 存在。它们目前包含的是 Pi0.5 fast path 需要的 host backend：

- cuBLASLt FP8/BF16 GEMM runner；
- CUTLASS FP8 shape specialization；
- FA2 attention host wrapper；
- Pi0.5 packed func ABI；
- Pi0.5 stream/scratch/tuning policy。

这些不是 devproc2 标准算子的 kernel registry 结果。标准 op 的 lowering 应该选择提前注册好的 `KernelSpec` / backend provider；显式 CUDA kernel 应该由前端 DSL 插入 `CudaCallOp`，再进入 artifact kernel table。二者都不应该依赖 runtime 启动时注册 `runtime.cuda.*` packed func。

目标目录建议：

```text
python/devproc2/models/pi05/cuda/
  kernels/
    pi05_kernels.cu
  backends/
    fp8_gemm/
      cublaslt_runner.cc
      cutlass_fp8_gemm_sm89.cu
    fa2/
      fa2_wrapper.cu
      flash_attn_2_src/
```

目标调用链建议：

```text
Pi0.5 graph
  -> pi05_ops semantic primitive
  -> pi05_ops emits call_dps_packed("pi05.cuda.*", ...)
  -> compiler records required_packed_funcs
  -> Pi0.5 recipe emits PackedBackendRecipe from PI05Config
  -> artifact builder compiles pi05-owned backend sources
  -> artifact records packed_backend_table.json
  -> runtime loader registers artifact-declared packed funcs
  -> VM dispatches PackedFuncRef with default CUDA stream trait
  -> Pi0.5-owned cuBLASLt/CUTLASS/FA2 host runner
```

重构要求：

- 删除 Pi0.5 fast path 对 `runtime.cuda.*` packed func 名称的依赖。
- 删除 runtime core 中无条件 `RegisterCUDAPackedFuncs()` 这类 Pi0.5 CUDA backend 注册路径。
- 新增通用 `PackedBackendRecipe` / `packed_backend_table.json` 路径，支持任意模型声明“编译这些 host backend sources 并注册这些 packed funcs”。
- Pi0.5 的 `PI05Config` 决定是否启用 `pi05.cuda` packed backend、目标 arch、CUTLASS/FA2 编译开关和 source 列表。
- 保留 cuBLASLt autotune、CUTLASS FP8 specialization、FA2 split-KV/scratch policy；不能用朴素 matmul 或朴素 attention kernel 替代。
- 如果需要 host runner，新增 Pi0.5-owned backend provider 或 artifact-declared extension；不要把 host runner 伪装成 source-symbol cubin kernel。
- VM 可以提供通用 stream 传入能力，但不能用 `runtime.cuda.` 前缀约定来决定执行语义。
- Pi0.5 backend 是否存在、需要哪些动态库/cubin/metadata，应由 Pi0.5 recipe 或 artifact manifest 声明。

第一阶段终局：

- `call_dps_packed("pi05.cuda.*", ...)` 是合法标准路径；
- `runtime` 不再无条件注册 Pi0.5 CUDA backend；
- `artifact` 根据 `PackedBackendRecipe` 编译并安装 `backends/pi05_cuda.so`；
- `runtime` 根据 `packed_backend_table.json` 加载该 backend 并调用 `devproc2_register_pi05_cuda_backend`；
- `abi.required_packed_funcs` 校验 `pi05.cuda.*` 是否已注册。

更长期可以把 `PackedFuncRef` 升级为显式 `BackendCallRef`，但这不是清理 `runtime.cuda.*` 的前置条件。

验收：

```bash
rg -n "runtime\\.cuda\\." python/devproc2/models/pi05 runtime/src runtime/include
rg -n "RegisterCUDAPackedFuncs|cuda_gemm" runtime/src runtime/include
```

最终应无输出。与此同时，Pi0.5 benchmark 和 oracle tests 必须继续通过，性能指标不能因为边界重构回退。

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
    "python/devproc2/export/pi05.py",
    "python/devproc2/artifact/pi05.py",
    "python/devproc2/integrations/pi05",
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

## 当前本地资产路径

这些路径只作为当前机器上的执行参数和文档示例使用，不能写入 devproc2 库代码默认值：

- ckpt 路径：`/root/autodl-tmp/tools/pi05-pytorch-base`
- OpenPI inputs 路径：`/root/autodl-tmp/openpi/outputs/pi05_torch_infer`
- OpenPI outputs 路径：`/root/autodl-tmp/openpi/outputs/pi05_torch_infer`
- OpenPI tokenizer 目录：`/root/autodl-tmp/openpi/outputs/pi05_torch_infer`
- OpenPI tokenizer model：`/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model`

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
  --recipe devproc2.models.pi05.model:sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_artifact \
  --weight-package-dir build/pi05_fp8.weights \
  --resource tokenizer=/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model \
  --sm-arch 89 \
  --compile-mode fast
```

### Python API

```python
from devproc2.export import export_artifact
from devproc2.models.pi05.model import pi05_recipe

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
- Pi0.5 FP8 GEMM、BF16 GEMM、FA2 attention 继续使用 cuBLASLt/CUTLASS/FA2 级别实现；禁止用朴素 CUDA kernel 作为重构替代品。
- Pi0.5 artifact 通过通用 `packed_backend_table.json` 声明并加载 `pi05.cuda` packed backend，`abi.required_packed_funcs` 中的 `pi05.cuda.*` 能由该 backend 注册满足。

边界验收：

```bash
find python/devproc2/export python/devproc2/artifact python/devproc2/integrations \
  -name '*pi05*' -print

rg -n "openpi|paligemma|pi05" \
  python/devproc2/export python/devproc2/artifact python/devproc2/integrations

rg -n "runtime\\.cuda\\.|RegisterCUDAPackedFuncs|cuda_gemm" \
  python/devproc2/models/pi05 runtime/src runtime/include
```

最终这些命令都应该无输出。Pi0.5 CUDA backend 可以存在，但归属必须在 `models/pi05/cuda/`，并通过通用 artifact-declared packed backend 路径注册，不在 runtime core 全局注册路径。

## Goal Mode 执行约束

- 按 Phase 顺序执行，每完成一个 Phase 必须保持 repo 可测试、可 diff、可回滚，不允许一次性跨 Phase 大重构。
- 精度、性能不能回退；任何会改变数值误差、benchmark 结果、CUDA graph capture 行为或 fast path backend 的改动，都必须先用现有 oracle/benchmark 证明等价。
- 禁止用朴素 CUDA kernel 替代 cuBLASLt/CUTLASS/FA2 fast path。
- `PackedBackendRecipe` 通用路径必须先落地，再迁移 Pi0.5 `cuda_gemm.cc` 和相关 backend source。
- 如果需要改变 artifact ABI、VM ABI、packed func 参数顺序、权重命名、kernel 参数顺序或 measured performance，必须停止并报告，不要猜测性推进。
- 每阶段至少执行 `PYTHONPATH=python pytest` 和 `git diff --check`。
- GPU/runtime 阶段必须保留现有 Pi0.5 oracle、benchmark、CUDA graph capture 能力。

## 一句话评价

当前方案是“功能优先的有效实现”，不是“框架边界优雅的实现”。真正优雅的 devproc2 应该让 Pi0.5 作为 recipe 被框架消费，让 Pi0.5 CUDA fast path 作为模型自有 backend 被 artifact 声明和调用，而不是让框架目录长出 Pi0.5 业务文件，也不是让 runtime core 全局注册 `runtime.cuda.*` 业务函数。下一轮重构的主线应当是 recipe 化、artifact 通用化、权重包通用化、producer 工具外移、Pi0.5 CUDA backend 收归模型包，然后直接删除 `export/pi05.py`、`artifact/pi05.py` 和 `integrations/pi05/`。
