# Pi0.5 编译流程重构方案

## 目标

期望的 Pi0.5 产品化流程应该分成四段，每段都有清晰输入和产物：

```text
1. build devproc2 runtime
   input: devproc2 source + CMake options
   output: C++ runtime core library / tools

2. convert weight
   input: OpenPI / HF checkpoint
   output: devproc2 quantized weight package

3. devproc2 build model
   input: model recipe / frontend DSL + quantized weight package + resources + target config
   output: self-contained model artifact, including model-owned CUDA kernels and backend extensions

4. runtime inference
   input: runtime + model artifact + user tensors
   output: actions
```

这里的核心变化是：第三步应该叫 `build model`，而不是一组手工拼装的 `export artifact` 命令。`build` 内部可以继续做 frontend DSL、权重映射、IR、pass、VM codegen、kernel/backend packaging，但这些细节不应该暴露成用户必须理解的操作顺序。

## 本机固定资产

本方案后续实现和验收默认使用同一组本机资产，避免 checkpoint、tokenizer、dump 不同源导致误判：

```text
devproc2 root: /root/tw/devproc2
checkpoint: /root/tools/pi05_libero_base
checkpoint safetensors: /root/tools/pi05_libero_base/model.safetensors
openpi outputs: /root/tw/openpi/outputs/pi05_torch_infer
tokenizer: /root/tw/openpi/outputs/pi05_torch_infer/tokenizer.model
runtime raw oracle: /root/tw/devproc2/build/pi05_torch_dump_oracle
current weight package: /root/tw/devproc2/build/pi05_fp8.weights
```

精度和性能验收必须保证 weight package、OpenPI dump、tokenizer 和 raw oracle 同源。若 `/root/tw/openpi/outputs/pi05_torch_infer/metadata.json` 中的 `ckpt` 不是 `/root/tools/pi05_libero_base`，应先按 `docs/pi05/08_build_run_profile.md` 重新生成 dump，再做回归判断。

## 当前链路

迁移前实际可跑流程大致是：

```text
tools/pi05/convert_weights.py
  -> build/pi05_fp8.weights

python -m devproc2.export.cli --recipe devproc2.models.pi05.recipe:sample_tokens
  -> GraphBuilder
  -> InferStructInfo / DPSLowering / MemoryPlanning / VMCodegen
  -> EmitExecutablePass / EmitABIPass
  -> prepare_artifact(...)
  -> copy weights/resources
  -> compile cubin
  -> locate or build devproc2_pi05_cuda_backend
  -> build/pi05_fp8_sample_tokens_..._artifact

runtime C++ / bench
  -> ModelSession::LoadArtifact(...)
  -> load weights / cubin / packed backend / tokenizer
  -> VM Invoke
```

相关入口：

- `tools/pi05/convert_weights.py`：从 safetensors 生成 devproc2 weight package。
- `python/devproc2/export/cli.py`：通用 recipe export CLI。
- `python/devproc2/export/pipeline.py`：通用 compile / emit / artifact pipeline。
- `python/devproc2/artifact/builder.py`：安装 weights、resources、kernels、packed backends。
- `python/devproc2/models/pi05/recipe.py`：Pi0.5 entrypoint recipe、legacy compile/export helper、packed backend recipe。
- `runtime/CMakeLists.txt`：迁移前同时构建 runtime core 和 `devproc2_pi05_cuda_backend`。
- `runtime/src/vm.cc`：加载 artifact、权重、kernel table、packed backend table 和 tokenizer resource。

## 不符合预期的地方

### 1. 文档顺序仍像验证手册，不像产品构建流程

`docs/pi05/08_build_run_profile.md` 当前为了 profile 闭环，把 PyTorch dump、raw oracle、artifact export、runtime benchmark 放在同一条流程里。这对验证有用，但不应该成为主编译流程。

产品主线应该先说明：

```text
runtime build -> weight convert -> model build -> runtime infer
```

PyTorch dump、raw oracle、Nsight profile 应该放到验证章节，作为 build 产物的验收方式，而不是 build 的必要前置步骤。

### 2. `export.cli` 是可用入口，但不是用户期望的 `build` 接口

当前命令是：

```bash
python -m devproc2.export.cli \
  --recipe devproc2.models.pi05.recipe:sample_tokens \
  --artifact-dir ... \
  --weight-package-dir ... \
  --resource tokenizer=... \
  --option prefix_rows=968 \
  --option max_prompt_len=200 \
  --option num_views=3 \
  --backend-build-dir build/root-cuda \
  --option use_static_act_scales=true
```

这暴露了太多内部概念：

- 用户需要知道 `recipe:sample_tokens` 是一个 entrypoint object，而不是通过模型注册表选 `pi05/sample_tokens`。
- 用户需要手动传 shape option，且这些 option 和 `PI05Config`、weight package metadata 没有统一校验。
- `--backend-build-dir` 把 runtime/backend build 目录暴露到 model build 命令里。
- `--option use_static_act_scales=true` 依赖权重包中是否真的有 `act_scale.*`，当前没有 build-time 硬校验。
- 命令名字是 `export`，但它实际承担了 model compile、kernel compile、resource packaging 和 backend packaging。

### 3. runtime build 和 model-specific backend build 边界不清

迁移前 `runtime/CMakeLists.txt` 在 `DEVPROC2_WITH_CUDA=ON` 时直接定义 `devproc2_pi05_cuda_backend`。这说明旧 runtime CMake 既在编译 devproc2 framework runtime，也在编译 Pi0.5 模型专用 CUDA backend。

同时，`artifact.builder.prepare_artifact(...)` 在找不到 packed backend shared library 时，会尝试在候选 CMake build dir 里执行：

```text
cmake --build <build_dir> --target devproc2_pi05_cuda_backend
```

这和期望的“先把 devproc2 runtime core 编译好，再 build model”不一致。更准确的边界是：model build 可以编译 Pi0.5 专用 backend extension，但必须作为 `devproc2 build` 的显式、可配置子阶段，而不是 artifact packager 找不到 `.so` 后偷偷猜 CMake build dir。

### 4. Pi0.5 recipe 里仍保留两套 compile/export 路径

`python/devproc2/models/pi05/recipe.py` 现在同时有：

- 底部的 `EntrypointRecipe` / `CompileRecipe`，这是正确方向。
- 大量 `build_pi05_*`、`compile_pi05_*`、`emit_pi05_*`、`export_pi05_*_artifact` helper。
- `_compile_pi05_ir_module(...)`、`_emit_compile_result(...)` 这类和 `devproc2.export.pipeline` 重复的逻辑。
- 一个 Pi0.5 专用 CLI `main(...)`。

这会让维护者不清楚哪个才是 canonical build path。目标形态应该是：Pi0.5 只声明 recipe、config、resources、backend requirements；通用 `devproc2.build` 负责 compile/emit/package。

### 5. weight package 没有成为 build 的强约束输入

当前 `prepare_artifact(...)` 基本是复制整个 weight package：

```text
weights.bin
weights.index.json
weight_map.json
quantization.json
convert_report.json
```

它没有在 build 时系统性校验：

- executable ABI 里的 weight 参数是否都能从 weight package 绑定。
- weight dtype/layout/shape 是否匹配 graph variant。
- `use_static_act_scales=true` 时是否存在所有 `act_scale.*`。
- `fp8_layout` 是否和 target/backend 选择一致。
- weight package 的 source checkpoint、action horizon、num steps、shape config 是否和 model config 一致。

因此错误可能延迟到 runtime 或精度 profile 阶段才暴露。

### 6. 基础 converter 和性能 baseline 不完全一致

`tools/pi05/convert_weights.py` 当前能生成 FP8 weights、support BF16 tensors 和 precomputed styles，但基础 converter 不会自动生成当前性能包中的 `act_scale.*`。文档只能提示“缺少 act_scale 时不要打开 static act scale”。

这不符合“convert weight 后拿到量化 ckpt，下一步直接 build model”的预期。目标状态里，converter 产物必须显式声明它支持哪些 graph variant：

```text
precision=fp8
fp8_layout=nk
activation_scales=static | dynamic
entrypoints_supported=[sample_tokens, ...]
shape_profile=pi05_libero_base_3v200
```

model build 根据声明选择 graph variant，或者在选项不兼容时直接失败。

### 7. config 默认值重复，shape/profile 没有产品化命名

`python/devproc2/models/pi05/config.py` 已经有 `PI05Config`，但 `recipe.py` 仍重复维护大量默认常量和 option parsing。用户还需要从 torch dump 推导 `prefix_rows=968`、`max_prompt_len=200`、`num_views=3`。

更合适的是把这些固定成 profile：

```text
pi05_libero_base_3v200:
  num_views: 3
  max_prompt_len: 200
  prefix_rows: 968
  action_horizon: 50
  action_dim: 32
  num_steps: 10
```

用户选择 profile，build 负责把它落到 frontend input specs、weight checks、artifact metadata 和 runtime ABI。

### 8. artifact inspect / metadata 仍有旧格式痕迹

当前 artifact 主 manifest 是 `metadata/artifact.json`，但 `devproc_cli.py inspect` 仍优先找 artifact root 下的 `manifest.json`。这类工具不影响模型能跑，但会让“build 后检查产物”的体验显得不统一。

目标状态里 `devproc2 build` 的输出应该有稳定结构和稳定 inspect 命令：

```text
artifact/
  executable.vm
  abi.json
  metadata/artifact.json
  metadata/build.json
  metadata/weight_map.json
  metadata/quantization.json
  metadata/kernel_table.json
  metadata/packed_backend_table.json
  weights/weights.bin
  weights/weights.index.json
  kernels/*.cubin
  backends/*.so
  resources/tokenizer.model
```

## 目标设计

### 1. 明确三类构建产物

#### Runtime Build

runtime build 只负责 devproc2 runtime core 和通用工具，不编译 OpenPI0.5 专用 CUDA backend：

```bash
cmake -S . -B build/runtime-sm89 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/runtime-sm89 \
  --target devproc2_runtime \
  -j
```

Pi0.5 的 `pi05.cuda` backend 是模型专用能力，不是 runtime core。当前 target 定义已经迁到 `python/devproc2/models/pi05/cuda/CMakeLists.txt`，由 `devproc2 build` backend substage 或 model CUDA project 直接构建。

如果需要跑本机验证工具，可以在 runtime build 后额外构建 benchmark/test binary；这些 binary 不是 runtime core 的一部分。

#### Weight Package

weight convert 只做 checkpoint 到 devproc2 quantized package 的转换：

```bash
python -m tools.pi05.convert_weights \
  --checkpoint-dir /root/tools/pi05_libero_base \
  --output-dir build/weights/pi05_libero_base.fp8_sm89 \
  --hardware rtx_sm89 \
  --activation-scales static
```

短期如果 `--activation-scales static` 尚未实现，必须让 converter report 明确写：

```json
{
  "activation_scales": "missing",
  "supports_static_act_scales": false
}
```

build 看到用户选择 static graph 时应直接失败，而不是生成一个 runtime 才暴露问题的 artifact。

#### Model Artifact

model build 消费 weight package、model recipe、resources 和 target config：

```bash
python -m devproc2.build \
  --model pi05 \
  --entry sample_tokens \
  --weights build/weights/pi05_libero_base.fp8_sm89 \
  --out build/artifacts/pi05_libero_base.sample_tokens.sm89 \
  --profile pi05_libero_base_3v200 \
  --target cuda \
  --sm-arch 89 \
  --build-backends auto \
  --backend-cache-dir build/model-backends \
  --resource tokenizer=/root/tw/openpi/outputs/pi05_torch_infer/tokenizer.model
```

等价 Python API：

```python
from devproc2.build import build
from devproc2.models.pi05.recipe import pi05_recipe
from devproc2.models.pi05.config import PI05Config

summary = build(
    recipe=pi05_recipe,
    entry="sample_tokens",
    weights="build/weights/pi05_libero_base.fp8_sm89",
    artifact_dir="build/artifacts/pi05_libero_base.sample_tokens.sm89",
    config=PI05Config.for_profile("pi05_libero_base_3v200"),
    target="cuda",
    sm_arch=89,
    build_backends="auto",
    backend_cache_dir="build/model-backends",
    resources={
        "tokenizer": "/root/tw/openpi/outputs/pi05_torch_infer/tokenizer.model",
    },
)
```

### 2. `devproc2.build` 内部职责

`build` 内部应该是当前 `export.pipeline + artifact.builder` 的产品化封装，顺序固定为：

```text
resolve model recipe / entry / config
  -> build frontend DSL module
  -> GraphBuilder -> IRModule
  -> infer struct info
  -> bind and validate weight package
  -> select graph variant / lowering config
  -> DPS lowering / kernel lowering
  -> memory planning
  -> VM codegen
  -> emit executable.vm / abi.json / metadata
  -> compile or install cubins
  -> build or install model backend extensions
  -> copy resources
  -> write artifact manifest and build lock
```

注意几个边界：

- `build` 可以编译 model-owned `.cu` source 成 cubin，因为这是模型产物的一部分。
- `build` 可以编译 model-owned backend extension，例如 `pi05.cuda`，但这必须是 `--build-backends` 控制的显式 model build 子阶段。
- `build` 不应该编译 devproc2 runtime core，也不应该让 artifact builder 隐式猜测 CMake build dir 后触发编译。
- `build` 可以复用 backend cache 或用户提供的 backend library；缺失或不兼容时给出明确诊断。
- `build` 必须在 package 前校验 weight package 与 graph variant 兼容。
- `build` 产物应该自包含运行所需的 weights、kernels、backend `.so` 和 tokenizer resource。

### 3. 模型 export 声明的长期形态

Pi0.5 目录应该只保留模型侧声明：

```text
python/devproc2/models/pi05/
  config.py
  model.py             # model class re-export + typed export declaration
  graph/
  ops.py
  weights.py
  cuda/
```

长期不需要单独保留一个 2000 行 `recipe.py`。更合理的形态是：先把 `recipe.py` 里的 compile/export/package 行为全部抽到通用 `devproc2.build`，再把剩下的“模型 export 声明”合并进 `model.py`。`model.py` 作为 Pi0.5 的模型类型入口，声明这个模型有哪些可编译 entrypoint、默认 config/profile、资源和 backend 依赖。

目标形态可以是：

```python
PI05_MODEL = ModelExportSpec(
    model_id="openpi0.5",
    config_cls=PI05Config,
    entrypoints={
        "sample_tokens": EntrySpec(
            build_module=build_sample_tokens,
            input_specs=sample_tokens_input_specs,
            model_name="openpi0.5-sample-actions-tokens",
            backends=("pi05.cuda",),
            resources=("tokenizer",),
        ),
        "sample_precomputed_prefix": EntrySpec(...),
    },
)
```

如果短期仍复用现有 `CompileRecipe` / `EntrypointRecipe` 类型，也应该从 `model.py` 导出声明对象，例如 `PI05_MODEL` 或 `pi05_recipe`。`recipe.py` 只作为兼容 re-export 存在一段时间：

```python
# python/devproc2/models/pi05/recipe.py
from devproc2.models.pi05.model import PI05_MODEL as pi05_recipe
```

模型 export 声明应能表达：

- `input_specs(config)`
- `build_module(config)`
- `required_resources(config)`
- `required_weight_profile(config)`
- `packed_backends(config)`
- `artifact_metadata(config)`
- `default_profiles`

明确不允许合并进 `model.py` 的内容：

- `_compile_pi05_ir_module(...)`
- `emit_pi05_*`
- `export_pi05_*_artifact`
- `prepare_pi05_artifact(...)`
- Pi0.5 专用 CLI `main(...)`
- 直接 import compiler passes、artifact builder、VM codegen 的逻辑

这些都属于通用 `devproc2.build` 的职责。否则只是把 `recipe.py` 的 2000 行问题移动到 `model.py`。

### 4. Weight binding 和 validation

新增 build-time weight validation 层：

```text
Graph / ABI required params
  + model weight spec
  + weight package manifest
  -> resolved weight binding
```

需要校验：

- 所有 required weight 参数在 `weight_map.json` 中存在。
- dtype、shape、layout 和 quant spec 匹配。
- FP8 graph 必须有 FP8 weight 和 scale。
- static activation graph 必须有完整 `act_scale.*`。
- BF16/FP16 graph 不应意外绑定 FP8-only tensor。
- package `convert_report.json` 的 `action_horizon`、`num_steps`、`fp8_layout` 与 config 一致。

产物中写入：

```text
metadata/build.json
metadata/weight_binding.json
metadata/config.json
```

这样 runtime 加载前就能知道 artifact 是怎么编出来的。

### 5. Packed backend 的处理

当前 `pi05.cuda` packed backend 是性能路径的一部分，不能简单移除。它也不是通用算子库，不应该成为 devproc2 runtime core 的一部分。重构目标是把它从“runtime CMake 里顺手定义的目标”和“artifact builder 的隐式 CMake side effect”，改成 `devproc2 build` 中由模型 export 声明驱动的显式 backend extension build 子阶段。

推荐默认流程：

```text
devproc2 build --model pi05 --entry sample_tokens --build-backends auto
  -> read PI05_MODEL backend declaration
  -> build pi05.cuda for sm89 into backend cache
  -> copy libdevproc2_model_pi05_cuda_backend.so to artifact/backends/
  -> write metadata/packed_backend_table.json
  -> write metadata/backend_build.json
```

backend build modes：

```text
--build-backends auto    # 默认：cache miss 或 artifact 缺 backend 时自动编译
--build-backends never   # 禁止编译，只能使用 --backend-library-dir / 预编译 .so
--build-backends force   # 忽略 cache，强制重编
```

cache key 至少应包含：

```text
model_id
backend name
target arch / sm_arch
CUDA toolkit version
CUTLASS config / source hash
backend source file hash
compile definitions and link options
```

`--build-backends never` 下缺失时错误信息应该是：

```text
Pi0.5 entry sample_tokens requires model backend pi05.cuda for target sm89.
Backend build is disabled by --build-backends never.
Pass --build-backends auto, --build-backends force, or --backend-library-dir <dir>.
```

长期可以把 model backend extension 从 runtime CMake 中移出，但第一阶段不必阻塞在 CMake 目录大搬迁。关键要求是：编译动作由 `devproc2 build` 的 backend substage 管理，artifact builder 只消费已经完成的 backend build result 并打包。

### 6. CLI 命名和兼容策略

新增 canonical CLI：

```bash
python -m devproc2.build ...
python -m devproc2.inspect <artifact>
```

保留兼容 CLI：

```bash
python -m devproc2.export.cli ...
python -m devproc2.models.pi05.recipe ...
```

但文档中只推荐 `devproc2.build`。兼容 CLI 标记为 internal/debug，并在后续删除 legacy Pi0.5 export helpers。

### 7. 文档拆分

建议把文档分成两条线：

```text
docs/pi05/08_build_run_profile.md
  保留验证和 profile 手册：torch dump、raw oracle、bench、nsys。

docs/pi05/13_product_build_quickstart.md
  新增产品构建手册：runtime build、weight convert、devproc2 build、runtime infer。
```

本方案落地前，`08_build_run_profile.md` 可以继续存在；等 `devproc2.build` 实现后，再把它的第 4/5 节改成调用新 build 接口。

### 8. Codex Goal 模式执行注意

后续如果用 Codex goal 模式实现本文任务，建议把 goal 写成分阶段可验收任务，而不是一次性“大改完”。执行注意：

- 先读并保护现有工作树，不能 revert 用户已有修改；每个阶段只触碰必要文件。
- Phase 1 先引入 `devproc2.build` 的薄封装，复用现有 `export.pipeline`，确保旧 `export.cli` 仍可跑。
- backend extension build 必须是 `devproc2.build` 的显式子阶段，禁止继续放在 `artifact.builder.prepare_artifact(...)` 的隐式 fallback 中。
- `model.py` 合并 recipe 时，只搬 typed model export declaration；不要把 compile/export/package helper 搬进去。
- 每个阶段都要保留兼容路径，直到新路径通过同权重精度和性能验收。
- 任何 CMake/backend/cache 设计都要写入 artifact metadata，保证之后能复现构建条件。
- 不要把 PyTorch/OpenPI oracle 生产逻辑塞进 model build；oracle 只用于验证。
- 不允许以降低精度、关闭 static act-scale、绕开 CUTLASS/FA2 性能路径的方式“通过”重构。

## 迁移计划

### Phase 0：固化现状，不改行为

- 保留当前 `export.cli` 和 Pi0.5 recipe。
- 在 docs 中明确生产主线和验证主线的区别。
- 给 artifact builder 增加“不要隐式 build backend”的开关或未来默认策略说明。
- 更新 `devproc_cli.py inspect`，支持 `metadata/artifact.json`。

验收：

- 当前 `docs/pi05/08_build_run_profile.md` 仍能跑。
- 新文档能指导后续实现，不要求代码行为变化。

### Phase 1：引入 `devproc2.build` API

新增通用 build API，但内部复用现有 `export.pipeline.export_artifact(...)`：

```python
build(
    model,
    entry,
    weights,
    artifact_dir,
    config,
    resources,
    target,
    sm_arch,
    build_backends="auto",
    backend_cache_dir="build/model-backends",
)
```

同时新增 CLI：

```bash
python -m devproc2.build --model pi05 --entry sample_tokens ...
```

验收：

- `devproc2.build` 和旧 `export.cli` 生成等价 artifact。
- Pi0.5 文档主线切到 `devproc2.build`。
- `build` 可以通过 `--build-backends auto` 编译并打包 `pi05.cuda`；`--build-backends never` 下缺失 backend 时明确报错。
- `artifact.builder` 不再负责隐式编译 model backend extension。

### Phase 2：权重包成为 build 强约束

- 扩展 `WeightPackageWriter` manifest。
- `convert_weights.py` 写入 activation scale 能力、shape profile、source checkpoint、target hardware。
- build-time 校验 required weights、FP8 layout、static act scales。
- 输出 `metadata/weight_binding.json` 和 `metadata/config.json`。

验收：

- 用缺少 `act_scale.*` 的 weight package 开启 static graph 时，build 阶段失败。
- 用动态 act-scale graph 可以继续 build smoke artifact。
- weight package 和 graph ABI 不匹配时，错误指向具体 missing/mismatched tensor。

### Phase 3：抽走 recipe 行为并合并模型 export 声明

- 删除或 deprecate `compile_pi05_*`、`emit_pi05_*`、`export_pi05_*_artifact`。
- 把 `_compile_pi05_ir_module(...)`、`_emit_compile_result(...)`、`prepare_pi05_artifact(...)` 和 Pi0.5 专用 CLI 行为迁到 generic `devproc2.build` 或删除。
- 把剩余的 entrypoint builders、input specs、backend/resource declarations 收敛成 typed model export declaration，并合并到 `model.py`。
- 短期保留 `recipe.py` 作为兼容 re-export；新文档、测试和 CLI 不再直接依赖 `devproc2.models.pi05.recipe`。
- 默认 shape/profile 从 `PI05Config` 读取，不再在 `recipe.py` 重复维护常量。

验收：

- Pi0.5 artifact 只通过 generic `devproc2.build` 生成。
- tests 不再调用 Pi0.5 legacy export helpers。
- `model.py` 只包含模型 re-export 和 typed export declaration，不 import compiler passes、artifact builder 或 VM codegen。
- `recipe.py` 要么删除，要么只剩兼容 re-export。

### Phase 4：整理 runtime/backend 边界

- 把 `devproc2_pi05_cuda_backend` 语义标注为 model backend extension。
- runtime build 只产出 devproc2 runtime core，不默认构建 Pi0.5 backend。
- `devproc2.build` 的 backend substage 负责构建或复用 Pi0.5 backend extension。
- Pi0.5 backend CMake target 定义位于 `python/devproc2/models/pi05/cuda`，
  不再由 runtime CMake tree 承载。
- artifact builder 只安装 backend substage 的 build result，不负责 CMake build。
- runtime core 保留 `ModelSession::LoadArtifact`、packed backend dlopen、kernel table load、WeightStore；不拥有 Pi0.5 专用默认路径。

验收：

- runtime core 可独立编译。
- Pi0.5 backend 可由 `devproc2 build --build-backends auto` 编译、缓存并打包进 artifact。
- `test_pi05_cuda_gemm` 由 Pi0.5 model CUDA project 构建，不属于 runtime core test tree。
- artifact 在另一台相同 ABI/SM 的机器上只依赖 runtime core 和 artifact 内容即可加载。

## 目标命令形态

最终用户应该只需要：

```bash
# 1. Build runtime once.
cmake -S . -B build/runtime-sm89 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/runtime-sm89 \
  --target devproc2_runtime bench_pi05_denoise \
  -j

# 2. Convert checkpoint to a quantized devproc2 weight package.
python -m tools.pi05.convert_weights \
  --checkpoint-dir /root/tools/pi05_libero_base \
  --output-dir build/weights/pi05_libero_base.fp8_sm89 \
  --hardware rtx_sm89

# 3. Build model artifact.
python -m devproc2.build \
  --model pi05 \
  --entry sample_tokens \
  --weights build/weights/pi05_libero_base.fp8_sm89 \
  --out build/artifacts/pi05_libero_base.sample_tokens.sm89 \
  --profile pi05_libero_base_3v200 \
  --target cuda \
  --sm-arch 89 \
  --build-backends auto \
  --backend-cache-dir build/model-backends \
  --resource tokenizer=/root/tw/openpi/outputs/pi05_torch_infer/tokenizer.model

# 4. Inspect and run.
python -m devproc2.inspect build/artifacts/pi05_libero_base.sample_tokens.sm89
build/runtime-sm89/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir build/artifacts/pi05_libero_base.sample_tokens.sm89 \
  --oracle-dir build/pi05_torch_dump_oracle/bf16_example0/raw \
  --max-prompt-len 200 \
  --prefix-valid-rows 895 \
  --num-views 3
```

第 4 步里的 `bench_pi05_denoise` 仍是验证工具。真正部署侧应该使用 `ModelSession::LoadArtifact(...)` 或后续模型级 C++ API。

## 验收标准

重构完成后，应该满足：

- 用户文档主流程是 `runtime build -> convert weight -> devproc2 build -> runtime infer`。
- `devproc2 build` 是唯一推荐的模型编译入口。
- Pi0.5 recipe 不再复制通用 compiler/export/artifact pipeline。
- devproc2 runtime core 不包含 Pi0.5 专用 backend。
- model-owned backend extension 由 `devproc2 build` 的显式 backend substage 构建或复用，不由 artifact builder 隐式触发。
- weight package 在 build 阶段完成 shape/dtype/layout/scale 校验。
- static act-scale graph 不能用缺少 `act_scale.*` 的 package 编译成功。
- artifact 自包含 weights、kernels、packed backend、resources 和 build metadata。
- runtime 只加载 artifact，不知道 build 流程，也不依赖本机默认 checkpoint/tokenizer 路径。
- profile/oracle 文档只作为验证补充，不再定义生产编译流程。
- 同权重精度不能回退：使用 `/root/tools/pi05_libero_base`、`/root/tw/openpi/outputs/pi05_torch_infer`、`/root/tw/devproc2/build/pi05_torch_dump_oracle` 验收，same-weight example0 `final_abs_max/final_abs_mean` 不应劣于当前 `0.021 / 0.002` 量级；10-example profile 的 `mean_abs_mean`、`max_abs_mean` 不应劣于当前 `0.0054 / 0.018` 量级。
- 性能不能回退：3-view / P=968 / `sample_tokens` / CUDA Graph 路径应维持当前 `mean_10step_ms ~= 33.7ms` 水平，默认允许不超过 5% 的测量波动；超过该范围必须定位并说明。
- 基础 runtime tests 必须通过：`test_cuda_graph`、`test_pi05_artifact_load`、`test_pi05_kernel_launch`。
- Pi0.5 model backend test 必须通过：`test_pi05_cuda_gemm`。
