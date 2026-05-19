# OpenPI0.5 在 devproc2 上的编译运行全流程

本文是一份技术分享稿，目标不是只给一串命令，而是把 OpenPI0.5 在 devproc2 中从 checkpoint 到 C++ runtime 跑出 actions 的完整链路讲清楚。当前产品化主线已经收敛为：

```text
build runtime -> convert weights -> build model artifact -> runtime inference
```

验证和性能分析还有两条辅助线：

```text
OpenPI PyTorch dump -> raw oracle -> actions 对点
runtime benchmark -> CUDA Graph / Nsight profile
```

这三条线要分开理解。模型构建不依赖 PyTorch oracle；oracle 只是用来确认 artifact 的数值和性能。

## 1. 一句话架构

devproc2 把 OpenPI0.5 拆成两类产物：

- Runtime：通用 C++ VM、WeightStore、CUDA kernel loader、packed backend loader、tokenizer packed func、CUDA Graph 封装。
- Model artifact：OpenPI0.5 的 VM bytecode、ABI、权重包、CUDA cubin、自定义 packed backend `.so`、tokenizer resource。

运行时只需要 runtime core 加上自包含 artifact：

```text
user tensors
  -> ModelSession::LoadArtifact(artifact)
  -> VMState::Invoke("main", args)
  -> builtin / CUDA cubin / pi05.cuda packed funcs
  -> actions [50, 32] float32
```

## 2. 当前固定资产

本机默认使用下面这组同源资产：

```bash
cd /root/tw/devproc2

export DEVPROC2_ROOT=/root/tw/devproc2
export OPENPI_ROOT=/root/tw/openpi
export OPENPI_PY=$OPENPI_ROOT/.venv/bin/python

export PI05_CKPT=/root/tools/pi05_libero_base
export PI05_TORCH_DUMP=/root/tw/openpi/outputs/pi05_torch_infer
export PI05_TOKENIZER=$PI05_TORCH_DUMP/tokenizer.model
export PI05_WEIGHT_PKG=$DEVPROC2_ROOT/build/pi05_fp8.weights
export PI05_RAW_ORACLE=$DEVPROC2_ROOT/build/pi05_torch_dump_oracle

export PYTHONPATH=$DEVPROC2_ROOT/python:${PYTHONPATH:-}
```

关键约束：

- checkpoint、weight package、tokenizer、PyTorch dump、raw oracle 必须同源，否则精度对点没有意义。
- 不要在 `/root/tw/openpi` 中执行 `uv sync`、`uv run` 或重装依赖。需要生成 oracle 时只使用 `/root/tw/openpi/.venv/bin/python`。
- 当前主性能目标是 RTX 4090 / SM89，FP8 layout 为 `nk`。

快速检查：

```bash
test -x "$OPENPI_PY"
test -f "$PI05_CKPT/model.safetensors"
test -d "$PI05_WEIGHT_PKG"
test -f "$PI05_TOKENIZER"
test -d "$PI05_RAW_ORACLE"
```

## 3. 推理图从哪里来

OpenPI0.5 的 PyTorch `sample_actions` 可以理解为：

```text
observation
  -> image / text / state preprocessing
  -> embed_prefix
       images -> SigLIP vision tower -> visual prefix embeddings
       prompt tokens -> language embedding
       concat visual/text prefix embeddings
  -> PaliGemma prefix transformer -> prefix KV cache
  -> 10-step Euler denoise loop
       embed_suffix(state, x_t, timestep)
       Gemma expert attention(prefix KV + suffix)
       action_out_proj -> v_t
       x_{t+1} = x_t + dt * v_t
  -> actions
```

devproc2 当前产品入口是 `sample_tokens`，也就是 tokenizer 前处理暂时仍在图外，运行时输入已经是 `images_u8 + token_ids + RoPE`：

```text
noise_f32                 [50, 32] float32
images_u8                 [3, 224, 224, 3] uint8
token_ids                 [200] int32
prefix_valid_rows          int64
prefix_rope_interleaved   [968, 256] bfloat16
suffix_rope_interleaved   [50, 256] bfloat16
```

输出是 `[50, 32]` float32 actions。

模型声明在 `python/devproc2/models/pi05/model.py`：

- `PI05_MODEL_ID = "openpi0.5"`
- 产品 entrypoint：`sample_tokens`
- 默认 profile：`pi05_libero_base_3v200`
- model-owned backend：`pi05.cuda`
- required packed funcs：FP8/BF16 GEMM 和 FA2 attention

形状默认值在 `python/devproc2/models/pi05/config.py`，核心参数是：

```text
num_views=3
image_size=224
max_prompt_len=200
prefix_rows=968
action_horizon=50
action_dim=32
num_steps=10
num_layers=18
head_dim=256
```

## 4. 阶段一：Build Runtime

runtime build 只负责编译 devproc2 通用 C++ runtime 和验证工具，不编译 OpenPI0.5 专用 CUDA backend。Pi0.5 backend 是 model artifact 的一部分，由后面的 `devproc2 build` 处理。

```bash
cmake -S . -B build/runtime-sm89 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/runtime-sm89 \
  --target devproc2_runtime bench_pi05_denoise test_cuda_graph \
  -j
```

这一阶段的关键产物：

```text
build/runtime-sm89/runtime/libdevproc2_runtime.a
build/runtime-sm89/runtime/tests/bench_pi05_denoise
build/runtime-sm89/runtime/tests/test_cuda_graph
```

runtime core 包含：

- `Executable::Load`：加载 `executable.vm`、`abi.json`、kernel table、packed backend table、weights。
- `WeightStore`：按 ABI 参数名自动绑定 artifact 内权重。
- `PackedFuncRegistry`：注册 builtin、tokenizer、model backend packed funcs。
- `CUDAKernelRegistry`：加载 artifact 里的 cubin 并按 kernel name 调度。
- `ModelSession`：部署侧薄封装，入口是 `LoadArtifact` 和 `Invoke`。

基础 smoke：

```bash
ctest --test-dir build/runtime-sm89/runtime \
  -R 'test_cuda_graph|test_pi05_artifact_load|test_pi05_kernel_launch' \
  --output-on-failure
```

## 5. 阶段二：Convert Weights

weight convert 把 OpenPI/HF checkpoint 转成 devproc2 weight package。它不做模型编译，只做权重抽取、重命名、融合、量化和 manifest 生成。

```bash
python -m tools.pi05.convert_weights \
  --checkpoint-dir "$PI05_CKPT" \
  --output-dir build/weights/pi05_libero_base.fp8_sm89 \
  --hardware rtx_sm89 \
  --activation-scales static \
  --shape-profile pi05_libero_base_3v200
```

输入：

```text
/root/tools/pi05_libero_base/model.safetensors
```

输出目录结构：

```text
build/weights/pi05_libero_base.fp8_sm89/
  manifest.json
  weights.bin
  weights.index.json
  weight_map.json
  quantization.json
  convert_report.json
```

每个文件的职责：

- `weights.bin`：连续二进制权重数据。
- `weights.index.json`：每个 tensor 在 `weights.bin` 中的 offset、size、dtype、shape。
- `weight_map.json`：graph ABI 参数名到权重条目的映射信息。
- `quantization.json`：FP8 quant/fusion 描述。
- `manifest.json`：package 格式、模型名、precision、文件名。
- `convert_report.json`：来源 checkpoint、target hardware、shape profile、FP8 layout、activation scale 能力。

FP8 路径的关键规则：

- RTX 4090/SM89 默认使用 `fp8_layout=nk`。
- FP8 权重名形如 `fp8.<logical_name>.weight`，scale 名形如 `fp8.<logical_name>.scale`。
- 静态 activation scale 名形如 `act_scale.<logical_name>`。
- `--activation-scales static` 只有在 package 里真的有完整 `act_scale.*` 时才是性能 baseline。

检查权重包：

```bash
python - <<'PY'
import json
from pathlib import Path

pkg = Path("build/weights/pi05_libero_base.fp8_sm89")
manifest = json.loads((pkg / "manifest.json").read_text())
report = json.loads((pkg / "convert_report.json").read_text())
wm = json.loads((pkg / "weight_map.json").read_text())["weights"]

print("model:", manifest.get("model"))
print("precision:", manifest.get("precision"))
print("source:", report.get("source"))
print("fp8_layout:", report.get("fp8_layout"))
print("activation_scales:", report.get("activation_scales"))
print("supports_static_act_scales:", report.get("supports_static_act_scales"))
print("num_weights:", len(wm))
print("act_scale entries:", sum(w["name"].startswith("act_scale.") for w in wm))
PY
```

如果基础 converter 生成的 package 没有 `act_scale.*`，`devproc2 build --activation-scales static` 会在 build 阶段失败。可以用 `--activation-scales dynamic` 做 smoke test，但它不代表当前静态 scale 性能路径。

## 6. 阶段三：Build Model Artifact

`devproc2 build` 是产品化模型编译入口。它把 Pi0.5 的 Python DSL model、weight package、target config、tokenizer resource 和 CUDA backend 打包成自包含 artifact。

推荐命令：

```bash
python -m devproc2.build \
  --model pi05 \
  --entry sample_tokens \
  --weights "$PI05_WEIGHT_PKG" \
  --out build/artifacts/pi05_libero_base.sample_tokens.sm89 \
  --profile pi05_libero_base_3v200 \
  --target cuda \
  --sm-arch 89 \
  --build-backends auto \
  --backend-cache-dir build/model-backends \
  --resource tokenizer="$PI05_TOKENIZER" \
  --activation-scales static
```

`--build-backends` 的语义：

```text
auto    cache miss 时自动构建 pi05.cuda backend，并复制进 artifact
never   禁止构建，只能从 --backend-library-dir 或环境变量取预编译 .so
force   忽略 cache，强制重建 backend
```

如果只想复用一个已经编好的 backend：

```bash
python -m devproc2.build \
  --model pi05 \
  --entry sample_tokens \
  --weights "$PI05_WEIGHT_PKG" \
  --out build/artifacts/pi05_libero_base.sample_tokens.sm89 \
  --profile pi05_libero_base_3v200 \
  --target cuda \
  --sm-arch 89 \
  --build-backends never \
  --backend-library-dir build/pi05-cuda-backend-sm89 \
  --resource tokenizer="$PI05_TOKENIZER" \
  --activation-scales static
```

`--backend-library-dir` 必须指向包含 `libdevproc2_pi05_cuda_backend.so` 的目录。全新构建时优先用 `--build-backends auto`，需要手工调试 backend 时再用 6.2 节的独立 CMake project 生成该 `.so`。

### 6.1 build 内部发生了什么

`python/devproc2/build.py` 内部顺序是：

```text
resolve model / entry / profile
  -> PI05Config.for_profile(...)
  -> entrypoint.build_module(options)
  -> entrypoint.input_specs(options)
  -> GraphBuilder 捕获 forward_fast
  -> InferStructInfo
  -> DPSLowering
  -> MemoryPlanning
  -> LowerTensorCreateToAlloc
  -> VMCodegen
  -> EmitExecutablePass / EmitABIPass
  -> validate_weight_package
  -> build or locate pi05.cuda backend
  -> prepare_artifact
  -> write metadata
```

几个关键点：

- `compile_mode=fast` 会选择各 module 的 `forward_fast`，图中会直接出现 Pi0.5 CUDA kernel 和 packed func 调用。
- `DPSLowering` 把 tensor-return op 改成 destination-passing style，即 `TensorCreateOp + CallDPSOp`。
- `MemoryPlanning` 计算中间 tensor storage 复用计划，写入 `metadata/storage_plan.json`。
- `VMCodegen` 生成 `executable.vm`，里面是 VM function table 和 bytecode。
- `EmitABIPass` 生成 `abi.json`，描述用户输入、权重参数、输出、required packed funcs。
- `validate_weight_package` 会检查 ABI 需要的权重在 weight package 里是否存在，shape/dtype/layout 是否匹配，static activation graph 是否有完整 `act_scale.*`。
- `prepare_artifact` 复制权重、资源、编译 cubin、安装 packed backend，并写 `metadata/artifact.json`。

### 6.2 Pi0.5 CUDA backend 怎么构建

`sample_tokens` 需要 `pi05.cuda` packed backend。这个 backend 不属于 runtime core，它在模型目录下单独维护：

```text
python/devproc2/models/pi05/cuda/CMakeLists.txt
python/devproc2/models/pi05/cuda/backends/fp8_gemm/*
python/devproc2/models/pi05/cuda/fa2/*
```

`devproc2 build --build-backends auto` 会配置这个 CMake project，构建 `devproc2_pi05_cuda_backend`，然后把 `.so` 放进：

```text
artifact/backends/pi05_cuda.so
metadata/backend_build.json
metadata/packed_backend_table.json
```

backend 提供的 packed funcs 包括：

```text
pi05.cuda.fp8_nn_bf16
pi05.cuda.fp8_nt_bf16
pi05.cuda.fp8_nn_bf16_accum
pi05.cuda.fp8_nt_bf16_accum
pi05.cuda.bf16_nn_bf16
pi05.cuda.bf16_nt_bf16
pi05.cuda.fa2_bf16
pi05.cuda.fa2_bf16_batched
```

需要单独调试 backend 时：

```bash
cmake -S python/devproc2/models/pi05/cuda \
  -B build/pi05-cuda-backend-sm89 \
  -DDEVPROC2_REPO_ROOT=$PWD \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_CUTLASS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/pi05-cuda-backend-sm89 \
  --target test_pi05_cuda_gemm -j

ctest --test-dir build/pi05-cuda-backend-sm89 \
  -R test_pi05_cuda_gemm --output-on-failure
```

### 6.3 artifact 长什么样

成功后 artifact 至少包含：

```text
build/artifacts/pi05_libero_base.sample_tokens.sm89/
  executable.vm
  abi.json
  weights/
    weights.bin
    weights.index.json
  kernels/
    *.cubin
  backends/
    pi05_cuda.so
  resources/
    tokenizer.model
  metadata/
    artifact.json
    build.json
    config.json
    weight_binding.json
    backend_build.json
    kernel_table.json
    packed_func_table.json
    packed_backend_table.json
    storage_plan.json
    weight_map.json
    quantization.json
    convert_report.json
```

检查 artifact：

```bash
python -m devproc2.inspect build/artifacts/pi05_libero_base.sample_tokens.sm89
```

重点看：

- `target_arch` 是否是 `sm89`。
- ABI 前 6 个输入是否是 `noise_f32/images_u8/token_ids/prefix_valid_rows/prefix_rope_interleaved/suffix_rope_interleaved`。
- `required_packed_funcs` 是否包含 `pi05.cuda.*`。
- `metadata/weight_binding.json` 中 `required_weights == bound_weights`。
- `metadata/artifact.json` 中是否有 `weights`、`kernels`、`packed_backends`、`resources`。

## 7. 阶段四：Runtime Inference

部署侧的核心 C++ API 是：

```cpp
auto session = devproc2::ModelSession::LoadArtifact(artifact_dir);
auto result = session.Invoke("main", {
    VMValue::ObjRef(noise_f32),
    VMValue::ObjRef(images_u8),
    VMValue::ObjRef(token_ids),
    VMValue::Int(prefix_valid_rows),
    VMValue::ObjRef(prefix_rope_interleaved),
    VMValue::ObjRef(suffix_rope_interleaved),
});
```

runtime 加载 artifact 时会做这些事：

```text
read executable.vm
read abi.json
load weights/weights.bin via weights.index.json
load kernels/*.cubin and register CUDA symbols
dlopen backends/pi05_cuda.so
call backend register_symbol
register pi05.cuda packed funcs
load tokenizer resource metadata
validate required packed funcs
```

VM 执行时按 function table 调度三类外部调用：

- builtin：runtime 内置函数。
- kernel：artifact 中 cubin 对应的 CUDA kernel。
- packed_func：`pi05.cuda` backend 中的 GEMM/FA2 等 opaque 函数。

## 8. 用 bench 跑通 example0

`bench_pi05_denoise` 是当前最方便的 C++ 验证入口。它从 raw oracle 读输入，加载 artifact，调用 `main`，最后把输出 actions 和 raw target 做差。

```bash
export ARTIFACT=build/artifacts/pi05_libero_base.sample_tokens.sm89
export META=$PI05_RAW_ORACLE/bf16_example0/metadata.json
export PREFIX_VALID_ROWS=$(python - <<'PY'
import json, os
print(json.load(open(os.environ["META"]))["prefix_valid_rows"])
PY
)

build/runtime-sm89/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir "$ARTIFACT" \
  --oracle-dir "$PI05_RAW_ORACLE/bf16_example0/raw" \
  --max-prompt-len 200 \
  --prefix-valid-rows "$PREFIX_VALID_ROWS" \
  --num-views 3
```

输出示例格式：

```text
pi05_denoise_bench iters=50 entry=sample_tokens mode=cuda_graph warmup_ms=...
mean_10step_ms=... mean_step_ms=... final_abs_max=... final_abs_mean=...
```

字段解读：

- `mode=cuda_graph`：已完成 CUDA Graph capture/replay，这是部署路径的主要计时方式。
- `mean_10step_ms`：一次完整 `sample_tokens` 的平均耗时，包括 vision encoder、language embedding、prefix KV materialization 和 10-step denoise。
- `mean_step_ms`：`mean_10step_ms / 10`，只是归一化展示，不代表真的只跑单 step。
- `final_abs_max/final_abs_mean`：runtime output vs raw oracle target。回归判断应重点看 `final_abs_mean` 和同源条件，不能只看单点 `abs_max`。

当前同权重 P=968 / 3-view batch 的参考观测值见 `08_build_run_profile.md`：`mean_10step_ms` 约 `33.7ms`，`final_abs_mean` 均值约 `0.0054`。

## 9. raw oracle 是什么

raw oracle 是 benchmark 直接读取的二进制输入和目标输出。`sample_tokens` 至少需要：

```text
images_u8.bin
token_ids_i32.bin
prefix_rope_interleaved_bf16.bin
rope_interleaved_bf16.bin
step_000/actions_f32.bin
step_009/x_next_f32.bin
```

benchmark 的通用 loader 还会读：

```text
prefix_k_cache_bf16.bin
prefix_v_cache_bf16.bin
prefix_embs_bf16.bin
```

对 `sample_tokens` 来说后三个不是图输入，可以是零占位。不要把这份只为 `sample_tokens` 生成的 raw 拿去跑 `step`、`loop` 或 precomputed-prefix 入口。

从 OpenPI dump 生成 raw 的完整脚本在 `08_build_run_profile.md`。当前本机 summary：

```text
prefix_rows = 968
prefix_valid_rows = [895, 900, 906, 900, 904, 894, 902, 906, 898, 905]
token_valid_len = [127, 132, 138, 132, 136, 126, 134, 138, 130, 137]
```

## 10. 生成 PyTorch oracle 的位置

PyTorch oracle 不属于构建必需步骤，只用于验证同权重 actions 对点。重新生成时使用 openpi 已有 venv：

```bash
OMP_NUM_THREADS=1 \
PYTHONPATH=$OPENPI_ROOT:$OPENPI_ROOT/src \
"$OPENPI_PY" "$OPENPI_ROOT/scripts/dump_pi05_torch_infer.py" \
  --ckpt "$PI05_CKPT" \
  --out "$PI05_TORCH_DUMP" \
  --device cuda \
  --num-examples 10 \
  --num-steps 10
```

检查 dump 的 checkpoint 来源：

```bash
python - <<'PY'
import json
from pathlib import Path

dump_meta = json.loads(Path("/root/tw/openpi/outputs/pi05_torch_infer/metadata.json").read_text())
report = json.loads(Path("build/pi05_fp8.weights/convert_report.json").read_text())
print("torch dump ckpt:", dump_meta.get("ckpt"))
print("weight pkg source:", report.get("source", {}).get("path"))
PY
```

有效对点时应看到两者都来自 `/root/tools/pi05_libero_base`。

## 11. Nsight profile 流程

先用 Nsight Systems 看整体耗时分布，再用 Nsight Compute 钻单 kernel。不要一开始就上 ncu。

```bash
mkdir -p build/pi05_profiles

export ARTIFACT=build/artifacts/pi05_libero_base.sample_tokens.sm89
export META=$PI05_RAW_ORACLE/bf16_example0/metadata.json
export PREFIX_VALID_ROWS=$(python - <<'PY'
import json, os
print(json.load(open(os.environ["META"]))["prefix_valid_rows"])
PY
)

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  -o build/pi05_profiles/sample_tokens_p968 \
  build/runtime-sm89/runtime/tests/bench_pi05_denoise 20 \
    --entry-kind sample_tokens \
    --artifact-dir "$ARTIFACT" \
    --oracle-dir "$PI05_RAW_ORACLE/bf16_example0/raw" \
    --max-prompt-len 200 \
    --prefix-valid-rows "$PREFIX_VALID_ROWS" \
    --num-views 3
```

profile 时关注：

- 是否走 CUDA Graph replay。
- GEMM/FA2 是否来自 `pi05.cuda` backend，而不是 fallback。
- 小 kernel 数量是否异常增多。
- vision/prefix encoder FP8 GEMM 是否仍是主瓶颈。

## 12. 常见问题

### static activation scale build 失败

错误含义通常是 weight package 没有完整 `act_scale.*`，但 build 选择了静态 activation graph。处理方式：

- 性能 baseline：换用带 `act_scale.*` 的权重包。
- smoke test：改用 `--activation-scales dynamic`。

### backend 找不到

典型错误是缺少 `pi05.cuda` compiled packed backend。处理方式：

```bash
python -m devproc2.build ... --build-backends auto
```

或者指定已编译 backend：

```bash
python -m devproc2.build ... \
  --build-backends never \
  --backend-library-dir build/pi05-cuda-backend-sm89
```

也可以用环境变量：

```bash
export DEVPROC2_PACKED_BACKEND_PI05_CUDA_SO=/path/to/libdevproc2_pi05_cuda_backend.so
```

### CUTLASS 找不到

Pi0.5 CUDA backend 默认打开 `DEVPROC2_WITH_CUTLASS=ON`。如果 CMake 找不到 CUTLASS，需要设置：

```bash
cmake -S python/devproc2/models/pi05/cuda \
  -B build/pi05-cuda-backend-sm89 \
  -DDEVPROC2_REPO_ROOT=$PWD \
  -DDEVPROC2_CUTLASS_ROOT=/path/to/cutlass \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_CUTLASS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

### benchmark 缺 raw 文件

确认 `--oracle-dir` 指向的是 `.../bf16_exampleN/raw`，不是 example 目录本身：

```bash
ls "$PI05_RAW_ORACLE/bf16_example0/raw"
```

`sample_tokens` 至少要有 image、token、RoPE、initial noise、target actions。

### 精度突然变差

先排查同源性：

- PyTorch dump 的 `metadata.json` 是否指向同一个 checkpoint。
- weight package 的 `convert_report.json` 是否指向同一个 `model.safetensors`。
- `tokenizer.model` 是否来自同一个 dump。
- `prefix_valid_rows` 是否读取了当前 example 的 metadata，而不是写死。
- artifact 的 `prefix_rows/max_prompt_len/num_views` 是否与 raw 匹配。

## 13. 分享时可以强调的边界

这个流程里最容易混淆的是“编译 runtime”和“编译模型”：

- runtime build 是 C++ 框架能力，目标是 `devproc2_runtime` 和验证 binary。
- model build 是 OpenPI0.5 产物生成，目标是自包含 artifact。
- Pi0.5 CUDA backend 是模型专用 extension，不是 runtime core。
- PyTorch oracle 是验证输入，不是 build 输入。
- tokenizer 目前作为 artifact resource 和 runtime packed func 使用；完整 prompt/state preprocessing 还没有并入 VM graph。

## 14. 最短闭环

如果现场只演示最短路径，用这四组命令：

```bash
cd /root/tw/devproc2
export PYTHONPATH=$PWD/python:${PYTHONPATH:-}
export PI05_CKPT=/root/tools/pi05_libero_base
export PI05_TOKENIZER=/root/tw/openpi/outputs/pi05_torch_infer/tokenizer.model
export PI05_WEIGHT_PKG=$PWD/build/pi05_fp8.weights
export PI05_RAW_ORACLE=$PWD/build/pi05_torch_dump_oracle
```

```bash
cmake -S . -B build/runtime-sm89 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/runtime-sm89 --target devproc2_runtime bench_pi05_denoise -j
```

```bash
python -m devproc2.build \
  --model pi05 \
  --entry sample_tokens \
  --weights "$PI05_WEIGHT_PKG" \
  --out build/artifacts/pi05_libero_base.sample_tokens.sm89 \
  --profile pi05_libero_base_3v200 \
  --target cuda \
  --sm-arch 89 \
  --build-backends auto \
  --backend-cache-dir build/model-backends \
  --resource tokenizer="$PI05_TOKENIZER" \
  --activation-scales static
```

```bash
export ARTIFACT=build/artifacts/pi05_libero_base.sample_tokens.sm89
export PREFIX_VALID_ROWS=$(python - <<'PY'
import json
print(json.load(open("build/pi05_torch_dump_oracle/bf16_example0/metadata.json"))["prefix_valid_rows"])
PY
)

build/runtime-sm89/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir "$ARTIFACT" \
  --oracle-dir "$PI05_RAW_ORACLE/bf16_example0/raw" \
  --max-prompt-len 200 \
  --prefix-valid-rows "$PREFIX_VALID_ROWS" \
  --num-views 3
```

如果这条线跑通，就说明：

```text
runtime 可加载 artifact
weights 可绑定
kernels 可注册
pi05.cuda backend 可 dlopen 并注册 packed funcs
VM main 可执行
CUDA Graph 可 capture/replay
输出 actions 可和 PyTorch raw target 对点
```
