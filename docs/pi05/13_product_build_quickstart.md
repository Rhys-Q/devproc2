# Pi0.5 产品构建 Quickstart

本文是 Pi0.5 的产品化主流程。验证、PyTorch oracle、raw dump 和 Nsight profile 仍在
`08_build_run_profile.md`，但不再作为模型构建的必要前置步骤。

固定本机资产：

```bash
cd /root/tw/devproc2
export DEVPROC2_ROOT=/root/tw/devproc2
export PI05_CKPT=/root/tools/pi05_libero_base
export PI05_TORCH_DUMP=/root/tw/openpi/outputs/pi05_torch_infer
export PI05_TOKENIZER=$PI05_TORCH_DUMP/tokenizer.model
export PI05_RAW_ORACLE=$DEVPROC2_ROOT/build/pi05_torch_dump_oracle
export PYTHONPATH=$DEVPROC2_ROOT/python:${PYTHONPATH:-}
```

## 1. Build Runtime

Runtime build 只需要 devproc2 runtime core 和要运行的 benchmark/test binary。Pi0.5
CUDA packed backend 是 model-owned extension，由第 3 步的 `devproc2 build`
显式构建或复用。

```bash
cmake -S . -B build/runtime-sm89 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/runtime-sm89 \
  --target devproc2_runtime bench_pi05_denoise \
  -j
```

## 2. Convert Weight

从 OpenPI/HF checkpoint 生成 devproc2 weight package：

```bash
python -m tools.pi05.convert_weights \
  --checkpoint-dir "$PI05_CKPT" \
  --output-dir build/weights/pi05_libero_base.fp8_sm89 \
  --hardware rtx_sm89 \
  --activation-scales static \
  --shape-profile pi05_libero_base_3v200
```

短期基础 converter 还不会生成 `act_scale.*` 时，`convert_report.json` 会写入
`"activation_scales": "missing"` 和 `"supports_static_act_scales": false`。
随后 `devproc2 build --activation-scales static` 会在 build 阶段直接失败；用
`--activation-scales dynamic` 可生成 smoke artifact，但不能代表当前静态 scale
性能 baseline。

## 3. Build Model Artifact

`devproc2 build` 是模型编译、kernel 打包、backend extension 打包和资源复制的
canonical 入口：

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
  --resource tokenizer="$PI05_TOKENIZER"
```

Backend modes：

```text
--build-backends auto    cache miss 时自动构建 model-owned backend
--build-backends never   禁止构建，只能从 cache 或 --backend-library-dir 取预编译 .so
--build-backends force   忽略 cache，强制重建
```

`auto/force` 会配置 `python/devproc2/models/pi05/cuda/CMakeLists.txt`，不再通过
runtime CMake tree 构建 Pi0.5 backend。需要直接调试 backend extension 时，可以单独
配置该 model CUDA project，并构建 `devproc2_pi05_cuda_backend` 或
`test_pi05_cuda_gemm`。

构建阶段会校验 graph ABI 需要的权重是否都存在，并写入：

```text
metadata/build.json
metadata/config.json
metadata/weight_binding.json
metadata/backend_build.json
metadata/artifact.json
```

## 4. Inspect And Run

```bash
python -m devproc2.inspect build/artifacts/pi05_libero_base.sample_tokens.sm89
```

用 runtime benchmark 做同权重 example0 验证：

```bash
build/runtime-sm89/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir build/artifacts/pi05_libero_base.sample_tokens.sm89 \
  --oracle-dir "$PI05_RAW_ORACLE/bf16_example0/raw" \
  --max-prompt-len 200 \
  --prefix-valid-rows 895 \
  --num-views 3
```

`bench_pi05_denoise` 是验证工具。部署侧应通过 `ModelSession::LoadArtifact(...)`
或后续模型级 C++ API 加载同一份 artifact。
