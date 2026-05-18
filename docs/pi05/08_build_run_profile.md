# Pi0.5 编译、运行与性能 Profile 手册

本文面向第一次接触 devproc2 的同学，目标是从已有 checkpoint 和 torch 对点数据出发，编译 C++ runtime，导出 Pi0.5 artifact，运行 full-token `sample_tokens` benchmark，并用 Nsight 做性能 profile。

本流程不安装、不更新、不修改 `/root/tw/openpi` 的 uv 环境。`/root/tw/openpi/outputs/pi05_torch_infer` 只作为已有输入 dump 和 tokenizer 的只读来源。

## 路径约定

先固定本文后面所有命令用到的路径：

```bash
cd /root/tw/devproc2

export DEVPROC2_ROOT=/root/tw/devproc2
export PI05_CKPT=/root/tools/pi05_libero_base
export PI05_TORCH_DUMP=/root/tw/openpi/outputs/pi05_torch_infer
export PI05_TOKENIZER=$PI05_TORCH_DUMP/tokenizer.model
export PI05_WEIGHT_PKG=$DEVPROC2_ROOT/build/pi05_fp8.weights
export PI05_ORACLE_RAW=$DEVPROC2_ROOT/build/pi05_torch_denoise_oracle/bf16_example0/raw

# 当前本机 FlashRT FA2 动态库位置。若路径不存在，见“常见问题”。
export DEVPROC2_FLASHRT_FA2_SO=/root/tw/FlashRT/flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
export PYTHONPATH=$DEVPROC2_ROOT/python:${PYTHONPATH:-}
```

检查关键文件是否存在：

```bash
test -f "$PI05_CKPT/model.safetensors"
test -f "$PI05_TOKENIZER"
test -f "$PI05_TORCH_DUMP/inputs.npz"
test -f "$PI05_TORCH_DUMP/fp16/outputs.npz"
test -f "$PI05_ORACLE_RAW/images_u8.bin"
test -f "$PI05_ORACLE_RAW/token_ids_i32.bin"
test -f "$PI05_ORACLE_RAW/step_000/actions_f32.bin"
test -f "$PI05_ORACLE_RAW/step_009/x_next_f32.bin"
```

如果 `PI05_ORACLE_RAW` 缺文件，说明 devproc2 benchmark 需要的 raw 对点数据还没有准备好。不要去 `/root/tw/openpi` 里执行 `uv sync`、`uv run` 或重装依赖；先找已有 raw dump，或让维护者在已经配置好的 openpi 环境中生成。

## 机器与软件要求

推荐环境是 RTX 4090 / SM89。本文命令默认 `sm_89`，也就是：

```bash
nvidia-smi
nvcc --version
cmake --version
```

需要：

- CUDA Toolkit，包含 `nvcc`、cuBLASLt、Nsight Systems `nsys`。
- CMake >= 3.24。
- Python >= 3.10。
- Python 包：`numpy`、`torch`、`safetensors`。权重转换需要 `torch` 和 `safetensors`；只跑已生成 artifact 时不需要重新转换。
- git submodules：`dlpack`、`json`、`tokenizers-cpp`。

只在 devproc2 环境里安装 Python 依赖，不要改 openpi 的 uv 环境：

```bash
cd "$DEVPROC2_ROOT"
git submodule update --init --recursive

# 如果已有 .venv，可以直接 source；没有再创建。
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m pip install torch safetensors
```

## 1. 准备权重包

如果 `build/pi05_fp8.weights` 已存在，并且里面有 `act_scale.*`，优先复用它。这是当前性能 profile 用的静态 activation scale 权重包。

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("build/pi05_fp8.weights/weight_map.json")
assert p.exists(), f"missing {p}"
names = [w["name"] for w in json.loads(p.read_text())["weights"]]
print("entries:", len(names))
print("act_scale entries:", sum(n.startswith("act_scale.") for n in names))
PY
```

期望看到 `act_scale entries` 大于 0。当前本机包通常是 `780` entries，其中 `250` 个 `act_scale.*`。

如果必须从 checkpoint 重新转换，可以运行：

```bash
python - <<'PY'
from devproc2.models.pi05 import convert_pi05_weights

convert_pi05_weights(
    checkpoint_dir="/root/tools/pi05_libero_base",
    output_dir="build/pi05_fp8.weights",
    hardware="rtx_sm89",
    include_bf16=False,
    include_support_bf16=True,
    include_fp8=True,
    include_precomputed_styles=True,
    action_horizon=50,
    device="cuda",
)
PY
```

注意：仓库内的基础 converter 只负责 safetensors 到 devproc2 权重和 FP8 权重量化。若新生成的包没有 `act_scale.*`，导出 artifact 时不要加 `--use-static-act-scales`；它可以用于 smoke test，但不等同于当前性能基线。

## 2. 导出 Pi0.5 Artifact

full-token artifact 直接消费：

- `noise_f32`
- `images_u8`
- `token_ids`
- `prefix_valid_rows`
- `prefix_rope_interleaved`
- `suffix_rope_interleaved`

当前主测两个形态：

- 3-view：`P=895`，`num_views=3`，`max_prompt_len=127`
- 2-view：`P=562`，`num_views=2`，`max_prompt_len=50`

导出 3-view artifact：

```bash
python -m devproc2.models.pi05.export \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
  --weight-package-dir "$PI05_WEIGHT_PKG" \
  --tokenizer-model-path "$PI05_TOKENIZER" \
  --prefix-rows 895 \
  --max-prompt-len 127 \
  --num-views 3 \
  --sm-arch 89 \
  --use-static-act-scales
```

导出 2-view artifact：

```bash
python -m devproc2.models.pi05.export \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_2v562_artifact \
  --weight-package-dir "$PI05_WEIGHT_PKG" \
  --tokenizer-model-path "$PI05_TOKENIZER" \
  --prefix-rows 562 \
  --max-prompt-len 50 \
  --num-views 2 \
  --sm-arch 89 \
  --use-static-act-scales
```

导出完成后检查 artifact：

```bash
python devproc_cli.py inspect build/pi05_fp8_sample_tokens_3v895_artifact
python devproc_cli.py inspect build/pi05_fp8_sample_tokens_2v562_artifact
```

至少应看到：

- `executable.vm`
- `metadata/pi05_artifact.json`
- `weights/weights.index.json`
- `resources/tokenizer.model`
- `kernels/*.cubin`

## 3. 编译 C++ Runtime 和 Benchmark

推荐先用 CUTLASS 打开当前性能路径。如果本机没有 CUTLASS checkout，先把 `DEVPROC2_WITH_CUTLASS` 改成 `OFF`，功能仍可跑，但性能数字会不同。

```bash
cmake -S . -B build/root-cuda \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DDEVPROC2_WITH_CUTLASS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/root-cuda \
  --target bench_pi05_denoise test_pi05_cuda_gemm test_cuda_graph \
  -j2
```

如果 CMake 报：

```text
DEVPROC2_WITH_CUTLASS=ON requires DEVPROC2_CUTLASS_ROOT
```

要么指定 CUTLASS：

```bash
cmake -S . -B build/root-cuda \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DDEVPROC2_WITH_CUTLASS=ON \
  -DDEVPROC2_CUTLASS_ROOT=/path/to/cutlass \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

要么关闭 CUTLASS：

```bash
cmake -S . -B build/root-cuda \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DDEVPROC2_WITH_CUTLASS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

基础 smoke test：

```bash
ctest --test-dir build/root-cuda/runtime \
  -R 'test_cuda_graph|test_pi05_cuda_gemm' \
  --output-on-failure
```

`test_pi05_denoise_oracle` 会依赖更多历史 artifact，例如 `build/pi05_fp8_artifact`、`build/pi05_fp8_loop_artifact`、`build/pi05_fp8_sample_precomputed_prefix_artifact`。只跑 full-token benchmark 时，不需要先跑这个测试。

## 4. 运行 Full-Token Benchmark

先确认 FlashRT FA2 和 libpython 能被加载。当前代码默认找 `/root/autodl-tmp/FlashRT/...`，本机应显式指定 `/root/tw/FlashRT/...`：

```bash
test -f "$DEVPROC2_FLASHRT_FA2_SO"

# 如果 libpython 不在默认 /root/miniconda3/lib/libpython3.12.so，也显式设置：
# export DEVPROC2_LIBPYTHON_SO=/path/to/libpython3.12.so
```

跑 3-view：

```bash
DEVPROC2_FLASHRT_FA2_SO="$DEVPROC2_FLASHRT_FA2_SO" \
build/root-cuda/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
  --max-prompt-len 127 \
  --prefix-valid-rows 895 \
  --num-views 3
```

跑 2-view：

```bash
DEVPROC2_FLASHRT_FA2_SO="$DEVPROC2_FLASHRT_FA2_SO" \
build/root-cuda/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_2v562_artifact \
  --max-prompt-len 50 \
  --prefix-valid-rows 562 \
  --num-views 2
```

输出类似：

```text
pi05_denoise_bench iters=50 entry=sample_tokens mode=cuda_graph warmup_ms=...
  mean_10step_ms=28.xxx mean_step_ms=2.xxx final_abs_max=... final_abs_mean=...
```

看结果时重点看：

- `mode=cuda_graph`：说明走的是部署侧 CUDA Graph replay。
- `mean_10step_ms`：benchmark 统计的主要延迟。full-token 路径包含 vision encoder、language embedding、prefix KV materialization 和 10-step denoise。
- `final_abs_max` / `final_abs_mean`：与 raw torch target 的动作输出误差。full-token 当前不是严格 bitwise 对齐，主要用于性能路径 smoke 和趋势观察。

当前 RTX 4090 参考值：

| 形态 | Artifact | 参考延迟 |
| --- | --- | --- |
| 2-view / P=562 | `build/pi05_fp8_sample_tokens_2v562_artifact` | 约 `23.4ms` |
| 3-view / P=895 | `build/pi05_fp8_sample_tokens_3v895_artifact` | 约 `28.5ms` |
| 3-view / P=769 对齐 realtime-vla | `build/pi05_fp8_sample_tokens_3v769_artifact` | 约 `29.2ms` |

## 5. Nsight Systems Profile

先用 Nsight Systems 看“时间花在哪里”。这一步不要一上来用 Nsight Compute；先知道主耗时 kernel 再钻单 kernel。

```bash
mkdir -p build/pi05_profiles

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o build/pi05_profiles/pi05_sample_tokens_3v895 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 20 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
    --max-prompt-len 127 \
    --prefix-valid-rows 895 \
    --num-views 3
```

生成统计表：

```bash
nsys stats build/pi05_profiles/pi05_sample_tokens_3v895.nsys-rep \
  --report cuda_gpu_kern_sum,cuda_api_sum
```

如果要看非 CUDA Graph 的 launch overhead，可加 `--no-graph`：

```bash
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o build/pi05_profiles/pi05_sample_tokens_3v895_stream \
  build/root-cuda/runtime/tests/bench_pi05_denoise 3 \
    --no-graph \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
    --max-prompt-len 127 \
    --prefix-valid-rows 895 \
    --num-views 3
```

当前预期结论是：主耗时集中在 cuBLASLt FP8 GEMM，尤其是 vision/prefix encoder FFN gate/up/down，而不是 tokenizer、attention fallback 或零散 elementwise kernel。

## 6. Nsight Compute Profile

Nsight Compute 用来回答“这个 kernel 为什么慢”。先从 `nsys stats` 里复制一个耗时最高的 kernel 名，再用 `ncu` 抓少量 launch。

常见 GEMM 名会包含 `sm89_xmma_gemm`，可以先这样抓：

```bash
ncu \
  --set speed-of-light \
  --target-processes all \
  --kernel-name regex:sm89_xmma_gemm \
  --launch-skip 20 \
  --launch-count 5 \
  --force-overwrite \
  -o build/pi05_profiles/ncu_sm89_xmma_3v895 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 3 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
    --max-prompt-len 127 \
    --prefix-valid-rows 895 \
    --num-views 3
```

如果要抓自定义 kernel，例如 GeGLU：

```bash
ncu \
  --set speed-of-light \
  --target-processes all \
  --kernel-name regex:pi05_geglu_to_fp8_bf16 \
  --launch-count 5 \
  --force-overwrite \
  -o build/pi05_profiles/ncu_geglu_3v895 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 3 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
    --max-prompt-len 127 \
    --prefix-valid-rows 895 \
    --num-views 3
```

读 ncu 时先看：

- `SOL` / `Speed Of Light`：判断更偏 compute-bound 还是 memory-bound。
- `Occupancy`：过低通常要看寄存器、shared memory、block size。
- `Memory Workload Analysis`：看 L2 命中、读写带宽、coalescing。
- `Launch Statistics`：确认抓到的是目标 kernel，不是 warmup 或无关小 kernel。

## 7. 常见问题

`missing pi05 artifact/oracle inputs`

说明 `bench_pi05_denoise` 找不到 `build/pi05_torch_denoise_oracle/bf16_example0/raw` 下的 raw 文件。full-token benchmark 至少需要：

```text
images_u8.bin
token_ids_i32.bin
prefix_embs_bf16.bin
prefix_rope_interleaved_bf16.bin
prefix_k_cache_bf16.bin
prefix_v_cache_bf16.bin
rope_interleaved_bf16.bin
step_000/actions_f32.bin
step_009/x_next_f32.bin
```

`failed to dlopen FlashRT FA2 library`

设置正确的动态库路径：

```bash
export DEVPROC2_FLASHRT_FA2_SO=/root/tw/FlashRT/flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
```

如果报 `failed to dlopen libpython`，再设置：

```bash
export DEVPROC2_LIBPYTHON_SO=/path/to/libpython3.12.so
```

`unknown --entry-kind` 或 artifact 形状不匹配

确认 benchmark 参数和导出 artifact 参数一致：

| 参数 | 2-view | 3-view |
| --- | --- | --- |
| `--artifact-dir` | `build/pi05_fp8_sample_tokens_2v562_artifact` | `build/pi05_fp8_sample_tokens_3v895_artifact` |
| `--max-prompt-len` | `50` | `127` |
| `--prefix-valid-rows` | `562` | `895` |
| `--num-views` | `2` | `3` |

`act_scale.* missing`

说明权重包不是当前性能包。可以先去掉 artifact export 里的 `--use-static-act-scales` 做 smoke test；要复现本文性能数字，需要带静态 activation scale 的权重包。

CUTLASS 构建失败

如果只是想先跑通，设 `-DDEVPROC2_WITH_CUTLASS=OFF`。如果要复现当前性能基线，需要提供可用 CUTLASS checkout，并打开 `DEVPROC2_WITH_CUTLASS=ON`。

性能明显变慢

先确认：

```bash
echo "$DEVPROC2_FLASHRT_FA2_SO"
echo "${DEVPROC2_CUBLASLT_FP8_FAST_ACCUM:-default_on}"
echo "${DEVPROC2_CUTLASS_FP8_NT:-default_on}"
```

性能 profile 默认使用：

- CUDA Graph 开启，不加 `--no-graph`。
- `DEVPROC2_CUBLASLT_FP8_FAST_ACCUM` 默认开启。
- `DEVPROC2_CUTLASS_FP8_NT` 默认开启，但只有编译时 `DEVPROC2_WITH_CUTLASS=ON` 才生效。

## 最小命令清单

已经有权重包和 raw oracle 时，最短路径是：

```bash
cd /root/tw/devproc2
source .venv/bin/activate

export PYTHONPATH=$PWD/python:${PYTHONPATH:-}
export PI05_TOKENIZER=/root/tw/openpi/outputs/pi05_torch_infer/tokenizer.model
export DEVPROC2_FLASHRT_FA2_SO=/root/tw/FlashRT/flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so

python -m devproc2.models.pi05.export \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
  --weight-package-dir build/pi05_fp8.weights \
  --tokenizer-model-path "$PI05_TOKENIZER" \
  --prefix-rows 895 \
  --max-prompt-len 127 \
  --num-views 3 \
  --sm-arch 89 \
  --use-static-act-scales

cmake -S . -B build/root-cuda \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DDEVPROC2_WITH_CUTLASS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/root-cuda --target bench_pi05_denoise -j2

build/root-cuda/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
  --max-prompt-len 127 \
  --prefix-valid-rows 895 \
  --num-views 3

nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o build/pi05_profiles/pi05_sample_tokens_3v895 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 20 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v895_artifact \
    --max-prompt-len 127 \
    --prefix-valid-rows 895 \
    --num-views 3
```
