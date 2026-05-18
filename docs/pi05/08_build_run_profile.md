# Pi0.5 编译、运行与性能 Profile 手册

本文面向第一次接触 devproc2 的同学，目标是从同一个 Pi0.5 checkpoint 出发，生成 openpi PyTorch oracle，导出 devproc2 full-token `sample_tokens` artifact，运行 actions 级精度对点，并用 Nsight 做性能 profile。

当前推荐流程使用已经安装好的 openpi venv：

```text
/root/tw/openpi/.venv
```

不要在 `/root/tw/openpi` 里执行 `uv sync`、`uv run` 或重装依赖。需要重新生成 PyTorch oracle 时，只使用这个 venv 里的 Python。

## 路径约定

```bash
cd /root/tw/devproc2

export DEVPROC2_ROOT=/root/tw/devproc2
export OPENPI_ROOT=/root/tw/openpi
export OPENPI_PY=$OPENPI_ROOT/.venv/bin/python

export PI05_CKPT=/root/tools/pi05_libero_base
export PI05_TORCH_DUMP=/root/tw/openpi/outputs/pi05_torch_infer_libero_base
export PI05_TOKENIZER=$PI05_TORCH_DUMP/tokenizer.model
export PI05_WEIGHT_PKG=$DEVPROC2_ROOT/build/pi05_fp8.weights
export PI05_DUMP_RAW=$DEVPROC2_ROOT/build/pi05_torch_dump_oracle_libero_base

export DEVPROC2_FLASHRT_FA2_SO=/root/tw/FlashRT/flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
export DEVPROC2_LIBPYTHON_SO=/root/miniforge3/envs/py312/lib/libpython3.12.so.1.0
export PYTHONPATH=$DEVPROC2_ROOT/python:${PYTHONPATH:-}
```

检查关键文件：

```bash
test -x "$OPENPI_PY"
test -f "$PI05_CKPT/model.safetensors"
test -d "$PI05_WEIGHT_PKG"
test -f "$DEVPROC2_FLASHRT_FA2_SO"
test -f "$DEVPROC2_LIBPYTHON_SO"
```

历史目录 `/root/tw/openpi/outputs/pi05_torch_infer` 的 metadata 指向 `/root/autodl-tmp/tools/pi05-pytorch-base`。如果 devproc2 权重包来自 `/root/tools/pi05_libero_base/model.safetensors`，不要拿这个历史 dump 做 actions 级结论；它只能作为输入格式参考。

## 机器与软件要求

推荐环境是 RTX 4090 / SM89：

```bash
nvidia-smi
nvcc --version
cmake --version
which nsys
```

devproc2 侧需要：

- CUDA Toolkit，包含 `nvcc`、cuBLASLt、Nsight Systems `nsys`。
- CMake >= 3.24。
- Python >= 3.10。
- Python 包：`numpy`、`torch`、`safetensors`。
- git submodules：`dlpack`、`json`、`tokenizers-cpp`。

如果 openpi 初始化报 `transformers_replace is not installed correctly`，按 openpi 自己的提示把 replacement 文件拷进 venv：

```bash
cp -r \
  /root/tw/openpi/src/openpi/models_pytorch/transformers_replace/* \
  /root/tw/openpi/.venv/lib/python3.11/site-packages/transformers/

OMP_NUM_THREADS=1 PYTHONPATH=$OPENPI_ROOT:$OPENPI_ROOT/src "$OPENPI_PY" - <<'PY'
from openpi.models import pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import torch

with torch.device("cuda"):
    PI0Pytorch(pi0_config.Pi0Config(pi05=True, dtype="bfloat16", pytorch_compile_mode=None))
print("transformers_replace ok")
PY
```

## 1. 准备权重包

当前性能 baseline 使用带静态 activation scale 的 FP8 权重包：

```bash
python - <<'PY'
import json
from pathlib import Path

wm = json.loads(Path("build/pi05_fp8.weights/weight_map.json").read_text())["weights"]
idx = json.loads(Path("build/pi05_fp8.weights/weights.index.json").read_text())["entries"]
report = json.loads(Path("build/pi05_fp8.weights/convert_report.json").read_text())
print("weight_map entries:", len(wm))
print("index entries:", len(idx))
print("act_scale entries:", sum(w["name"].startswith("act_scale.") for w in wm))
print("source.path:", report["source"]["path"])
PY
```

本机当前结果：

```text
weight_map entries: 780
index entries: 780
act_scale entries: 250
source.path: /root/tools/pi05_libero_base/model.safetensors
```

确认 PyTorch dump 和 devproc2 权重包同源：

```bash
python - <<'PY'
import json
from pathlib import Path

dump_meta = json.loads(Path("/root/tw/openpi/outputs/pi05_torch_infer_libero_base/metadata.json").read_text())
report = json.loads(Path("build/pi05_fp8.weights/convert_report.json").read_text())
print("torch dump ckpt:", dump_meta.get("ckpt"))
print("weight pkg source:", report.get("source", {}).get("path"))
PY
```

同权重对点时应看到：

```text
torch dump ckpt: /root/tools/pi05_libero_base
weight pkg source: /root/tools/pi05_libero_base/model.safetensors
```

如果必须从 checkpoint 重新转换权重：

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

注意：基础 converter 不会自动生成当前性能包里的 `act_scale.*`。如果新包缺少 `act_scale.*`，导出 artifact 时不要加 `--use-static-act-scales`；它能用于 smoke test，但不等同于当前性能 baseline。

## 2. 生成同权重 PyTorch Dump

用 openpi venv 重新跑 PyTorch BF16/FP16 oracle：

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

本机 2026-05-18 实测输出：

```text
shape: [10, 50, 32]
bf16_vs_fp16 abs_max: 1.0936790704727173
bf16_vs_fp16 abs_mean: 0.0009669848368503153
bf16_vs_fp16 abs_p95: 0.0026625737547874436
bf16_vs_fp16 abs_p99: 0.011909374296665205
```

`abs_max` 的单点 outlier 来自 PyTorch BF16/FP16 自身差异；看 oracle sanity 时要同时看 `abs_mean/p95/p99`。

可用 openpi 自带 check 脚本确认 FP16 dump 可重放：

```bash
OMP_NUM_THREADS=1 \
PYTHONPATH=$OPENPI_ROOT:$OPENPI_ROOT/src \
"$OPENPI_PY" "$OPENPI_ROOT/scripts/check_pi05_fp16_infer.py" \
  --ckpt "$PI05_CKPT" \
  --dump-dir "$PI05_TORCH_DUMP" \
  --device cuda \
  --num-steps 10 \
  --rtol 1e-2 \
  --atol 1e-2
```

本机结果：

```text
allclose: true
abs_max: 0.0
abs_mean: 0.0
shape: [10, 50, 32]
```

## 3. 从 Dump 生成 Runtime Raw

`bench_pi05_denoise` 不直接读 `.npz`，它读取 raw 文件。full-token `sample_tokens` 入口需要：

```text
images_u8.bin
token_ids_i32.bin
prefix_rope_interleaved_bf16.bin
rope_interleaved_bf16.bin
step_000/actions_f32.bin
step_009/x_next_f32.bin
```

benchmark 的通用 loader 还会读取 `prefix_k_cache_bf16.bin`、`prefix_v_cache_bf16.bin`、`prefix_embs_bf16.bin`。对 `sample_tokens` 来说这三个文件不被计算图消费，可以写零占位。不要用这份 raw 跑 `step`、`loop` 或 precomputed-prefix 入口。

当前 openpi 配置是 3-view / `max_token_len=200`：

```text
image rows = 3 * 256 = 768
prefix_rows P = 768 + 200 = 968
prefix_valid_rows = 768 + token_valid_len
```

生成 raw 时使用 openpi 自己的 transform/tokenizer，避免手写 prompt/state 离散化规则：

```bash
OMP_NUM_THREADS=1 \
PYTHONPATH=$OPENPI_ROOT:$OPENPI_ROOT/src \
"$OPENPI_PY" - <<'PY'
import json
from pathlib import Path

import numpy as np
import torch
from openpi.models import pi0_config
from scripts import dump_pi05_torch_infer as dump

src = Path("/root/tw/openpi/outputs/pi05_torch_infer_libero_base")
out_root = Path("/root/tw/devproc2/build/pi05_torch_dump_oracle_libero_base")
inputs = dump._load_inputs(src / "inputs.npz")  # noqa: SLF001
targets = np.load(src / "bf16" / "outputs.npz")["actions"].astype(np.float32)
config = pi0_config.Pi0Config(pi05=True, dtype="bfloat16", pytorch_compile_mode=None)
transform = dump._make_transform(config)  # noqa: SLF001

V = 3
TOK = int(config.max_token_len)
P = V * 256 + TOK
T = int(config.action_horizon)
HD = 256
L = 18
HKV = 1

def rope_bits(position_ids):
    pos = torch.as_tensor(position_ids, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    freqs = pos[:, None] * inv[None, :]
    cos = torch.cos(freqs).to(torch.bfloat16)
    sin = torch.sin(freqs).to(torch.bfloat16)
    rope = torch.empty((pos.numel(), HD), dtype=torch.bfloat16)
    rope[:, 0::2] = cos
    rope[:, 1::2] = sin
    return rope.view(torch.uint16).numpy().copy()

def write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr).tofile(path)

zeros_prefix_kv = np.zeros((L, P, HKV, HD), dtype=np.uint16)
zeros_prefix_embs = np.zeros((P, 2048), dtype=np.uint16)
summary = []

for i in range(targets.shape[0]):
    transformed = transform(dump._example_from_inputs(inputs, i))  # noqa: SLF001
    token_ids = transformed["tokenized_prompt"].astype(np.int32)
    token_mask = transformed["tokenized_prompt_mask"].astype(np.bool_)
    prefix_mask = np.concatenate([np.ones((V * 256,), dtype=np.bool_), token_mask])
    position_ids = np.cumsum(prefix_mask.astype(np.int64)) - 1
    prefix_valid_rows = int(prefix_mask.sum())
    token_valid_len = int(token_mask.sum())

    images = np.stack(
        [
            inputs["base_0_rgb"][i],
            inputs["left_wrist_0_rgb"][i],
            inputs["right_wrist_0_rgb"][i],
        ],
        axis=0,
    ).astype(np.uint8)

    raw = out_root / f"bf16_example{i}" / "raw"
    write(raw / "images_u8.bin", images)
    write(raw / "state_f32.bin", inputs["state"][i].astype(np.float32))
    write(raw / "token_ids_i32.bin", token_ids)
    write(raw / "prefix_rope_interleaved_bf16.bin", rope_bits(position_ids))
    write(raw / "rope_interleaved_bf16.bin", rope_bits(np.arange(prefix_valid_rows, prefix_valid_rows + T)))
    write(raw / "prefix_k_cache_bf16.bin", zeros_prefix_kv)
    write(raw / "prefix_v_cache_bf16.bin", zeros_prefix_kv)
    write(raw / "prefix_embs_bf16.bin", zeros_prefix_embs)
    write(raw / "step_000" / "actions_f32.bin", inputs["noise"][i].astype(np.float32))
    write(raw / "step_009" / "x_next_f32.bin", targets[i].astype(np.float32))
    (raw / "prompt.txt").write_text(str(inputs["prompt"][i]) + "\n")

    meta = {
        "format": "devproc2.pi05_sample_tokens_raw_from_openpi_dump",
        "source": str(src),
        "ckpt": "/root/tools/pi05_libero_base",
        "example_index": i,
        "target_precision": "bf16",
        "num_views": V,
        "max_prompt_len": TOK,
        "prefix_rows": P,
        "prefix_valid_rows": prefix_valid_rows,
        "token_valid_len": token_valid_len,
        "token_head": token_ids[:16].astype(int).tolist(),
    }
    (out_root / f"bf16_example{i}" / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    summary.append(meta)

(out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "out_root": str(out_root),
    "num_examples": len(summary),
    "prefix_rows": P,
    "prefix_valid_rows": [m["prefix_valid_rows"] for m in summary],
    "token_valid_len": [m["token_valid_len"] for m in summary],
}, indent=2, sort_keys=True))
PY
```

本机生成结果：

```text
prefix_rows: 968
prefix_valid_rows: [895, 900, 906, 900, 904, 894, 902, 906, 898, 905]
token_valid_len: [127, 132, 138, 132, 136, 126, 134, 138, 130, 137]
```

## 4. 导出 Pi0.5 Artifact

真实 dump 对点主线使用 3-view / P=968：

```bash
python -m devproc2.models.pi05.export \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
  --weight-package-dir "$PI05_WEIGHT_PKG" \
  --tokenizer-model-path "$PI05_TOKENIZER" \
  --prefix-rows 968 \
  --max-prompt-len 200 \
  --num-views 3 \
  --sm-arch 89 \
  --use-static-act-scales
```

性能 smoke 可继续导出 3-view / P=769 的历史 synthetic 形态：

```bash
python -m devproc2.models.pi05.export \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v769_artifact \
  --weight-package-dir "$PI05_WEIGHT_PKG" \
  --tokenizer-model-path "$PI05_TOKENIZER" \
  --prefix-rows 769 \
  --max-prompt-len 1 \
  --num-views 3 \
  --sm-arch 89 \
  --use-static-act-scales
```

导出完成后检查：

```bash
python devproc_cli.py inspect build/pi05_fp8_sample_tokens_3v968_artifact
```

至少应看到：

- `executable.vm`
- `metadata/pi05_artifact.json`
- `weights/weights.index.json`
- `resources/tokenizer.model`
- `kernels/*.cubin`

## 5. 编译 C++ Runtime 和 Benchmark

推荐打开 CUTLASS 复现当前性能路径：

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

如果 CMake 报 `DEVPROC2_WITH_CUTLASS=ON requires DEVPROC2_CUTLASS_ROOT`，要么指定 CUTLASS checkout，要么先用 `-DDEVPROC2_WITH_CUTLASS=OFF` 跑功能 smoke。关闭 CUTLASS 后性能数字不能和本文 baseline 直接比较。

基础 smoke test：

```bash
ctest --test-dir build/root-cuda/runtime \
  -R 'test_cuda_graph|test_pi05_artifact_load|test_pi05_kernel_launch|test_pi05_cuda_gemm' \
  --output-on-failure
```

`test_pi05_denoise_oracle` 和 `test_pi05_sample_tokens_tokenizer` 依赖历史 raw oracle 的具体形状和额外文件。如果本地 raw 是只给 `sample_tokens` 用的 `.npz` 转换 raw，这些测试失败不等同于 full-token artifact 不能跑。

## 6. 运行 Full-Token Benchmark

跑同权重 dump example0 / P=968：

```bash
export META=$PI05_DUMP_RAW/bf16_example0/metadata.json
export PREFIX_VALID_ROWS=$(python - <<'PY'
import json, os
print(json.load(open(os.environ["META"]))["prefix_valid_rows"])
PY
)

DEVPROC2_FLASHRT_FA2_SO="$DEVPROC2_FLASHRT_FA2_SO" \
DEVPROC2_LIBPYTHON_SO="$DEVPROC2_LIBPYTHON_SO" \
build/root-cuda/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
  --oracle-dir "$PI05_DUMP_RAW/bf16_example0/raw" \
  --max-prompt-len 200 \
  --prefix-valid-rows "$PREFIX_VALID_ROWS" \
  --num-views 3
```

看结果时重点看：

- `mode=cuda_graph`：走部署侧 CUDA Graph replay。
- `mean_10step_ms`：full-token 路径整体延迟，包含 vision encoder、language embedding、prefix KV materialization 和 10-step denoise。
- `final_abs_max/final_abs_mean`：devproc2 FP8 artifact output vs PyTorch BF16 raw target。只有 checkpoint、token、RoPE、prefix rows 和 target 都同源时才是有效 actions 级对点。

本机 2026-05-18 同权重 P=968 batch 观测值：

```text
mean_10step_ms mean: 33.771
mean_10step_ms max: 34.352
final_abs_max mean: 0.795
final_abs_max worst: 1.983
final_abs_mean mean: 0.0054
final_abs_mean worst: 0.018
```

当前延迟参考：

| 形态 | Artifact | raw 来源 | 参考延迟 |
| --- | --- | --- | --- |
| 3-view / P=968 / max_token_len=200 | `build/pi05_fp8_sample_tokens_3v968_artifact` | 同权重 openpi dump | `~33.8ms` |
| 3-view / P=769 / max_prompt_len=1 | `build/pi05_fp8_sample_tokens_3v769_artifact` | 历史 synthetic raw | `~29.3ms` |
| 3-view / P=895 / max_prompt_len=127 | `build/pi05_fp8_sample_tokens_3v895_artifact` | 需匹配 raw | `~28-31ms` |
| 2-view / P=562 / max_prompt_len=50 | `build/pi05_fp8_sample_tokens_2v562_artifact` | 需匹配 raw | `~23.4ms` |

## 7. 批量跑 Dump 精度对点

下面命令会跑 10 个 dump 样本，每个样本从自己的 metadata 读取 `prefix_valid_rows`：

```bash
python - <<'PY'
import json
import os
import re
import subprocess
from pathlib import Path

root = Path("/root/tw/devproc2")
raw_root = root / "build/pi05_torch_dump_oracle_libero_base"
artifact = root / "build/pi05_fp8_sample_tokens_3v968_artifact"
bin_path = root / "build/root-cuda/runtime/tests/bench_pi05_denoise"
summary = json.loads((raw_root / "summary.json").read_text())
env = os.environ.copy()
env["DEVPROC2_FLASHRT_FA2_SO"] = "/root/tw/FlashRT/flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so"
env["DEVPROC2_LIBPYTHON_SO"] = "/root/miniforge3/envs/py312/lib/libpython3.12.so.1.0"
pat = re.compile(
    r"mean_10step_ms=([0-9.]+).*mean_step_ms=([0-9.]+).*"
    r"final_abs_max=([0-9.eE+-]+).*final_abs_mean=([0-9.eE+-]+)"
)
rows = []

for meta in summary:
    i = meta["example_index"]
    cmd = [
        str(bin_path),
        "20",
        "--entry-kind",
        "sample_tokens",
        "--artifact-dir",
        str(artifact),
        "--oracle-dir",
        str(raw_root / f"bf16_example{i}" / "raw"),
        "--max-prompt-len",
        "200",
        "--prefix-valid-rows",
        str(meta["prefix_valid_rows"]),
        "--num-views",
        "3",
    ]
    out = subprocess.check_output(cmd, cwd=str(root), env=env, text=True, stderr=subprocess.STDOUT)
    line = out.strip().splitlines()[-1]
    mean_ms, step_ms, abs_max, abs_mean = map(float, pat.search(line).groups())
    row = {
        "example": i,
        "tok": meta["token_valid_len"],
        "pv": meta["prefix_valid_rows"],
        "mean_ms": mean_ms,
        "abs_max": abs_max,
        "abs_mean": abs_mean,
    }
    rows.append(row)
    print(
        f"example={i} tok={row['tok']} pv={row['pv']} "
        f"mean_10step_ms={mean_ms:.3f} final_abs_max={abs_max:.6f} "
        f"final_abs_mean={abs_mean:.6f}"
    )

print("summary")
print("mean_ms=%.3f max_ms=%.3f mean_abs_max=%.6f max_abs_max=%.6f mean_abs_mean=%.6f max_abs_mean=%.6f" % (
    sum(r["mean_ms"] for r in rows) / len(rows),
    max(r["mean_ms"] for r in rows),
    sum(r["abs_max"] for r in rows) / len(rows),
    max(r["abs_max"] for r in rows),
    sum(r["abs_mean"] for r in rows) / len(rows),
    max(r["abs_mean"] for r in rows),
))
PY
```

本次实测逐样本结果：

| example | token_valid_len | prefix_valid_rows | mean_10step_ms | final_abs_max | final_abs_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 127 | 895 | 34.352 | 0.026 | 0.002 |
| 1 | 132 | 900 | 33.278 | 0.037 | 0.003 |
| 2 | 138 | 906 | 33.793 | 1.983 | 0.018 |
| 3 | 132 | 900 | 33.896 | 0.022 | 0.003 |
| 4 | 136 | 904 | 34.172 | 0.044 | 0.004 |
| 5 | 126 | 894 | 33.338 | 1.946 | 0.007 |
| 6 | 134 | 902 | 33.396 | 0.039 | 0.003 |
| 7 | 138 | 906 | 33.408 | 1.900 | 0.006 |
| 8 | 130 | 898 | 34.079 | 1.913 | 0.005 |
| 9 | 137 | 905 | 34.002 | 0.040 | 0.003 |

解读：

- 同 checkpoint 后，旧的权重不一致导致的 `final_abs_mean≈0.36` 消失，当前 mean error 是 `0.0054`。
- 仍有少数样本出现接近动作范围上界的单点最大误差。这个现象在 `--no-graph` 下仍存在，且把 worst sample 导出成 token 长度精确匹配的 P=906 artifact 后仍存在，因此不是 CUDA Graph replay 或 P=968 padding 形状单独造成的。
- 当前 full-token production artifact 是 FP8 权重/静态 act-scale 路径，PyTorch oracle 是 BF16/FP16。评估回归时不要只看 `final_abs_max`，同时看 `final_abs_mean`、BF16/FP16 oracle sanity，以及同一 weight package/raw 下 refactor 前后的 runtime 输出。

## 8. Nsight Systems Profile

先用 Nsight Systems 看“时间花在哪里”。这一步不要一上来用 Nsight Compute；先知道主耗时 kernel 再钻单 kernel。

同权重 dump example0 / P=968：

```bash
mkdir -p build/pi05_profiles

export META=$PI05_DUMP_RAW/bf16_example0/metadata.json
export PREFIX_VALID_ROWS=$(python - <<'PY'
import json, os
print(json.load(open(os.environ["META"]))["prefix_valid_rows"])
PY
)

DEVPROC2_FLASHRT_FA2_SO="$DEVPROC2_FLASHRT_FA2_SO" \
DEVPROC2_LIBPYTHON_SO="$DEVPROC2_LIBPYTHON_SO" \
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o build/pi05_profiles/pi05_sample_tokens_3v968_same_weight_ex0 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 20 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
    --oracle-dir "$PI05_DUMP_RAW/bf16_example0/raw" \
    --max-prompt-len 200 \
    --prefix-valid-rows "$PREFIX_VALID_ROWS" \
    --num-views 3
```

P=769 synthetic smoke：

```bash
DEVPROC2_FLASHRT_FA2_SO="$DEVPROC2_FLASHRT_FA2_SO" \
DEVPROC2_LIBPYTHON_SO="$DEVPROC2_LIBPYTHON_SO" \
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o build/pi05_profiles/pi05_sample_tokens_3v769 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 20 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v769_artifact \
    --max-prompt-len 1 \
    --prefix-valid-rows 769 \
    --num-views 3
```

生成统计表：

```bash
nsys stats build/pi05_profiles/pi05_sample_tokens_3v968_same_weight_ex0.nsys-rep \
  --report cuda_gpu_kern_sum,cuda_api_sum
```

当前预期结论：主耗时集中在 FP8 GEMM，尤其是 vision/prefix encoder FFN gate/up/down；FlashRT attention 次之；不是 tokenizer、attention fallback 或零散 elementwise kernel。

本机 2026-05-18 same-weight example0 profile 已生成：

```text
build/pi05_profiles/pi05_sample_tokens_3v968_same_weight_ex0.nsys-rep
bench output: mean_10step_ms=34.050 final_abs_max=0.021 final_abs_mean=0.002
```

`nsys stats` 的 top GPU kernel 仍是 FP8 GEMM 和 FlashRT attention：

| Kernel 类别 | Time |
| --- | ---: |
| `sm89_xmma_gemm...128x128...` | 24.2% |
| `sm89_xmma_gemm...64x64...` | 20.7% |
| `sm89_xmma_gemm...32x64...` | 10.1% |
| FlashRT `flash_fwd_splitkv_kernel` | 8.8% |
| `sm89_xmma_gemm...64x128...` | 8.0% |
| `pi05_geglu_to_fp8_bf16` | 4.0% |

如果要看非 CUDA Graph 的 launch overhead，可加 `--no-graph`：

```bash
DEVPROC2_FLASHRT_FA2_SO="$DEVPROC2_FLASHRT_FA2_SO" \
DEVPROC2_LIBPYTHON_SO="$DEVPROC2_LIBPYTHON_SO" \
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o build/pi05_profiles/pi05_sample_tokens_3v968_same_weight_ex0_stream \
  build/root-cuda/runtime/tests/bench_pi05_denoise 3 \
    --no-graph \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
    --oracle-dir "$PI05_DUMP_RAW/bf16_example0/raw" \
    --max-prompt-len 200 \
    --prefix-valid-rows "$PREFIX_VALID_ROWS" \
    --num-views 3
```

## 9. Nsight Compute Profile

Nsight Compute 用来回答“这个 kernel 为什么慢”。先从 `nsys stats` 里复制一个耗时最高的 kernel 名，再用 `ncu` 抓少量 launch。

常见 GEMM 名会包含 `sm89_xmma_gemm`：

```bash
ncu \
  --set speed-of-light \
  --target-processes all \
  --kernel-name regex:sm89_xmma_gemm \
  --launch-skip 20 \
  --launch-count 5 \
  --force-overwrite \
  -o build/pi05_profiles/ncu_sm89_xmma_3v968 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 3 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
    --oracle-dir "$PI05_DUMP_RAW/bf16_example0/raw" \
    --max-prompt-len 200 \
    --prefix-valid-rows "$PREFIX_VALID_ROWS" \
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
  -o build/pi05_profiles/ncu_geglu_3v968 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 3 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
    --oracle-dir "$PI05_DUMP_RAW/bf16_example0/raw" \
    --max-prompt-len 200 \
    --prefix-valid-rows "$PREFIX_VALID_ROWS" \
    --num-views 3
```

读 ncu 时先看：

- `SOL` / `Speed Of Light`：判断更偏 compute-bound 还是 memory-bound。
- `Occupancy`：过低通常要看寄存器、shared memory、block size。
- `Memory Workload Analysis`：看 L2 命中、读写带宽、coalescing。
- `Launch Statistics`：确认抓到的是目标 kernel，不是 warmup 或无关小 kernel。

## 10. 常见问题

`transformers_replace is not installed correctly`

openpi venv 里的 transformers 没有打 replacement patch。执行本文“机器与软件要求”中的 `cp -r .../transformers_replace/* .../site-packages/transformers/`，然后重新跑初始化检查。

`missing pi05 artifact/oracle inputs`

说明 `bench_pi05_denoise` 找不到 raw 文件。full-token benchmark 至少需要：

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

使用本文的 dump 转换脚本生成 raw，或者用 `--oracle-dir` 指到已有 raw 目录。

`failed to dlopen FlashRT FA2 library`

```bash
export DEVPROC2_FLASHRT_FA2_SO=/root/tw/FlashRT/flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
```

`failed to dlopen libpython`

```bash
find /root /usr -name 'libpython3.12.so*' -type f 2>/dev/null | sort
export DEVPROC2_LIBPYTHON_SO=/root/miniforge3/envs/py312/lib/libpython3.12.so.1.0
```

`unknown --entry-kind` 或 artifact 形状不匹配

确认 benchmark 参数、artifact 参数和 raw 文件形状一致：

| 参数 | 同权重 dump 3-view | synthetic 3-view |
| --- | --- | --- |
| `--artifact-dir` | `build/pi05_fp8_sample_tokens_3v968_artifact` | `build/pi05_fp8_sample_tokens_3v769_artifact` |
| `--max-prompt-len` | `200` | `1` |
| `--prefix-valid-rows` | metadata 中的值 | `769` |
| `--num-views` | `3` | `3` |
| `--oracle-dir` | `$PI05_DUMP_RAW/bf16_exampleN/raw` | 可省略，使用历史默认 raw |

`final_abs_max/final_abs_mean` 偏大

先分清三件事：

- `bf16/outputs.npz` vs `fp16/outputs.npz` 是 PyTorch oracle 自身 sanity check。
- `bench_pi05_denoise` 的 `final_abs_*` 是 devproc2 FP8 artifact output vs PyTorch raw target。
- 从 `.npz` 生成的 raw 只适用于 `sample_tokens` 入口；其中 prefix cache 和 prefix embeddings 是占位文件。

排查顺序：

1. 确认 `build/pi05_fp8.weights/convert_report.json` 的 source 和 `PI05_TORCH_DUMP/metadata.json` 的 `ckpt` 同源。
2. 确认 `--prefix-rows`、`--max-prompt-len`、`--prefix-valid-rows` 和 raw metadata 匹配。
3. 确认 `token_ids_i32.bin` 是用 openpi transform 生成，而不是手写 tokenizer 规则。
4. 确认 `prefix_rope_interleaved_bf16.bin` 和 `rope_interleaved_bf16.bin` 按对应 sample 的 valid mask 生成。
5. 对回归判断，优先比较同一个 weight package、同一个 raw 输入下 refactor 前后的 runtime output。

`act_scale.* missing`

说明权重包不是当前性能包。可以先去掉 artifact export 里的 `--use-static-act-scales` 做 smoke test；要复现本文性能数字，需要带静态 activation scale 的权重包。

CUTLASS 构建失败

如果只是想先跑通，设 `-DDEVPROC2_WITH_CUTLASS=OFF`。如果要复现当前性能 baseline，需要提供可用 CUTLASS checkout，并打开 `DEVPROC2_WITH_CUTLASS=ON`。

性能明显变慢

先确认：

```bash
echo "$DEVPROC2_FLASHRT_FA2_SO"
echo "$DEVPROC2_LIBPYTHON_SO"
echo "${DEVPROC2_CUBLASLT_FP8_FAST_ACCUM:-default_on}"
echo "${DEVPROC2_CUTLASS_FP8_NT:-default_on}"
```

性能 profile 默认使用：

- CUDA Graph 开启，不加 `--no-graph`。
- `DEVPROC2_CUBLASLT_FP8_FAST_ACCUM` 默认开启。
- `DEVPROC2_CUTLASS_FP8_NT` 默认开启，但只有编译时 `DEVPROC2_WITH_CUTLASS=ON` 才生效。

## 最小命令清单

```bash
cd /root/tw/devproc2

export DEVPROC2_ROOT=/root/tw/devproc2
export OPENPI_ROOT=/root/tw/openpi
export OPENPI_PY=$OPENPI_ROOT/.venv/bin/python
export PI05_CKPT=/root/tools/pi05_libero_base
export PI05_TORCH_DUMP=/root/tw/openpi/outputs/pi05_torch_infer_libero_base
export PI05_TOKENIZER=$PI05_TORCH_DUMP/tokenizer.model
export PI05_WEIGHT_PKG=$PWD/build/pi05_fp8.weights
export PI05_DUMP_RAW=$PWD/build/pi05_torch_dump_oracle_libero_base
export DEVPROC2_FLASHRT_FA2_SO=/root/tw/FlashRT/flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
export DEVPROC2_LIBPYTHON_SO=/root/miniforge3/envs/py312/lib/libpython3.12.so.1.0
export PYTHONPATH=$PWD/python:${PYTHONPATH:-}

# 1. 用 openpi venv 生成同权重 PyTorch dump。
OMP_NUM_THREADS=1 PYTHONPATH=$OPENPI_ROOT:$OPENPI_ROOT/src "$OPENPI_PY" \
  "$OPENPI_ROOT/scripts/dump_pi05_torch_infer.py" \
  --ckpt "$PI05_CKPT" \
  --out "$PI05_TORCH_DUMP" \
  --device cuda \
  --num-examples 10 \
  --num-steps 10

# 2. 按“从 Dump 生成 Runtime Raw”一节生成 $PI05_DUMP_RAW。

# 3. 导出 P=968 full-token artifact。
python -m devproc2.models.pi05.export \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
  --weight-package-dir "$PI05_WEIGHT_PKG" \
  --tokenizer-model-path "$PI05_TOKENIZER" \
  --prefix-rows 968 \
  --max-prompt-len 200 \
  --num-views 3 \
  --sm-arch 89 \
  --use-static-act-scales

# 4. 编译 benchmark。
cmake --build build/root-cuda --target bench_pi05_denoise -j2

# 5. 跑 example0 benchmark。
export META=$PI05_DUMP_RAW/bf16_example0/metadata.json
export PREFIX_VALID_ROWS=$(python - <<'PY'
import json, os
print(json.load(open(os.environ["META"]))["prefix_valid_rows"])
PY
)

build/root-cuda/runtime/tests/bench_pi05_denoise 50 \
  --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
  --oracle-dir "$PI05_DUMP_RAW/bf16_example0/raw" \
  --max-prompt-len 200 \
  --prefix-valid-rows "$PREFIX_VALID_ROWS" \
  --num-views 3

# 6. 跑 nsys。
mkdir -p build/pi05_profiles
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o build/pi05_profiles/pi05_sample_tokens_3v968_same_weight_ex0 \
  build/root-cuda/runtime/tests/bench_pi05_denoise 20 \
    --entry-kind sample_tokens \
    --artifact-dir build/pi05_fp8_sample_tokens_3v968_artifact \
    --oracle-dir "$PI05_DUMP_RAW/bf16_example0/raw" \
    --max-prompt-len 200 \
    --prefix-valid-rows "$PREFIX_VALID_ROWS" \
    --num-views 3
```
