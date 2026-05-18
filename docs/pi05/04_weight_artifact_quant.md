# 权重映射、Artifact 与量化

## 当前实现状态

本阶段已经从“量化预留”推进到可执行的 Pi0.5 FP8 权重包与 artifact 资源装配：

- `devproc2.models.pi05.convert_pi05_weights(...)` 可将 `/root/autodl-tmp/tools/pi05-pytorch-base/model.safetensors` 转成 devproc2 权重包。
- RTX 4090 / SM89 使用 FP8 `nk` layout，与 FlashRT 的 4090 cuBLASLt `fp8_nt` 路径一致。
- 已生成本地包：`build/pi05_fp8.weights/`。
- 已生成本地 resource artifact：`build/pi05_fp8_artifact/`。
- artifact 包含 `weights/weights.bin`、`weights/weights.index.json`、`metadata/weight_map.json`、`metadata/quantization.json`、`resources/tokenizer.model`、`metadata/kernel_table.json`。
- C++ runtime 已实现 `WeightStore`，可加载 artifact 内权重 index，创建 CPU tensor view，并按需缓存到 CUDA device tensor。
- `WeightStore` 读取 3.7GB `weights.bin` 时使用一次性 sized read，避免 iterator 逐字节路径拖慢 artifact load。
- artifact resource install 会优先使用同盘 hardlink 安装 `weights.bin`，避免多个 Pi0.5 executable artifact 重复占用 3.7GB 权重空间；跨文件系统时回退到 copy。
- VM bytecode v3 已序列化函数参数名；`VMState::Invoke` 在少传参数时可从 artifact `WeightStore` 自动补齐同名权重参数。
- C++ runtime 已实现 tokenizer packed func，artifact 内 `resources/tokenizer.model` 优先于默认 oracle 路径；`test_pi05_sample_tokens_tokenizer` 已验证 tokenizer 生成 token ids 后可直接驱动 full `sample_tokens` artifact。
- CUDA source provider 可把 `python/devproc2/models/pi05/cuda/pi05_kernels.cu` 编译成 SM89 cubin，并写入 artifact `kernels/`。
- FlashRT vendored FA2 已通过 packed func `runtime.cuda.pi05_fa2_bf16` / `runtime.cuda.pi05_fa2_bf16_batched` 接入，denoise、prefix encoder 和 vision encoder fast path 默认走 FA2，BF16 attention kernel 仅作为 fallback/debug。runtime 提供 `DEVPROC2_FA2_SPLIT_Q_THRESHOLD` / `DEVPROC2_FA2_NUM_SMS` profile 开关；默认保持 FlashRT 原 split-KV 启发式，因为只对长 Q 禁用 split-KV 在完整 CUDA Graph 中回退。
- 动态 activation quant 已拆为 parallel amax reduce、scale materialize 和 static quant 三段；当前主性能 artifact 使用离线校准后的静态 activation scales，避免 runtime amax 成为 4090 主瓶颈。
- cuBLASLt FP8 GEMM 已支持 shape-level autotune、默认 FP8 FAST_ACCUM 性能模式，以及 `runtime.cuda.fp8_nt_bf16_accum` in-place residual accumulate，用于 prefix encoder O projection / FFN down 融合。`beta=1` accum 路径 autotune 已修正为调参前备份 `D`、调参后恢复，避免首次真实 inference residual 被 tune 覆盖。
- 可选 `DEVPROC2_WITH_CUTLASS=ON` build 已接入 SM89 CUTLASS FP8 NT -> BF16 GEMM prototype，runtime 通过 `DEVPROC2_CUTLASS_FP8_NT` 开关进入 shape-specialized path。当前只默认接管 vision FFN down 的 `m=512/768,n=1152,k=4304,beta=0` 形状，使用现有 device `A_scale/B_scale` 做双 scale epilogue；prefix FFN 的 plain CUTLASS probe 仍慢于或接近 cuBLASLt，所以不替换主路径。
- artifact 现已包含 action-expert denoise fast 子图、10-step unrolled denoise loop、precomputed-prefix `sample_actions` 后半段、prefix-embeddings 到 actions 的单 artifact，以及直接消费 `images_u8 + token_ids` 的 `sample_tokens` artifact。当前 tokenizer packed func 已可用，但完整 prompt/state 到 token ids 的前处理尚未并入 VM graph。
- runtime 已提供 `CUDAGraphExec` RAII wrapper；`bench_pi05_denoise` 通过该 API 做 CUDA Graph capture/replay，用于部署形态的 10-step denoise 延迟验证。
- prefix KV materialization fast path 已新增 `pi05_qkv_split_rope_cache_bf16`，decoder attention fast path 已新增 `pi05_qkv_split_rope_concat_bf16`；二者均在 DSL frontend 中通过 Pi0.5 CUDA helper 注入为 `CudaCallOp`，把 QKV split、Q/K RoPE、K/V cache 写回或 full-KV concat 合到单个 CUDA kernel；默认 `forward()` 仍保留标准 IR op 结构，快速路径放在 `forward_fast_dynamic()`。
- 已对 `/root/autodl-tmp/realtime-vla` 的 Pi0.5 Triton 实现做对照分析。该项目没有 FP8 量化，4090 上 Pi05 1/2/3-view 参考为 `22.1ms / 29.2ms / 38.9ms`。它的主要收益来自固定形状 Triton GEMM、GEMM tile 内 bias/residual/GELU/gate 融合、QKV+RoPE 写回融合、预计算 denoise style、action_out 融合 Euler `dt`、CUDA Graph capture，以及把 prefix/suffix KV 放入同一缓存布局。
- realtime-vla 中已经被当前 devproc2 采用的策略包括 CUDA Graph、precomputed denoise style、action_out `-1/num_steps` folding、PaliGemma prefix RMS scale folding、shape-level GEMM autotune、FlashRT FA2、静态 activation scale、FP8 FAST_ACCUM、vision QKV bias+split 小融合、vision FFN bias+GELU+static FP8 quant 小融合，以及 vision FFN down 的 shape-specialized CUTLASS FP8 route。
- realtime-vla 中已验证但当前不保留的策略包括：full KV cache 布局写 suffix K/V 到 prefix cache 后部（storage 从约 2.6MB 增至约 20.3MB，precomputed-prefix latency 变差）、独立 scalar fusion kernels（launch 数减少但 full path latency 变差）、vision residual 用 cuBLASLt `beta=1` 累加再单独加 bias（2-view 收益不稳定，3-view latency 和误差变差）、cuBLASLt bias epilogue（当前 row-major FP8/BF16 layout 下 heuristic 返回 status 15，不可用）。

当前实测本地 FP8 包：

```text
build/pi05_fp8.weights/
  weights.bin              3877659908 bytes
  weights.index.json       780 entries
  weight_map.json          780 entries
  quantization.json        253 entries
  convert_report.json      ruleset=openpi05_hf_to_devproc2_flashrt_v1
                           include_precomputed_styles=true
                           include_support_bf16=true
                           action_horizon=50
                           fp8_layout=nk
```

当前 `build/pi05_fp8_artifact/metadata/pi05_artifact.json` 记录：

```json
{
  "model": "openpi0.5",
  "target": "cuda",
  "sm_arch": 89,
  "weights": {
    "entries": 780,
    "precision": "fp8",
    "fp8_layout": "nk"
  },
  "kernels": {
    "count": 41,
    "compiled": true
  },
  "tokenizer": "resources/tokenizer.model"
}
```

当前 `build/pi05_fp8_artifact/executable.vm` 是 denoise fast 子图：

```text
entry: main
runtime inputs: 6
weight/scale params: 152
output: bfloat16[50, 32] action delta
VM instructions: 1632
temporary storage: 2687744 bytes
```

当前 `build/pi05_fp8_loop_artifact/executable.vm` 是 10-step denoise loop 子图：

```text
entry: main
runtime inputs: 5
weight/scale params: 152
output: float32[50, 32] actions
VM instructions: 16212
temporary storage: 2684416 bytes
```

当前 `build/pi05_fp8_sample_precomputed_prefix_artifact/executable.vm` 是 precomputed-prefix `sample_actions` 后半段：

```text
entry: main
runtime inputs: 5
first inputs: noise_f32, prefix_k_cache, prefix_v_cache, prefix_valid_rows, rope_interleaved
weight/scale params: 152
output: float32[50, 32] actions
VM instructions: 16212
```

当前 `build/pi05_fp8_sample_precomputed_prefix_embs_artifact/executable.vm` 是 prefix-embeddings 到 actions 的单 artifact 桥接路径：

```text
entry: main
runtime inputs: 5
first inputs: noise_f32, prefix_embs, prefix_valid_rows, prefix_rope_interleaved, suffix_rope_interleaved
weight/scale params: 431
output: float32[50, 32] actions
VM instructions: 10686
temporary storage: 108565504 bytes
kernels: 41
tokenizer: resources/tokenizer.model
```

当前推荐的 full-token artifact 是 `build/pi05_fp8_sample_tokens_127_artifact/executable.vm`：

```text
entry: main
runtime inputs: 6
first inputs: noise_f32, images_u8, token_ids, prefix_valid_rows, prefix_rope_interleaved, suffix_rope_interleaved
weight/scale params: 773
output: float32[50, 32] actions
VM instructions: 12389
temporary storage: 115201024 bytes
kernels: 41
tokenizer: resources/tokenizer.model
```

当前 `build/pi05_fp8_vision_encoder_executable/executable.vm` 是 SigLIP vision encoder prefix 切片：

```text
entry: main
runtime inputs: 1
first input: images_u8
weight/scale params: 341
output: bfloat16[768, 2048] image embeddings
VM instructions: 1702
temporary storage: 23949312 bytes
kernels: 41
```

当前 `build/pi05_fp8_paligemma_prefix_encoder_artifact/executable.vm` 是 compact PaliGemma prefix transformer 切片：

```text
entry: main
runtime inputs: 2
first inputs: prefix_embs, rope_interleaved
weight/scale params: 216
output: bfloat16[968, 2048] prefix hidden states
VM instructions: 583
temporary storage: 90697728 bytes
kernels: 41
tokenizer: resources/tokenizer.model
```

当前 `build/pi05_fp8_paligemma_prefix_kv_encoder_artifact/executable.vm` 是 compact PaliGemma prefix KV cache 切片：

```text
entry: main
runtime inputs: 3
first inputs: prefix_embs, prefix_valid_rows, rope_interleaved
weight/scale params: 207
outputs:
  bfloat16[18, 968, 1, 256] prefix_k_cache
  bfloat16[18, 968, 1, 256] prefix_v_cache
VM instructions: 606
temporary storage: 108044288 bytes
kernels: 41
tokenizer: resources/tokenizer.model
```

当前 RTX 4090 实测 denoise 子图：

```text
build/root-cuda/runtime/tests/test_pi05_denoise_oracle
  strict FP8 accumulation
  step0 abs_max=0.00868897 abs_mean=0.00144342
  10-step final_abs_max=0.078042 final_abs_mean=0.0104354
  VM-loop final_abs_max=0.078042 final_abs_mean=0.0104354
  bf16 multi-oracle count=10 worst_abs_max=0.149316 worst_abs_mean=0.0131491
  fp16 outputs compare count=10 worst_abs_max=0.165487 worst_abs_mean=0.0158259
  sample-precomputed-prefix count=10 same thresholds/metrics as denoise loop
  standalone prefix-kv raw-cache smoke is diagnostic; downstream action check is covered by sample-prefix-embs single artifact
  sample-prefix-embs single artifact example0 final_abs_max=0.199095 final_abs_mean=0.024508

build/root-cuda/runtime/tests/bench_pi05_denoise 30 --entry-kind sample_precomputed_prefix
  entry=sample_precomputed_prefix mode=cuda_graph mean_10step_ms=13.286 mean_step_ms=1.329

build/root-cuda/runtime/tests/bench_pi05_denoise 5 --no-graph
  entry=step mode=stream mean_10step_ms=71.003 mean_step_ms=7.100

build/root-cuda/runtime/tests/bench_pi05_denoise 30 --entry-kind sample_precomputed_prefix_embs
  entry=sample_precomputed_prefix_embs mode=cuda_graph mean_10step_ms=25.812 mean_step_ms=2.581

build/root-cuda/runtime/tests/bench_pi05_denoise 30 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_127_artifact \
  --max-prompt-len 127 --prefix-valid-rows 895 --num-views 3
  entry=sample_tokens mode=cuda_graph mean_10step_ms=28.548

build/root-cuda/runtime/tests/bench_pi05_denoise 50 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_2v562_cache_artifact \
  --max-prompt-len 50 --prefix-valid-rows 562 --num-views 2
  entry=sample_tokens mode=cuda_graph mean_10step_ms=23.425
```

realtime-vla 对齐实验：

```text
build/root-cuda/runtime/tests/bench_pi05_denoise 20 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v769_artifact \
  --max-prompt-len 1 --prefix-valid-rows 769 --num-views 3
  entry=sample_tokens mode=cuda_graph mean_10step_ms=29.246
```

说明：realtime-vla 的 3-view benchmark 默认 `prompt_len=0`，实际 prefix valid rows 约为 `3 * 256 + 1 = 769`。把 devproc2 full-token artifact 对齐到 `P=769` 后只带来约 `0.5ms` 级收益，所以当前差距不主要来自 prompt 长度或 prefix valid rows，而来自 vision/prefix GEMM 的 tile 级实现差异。

最新 realtime-vla audit profile（2-view/P=562，Nsight Systems）显示 kernel 时间仍集中在 cuBLASLt FP8 GEMM，而不是 tokenizer、attention fallback 或小 elementwise kernel：

```text
top CUDA kernels by total device time:
  sm89_xmma_gemm tilesize64x128x64       112 calls  16.531ms total
  sm89_xmma_gemm tilesize64x64x64        652 calls  12.767ms total
  sm89_xmma_gemm tilesize32x64x64       1179 calls  10.486ms total
  FlashRT FA2 split-k main               394 calls   4.381ms total
  pi05_geglu_to_fp8_bf16                 394 calls   2.149ms total
  pi05_gate_residual_ada_norm_to_fp8     360 calls   1.527ms total
  pi05_ada_rms_norm_style_to_fp8         360 calls   1.496ms total
```

对应 cuBLASLt shape tune log 中，prefix FFN 仍是最大单项：2-view `m=562,n=32768,k=2048` 约 `0.253ms/layer`，`m=562,n=2048,k=16384` 约 `0.146ms/layer`；3-view `m=895,n=32768,k=2048` 约 `0.385ms/layer`，`m=895,n=2048,k=16384` 约 `0.218ms/layer`。把 action horizon 从 50 降到 10 只把 2-view full-token 从约 `23.9ms` 降到 `23.388ms`；FA2 split-KV sweep 显示只对长 Q 禁用 split-KV 在 isolated microbench 中可能更快，但 full CUDA Graph 反而回退，因此默认保持 FlashRT split heuristic。realtime-vla 的 vision FFN down split/two-part 思路已先用 CUTLASS FP8 shape-specialized route 迁移：2-view/P=562 从 `23.653ms` 到 `23.368ms`，3-view/P=895 从 `28.766ms` 到 `28.533ms`，收益稳定但不是数量级变化。若继续压低 latency，需要把 prefix FFN 的 gate/up、down 也改成 GEMM tile 内 epilogue fusion 或更专门的 CUTLASS/CuTe/Triton kernel，而不是继续堆独立 CUDA elementwise fusion。

验证命令：

```bash
PYTHONPATH=python pytest \
  tests/compiler/test_pi05_weight_package.py \
  tests/compiler/test_pi05_artifact.py -q

cmake -S . -B build/root-cuda \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DDEVPROC2_WITH_CUTLASS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

ctest --test-dir build/root-cuda/runtime \
  -R 'test_pi05_artifact_load|test_pi05_weight_store|test_pi05_kernel_launch|test_pi05_cuda_gemm|test_pi05_denoise_oracle|test_pi05_sample_tokens_tokenizer' \
  --output-on-failure

ctest --test-dir build/root-tokenizers/runtime \
  -R test_pi05_tokenizer --output-on-failure
```

仍未完成：

- 直接消费 `images_u8 + token_ids` 的 `sample_tokens` VM executable 已生成；仍未完成的是把 prompt/state tokenizer 前处理也并入 VM graph，并输出最终部署级 C++ sample_actions API。
- prefix path 已接入：vision patch embedding、SigLIP vision encoder、language embedding、PaliGemma prefix transformer、prefix KV 生成和 denoise loop 均已有标准/fast DSL 切片；当前缺口是 tokenizer/state preprocessing 的整图 ABI 与更深的 prefix GEMM 优化。
- 权重 package 已可加载，并具备 VM 参数名到 artifact 权重的自动补齐路径；denoise 子图已具备真实参数名，完整 Pi0.5 model graph 尚未生成，所以还没有整图绑定验证。
- FP8 权重已量化，静态 activation scale 已写入权重包；denoise 子图在 strict FP8 accumulation 下已跑通 10 个 torch oracle 的 10-step loop，并与 fp16 torch outputs 做动作级对比；full `sample_tokens` example0 当前 `final_abs_max=0.159`、`final_abs_mean=0.026`，仍需多样本 actions 级报告。
- 非量化部分当前仍是 BF16 fast path，主要用于 torch oracle 对齐和现有辅助 kernel 过渡。4090 microbench 显示 Pi0.5 关键 GEMM 上 FP16/BF16 总体同级，FP16 仅在 2-view prefix FFN down 等局部 shape 有明显优势；后续 TODO 优先补 FP16 权重/runtime variant，对 embedding、norm、bias/small GEMM 等非 FP8 路径做 4090 实测取舍。
- denoise/precomputed-prefix sample_actions 后半段已达到 4090 上 `13.3ms` CUDA Graph replay（strict/oracle artifact）；prefix-embeddings single artifact 当前为 `25.8ms`；CUTLASS-enabled full 3-view `sample_tokens` 当前为 `28.548ms`，2-view/P=562 当前为 `23.425ms`。主要瓶颈已转移到 vision/prefix encoder FP8 GEMM。FlashRT PR #19 的 `~21ms` 对照使用 `num_views=2`、`num_steps=10`、`max_prompt_len=50`，且报告 GPU 是 RTX 4060 Ti；公开 PR head 缺失文件且不能按说明复现。本地 FlashRT main 4090 2-view sanity check median 为 `23.41ms`，与 devproc2 当前 `23.425ms` 同级，因此当前性能快照接受该水平。后续若继续优化，方向应转为 CUTLASS/CuTe/Triton 风格的 shape-specialized GEMM epilogue fusion，重点是 prefix FFN gate/up/down，而不是独立 elementwise launch fusion。

## 目标

openpi0.5 的原始权重来自 HuggingFace/PyTorch safetensors，但 devproc2 的编译和运行阶段不应该直接消费 HuggingFace 权重。需要在编译前增加独立的 `convert_weight` 环节，将外部 checkpoint 转换为 devproc2 自有权重格式并保存到本地。后续 compile、artifact emit、C++ runtime 都只依赖 devproc2 权重包。

同时，本设计需要能承载真实量化输出，而不是只预留 metadata。当前实现已经生成 SM89/RTX 4090 使用的 FP8 E4M3 权重包，并写入 weight scale 与 layout 信息；activation scale 已有动态/静态 quant kernel 路径，当前性能快照已接受 2-view/P=562 `23.425ms` 与 3-view/P=895 `28.548ms`。后续仍需补齐完整 Pi0.5 graph 上的多样本 actions 精度报告，并在不回退性能的前提下按 devproc2 设计重构 fast path。

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
- 量化 metadata 是 runtime ABI 的一部分；SM89 当前固定使用 FlashRT 对齐的 `nk` FP8 weight layout 与 `runtime.cuda.fp8_nt_bf16` GEMM。

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

## 量化 metadata

当前 `QuantSpec` 已用于真实 FP8 权重量化输出：

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

当前 FP8 示例：

```json
{
  "scheme": "fp8_e4m3_per_tensor",
  "storage_dtype": "fp8_e4m3",
  "compute_dtype": "bfloat16",
  "scale_name": "fp8.encoder_ffn_down_w_0.scale",
  "zero_point_name": null,
  "group_size": null,
  "axis": null,
  "packed_layout": "nk"
}
```

原则：

- 权重量化在 `convert_weight` 阶段显式完成，当前 SM89 路径使用 FP8 E4M3 per-tensor scale。
- 后续也可以接外部框架输出，例如 torchao，再通过 `convert_weight` 转成 devproc2 quantized weight package。
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
- FP8 quantized kernel/GEMM 能执行；完整 actions 精度仍需要 activation scale 校准和 E2E graph。
