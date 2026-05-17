# Pi0.5 性能快照与优化结论

## 快照范围

本文件记录当前 Pi0.5 性能分支的工程状态，目标是为下一阶段重构提供边界：当前实现优先证明 4090 上的可运行性能，不代表最终 devproc2 设计形态。后续重构应保持这里的性能基线，同时把 fast path、kernel 注入和部署 ABI 收回到更干净的 devproc2 架构中。

当前接受的性能口径是 RTX 4090 / SM89、batch size 1、`num_steps=10`、CUDA Graph replay：

| Path | Artifact / config | Latency |
| --- | --- | --- |
| denoise 后半段 | `sample_precomputed_prefix` | `13.286ms` |
| prefix embeddings 到 actions | `sample_precomputed_prefix_embs` | `25.812ms` |
| full token 2-view | `sample_tokens_2v562_cache`, `P=562`, `max_prompt_len=50` | `23.425ms` |
| full token 3-view | `sample_tokens_127`, `P=895`, `max_prompt_len=127` | `28.548ms` |

FlashRT PR #19 的 `~21ms` 口径已经验证为不可直接复现：公开 PR head 缺少 `flash_rt.frontends.torch.pi05_sm89_fp8_ffn`，且 pipeline 内调用未绑定的 `cutlass_ada_fp8_gemm_bf16_simple`。本地 FlashRT main 的可运行 2-view sanity check 在 4090 上 median 为 `23.41ms`，与 devproc2 当前 `23.425ms` 同级。因此这个分支接受 `~23.4ms` 作为当前 2-view 性能基线。

## 已采用的优化

**Artifact 与权重**

- 使用独立 `convert_weight` 生成 devproc2 权重包，runtime 不直接读取 HuggingFace/PyTorch checkpoint。
- RTX 4090 / SM89 使用 FP8 E4M3 `nk` weight layout，对应 cuBLASLt `fp8_nt` 访问路径。
- 权重包内保存静态 activation scale，主性能 artifact 避免 runtime dynamic amax。
- `WeightStore` 支持 artifact 内权重 index、按需 CUDA device cache，以及 `weights.bin` sized read；artifact 安装同盘 hardlink，避免多个 executable 重复占用 3.7GB 权重。
- VM bytecode v3 保留参数名，`VMState::Invoke` 可按名称自动从 artifact 权重补齐缺省参数。

**Frontend / IR / DSL**

- 标准 `forward()` 仍保留标准 IR op 结构，便于阅读、分析和后续重构。
- 性能路径放在 `forward_fast()` / `forward_fast_dynamic()`，CUDA source kernel 通过 Pi0.5 helper 展开为 `dp.cuda_call(...)` 无注册接入，cuBLASLt/FA2/CUTLASS runtime func 继续通过 `dp.call_dps_packed(...)` 接入。
- `TensorViewOp` / `dp.tensor_view(...)` 已用于 fast path 中零拷贝切分 per-layer KV cache 与 per-step style table。
- Pi0.5 CUDA kernel catalog 当前包含 41 个 SM89 CUDA source kernels，覆盖 image normalize、embedding gather、patch im2col、norm、RoPE、QKV split、KV cache、GeGLU、AdaRMSNorm、static/dynamic FP8 quant、Euler update 和 attention correctness fallback。

**GEMM / 量化**

- cuBLASLt FP8 GEMM 已接入 `runtime.cuda.fp8_nt_bf16` / `runtime.cuda.fp8_nn_bf16`，并支持 shape-level autotune。
- 默认启用 FP8 FAST_ACCUM；严格累积可用 `DEVPROC2_CUBLASLT_FP8_FAST_ACCUM=0` 回退。
- `runtime.cuda.fp8_*_accum` 用于 O projection / FFN down 的 `beta=1` residual accumulate；autotune 期间会备份和恢复 `D`，避免首次真实 inference residual 被调参覆盖。
- 非量化投影使用 cuBLASLt BF16 packed func，补齐 patch/action/output 等路径。
- 可选 `DEVPROC2_WITH_CUTLASS=ON` 接入 SM89 CUTLASS FP8 NT -> BF16 path，目前只默认接管 vision FFN down `m=512/768,n=1152,k=4304,beta=0`。A/B 结果：2-view `23.653ms -> 23.368ms`，3-view `28.766ms -> 28.533ms`。

**Attention / KV**

- 性能路径使用 FlashRT vendored FA2 BF16 packed func `runtime.cuda.pi05_fa2_bf16` / `runtime.cuda.pi05_fa2_bf16_batched`。
- `pi05_attention_bf16` 和 `pi05_attention_prefix_bf16` 只作为 correctness fallback/debug kernel 保留。
- Prefix KV materialization 使用 `pi05_qkv_split_rope_cache_bf16`，把 QKV split、Q/K RoPE 和 K/V cache 写回合为单 kernel。
- Decoder attention 使用 `pi05_qkv_split_rope_concat_bf16`，把 QKV split、Q/K RoPE 和 prefix+suffix full-KV concat 合为单 kernel。

**Denoise / graph**

- 10-step denoise loop 已静态展开到 DSL/VM。
- `action_out_proj` 权重/bias 融合 Euler `dt=-1/num_steps`，减少每步输出后的独立 scale/update 逻辑。
- decoder time/style 已预计算并写入 artifact，runtime 只按 step/layer 切片。
- `CUDAGraphExec` 封装 capture、instantiate、upload、launch 和销毁；`bench_pi05_denoise` 使用它验证部署形态 replay latency。

**Tokenizer**

- tokenizers-cpp 已作为 runtime packed func 接入。
- artifact 内 `resources/tokenizer.model` 优先于默认 oracle 路径。
- `test_pi05_sample_tokens_tokenizer` 已验证 tokenizer 生成 token ids 后可以直接驱动 full `sample_tokens` artifact。

## 验证后未采用的结论

- FlashRT PR #19 不能作为可复现硬基线。公开 PR head 缺文件且缺 pybind binding，不能按 PR 说明实例化运行；其 `MSE=0, Cosine=1.0` 也不是从公开脚本实际计算出的证明。
- realtime-vla 本身没有 FP8 量化；它的价值是 shape-specialized Triton GEMM 与 tile-level epilogue fusion 思路，而不是可以直接复制的量化方案。
- full KV cache 布局曾尝试把 suffix K/V 写到 prefix cache 后部，但 storage 从约 2.6MB 增至约 20.3MB，precomputed-prefix latency 变差，因此回退。
- 独立 scalar fusion kernels 虽减少 launch 数，但 full path latency 变差；当前瓶颈已不在这些小 kernel 的 launch overhead。
- vision residual 使用 cuBLASLt `beta=1` accumulate 后再单独加 bias，对 2-view 收益不稳定，3-view latency 和误差变差，因此不保留。
- cuBLASLt bias epilogue 在当前 row-major FP8/BF16 layout 下 heuristic 返回不可用状态，不能作为主路径。
- FA2 split-KV sweep 显示只对长 Q 禁用 split-KV 在 isolated microbench 中可能更快，但 full CUDA Graph 反而回退，因此保留 FlashRT 默认 split heuristic。
- 把 action horizon 从 50 降到 10 只让 2-view full-token 从约 `23.9ms` 到 `23.388ms`，不是主要杠杆。
- prompt length / prefix valid rows 不是主要瓶颈。3-view 对齐 realtime-vla 的 `P=769` 后只得到约 `0.5ms` 收益。
- FP16 与 BF16 在 4090 Pi0.5 关键 GEMM 上总体同级；FP16 只在个别 2-view prefix FFN down shape 有局部优势，因此 FP16 variant 是后续 TODO，不是当前性能闭环主路径。
- plain CUTLASS FP8 prefix FFN probe 暂未优于 cuBLASLt，当前只保留 vision FFN down 的 shape-specialized CUTLASS route。

## Profile 结论

当前 full-token profile 的主瓶颈是 vision/prefix encoder FP8 GEMM，尤其是 prefix FFN gate/up 与 down，而不是 tokenizer、attention fallback、action horizon 或独立 elementwise kernel。

2-view/P=562 的 Nsight Systems top device time 仍集中在 cuBLASLt FP8 GEMM：

```text
sm89_xmma_gemm tilesize64x128x64       112 calls  16.531ms total
sm89_xmma_gemm tilesize64x64x64        652 calls  12.767ms total
sm89_xmma_gemm tilesize32x64x64       1179 calls  10.486ms total
FlashRT FA2 split-k main               394 calls   4.381ms total
pi05_geglu_to_fp8_bf16                 394 calls   2.149ms total
pi05_gate_residual_ada_norm_to_fp8     360 calls   1.527ms total
pi05_ada_rms_norm_style_to_fp8         360 calls   1.496ms total
```

继续优化时，优先级应该是把 FFN gate/up、GeGLU、down、bias/residual 合到 GEMM tile-level epilogue，或写 shape-specialized CUTLASS/CuTe/Triton kernel。继续增加独立 CUDA elementwise fusion 不是当前方向。

## 设计债务

当前分支是性能快照，不是最终 devproc2 形态。主要设计债务：

- runtime packed func、CUDA kernel 和 artifact ABI 已经为 Pi0.5 shape 做了大量特化，需要后续抽成 backend capability 与 kernel selection，而不是散落在模块实现和 env 开关里。
- `forward_fast()` 已证明手动 fast path 可行，但后续需要更明确地区分标准 IR、手写 CUDA、Triton、CuTeDSL 和 CUTLASS backend。
- `sample_tokens` / prefix artifacts 仍偏 benchmark 驱动，最终需要统一成部署级 C++ `sample_actions` API。
- tokenizer/state preprocessing 还未进入整图 ABI。
- CUTLASS path 当前是可选 prototype，只覆盖 vision FFN down；后续应将 shape-specialized GEMM 纳入正式 kernel registry 和 profile database。
- FlashRT FA2 仍以 vendored `.so` packed func 方式接入，后续需要整理为可配置依赖或本仓统一 build target。

下一阶段重构目标：保持当前 2-view `~23.4ms` 和 3-view `~28.5ms` 性能不回退，同时把这些性能路径重新落到 devproc2 的标准设计上。

## 常用验证命令

```bash
cmake -S . -B build/root-cuda \
  -DDEVPROC2_WITH_CUDA=ON \
  -DDEVPROC2_WITH_TOKENIZERS=ON \
  -DDEVPROC2_BUILD_TESTS=ON \
  -DDEVPROC2_WITH_CUTLASS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build/root-cuda \
  --target bench_pi05_denoise test_pi05_cuda_gemm test_pi05_denoise_oracle -j2

ctest --test-dir build/root-cuda/runtime \
  -R 'test_pi05_cuda_gemm|test_pi05_denoise_oracle' \
  --output-on-failure

build/root-cuda/runtime/tests/bench_pi05_denoise 50 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_2v562_cache_artifact \
  --max-prompt-len 50 --prefix-valid-rows 562 --num-views 2

build/root-cuda/runtime/tests/bench_pi05_denoise 50 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_127_artifact \
  --max-prompt-len 127 --prefix-valid-rows 895 --num-views 3
```
