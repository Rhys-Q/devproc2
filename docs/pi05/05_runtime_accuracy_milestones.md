# Runtime、精度对齐与实施阶段

## 当前实现状态

截至当前实现，Pi0.5 runtime 已具备以下可验证能力：

- `Executable::Load(artifact_dir)` 会加载 artifact 内 `weights/`，并在 CUDA build 下加载 `metadata/kernel_table.json` 注册 cubin。
- `ModelSession::LoadArtifact(artifact_dir)` 已封装 `Executable + VMState`，部署侧可直接持有 session 并调用 `Invoke(...)`。
- `WeightStore` 支持 `float16`、`float32`、`bfloat16`、`fp8_e4m3`、`uint8`、`int32` 的 CPU tensor view，并支持 CUDA device cache。
- VM bytecode v3 已包含函数参数名；`VMState::Invoke` 可将缺省参数按名称从 artifact 权重自动绑定成 CPU/CUDA tensor。
- `Parameter(name=...)` 显式命名现在会保留到 VM 参数表，用于将模块参数直接绑定到 artifact 权重名，例如 `fp8.encoder_ffn_gate_up_w_0.weight`。
- Pi0.5 fast modules 已支持显式 artifact 参数名；`PI05Linear` 和 `PI05FFN` 可在构造时传入 weight/scale 名称，避免真实模型接线时依赖 module path 猜测。
- `runtime.tokenizer.paligemma_encode` / `runtime.tokenizer.encode` / `runtime.tokenizer.paligemma_pi05_encode` 已接入 tokenizers-cpp，artifact 内 `resources/tokenizer.model` 优先于默认路径；`test_pi05_sample_tokens_tokenizer` 已覆盖 tokenizer 生成 token ids，再把 token ids 喂给 full `sample_tokens` artifact。
- `CUDAKernelLauncher_Launch` 已修正 driver API 参数打包：tensor 参数现在传递 device pointer 的地址，而不是误把 device pointer 当参数地址。
- CUDA packed GEMM 会继承 VM 默认 stream；`runtime.cuda.fp8_*`、FlashRT FA2 与手写 CUDA kernels 已能放进同一 CUDA Graph capture。
- `ModelSession::GetDefaultStream(...)` 已暴露 VM 默认 stream，部署工具可在同一 stream 上串接 side kernel、捕获 CUDA Graph 并 replay。
- `CUDAGraphExec` 已提供 runtime-level RAII wrapper，封装 capture、instantiate、upload、launch 和销毁，避免部署工具直接管理 `cudaGraph_t` 生命周期。
- 已对 `/root/autodl-tmp/realtime-vla` 的 Pi0.5 Triton runtime 做结构对照。realtime-vla 不做 FP8 量化，4090 Pi05 1/2/3-view 公开参考为 `22.1ms / 29.2ms / 38.9ms`，其性能主要来自 shape-specialized Triton GEMM、GEMM tile 内 epilogue fusion、QKV+RoPE 融合写回、预计算 style/action update 和 CUDA Graph，而不是量化。
- `TensorViewOp` / `dp.tensor_view(...)` 已进入 IR、VM builtin 和 alias analysis，用于 fast path 零拷贝切分 per-layer KV cache 与 per-step style table。
- Pi0.5 CUDA kernel catalog 已通过 frontend DSL 注册，当前包含 41 个 SM89 CUDA source kernels，覆盖 FlashRT Pi0.5 pipeline 的主要辅助 kernel，并提供 BF16 attention correctness fallback：
  - `pi05_image_u8_to_bf16_norm`
  - `pi05_cast_f32_to_bf16`
  - `pi05_embedding_gather_bf16`
  - `pi05_patch_im2col_bf16`
  - `pi05_rope_qwen3_bf16`
  - `pi05_qkv_split_bf16`
  - `pi05_qkv_bias_split_bf16`
  - `pi05_qkv_split_rope_bf16`
  - `pi05_qkv_split_rope_cache_bf16`
  - `pi05_qkv_split_rope_concat_bf16`
  - `pi05_kv_concat_bf16`
  - `pi05_copy_kv_cache_layer_bf16`
  - `pi05_layer_norm_bf16`
  - `pi05_layer_norm_to_fp8_bf16`
  - `pi05_rms_norm_bf16`
  - `pi05_rms_norm_unit_bf16`
  - `pi05_rms_norm_unit_to_fp8_bf16`
  - `pi05_rms_norm_to_fp8_bf16`
  - `pi05_residual_rms_norm_to_fp8_bf16`
  - `pi05_reduce_amax_bf16`
  - `pi05_amax_to_scale`
  - `pi05_bias_residual_bf16`
  - `pi05_bias_add_bf16`
  - `pi05_position_add_bf16`
  - `pi05_residual_add_bf16`
  - `pi05_gate_mul_residual_bf16`
  - `pi05_geglu_to_fp8_bf16`
  - `pi05_geglu_bf16`
  - `pi05_gelu_inplace_bf16`
  - `pi05_bias_gelu_to_fp8_bf16`
  - `pi05_ada_rms_norm_style_bf16`
  - `pi05_ada_rms_norm_style_to_fp8_bf16`
  - `pi05_gate_residual_ada_norm_to_fp8_bf16`
  - `pi05_quantize_fp8_static_bf16`
  - `pi05_quantize_fp8_dynamic_bf16`
  - `pi05_attention_bf16`（correctness fallback；性能目标路径已使用 vendored FA2 packed func）
  - `pi05_attention_prefix_bf16`（支持 prefix valid rows 的 correctness fallback）
  - `pi05_encoder_ffn_fp8_fused`（correctness/reference fallback；性能路径使用 split GEMM）
  - `pi05_euler_update_f32`
  - `pi05_euler_update_bf16`
- `PI05FFN.forward()` 保持标准 IR op，可读、可分析。
- `PI05Linear.forward_fast()` 现在使用 `runtime.cuda.bf16_nn_bf16`，用于 patch/action/output 等 BF16 投影。
- `PI05Attention.forward_fast()`、`PI05PaliGemmaEncoderLayer.forward_fast_dynamic(..., prefix_valid_rows=...)` 和 decoder attention 的性能路径现在使用 FlashRT vendored FA2 packed func `runtime.cuda.pi05_fa2_bf16`；`pi05_attention_bf16` / `pi05_attention_prefix_bf16` 作为 correctness fallback 和 debug kernel 保留。
- `PI05FFN.forward_fast()` 现在走显式 fast path：
  1. `dp.call_dps_kernel("pi05_quantize_fp8_static_bf16", ...)`
  2. `dp.call_dps_packed("runtime.cuda.fp8_nt_bf16", ...)`
  3. `dp.call_dps_kernel("pi05_geglu_to_fp8_bf16", ...)`
  4. `dp.call_dps_packed("runtime.cuda.fp8_nt_bf16", ...)`
- `PI05FFN.forward_fast_dynamic()` 现在走动态 activation scale 路径：`pi05_reduce_amax_bf16` 做 parallel amax、`pi05_amax_to_scale` materialize scale，再复用 static quant kernel，用于校准和没有静态 act scale artifact 时的 correctness fallback。
- CUDA packed func 已提供 `runtime.cuda.fp8_nt_bf16` / `runtime.cuda.fp8_nn_bf16`，底层为 cuBLASLt FP8 E4M3 -> BF16 GEMM，支持 SM89 `nk` weight layout；当前默认启用 FP8 FAST_ACCUM，可用 `DEVPROC2_CUBLASLT_FP8_FAST_ACCUM=0` 回退严格累积。
- CUDA packed func 已提供 `runtime.cuda.fp8_nt_bf16_accum` / `runtime.cuda.fp8_nn_bf16_accum`，用于把 FP8 GEMM 结果以 `beta=1` 直接累加到 residual tensor，当前接入 prefix encoder O projection 和 FFN down。
- FP8 GEMM runner 已做 shape-level cuBLASLt autotune，默认每个 shape 测 8 个 heuristic 候选；可用 `DEVPROC2_CUBLASLT_FP8_TUNE_ALGOS` / `DEVPROC2_CUBLASLT_FP8_TUNE_REPEATS` 调整。`beta=1` accum 路径的 autotune 已修正为调参前备份 `D`、调参后恢复，避免首次真实 inference 累加到被 tune 覆盖过的 residual。
- 可选 `DEVPROC2_WITH_CUTLASS=ON` build 已接入 SM89 CUTLASS FP8 NT -> BF16 shape-specialized path，runtime 通过 `DEVPROC2_CUTLASS_FP8_NT` 开关控制。当前只默认接管 vision FFN down `m=512/768,n=1152,k=4304,beta=0`，使用现有 device `A_scale/B_scale` 做双 scale epilogue；prefix FFN plain CUTLASS probe 暂未优于 cuBLASLt 主路径。
- CUDA packed func 已提供 `runtime.cuda.bf16_nn_bf16` / `runtime.cuda.bf16_nt_bf16`，底层为 cuBLASLt BF16 -> BF16 GEMM，补齐 patch/action/output 等非 FP8 投影路径。
- CUDA packed func 已提供 `runtime.cuda.pi05_fa2_bf16`，通过 `/root/autodl-tmp/FlashRT/flash_rt/libflashrt_fa2_raw.so` 调用 FlashRT vendored FA2 BF16 kernel。FA2 split-KV 可用 `DEVPROC2_FA2_SPLIT_Q_THRESHOLD` 和 `DEVPROC2_FA2_NUM_SMS` 做 profile sweep；默认保持 FlashRT 原启发式，实测只对长 Q 禁用 split-KV 会拖慢 full graph。
- `pi05_rms_norm_unit_bf16` 已补齐 Gemma/PaliGemma encoder 的 RMSNorm 基础路径；权重转换已将 `(1 + norm_weight)` 融入 encoder QKV/FFN FP8 权重，runtime 只需做 unit RMS 归一化。
- `PI05VisionPatchEmbedding.forward_fast()` 已通过 DSL 注入 image normalize、patch im2col、BF16 GEMM、bias add 和 position add；`PI05VisionEncoderLayer` / `PI05VisionEncoder` 已覆盖 SigLIP vision tower block、final norm 和 multimodal projector，并可导出独立 VM executable；`PI05LanguageEmbedding.forward_fast()` 已通过 DSL 注入 PaliGemma language embedding gather；`PI05PaliGemmaPrefixEncoder` 已覆盖 compact prefix transformer 的 unit RMSNorm、RoPE attention 和 FFN fast path。prefix KV materialization fast path 当前使用 `pi05_qkv_split_rope_cache_bf16`，decoder attention fast path 当前使用 `pi05_qkv_split_rope_concat_bf16`，分别把 QKV split、Q/K RoPE、K/V cache 写回或 full-KV concat 合到单个 DSL-injected CUDA kernel。
- `PI05DecoderLayer.forward_fast_dynamic()` 与 `PI05DenoiseStep.forward_fast_dynamic()` 已可通过 `GraphBuilder -> DPSLowering -> MemoryPlanning -> VMCodegen -> EmitABI` 导出真实 VM 子图。
- `build/pi05_fp8_artifact/executable.vm` 当前入口为 `main`，覆盖 18 层 action-expert denoise step；ABI 含 6 个运行时输入、152 个权重/scale 参数，输出 `[50, 32]` BF16 action delta，`main` 当前 1632 条 VM 指令、12 个 planned storage slot。
- `PI05DenoiseLoop.forward_fast_dynamic()` 已将固定 10-step Euler loop 静态展开到 DSL/VM；`build/pi05_fp8_loop_artifact` 当前入口为 `main`，ABI 含 5 个运行时输入、152 个权重/scale 参数，输出 `[50, 32]` float32 actions，`main` 当前 16212 条 VM 指令。
- `build/pi05_fp8_sample_precomputed_prefix_artifact` 已提供 `sample_actions` 后半段 ABI：输入 `noise_f32 + prefix_k_cache + prefix_v_cache + prefix_valid_rows + rope_interleaved`，输出 `[50, 32]` float32 actions；当前 152 个权重/scale 参数、16212 条 VM 指令。
- `build/pi05_fp8_sample_precomputed_prefix_embs_artifact` 已提供 prefix-embeddings 单 artifact 桥接路径：输入 `noise_f32 + prefix_embs + prefix_valid_rows + prefix_rope_interleaved + suffix_rope_interleaved`，在同一 VM graph 内完成 compact PaliGemma prefix KV materialization 和 10-step denoise，输出 `[50, 32]` float32 actions；当前 431 个权重/scale 参数、10686 条 VM 指令、108565504 bytes temporary storage。
- `build/pi05_fp8_sample_tokens_127_artifact` 已提供 full-token artifact：输入 `noise_f32 + images_u8 + token_ids + prefix_valid_rows + prefix_rope_interleaved + suffix_rope_interleaved`，在同一 VM graph 内完成 vision encoder、language embedding、prefix concat、prefix KV materialization 和 10-step denoise；当前 773 个权重/scale 参数、12389 条 VM 指令、115201024 bytes temporary storage。
- `build/pi05_fp8_vision_encoder_executable` 已提供 SigLIP vision encoder prefix slice ABI：输入 `images_u8`，输出 `[768, 2048]` BF16 image embeddings；当前 341 个权重/scale 参数、1702 条 VM 指令。
- `build/pi05_fp8_paligemma_prefix_encoder_artifact` 已提供 compact PaliGemma prefix transformer slice ABI：输入 `prefix_embs + rope_interleaved`，输出 `[968, 2048]` BF16 prefix hidden states；当前 216 个权重/scale 参数、583 条 VM 指令、90697728 bytes temporary storage。
- `build/pi05_fp8_paligemma_prefix_kv_encoder_artifact` 已提供 compact PaliGemma prefix KV cache slice ABI：输入 `prefix_embs + prefix_valid_rows + rope_interleaved`，输出 `[18, 968, 1, 256]` BF16 `prefix_k_cache` 和 `prefix_v_cache`；当前 207 个权重/scale 参数、606 条 VM 指令、108044288 bytes temporary storage。
- `devproc2.pi05.torch_oracle` 可从本地 openpi/PyTorch checkpoint dump denoise/prefix oracle，输出 `prefix_embs`、`prefix_rope_interleaved`、`prefix_k_cache`、`prefix_v_cache`、`prefix_valid_rows`、suffix `rope_interleaved`、`actions_f32` 和 torch target delta。
- `test_pi05_denoise_oracle` 已可加载真实 `build/pi05_fp8_artifact` / `build/pi05_fp8_loop_artifact` / `build/pi05_fp8_sample_precomputed_prefix_artifact` / `build/pi05_fp8_paligemma_prefix_kv_encoder_artifact` / `build/pi05_fp8_sample_precomputed_prefix_embs_artifact`，用 torch bf16 oracle 调用 VM `main`。当前 strict FP8 accumulation 下 step0 指标为 `abs_max=0.00868897`、`abs_mean=0.00144342`；example0 的 10-step closed-loop 和 VM 内 loop 指标均为 `final_abs_max=0.078042`、`final_abs_mean=0.0104354`；10-example bf16 multi-oracle 指标为 `worst_abs_max=0.149316`、`worst_abs_mean=0.0131491`；runtime-vs-fp16 torch outputs 指标为 `worst_abs_max=0.165487`、`worst_abs_mean=0.0158259`。standalone prefix KV artifact 目前作为 raw-cache smoke；单 artifact prefix-embs 路径为 `final_abs_max=0.199095`、`final_abs_mean=0.024508`。
- `WeightStore::Load` 已改为按文件 size 一次性读取 `weights.bin`，避免 3.7GB 权重包通过 iterator 读入导致分钟级加载。
- `bench_pi05_denoise` 已通过 runtime `CUDAGraphExec` 支持完整 10-step CUDA Graph capture/replay。RTX 4090 当前实测：precomputed-prefix 后半段 `13.286ms`（strict/oracle artifact），prefix-embeddings single artifact `25.812ms`，CUTLASS-enabled 3-view full `sample_tokens` `28.548ms`，2-view/P=562 full `sample_tokens` `23.425ms`。
- 非量化路径当前以 BF16 为主，主要服务 torch oracle 对齐和现有辅助 kernel 过渡。RTX 4090 上 Pi0.5 关键 GEMM microbench 显示 FP16/BF16 总体同级：3-view prefix FFN gate/up `0.768ms` vs `0.766ms`，down `0.383ms` vs `0.380ms`，vision QKV `0.049ms` vs `0.051ms`；2-view prefix FFN down FP16 有局部优势 `0.255ms` vs `0.292ms`。因此 FP16 variant 是后续兼容和局部 profile TODO，但不是当前性能闭环的主杠杆。
- realtime-vla 中已经被当前 runtime 采用的策略包括 CUDA Graph、precomputed style、action_out `-1/num_steps` folding、prefix RMS scale folding、FA2、shape-level FP8 GEMM autotune、静态 activation scale、FP8 FAST_ACCUM、vision QKV bias+split 小融合、vision FFN bias+GELU+static FP8 quant 小融合、vision FFN down shape-specialized CUTLASS FP8 route、prefix QKV split/RoPE/cache-write 小融合，以及 decoder QKV split/RoPE/full-KV concat 小融合。已尝试但不保留的策略包括 full KV cache 布局、独立 scalar fusion kernels、vision residual 用 cuBLASLt `beta=1` 累加后再单独加 bias，以及 cuBLASLt bias epilogue；full KV/scalar fusion 在当前 VM/storage/layout 下实测变慢，vision residual accumulate 对 2-view 收益不稳定且拖慢 3-view，bias epilogue 在当前 row-major FP8/BF16 layout 下无可用 heuristic。

已通过的关键 runtime 验证：

```bash
ctest --test-dir build/root-cuda/runtime \
  -R 'test_cuda_graph|test_pi05_artifact_load|test_pi05_weight_store|test_pi05_kernel_launch|test_pi05_cuda_gemm|test_pi05_denoise_oracle|test_pi05_sample_tokens_tokenizer' \
  --output-on-failure

ctest --test-dir build/root-tokenizers/runtime \
  -R test_pi05_tokenizer --output-on-failure
```

当前 4090 denoise 性能验证：

```bash
build/root-cuda/runtime/tests/test_cuda_graph
# test_cuda_graph passed

build/root-cuda/runtime/tests/bench_pi05_denoise 30 --entry-kind sample_precomputed_prefix
# pi05_denoise_bench iters=30 entry=sample_precomputed_prefix mode=cuda_graph mean_10step_ms=13.286 mean_step_ms=1.329 final_abs_max=0.070 final_abs_mean=0.011

build/root-cuda/runtime/tests/bench_pi05_denoise 30 --entry-kind sample_precomputed_prefix_embs
# pi05_denoise_bench iters=30 entry=sample_precomputed_prefix_embs mode=cuda_graph mean_10step_ms=25.812 mean_step_ms=2.581 final_abs_max=0.236 final_abs_mean=0.031

build/root-cuda/runtime/tests/bench_pi05_denoise 30 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_127_artifact \
  --max-prompt-len 127 --prefix-valid-rows 895 --num-views 3
# pi05_denoise_bench entry=sample_tokens mode=cuda_graph mean_10step_ms=28.548 final_abs_max≈0.14-0.16 final_abs_mean≈0.026-0.028

build/root-cuda/runtime/tests/bench_pi05_denoise 50 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_2v562_cache_artifact \
  --max-prompt-len 50 --prefix-valid-rows 562 --num-views 2
# pi05_denoise_bench entry=sample_tokens mode=cuda_graph mean_10step_ms=23.425

build/root-cuda/runtime/tests/bench_pi05_denoise 20 --entry-kind sample_tokens \
  --artifact-dir build/pi05_fp8_sample_tokens_3v769_artifact \
  --max-prompt-len 1 --prefix-valid-rows 769 --num-views 3
# pi05_denoise_bench iters=20 entry=sample_tokens mode=cuda_graph mean_10step_ms=29.246

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o build/pi05_graph_profile \
  build/root-cuda/runtime/tests/bench_pi05_denoise 3
```

realtime-vla 对照结论：

- 其 Pi05 benchmark 默认 `prompt_len=0`，`Pi05Inference` 初始化阶段把 `valid_encoder_len` 设为 `num_views * 256 + 1`；3-view 即 `769` rows。devproc2 对齐到 `P=769` 后只比 `P=895` 快约 `0.5ms`，所以 prompt length 不是主要瓶颈。
- realtime-vla 的有效优势在 GEMM 内部融合：bias/residual/GELU/gate 在 tile 内完成，QKV matmul 直接做 RoPE 并写入 Q/K/V，vision FFN down 对部分 shape 使用 split-K/two-part 写法。当前 devproc2 已用 CUTLASS FP8 shape-specialized route 迁移 vision FFN down 思路，2-view/P=562 从 `23.653ms` 到 `23.368ms`，3-view/P=895 从 `28.766ms` 到 `28.533ms`；独立 CUDA elementwise fusion 降低了 launch 数但没有降低端到端 latency，说明下一步若继续优化，应把融合移入 GEMM epilogue 或写 shape-specialized CUTLASS/CuTe/Triton kernel。
- full KV cache 布局虽然接近 realtime-vla，但在 devproc2 当前 compact prefix cache + FlashRT FA2 路径下增加 storage 并拖慢 precomputed-prefix，因此已回退。KV 布局是否再做，需要和新的 attention/GEMM kernel 一起重评。
- 最新 2-view/P=562 Nsight Systems audit 中，top device time 仍由 cuBLASLt FP8 GEMM 占据；prefix FFN tune log 为 `m=562,n=32768,k=2048` 约 `0.253ms/layer`、`m=562,n=2048,k=16384` 约 `0.146ms/layer`。3-view/P=895 对应约 `0.385ms/layer` 和 `0.218ms/layer`。把 action horizon 从 50 降到 10 只把 2-view full-token 从约 `23.9ms` 降到 `23.388ms`。FA2 split-KV sweep 显示只对长 Q 禁用 split-KV 在 isolated microbench 中可能更快，但 full CUDA Graph 反而回退，因此默认保持 FlashRT split heuristic。因此进一步压低 latency 的主要方向是 GEMM tile-level epilogue fusion/shape specialization，不是 prompt 长度、action horizon、tokenizer、attention fallback 或独立小 kernel launch。

这些测试分别覆盖：

- artifact load 会注册 CUDA kernel table 并加载 WeightStore；
- 真实 Pi0.5 cubin 的 image normalize、BF16 LayerNorm、QKV split/RoPE、QKV split/RoPE/cache-write、QKV split/RoPE/full-KV concat、KV concat、prefix KV cache copy、bias add、Euler BF16 update、attention fallback、parallel amax/scale、dynamic FP8 quantize 等 kernels 可 launch 并通过小规模数值检查；
- `runtime.cuda.fp8_nt_bf16` 和 `runtime.cuda.bf16_nn_bf16` 可在 CUDA/cuBLASLt 上完成小矩阵 GEMM 并对齐 CPU reference；
- 真实 denoise VM 子图、unrolled denoise-loop 子图和 precomputed-prefix sample_actions 子图可消费 torch oracle prefix/action 输入；strict FP8 accumulation 下 step0、host 10-step closed-loop、VM 内 10-step loop、10-example bf16 multi-oracle 和 fp16 outputs comparison 精度已进入可用范围；
- standalone prefix KV artifact 当前作为 raw-cache smoke；prefix-embeddings single artifact 可消费 torch oracle prefix embeddings/RoPE 输入，并在单 VM graph 内生成 prefix KV 后接 denoise loop，example0 actions 精度已进入当前阈值；
- tokenizers-cpp 对本地 PaliGemma `tokenizer.model` 的输出与 openpi prompt token 序列一致，并已验证 tokenizer 生成的 127-token prompt 可直接驱动 `build/pi05_fp8_sample_tokens_127_artifact`。
- `CUDAGraphExec` unit test 覆盖 wrapper capture/upload/replay；denoise 后半段 benchmark 可捕获并 replay 10-step denoise loop，在 4090 上低于 21ms 目标。

仍未达成的验收项：

- 直接消费 `images_u8 + token_ids` 的 Pi0.5 `sample_tokens` executable 已生成；仍未完成的是 prompt/state tokenizer 前处理并入 VM graph、最终 C++ sample_actions API，以及 full path 多样本 actions 级报告。
- precomputed-prefix denoise 后半段已与 `/root/autodl-tmp/openpi/outputs/pi05_torch_infer/fp16/outputs.npz` 做 runtime-vs-fp16 actions 对比；完整 vision/text `sample_actions` 尚未做整图 actions 对齐。
- denoise oracle 已收紧到 step0 和 10-step thresholds；完整 sample_actions 仍需要 actions 级正式报告。
- 离线/静态 activation scale 已接入权重包并用于主性能 artifact；full path 仍需要 10-example 级 actions 精度报告和局部 FP16/动态 scale 取舍。
- CUDA Graph replay 已固化为 runtime wrapper；prefix-embeddings single artifact 已可 capture/replay，当前 `25.812ms`；CUTLASS-enabled 3-view full `sample_tokens` 当前 `28.548ms`，2-view/P=562 当前 `23.425ms`。nsys 显示主瓶颈为 vision/prefix encoder FP8 GEMM，而不是 attention fallback。
- FlashRT PR #19 的 `~21ms` 对照使用 `num_views=2`、`num_steps=10`、`max_prompt_len=50`，并报告在 RTX 4060 Ti 上测得；公开 PR head 缺失 `pi05_sm89_fp8_ffn.py` 且调用未绑定的 `cutlass_ada_fp8_gemm_bf16_simple`，不能按说明复现。本地 FlashRT main 4090 2-view sanity check median 为 `23.41ms`，与 devproc2 当前 `23.425ms` 同级。当前分支接受该性能，下一阶段重点是保持性能并按 devproc2 设计重构。

## C++ Runtime 目标

最终运行方式：

```python
import devproc2.runtime as rt

runner = rt.load_artifact("build/pi05_fp16")
actions = runner.run({
    "base_0_rgb": base,
    "left_wrist_0_rgb": left,
    "right_wrist_0_rgb": right,
    "state": state,
    "noise": noise,
    "prompt_tokens": tokens,
    "prompt_mask": mask,
})
```

Python 只负责调用 C++ 入口。实际执行不依赖 Python，不加载 PyTorch，不读取原始 safetensors。

## Artifact 加载流程

C++ `Executable::Load(artifact_dir)` 扩展为：

1. 读取 `executable.vm`。
2. 读取并校验 `abi.json`。
3. 读取 `metadata/kernel_table.json`，加载 cubin 并注册 kernel。
4. 读取 `metadata/weight_map.json` 和 `weights/weights.index.json`。
5. 读取 `weights/weights.bin`，创建 host/device weight tensor。
6. 创建 `ModelSession`，持有 executable、VMState、weights、kernels、default stream。

当前已新增：

```cpp
class ModelSession {
public:
    static ModelSession LoadArtifact(const std::string& artifact_dir);
    VMValue Invoke(const std::string& func_name, std::vector<VMValue> args = {});
    void* GetDefaultStream(const Device& dev);
};
```

## 输入输出 ABI

首版可将 tokenizer 和图像预处理留在 Python 测试 harness 中，C++ 入口接收模型数值输入：

- images：已 resize/normalize 后的 tensor，shape 与 openpi preprocessing 对齐。
- prompt tokens：int tensor。
- prompt mask：bool tensor。
- state：float32 tensor。
- noise：float32 tensor。

后续再把 tokenizer 和 image preprocessing 纳入 C++ 或独立前处理 artifact。

输出：

- `actions`: float32 tensor，shape `[B, action_horizon, action_dim]`。

## 精度对齐 oracle

使用：

```bash
cd /root/autodl-tmp/openpi
uv run python scripts/check_pi05_fp16_infer.py \
  --ckpt /root/autodl-tmp/tools/pi05-pytorch-base \
  --dump-dir outputs/pi05_torch_infer \
  --device cuda \
  --num-steps 10 \
  --rtol 1e-2 \
  --atol 1e-2
```

devproc2 对齐目标：

- 完整 actions `np.allclose(actual, expected, rtol=1e-2, atol=1e-2)`。
- 同时记录 `abs_max`、`abs_mean`、`abs_p95`、`rel_mean`。

## 分层对齐流程

### Level 1：单 op

每个 kernel 独立与 PyTorch 对齐：

- `matmul`
- `embedding`
- `layer_norm`
- `rms_norm/adarms_norm`
- `gelu_tanh`
- `silu`
- `rope`
- attention 标准 op 序列：`matmul + mask + softmax + matmul`

### Level 2：基础 block

对齐：

- MLP block：`matmul/add -> gelu -> matmul/add`
- self-attention block：qkv projection、rope、`matmul + mask + softmax + matmul`、o_proj
- norm + gated residual

### Level 3：denoise step

输入使用 PyTorch dump 的：

- state
- prefix pad masks
- past_key_values
- x_t
- timestep

输出对齐 `v_t`。

### Level 4：Euler loop

固定 `num_steps=10`，逐 step 对齐：

- 每一步 `timestep`
- 每一步 `v_t`
- 每一步更新后的 `x_t`

### Level 5：完整 sample_actions

完整输入到 actions 对齐。

## Debug dump 机制

建议编译和运行支持 debug mode：

- 按 op name/path dump tensor。
- dump 文件使用 `.npz`，key 为 stable value path。
- PyTorch 和 devproc2 使用相同 key 命名，便于自动 diff。
- 默认不启用，避免影响性能和 artifact 大小。

## 实施阶段

### M0：文档和 oracle 固化

产出：

- 本目录设计文档。
- openpi0.5 推理路径说明。
- PyTorch oracle 命令和阈值。

验收：

- 另一个工程师可以按文档开始实现，不需要重新判断接口边界。

### M1：前端最小可用

产出：

- `devproc2.nn.Module`
- `Parameter/Weight`
- `Linear/Embedding/LayerNorm/RMSNorm/GELU/SiLU`
- stable `named_parameters()`

验收：

- 可构建 openpi0.5 denoise path 的模块骨架。

### M2：IR attr 与 op schema

产出：

- `CallOp.attrs` / `CallDPSOp.attrs`
- AttrValue printer/serializer/verifier
- 基础 op schema 和 shape infer
- 每个 tensor op 必须有独立 infer shape/struct info 函数；缺失 infer 的 op 不能 lowering

验收：

- `matmul/add/layer_norm/gelu/cat/reshape` 的 IR 可打印、可 infer shape、可 lower，attrs 不丢失。

### M3：convert_weight 和权重 artifact

产出：

- `WeightSpec`
- `convert_weight` 工具：safetensors checkpoint 到 devproc2 weight package
- devproc2 weight package manifest
- `weights.bin`
- `weights.index.json`
- `metadata/weight_map.json`

验收：

- compile 只接受 devproc2 weight package，拒绝直接读取 safetensors。
- C++ runtime 能加载最小权重 artifact。
- shape/dtype mismatch 会报错。

### M4：基础 kernel 与 C++ dispatch

产出：

- elementwise、matmul、embedding、norm kernel
- kernel table
- cubin 加载注册

验收：

- C++ VM 能调用 artifact 内 cubin。
- 单 op 与 PyTorch 对齐。

### M5：attention pattern 和 denoise step

产出：

- rope kernel
- attention 标准 op 序列；可选 fused attention pattern lowering
- KV cache 表达和读取
- denoise step 编译：当前已生成 action-expert denoise fast 子图 VM executable

验收：

- 单次 denoise step 输出 `v_t` 与 PyTorch 对齐。当前已有可运行子图导出；strict FP8 accumulation 下 step0 `abs_max=0.00868897`、10-step closed-loop `final_abs_mean=0.0104354`。

### M6：完整 sample_actions

产出：

- 固定 `num_steps=10` 的 Euler loop。
- prefix KV cache。
- 完整 artifact 编译。当前 action-expert denoise loop executable、precomputed-prefix 后半段 artifact、prefix KV artifact 和 prefix-embeddings single artifact 已生成；raw images/tokens/state/noise 的完整 model ABI 仍未接入。

验收：

- `actions` 与 PyTorch fp16 dump 在 `rtol=1e-2, atol=1e-2` 内。
- 当前 prefix-embeddings single artifact example0 已达 `final_abs_max=0.199095`、`final_abs_mean=0.024508`；10-example 级别和 raw-input 完整路径仍待补齐。

### M7：稳定化

产出：

- debug dump 工具。
- 性能基线。
- 错误信息整理。
- 文档更新为实际实现状态。
- 将 CUDA Graph replay API 接入完整 sample_actions runner。
- 非量化 BF16 路径的 FP16 variant 与 4090 profile 对比。

验收：

- CI 或本地脚本可以一键运行 openpi0.5 fp16 对齐测试。
- 4090 上完整 sample_actions 端到端达到当前接受的目标延迟；当前 precomputed-prefix 后半段为 `13.286ms`，prefix-embeddings single artifact 为 `25.812ms`，CUTLASS-enabled 3-view full `sample_tokens` 为 `28.548ms`，2-view/P=562 full `sample_tokens` 为 `23.425ms`。后续重点是保持该性能并按 devproc2 设计重构；若继续压低 latency，再针对 vision/prefix encoder FP8 GEMM 做 tile-level epilogue fusion 和 shape-specialized GEMM profile。

## 风险与默认策略

- Vision tower kernel 复杂度高：先把 prefix path 拆出来逐层对齐，必要时短期用 precomputed prefix embeddings/KV cache 解锁 denoise path。
- attention 精度敏感：首版使用 correctness-first 标准 op 序列，确认精度后再做 pattern fusion。
- dynamic shape 增加复杂度：首版固定 batch/image/action horizon/num_steps，metadata 记录约束。
- FP8 权重量化已经进入 runtime 路径；剩余风险在完整 graph 的 activation scale 校准、完整 sample_actions 接线和 actions 级精度/性能验收。
- FP16 非量化路径作为后续优化项保留；4090 microbench 显示 FP16/BF16 Tensor Core 路径总体同级，FP16 不是当前性能闭环的主杠杆。后续应先以现有 BF16/FP8 fast path 建立可运行基线，再用 FP16 variant 对局部 embedding、norm、bias/small GEMM 做替换和 profile 对比。
