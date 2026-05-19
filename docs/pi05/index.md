# openpi0.5 支持方案总览

## 背景

目标是在 devproc2 中支持 openpi0.5 模型的编译和纯 C++ 推理运行。当前本机操作默认复用已有 PyTorch dump 和 tokenizer，不安装、不更新、不修改 openpi 的 uv 环境：

- torch 对点数据和 tokenizer：`/root/tw/openpi/outputs/pi05_torch_infer`
- checkpoint：`/root/tools/pi05_libero_base/model.safetensors`

当前 PyTorch dump 包含 `inputs.npz`、`fp16/outputs.npz`、`bf16/outputs.npz` 和 `tokenizer.model`。devproc2 benchmark 侧还需要 `build/pi05_torch_denoise_oracle/bf16_example0/raw` 下的 raw 对点文件；如果缺失，应复用已有 raw dump 或由维护者在已配置好的 openpi 环境中生成，不在本流程里执行 openpi `uv` 安装/更新。

本目录最初描述设计方案和实施阶段；当前实现已开始落地，状态记录见
`04_weight_artifact_quant.md` 和 `05_runtime_accuracy_milestones.md` 顶部的
“当前实现状态”。性能快照和优化取舍总结见
`06_performance_snapshot.md`。

## 当前 devproc2 状态

已有能力：

- Python DSL 可通过 `@dp.function` 捕获函数 AST，生成 `IRModule`、`Function`、`CallOp`、`CallDPSOp`、`TensorCreateOp` 等 IR。
- 已有 `TensorStructInfo`，可表达 shape、dtype、device。
- 已有 DPS lowering：`CallOp` 可被 lower 为 `TensorCreateOp + CallDPSOp(kernel)`。
- 已有 `KernelRegistry`、`KernelSpec`、`@dp.kernel` 和 CUDA source provider，可把 Pi0.5 CUDA source 编译为 SM89 cubin 并写入 artifact。
- 已有 VM bytecode、artifact `executable.vm`、`abi.json`，以及 C++ `Executable::Load` / `VMState::Invoke`。
- C++ runtime 已支持 builtin、packed_func、kernel 三类外部调用。
- 已生成本地 Pi0.5 FP8 权重包和自包含 resource artifact：`build/pi05_fp8.weights/`、`build/pi05_fp8_artifact/`。
- RTX 4090/SM89 FP8 权重 layout 为 `nk`，已接 cuBLASLt FP8/BF16 packed GEMM。
- 已接 tokenizers-cpp，artifact 内 `resources/tokenizer.model` 可作为 runtime tokenizer source；tokenizer 生成 token ids 后直连 full `sample_tokens` artifact 的 C++ 测试已覆盖。
- 已有 41 个 Pi0.5 CUDA source kernels，包括 image normalize、language embedding gather、vision patch im2col、position add、LayerNorm/RMSNorm、norm+static FP8 quant、unit RMSNorm、parallel activation amax/scale、quant、GeGLU、vision QKV bias+split、bias+GELU+FP8、QKV split/RoPE、QKV split/RoPE 直接写 prefix KV cache、decoder QKV split/RoPE/full-KV concat 融合、KV concat、prefix KV cache layer copy、AdaRMSNorm、bias add、Euler update 和 prefix-valid BF16 attention correctness fallback。
- 已有 `PI05Linear.forward_fast`、`PI05FFN.forward_fast`、`PI05FFN.forward_fast_dynamic`、`PI05Attention.forward_fast` 等 DSL-injected fast path。
- `WeightStore`、VM 参数名自动绑权重、`ModelSession::LoadArtifact` 已进入 C++ runtime。
- 已有 `TensorViewOp` / `dp.tensor_view(...)`，用于在 fast path 中零拷贝切分 per-layer KV cache 和 per-step style table。
- 已生成 `build/pi05_fp8_artifact/executable.vm` denoise fast 子图：入口 `main`，6 个运行时输入，152 个权重/scale 参数，输出 `[50, 32]` BF16 action delta。
- 已新增 unrolled denoise loop export：`build/pi05_fp8_loop_artifact` 入口 `main`，5 个运行时输入，152 个权重/scale 参数，16212 条 VM 指令，输出 `[50, 32]` float32 actions。
- 已新增 `sample_actions` precomputed-prefix ABI：`build/pi05_fp8_sample_precomputed_prefix_artifact` 入口 `main`，输入 `noise_f32 + prefix_k_cache + prefix_v_cache + prefix_valid_rows + rope_interleaved`，输出 `[50, 32]` float32 actions，4090 CUDA Graph 最新复测 `13.286ms/10-step`（strict/oracle artifact，低于 21ms 目标）。
- 已新增 prefix 前半段的可编译切片：`PI05VisionPatchEmbedding.forward_fast()` 通过 DSL 注入 image normalize、patch im2col、BF16 GEMM、bias/position add；`PI05VisionEncoderLayer` / `PI05VisionEncoder` 已覆盖 SigLIP vision tower block、final norm 和 multimodal projector，并可导出 `build/pi05_fp8_vision_encoder_executable`。`PI05LanguageEmbedding.forward_fast()` 已通过 DSL 注入 PaliGemma language embedding gather；`PI05PaliGemmaPrefixEncoder` 已覆盖 compact prefix transformer 和 prefix KV cache materialization，可导出 `build/pi05_fp8_paligemma_prefix_encoder_artifact` 与 `build/pi05_fp8_paligemma_prefix_kv_encoder_artifact`。prefix KV materialization fast path 当前使用 `pi05_qkv_split_rope_cache_bf16`，decoder attention fast path 使用 `pi05_qkv_split_rope_concat_bf16`，分别把 prefix cache 写回和 suffix full-KV concat 融成 DSL-injected CUDA kernel。
- 已新增单 artifact 桥接形态：`build/pi05_fp8_sample_precomputed_prefix_embs_artifact` 入口 `main`，输入 `noise_f32 + prefix_embs + prefix_valid_rows + prefix_rope_interleaved + suffix_rope_interleaved`，在一个 VM graph 内完成 prefix transformer KV materialization 和 10-step denoise。4090 CUDA Graph 当前 `25.812ms`，说明 prefix transformer FP8 GEMM 仍是完整路径的主要优化缺口。
- 已新增 full-token artifact：`build/pi05_fp8_sample_tokens_3v895_artifact` 入口 `main`，输入 `noise_f32 + images_u8 + token_ids + prefix_valid_rows + prefix_rope_interleaved + suffix_rope_interleaved`，在一个 VM graph 内完成 vision encoder、language embedding、prefix concat、prefix KV materialization 和 10-step denoise；CUTLASS-enabled 4090 CUDA Graph 当前复测 3-view/P=895 `28.548ms`，2-view/P=562 `23.425ms`，example0 `final_abs_max≈0.14-0.16`、`final_abs_mean≈0.026-0.028`。
- denoise fast path 已接 FlashRT vendored FA2 packed func，fallback BF16 attention kernel 仍保留在 catalog 用于 correctness/debug。
- 已新增 torch denoise oracle dump 和 C++ VM smoke：strict FP8 accumulation 下 bf16 example0 step0 当前 `abs_max=0.00868897`、`abs_mean=0.00144342`；example0 10-step closed-loop 当前 `final_abs_max=0.078042`、`final_abs_mean=0.0104354`；10-example bf16 multi-oracle 当前 `worst_abs_max=0.149316`、`worst_abs_mean=0.0131491`；runtime-vs-fp16 torch outputs 当前 `worst_abs_max=0.165487`、`worst_abs_mean=0.0158259`。
- RTX 4090 上 denoise/full-token 路径已完成 CUDA Graph capture/replay：precomputed-prefix 后半段 `13.286ms`（strict/oracle artifact），prefix-embeddings single artifact `25.812ms`，CUTLASS-enabled 3-view full-token artifact `28.548ms`，2-view/P=562 full-token artifact `23.425ms`。
- 已对 `/root/autodl-tmp/realtime-vla` 的 Pi0.5 Triton 实现做结构对照。realtime-vla 不做 FP8 量化，4090 Pi05 1/2/3-view 参考为 `22.1ms / 29.2ms / 38.9ms`；它采用固定形状 Triton GEMM、GEMM tile 内 bias/residual/GELU/gate 融合、QKV+RoPE 写回融合、预计算 style/action update 和 CUDA Graph。当前 devproc2 已采用 CUDA Graph、precomputed style、action_out folding、prefix RMS folding、FA2、shape-level GEMM autotune、静态 activation scale、FP8 FAST_ACCUM，并吸收了 vision QKV bias+split、vision bias+GELU+static FP8 quant、vision FFN down CUTLASS shape-specialized route 等优化；已尝试但不保留 full KV cache、独立 scalar fusion、vision residual cuBLASLt `beta=1` accumulate、cuBLASLt bias epilogue，因为这些路径在当前 VM/storage/FP8 layout 下变慢、不稳定或无可用 heuristic。
- 已复现审计 FlashRT PR #19：公开 PR head 缺失 `pi05_sm89_fp8_ffn.py`，且 pipeline 调用未绑定的 `cutlass_ada_fp8_gemm_bf16_simple`，不能按 PR 自带说明跑出 `21ms`。本地 FlashRT main 的 4090 2-view sanity check median 为 `23.41ms`，与 devproc2 当前 `23.425ms` 同级，因此当前性能快照接受 `~23.4ms` 作为 2-view 基线。

主要缺口：

- 直接消费 `images_u8 + token_ids` 的 full-token executable 已生成；尚未把 prompt/state tokenizer 前处理并入 VM graph，也尚未收口最终部署级 C++ sample_actions API。
- vision/text 拼接、vision encoder、language embedding、PaliGemma prefix transformer、prefix KV cache materialization 和 denoise loop 已接成单 VM graph；当前主要缺口是 tokenizer/state preprocessing 的整图 ABI、多样本 actions 精度报告和 prefix GEMM 性能。
- attention 性能路径已经接 FlashRT vendored FA2；CUDA Graph replay 已抽成 runtime RAII API，后续需要继续减少 dynamic quant 小 kernel 数量并接入完整 sample_actions。
- FP8 权重量化和静态 activation scale 已完成并进入主性能 artifact；完整 graph 的多样本 actions 级精度报告仍需补齐。
- denoise 子图已能运行并通过 step0/10-step torch oracle 阈值；仍不是完整 `sample_actions` 的最终精度验收。
- 非量化部分当前以 BF16 为主，主要用于 torch oracle 对齐和现有辅助 kernel 过渡。RTX 4090 上针对 Pi0.5 关键 GEMM shape 的 FP16/BF16 microbench 显示二者总体同级：3-view prefix FFN gate/up `0.768ms` vs `0.766ms`，down `0.383ms` vs `0.380ms`，vision QKV `0.049ms` vs `0.051ms`；2-view prefix FFN down FP16 有局部优势 `0.255ms` vs `0.292ms`。因此 FP16 variant 是后续兼容和局部 profile TODO，但不是当前性能闭环的主杠杆。
- precomputed-prefix denoise 后半段已与 `/root/tw/openpi/outputs/pi05_torch_infer/fp16/outputs.npz` 做 runtime-vs-fp16 actions 对比；完整 vision/text `sample_actions` 尚未做整图 actions 对齐。
- full-token sample path 已用 nsys profile 验证 CUDA Graph 路径；当前接受 2-view/P=562 `23.425ms` 和 3-view/P=895 `28.548ms` 作为性能快照。FlashRT PR #19 的 `~21ms` 对照无法从公开 head 复现，且报告环境为 RTX 4060 Ti，不再作为硬验收口径。当前 profile 显示主瓶颈为 vision/prefix encoder FP8 GEMM；下一阶段重点是保持该性能并按 devproc2 设计重构 fast path，而不是继续追逐独立 elementwise launch fusion。

## openpi0.5 推理路径

PyTorch 侧 `sample_actions` 的核心流程：

1. 预处理 observation：图片 resize/tokenize/state/action padding。
2. `embed_prefix`：三路图片经 SigLIP/PaliGemma vision tower，语言 tokens 经 embedding，拼接 prefix embeddings 和 masks。
3. prefix 进入 PaliGemma language model，生成 prefix KV cache。
4. 初始化 `x_t = noise`，`time = 1.0`，按 `dt = -1 / num_steps` 做 Euler denoise loop。
5. 每一步调用 `denoise_step`：
   - `embed_suffix(state, x_t, timestep)` 生成 action tokens 和 adaRMS 条件。
   - Gemma expert 使用 prefix KV cache、suffix masks、position ids 做 transformer forward。
   - 取最后 `action_horizon` 个输出，经 `action_out_proj` 得到 `v_t`。
6. 更新 `x_t = x_t + dt * v_t`，最终返回 actions。

首个端到端目标固定：

- precision：fp16
- batch size：1
- image size：224
- `num_steps=10`
- 输出：`actions`，shape 与 PyTorch dump 一致
- 精度：`np.allclose(actual, expected, rtol=1e-2, atol=1e-2)`

## 文档结构

- [01_frontend_nn_module.md](01_frontend_nn_module.md)：类 torch `nn.Module` 前端、参数/子模块注册、模型结构表达。
- [02_ir_attr_ops.md](02_ir_attr_ops.md)：IR attr 系统、基础 op 和 openpi0.5 所需 op 语义。
- [03_kernels_lowering.md](03_kernels_lowering.md)：Triton kernel、kernel selection、DPS lowering、cubin artifact。
- [04_weight_artifact_quant.md](04_weight_artifact_quant.md)：编译前 `convert_weight`、devproc2 权重包、artifact 扩展、FP8 权重量化状态。
- [05_runtime_accuracy_milestones.md](05_runtime_accuracy_milestones.md)：C++ runtime、精度对齐流程、分阶段 milestone。
- [06_performance_snapshot.md](06_performance_snapshot.md)：当前性能快照、已采用优化、未采用结论和下一阶段设计债务。
- [07_frontend_dsl_refactor.md](07_frontend_dsl_refactor.md)：`forward_fast()` 一等接口和 CUDA 自定义算子无注册接入重构。
- [08_build_run_profile.md](08_build_run_profile.md)：面向新手的 Pi0.5 编译、运行和 Nsight profile 操作手册。
- [09_pi05_refactor_plan.md](09_pi05_refactor_plan.md)：当前 Pi0.5 实现锐评、边界问题和保持性能不回退的重构方案。
- [11_framework_boundary_refactor.md](11_framework_boundary_refactor.md)：Pi0.5 业务模型不应进入 devproc2 框架层的锐评、目标结构和迁移方案。
- [12_compile_flow_refactor.md](12_compile_flow_refactor.md)：按 runtime build、weight convert、model build、runtime inference 重塑 Pi0.5 产品化编译流程。
- [13_product_build_quickstart.md](13_product_build_quickstart.md)：产品化主线：runtime build、weight convert、`devproc2 build`、runtime inference。
- [14_openpi05_devproc_compile_run_flow.md](14_openpi05_devproc_compile_run_flow.md)：技术分享稿：OpenPI0.5 在 devproc2 上从 checkpoint 到 C++ runtime actions 的完整编译运行流程。

## Milestones

### M0：现状审计与最小子图

- 固化本目录设计文档。
- 明确 PyTorch oracle、输入 dump、输出 dump、误差阈值。
- 将 openpi0.5 推理拆成 prefix、denoise step、Euler loop 三段。

### M1：`nn.Module` 前端

- 新增 `devproc2.nn.Module`、`Parameter`、基础 nn modules；不单独引入 Buffer，推理所需持久 tensor 统一作为 Parameter/Weight 管理。
- 支持子模块路径、参数路径、`state_dict` 命名和模型结构构建。
- 用 devproc2 nn 前端重写 openpi0.5 的模型结构骨架。

### M2：IR attr 与 op 集

- 为 `CallOp` / `CallDPSOp` 增加 attrs。
- 定义 AttrValue 数据模型、printer、serializer、verifier。
- 覆盖 `matmul`、`embedding`、norm、GELU/SiLU、reshape、transpose、cat、slice 等标准基础 op；`Linear` 只作为前端 nn module，forward 必须展开为 `matmul` 加可选 bias `add`，不能作为 IR op。

### M3：convert_weight 与权重包

- 设计并实现 `WeightSpec` / `WeightMap`。
- 支持编译前 `convert_weight`，将 safetensors checkpoint 转成 devproc2 weight package。
- compile 阶段只消费 devproc2 weight package，不直接读取 HuggingFace/PyTorch 权重。
- artifact 增加 `weights/` 和 `metadata/weight_map.json`，部署产物必须自包含。

### M4：kernel 与 lowering

- 实现基础 op Triton kernels。
- 扩展 kernel selection 支持 attrs/layout/shape predicate。
- 编译产物中包含 cubin、kernel table、launch ABI。

### M5：openpi0.5 子图编译

- 先编译 `embed_suffix + denoise_step`。
- 再支持 prefix KV cache 和 attention mask。
- 最后编译完整 `sample_actions`，固定 `num_steps=10`。

### M6：纯 C++ runtime 执行

- C++ runtime 加载 executable、weights、kernels。
- 提供模型级 C++ 推理入口。
- Python 只作为测试 harness 调用 C++。

### M7：精度对齐与性能基线

- 建立单 op、单 layer、denoise step、完整 actions 的逐级对齐。
- 记录误差指标、失败定位方式、基础吞吐/延迟指标。
