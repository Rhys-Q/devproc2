# openpi0.5 支持方案总览

## 背景

目标是在 devproc2 中支持 openpi0.5 模型的编译和纯 C++ 推理运行。PyTorch 参考实现位于 `/root/autodl-tmp/openpi`，参考脚本为：

```bash
cd /root/autodl-tmp/openpi
uv run python scripts/check_pi05_fp16_infer.py
```

该脚本加载 `/root/autodl-tmp/tools/pi05-pytorch-base/model.safetensors`，使用 `Pi0Config(pi05=True, dtype="float16")`，运行 `PI0Pytorch.sample_actions(..., num_steps=10)`，并与已 dump 的 fp16 输出做 `rtol=1e-2, atol=1e-2` 对齐。

本目录只描述设计方案和实施阶段，不做代码实现。

## 当前 devproc2 状态

已有能力：

- Python DSL 可通过 `@dp.function` 捕获函数 AST，生成 `IRModule`、`Function`、`CallOp`、`CallDPSOp`、`TensorCreateOp` 等 IR。
- 已有 `TensorStructInfo`，可表达 shape、dtype、device。
- 已有 DPS lowering：`CallOp` 可被 lower 为 `TensorCreateOp + CallDPSOp(kernel)`。
- 已有 `KernelRegistry`、`KernelSpec`、`@dp.kernel` 和 Triton AOT 编译到 cubin 的雏形。
- 已有 VM bytecode、artifact `executable.vm`、`abi.json`，以及 C++ `Executable::Load` / `VMState::Invoke`。
- C++ runtime 已支持 builtin、packed_func、kernel 三类外部调用。

主要缺口：

- DSL 只能描述函数，不能自然描述 torch `nn.Module` 风格模型结构。
- IR op 没有 attr 系统，无法表达 `axis`、`eps`、`approximate`、`transpose_b`、`num_heads`、`layout` 等语义。
- 当前 `InferStructInfoPass` 只对 `TensorCreateOp` 和简单 elementwise 形态做 MVP 推导；后续每个 tensor op 都必须有独立 infer shape/struct info 函数，不能依赖“从第一个参数传播”的默认行为。
- `KernelRegistry` 当前主要按 op/device/input dtype 匹配，缺少 attrs/layout/shape 级别的选择能力。
- artifact 的 `constants/`、`kernels/` 还偏占位；权重加载、权重 metadata、weight tensor view 尚未形成机制。
- C++ runtime 尚未定义模型级输入输出 ABI、weight mmap/read、kernel table 注册和纯 C++ openpi0.5 调用接口。
- 缺少逐 op、逐 block、逐 denoise step 的精度定位工具链。

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
- [04_weight_artifact_quant.md](04_weight_artifact_quant.md)：编译前 `convert_weight`、devproc2 权重包、artifact 扩展、未来量化预留。
- [05_runtime_accuracy_milestones.md](05_runtime_accuracy_milestones.md)：C++ runtime、精度对齐流程、分阶段 milestone。

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
