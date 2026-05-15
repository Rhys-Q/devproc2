# Runtime、精度对齐与实施阶段

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

建议新增概念：

```cpp
class ModelSession {
public:
    explicit ModelSession(std::shared_ptr<Executable> exec, WeightStore weights);
    Tensor Invoke(std::string_view entry, std::vector<Tensor> inputs);
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
- denoise step 编译

验收：

- 单次 denoise step 输出 `v_t` 与 PyTorch 对齐。

### M6：完整 sample_actions

产出：

- 固定 `num_steps=10` 的 Euler loop。
- prefix KV cache。
- 完整 artifact 编译。

验收：

- `actions` 与 PyTorch fp16 dump 在 `rtol=1e-2, atol=1e-2` 内。

### M7：稳定化

产出：

- debug dump 工具。
- 性能基线。
- 错误信息整理。
- 文档更新为实际实现状态。

验收：

- CI 或本地脚本可以一键运行 openpi0.5 fp16 对齐测试。

## 风险与默认策略

- Vision tower kernel 复杂度高：先把 prefix path 拆出来逐层对齐，必要时短期用 precomputed prefix embeddings/KV cache 解锁 denoise path。
- attention 精度敏感：首版使用 correctness-first 标准 op 序列，确认精度后再做 pattern fusion。
- dynamic shape 增加复杂度：首版固定 batch/image/action horizon/num_steps，metadata 记录约束。
- 量化不进入首版 runtime：只保留 metadata 和 kernel selection 扩展点。
