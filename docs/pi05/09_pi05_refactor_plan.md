# Pi0.5 实现锐评与重构方案

## 结论

`python/devproc2/models/pi05` 当前已经证明了两件事：Pi0.5 可以在 devproc2 里跑通，性能也能站住。但这份实现不应该成为 devproc2 的模型实现范式。它现在更像一份被性能压力推着长出来的战时产物：能打，但脏；精度和 latency 达标，但结构边界被打穿了。

最核心的问题不是 `forward_fast()` 存在，也不是 `forward()` 和 `forward_fast()` 放在同一个 `nn.Module` 里。恰恰相反，这个双路径接口是正确方向：`forward()` 负责标准 op 语义，`forward_fast()` 负责显式接入高性能 HPC 路径。真正的问题是 fast path 把 CUDA kernel ABI、packed func 参数顺序、FP8 scale、KV cache byte layout、VM alias 细节和 artifact 约定全部泄漏进了模型代码。模型层不再像模型，而像半个 compiler backend。

本轮重构目标不是重写所有性能优化，而是在不牺牲现有性能基线的前提下，把 Pi0.5 从“能跑的性能脚本”收敛成“可维护的 devproc2 模型实现”。

## 当前本地资产路径

- ckpt 路径：`/root/autodl-tmp/tools/pi05-pytorch-base`
- OpenPI inputs 路径：`/root/autodl-tmp/openpi/outputs/pi05_torch_infer`
- OpenPI outputs 路径：`/root/autodl-tmp/openpi/outputs/pi05_torch_infer`
- OpenPI tokenizer 路径：`/root/autodl-tmp/openpi/outputs/pi05_torch_infer`

## 锐评

### `modules.py` 是失控的聚合体

`modules.py` 接近 3000 行，里面同时存在：

- 模型结构：vision tower、PaliGemma prefix encoder、action expert、denoise loop。
- 标准 op reference path：`matmul`、`attention`、norm、GELU、cat。
- fast path kernel 编排：`dp.cuda_call(...)`、`dp.call_dps_packed(...)`、launch spec。
- 数据布局知识：QKV packed layout、KV cache layer stride、style table step/layer stride。
- 量化策略：dynamic/static FP8 activation scale、FP8 weight scale、accum path。
- artifact 命名约定：`fp8.decoder_attn_qkv_w_0.weight` 这类名字直接散落在 Module 初始化里。

这是一个典型的“所有东西都知道所有东西”的文件。任何人想改一个 decoder layer，都必须同时理解 frontend DSL、CUDA kernel 签名、VM memory alias、FP8 GEMM ABI 和权重转换规则。这个复杂度不是 Pi0.5 本身要求的，是边界没有守住造成的。

### `forward()` / `forward_fast()` 的契约还不够硬

好的设计应该是：

- `forward()` 只使用标准 op，表达数学语义和 reference graph。
- `forward_fast()` 和 `forward()` 放在同一个 Module 中，表达同一个语义节点的两种实现。
- `forward_fast()` 可以使用后端能力，但应调用语义化 HPC primitive，不能把后端 ABI 暴露成模型主体逻辑。
- compile mode 明确选择入口：normal 永远不偷跑 fast，fast 可在没有 fast path 时 fallback 到 `forward()`。

当前实现已经有 `_validate_pi05_module_contract()` 检查 `forward()` 是否调用 `forward_fast()`，这是正确方向。但还不够。`forward()` 里仍出现 `dp.tensor_view(...)` 这类 storage/view 细节，且大量 reference path 直接复用 FP8 artifact 权重命名和 `_fp8_linear_ref(...)`。这使 normal path 看起来像 reference，实则仍被 deploy artifact 的布局牵着走。

### CUDA 调用无注册化完成了一半

`dp.cuda_call(source::symbol, ...)` 已经摆脱了 Pi0.5 静态 kernel catalog，这是进步。但模型层仍然需要手写：

- kernel symbol；
- launch；
- `--std=c++17`；
- output index / metadata；
- 参数顺序；
- scalar bit packing；
- shape 到 grid 的映射。

这不是模型工程师应该写的层级。Pi0.5 模型应该表达“做 QKV split + RoPE + 写 KV cache”，而不是表达 `pi05_qkv_split_rope_cache_bf16` 的每一个 CUDA 参数怎么排。后者应该进 `pi05_ops` 或 backend facade。

### `dp.tensor_view` 暴露在前端是设计债

`dp.tensor_view(base, byte_offset, shape, byte_stride=..., base_offset=...)` 出现在模型代码里，是最刺眼的抽象泄漏之一。它把 VM/storage 层概念直接塞进前端 DSL：

- `byte_stride=prefix_rows * num_kv_heads * head_dim * 2`
- `base_offset=layer_idx * rows * 3 * hidden_size * 2`
- BF16 element size `2` 在多个地方手写

模型层关心的应该是“取第 `layer_idx` 层 KV cache”或“取第 `step/layer` 个 style slice”，而不是字节偏移。byte math 一旦散落，就会导致后续任何 layout 调整都变成全局搜索替换，并且极难验证。

迁移期可以保留底层 storage alias 表达，但它必须退到 compiler internal；前端模型、高层 IR 和 Pi0.5 layout helper 都不应该继续手写 `dp.tensor_view` 的 byte offset。

### `call_dps_packed` 的使用还不够统一

计划里要求的 destination-passing 形式是对的：

```python
out = dp.empty(...)
dp.call_dps_packed("runtime.cuda.fp8_nt_bf16", inputs=[..., out])
```

当前大多数路径通过 `_packed_call(...)` 已经接近这个约定，但仍存在裸 `dp.call_dps_packed(...)`，尤其是 accum path。这些调用没有统一命名，也没有显式表达“这个 packed func 会原地累加到 residual”。结果是语义只能靠读参数顺序猜：

```python
dp.call_dps_packed(
    "runtime.cuda.fp8_nt_bf16_accum",
    inputs=[attn_fp8, self.o_w_fp8, hidden, rows, ...],
)
```

这里 `hidden` 是输入还是输出？从模型层看不出来。它实际是 in-place destination。这种写法对 compiler、memory planner 和读代码的人都不友好。

### `forward_kv` 是前端能力缺口的症状

`PI05PaliGemmaPrefixEncoder.forward_kv(...)` 不是一个自然的模型 API。它存在的本质原因是当前 `nn.Module` / `GraphBuilder` 对多入口模块支持不够优雅：

- `forward()` 表示 prefix hidden states；
- KV materialization 是另一个合法 graph entry；
- export 需要选择这个 entry；
- 但现在只能通过 `normal="forward_kv"` 这类字符串和特殊方法绕过去。

这不是 Pi0.5 的业务语义，而是 frontend 缺少“命名入口 / 多方法构图 contract”。继续把这种 workaround 放在模型里，会让每个需要多输出、多 entry 的模型都长出自己的私有命名。

### 权重命名和布局规则离模型太近

`Parameter(name=f"fp8.decoder_attn_qkv_w_{layer_idx}.weight")` 这类命名大量散落在 Module init 中。它把 checkpoint conversion、artifact weight map、runtime binding 和模型结构硬绑在一起。

模型结构应该声明“这一层需要 decoder attention qkv weight”。具体权重名、FP8/BF16 variant、layout 和 scale tensor 应该由 weight spec/layout 层集中定义。否则权重转换规则一改，模型代码要陪着改；模型结构一复用，权重命名又成了阻碍。

## 目标形态

重构后的 Pi0.5 目录应该只保留 Pi0.5 业务模型资产。Pi0.5 是 devproc2 内置业务模型，不是 plugin；但它也不应该重新实现 framework export、artifact、oracle、quantization pipeline。关键原则是：不要按 normal/fast 拆成两套模型；同一个 Module 仍然同时拥有 `forward()` 和 `forward_fast()`，只是 `forward_fast()` 通过 Pi0.5 CUDA/HPC op 边界接入高性能后端。

另一个必须钉死的边界是：precision/quantization 可以决定 DSL graph variant，device/architecture 只能决定 backend lowering variant。FP8 量化和 FP16/BF16 非量化路径不必强行共享同一张 DSL 图；FP8 引入 scale、quant/dequant、FP8 artifact layout、fused norm/activation-to-FP8 等可见数值路径，硬塞进一张图只会制造到处都是 `if quantized` 的脏抽象。4090、Thor 这类 target 则不应该拆模型图；它们共享同一个 precision graph，通过 `ops.py` / lowering / kernel registry 选择不同手写 kernel。

- `config.py`：唯一的 Pi0.5 配置入口，定义 `PI05Config` dataclass；模型结构、shape、entrypoint、precision/quantization variant、layout/fusion policy、kernel source 列表、artifact recipe 默认值都从这里来。
- `model.py`：Pi0.5 Module 结构，每个 Module 内保留 `forward()` / `forward_fast()` 双路径；FP8 和非 FP8 可以导出不同 fast graph variant，但共享 Module skeleton 和非量化子图 helper。
- `ops.py`：Pi0.5 专用 CUDA/HPC op 边界，封装手写 kernel、packed func、kernel spec 注册和 target-specific kernel 选择；这是 CUDA 相关文件，应该保留在 Pi0.5 目录。模型代码可以 `from . import ops as pi05_ops` 提高可读性；如果迁移期需要 `pi05_ops.py`，它只能是兼容 re-export，不能形成第二套实现。
- `weights.py`：Pi0.5 checkpoint conversion、deploy fusion mapping、weight package assembly；消费 `PI05Config`、框架层 quantization manifest 和 requant helper，不自研量化算法。
- `cuda/`：Pi0.5 自有 CUDA source、vendored kernel 子集和本模型专用 backend wrapper 的唯一归属地。
- `python/devproc2/export/`：框架层构图、compile、emit executable/ABI、entrypoint selection、CLI scaffolding。
- `python/devproc2/artifact/`：框架层 artifact builder、resource copy、kernel packaging、manifest 写入。
- `python/devproc2/quantization/`：框架层 quantization manifest schema、通用 requant helper 和 fusion/requant 规则；不依赖业务 PyTorch 模型，不引入 ModelOpt runtime 依赖。
- `python/devproc2/integrations/modelopt/` 或 `tools/quant/modelopt_export.py`：可选生产端 exporter，从业务侧已经量化好的 PyTorch model / ModelOpt state 导出标准 manifest；不进入 runtime，也不放进 Pi0.5 模型目录。
- `tools/pi05/dump_torch_oracle.py`：OpenPI/PyTorch oracle producer；devproc2 可以消费 oracle artifact，但 Pi0.5 runtime/model package 不拥有业务 PyTorch producer。

目标不是让文件数量变多，而是让依赖方向变干净：

```text
model.py      -> ops.py -> devproc2 DSL/backend
model.py      -> config.py
ops.py        -> config.py / cuda/
weights.py    -> config.py
weights.py    -> devproc2.quantization
devproc2.export   -> PI05Config / PI05Model
devproc2.artifact -> PI05Config artifact recipe
```

Module 可以知道“这里有一个 fast implementation”，也可以根据 `PI05Config.precision` 或构图时传入的 dtype/quant mode 进入 FP8 或 BF16/FP16 fast helper；但 Module 不应该知道 CUDA 参数顺序、target-specific kernel symbol 或 device-specific launch 策略。`ops.py` 可以知道这些 CUDA 细节；`weights.py` 不应该构图；Pi0.5 目录不应该拥有长期 `export.py`、`artifact.py` 或 `torch_oracle.py`。

## 重构方案

### 1. 固化 Module contract

新增通用或 Pi0.5 专用 validator，作为构图前的硬门禁：

- 每个 `nn.Module` 必须实现 `forward()`。
- `forward()` 禁止引用 `forward_fast`。
- normal compile 只能使用 standard op；构图后不得出现 `CudaCallOp`、`CallDPSOp`、backend packed func。
- Pi0.5 `model.py` 禁止直接调用 `dp.cuda_call`、`dp.call_dps_packed`、`dp.tensor_view`；CUDA/packed 只能出现在 `ops.py` 这个受控边界，tensor 切片必须通过标准 `select`、`slice`、`index`、`split`、`reshape` 表达。
- fast path 允许 fallback 到 `forward()`，但 fallback 由 compile mode 选择逻辑完成，不由子模块手动调用。
- `forward_fast()` 不强制接受 `ctx`；precision/quant mode 可以来自 `PI05Config`、构图参数或显式 dtype 参数。模型层最多按 precision/quant mode 选择 FP8 / 非 FP8 graph variant，不允许按 4090、Thor、SM 版本散落分支。
- 同一个 `nn.Module` 可以实现 4090 和 Thor 的 fast path，但 target 分发必须停在 `ops.py` / lowering / backend registry，不能在 layer 主体里手写不同 kernel symbol。

这一步先以 lint/test 约束落地，不急着一次性拆完所有文件。先把“以后不许继续变脏”钉住。

### 2. 建立 Pi0.5 fast primitive facade

把裸 backend 调用集中收口成 Pi0.5 私有 CUDA/HPC primitive。模型层调用：

```python
qkv = pi05_ops.fp8_linear(x, weight, x_scale, w_scale, out_shape)
q, k, v = pi05_ops.qkv_split_rope(qkv, rope, layout)
hidden = pi05_ops.fp8_accum_linear_(attn, weight, hidden, scales)
```

而不是直接调用：

```python
dp.cuda_call("...cu::pi05_qkv_split_rope_concat_bf16", ...)
dp.call_dps_packed("runtime.cuda.fp8_nt_bf16_accum", inputs=[...])
```

首批 facade 至少覆盖：

- `bf16_linear(...)`
- `fp8_linear(...)`
- `fp8_linear_accum_(...)`
- `quantize_fp8_static(...)`
- `quantize_fp8_dynamic(...)`
- `layer_norm_to_fp8(...)`
- `rms_norm_to_fp8(...)`
- `qkv_split_rope(...)`
- `qkv_split_rope_cache(...)`
- `qkv_split_rope_concat(...)`
- `bias_add_ / bias_residual_ / gate_residual_`
- `geglu_to_fp8(...)`
- `attention_fa2(...)`

`ops.py` 初期可以沿用现有 `dp.cuda_call` 和 `dp.call_dps_packed` 形态，先保证性能路径不变；但 `attention_fa2(...)` 的最终 lowering 必须指向 Pi0.5-owned CUDA kernel 或 Pi0.5-owned runtime backend，不能继续指向 FlashRT 产物。变化不是把 ABI 换个地方藏起来，而是先从 `model.py` 赶出去，再把 Pi0.5 专用 backend ownership 收回来。

### 3. 固化 precision graph variant 和 target lowering 边界

FP8 量化路径和非 FP8 路径不强求共享同一张 DSL graph。FP8 不只是 dtype 变化，它改变了权重 artifact、activation scale、quant/dequant 点、fused norm/activation kernel 和误差模型；这些差异进入 graph variant 是合理的。强行把 FP8 和 BF16/FP16 合成一张 graph，只会让前端图充满量化策略分支，表面复用，实际更难验证。

4090、Thor 等设备/架构差异则相反：它们不应该产生两套 Pi0.5 模型图。对同一个 precision graph，target 只影响 lowering 选择：

```text
fp8 graph
  target=rtx4090_sm89 -> sm89 handwritten kernel / cuBLASLt / Pi0.5 FA2
  target=thor         -> thor handwritten kernel / target-specific backend
  target=generic      -> decomposed fallback

bf16/fp16 graph
  target=rtx4090_sm89 -> bf16/fp16 GEMM + handwritten composite kernels
  target=thor         -> thor backend
  target=generic      -> standard op fallback
```

推荐接口形态：

```python
@dataclass(frozen=True)
class PI05Config:
    precision: str = "fp8"
    target: str = "rtx4090_sm89"
    fp8_layout: str = "nk"

class PI05DecoderLayer(nn.Module):
    def __init__(self, config: PI05Config, layer_idx: int):
        self.config = config

    def forward_fast(self, hidden):
        if self.config.precision == "fp8":
            return self._forward_fast_fp8(hidden)
        return self._forward_fast_bf16(hidden)
```

这里允许按 `precision` 选择 graph variant，因为量化和非量化确实是不同数值路径；但不允许在 layer 主体里写：

```python
if self.config.target == "rtx4090_sm89":
    dp.cuda_call("pi05_xxx_sm89", ...)
elif self.config.target == "thor":
    dp.cuda_call("pi05_xxx_thor", ...)
```

target-specific 分发必须写在 `ops.py` / lowering registry：

```python
qkv = pi05_ops.fp8_linear(hidden, self.qkv, config=self.config)
q, k, v = pi05_ops.qkv_split_rope_cache(qkv, rope, cache, config=self.config)
attn = pi05_ops.attention(q, k, v, cache=cache, config=self.config)
```

对应 lowering 规则：

```text
pi05_ops.qkv_split_rope_cache
  precision=fp8, target=rtx4090_sm89 -> pi05_qkv_split_rope_cache_bf16_sm89
  precision=fp8, target=thor         -> pi05_qkv_split_rope_cache_bf16_thor
  precision=bf16, target=*           -> split + rope + cache store or bf16 composite kernel

pi05_ops.ada_rms_norm_style_to_fp8
  target=rtx4090_sm89 -> pi05_ada_rms_norm_style_to_fp8_bf16_sm89
  target=thor         -> pi05_ada_rms_norm_style_to_fp8_bf16_thor
  fallback            -> rms_norm + style affine + quantize_fp8
```

手写 kernel 是一等 backend implementation，但不是前端表达。每个手写 kernel 都应该有 registry metadata，而不是裸字符串散落在模型层：

```python
KernelSpec(
    semantic="pi05.qkv_split_rope_cache",
    symbol="pi05_qkv_split_rope_cache_bf16_sm89",
    precision=["fp8"],
    targets=["rtx4090_sm89"],
    input_layouts=[...],
    output_layouts=[...],
    effects=["write_kv_cache"],
    fallback="decompose",
)
```

这条边界的验收口径很简单：FP8 和非 FP8 可以导出两张 fast graph；4090 和 Thor 对同一个 precision graph 必须只生成不同 lowered backend graph / executable，不复制模型结构、不复制 DSL 构图代码。

### 4. 替换前端裸 `dp.tensor_view`

彻底策略：前端和高层 IR 不再暴露 byte-level view API。模型层只能写标准张量语义 op：`select`、`slice`、`index`、`split`、`reshape`。`reshape` 就叫 `reshape`，不引入 `reshape_view` 这种带实现细节的名字；是否 no-copy 是 verifier/lowering 的责任，不是 op 名字的一部分。

Pi0.5 模型代码应从：

```python
prefix_k = dp.tensor_view(prefix_k_cache, layer_idx, ...)
style_attn = dp.tensor_view(style_attn_table, step, ...)
q = dp.tensor_view(qkv, 0, ...)
```

改成：

```python
prefix_k = dp.select(prefix_k_cache, axis=0, index=layer_idx)

style_attn = dp.select(style_attn_table, axis=0, index=step)
style_attn = dp.select(style_attn, axis=0, index=layer_idx)

q_raw, k_raw, v_raw = dp.split(
    qkv,
    sections=(q_dim, kv_dim, kv_dim),
    axis=1,
)
q = dp.reshape(q_raw, (rows, q_heads, head_dim))
k = dp.reshape(k_raw, (rows, kv_heads, head_dim))
v = dp.reshape(v_raw, (rows, kv_heads, head_dim))
```

高层 IR 对应新增或规范化这些 op：

- `select(base, axis, index)`：取单个 axis index，结果降一维。
- `slice(base, starts, sizes, strides=None)`：多维区间切片。
- `index(base, selectors)`：通用索引 sugar，支持 `int/scalar`、`slice`、`all` 三类 selector，规范化成 `select` + `slice`。
- `split(base, sections, axis)`：按 axis 切分，返回 tuple。
- `reshape(base, shape)`：标准 reshape；当用于 fast path alias 时由 verifier 确认元素数一致、layout 可 no-copy。

这些 op 的 `struct_info` 直接推导 shape/dtype/device，alias analysis 标记为 `view_of(base)`。它们不计算 byte offset，也不接受 byte stride。

`select` / `slice` / `index` 的语义边界要克制。不要一开始实现 NumPy/PyTorch 那种大而全的 advanced indexing；Pi0.5 当前只需要：

- `int` 或 runtime scalar：走 `select`，结果降维。
- `slice(start, stop, step)`：走 `slice`，结果保留维度。
- `all` / `:`：保留整维。

例如：

```python
prefix_k = dp.index(prefix_k_cache, [layer_idx, slice(None), slice(None), slice(None)])
style = dp.index(style_table, [step, layer_idx, slice(None), slice(None)])
```

只是语法糖，前端 normalize 后等价于：

```python
prefix_k = dp.select(prefix_k_cache, axis=0, index=layer_idx)

style = dp.select(style_table, axis=0, index=step)
style = dp.select(style, axis=0, index=layer_idx)
```

byte offset 只允许在 lowering 之后出现：

```text
select/slice/index/split/reshape
  -> storage alias lowering
  -> low-level storage alias / VM tensor descriptor
```

这个 lowering 统一使用 dtype element size、physical layout、compact stride 和 slice/index/reshape 语义计算地址。`dp.tensor_view` 最终应从公开 DSL API 中移除；若底层仍需要类似能力，也只能作为 compiler-internal low-level op 存在。

lowering 规则保持机械、可证明：

- `select(axis=i, index=v)`：`new_offset = base_offset + v * old_stride[i] * elem_size`，删除第 `i` 维 shape/stride。
- `slice(starts, sizes, strides)`：`new_offset = base_offset + sum(starts[d] * old_stride[d] * elem_size)`，shape 变成 `sizes`，stride 乘以 slice stride。
- `split(sections, axis)`：lower 成多个 `slice`，每个输出是一个 alias descriptor。
- `reshape(shape)`：只在 layout compatible 时变成 alias descriptor；normal path 可以 materialize，fast path verifier 应直接报错，避免隐式 copy。

### 5. 统一 destination-passing packed func

保留并推广当前 `_packed_call` 思路，但不要让它躲在 `modules.py` 私有 helper 里。建议收口为：

```python
out = pi05_ops.call_packed_out(
    "runtime.cuda.fp8_nt_bf16",
    args=[...],
    shape=(rows, out_features),
    dtype="bfloat16",
)
```

in-place accum 必须显式命名：

```python
pi05_ops.fp8_linear_accum_(
    x_fp8,
    weight,
    residual=hidden,
    shape=(rows, hidden_size),
    scales=(x_scale, w_scale),
)
```

禁止模型层写裸 `dp.call_dps_packed(...)`。这样 memory effect 和 readable intent 对齐，后续 verifier 也能检查所有 packed func 都有明确 destination。

### 6. 收回 Pi0.5 CUDA kernel ownership

Pi0.5 的部署路径不能依赖 FlashRT FA2。这里的“不能依赖”包括：

- 不能要求运行环境存在 `/root/autodl-tmp/FlashRT`、`/root/tw/FlashRT`、`build-fa2-sm89` 这类外部 checkout 或 build 目录。
- 不能通过 `DEVPROC2_FLASHRT_FA2_SO` 指向 FlashRT 产物作为正式路径。
- 不能在 runtime 中硬编码 FlashRT pybind `.so` 或 `libflashrt_fa2_raw.so` 的绝对路径。
- 不能把 FlashRT 的 Python extension 当作 Pi0.5 artifact 的隐式依赖。

如果当前性能路径确实使用了 FlashRT FA2 里的 kernel，可以把必要 kernel 拆出来放到 Pi0.5 自己的 CUDA 目录下：

```text
python/devproc2/models/pi05/cuda/
  pi05_kernels.cu
  fa2/
    fa2_wrapper.cu
    fa2_wrapper.h
    flash_attn_2_src/...   # 最小必要子集，带 provenance/license 记录
  gemm/
    ...
```

拆出来之后，kernel source、build rule、symbol ABI 和 artifact packaging 都归 devproc2/Pi0.5 所有。FlashRT 可以作为一次性参考来源，不能作为运行时依赖来源。

更优雅的落点有两种：

- 优先方案：FA2 wrapper 作为 Pi0.5 CUDA source provider 的一部分，在 export 阶段 AOT 编译为 cubin，并进入 artifact `kernels/` 与 `metadata/kernel_table.json`。
- 过渡方案：如果 FA2 模板体量太大、暂时不适合每个 symbol 都进 kernel table，则在 devproc2 build 中生成 Pi0.5-owned shared library，并从 devproc2 package 或 artifact 相对路径加载；仍然禁止外部 FlashRT 路径。

同时，Pi0.5 专用 packed func 不应继续混在通用 `runtime/src/cuda/cuda_gemm.cc` 里。`runtime.cuda.fp8_*` / `runtime.cuda.bf16_*` 这类 cuBLASLt GEMM 可以保留为通用 runtime 能力；但 `runtime.cuda.pi05_fa2_bf16` / `runtime.cuda.pi05_fa2_bf16_batched` 属于 Pi0.5 attention backend，应迁移到 Pi0.5 backend 注册边界，或者由 `pi05_ops.attention_fa2(...)` 直接 lowering 到 Pi0.5-owned kernel。

这一步的设计底线是：Pi0.5 fast path 可以复用 FA2 算法和 kernel 实现思想，但不能把 FlashRT repo、FlashRT build 产物或 FlashRT Python module 作为系统依赖。

### 7. 用命名入口替代 `forward_kv`

新增 GraphBuilder 或 export 层接口：

```python
GraphBuilder().build_method(module, "materialize_kv", input_specs)
```

然后将 `PI05PaliGemmaPrefixEncoder` 改成：

- `forward(prefix_embs, rope_interleaved) -> hidden`
- `materialize_kv(prefix_embs, prefix_valid_rows, rope_interleaved) -> (k_cache, v_cache)`
- `materialize_kv_fast(...) -> (k_cache, v_cache)`，或由 compile mode 自动选择 fast method。

export API 不再传 `normal="forward_kv"`，而是声明 entrypoint：

```python
build_pi05_paligemma_prefix_kv_encoder_module(entrypoint="materialize_kv")
```

这会把“多入口模型”提升为 frontend 能力，而不是 Pi0.5 的私有绕法。

### 8. 用 QuantizationManifest 切开 ModelOpt 和业务 PyTorch 边界

devproc2 不应该继续自研量化算法、校准流程和 activation scale 生成逻辑；但 devproc2 也不应该拥有业务 PyTorch 模型。ModelOpt 只吃 PyTorch，这意味着真正的 PTQ/QAT/calibration 必须发生在业务生产端：那里才知道如何加载 OpenPI/PyTorch model、processor、tokenizer、dataloader、calibration sample 和业务 config。把这些依赖塞进 devproc2，会让 runtime/inference 框架越界成业务训练栈。

正确边界是：devproc2 core 只定义和消费稳定的部署量化协议；ModelOpt adapter 最多作为 optional producer-side exporter 存在。Pi0.5 只声明模型特定的 deploy mapping / fusion spec。

```text
业务 PyTorch/OpenPI repo
  -> ModelOpt PTQ/QAT/calibration
  -> 业务侧已经量化好的 torch module / ModelOpt state
  -> optional devproc2.integrations.modelopt exporter
  -> QuantizationManifest artifact
  -> devproc2 core consumes manifest + FP/BF16 weights
  -> Pi0.5 deploy fusion spec + framework requant helper
  -> Pi0.5 weight package
  -> pi05_ops lowering / runtime ABI
```

ModelOpt 只作为业务生产端的离线量化前端使用，不进入 devproc2 runtime，不成为 artifact 运行时依赖。正式部署 artifact 仍然只依赖 devproc2 自己的权重包、kernel artifact、CUDA/cuBLASLt 等系统库。

devproc2 core 应提供：

- `QuantizationManifest` / `QuantTensorSpec` / `CalibrationManifest`：与模型无关的中间格式，描述原始 module path、weight/input/output quantizer、amax、scale、axis、block size、dtype。
- 通用 FP8 requant helper：从 FP/BF16 tensor 或 dequantized tensor 按指定 common scale 重新量化；禁止直接拼接不同 scale 的 FP8 tensor。
- 通用 fusion manifest 表达：支持“多个 component tensor -> 一个 deploy tensor”的映射和 provenance 记录，但不内置 Pi0.5 的 qkv/gate-up 名字。

可选 ModelOpt exporter 的边界必须更窄：

- 可以读取一个已经由业务代码完成 ModelOpt 量化/校准的 torch module 或 ModelOpt state。
- 可以抽取 quantizer amax/scale、quant config、algorithm、ModelOpt version 和 calibration metadata，写成 `QuantizationManifest`。
- 不负责加载 OpenPI，不构造业务 dataloader，不执行 calibration loop，不 import 业务 repo。
- 不作为 devproc2 core dependency；建议放在 `python/devproc2/integrations/modelopt/` 或 `tools/quant/modelopt_export.py`，并通过 optional extra 安装 `torch + nvidia-modelopt`。

Pi0.5 层只提供模型特定信息：

- 原始 Torch module path 到 devproc2 logical weight name 的映射。
- QKV、gate-up、action folding、Q/K interleave、`nk/kn` layout 等 deploy transform。
- 哪些 component 共享一个 fused activation scale。
- artifact 中 `fp8.*.weight`、`fp8.*.scale`、`act_scale.*` 的命名 spec。

本阶段采用“不改原始 Torch 模型，也不让 devproc2 管 Torch 模型”的方案：允许业务侧 ModelOpt 按 OpenPI 原始 module 粒度量化和校准，例如 `q_proj/k_proj/v_proj`、`gate_proj/up_proj` 各自拥有 weight/input quantizer 统计；producer-side exporter 只把这些统计标准化成 `QuantizationManifest`，再由 Pi0.5 deploy spec 把它们投影到当前 fused ABI。

核心规则是：不要直接拼接 ModelOpt 已经量化好的 FP8 tensor。因为每段 FP8 code 依赖自己的 scale，直接拼接后再传一个 common `B_scale` 是错误语义。framework requant helper 必须从原始 FP/BF16 权重或 dequantized 权重重新量化到 fused weight 的 common scale。

对于 QKV / gate-up 这类 fused projection，Pi0.5 deploy spec 驱动 fusion pass：

```text
原始权重 + ModelOpt component amax
  -> 按 devproc2 layout 规则做 transpose/interleave/fold/concat
  -> common_amax = max(component_amax...)
  -> common_scale = max(common_amax / 448.0, 1e-12)
  -> fused_fp8 = quantize_e4m3(fused_fp_weight, common_scale)
  -> 写入 fp8.*.weight + fp8.*.scale
```

activation scale 也按 fused op 语义生成。若 `q_proj/k_proj/v_proj` 在 Torch 图中消费同一个 input tensor，deploy fused QKV 只产生一个 `act_scale.decoder_attn_qkv_w_i`；其值来自对应 component input amax 的 conservative merge：

```text
common_input_amax = max(input_amax_q, input_amax_k, input_amax_v)
common_input_scale = max(common_input_amax / 448.0, 1e-12)
```

`gate_proj/up_proj` 同理。这个策略会牺牲一点 per-branch scale 的精细度，但换来三个关键收益：

- 不改 OpenPI Torch 模型，不要求构造 deploy-aligned Torch mirror model，也不要求 devproc2 知道如何加载业务 PyTorch 模型。
- 不改当前 `runtime.cuda.fp8_nt_bf16` scalar `A_scale/B_scale` ABI。
- 把量化算法、校准数据和 scale 选择交给业务侧 ModelOpt 流程，把 manifest/requant 能力放到 devproc2 core，把 ModelOpt 读取逻辑限制在 optional producer-side exporter，把 deploy layout、fused weight ABI 和 artifact 映射留在 Pi0.5 spec。

长期可以保留一个更精细的 backend 路线：新增 `pi05_ops.fp8_linear_grouped_scale(...)`，让 fused QKV / gate-up 在同一个 kernel 中按 output slice 使用不同 weight scale。但这需要 CUTLASS/CuTe 或自有 epilogue 支持，不属于第一阶段。第一阶段的硬约束是行为等价于当前 scalar-scale GEMM ABI，先把手写量化和 out-of-band `act_scale.*` 收回到可复现的 manifest-driven 权重生成流程。

框架层 manifest + Pi0.5 deploy manifest 至少记录：

- ModelOpt package version、quant config、algorithm。
- calibration dataset / sample set / hash。
- 原始 Torch module path 到 devproc2 logical weight name 的映射。
- component amax/scale 和 fused common scale。
- fused scale policy：`max_component_amax`。
- 是否从 original FP/BF16 weight 重新量化，而不是拼接已量化 FP8。
- devproc2 target layout：`nk` / `kn`。

### 9. `PI05Config` / `weights.py` spec 集中化

不新增长期 `layout.py` 或 `weight_spec.py`。配置类和权重映射文件共同定义：

- 每类权重的 logical name。
- BF16 / FP8 artifact name。
- scale name。
- shape。
- layout：`kn` / `nk` / row-major。
- 是否 per-layer。
- 是否 constant tensor。

Module init 不再拼接字符串，只从 `PI05Config` / `weights.py` 请求 spec：

```python
self.qkv = PI05WeightSpec.decoder_attn_qkv(layer_idx).parameter(device)
```

这样 `weights.py` 和 `model.py` 使用同一份 spec。checkpoint conversion 不再通过字符串约定和 Module 隐式对齐。

### 10. 收敛 Pi0.5 业务模型目录

在 contract、`PI05Config` 和 `ops.py` 建立后，再收敛文件。目标不是把 `modules.py` 拆成更多模型文件，而是把 Pi0.5 目录压回业务模型应该拥有的最小职责。

建议顺序：

1. 新增 `config.py`，把 shape、layer 数、head 数、entrypoint、precision、layout/fusion、kernel source、artifact recipe 默认值统一收到 `PI05Config` dataclass。
2. 新增 `ops.py`，迁移 `_packed_call`、CUDA helper、FP8 quant/norm/attention wrapper、Pi0.5 kernel spec 注册；若保留 `pi05_ops.py`，只能 re-export `ops.py`。
3. 将 `modules.py` 收敛成 `model.py`；短期可以保留 `modules.py` 兼容 re-export，长期删除。
4. 将 `layout.py` / `target.py` 这类想象中的独立文件并入 `config.py`，除非未来配置体量证明必须再拆。
5. 将 `export.py` / `artifact.py` 的通用逻辑迁到 `python/devproc2/export/` 和 `python/devproc2/artifact/`；Pi0.5 只通过 `PI05Config` 暴露模型 entrypoint 和 artifact recipe。
6. 将 `torch_oracle.py` 移到 `tools/pi05/dump_torch_oracle.py` 或业务生产端，不从 Pi0.5 runtime/model package 暴露。

建议目标结构：

```text
python/devproc2/models/pi05/
  __init__.py
  config.py
  model.py
  ops.py            # Pi0.5 CUDA/HPC op boundary
  weights.py
  cuda/
    pi05_kernels.cu
    fa2/
    gemm/
```

框架层目标结构：

```text
python/devproc2/export/
  pipeline.py
  entrypoint.py
  emit.py
  cli.py

python/devproc2/artifact/
  builder.py
  resources.py
  kernels.py
  manifest.py

tools/pi05/
  dump_torch_oracle.py
```

迁移过程中对外 import 暂时保持兼容，例如 `from devproc2.models.pi05.modules import PI05DenoiseLoop` 仍然可用，内部 re-export 到 `model.py`；但这只是迁移期兼容，不是最终结构。

## Codex Goal 模式任务书

可以直接把下面这段作为 Codex goal objective 使用：

```text
Implement the Pi0.5 refactor described in docs/pi05/09_pi05_refactor_plan.md as a staged, behavior-preserving refactor.

Primary objective:
- Turn python/devproc2/models/pi05 from a performance-script-shaped implementation into a clean first-party business model package.
- Keep forward() and forward_fast() in the same nn.Module: forward() expresses standard op semantics; forward_fast() connects to the high-performance path through python/devproc2/models/pi05/ops.py.
- Do not regress existing Pi0.5 accuracy, export behavior, artifact layout, or performance.

Hard constraints:
- Preserve public imports during migration with thin compatibility wrappers.
- Do not change CUDA kernel argument order, output shape, launch policy, weight file names, or artifact metadata unless a phase explicitly requires it and tests are updated.
- Do not put generic export, artifact, quantization, or oracle producer logic in python/devproc2/models/pi05.
- Do not let devproc2 core quantization import torch, nvidia-modelopt, OpenPI, or a business repo.
- Do not leave formal runtime/export paths depending on FlashRT checkout paths, FlashRT build outputs, DEVPROC2_FLASHRT_FA2_SO, FlashRT pybind modules, or absolute external .so paths.
- Do not hand-write target-specific CUDA symbols in model.py; target dispatch belongs in ops.py, lowering, kernel registry, or Pi0.5-owned CUDA backend.
- Do not use dp.tensor_view in public model code after the view-op phase; public DSL/view semantics must be select, slice, index, split, reshape.

Execution rule:
- Work phase by phase. After each phase, run the phase gate commands and fix failures before continuing.
- If a phase cannot be completed without changing public ABI, kernel ABI, artifact file names, or measured numeric/performance behavior, stop and report the blocker instead of continuing with a speculative redesign.
```

Goal 模式执行时不要把这当成一次“大搬家”。正确节奏是每个阶段都保持 repo 可运行、可测试、可 diff；如果某一阶段变成需要同时重写 frontend、runtime、kernel 和 artifact，说明阶段切分失败，应该收缩本阶段目标。

## 分阶段执行和阶段门禁

### Phase 0：建立 baseline 和硬护栏

本阶段只加保护，不做结构搬迁。

- 记录当前 Pi0.5 test baseline、导出路径、artifact 结构和可运行 benchmark 命令。
- 增加 Pi0.5 source lint：模型层禁止新增裸 `dp.tensor_view`、裸 `dp.call_dps_packed`、裸 `dp.cuda_call`。
- 扩展 `_validate_pi05_module_contract()`：normal path 构图后不得出现 runtime/backend op。
- 增加 import-compat 测试，保证迁移期间旧 import 不断。

阶段门禁：

```bash
pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_artifact.py tests/compiler/test_pi05_weight_package.py tests/compiler/test_pi05_nn_frontend.py
rg "dp\\.cuda_call|dp\\.call_dps_packed|dp\\.tensor_view" python/devproc2/models/pi05/modules.py
rg "FlashRT|DEVPROC2_FLASHRT_FA2_SO|build-fa2-sm89|libflashrt" python/devproc2 runtime tests
```

`rg` 命令在 Phase 0 可以列出现状债务；从 Phase 1 开始必须作为“不得新增脏调用”的对照。

### Phase 1：新增 `PI05Config` 和 `ops.py`，封装不改行为

本阶段目标是把边界立起来，不追求一次性清空旧文件。

- 新增 `config.py`，定义 frozen `PI05Config` dataclass。避免把它做成巨型平铺 dict；建议至少分出 shape、precision、layout、entrypoint、kernel、artifact recipe 这些 typed/nested spec。
- 新增 `ops.py`，迁移 `_packed_call`、CUDA helper、FP8 quant、norm-to-FP8、attention wrapper、Pi0.5 kernel spec 注册。
- 模型里允许 `from . import ops as pi05_ops`，但长期实现文件只有 `ops.py`；若保留 `pi05_ops.py`，只能 re-export `ops.py`。
- 将裸 `dp.cuda_call` / `dp.call_dps_packed` 的调用点逐步替换为 `ops.py` facade。保持 kernel 参数顺序、output shape、launch 和 packed func 名称不变。
- 构图入口按 `PI05Config.precision` 或构图 dtype/quant mode 选择 FP8 / BF16-FP16 fast graph variant；4090、Thor 等 target 不改变 DSL 构图，只改变 `ops.py` / lowering / backend selection。
- 在 `ops.py` 或 backend registry 中注册 target-specific handwritten kernel；`model.py` 不得出现 target-specific CUDA symbol 分支。

阶段门禁：

```bash
pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_artifact.py tests/compiler/test_pi05_weight_package.py tests/compiler/test_pi05_nn_frontend.py
rg "dp\\.cuda_call|dp\\.call_dps_packed" python/devproc2/models/pi05/model.py python/devproc2/models/pi05/modules.py
rg "if .*target|config\\.target|sm89|thor" python/devproc2/models/pi05/model.py python/devproc2/models/pi05/modules.py
```

Phase 1 后，裸 backend 调用只能继续存在于 `ops.py` 或迁移期尚未收敛的旧债位置；不能新增到模型主体。

### Phase 2：前端 view 语义升级

这一阶段必须早于最终收敛 `model.py`。否则会出现一边要求模型层没有 `dp.tensor_view`，一边 frontend 还没有替代 op 的矛盾。

- 设计并实现标准张量语义 op：`select`、`slice`、`index`、`split`、`reshape`。
- `index` 只作为 `select` + `slice` 的 sugar，不做 advanced indexing。
- 不引入 `reshape_view` / `slice_view` / `TensorLayoutView` 这类把实现方式写进名字的 op。
- 将 byte offset / stride 计算下沉到 storage alias lowering、VM codegen 或 runtime tensor descriptor 生成阶段。
- `dp.tensor_view` 的底层能力可以暂时保留为 compiler-internal low-level op，但不得作为 public DSL / model API 暴露。
- 将 Pi0.5 KV/style/QKV 切片 helper 从裸 `dp.tensor_view` 迁移到 `select`、`slice`、`index`、`split`、`reshape`。

阶段门禁：

```bash
pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_nn_frontend.py
rg "dp\\.tensor_view|tensor_view\\(" python/devproc2/models/pi05/model.py python/devproc2/models/pi05/modules.py
rg "reshape_view|slice_view|TensorLayoutView" python/devproc2 python/tests tests
```

Phase 2 后，Pi0.5 模型主体不得再手写 byte offset 或 byte stride；如果底层仍有 alias op，它只能出现在 compiler internal lowering/codegen。

### Phase 3：收敛 Pi0.5 模型文件结构

本阶段做模型目录收敛，但仍保留兼容 wrapper。

- 将 `modules.py` 收敛到 `model.py`，短期 `modules.py` 只做兼容导出。
- `model.py` 保留 `forward()` / `forward_fast()` 双路径；`forward()` 不包含 fast path、CUDA、packed func 或 storage byte math。
- `forward_fast()` 可以根据 `PI05Config.precision` 或构图 dtype/quant mode 选择 FP8 / 非 FP8 graph variant；不得根据 target 在 layer 主体里手写不同 CUDA symbol 或 packed func。
- 将 `_view_bf16_row`、`_qkv_views`、KV cache/style table view 这类 helper 删除或改成标准 view op helper；不新增长期 `layout.py`。
- `weights.py` conversion 和 Module parameter creation 共用同一份 spec，Module init 不再散落 artifact 字符串拼接。

阶段门禁：

```bash
pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_artifact.py tests/compiler/test_pi05_weight_package.py tests/compiler/test_pi05_nn_frontend.py
rg "dp\\.cuda_call|dp\\.call_dps_packed|dp\\.tensor_view" python/devproc2/models/pi05/model.py
rg "Parameter\\(name=f|Parameter\\(name=.*fp8\\.|act_scale\\." python/devproc2/models/pi05/model.py
python - <<'PY'
import devproc2.models.pi05.modules as old
import devproc2.models.pi05.model as new
assert old.PI05DenoiseLoop is new.PI05DenoiseLoop
PY
```

### Phase 4：框架化 export / artifact / oracle 边界

本阶段把通用管线迁出 Pi0.5 目录。

- 引入框架层 `python/devproc2/export/`：通用 GraphBuilder entrypoint selection、compile pipeline、EmitExecutable、EmitABI、CLI scaffolding 都迁出 Pi0.5。
- 引入框架层 `python/devproc2/artifact/`：通用 artifact builder、resource copy、kernel packaging、manifest 写入都迁出 Pi0.5。
- Pi0.5 不再长期保留 `export.py` / `artifact.py`；迁移期只允许薄兼容 wrapper，调用框架层 pipeline 并传入 `PI05Config`。
- 将 `torch_oracle.py` 移到 `tools/pi05/dump_torch_oracle.py` 或业务生产端；Pi0.5 runtime/model package 不 import OpenPI/PyTorch oracle producer。
- 引入 `GraphBuilder.build_method(...)` 或框架 export-level named entrypoint。
- 将 `forward_kv` 重命名为 `materialize_kv`。
- 删除 `forward_fast(prefix_embs, rope_or_valid_rows, rope_interleaved=None)` 这种通过参数个数重载语义的写法。

阶段门禁：

```bash
pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_artifact.py tests/compiler/test_pi05_weight_package.py tests/compiler/test_pi05_nn_frontend.py
rg "openpi|torch_oracle|dump_torch|forward_kv|normal=\"forward_kv\"" python/devproc2/models/pi05 python/devproc2/export tools tests
rg "compile_pi05|emit_pi05|export_pi05|prepare_pi05_artifact" python/devproc2/models/pi05
```

Phase 4 后，`python/devproc2/models/pi05/export.py` 和 `artifact.py` 若仍存在，只能是薄 wrapper；不能保留通用 pass pipeline 或 artifact builder 主体。

### Phase 5：框架层 ModelOpt 量化边界和权重 spec 收口

本阶段先落 devproc2 的量化协议，不要求 devproc2 负责业务 PyTorch 或 calibration。

- 新增框架层 `python/devproc2/quantization/`，提供通用 manifest schema、fusion manifest 表达和 FP8 requant helper；该包不依赖 `torch`、`nvidia-modelopt` 或 OpenPI。
- 可选新增 `python/devproc2/integrations/modelopt/` 或 `tools/quant/modelopt_export.py`，作为 producer-side exporter；它只从业务侧已经量化好的 torch module / ModelOpt state 导出 `QuantizationManifest`。
- Pi0.5 目录不放通用 ModelOpt adapter；只在 `config.py` / `weights.py` 中声明 Pi0.5 deploy quant mapping/fusion spec。
- exporter 不加载 OpenPI，不构造业务 dataloader，不执行 calibration loop，不 import 业务 repo；这些属于业务 PyTorch repo 的生产端职责。
- 对 QKV、gate-up 等 fused projection 执行 common-scale requantization：Pi0.5 spec 声明 fusion group，框架 helper 从原始 FP/BF16 权重重新量化到 fused scalar scale，禁止直接拼接不同 scale 的 FP8 tensor。
- 将 `act_scale.*` 生成纳入框架层 calibration manifest + Pi0.5 deploy manifest，禁止依赖 out-of-band 手工注入的性能包资产。
- 保持 artifact 文件名兼容；只改变代码生成这些名字的位置。

阶段门禁：

```bash
pytest tests/compiler/test_pi05_weight_package.py tests/compiler/test_pi05_artifact.py
python - <<'PY'
import ast
from pathlib import Path
for path in Path("python/devproc2/quantization").rglob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in getattr(node, "names", [])]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            banned = {"torch", "modelopt", "nvidia_modelopt", "openpi"}
            assert not (set(names) & banned), f"{path} imports banned dependency: {set(names) & banned}"
PY
rg "nvidia-modelopt|modelopt|openpi|torch" python/devproc2/quantization python/devproc2/models/pi05
```

### Phase 6：去除 FlashRT FA2 运行时依赖

本阶段可以在 `ops.py` facade 稳定后执行；不要和模型结构收敛混在同一个 diff 里。

- 审计所有 `runtime.cuda.pi05_fa2_*` 调用和 runtime `dlopen` 路径。
- 将使用到的 FA2 wrapper/kernel 最小子集迁入 `python/devproc2/models/pi05/cuda/fa2/`，记录来源、license 和本地修改点。
- 建立 Pi0.5-owned build/export 路径：优先 AOT cubin + kernel table；若暂时使用 shared library，也必须从 devproc2 package 或 artifact 相对路径加载。
- 从正式 runtime path 中删除 FlashRT 绝对路径、`DEVPROC2_FLASHRT_FA2_SO` 依赖和 Python extension 依赖。
- 保留 FlashRT 只作为 benchmark/reference 对照，不作为验收路径。

阶段门禁：

```bash
pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_artifact.py
rg "FlashRT|DEVPROC2_FLASHRT_FA2_SO|build-fa2-sm89|libflashrt|flashrt_fa2" python/devproc2 runtime tests
find python/devproc2/models/pi05/cuda -maxdepth 4 -type f | sort
```

Phase 6 后，正式 artifact/package 必须自包含 Pi0.5 所需 kernel；除 CUDA/cuBLASLt 等系统库外，不要求外部 FlashRT 安装。

### Phase 7：最终文件结构收敛

这是最后清理阶段，只在前面所有行为门禁稳定后做。

- Pi0.5 目录长期只保留 `config.py`、`model.py`、`ops.py`、`weights.py`、`cuda/` 和最小 `__init__.py`。
- 删除或迁走 Pi0.5 目录内长期 `export.py`、`artifact.py`、`torch_oracle.py`、`layout.py`、`target.py`。
- `modules.py` 只允许作为短期兼容 re-export；如果所有下游 import 已迁移，可以删除。
- 更新 tests import，保留旧 import 的兼容测试，直到兼容期结束。

阶段门禁：

```bash
find python/devproc2/models/pi05 -maxdepth 1 -type f | sort
pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_artifact.py tests/compiler/test_pi05_weight_package.py tests/compiler/test_pi05_nn_frontend.py
git diff --check
```

## 验收标准

结构验收：

- `python/devproc2/models/pi05/` 长期只保留 `config.py`、`model.py`、`ops.py`、`weights.py`、`cuda/` 和最小 `__init__.py`；`modules.py` 只允许作为迁移期兼容导出层。
- `python/devproc2/models/pi05/` 长期不保留 `export.py`、`artifact.py`、`torch_oracle.py`、`layout.py`、`target.py`；通用 export/artifact/oracle producer 迁到框架层或 tools。
- `model.py` 没有裸 `dp.cuda_call`、裸 `dp.call_dps_packed`、裸 `dp.tensor_view`。
- `ops.py` 是 Pi0.5 CUDA/HPC op 边界，可以封装手写 kernel、packed func、kernel spec 注册和 target-specific dispatch，但不得承载通用 export/artifact/oracle 逻辑。
- `forward()` 不包含 fast path、CUDA、packed func 或 storage byte math。
- fast path 所有 backend 调用都经过 `ops.py` / `pi05_ops.py` facade。
- FP8 和 BF16/FP16 fast path 可以导出不同 DSL graph variant；共享部分通过 helper/子模块复用，不用一张带大量量化条件分支的图伪复用。
- 4090、Thor 对同一个 precision graph 共享 DSL 构图；target-specific 差异只允许出现在 `ops.py`、lowering registry、kernel registry 和 Pi0.5-owned CUDA backend。
- `forward_fast()` 可以根据 `PI05Config.precision` 或构图 dtype/quant mode 选择 graph variant，但不得根据 target 在 layer 主体里手写不同 CUDA symbol 或 packed func。
- `PI05Config` 是 Pi0.5 配置唯一事实来源；模型结构、shape、entrypoint、precision、layout/fusion、kernel source、artifact recipe 默认值不得散落在 `export.py`/`artifact.py`/临时 helper 中。
- 前端和高层 IR 中没有 byte-level view；KV/style/QKV 切片只通过 `select`、`slice`、`index`、`split`、`reshape` 表达。
- devproc2 不再自研 Pi0.5 量化算法和校准流程；ModelOpt 作为业务生产端离线量化前端，devproc2 core 只消费标准 manifest + FP/BF16 weights。
- `python/devproc2/quantization/` 不 import `torch`、`nvidia-modelopt`、OpenPI 或业务 repo；ModelOpt 读取逻辑只能存在于 optional producer-side exporter。
- `act_scale.*`、`fp8.*.scale` 都由可复现的 framework quantization manifest + Pi0.5 deploy manifest 生成；manifest 记录 ModelOpt version/config、calibration source、component scale、fused common scale 和 layout。
- Pi0.5 模型目录不得私有实现一套 ModelOpt importer/exporter；模型层只能声明 mapping/fusion/layout spec。
- optional ModelOpt exporter 不拥有业务 PyTorch 模型：不得加载业务 checkpoint、不得构造 dataloader、不得执行 calibration，只能抽取业务侧已经产出的量化状态。
- QKV / gate-up fused FP8 权重必须从原始 FP/BF16 权重按 common scale 重新量化，不能直接拼接不同 scale 的 ModelOpt FP8 tensor。
- Pi0.5 正式运行路径不依赖 FlashRT checkout、FlashRT build 目录、FlashRT pybind `.so` 或 `DEVPROC2_FLASHRT_FA2_SO`。
- Pi0.5 使用的 FA2 kernel source、wrapper、build rule 和 runtime load path 都归属 `python/devproc2/models/pi05/cuda` 或 artifact/package 内的 Pi0.5-owned 产物。

行为验收：

- normal compile 只生成 standard op graph。
- fast compile 仍生成等价 CUDA/packed func 调用序列或 Pi0.5-owned kernel 调用序列，不发生性能路径退化。
- 对同一 precision graph，`target=rtx4090_sm89` 和 `target=thor` 的差异体现在 lowered backend graph / executable，而不是两套模型代码或两套 DSL 构图。
- 框架层 `devproc2.export` 可以基于 `PI05Config` 完成 Pi0.5 entrypoint 构图、compile、emit；Pi0.5 目录不自己实现 pass pipeline。
- 框架层 `devproc2.artifact` 可以基于 `PI05Config` artifact recipe 打包 weights/resources/kernels/metadata；Pi0.5 目录不自己实现通用 artifact builder。
- artifact 仍生成完整 `metadata/kernel_table.json`。
- artifact/package 自包含 Pi0.5 所需 kernel；除 CUDA/cuBLASLt 等系统库外，不要求外部 FlashRT 安装。
- 现有 `tests/compiler/test_pi05_fast_modules.py`、artifact tests、runtime smoke 全部通过。

性能和精度验收：

- `sample_precomputed_prefix` 维持 `~13.3ms` 量级。
- full-token 2-view 维持 `~23.4ms` 量级。
- full-token 3-view 维持 `~28.5ms` 量级。
- 已记录的 oracle 误差指标不回退。

## 默认假设

- 不为了代码洁癖牺牲已经拿到的性能基线。
- 不为了代码洁癖牺牲已经拿到的性能基线，但 FlashRT FA2 外部依赖必须移除；迁移时可以先保持 ABI 和数值路径，再把 kernel ownership 收回到 Pi0.5。
- 量化算法、校准和 scale 搜集交给业务侧 ModelOpt 流程；devproc2 不把 ModelOpt 放入 runtime，也不负责加载或校准业务 PyTorch 模型。
- devproc2 core 的量化边界是标准 manifest + requant helper；ModelOpt 读取逻辑若存在，只能是 optional producer-side exporter，不属于 Pi0.5 模型目录。
- 第一阶段不改 OpenPI Torch 模型，不改当前 scalar `A_scale/B_scale` GEMM ABI；通过 deploy-side fusion requantization 对齐当前 runtime。
- 通用 cuBLASLt GEMM packed func 可以继续留在 runtime；Pi0.5 专用 FA2 packed func 应迁移出通用 `cuda_gemm.cc`。
- `forward_fast()` 保留为 `nn.Module` 的一等接口，并与 `forward()` 放在同一个语义 Module 中；不强制引入 `ctx` 参数。precision/target/layout 等配置由 `PI05Config` 或框架构图配置提供。
- FP8 与 BF16/FP16 的差异是 precision/quantization graph variant 差异，可以分图；4090 与 Thor 的差异是 target/backend lowering 差异，不应该分模型图。
- `ops.py` / `pi05_ops.py` 保留在 Pi0.5 目录，作为 CUDA/HPC op 边界。手写 kernel 保留为最高优先级 backend implementation；它通过 `ops.py`、lowering registry 和 `KernelSpec` 接入，不通过 `model.py` 裸 `dp.cuda_call` 接入。
- `dp.tensor_view` 的底层能力可以暂时保留给 compiler internal lowering；公开 DSL 和高层 IR 应迁移到 `select`、`slice`、`index`、`split`、`reshape`。
- `forward_kv` 的根因是多入口构图能力不足，应该修 frontend/export contract，而不是继续给模型加私有方法名。
