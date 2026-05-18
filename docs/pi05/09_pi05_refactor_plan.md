# Pi0.5 实现锐评与重构方案

## 结论

`python/devproc2/models/pi05` 当前已经证明了两件事：Pi0.5 可以在 devproc2 里跑通，性能也能站住。但这份实现不应该成为 devproc2 的模型实现范式。它现在更像一份被性能压力推着长出来的战时产物：能打，但脏；精度和 latency 达标，但结构边界被打穿了。

最核心的问题不是 `forward_fast()` 存在，也不是 `forward()` 和 `forward_fast()` 放在同一个 `nn.Module` 里。恰恰相反，这个双路径接口是正确方向：`forward()` 负责标准 op 语义，`forward_fast()` 负责显式接入高性能 HPC 路径。真正的问题是 fast path 把 CUDA kernel ABI、packed func 参数顺序、FP8 scale、KV cache byte layout、VM alias 细节和 artifact 约定全部泄漏进了模型代码。模型层不再像模型，而像半个 compiler backend。

本轮重构目标不是重写所有性能优化，而是在不牺牲现有性能基线的前提下，把 Pi0.5 从“能跑的性能脚本”收敛成“可维护的 devproc2 模型实现”。

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

重构后的 Pi0.5 目录应该让每一层只知道自己该知道的东西。关键原则是：不要按 normal/fast 拆成两套模型；同一个 Module 仍然同时拥有 `forward()` 和 `forward_fast()`，只是 `forward_fast()` 通过受控 facade 接入 HPC 后端。

- `model/*.py`：Pi0.5 Module 结构，每个 Module 内保留 `forward()` / `forward_fast()` 双路径。
- `ops.py` 或 `pi05_ops.py`：Pi0.5 high-level primitives，封装 CUDA source kernel 和 packed func。
- `layout.py`：QKV、KV cache、style table、FP8 weight layout 的唯一事实来源。
- `weights.py`：checkpoint conversion、weight package、quantization；只输出 layout/spec，不反向污染模型。
- `export.py`：构图、compile、artifact export orchestration；不承载模型细节。

目标不是让文件数量变多，而是让依赖方向变干净：

```text
model modules -> pi05_ops -> devproc2 DSL/backend
model modules -> layout
weights       -> layout
export        -> model entrypoints
```

Module 可以知道“这里有一个 fast implementation”，但不应该知道 CUDA 参数顺序；`layout.py` 不应该调用 CUDA；`weights.py` 不应该构图；`export.py` 不应该理解 QKV byte stride。

## 重构方案

### 1. 固化 Module contract

新增通用或 Pi0.5 专用 validator，作为构图前的硬门禁：

- 每个 `nn.Module` 必须实现 `forward()`。
- `forward()` 禁止引用 `forward_fast`。
- normal compile 只能使用 standard op；构图后不得出现 `CudaCallOp`、`CallDPSOp`、backend packed func。
- Pi0.5 模型层禁止直接调用 `dp.cuda_call`、`dp.call_dps_packed`、`dp.tensor_view`；CUDA/packed 只能出现在 `pi05_ops.py` 这类受控边界，tensor 切片必须通过标准 `select`、`slice`、`split`、`reshape` 表达。
- fast path 允许 fallback 到 `forward()`，但 fallback 由 compile mode 选择逻辑完成，不由子模块手动调用。

这一步先以 lint/test 约束落地，不急着一次性拆完所有文件。先把“以后不许继续变脏”钉住。

### 2. 建立 Pi0.5 fast primitive facade

把裸 backend 调用集中收口成 Pi0.5 私有 primitive。模型层调用：

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

facade 内部继续使用现有 `dp.cuda_call` 和 `dp.call_dps_packed`，因此性能路径不变；变化只是把 ABI 污染从模型层赶出去。

### 3. 替换前端裸 `dp.tensor_view`

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
prefix_k = dp.index(prefix_k_cache, [layer_idx, :, :, :])
style = dp.index(style_table, [step, layer_idx, :, :])
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

### 4. 统一 destination-passing packed func

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

### 5. 用命名入口替代 `forward_kv`

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

### 6. 权重/layout spec 集中化

新增 `layout.py` 或 `weight_spec.py`，集中定义：

- 每类权重的 logical name。
- BF16 / FP8 artifact name。
- scale name。
- shape。
- layout：`kn` / `nk` / row-major。
- 是否 per-layer。
- 是否 constant tensor。

Module init 不再拼接字符串，只请求 spec：

```python
self.qkv = PI05WeightSpec.decoder_attn_qkv(layer_idx).parameter(device)
```

这样 `weights.py` 和模型代码使用同一份 spec。checkpoint conversion 不再通过字符串约定和 Module 隐式对齐。

### 7. 拆分 `modules.py`

在 contract 和 facade 建立后，再拆文件。建议顺序：

1. 提取 layout helper，不改变行为。
2. 提取 `pi05_ops` facade，不改变行为。
3. 把 vision、prefix encoder、decoder/action expert 拆成独立文件；每个文件内继续让同一个 Module 同时包含 `forward()` 和 `forward_fast()`。
4. 最后将 `modules.py` 变成兼容 re-export 层，避免一次性破坏 import。

建议目标结构：

```text
python/devproc2/models/pi05/
  __init__.py
  artifact.py
  export.py
  layout.py
  ops.py
  modules.py        # compatibility re-export only
  model/
    vision.py
    prefix.py
    decoder.py
    sample.py
  weights.py
  torch_oracle.py
  cuda/pi05_kernels.cu
```

拆分过程中对外 import 暂时保持兼容：

```python
from devproc2.models.pi05.modules import PI05DenoiseLoop
```

仍然可用，内部再 re-export。

## 分阶段执行

### Phase 0：加护栏

- 增加 Pi0.5 source lint：模型层禁止直接使用 `dp.tensor_view`、裸 `dp.call_dps_packed`、裸 `dp.cuda_call`。
- 扩展 `_validate_pi05_module_contract()`：normal path 构图后不得出现 runtime/backend op。
- 对所有现有 build/export path 跑现有 compiler tests，先确认 baseline。

### Phase 1：封装不改行为

- 新增 `layout.py`，迁移 `_view_bf16_row`、`_qkv_views`、KV cache/style table view。
- 新增 `ops.py`，迁移 `_packed_call`、FP8 quant、norm-to-FP8、CUDA helper。
- 模型代码只替换调用点，不改 kernel 参数顺序、不改 output shape、不改 launch。

### Phase 2：修正入口语义

- 引入 `GraphBuilder.build_method(...)` 或 export-level named entrypoint。
- 将 `forward_kv` 重命名为 `materialize_kv`。
- 删除 `forward_fast(prefix_embs, rope_or_valid_rows, rope_interleaved=None)` 这种通过参数个数重载语义的写法。

### Phase 3：权重 spec 收口

- 建立 Pi0.5 weight/layout spec。
- `weights.py` conversion 和 Module parameter creation 共用 spec。
- 保持 artifact 文件名兼容；只改变代码生成这些名字的位置。

### Phase 4：文件拆分

- 拆 vision/prefix/decoder/sample 模块。
- `modules.py` 只做兼容导出。
- 更新 tests import，保留旧 import 的兼容测试。

### Phase 5：前端 view 语义升级

- 设计并实现标准张量语义 op：`select`、`slice`、`index`、`split`、`reshape`；`index` 只作为 `select` + `slice` 的 sugar，不做 advanced indexing；不引入 `reshape_view` / `slice_view` / `TensorLayoutView` 这类把实现方式写进名字的 op。
- 将 Pi0.5 layout helper 从 `dp.tensor_view` 迁移到这些标准 op。
- 将 byte offset / stride 计算下沉到 storage alias lowering、VM codegen 或 runtime tensor descriptor 生成阶段。
- 最终禁止任何模型层或 Pi0.5 layout 层手写 byte offset，除 low-level lowering/codegen 外。

## 验收标准

结构验收：

- `python/devproc2/models/pi05/modules.py` 不再承载全部实现；目标是兼容导出层。
- 模型层没有裸 `dp.cuda_call`、裸 `dp.call_dps_packed`、裸 `dp.tensor_view`。
- `forward()` 不包含 fast path、CUDA、packed func 或 storage byte math。
- fast path 所有 backend 调用都经过 `pi05_ops` facade。
- 前端和高层 IR 中没有 byte-level view；KV/style/QKV 切片只通过 `select`、`slice`、`index`、`split`、`reshape` 表达。

行为验收：

- normal compile 只生成 standard op graph。
- fast compile 仍生成当前 CUDA/packed func 调用序列，不发生性能路径退化。
- artifact 仍生成完整 `metadata/kernel_table.json`。
- 现有 `tests/compiler/test_pi05_fast_modules.py`、artifact tests、runtime smoke 全部通过。

性能和精度验收：

- `sample_precomputed_prefix` 维持 `~13.3ms` 量级。
- full-token 2-view 维持 `~23.4ms` 量级。
- full-token 3-view 维持 `~28.5ms` 量级。
- 已记录的 oracle 误差指标不回退。

## 默认假设

- 不为了代码洁癖牺牲已经拿到的性能基线。
- 不一次性替换 CUDA kernel 或 packed func，只先收敛调用边界。
- `forward_fast()` 保留为 `nn.Module` 的一等接口，并与 `forward()` 放在同一个语义 Module 中；重构目标是让它干净，而不是消灭它或拆成另一套模型。
- `dp.tensor_view` 的底层能力可以暂时保留给 compiler internal lowering；公开 DSL 和高层 IR 应迁移到 `select`、`slice`、`index`、`split`、`reshape`。
- `forward_kv` 的根因是多入口构图能力不足，应该修 frontend/export contract，而不是继续给模型加私有方法名。
