# Kernel 与 Lowering 设计

## 当前状态

devproc2 已有：

- `KernelRegistry`
- `KernelSpec`
- `@dp.kernel`
- `DPSLoweringPass`
- `TritonAOTCompilePass`
- C++ `CUDAKernelRegistry` 与 VM `kKernel` dispatch

当前不足：

- kernel selection 只按 `(op_name, device, input_dtypes)` 初筛。
- `KernelSpec` 缺少 attrs/layout/shape 约束的结构化描述。
- dynamic grid 和 kernel 参数 ABI 尚未系统化。
- cubin 与 artifact kernel table 尚未完整贯通。

## KernelSpec 扩展

建议扩展字段：

```python
@dataclass(frozen=True)
class KernelSpec:
    op_name: str
    device: str
    input_dtypes: tuple[str, ...]
    output_dtype: str | None
    kernel_name: str
    backend: str = "triton"
    sm_arches: tuple[int, ...] = ()
    priority: int = 0
    attr_constraints: Mapping[str, AttrConstraint] = field(default_factory=dict)
    layout_constraints: tuple[str, ...] = ()
    match: Callable[[CallOp], bool] | None = None
    grid_fn: Callable[[CallOp], tuple[GridExpr, GridExpr, GridExpr]] | None = None
    launch_meta: LaunchMeta = LaunchMeta()
```

selection 顺序：

1. op name、device、input dtype 精确匹配。
2. SM arch 过滤。
3. attr constraints 过滤。
4. layout constraints 过滤。
5. shape/custom predicate 过滤。
6. priority 选择最高优先级。

## DPS lowering

高层：

```text
%wt = call @transpose(%w) {dim0=0, dim1=1}
%y0 = call @matmul(%x, %wt)
%y = call @add(%y0, %b)
```

lower 后：

```text
%wt = tensor_create.empty(...)
call_dps @kernel.transpose_fp16(%w, %wt) {dim0=0, dim1=1}
%y0 = tensor_create.empty(...)
call_dps @kernel.matmul_fp16(%x, %wt, %y0)
%y = tensor_create.empty(...)
call_dps @kernel.add_fp16(%y0, %b, %y)
```

规则：

- 所有 matched tensor-producing `CallOp` lower 为 DPS kernel。
- attrs 保留到 `CallDPSOp`。
- output tensor shape/dtype/device 来自 `InferStructInfoPass`。
- 若 op 输出 tuple，第一阶段优先在 high-level 拆成多个单输出 op，降低 DPS lowering 复杂度。
- high-level IR 必须保留标准 op。fused linear kernel 只能作为 `transpose + matmul + add` pattern 的 lowering 优化，不能要求前端生成 `linear` op。

## Kernel 参数 ABI

建议 kernel 参数顺序固定：

1. 所有 input tensors。
2. 所有 weight tensors。
3. output tensor。
4. runtime shape scalars。
5. attrs 中需要 runtime 传入的 scalar 常量。

对 Triton kernel，tensor 参数传 raw pointer，shape/stride/size 传 `int64` 或 `int32`。attrs 如果能作为 constexpr 编译进 cubin，则不进入 runtime 参数；否则进入参数列表。

metadata 需要记录：

```json
{
  "name": "kernel.matmul_fp16",
  "op": "matmul",
  "cubin": "kernels/kernel.matmul_fp16.sm90.cubin",
  "symbol": "matmul_fp16",
  "grid": ["ceildiv(M*N, BLOCK_MN)", 1, 1],
  "block": [256, 1, 1],
  "shared_memory_bytes": 0,
  "params": [
    {"name": "x", "kind": "tensor"},
    {"name": "weight", "kind": "tensor"},
    {"name": "bias", "kind": "tensor_optional"},
    {"name": "out", "kind": "tensor"},
    {"name": "M", "kind": "i64"},
    {"name": "N", "kind": "i64"},
    {"name": "K", "kind": "i64"}
  ]
}
```

## 首批 kernel 顺序

### 阶段 A：基础 activation 与 elementwise

- `add`
- `mul`
- `silu`
- `gelu_tanh`
- `where`
- `mask_fill`

这些 kernel 简单，适合先打通 attrs、grid、artifact、C++ dispatch。

### 阶段 B：dense 和 norm

- `matmul_fp16`
- `embedding`
- `layer_norm`
- `rms_norm`
- `adarms_norm`

PyTorch `Linear` 前端展开为 `matmul + add`。kernel 层可以先使用通用 matmul/add kernel，后续通过 pattern fusion 选择 shape-specialized fused kernel。

### 阶段 C：shape/movement

- `reshape/view` 尽量 metadata-only，不生成 kernel。
- `transpose/permute` 先支持必要模式。
- `cat` 支持 prefix/suffix 拼接。
- `slice/gather` 支持取 action horizon、token embedding。

### 阶段 D：attention

先实现 correctness-first：

- q/k/v projection 使用 `matmul + add`。
- rope 单独 kernel。
- attention 路径先按 `matmul + mask + softmax + matmul` 的标准 op 序列分别 lower；后续可在 lowering 阶段识别该 pattern 并选择 fused kernel。
- 后续再融合成 FlashAttention-like kernel。

## Triton AOT 编译

编译流程：

1. `KernelSelectPass` 收集所有被选中的 `KernelSpec`。
2. `EmitKernelsPass` 对每个 spec 调用 `TritonAOTCompilePass`。
3. cubin 写入 `artifact/kernels/`。
4. `metadata/kernel_table.json` 记录 cubin、symbol、launch config、参数 ABI。
5. C++ load artifact 时读取 kernel table，将 cubin 注册到 `CUDAKernelRegistry`。

## shape-specialization 策略

openpi0.5 首版固定：

- batch size 1
- image size 224
- action horizon 和 action dim 来自 Pi0Config
- `num_steps=10`

因此首版允许对常见 shape 编译 specialized kernels。动态 shape 支持后续扩展，但 metadata 仍要记录 symbol 与 shape constraints，避免错误复用 cubin。

## 测试策略

- registry attr/layout/shape selection 单测。
- DPS lowering 后 attrs 和 kernel name 正确。
- mocked Triton compile 写入 cubin 和 kernel table。
- C++ runtime 可加载 kernel table，并在缺失 cubin/symbol 时给出明确错误。
- 每个 kernel 与 PyTorch/numpy reference 单独对齐。
