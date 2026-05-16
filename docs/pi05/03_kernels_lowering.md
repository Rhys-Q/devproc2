# Kernel 与 Lowering 机制

## 当前结论

标准 op 是唯一高层语义。Triton、CuTeDSL、CUDA C++、预编译 cubin/PTX 都只是同一个标准 op 的 CUDA backend implementation。

因此 kernel lowering 分三层：

1. `KernelSpec` 描述实现选择、ABI、launch、artifact metadata。
2. `DPSLoweringPass` 只把标准 op lower 到选中的 `CallDPSOp + KernelRef(spec)`。
3. runtime 从 `metadata/kernel_table.json` 加载 cubin/symbol/launch，不从 VM CALL 参数尾部猜测 grid。

## KernelSpec

`KernelSpec` 是 backend-neutral descriptor：

```python
KernelSpec(
    op_name="matmul",
    device="cuda",
    input_dtypes=("float16", "float16"),
    output_dtype="float16",
    kernel_name="kernel.matmul_fp16_sm90",
    backend="cutedsl",          # triton | cutedsl | cuda | python | llvm
    symbol="matmul_fp16_sm90",
    sm_arches=(90,),
    priority=10,
    attr_constraints={...},
    layout_constraints=("contiguous", "contiguous"),
    launch=KernelLaunchSpec(
        grid=(ceildiv(M, 16), ceildiv(N, 16), 1),
        block=(256, 1, 1),
        shared_memory_bytes=0,
    ),
    params=(
        KernelParamSpec("a", "tensor", source="input", index=0),
        KernelParamSpec("b", "tensor", source="input", index=1),
        KernelParamSpec("out", "tensor", source="output", index=0),
    ),
    compile_options={"num_warps": 8},
)
```

选择顺序固定：

1. `(op_name, device, input_dtypes)` 精确匹配。
2. `sm_arches` 过滤。
3. `attr_constraints` 过滤。
4. `layout_constraints` 过滤。
5. `match(call_op)` 作为 shape/custom predicate escape hatch。
6. `priority` 预排序，最高优先。

## Backend Provider

backend provider 只负责把 implementation 编译为 artifact：

```python
provider.compile(spec, kernel_impl, output_dir=..., sm_arch=...)
```

返回 `KernelCompileResult(kernel_name, backend, symbol, artifact_kind, data, metadata)`。

registry/lowering/runtime 不 import Triton、CuTeDSL 或 CUDA 编译 API。后续新增 backend 只注册 provider，不改标准 op lowering 机制。

## DPS Lowering

高层 IR 保留标准 op：

```text
%y = call @gelu(%x) {approximate="tanh"}
```

lower 后：

```text
%y = tensor_create.empty(...)
call_dps @kernel.gelu_tanh_sm90(%x, %y) {approximate="tanh"}
```

规则：

- 只 lower `LoweringKind.kernel` 的 tensor-producing 标准 op。
- `CallDPSOp.attrs` 保留标准 op attrs。
- `KernelRef.spec` 保存选中的 `KernelSpec`。
- tuple-output kernel lowering 暂不展开；当前阶段保持单 tensor output。

## Launch ABI

kernel 普通 ABI 参数只包含 `KernelParamSpec` 对应的输入、输出、shape/stride/runtime scalar。

launch 不属于普通参数：

- 静态 launch 写入 `metadata/kernel_table.json`。
- 动态 launch 在 `VMCodegenPass` 中 materialize 为 `Instruction.launch_regs`。
- C++ launcher 接收独立的 `launch_args = grid3 + block3 + shared_memory_bytes`。
- 不再使用“最后三个 int 是 grid”的旧约定。

## Artifact

`metadata/kernel_table.json` 是 runtime kernel metadata 的事实来源：

```json
{
  "name": "kernel.relu_fp16",
  "kind": "kernel",
  "backend": "cuda",
  "op": "relu",
  "cubin": "kernels/relu_fp16.cubin",
  "symbol": "relu_fp16",
  "launch": {
    "grid": [128, 1, 1],
    "block": [256, 1, 1],
    "shared_memory_bytes": 0,
    "cluster": [1, 1, 1],
    "cooperative": false
  },
  "params": [
    {"name": "x", "kind": "tensor", "source": "input", "index": 0},
    {"name": "out", "kind": "tensor", "source": "output", "index": 0}
  ]
}
```

C++ `Executable::Load()` 读取 kernel table，加载 cubin，并注册到 `CUDAKernelRegistry`。缺失 cubin、symbol 或非法 launch 字段必须报出包含 kernel name 和字段名的错误。

## 首批 kernel 顺序

1. elementwise：`add`、`mul`、`silu`、`gelu_tanh`、`where`、`mask_fill`。
2. dense/norm：`matmul_fp16`、`embedding`、`layer_norm`、`rms_norm`、`adarms_norm`。
3. movement：metadata-only `reshape/view`，必要 `permute_dims`、`cat`、`slice/gather`。
4. attention：先 correctness-first 标准 op 序列，再在 lowering 中做 pattern fusion。

PyTorch `Linear` 前端继续展开为 `matmul + add`；fused linear 只能作为 lowering/pattern selection 优化，不能要求前端生成 `linear` op。

## 测试策略

- registry：SM、attrs、layout、priority、custom predicate。
- lowering：attrs 保留，`KernelRef.spec` 正确。
- VM codegen：launch_regs 与 arg_regs 分离，动态 `PrimExpr` launch 可 materialize。
- artifact：kernel table 输出 backend、symbol、launch、params、cubin。
- runtime：artifact load 注册 kernel，launcher 不解析普通参数尾部。
