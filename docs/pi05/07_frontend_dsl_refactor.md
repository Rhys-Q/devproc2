# Pi0.5 前端 DSL 与 CUDA 自定义算子接入重构

## 目标

Pi0.5 的 MVP 性能主要来自自定义 CUDA kernel。`nn.Module.forward_fast()` 因此是正式的一等接口，不是临时旁路：

- `forward()` 保留标准 op 语义，用于可读性、精度对点和 reference IR。
- `forward_fast()` 允许接入自定义算子，但模型代码不应手写 `KernelSpec`、registry、`KernelRef` 或 DPS lowering 细节。
- 本轮只实现 CUDA source-symbol 接入；Triton/CuTeDSL 后续可复用同一 destination-passing 抽象。

## 当前不优雅点

当前 Pi0.5 fast path 的主要问题不是有 `forward_fast()`，而是 fast path 不够规范：

- Module 里直接调用 `dp.call_dps_kernel()`，导致模型结构、kernel symbol、launch、output spec、effect 和 lowering 细节混在一起。
- 每个 fast path 都要显式做静态 kernel 注册，自定义算子不是无感接入。
- CUDA 参数 ABI 容易被 `inputs + outputs` 重排；而手写 CUDA kernel 的参数顺序必须与源码签名一致。
- 权重布局、activation scale、byte stride、cache layout 等低层细节大量散落在 Module 中。

## CUDA 接入 API

新增 `dp.cuda_call(source_symbol, *args, attrs=None, metadata=None)`：

```python
class AddOne(nn.Module):
    def forward_fast(self, x):
        y = dp.empty((4,), dtype="float32", device="cuda")
        dp.cuda_call(
            "kernels/add_one.cu::add_one",
            x,
            y,
            4,
            metadata={"grid": (1, 1, 1), "block": (64, 1, 1)},
        )
        return y
```

约定：

- `source_symbol` 固定为 `"path/to/file.cu::symbol"`。
- `dp.empty()` 创建且首次传入 `cuda_call` 的 tensor 会被自动识别为输出。
- `metadata["outputs"]` 或 `metadata["output_indices"]` 可显式指定输出参数索引。
- 参数传给 CUDA kernel 时保持用户书写顺序；输出只通过 `effect.writes` 标记，不会被移动到参数尾部。
- `metadata` 支持 `grid`、`block`、`shared_memory_bytes`、`sm_arches`、`include_dirs`、`extra_nvcc_flags`、`compile_options`、`kernel_name`、`params`、`input_dtypes`、`output_dtype`、`effect`。

## Lowering 语义

trace 阶段生成 `CudaCallOp`，记录：

- CUDA source path 与 symbol。
- 原始参数顺序。
- 输出参数索引。
- launch metadata。
- attrs/effect。

`DPSLoweringPass` 将 `CudaCallOp` 自动转换为：

- `KernelSpec(backend="cuda", source_path=..., symbol=...)`
- `CallDPSOp(KernelRef(spec), inputs=<原始参数顺序>, outputs=())`
- `effect.writes=<输出 tensor>`

这样用户不需要写 `@dp.kernel` 或注册 `KernelRegistry`，artifact metadata 仍然得到完整 kernel table。`EmitKernelsPass.compile_specs(exe.kernel_specs.values(), artifact_dir, sm_arch=...)` 可将 lowering 自动生成的 CUDA specs 通过 provider 编译进 artifact 的 `kernels/` 目录。

## Pi0.5 迁移结果

Pi0.5 前端模型已迁到独立模型命名空间：

- canonical package 为 `python/devproc2/models/pi05` / `devproc2.models.pi05`；旧 `python/devproc2/pi05` 包已删除，不保留双命名空间。
- Pi0.5 Module 不再调用静态 kernel 注册入口。
- CUDA source kernels 通过模型内的 `dp.cuda_call(...)` 接入；Pi0.5 不再保留独立 kernel catalog/helper 文件。
- 多输出 kernel 用多个 `dp.empty()` 输出并传入同一个 `cuda_call`，lowering 后输出通过 `effect.writes` 表达，CUDA 参数顺序保持为原始 ABI 顺序。
- Pi0.5 fast path 在调用点提供 `kernel_name`、launch 和 `--std=c++17` 编译参数；artifact ABI 以导出阶段生成的 `metadata/kernel_table.json` 为准。
- cuBLASLt、FA2、CUTLASS packed func 暂时保留现有 `dp.call_dps_packed()` 路径；本轮只处理 `.cu` source kernel。
- `forward()` 继续保留标准 op reference path，不因 fast path 迁移而降低可读性。

## 已实现验收

- `dp.cuda_call()` 可在 `nn.GraphBuilder` trace 中无注册生成 `CudaCallOp`。
- lowering 自动生成 CUDA `KernelSpec`，无需 `@dp.kernel`。
- VM codegen 保持 CUDA 参数顺序，并把 launch 放入 `launch_regs`。
- `EmitABIPass` 输出的 `metadata/kernel_table.json` 包含 source、symbol、launch 和 params。
- `EmitKernelsPass.compile_specs(...)` 可直接编译 `exe.kernel_specs` 中的自动 CUDA specs。
- Pi0.5 fast path 已迁到无注册 CUDA helper，覆盖测试见 `tests/compiler/test_cuda_custom_call.py` 和 `tests/compiler/test_pi05_fast_modules.py`。
