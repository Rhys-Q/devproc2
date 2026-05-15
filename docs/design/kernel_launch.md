# M11：@dp.kernel + Triton Cubin + CUDA 启动 设计文档

> 适合读者：了解基本 Python / CUDA 概念，没有编译器背景的初学者。
>
> 读完本文，你将理解：devproc2 如何把一个用 Triton 写的 Python GPU kernel，
> 编译成 cubin 二进制，打包进 Artifact，在 VM 里通过 `cuLaunchKernel` 启动。

---

## 1. 背景：为什么需要 @dp.kernel？

在 M6 / DPS Lowering 之后，devproc2 已经能把 `dp.ops.relu(x)` 这种高层调用翻译成：

```
%y = dp.empty(shape, dtype, device)
call_dps @kernel.relu_fp16(inputs=[%x], output=%y, effect=write(%y))
```

但这里的 `@kernel.relu_fp16` 只是一个**名字**。VM 在 `kKernel` 分支调用时，需要真正的可执行二进制（cubin）。

**M11 解决的问题：从名字到可执行。**

用户只需要写一个 Triton kernel，加上装饰器，devproc2 全自动完成：

1. 把 Triton Python → cubin 二进制
2. 把 cubin 打包进 Artifact
3. 运行时加载 cubin，调用 `cuLaunchKernel`

---

## 2. 全局流程概览

```
用户代码
   │
   │  @dp.kernel(op="relu", ...)
   │  def relu_kernel(x, out): ...   ← Triton Python kernel
   │
   ▼
[注册阶段] KernelRegistry.register(KernelSpec)
   │         ↑ kernel 名、device、dtype、grid_fn
   │
   ▼
[编译阶段] DPSLoweringPass
   │         ↑ CallOp @relu → TensorCreateOp + CallDPSOp @kernel.relu_kernel
   │
   ▼
[编译阶段] TritonAOTCompilePass        [可选，需要 triton 已安装]
   │         ↑ relu_kernel.__triton_fn → cubin bytes
   │
   ▼
[打包阶段] EmitKernelsPass
   │         ↑ cubin bytes → artifact/kernels/relu_kernel.cubin
   │
   ▼
[运行时] Executable::Load(artifact_dir)
   │         ↑ 读 kernels/*.cubin → CUDAKernelRegistry::Register
   │
   ▼
[运行时] VMState::DispatchExternal (kKernel)
   │         ↑ CUDAKernelRegistry::Get(name) → KernelObj
   │         ↑ CUDAKernelLauncher_Launch(kernel, args, stream)
   │
   ▼
[GPU]  cuLaunchKernel(CUfunction, grid, block, args, stream)
```

---

## 3. 注册阶段：@dp.kernel 装饰器

### 3.1 用法

```python
import devproc2.frontend.dsl as dp

@dp.kernel(
    op="relu",             # 对应 dp.ops.relu 的算子名
    backend="triton",      # 实现后端
    device="cuda",
    dtype="float16",
    grid=lambda: (128, 1, 1),  # 静态 grid：(grid_x, grid_y, grid_z)
    sm_arches=(80, 90),    # 支持的 SM 架构（空 = 任意）
)
def relu_kernel(x, out):
    """Triton kernel：element-wise relu。"""
    import triton
    import triton.language as tl

    pid = tl.program_id(0)
    BLOCK = 256
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x_vals = tl.load(x + offs)
    out_vals = tl.maximum(x_vals, 0.0)
    tl.store(out + offs, out_vals)
```

### 3.2 装饰器做了什么

```python
def kernel(*, op, backend, device, dtype, grid=None, sm_arches=()):
    def decorator(fn):
        spec = KernelSpec(
            op_name=op,
            device=device,
            input_dtypes=(dtype,),
            kernel_name=f"kernel.{fn.__name__}",  # "kernel.relu_kernel"
            sm_arches=sm_arches,
            grid_fn=grid,       # ← 记住 grid 表达式
        )
        _kernel_registry.register(spec)   # ← 注册到模块级 registry
        fn._kernel_spec = spec            # ← 挂在函数对象上
        return fn
    return decorator
```

注册之后，`dp.get_kernel_registry()` 就能查到：

```python
from devproc2.kernel.registry import KernelMatchKey

reg = dp.get_kernel_registry()
spec = reg.lookup(KernelMatchKey("relu", "cuda", ("float16",)))
print(spec.kernel_name)   # "kernel.relu_kernel"
print(spec.grid_fn())     # (128, 1, 1)
```

---

## 4. KernelSpec：kernel 的元数据

```python
@dataclass(frozen=True)
class KernelSpec:
    op_name:      str                  # 匹配 CallOp 的算子名（去掉 "@"）
    device:       str                  # "cuda" / "cpu"
    input_dtypes: tuple[str, ...]      # 输入张量的 dtype 列表
    kernel_name:  str                  # CallDPSOp 中用的名字
    sm_arches:    tuple[int, ...] = () # () = 不限 SM；(80,90) = Ampere/Hopper
    priority:     int = 0              # 多 kernel 候选时优先级高的胜出
    match:        Optional[Callable]   # 自定义 predicate（可选）
    grid_fn:      Optional[Callable]   # 返回 (grid_x, grid_y, grid_z) 的函数
```

### 4.1 两级 dispatch

`KernelRegistry.lookup(key, sm_arch, call_op)` 的匹配逻辑：

```
Level 1（O(1)，字典查找）：
  key = (op_name, device, input_dtypes)
  → 找到候选列表（按 priority 降序排好）

Level 2（线性扫描候选）：
  ① SM 架构过滤：sm_arches 非空且 sm_arch 不在里面 → 跳过
  ② 自定义 predicate：spec.match(call_op) 返回 False → 跳过
  返回第一个通过的 spec
```

---

## 5. 编译阶段：VMCodegenPass 如何处理 grid dims

当 IR 中有 `CallDPSOp(callee="kernel.relu_kernel", callee_kind=kernel, ...)` 时，VMCodegenPass 需要把 **grid 维度** 也编码进 CALL 指令的参数里。

### 5.1 为什么 grid dims 要作为参数传递？

VM 的 CALL 指令格式只有：

```
CALL dst_reg, func_idx, [arg_regs...]
```

没有专门的"launch config"字段。devproc2 的解决方案：**把 grid_x/grid_y/grid_z 作为最后 3 个整数参数追加到 arg_regs**。

```
CALL dst=-1, @kernel.relu_kernel, [r_x, r_out, r_128, r_1, r_1]
                                                ↑      ↑   ↑
                                            grid_x  grid_y  grid_z
```

C++ 侧（`CUDAKernelLauncher_Launch`）通过探测最后 3 个 arg 是否都是 Int 来提取 grid dims，其余 args 是 tensor 指针。

### 5.2 codegen 中的实现

```python
# vm_codegen.py
def _lower_calldps(self, op: CallDPSOp, ctx: _FnCtx) -> None:
    arg_regs = [ctx.reg_of(v) for v in op.inputs]
    if op.output is not None:
        arg_regs.append(ctx.reg_of(op.output))

    # 对 kernel callee：追加静态 grid dims
    if op.callee_kind == IRCalleeKind.kernel:
        spec = self._kernel_specs.get(op.callee)
        if spec is not None and spec.grid_fn is not None:
            grid = spec.grid_fn()     # 编译时计算，例如 (128, 1, 1)
            for g in grid:
                arg_regs.append(ctx.reg_for_int(int(g)))

    ctx.emit(Instruction(opcode=CALL, dst_reg=-1,
                         func_idx=..., arg_regs=arg_regs))
```

---

## 6. TritonAOTCompilePass：Triton Python → cubin

### 6.1 什么是 AOT 编译？

通常 Triton 是**即时编译（JIT）**的：第一次调用 kernel 时才触发 LLVM/PTXAS 编译。

devproc2 使用**提前编译（AOT）**：在构建 Artifact 时就把 `.ptx`/`.cubin` 生成好，装进包里。

好处：
- 部署时不需要 Triton / NVCC 环境
- 启动时间固定，没有 JIT 预热延迟
- cubin 可以做版本管理、签名验证

### 6.2 使用方式

```python
from devproc2.compiler.passes.triton_aot_compile import TritonAOTCompilePass

cubin = TritonAOTCompilePass().run(
    kernel_fn=relu_kernel,    # @triton.jit 装饰的函数
    output_dir="/tmp/my_model",
    sm_arch=80,               # A100 = sm_80
)
# 同时在 /tmp/my_model/kernels/relu_kernel.cubin 写入文件
# 返回 cubin bytes
```

### 6.3 内部流程

```python
def run(self, kernel_fn, output_dir, sm_arch=90, signature=None):
    import triton
    import triton.compiler as tc

    # 1. 构建编译目标
    source = tc.ASTSource(fn=kernel_fn, signature=signature or {})
    target = tc.GPUTarget("cuda", sm_arch, 32)

    # 2. 触发 Triton 编译
    compiled = triton.compile(source, target=target)

    # 3. 提取 cubin bytes
    cubin_bytes = compiled.asm["cubin"]

    # 4. 写入 artifact/kernels/<name>.cubin
    os.makedirs(os.path.join(output_dir, "kernels"), exist_ok=True)
    with open(f"{output_dir}/kernels/{kernel_fn.__name__}.cubin", "wb") as f:
        f.write(cubin_bytes)

    return cubin_bytes
```

如果没安装 Triton，会得到清晰的错误：
```
ImportError: triton is required for TritonAOTCompilePass.
Install with: pip install triton
```

---

## 7. EmitKernelsPass：把 cubin 写进 Artifact

```python
from devproc2.compiler.passes.emit_kernels import EmitKernelsPass

EmitKernelsPass().run(
    kernel_cubins={
        "kernel.relu_kernel":   b"<cubin bytes>",
        "kernel.matmul_fp16":   b"<cubin bytes>",
    },
    output_dir="/tmp/my_model"
)
```

结果：
```
/tmp/my_model/
  kernels/
    relu_kernel.cubin      ← 注意去掉了 "kernel." 前缀
    matmul_fp16.cubin
```

---

## 8. 运行时：C++ 侧如何加载和执行 kernel

### 8.1 CUDAKernelRegistry

类似 PackedFuncRegistry，但存的是 `KernelObj`（含 cubin 数据 + 函数名 + block dims）：

```cpp
class CUDAKernelRegistry {
    // 注册：名字 + cubin bytes + CUDA 函数名 + block_dims
    void Register(const std::string& name,
                  const std::vector<uint8_t>& cubin_data,
                  const std::string& func_name,
                  std::array<int32_t,3> block_dims = {128,1,1});

    // 查找：返回 KernelObj* 或 nullptr
    KernelObj* Get(const std::string& name) const;
};
```

`KernelObj` 结构：

```cpp
class KernelObj : public Object {
    std::string name;                  // "kernel.relu_kernel"
    std::string func_name;            // cubin 内的 CUDA 函数名
    std::vector<uint8_t> cubin_data;  // cubin 二进制内容
    std::array<int32_t,3> block_dims; // {128, 1, 1}
};
```

### 8.2 Executable::Load() 如何注册 kernels

```cpp
// 加载 artifact 时的流程（伪代码）
auto exec = Executable::Load("/tmp/my_model");
// Load 内部：
// 1. 读 executable.vm → Deserialize()
// 2. 读 abi.json → 版本检查 + required_packed_funcs 检查
// 3. 读 metadata/kernel_table.json → 每个 kernel 的元数据
// 4. for each kernel_entry:
//      data = read_file("kernels/<name>.cubin")
//      CUDAKernelRegistry::Global().Register(name, data, func_name, block_dims)
```

### 8.3 VM 执行时的 kernel dispatch

```cpp
// vm.cc DispatchExternal
case VMCalleeKind::kKernel: {
#ifdef DEVPROC2_WITH_CUDA
    auto* k = CUDAKernelRegistry::Global().Get(callee.name);
    if (!k) {
        throw std::runtime_error(
            "Kernel '" + callee.name + "' not registered in CUDAKernelRegistry");
    }
    Device cuda_dev{kDLCUDA, 0};
    void* stream = GetDefaultStream(cuda_dev);   // 懒初始化的 per-device stream
    CUDAKernelLauncher_Launch(k, args, stream);
    return VMValue{};
#else
    throw std::runtime_error("kKernel requires DEVPROC2_WITH_CUDA");
#endif
}
```

### 8.4 CUDAKernelLauncher_Launch：解包参数，调用 cuLaunchKernel

```cpp
void CUDAKernelLauncher_Launch(
    const KernelObj* kernel,
    std::vector<VMValue>& args,
    void* stream
) {
    // 1. 从尾部提取 grid dims（如果最后 3 个 arg 是 Int）
    uint32_t grid_x = 1, grid_y = 1, grid_z = 1;
    if (args.size() >= 3 &&
        args[tail].IsInt() && args[tail-1].IsInt() && args[tail-2].IsInt()) {
        grid_z = args[tail].AsInt();
        grid_y = args[tail-1].AsInt();
        grid_x = args[tail-2].AsInt();
        tensor_count -= 3;
    }

    // 2. 把 tensor VMValue 转成 void* 数据指针
    std::vector<void*> raw_args;
    for (int i = 0; i < tensor_count; ++i) {
        if (args[i].IsObjectRef()) {
            auto* tobj = args[i].AsObjectAs<TensorObj>();
            raw_args.push_back(tobj->dl().data);  // GPU 数据指针
        }
    }

    // 3. 加载 cubin（带缓存，每个 kernel 只加载一次）
    CUfunction fn = get_or_load_function(kernel);
    //   内部：cuModuleLoadData(cubin_data) + cuModuleGetFunction(func_name)

    // 4. 启动！
    cuLaunchKernel(
        fn,
        grid_x, grid_y, grid_z,     // grid
        block_x, block_y, block_z,  // block（来自 kernel->block_dims）
        0,                          // sharedMemBytes
        static_cast<CUstream>(stream),
        raw_args.data(),            // kernel 参数指针数组
        nullptr
    );
}
```

---

## 9. module 缓存：cubin 只加载一次

`get_or_load_function` 维护一个进程级缓存：

```
g_module_cache: unordered_map<kernel_name, ModuleEntry>
                                              ├── CUmodule
                                              └── func_cache: map<func_name, CUfunction>
```

第一次调用 `@kernel.relu_kernel`：
```
g_module_cache["kernel.relu_kernel"] 不存在
→ cuModuleLoadData(cubin_data)    → CUmodule
→ cuModuleGetFunction("relu_kernel_fp16") → CUfunction
→ 存入缓存
```

之后每次调用：直接从缓存拿，不再 `cuModuleLoadData`。

---

## 10. 完整 Demo：从 @dp.kernel 到 GPU 执行

### 10.1 定义 kernel 和模型

```python
import devproc2.frontend.dsl as dp

# ① 定义 Triton kernel，注册为 relu op 的 cuda float16 实现
@dp.kernel(
    op="relu",
    backend="triton",
    device="cuda",
    dtype="float16",
    grid=lambda: (32, 1, 1),    # 32 个 block，每 block 256 线程 → 8192 元素
)
def relu_fp16(x, out):
    import triton.language as tl
    pid = tl.program_id(0)
    BLOCK = 256
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < 8192
    v = tl.load(x + offs, mask=mask)
    tl.store(out + offs, tl.maximum(v, 0.0), mask=mask)

# ② 定义模型函数
@dp.function
def apply_relu(x: dp.Tensor[(8192,), "float16", "cuda"]):
    return dp.ops.relu(x)
```

### 10.2 AOT 编译 + 打包

```python
import os, tempfile
from devproc2.compiler.passes.triton_aot_compile import TritonAOTCompilePass
from devproc2.compiler.passes.emit_kernels import EmitKernelsPass

output_dir = "/tmp/relu_demo"
os.makedirs(output_dir, exist_ok=True)

# 编译 Triton kernel → cubin
cubin = TritonAOTCompilePass().run(relu_fp16, output_dir, sm_arch=80)
print(f"cubin 大小：{len(cubin)} bytes")
# cubin 大小：12288 bytes

# 写入 artifact
EmitKernelsPass().run({"kernel.relu_fp16": cubin}, output_dir)
# 生成 /tmp/relu_demo/kernels/relu_fp16.cubin
```

### 10.3 完整编译 pipeline

```python
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.compiler.pass_context import PassContext

module = dp.get_module()
kernel_reg = dp.get_kernel_registry()   # 包含 relu_fp16 的 spec

module = InferStructInfoPass().run(module)
module = DPSLoweringPass(kernel_reg).run(module)    # relu CallOp → kernel CallDPSOp
ctx = PassContext()
MemoryPlanningPass().run(module, ctx)
module = LowerTensorCreateToAllocPass(ctx).run(module)

# 传入 kernel_specs，让 codegen 知道 grid_fn
kernel_specs = {"kernel.relu_fp16": kernel_reg.lookup(
    KernelMatchKey("relu", "cuda", ("float16",))
)}
exe = VMCodegenPass(kernel_specs=kernel_specs).run(module)
```

### 10.4 运行时（需要 GPU）

```python
# 假设使用 C++ VMState
import ctypes
lib = ctypes.CDLL("libdevproc2_runtime.so")

# 加载 artifact（内部注册 kernel 到 CUDAKernelRegistry）
exec_ptr = lib.Executable_Load("/tmp/relu_demo")

# 准备输入：全 -1.0 的 float16 tensor（在 GPU 上）
import torch
x_torch = torch.full((8192,), -1.0, dtype=torch.float16, device="cuda")

# 执行（通过 dlpack 桥接）
vm = lib.VMState_Create(exec_ptr)
result = lib.VMState_Invoke(vm, "apply_relu", [x_torch])
# 结果：全 0.0（relu(-1.0) = 0.0）
```

### 10.5 Python 端用 mock kernel 测试（无需 GPU）

```python
from devproc2.vm.interpreter import VMInterpreter, _Storage, _Tensor

vm = VMInterpreter(exe)

def relu_mock(args):
    """numpy relu mock，模拟 GPU kernel 行为。"""
    import struct, numpy as np
    in_t, out_t = args[0], args[1]
    data = np.frombuffer(in_t.storage.data, dtype=np.float16)
    result = np.maximum(data, 0)
    out_t.storage.data[:] = result.tobytes()

vm.register_kernel("kernel.relu_fp16", relu_mock)
# ... 调用 vm.invoke("apply_relu", [x_tensor])
```

---

## 11. 关键设计决策总结

| 设计决策 | 原因 |
|---|---|
| grid dims 追加为最后 3 个 Int arg | 复用 CALL 指令，不新增 opcode |
| cubin 缓存在进程级 static map | kernel 通常固定，避免每次 `cuModuleLoadData` 的开销 |
| `CUDAKernelRegistry` 与 `PackedFuncRegistry` 分离 | kernel 携带二进制数据，不适合 `std::function` 包装 |
| AOT 编译（非 JIT） | 部署环境不需要 Triton/NVCC，启动延迟固定 |
| `#ifdef DEVPROC2_WITH_CUDA` 条件编译 | 非 CUDA 机器可以正常编译和跑 CPU 测试 |
| `@dp.kernel` 注册到模块级 `_kernel_registry` | 与 `@dp.function` 用同一个模块生命周期，`reset_module()` 可以清空 |

---

## 12. 与其他组件的关系

```
@dp.kernel
   ├── 注册 KernelSpec → _kernel_registry
   ├── KernelSpec 被 DPSLoweringPass 查询 → CallDPSOp
   ├── KernelSpec.grid_fn 被 VMCodegenPass 调用 → grid dim 常量
   └── KernelSpec 指向 Triton kernel fn → TritonAOTCompilePass 编译

cubin
   ├── TritonAOTCompilePass 生成
   ├── EmitKernelsPass 写入 artifact
   └── CUDAKernelRegistry 加载后存入 KernelObj

CUDAKernelRegistry
   ├── Executable::Load() 填充
   └── VMState::DispatchExternal (kKernel) 查询

CUDAKernelLauncher_Launch
   ├── cuModuleLoadData (带缓存)
   ├── cuModuleGetFunction
   └── cuLaunchKernel
```
