"""CUDA source-symbol custom call frontend tests."""
from __future__ import annotations

import json

import devproc2 as dp
import devproc2.nn as nn
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_abi import EmitABIPass
from devproc2.compiler.passes.emit_kernels import EmitKernelsPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir import CudaCallOp, KernelRef, verify
from devproc2.ir.ops import CallDPSOp
from devproc2.kernel.registry import KernelRegistry
from devproc2.kernel.provider import KernelCompileResult, KernelProviderRegistry
from devproc2.vm.executable import Opcode


class _CudaAddOne(nn.Module):
    def __init__(self, source_symbol: str) -> None:
        super().__init__()
        self.source_symbol = source_symbol

    def forward_fast(self, x):
        y = dp.empty((4,), dtype="float32", device="cuda")
        dp.cuda_call(
            self.source_symbol,
            x,
            y,
            4,
            metadata={"grid": (1, 1, 1), "block": (64, 1, 1)},
        )
        return y


def _build_module(source_symbol: str):
    return nn.GraphBuilder().build(
        _CudaAddOne(source_symbol).forward_fast,
        {"x": nn.TensorSpec((4,), "float32", device="cuda")},
    )


def test_cuda_call_traces_without_kernel_registration(tmp_path):
    src = tmp_path / "add_one.cu"
    source_symbol = f"{src}::add_one"
    module = _build_module(source_symbol)
    verify(module)

    ops = module.functions["forward_fast"].body.entry_block.ops
    cuda_ops = [op for op in ops if isinstance(op, CudaCallOp)]

    assert len(cuda_ops) == 1
    cuda = cuda_ops[0]
    assert cuda.source_path == str(src)
    assert cuda.symbol == "add_one"
    assert cuda.output_indices == (1,)
    assert cuda.launch.grid == (1, 1, 1)
    assert cuda.launch.block == (64, 1, 1)


def test_cuda_call_lowers_to_kernel_spec_without_registry(tmp_path):
    src = tmp_path / "add_one.cu"
    module = InferStructInfoPass().run(_build_module(f"{src}::add_one"))
    lowered = DPSLoweringPass(KernelRegistry(), sm_arch=89).run(module)
    verify(lowered, stage="DPSIR")

    dps_ops = [
        op
        for op in lowered.functions["forward_fast"].body.entry_block.ops
        if isinstance(op, CallDPSOp)
    ]
    assert len(dps_ops) == 1
    dps = dps_ops[0]
    assert isinstance(dps.target_ref, KernelRef)
    assert dps.outputs == ()
    assert len(dps.inputs) == 3
    assert dps.effect.writes == (dps.inputs[1],)

    spec = dps.target_ref.spec
    assert spec is not None
    assert spec.backend == "cuda"
    assert spec.source_path == str(src)
    assert spec.symbol == "add_one"
    assert spec.sm_arches == (89,)
    assert spec.launch.block == (64, 1, 1)
    assert [param.source for param in spec.params] == ["input", "output", "input"]
    assert [param.kind for param in spec.params] == ["tensor", "tensor", "scalar"]


def test_cuda_call_vm_codegen_preserves_argument_order(tmp_path):
    src = tmp_path / "add_one.cu"
    module = InferStructInfoPass().run(_build_module(f"{src}::add_one"))
    module = DPSLoweringPass(KernelRegistry(), sm_arch=89).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)

    kernel_name = next(name for name in exe.kernel_specs if "add_one" in name)
    k_idx = next(i for i, fe in enumerate(exe.function_table) if fe.name == kernel_name)
    main = next(fe for fe in exe.function_table if fe.name == "forward_fast")
    call = next(
        instr
        for instr in exe.instructions[main.instr_offset:main.instr_offset + main.instr_count]
        if instr.opcode == Opcode.CALL and instr.func_idx == k_idx
    )

    assert len(call.arg_regs) == 3
    assert len(call.launch_regs) == 7


def test_cuda_call_kernel_table_contains_auto_metadata(tmp_path):
    src = tmp_path / "add_one.cu"
    module = InferStructInfoPass().run(_build_module(f"{src}::add_one"))
    abi_module = module
    module = DPSLoweringPass(KernelRegistry(), sm_arch=89).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)

    EmitABIPass().run(abi_module, exe, ctx, str(tmp_path), target="cuda", target_arch="sm89")
    kernel_table = json.loads((tmp_path / "metadata" / "kernel_table.json").read_text())

    assert len(kernel_table) == 1
    entry = kernel_table[0]
    assert entry["backend"] == "cuda"
    assert entry["source"] == str(src)
    assert entry["symbol"] == "add_one"
    assert entry["launch"]["block"] == [64, 1, 1]
    assert [param["source"] for param in entry["params"]] == ["input", "output", "input"]


def test_cuda_call_generated_specs_can_compile_via_emit_kernels(tmp_path):
    src = tmp_path / "add_one.cu"
    module = InferStructInfoPass().run(_build_module(f"{src}::add_one"))
    module = DPSLoweringPass(KernelRegistry(), sm_arch=89).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)

    class FakeCudaProvider:
        backend = "cuda"

        def compile(self, spec, kernel_impl, *, output_dir: str, sm_arch: int):
            assert spec.source_path == str(src)
            assert spec.symbol == "add_one"
            assert sm_arch == 89
            return KernelCompileResult(
                kernel_name=spec.kernel_name,
                backend="cuda",
                symbol=spec.symbol,
                artifact_kind="cubin",
                data=b"CUBIN",
            )

    providers = KernelProviderRegistry()
    providers.register(FakeCudaProvider())

    cubins = EmitKernelsPass(providers).compile_specs(
        exe.kernel_specs.values(),
        str(tmp_path),
        sm_arch=89,
    )

    assert len(cubins) == 1
    cubin_name = next(iter(cubins)).removeprefix("kernel.")
    assert (tmp_path / "kernels" / f"{cubin_name}.cubin").read_bytes() == b"CUBIN"
