"""Kernel implementation registry, launch metadata, and cubin emit tests."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.passes.emit_kernels import EmitKernelsPass
from devproc2.compiler.passes.triton_aot_compile import TritonAOTCompilePass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir import (
    Block,
    CallOp,
    EffectSummary,
    Function,
    IRModule,
    KernelRef,
    Region,
    ReturnOp,
    TensorStructInfo,
    Var,
    StandardOpRef,
)
from devproc2.ir.ops import AllocStorageOp, AllocTensorOp, CallDPSOp
from devproc2.ir.prim_expr import IntImm, PrimVar, ceildiv
from devproc2.kernel.registry import (
    AttrConstraint,
    KernelLaunchSpec,
    KernelMatchKey,
    KernelParamSpec,
    KernelRegistry,
    KernelSpec,
)
from devproc2.kernel.provider import KernelCompileResult, KernelProviderRegistry
from devproc2.vm.executable import Opcode
from devproc2.vm.interpreter import VMInterpreter, _Storage, _Tensor


@pytest.fixture(autouse=True)
def reset_dsl():
    dp.reset_module()
    yield
    dp.reset_module()


def _simple_kernel_module(spec: KernelSpec | None = None) -> IRModule:
    x = Var("x", TensorStructInfo((IntImm(4),), "float32", "cpu"))
    s0 = AllocStorageOp("s0", IntImm(16), 256, "cpu")
    out = AllocTensorOp("out", s0.results[0], 0, (IntImm(4),), "float32")
    k_call = CallDPSOp(
        KernelRef("kernel.relu_fp32", spec),
        (x,),
        (out.results[0],),
        EffectSummary.opaque_call(),
    )
    ret = ReturnOp(values=(out.results[0],))
    fn = Function(body=Region(blocks=(Block(args=(x,), ops=(s0, out, k_call, ret)),)))
    return IRModule(functions={"main": fn})


def test_decorator_registers_backend_neutral_spec():
    launch = KernelLaunchSpec(grid=(8, 1, 1), block=(128, 1, 1), shared_memory_bytes=256)
    params = (
        KernelParamSpec("x", "tensor", source="input", index=0),
        KernelParamSpec("out", "tensor", source="output", index=0),
    )

    @dp.kernel(
        op="relu",
        backend="cutedsl",
        device="cuda",
        dtype="float16",
        output_dtype="float16",
        symbol="relu_kernel_sm90",
        sm_arches=(90,),
        launch=launch,
        params=params,
        compile_options={"arch": "sm90"},
    )
    def relu_kernel(x, out):
        pass

    spec = dp.get_kernel_registry().lookup(
        KernelMatchKey("relu", "cuda", ("float16",)),
        sm_arch=90,
    )
    assert spec is not None
    assert spec.backend == "cutedsl"
    assert spec.launch == launch
    assert spec.params == params
    assert spec.symbol == "relu_kernel_sm90"
    assert spec.compile_options == {"arch": "sm90"}
    assert relu_kernel._kernel_spec is spec


def test_kernel_provider_registry_is_backend_keyed():
    class FakeCudaProvider:
        backend = "cuda"

        def compile(self, spec, kernel_impl, *, output_dir: str, sm_arch: int):
            return KernelCompileResult(
                kernel_name=spec.kernel_name,
                backend=self.backend,
                symbol=spec.symbol,
                artifact_kind="cubin",
                data=b"CUBIN",
                metadata={"sm_arch": sm_arch},
            )

    registry = KernelProviderRegistry()
    provider = FakeCudaProvider()
    spec = KernelSpec(
        op_name="relu",
        device="cuda",
        input_dtypes=("float16",),
        kernel_name="kernel.relu_cuda",
        backend="cuda",
    )

    registry.register(provider)
    result = registry.get("cuda").compile(spec, object(), output_dir="/tmp", sm_arch=90)

    assert result.backend == "cuda"
    assert result.symbol == "relu_cuda"
    assert result.metadata == {"sm_arch": 90}


def test_registry_filters_attrs_layout_sm_and_priority():
    generic = KernelSpec(
        op_name="gelu",
        device="cuda",
        input_dtypes=("float16",),
        kernel_name="kernel.gelu_generic",
        backend="triton",
        priority=0,
    )
    tanh_sm90 = KernelSpec(
        op_name="gelu",
        device="cuda",
        input_dtypes=("float16",),
        kernel_name="kernel.gelu_tanh_sm90",
        backend="cuda",
        sm_arches=(90,),
        priority=10,
        attr_constraints={"approximate": AttrConstraint.eq("tanh")},
        layout_constraints=("contiguous",),
    )
    reg = KernelRegistry()
    reg.register(generic)
    reg.register(tanh_sm90)

    x = Var("x", TensorStructInfo((IntImm(8),), "float16", "cuda"))
    call = CallOp(StandardOpRef("gelu"), (x,), result_name="y", attrs={"approximate": "tanh"})
    key = KernelMatchKey("gelu", "cuda", ("float16",))

    assert reg.lookup(key, sm_arch=90, call_op=call).kernel_name == "kernel.gelu_tanh_sm90"
    assert reg.lookup(key, sm_arch=80, call_op=call).kernel_name == "kernel.gelu_generic"


def test_vm_codegen_keeps_launch_metadata_out_of_kernel_args():
    spec = KernelSpec(
        op_name="relu",
        device="cpu",
        input_dtypes=("float32",),
        kernel_name="kernel.relu_fp32",
        backend="cuda",
        launch=KernelLaunchSpec(grid=(4, 2, 1), block=(128, 1, 1), shared_memory_bytes=64),
    )
    exe = VMCodegenPass().run(_simple_kernel_module(spec))
    main = next(fe for fe in exe.function_table if fe.name == "main")
    k_idx = next(i for i, fe in enumerate(exe.function_table) if fe.name == "kernel.relu_fp32")
    call = next(
        instr
        for instr in exe.instructions[main.instr_offset:main.instr_offset + main.instr_count]
        if instr.opcode == Opcode.CALL and instr.func_idx == k_idx
    )

    assert len(call.arg_regs) == 2
    assert len(call.launch_regs) == 7
    reg_to_val = {ci.reg_idx: exe.constants[ci.const_idx] for ci in main.const_inits}
    assert [reg_to_val[r] for r in call.launch_regs] == [4, 2, 1, 128, 1, 1, 64]


def test_vm_codegen_materializes_dynamic_launch_expr():
    n = PrimVar("N", upper=1024)
    x = Var("x", TensorStructInfo((n,), "float32", "cpu"))
    s0 = AllocStorageOp("s0", n * 4, 256, "cpu")
    out = AllocTensorOp("out", s0.results[0], 0, (n,), "float32")
    spec = KernelSpec(
        op_name="relu",
        device="cpu",
        input_dtypes=("float32",),
        kernel_name="kernel.relu_fp32",
        backend="cuda",
        launch=KernelLaunchSpec(grid=(ceildiv(n, 256), 1, 1), block=(256, 1, 1)),
    )
    call = CallDPSOp(KernelRef("kernel.relu_fp32", spec), (x,), (out.results[0],))
    ret = ReturnOp((out.results[0],))
    fn = Function(Region((Block((x,), (s0, out, call, ret)),)))
    exe = VMCodegenPass().run(IRModule({"main": fn}))
    main = next(fe for fe in exe.function_table if fe.name == "main")
    k_idx = next(i for i, fe in enumerate(exe.function_table) if fe.name == "kernel.relu_fp32")
    kernel_call = next(instr for instr in exe.instructions if instr.func_idx == k_idx)

    assert len(kernel_call.arg_regs) == 2
    assert len(kernel_call.launch_regs) == 7
    # Dynamic grid_x is produced by a ceildiv builtin, not a constant tail arg.
    assert kernel_call.launch_regs[0] not in {
        ci.reg_idx for ci in main.const_inits if exe.constants[ci.const_idx] == 1
    }


def test_mock_kernel_receives_only_abi_args():
    module = _simple_kernel_module()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    received = []

    vm.register_kernel("kernel.relu_fp32", lambda args: received.extend(args))
    in_storage = _Storage(bytearray(16), 1, 0)
    in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
    vm.invoke("main", [in_tensor])

    assert len(received) == 2
    assert all(isinstance(arg, _Tensor) for arg in received)


def test_triton_compile_uses_compile_options(tmp_path):
    fake_cubin = b"\x00FAKE_CUBIN_DATA"

    def mock_kernel():
        pass
    mock_kernel.__name__ = "relu_triton"

    mock_triton = MagicMock()
    mock_compiled = MagicMock()
    mock_compiled.asm = {"cubin": fake_cubin}
    mock_triton.compile.return_value = mock_compiled
    mock_tc = MagicMock()
    mock_tc.ASTSource.return_value = MagicMock()
    mock_tc.GPUTarget.return_value = MagicMock()

    with patch.dict(sys.modules, {"triton": mock_triton, "triton.compiler": mock_tc}):
        result = TritonAOTCompilePass().run(
            mock_kernel,
            str(tmp_path),
            sm_arch=90,
            compile_options={"num_warps": 8},
        )

    assert result == fake_cubin
    assert os.path.exists(tmp_path / "kernels" / "relu_triton.cubin")
    assert mock_triton.compile.call_args.kwargs["options"] == {"num_warps": 8}


def test_triton_missing_import_raises(tmp_path):
    with patch.dict(sys.modules, {"triton": None, "triton.compiler": None}):
        with pytest.raises(ImportError, match="triton"):
            TritonAOTCompilePass().run(lambda: None, str(tmp_path), sm_arch=90)


def test_emit_kernels_writes_cubin_files(tmp_path):
    EmitKernelsPass().run(
        {
            "kernel.relu_fp16": b"RELU",
            "kernel.matmul_fp16": b"MATMUL",
        },
        str(tmp_path),
    )

    assert (tmp_path / "kernels" / "relu_fp16.cubin").read_bytes() == b"RELU"
    assert (tmp_path / "kernels" / "matmul_fp16.cubin").read_bytes() == b"MATMUL"
