"""M11 @dp.kernel + Triton Cubin MVP tests."""
from __future__ import annotations

import os
import sys
import struct
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_kernels import EmitKernelsPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.triton_aot_compile import TritonAOTCompilePass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir import (
    Block,
    Function,
    IRModule,
    OpaqueEffect,
    Region,
    ReturnOp,
    TensorStructInfo,
    Var,
)
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    CallDPSOp,
    CalleeKind as IRCalleeKind,
)
from devproc2.ir.prim_expr import IntImm
from devproc2.kernel.registry import KernelMatchKey, KernelRegistry, KernelSpec
from devproc2.vm.interpreter import VMInterpreter, _Storage, _Tensor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_dsl():
    dp.reset_module()
    yield
    dp.reset_module()


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def _simple_kernel_module() -> IRModule:
    """IR: main(x: Tensor[(4,), f32, cpu]) → out via kernel call."""
    x = Var("x", TensorStructInfo((IntImm(4),), "float32", "cpu"))
    s0 = AllocStorageOp("s0", IntImm(16), 256, "cpu")
    out = AllocTensorOp("out", s0.results[0], 0, (IntImm(4),), "float32")
    k_call = CallDPSOp("kernel.relu_fp32", IRCalleeKind.kernel,
                       (x,), out.results[0], OpaqueEffect())
    ret = ReturnOp(values=(out.results[0],))
    fn = Function(body=Region(blocks=(Block(args=(x,), ops=(s0, out, k_call, ret)),)),)
    return IRModule(functions={"main": fn})


# ---------------------------------------------------------------------------
# Group A: @dp.kernel decorator
# ---------------------------------------------------------------------------

class TestKernelDecorator:
    def test_decorator_registers_spec(self):
        @dp.kernel(op="relu", backend="triton", device="cuda", dtype="float16")
        def relu_kernel(x, out):
            pass

        reg = dp.get_kernel_registry()
        spec = reg.lookup(KernelMatchKey("relu", "cuda", ("float16",)))
        assert spec is not None
        assert spec.op_name == "relu"

    def test_decorator_sets_kernel_name_from_fn_name(self):
        @dp.kernel(op="silu", backend="triton", device="cuda", dtype="float16")
        def my_silu_kernel(x, out):
            pass

        reg = dp.get_kernel_registry()
        spec = reg.lookup(KernelMatchKey("silu", "cuda", ("float16",)))
        assert spec is not None
        assert spec.kernel_name == "kernel.my_silu_kernel"

    def test_decorator_stores_spec_on_function(self):
        @dp.kernel(op="gelu", backend="triton", device="cuda", dtype="float16")
        def gelu_kernel(x, out):
            pass

        assert hasattr(gelu_kernel, "_kernel_spec")
        assert gelu_kernel._kernel_spec.op_name == "gelu"

    def test_decorator_with_grid_fn(self):
        grid = lambda *inputs: (8, 1, 1)

        @dp.kernel(op="relu2", backend="triton", device="cuda", dtype="float16",
                   grid=grid)
        def relu2_kernel(x, out):
            pass

        spec = relu2_kernel._kernel_spec
        assert spec.grid_fn is not None
        assert spec.grid_fn() == (8, 1, 1)

    def test_decorator_with_sm_arches(self):
        @dp.kernel(op="matmul", backend="triton", device="cuda", dtype="float16",
                   sm_arches=(80, 90))
        def matmul_kernel(x, w, out):
            pass

        spec = matmul_kernel._kernel_spec
        assert spec.sm_arches == (80, 90)

    def test_reset_module_clears_kernel_registry(self):
        @dp.kernel(op="relu_to_clear", backend="triton", device="cuda", dtype="float16")
        def relu_clear(x, out):
            pass

        dp.reset_module()
        reg = dp.get_kernel_registry()
        spec = reg.lookup(KernelMatchKey("relu_to_clear", "cuda", ("float16",)))
        assert spec is None


# ---------------------------------------------------------------------------
# Group B: KernelSpec grid_fn field
# ---------------------------------------------------------------------------

class TestKernelSpecGridFn:
    def test_grid_fn_callable_returns_triple(self):
        spec = KernelSpec(
            op_name="relu",
            device="cuda",
            input_dtypes=("float16",),
            kernel_name="kernel.relu",
            grid_fn=lambda: (4, 2, 1),
        )
        assert spec.grid_fn is not None
        result = spec.grid_fn()
        assert result == (4, 2, 1)
        assert len(result) == 3

    def test_grid_fn_none_by_default(self):
        spec = KernelSpec(
            op_name="relu",
            device="cuda",
            input_dtypes=("float16",),
            kernel_name="kernel.relu",
        )
        assert spec.grid_fn is None

    def test_registry_lookup_unaffected_by_grid_fn(self):
        reg = KernelRegistry()
        spec = KernelSpec(
            op_name="relu",
            device="cuda",
            input_dtypes=("float16",),
            kernel_name="kernel.relu",
            grid_fn=lambda: (1, 1, 1),
        )
        reg.register(spec)
        found = reg.lookup(KernelMatchKey("relu", "cuda", ("float16",)))
        assert found is not None
        assert found.grid_fn is not None


# ---------------------------------------------------------------------------
# Group C: TritonAOTCompilePass
# ---------------------------------------------------------------------------

class TestTritonAOTCompilePass:
    def test_missing_triton_raises_import_error(self, tmp_dir):
        pass_obj = TritonAOTCompilePass()
        # Patch triton import to fail
        with patch.dict(sys.modules, {"triton": None, "triton.compiler": None}):
            with pytest.raises(ImportError, match="triton"):
                pass_obj.run(lambda: None, tmp_dir, sm_arch=90)

    def test_mocked_triton_writes_cubin(self, tmp_dir):
        """Mock triton compilation: verify cubin written to correct path."""
        fake_cubin = b"\x00FAKE_CUBIN_DATA"

        def mock_kernel():
            pass
        mock_kernel.__name__ = "relu_triton"

        # Mock the triton module
        mock_triton = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.asm = {"cubin": fake_cubin}
        mock_triton.compile.return_value = mock_compiled

        mock_tc = MagicMock()
        mock_tc.ASTSource.return_value = MagicMock()
        mock_tc.GPUTarget.return_value = MagicMock()

        with patch.dict(sys.modules, {"triton": mock_triton, "triton.compiler": mock_tc}):
            result = TritonAOTCompilePass().run(mock_kernel, tmp_dir, sm_arch=90)

        assert result == fake_cubin
        cubin_path = os.path.join(tmp_dir, "kernels", "relu_triton.cubin")
        assert os.path.exists(cubin_path)
        with open(cubin_path, "rb") as f:
            assert f.read() == fake_cubin


# ---------------------------------------------------------------------------
# Group D: EmitKernelsPass
# ---------------------------------------------------------------------------

class TestEmitKernelsPass:
    def test_writes_cubin_files(self, tmp_dir):
        cubins = {
            "kernel.relu_fp16":    b"\x01\x02\x03RELU",
            "kernel.matmul_fp16":  b"\x04\x05\x06MATMUL",
        }
        EmitKernelsPass().run(cubins, tmp_dir)

        relu_path = os.path.join(tmp_dir, "kernels", "relu_fp16.cubin")
        matmul_path = os.path.join(tmp_dir, "kernels", "matmul_fp16.cubin")
        assert os.path.exists(relu_path)
        assert os.path.exists(matmul_path)
        with open(relu_path, "rb") as f:
            assert f.read() == cubins["kernel.relu_fp16"]
        with open(matmul_path, "rb") as f:
            assert f.read() == cubins["kernel.matmul_fp16"]

    def test_creates_kernels_directory(self, tmp_dir):
        EmitKernelsPass().run({"kernel.noop": b"\x00"}, tmp_dir)
        assert os.path.isdir(os.path.join(tmp_dir, "kernels"))

    def test_empty_cubins_no_error(self, tmp_dir):
        EmitKernelsPass().run({}, tmp_dir)  # should not raise
        assert os.path.isdir(os.path.join(tmp_dir, "kernels"))

    def test_kernel_prefix_stripped_from_filename(self, tmp_dir):
        EmitKernelsPass().run({"kernel.my_kernel": b"\xAB"}, tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "kernels", "my_kernel.cubin"))


# ---------------------------------------------------------------------------
# Group E: VMInterpreter kernel dispatch
# ---------------------------------------------------------------------------

class TestVMInterpreterKernel:
    def test_register_kernel_mock_invoked(self):
        module = _simple_kernel_module()
        exe = VMCodegenPass().run(module)
        vm = VMInterpreter(exe)
        called = []

        def mock_kernel(args):
            called.append(True)

        vm.register_kernel("kernel.relu_fp32", mock_kernel)
        in_storage = _Storage(bytearray(16), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
        vm.invoke("main", [in_tensor])
        assert len(called) == 1

    def test_kernel_receives_input_and_output_args(self):
        module = _simple_kernel_module()
        exe = VMCodegenPass().run(module)
        vm = VMInterpreter(exe)
        received_args = []

        def mock_kernel(args):
            received_args.extend(args)

        vm.register_kernel("kernel.relu_fp32", mock_kernel)
        in_storage = _Storage(bytearray(16), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
        vm.invoke("main", [in_tensor])

        # At minimum: input tensor + output tensor
        assert len(received_args) >= 2
        assert isinstance(received_args[0], _Tensor)
        assert isinstance(received_args[1], _Tensor)

    def test_kernel_unregistered_returns_none(self):
        """Unregistered kernel is a no-op (returns None) — existing M8 behavior."""
        module = _simple_kernel_module()
        exe = VMCodegenPass().run(module)
        vm = VMInterpreter(exe)
        # Don't register the kernel — should return successfully (no-op)
        in_storage = _Storage(bytearray(16), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
        result = vm.invoke("main", [in_tensor])
        assert isinstance(result, _Tensor)


# ---------------------------------------------------------------------------
# Group F: VMCodegen with kernel_specs + grid dims
# ---------------------------------------------------------------------------

class TestVMCodegenWithKernelSpecs:
    def test_grid_dims_emitted_as_const_args(self):
        """When kernel_specs has a spec with grid_fn, 3 extra const args are emitted."""
        from devproc2.vm.executable import Opcode

        module = _simple_kernel_module()
        specs = {
            "kernel.relu_fp32": KernelSpec(
                op_name="relu",
                device="cpu",
                input_dtypes=("float32",),
                kernel_name="kernel.relu_fp32",
                grid_fn=lambda: (4, 2, 1),
            )
        }
        exe = VMCodegenPass(kernel_specs=specs).run(module)

        main_entry = next(fe for fe in exe.function_table if fe.name == "main")
        k_idx = next(i for i, fe in enumerate(exe.function_table)
                     if fe.name == "kernel.relu_fp32")
        kernel_calls = [
            instr for instr in exe.instructions[
                main_entry.instr_offset:main_entry.instr_offset + main_entry.instr_count
            ]
            if instr.opcode == Opcode.CALL and instr.func_idx == k_idx
        ]
        assert len(kernel_calls) == 1
        # input_tensor + output_tensor + grid_x + grid_y + grid_z = 5
        assert len(kernel_calls[0].arg_regs) == 5

    def test_grid_dim_constants_correct(self):
        """The 3 grid dim constants match what grid_fn returns."""
        module = _simple_kernel_module()
        specs = {
            "kernel.relu_fp32": KernelSpec(
                op_name="relu",
                device="cpu",
                input_dtypes=("float32",),
                kernel_name="kernel.relu_fp32",
                grid_fn=lambda: (8, 4, 2),
            )
        }
        exe = VMCodegenPass(kernel_specs=specs).run(module)

        main_entry = next(fe for fe in exe.function_table if fe.name == "main")
        k_idx = next(i for i, fe in enumerate(exe.function_table)
                     if fe.name == "kernel.relu_fp32")
        kernel_call = next(
            instr for instr in exe.instructions[
                main_entry.instr_offset:main_entry.instr_offset + main_entry.instr_count
            ]
            if instr.func_idx == k_idx
        )
        # Last 3 arg regs hold grid dims; read via const_inits
        grid_regs = kernel_call.arg_regs[-3:]
        # Build register → constant value map from const_inits
        reg_to_val = {}
        for ci in main_entry.const_inits:
            reg_to_val[ci.reg_idx] = exe.constants[ci.const_idx]

        grid_vals = [reg_to_val.get(r) for r in grid_regs]
        assert grid_vals == [8, 4, 2]

    def test_no_grid_dims_without_spec(self):
        """Without kernel_specs, kernel call has only input + output args."""
        from devproc2.vm.executable import Opcode

        module = _simple_kernel_module()
        exe = VMCodegenPass().run(module)  # no kernel_specs

        main_entry = next(fe for fe in exe.function_table if fe.name == "main")
        k_idx = next(i for i, fe in enumerate(exe.function_table)
                     if fe.name == "kernel.relu_fp32")
        kernel_calls = [
            instr for instr in exe.instructions[
                main_entry.instr_offset:main_entry.instr_offset + main_entry.instr_count
            ]
            if instr.opcode == Opcode.CALL and instr.func_idx == k_idx
        ]
        assert len(kernel_calls) == 1
        # input + output = 2 args (no grid dims)
        assert len(kernel_calls[0].arg_regs) == 2
