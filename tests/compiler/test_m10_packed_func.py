"""M10 PackedFunc + call_dps_packed MVP tests."""
from __future__ import annotations

import json
import os
import struct
import tempfile

import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_abi import EmitABIPass
from devproc2.compiler.passes.emit_executable import EmitExecutablePass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
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
    TensorCreateOp,
    TensorCreateKind,
)
from devproc2.ir.prim_expr import IntImm
from devproc2.kernel.registry import KernelRegistry, KernelSpec
from devproc2.vm import Executable, serializer
from devproc2.vm.executable import CalleeKind as VMCalleeKind
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


def _spec(op: str, **kw) -> KernelSpec:
    defaults = dict(device="cpu", input_dtypes=("float16",),
                    kernel_name=f"kernel.{op}_fp16")
    defaults.update(kw)
    return KernelSpec(op_name=op, **defaults)


def _run_pipeline(module: IRModule, *specs: KernelSpec):
    reg = KernelRegistry()
    for s in specs:
        reg.register(s)
    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(reg).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return exe


# ---------------------------------------------------------------------------
# Group A: DSL + IR construction
# ---------------------------------------------------------------------------

class TestDSLEmitsCorrectIR:
    def test_empty_emits_tensor_create_op(self):
        @dp.function
        def f(x: dp.Tensor[(4,), "float32", "cpu"]):
            buf = dp.empty((4,), dtype="float32", device="cpu")
            return buf

        fn = dp.get_module().functions["f"]
        ops = fn.body.blocks[0].ops
        tc_ops = [o for o in ops if isinstance(o, TensorCreateOp)]
        assert len(tc_ops) == 1
        assert tc_ops[0].kind == TensorCreateKind.empty
        assert tc_ops[0].dtype == "float32"
        assert tc_ops[0].device == "cpu"

    def test_call_dps_packed_emits_calldpsop(self):
        @dp.function
        def f(text, max_len):
            tokens = dp.empty((max_len,), dtype="int32", device="cpu")
            dp.call_dps_packed(
                "runtime.tokenizer.encode",
                inputs=[text],
                output=tokens,
            )
            return tokens

        fn = dp.get_module().functions["f"]
        ops = fn.body.blocks[0].ops
        dps_ops = [o for o in ops if isinstance(o, CallDPSOp)]
        assert len(dps_ops) == 1
        assert dps_ops[0].callee == "runtime.tokenizer.encode"
        assert dps_ops[0].callee_kind == IRCalleeKind.packed_func

    def test_call_dps_packed_output_references_tensor_create(self):
        @dp.function
        def f(text, max_len):
            tokens = dp.empty((max_len,), dtype="int32", device="cpu")
            dp.call_dps_packed(
                "runtime.tokenizer.encode",
                inputs=[text],
                output=tokens,
            )
            return tokens

        fn = dp.get_module().functions["f"]
        ops = fn.body.blocks[0].ops
        tc_ops = [o for o in ops if isinstance(o, TensorCreateOp)]
        dps_ops = [o for o in ops if isinstance(o, CallDPSOp)]
        assert len(tc_ops) == 1
        assert len(dps_ops) == 1
        # The output of CallDPSOp must be the result of TensorCreateOp
        assert dps_ops[0].output is tc_ops[0].results[0]

    def test_call_dps_packed_with_none_output(self):
        @dp.function
        def f(k_cache, v_cache):
            dp.call_dps_packed(
                "runtime.update_cache",
                inputs=[k_cache, v_cache],
                output=None,
            )
            return k_cache

        fn = dp.get_module().functions["f"]
        ops = fn.body.blocks[0].ops
        dps_ops = [o for o in ops if isinstance(o, CallDPSOp)]
        assert len(dps_ops) == 1
        assert dps_ops[0].output is None
        assert dps_ops[0].callee_kind == IRCalleeKind.packed_func


# ---------------------------------------------------------------------------
# Group B: VM codegen
# ---------------------------------------------------------------------------

class TestVMCodegen:
    def _build_simple_packed_func_module(self) -> IRModule:
        """IR: main(x: Tensor[(4,), f32, cpu]) → tokens via packed_func."""
        x = Var("x", TensorStructInfo((IntImm(4),), "float32", "cpu"))
        s0 = AllocStorageOp("s0", IntImm(16), 256, "cpu")
        out = AllocTensorOp("out", s0.results[0], 0, (IntImm(4),), "float32")
        pf_call = CallDPSOp("runtime.tokenizer.encode", IRCalleeKind.packed_func,
                            (x,), out.results[0], OpaqueEffect())
        ret = ReturnOp(values=(out.results[0],))
        si = TensorStructInfo((IntImm(4),), "float32", "cpu")
        fn = Function(body=Region(blocks=(Block(args=(x,), ops=(s0, out, pf_call, ret)),)),
                      ret_struct_info=si)
        return IRModule(functions={"main": fn})

    def test_packed_func_in_function_table(self):
        module = self._build_simple_packed_func_module()
        exe = VMCodegenPass().run(module)
        names = [fe.name for fe in exe.function_table]
        assert "runtime.tokenizer.encode" in names

    def test_packed_func_entry_kind(self):
        module = self._build_simple_packed_func_module()
        exe = VMCodegenPass().run(module)
        entry = next(fe for fe in exe.function_table
                     if fe.name == "runtime.tokenizer.encode")
        assert entry.kind == VMCalleeKind.packed_func

    def test_calldps_packed_dst_reg_is_minus_one(self):
        module = self._build_simple_packed_func_module()
        exe = VMCodegenPass().run(module)
        from devproc2.vm.executable import Opcode
        main_entry = next(fe for fe in exe.function_table if fe.name == "main")
        pf_idx = next(i for i, fe in enumerate(exe.function_table)
                      if fe.name == "runtime.tokenizer.encode")
        call_instrs = [
            instr for instr in exe.instructions[
                main_entry.instr_offset:main_entry.instr_offset + main_entry.instr_count
            ]
            if instr.opcode == Opcode.CALL and instr.func_idx == pf_idx
        ]
        assert len(call_instrs) == 1
        assert call_instrs[0].dst_reg == -1


# ---------------------------------------------------------------------------
# Group C: VMInterpreter execution
# ---------------------------------------------------------------------------

class TestVMInterpreterPackedFunc:
    def _build_module(self) -> IRModule:
        x = Var("x", TensorStructInfo((IntImm(4),), "float32", "cpu"))
        s0 = AllocStorageOp("s0", IntImm(16), 256, "cpu")
        out = AllocTensorOp("out", s0.results[0], 0, (IntImm(4),), "float32")
        pf_call = CallDPSOp("runtime.fill_ones", IRCalleeKind.packed_func,
                            (x,), out.results[0], OpaqueEffect())
        ret = ReturnOp(values=(out.results[0],))
        fn = Function(body=Region(blocks=(Block(args=(x,), ops=(s0, out, pf_call, ret)),)),)
        return IRModule(functions={"main": fn})

    def test_interpreter_packed_func_called(self):
        module = self._build_module()
        exe = VMCodegenPass().run(module)
        vm = VMInterpreter(exe)
        called = []

        def fill_fn(args):
            called.append(True)

        vm.register_packed_func("runtime.fill_ones", fill_fn)
        in_storage = _Storage(bytearray(16), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
        vm.invoke("main", [in_tensor])
        assert len(called) == 1

    def test_interpreter_packed_func_writes_output(self):
        module = self._build_module()
        exe = VMCodegenPass().run(module)
        vm = VMInterpreter(exe)

        def fill_fn(args):
            # args: [input_tensor, output_tensor]
            output = args[-1]
            # Write 1.0 (float32 = 0x3f800000) into each element
            data = output.storage.data
            for i in range(4):
                struct.pack_into("<f", data, output.offset + i * 4, 1.0)

        vm.register_packed_func("runtime.fill_ones", fill_fn)
        in_storage = _Storage(bytearray(16), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
        result = vm.invoke("main", [in_tensor])
        assert isinstance(result, _Tensor)
        for i in range(4):
            val = struct.unpack_from("<f", result.storage.data, result.offset + i * 4)[0]
            assert abs(val - 1.0) < 1e-6, f"element {i} = {val}, expected 1.0"

    def test_interpreter_missing_packed_func_raises(self):
        module = self._build_module()
        exe = VMCodegenPass().run(module)
        vm = VMInterpreter(exe)
        in_storage = _Storage(bytearray(16), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
        with pytest.raises(RuntimeError, match="runtime.fill_ones"):
            vm.invoke("main", [in_tensor])

    def test_interpreter_none_output_packed_func(self):
        """output=None packed_func: call succeeds, no tensor return needed."""
        x = Var("x", TensorStructInfo((IntImm(4),), "float32", "cpu"))
        pf_call = CallDPSOp("runtime.log_call", IRCalleeKind.packed_func,
                            (x,), None, OpaqueEffect())
        ret = ReturnOp(values=(x,))
        fn = Function(body=Region(blocks=(Block(args=(x,), ops=(pf_call, ret)),)),)
        module = IRModule(functions={"main": fn})
        exe = VMCodegenPass().run(module)
        vm = VMInterpreter(exe)
        calls = []

        def log_fn(args):
            calls.append(len(args))

        vm.register_packed_func("runtime.log_call", log_fn)
        in_storage = _Storage(bytearray(16), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (4,), 2, 32, 1)
        vm.invoke("main", [in_tensor])
        assert calls == [1]  # called once with 1 arg (no output appended)


# ---------------------------------------------------------------------------
# Group D: Full DSL acceptance criterion
# ---------------------------------------------------------------------------

class TestDSLAcceptance:
    def test_tokenize_dsl_acceptance(self):
        """M10 acceptance: tokenize function with dp.empty + call_dps_packed."""
        @dp.function
        def tokenize(text: dp.Tensor[(1,), "int32", "cpu"],
                     max_len: dp.Tensor[(1,), "int32", "cpu"]):
            tokens = dp.empty((8,), dtype="int32", device="cpu")
            dp.call_dps_packed(
                "runtime.tokenizer.encode",
                inputs=[text],
                output=tokens,
            )
            return tokens

        module = dp.get_module()
        fn = module.functions["tokenize"]
        ops = fn.body.blocks[0].ops
        tc_ops = [o for o in ops if isinstance(o, TensorCreateOp)]
        dps_ops = [o for o in ops if isinstance(o, CallDPSOp)]
        assert len(tc_ops) == 1
        assert len(dps_ops) == 1
        assert dps_ops[0].callee == "runtime.tokenizer.encode"
        assert dps_ops[0].callee_kind == IRCalleeKind.packed_func
        assert dps_ops[0].output is tc_ops[0].results[0]

    def test_tokenize_full_pipeline(self):
        """Full pipeline: DSL → IR → memory planning → VMCodegen → VMInterpreter."""
        @dp.function
        def tokenize(text: dp.Tensor[(1,), "int32", "cpu"]):
            tokens = dp.empty((4,), dtype="int32", device="cpu")
            dp.call_dps_packed(
                "runtime.tokenizer.encode",
                inputs=[text],
                output=tokens,
            )
            return tokens

        module = dp.get_module()
        ctx = PassContext()
        module = InferStructInfoPass().run(module)
        module = DPSLoweringPass(KernelRegistry()).run(module)
        MemoryPlanningPass().run(module, ctx)
        module = LowerTensorCreateToAllocPass(ctx).run(module)
        exe = VMCodegenPass().run(module)

        vm = VMInterpreter(exe)
        sentinel = [42, 7, 13, 99]

        def encode_fn(args):
            output = args[-1]
            for i, v in enumerate(sentinel):
                struct.pack_into("<i", output.storage.data, output.offset + i * 4, v)

        vm.register_packed_func("runtime.tokenizer.encode", encode_fn)
        in_storage = _Storage(bytearray(4), 1, 0)
        in_tensor = _Tensor(in_storage, 0, (1,), 0, 32, 1)
        result = vm.invoke("tokenize", [in_tensor])

        assert isinstance(result, _Tensor)
        assert result.shape == (4,)
        for i, expected in enumerate(sentinel):
            val = struct.unpack_from("<i", result.storage.data, result.offset + i * 4)[0]
            assert val == expected, f"tokens[{i}] = {val}, expected {expected}"

    def test_none_output_not_dce_deleted(self):
        """output=None call_dps_packed must NOT be eliminated by pipeline."""
        @dp.function
        def f(k_cache: dp.Tensor[(4,), "float32", "cpu"],
              v_cache: dp.Tensor[(4,), "float32", "cpu"]):
            dp.call_dps_packed(
                "runtime.update_cache",
                inputs=[k_cache, v_cache],
                output=None,
            )
            return k_cache

        module = dp.get_module()
        ctx = PassContext()
        module = InferStructInfoPass().run(module)
        module = DPSLoweringPass(KernelRegistry()).run(module)
        MemoryPlanningPass().run(module, ctx)
        module = LowerTensorCreateToAllocPass(ctx).run(module)
        exe = VMCodegenPass().run(module)

        names = [fe.name for fe in exe.function_table]
        assert "runtime.update_cache" in names


# ---------------------------------------------------------------------------
# Group E: ABI artifact
# ---------------------------------------------------------------------------

class TestABIArtifact:
    def test_abi_lists_required_packed_func(self, tmp_dir):
        x = Var("x", TensorStructInfo((IntImm(4),), "float32", "cpu"))
        s0 = AllocStorageOp("s0", IntImm(16), 256, "cpu")
        out = AllocTensorOp("out", s0.results[0], 0, (IntImm(4),), "float32")
        pf_call = CallDPSOp("runtime.tokenizer.encode", IRCalleeKind.packed_func,
                            (x,), out.results[0], OpaqueEffect())
        ret = ReturnOp(values=(out.results[0],))
        si = TensorStructInfo((IntImm(4),), "float32", "cpu")
        fn = Function(body=Region(blocks=(Block(args=(x,), ops=(s0, out, pf_call, ret)),)),
                      ret_struct_info=si)
        pre_module = IRModule(functions={"main": fn})
        exe = VMCodegenPass().run(pre_module)
        EmitExecutablePass().run(exe, tmp_dir)

        module = IRModule(functions={"main": fn})
        EmitABIPass().run(module, exe, PassContext(), tmp_dir)

        with open(os.path.join(tmp_dir, "abi.json")) as f:
            abi = json.load(f)
        assert "runtime.tokenizer.encode" in abi.get("required_packed_funcs", [])
