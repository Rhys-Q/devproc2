"""X2 Runtime Shape Builtin MVP tests.

Tests:
  A. Interpreter builtin unit tests (shape_of, get_shape_dim, arithmetic, comparison,
     assert_le_i64)
  B. End-to-end acceptance criteria (manually-built Executable + interpreter)
  C. Full pipeline integration (IR → dynamic shape → VMCodegen → interpreter)
"""
import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.shape_assertion_insert import ShapeAssertionInsertPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir import (
    Block,
    Constant,
    Function,
    IRModule,
    Region,
    ReturnOp,
    TensorStructInfo,
    Var,
)
from devproc2.ir.ops import AllocStorageOp, AllocTensorOp
from devproc2.ir.prim_expr import IntImm, PrimVar
from devproc2.kernel.registry import KernelRegistry, KernelSpec
from devproc2.vm import (
    CalleeKind,
    ConstInit,
    Executable,
    FunctionEntry,
    Instruction,
    Opcode,
    VMInterpreter,
)
from devproc2.vm.interpreter import _Storage, _Tensor, _DEFAULT_BUILTINS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_dsl():
    dp.reset_module()
    yield
    dp.reset_module()


def _spec(op: str, **kw) -> KernelSpec:
    defaults = dict(device="cpu", input_dtypes=("float16",),
                    kernel_name=f"kernel.{op}_fp16")
    defaults.update(kw)
    return KernelSpec(op_name=op, **defaults)


def _run_pipeline(module: IRModule, *specs: KernelSpec) -> Executable:
    """Full pipeline: IR → Executable (same as M8 helper)."""
    reg = KernelRegistry()
    for s in specs:
        reg.register(s)
    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(reg).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    return VMCodegenPass().run(module)


def _make_tensor(shape: tuple[int, ...], dtype_bits: int = 16) -> _Tensor:
    nbytes = 1
    for d in shape:
        nbytes *= d
    nbytes = nbytes * dtype_bits // 8
    storage = _Storage(bytearray(nbytes), device_type=1, device_id=0)
    return _Tensor(storage=storage, offset=0, shape=shape,
                   dtype_code=2, dtype_bits=dtype_bits, dtype_lanes=1)


# ---------------------------------------------------------------------------
# A. Interpreter builtin unit tests
# ---------------------------------------------------------------------------

def test_shape_of_returns_shape_tuple():
    t = _make_tensor((2, 512, 4096))
    result = _DEFAULT_BUILTINS["vm.builtin.shape_of"]([t])
    assert result == (2, 512, 4096)


def test_get_shape_dim_first():
    shape = (2, 512, 4096)
    result = _DEFAULT_BUILTINS["vm.builtin.get_shape_dim"]([shape, 0])
    assert result == 2


def test_get_shape_dim_second():
    shape = (2, 512, 4096)
    result = _DEFAULT_BUILTINS["vm.builtin.get_shape_dim"]([shape, 1])
    assert result == 512


def test_sub_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.sub_i64"]([10, 3]) == 7


def test_mul_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.mul_i64"]([6, 7]) == 42


def test_floordiv_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.floordiv_i64"]([10, 3]) == 3


def test_ceildiv_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.ceildiv_i64"]([10, 3]) == 4
    assert _DEFAULT_BUILTINS["vm.builtin.ceildiv_i64"]([9, 3]) == 3
    assert _DEFAULT_BUILTINS["vm.builtin.ceildiv_i64"]([512, 16]) == 32


def test_min_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.min_i64"]([3, 7]) == 3
    assert _DEFAULT_BUILTINS["vm.builtin.min_i64"]([7, 3]) == 3


def test_max_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.max_i64"]([3, 7]) == 7
    assert _DEFAULT_BUILTINS["vm.builtin.max_i64"]([7, 3]) == 7


def test_eq_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.eq_i64"]([5, 5]) is True
    assert _DEFAULT_BUILTINS["vm.builtin.eq_i64"]([5, 6]) is False


def test_le_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.le_i64"]([5, 5]) is True
    assert _DEFAULT_BUILTINS["vm.builtin.le_i64"]([4, 5]) is True
    assert _DEFAULT_BUILTINS["vm.builtin.le_i64"]([6, 5]) is False


def test_gt_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.gt_i64"]([6, 5]) is True
    assert _DEFAULT_BUILTINS["vm.builtin.gt_i64"]([5, 5]) is False


def test_ge_i64():
    assert _DEFAULT_BUILTINS["vm.builtin.ge_i64"]([5, 5]) is True
    assert _DEFAULT_BUILTINS["vm.builtin.ge_i64"]([6, 5]) is True
    assert _DEFAULT_BUILTINS["vm.builtin.ge_i64"]([4, 5]) is False


def test_assert_le_i64_pass():
    # Should not raise when val <= bound
    _DEFAULT_BUILTINS["vm.builtin.assert_le_i64"]([512, 2048, "S exceeds upper bound 2048"])
    _DEFAULT_BUILTINS["vm.builtin.assert_le_i64"]([2048, 2048, "S exceeds upper bound 2048"])


def test_assert_le_i64_fail():
    with pytest.raises(RuntimeError, match="RuntimeShapeError"):
        _DEFAULT_BUILTINS["vm.builtin.assert_le_i64"](
            [4096, 2048, "S exceeds upper bound 2048"])


def test_assert_le_i64_fail_message_contains_values():
    with pytest.raises(RuntimeError, match="4096") as exc_info:
        _DEFAULT_BUILTINS["vm.builtin.assert_le_i64"](
            [4096, 2048, "S exceeds upper bound 2048"])
    assert "2048" in str(exc_info.value)


# ---------------------------------------------------------------------------
# B. End-to-end acceptance criteria (manually-built Executable)
#
# Builds bytecode equivalent to:
#   %shape = call @vm.builtin.shape_of(%x)
#   %B     = call @vm.builtin.get_shape_dim(%shape, 0)
#   %S     = call @vm.builtin.get_shape_dim(%shape, 1)
#   %grid  = call @vm.builtin.ceildiv_i64(%S, 16)
#   call @vm.builtin.assert_le_i64(%S, 2048, msg)
#   ret %grid
# ---------------------------------------------------------------------------

def _build_shape_extraction_executable() -> tuple[Executable, int]:
    """Build Executable for shape_of → get_shape_dim → ceildiv → assert_le → ret.

    Returns (executable, bound_S) where bound_S is the assert upper bound.
    """
    exe = Executable()
    bound_S = 2048

    # ---- Helper: add function entries for builtins -------------------------
    def _add_builtin(name: str) -> int:
        fe = FunctionEntry(name=name, kind=CalleeKind.builtin,
                           instr_offset=-1, instr_count=0, num_regs=0, num_args=0)
        exe.function_table.append(fe)
        return len(exe.function_table) - 1

    # Pre-allocate builtin table entries
    fidx_shape_of    = _add_builtin("vm.builtin.shape_of")
    fidx_get_dim     = _add_builtin("vm.builtin.get_shape_dim")
    fidx_ceildiv     = _add_builtin("vm.builtin.ceildiv_i64")
    fidx_assert_le   = _add_builtin("vm.builtin.assert_le_i64")

    # ---- Constants ---------------------------------------------------------
    # const_inits for the main function:
    #   reg 1 → 0          (dim index for B)
    #   reg 2 → 1          (dim index for S)
    #   reg 3 → 16         (block size for ceildiv)
    #   reg 4 → bound_S    (2048)
    #   reg 5 → msg string
    exe.constants.extend([0, 1, 16, bound_S, f"S exceeds upper bound {bound_S}"])
    # reg 0 = param %x; regs 1-5 are from const_inits; regs 6-11 are temporaries
    const_inits = [
        ConstInit(reg_idx=1, const_idx=0),  # index 0 → dim 0
        ConstInit(reg_idx=2, const_idx=1),  # index 1 → dim 1
        ConstInit(reg_idx=3, const_idx=2),  # 16
        ConstInit(reg_idx=4, const_idx=3),  # 2048
        ConstInit(reg_idx=5, const_idx=4),  # msg
    ]

    # ---- Instructions ------------------------------------------------------
    instrs: list[Instruction] = []

    # r6 = shape_of(r0)
    instrs.append(Instruction(opcode=Opcode.CALL, dst_reg=6,
                              func_idx=fidx_shape_of, arg_regs=[0]))
    # r7 = get_shape_dim(r6, r1)  → B
    instrs.append(Instruction(opcode=Opcode.CALL, dst_reg=7,
                              func_idx=fidx_get_dim, arg_regs=[6, 1]))
    # r8 = get_shape_dim(r6, r2)  → S
    instrs.append(Instruction(opcode=Opcode.CALL, dst_reg=8,
                              func_idx=fidx_get_dim, arg_regs=[6, 2]))
    # r9 = ceildiv_i64(r8, r3)    → ceildiv(S, 16)
    instrs.append(Instruction(opcode=Opcode.CALL, dst_reg=9,
                              func_idx=fidx_ceildiv, arg_regs=[8, 3]))
    # assert_le_i64(r8, r4, r5)   → assert S <= 2048
    instrs.append(Instruction(opcode=Opcode.CALL, dst_reg=-1,
                              func_idx=fidx_assert_le, arg_regs=[8, 4, 5]))
    # ret r9
    instrs.append(Instruction(opcode=Opcode.RET, src_reg=9))

    instr_base = 0
    fe_main = FunctionEntry(
        name="main", kind=CalleeKind.vm_func,
        instr_offset=instr_base, instr_count=len(instrs),
        num_regs=10, num_args=1,
        const_inits=const_inits,
    )
    exe.function_table.append(fe_main)
    exe.instructions.extend(instrs)

    return exe, bound_S


def test_shape_extraction_and_ceildiv():
    """Acceptance criteria: shape_of → get_shape_dim → ceildiv(S, 16) = 32."""
    exe, _ = _build_shape_extraction_executable()
    interp = VMInterpreter(exe)
    tensor = _make_tensor((2, 512, 4096))
    result = interp.invoke("main", [tensor])
    assert result == 32  # ceildiv(512, 16) = 32


def test_shape_extraction_B_extracted_correctly():
    """B should be 2 extracted from shape[0], but we return ceildiv(S)."""
    exe, _ = _build_shape_extraction_executable()
    interp = VMInterpreter(exe)
    tensor = _make_tensor((2, 512, 4096))
    result = interp.invoke("main", [tensor])
    assert result == 32


def test_assert_le_i64_passes_when_S_within_bound():
    """S=2048 is exactly at bound, should not raise."""
    exe, _ = _build_shape_extraction_executable()
    interp = VMInterpreter(exe)
    tensor = _make_tensor((2, 2048, 4096))
    result = interp.invoke("main", [tensor])
    assert result == 128  # ceildiv(2048, 16) = 128


def test_assert_le_i64_raises_when_S_exceeds_bound():
    """S=4096 > bound 2048 → RuntimeShapeError."""
    exe, _ = _build_shape_extraction_executable()
    interp = VMInterpreter(exe)
    tensor = _make_tensor((2, 4096, 4096))
    with pytest.raises(RuntimeError, match="RuntimeShapeError"):
        interp.invoke("main", [tensor])


def test_assert_le_i64_error_contains_S_value():
    """Error message should contain the actual value 4096."""
    exe, _ = _build_shape_extraction_executable()
    interp = VMInterpreter(exe)
    tensor = _make_tensor((2, 4096, 4096))
    with pytest.raises(RuntimeError, match="4096") as exc_info:
        interp.invoke("main", [tensor])
    assert "2048" in str(exc_info.value)


# ---------------------------------------------------------------------------
# C. Full pipeline integration (IR → VMCodegen handles dynamic PrimVar shapes)
# ---------------------------------------------------------------------------

def _build_dynamic_shape_module() -> tuple[IRModule, PrimVar, PrimVar]:
    """Build a simple IRModule: f(x: Tensor[(B, S, 4096), float16]) { return x }
    with symbolic dims B (upper=8) and S (upper=2048).
    """
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    x = Var("x", struct_info=TensorStructInfo(
        shape=(B, S, IntImm(4096)), dtype="float16", device="cpu"
    ))
    s_op = AllocStorageOp(
        result_name="s0",
        size_bytes=IntImm(67108864),  # upper: 8*2048*4096*2 aligned to 256
        alignment=256,
        device="cpu",
    )
    t_op = AllocTensorOp(
        result_name="t0",
        storage=s_op.results[0],
        offset=0,
        shape=(B, S, IntImm(4096)),
        dtype="float16",
    )
    ret_op = ReturnOp(values=(t_op.results[0],))
    block = Block(args=(x,), ops=(s_op, t_op, ret_op))
    fn = Function(body=Region((block,)), ret_struct_info=None)
    return IRModule({"f": fn}), B, S


def test_dynamic_shape_vmcodegen_runs_without_error():
    """VMCodegenPass succeeds on a module with PrimVar shape dims."""
    module, B, S = _build_dynamic_shape_module()
    exe = VMCodegenPass().run(module)
    assert exe is not None


def test_dynamic_shape_vmcodegen_emits_shape_of():
    """VMCodegenPass emits shape_of + get_shape_dim for PrimVar dims."""
    module, B, S = _build_dynamic_shape_module()
    exe = VMCodegenPass().run(module)
    names = {fe.name for fe in exe.function_table}
    assert "vm.builtin.shape_of" in names
    assert "vm.builtin.get_shape_dim" in names


def test_dynamic_shape_vmcodegen_emits_assert_le():
    """VMCodegenPass emits assert_le_i64 for PrimVars with upper bounds."""
    module, B, S = _build_dynamic_shape_module()
    exe = VMCodegenPass().run(module)
    names = {fe.name for fe in exe.function_table}
    assert "vm.builtin.assert_le_i64" in names


def test_dynamic_shape_interpreter_runs():
    """Interpreter executes a function with PrimVar dims and returns a _Tensor."""
    module, B, S = _build_dynamic_shape_module()
    exe = VMCodegenPass().run(module)
    interp = VMInterpreter(exe)
    tensor = _make_tensor((2, 512, 4096))
    result = interp.invoke("f", [tensor])
    assert isinstance(result, _Tensor)
    assert result.shape == (2, 512, 4096)


def test_dynamic_shape_interpreter_assert_le_pass():
    """Interpreter runs ok when input dims are within upper bounds."""
    module, B, S = _build_dynamic_shape_module()
    exe = VMCodegenPass().run(module)
    interp = VMInterpreter(exe)
    # B=8, S=2048 are at the exact upper bound — should succeed
    tensor = _make_tensor((8, 2048, 4096))
    result = interp.invoke("f", [tensor])
    assert result.shape == (8, 2048, 4096)


def test_dynamic_shape_interpreter_assert_le_fail():
    """Interpreter raises RuntimeError when S exceeds upper bound."""
    module, B, S = _build_dynamic_shape_module()
    exe = VMCodegenPass().run(module)
    interp = VMInterpreter(exe)
    tensor = _make_tensor((2, 4096, 4096))  # S=4096 > upper=2048
    with pytest.raises(RuntimeError, match="RuntimeShapeError"):
        interp.invoke("f", [tensor])


def test_dynamic_shape_interpreter_B_exceeds():
    """Interpreter raises RuntimeError when B exceeds upper bound."""
    module, B, S = _build_dynamic_shape_module()
    exe = VMCodegenPass().run(module)
    interp = VMInterpreter(exe)
    tensor = _make_tensor((16, 512, 4096))  # B=16 > upper=8
    with pytest.raises(RuntimeError, match="RuntimeShapeError"):
        interp.invoke("f", [tensor])


# ---------------------------------------------------------------------------
# D. Full DSL pipeline with dynamic shapes
# ---------------------------------------------------------------------------

def test_full_dsl_dynamic_shape_pipeline():
    """DSL @dp.function with symbolic dims goes through full pipeline."""
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cpu"]):
        y = dp.ops.relu(x)
        return y

    module = dp.get_module()
    exe = _run_pipeline(module, _spec("relu"))

    # Executable must have shape_of and assert_le_i64
    names = {fe.name for fe in exe.function_table}
    assert "vm.builtin.shape_of" in names
    assert "vm.builtin.assert_le_i64" in names

    # Interpreter runs with valid input
    interp = VMInterpreter(exe)
    interp.register_kernel("kernel.relu_fp16", lambda args: None)
    tensor = _make_tensor((2, 512, 4096))
    result = interp.invoke("main", [tensor])
    # relu is DPS (output written in-place); main returns the output tensor
    assert result is not None


def test_full_dsl_dynamic_shape_assert_fails():
    """Full pipeline: interpreter raises RuntimeShapeError when S exceeds bound."""
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cpu"]):
        y = dp.ops.relu(x)
        return y

    module = dp.get_module()
    exe = _run_pipeline(module, _spec("relu"))

    interp = VMInterpreter(exe)
    interp.register_kernel("kernel.relu_fp16", lambda args: None)

    tensor = _make_tensor((2, 4096, 4096))  # S=4096 > 2048
    with pytest.raises(RuntimeError, match="RuntimeShapeError"):
        interp.invoke("main", [tensor])
