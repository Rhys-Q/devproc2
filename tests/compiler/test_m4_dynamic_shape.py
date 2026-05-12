"""M4 Dynamic Shape MVP tests."""
import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.shape_assertion_insert import ShapeAssertionInsertPass
from devproc2.compiler.passes.shape_constraint_verify import (
    RuntimeShapeError,
    ShapeConstraintVerifyPass,
)
from devproc2.ir import (
    IRModule,
    TensorStructInfo,
    Var,
    print_module,
    verify,
)
from devproc2.ir.nodes import IntImm
from devproc2.ir.ops import CallOp, ShapeAssertOp
from devproc2.ir.prim_expr import PrimVar


@pytest.fixture(autouse=True)
def reset():
    dp.reset_module()
    yield
    dp.reset_module()


# ---------------------------------------------------------------------------
# symbolic_dim / Tensor annotation primitives
# ---------------------------------------------------------------------------

def test_symbolic_dim_returns_primvar():
    B = dp.symbolic_dim("B", upper=8)
    assert isinstance(B, PrimVar)
    assert B.name == "B"
    assert B.upper == 8


def test_symbolic_dim_no_upper():
    N = dp.symbolic_dim("N")
    assert isinstance(N, PrimVar)
    assert N.upper is None


def test_tensor_annotation_two_args():
    B = dp.symbolic_dim("B", upper=8)
    si = dp.Tensor[(B, 4096), "float16"]
    assert isinstance(si, TensorStructInfo)
    assert si.shape[0] is B
    assert si.shape[1] == IntImm(4096)
    assert si.dtype == "float16"
    assert si.device == "cuda"


def test_tensor_annotation_three_args():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)
    si = dp.Tensor[(B, S, 4096), "float16", "cuda"]
    assert isinstance(si, TensorStructInfo)
    assert si.shape[0] is B
    assert si.shape[1] is S
    assert si.shape[2] == IntImm(4096)
    assert si.device == "cuda"


def test_tensor_annotation_rejects_wrong_arity():
    B = dp.symbolic_dim("B", upper=8)
    with pytest.raises(TypeError):
        dp.Tensor[(B, 4096), "float16", "cuda", "extra"]  # 4 args


# ---------------------------------------------------------------------------
# @dp.function with type annotations
# ---------------------------------------------------------------------------

def test_function_typed_params_struct_info():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    fn = dp.get_module().functions["main"]
    x_param = fn.params[0]
    assert isinstance(x_param.struct_info, TensorStructInfo)
    assert x_param.struct_info.shape[0] is B
    assert x_param.struct_info.shape[1] is S


def test_function_unannotated_params_unchanged():
    @dp.function
    def f(x, y):
        z = dp.ops.relu(x)
        return z

    fn = dp.get_module().functions["f"]
    assert fn.params[0].struct_info is None
    assert fn.params[1].struct_info is None


def test_function_typed_params_verifier_passes():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    verify(dp.get_module())


# ---------------------------------------------------------------------------
# InferStructInfoPass
# ---------------------------------------------------------------------------

def test_infer_struct_info_propagates_through_call():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    fn = module.functions["main"]
    call_op = fn.body.entry_block.ops[0]
    assert isinstance(call_op, CallOp)
    assert call_op.results[0].struct_info is not None
    assert call_op.results[0].struct_info == TensorStructInfo((B, S, IntImm(4096)), "float16", "cuda")


def test_infer_struct_info_no_annotation_unchanged():
    @dp.function
    def f(x):
        y = dp.ops.relu(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    fn = module.functions["f"]
    call_op = fn.body.entry_block.ops[0]
    assert call_op.results[0].struct_info is None


def test_infer_struct_info_verifier_still_passes():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 4096), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    verify(module)


# ---------------------------------------------------------------------------
# ShapeAssertionInsertPass
# ---------------------------------------------------------------------------

def test_shape_assertion_insert_ops_prepended():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    module = ShapeAssertionInsertPass().run(dp.get_module())
    fn = module.functions["main"]
    ops = fn.body.entry_block.ops

    # First two ops are ShapeAssertOps for B and S
    assert isinstance(ops[0], ShapeAssertOp)
    assert isinstance(ops[1], ShapeAssertOp)

    assert ops[0].tensor.name == "x"
    assert ops[0].dim_idx == 0
    assert ops[0].upper == 8

    assert ops[1].tensor.name == "x"
    assert ops[1].dim_idx == 1
    assert ops[1].upper == 2048


def test_shape_assertion_insert_no_upper_no_assert():
    N = dp.symbolic_dim("N")  # no upper

    @dp.function
    def f(x: dp.Tensor[(N, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = ShapeAssertionInsertPass().run(dp.get_module())
    fn = module.functions["f"]
    assert not any(isinstance(op, ShapeAssertOp) for op in fn.body.entry_block.ops)


def test_shape_assertion_insert_printed_ir():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    text = print_module(ShapeAssertionInsertPass().run(dp.get_module()))
    assert "assert %x.shape[0] <= 8" in text
    assert "assert %x.shape[1] <= 2048" in text


def test_shape_assertion_insert_verifier_passes():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    verify(ShapeAssertionInsertPass().run(dp.get_module()))


def test_shape_assertion_insert_no_annotated_param_unchanged():
    @dp.function
    def f(x, y):
        z = dp.ops.relu(x)
        return z

    before_ops = tuple(dp.get_module().functions["f"].body.entry_block.ops)
    module = ShapeAssertionInsertPass().run(dp.get_module())
    after_ops = tuple(module.functions["f"].body.entry_block.ops)
    assert before_ops == after_ops


# ---------------------------------------------------------------------------
# ShapeConstraintVerifyPass
# ---------------------------------------------------------------------------

def _make_asserted_module():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    return ShapeAssertionInsertPass().run(dp.get_module())


def test_shape_constraint_verify_valid_binding():
    module = _make_asserted_module()
    ShapeConstraintVerifyPass().run(module, bindings={"B": 4, "S": 512})


def test_shape_constraint_verify_at_upper_bound():
    module = _make_asserted_module()
    ShapeConstraintVerifyPass().run(module, bindings={"B": 8, "S": 2048})  # exact bound ok


def test_shape_constraint_verify_violation_S():
    module = _make_asserted_module()
    with pytest.raises(RuntimeShapeError, match="S"):
        ShapeConstraintVerifyPass().run(module, bindings={"B": 4, "S": 4096})


def test_shape_constraint_verify_violation_B():
    module = _make_asserted_module()
    with pytest.raises(RuntimeShapeError, match="B"):
        ShapeConstraintVerifyPass().run(module, bindings={"B": 16, "S": 512})


def test_shape_constraint_verify_violation_B_at_boundary():
    """B=9 is exactly one over the upper bound of 8; must raise."""
    module = _make_asserted_module()
    with pytest.raises(RuntimeShapeError, match="B"):
        ShapeConstraintVerifyPass().run(module, bindings={"B": 9, "S": 512})


def test_shape_constraint_verify_no_bindings_noop():
    module = _make_asserted_module()
    result = ShapeConstraintVerifyPass().run(module)
    assert result is module


def test_shape_constraint_verify_returns_module():
    module = _make_asserted_module()
    result = ShapeConstraintVerifyPass().run(module, bindings={"B": 1, "S": 1})
    assert result is module


# ---------------------------------------------------------------------------
# Full pipeline: InferStructInfo → ShapeAssertionInsert → Verify
# ---------------------------------------------------------------------------

def test_full_m4_pipeline():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    module = dp.get_module()
    module = InferStructInfoPass().run(module)
    module = ShapeAssertionInsertPass().run(module)

    verify(module)

    text = print_module(module)
    assert "assert %x.shape[0] <= 8" in text
    assert "assert %x.shape[1] <= 2048" in text

    fn = module.functions["main"]
    call_op = next(op for op in fn.body.entry_block.ops if isinstance(op, CallOp))
    assert call_op.results[0].struct_info is not None

    # Valid shape passes
    ShapeConstraintVerifyPass().run(module, bindings={"B": 8, "S": 2048})

    # Exceeding S raises RuntimeShapeError
    with pytest.raises(RuntimeShapeError, match="S"):
        ShapeConstraintVerifyPass().run(module, bindings={"B": 4, "S": 4096})


# ---------------------------------------------------------------------------
# M2/M3 regression
# ---------------------------------------------------------------------------

def test_m3_regression_no_annotations():
    @dp.function
    def decode_step(x, flag, n):
        if flag:
            y = dp.ops.relu(x)
        else:
            y = dp.ops.silu(x)
        for i in dp.range(0, n):
            y = dp.ops.layernorm(y)
        return y

    verify(dp.get_module())
