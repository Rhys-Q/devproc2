import pytest

from devproc2.compiler.op import get_op
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.ir import (
    Block,
    CallKind,
    CallOp,
    Function,
    IRModule,
    IRVerificationError,
    Region,
    ReturnOp,
    TensorStructInfo,
    Var,
    verify,
)
from devproc2.ir.ops import CallDPSOp
from devproc2.ir.prim_expr import IntImm
from devproc2.kernel.registry import KernelRegistry, KernelSpec


def test_callop_resolves_registered_op_without_forcing_default_attrs():
    x = Var("x", TensorStructInfo((2, 4), "float16", "cuda"))
    op = CallOp("@gelu", (x,), result_name="y")

    assert op.op is get_op("gelu")
    assert op.op_name == "gelu"
    assert op.attrs == {}


def test_registered_op_rejects_unknown_or_wrong_attrs():
    x = Var("x", TensorStructInfo((2, 4), "float16", "cuda"))

    with pytest.raises(ValueError, match="unknown attrs"):
        CallOp("@gelu", (x,), result_name="y", attrs={"bad": 1})

    with pytest.raises(TypeError, match="expects str"):
        CallOp("@gelu", (x,), result_name="y", attrs={"approximate": 1})


def test_verifier_rejects_unknown_standard_call_but_allows_external_call():
    x = Var("x", TensorStructInfo((2,), "float16", "cuda"))
    standard = CallOp("@custom_runtime", (x,), result_name="y")
    y = standard.results[0]
    bad = IRModule({"f": Function(Region((Block((x,), (standard, ReturnOp((y,)))),)))})

    with pytest.raises(IRVerificationError, match="unknown standard op"):
        verify(bad)

    external = CallOp(
        "@custom_runtime",
        (x,),
        result_name="y",
        call_kind=CallKind.external,
    )
    y = external.results[0]
    ok = IRModule({"f": Function(Region((Block((x,), (external, ReturnOp((y,)))),)))})
    verify(ok)


def test_broadcast_and_matmul_infer_use_registered_rules():
    a = TensorStructInfo((IntImm(2), IntImm(3), IntImm(4)), "float16", "cuda")
    b = TensorStructInfo((IntImm(4),), "float16", "cuda")
    assert get_op("add").infer_struct_info((a, b), {}) == TensorStructInfo(
        (IntImm(2), IntImm(3), IntImm(4)),
        "float16",
        "cuda",
    )

    lhs = TensorStructInfo((IntImm(2), IntImm(3)), "float16", "cuda")
    rhs = TensorStructInfo((IntImm(4), IntImm(5)), "float16", "cuda")
    with pytest.raises(ValueError, match="reduction dims"):
        get_op("matmul").infer_struct_info((lhs, rhs), {})


def test_permute_dims_uses_axes_attr_like_relax():
    x = TensorStructInfo((IntImm(2), IntImm(3), IntImm(4)), "float16", "cuda")

    assert get_op("permute_dims").infer_struct_info(
        (x,),
        {"axes": (2, 0, 1)},
    ) == TensorStructInfo((IntImm(4), IntImm(2), IntImm(3)), "float16", "cuda")

    assert get_op("permute_dims").infer_struct_info(
        (x,),
        {"axes": None},
    ) == TensorStructInfo((IntImm(4), IntImm(3), IntImm(2)), "float16", "cuda")

    with pytest.raises(ValueError, match="permutation"):
        get_op("permute_dims").infer_struct_info((x,), {"axes": (0, 0, 1)})


def test_norm_ops_use_axes_and_epsilon_attrs():
    layer_norm = get_op("layer_norm")
    rms_norm = get_op("rms_norm")

    assert [attr.name for attr in layer_norm.attrs] == [
        "axes",
        "epsilon",
        "center",
        "scale",
    ]
    assert [attr.name for attr in rms_norm.attrs] == ["axes", "epsilon"]
    assert "use_adarms" not in {attr.name for attr in rms_norm.attrs}


def test_dps_lowering_preserves_normalized_standard_op_attrs():
    x = Var("x", TensorStructInfo((2, 4), "float16", "cuda"))
    gelu = CallOp("@gelu", (x,), result_name="y", attrs={"approximate": "none"})
    y = gelu.results[0]
    module = IRModule({"f": Function(Region((Block((x,), (gelu, ReturnOp((y,)))),)))})
    module = InferStructInfoPass().run(module)

    registry = KernelRegistry()
    registry.register(
        KernelSpec(
            op_name="gelu",
            device="cuda",
            input_dtypes=("float16",),
            kernel_name="kernel.gelu_fp16",
        )
    )
    lowered = DPSLoweringPass(registry).run(module)
    dps = next(
        op
        for op in lowered.functions["f"].body.entry_block.ops
        if isinstance(op, CallDPSOp)
    )

    assert dps.attrs == {"approximate": "none"}
