import pytest

from devproc2.compiler.op import get_op
from devproc2.compiler.op.registry import register
from devproc2.compiler.op.schema import LoweringPolicy, OpDef
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.ir import (
    AttrDict,
    Block,
    CallOp,
    DialectKind,
    ExternalFuncRef,
    Function,
    IRModule,
    IRVerificationError,
    Region,
    ReturnOp,
    StandardOpRef,
    TensorStructInfo,
    Var,
    print_module,
    verify,
)
from devproc2.ir.ops import CallDPSOp
from devproc2.ir.prim_expr import IntImm
from devproc2.kernel.registry import KernelRegistry, KernelSpec


def test_callop_resolves_registered_op_without_forcing_default_attrs():
    x = Var("x", TensorStructInfo((2, 4), "float16", "cuda"))
    op = CallOp(StandardOpRef("gelu"), (x,), result_name="y")

    assert op.op_def is get_op("gelu")
    assert op.op_ref.name == "gelu"
    assert op.attrs == {}


def test_standard_op_ref_inherits_registered_dialect():
    name = "test_shape_marker"
    op_def = get_op(name)
    if op_def is None:
        op_def = register(
            OpDef(
                name=name,
                inputs=(),
                attrs=(),
                outputs=(),
                infer=lambda ctx: None,
                dialect=DialectKind.shape,
                lowering=LoweringPolicy.none(),
            )
        )

    ref = StandardOpRef(name, op_def)
    call = CallOp(StandardOpRef(name), (), result_name="")

    assert ref.dialect is DialectKind.shape
    assert call.dialect is DialectKind.shape


def test_registered_op_rejects_unknown_or_wrong_attrs():
    x = Var("x", TensorStructInfo((2, 4), "float16", "cuda"))

    with pytest.raises(ValueError, match="unknown attrs"):
        CallOp(StandardOpRef("gelu"), (x,), result_name="y", attrs={"bad": 1})

    with pytest.raises(TypeError, match="expects string"):
        CallOp(StandardOpRef("gelu"), (x,), result_name="y", attrs={"approximate": 1})


def test_attrdict_json_roundtrip_is_stable():
    attrs = get_op("gelu").normalize_attrs({"approximate": "none"})
    payload = attrs.to_json_obj()

    assert payload == {"approximate": "none"}
    assert AttrDict.from_python(payload) == attrs


def test_verifier_rejects_unknown_standard_call_but_allows_external_call():
    x = Var("x", TensorStructInfo((2,), "float16", "cuda"))
    standard = CallOp(StandardOpRef("custom_runtime"), (x,), result_name="y")
    y = standard.results[0]
    bad = IRModule({"f": Function(Region((Block((x,), (standard, ReturnOp((y,)))),)))})

    with pytest.raises(IRVerificationError, match="unknown standard op"):
        verify(bad)

    external = CallOp(
        ExternalFuncRef("custom_runtime"),
        (x,),
        result_name="y",
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


def test_reshape_infer_and_shape_attr_printing_are_stable():
    x_info = TensorStructInfo((IntImm(2), IntImm(3), IntImm(4)), "float16", "cuda")

    assert get_op("reshape").infer_struct_info(
        (x_info,),
        {"shape": (IntImm(6), IntImm(4))},
    ) == TensorStructInfo((IntImm(6), IntImm(4)), "float16", "cuda")

    with pytest.raises(ValueError, match="element count mismatch"):
        get_op("reshape").infer_struct_info((x_info,), {"shape": (5, 5)})

    attrs = get_op("reshape").normalize_attrs({"shape": (6, 4)})
    assert attrs.to_json_obj() == {"shape": [6, 4]}

    x = Var("x", x_info)
    op = CallOp(StandardOpRef("reshape"), (x,), result_name="y", attrs={"shape": (6, 4)})
    y = op.results[0]
    module = IRModule({"f": Function(Region((Block((x,), (op, ReturnOp((y,)))),)))})
    assert "@reshape(%x) {shape=[6, 4]}" in print_module(module)

    with pytest.raises(TypeError, match="expects shape"):
        CallOp(StandardOpRef("reshape"), (x,), result_name="bad", attrs={"shape": ("bad",)})


def test_cat_infer_checks_axis_shape_dtype_and_device():
    lhs = TensorStructInfo((IntImm(2), IntImm(3), IntImm(4)), "float16", "cuda")
    rhs = TensorStructInfo((IntImm(2), IntImm(5), IntImm(4)), "float16", "cuda")

    assert get_op("cat").infer_struct_info((lhs, rhs), {"axis": 1}) == TensorStructInfo(
        (IntImm(2), IntImm(8), IntImm(4)),
        "float16",
        "cuda",
    )
    assert get_op("cat").infer_struct_info((lhs, rhs), {"axis": -2}) == TensorStructInfo(
        (IntImm(2), IntImm(8), IntImm(4)),
        "float16",
        "cuda",
    )

    bad_shape = TensorStructInfo((IntImm(2), IntImm(5), IntImm(7)), "float16", "cuda")
    with pytest.raises(ValueError, match="non-axis dimension"):
        get_op("cat").infer_struct_info((lhs, bad_shape), {"axis": 1})

    bad_dtype = TensorStructInfo((IntImm(2), IntImm(5), IntImm(4)), "float32", "cuda")
    with pytest.raises(ValueError, match="dtype mismatch"):
        get_op("cat").infer_struct_info((lhs, bad_dtype), {"axis": 1})

    bad_device = TensorStructInfo((IntImm(2), IntImm(5), IntImm(4)), "float16", "cpu")
    with pytest.raises(ValueError, match="device mismatch"):
        get_op("cat").infer_struct_info((lhs, bad_device), {"axis": 1})


def test_matmul_transpose_attrs_affect_infer():
    lhs = TensorStructInfo((IntImm(2), IntImm(5), IntImm(3)), "float16", "cuda")
    rhs = TensorStructInfo((IntImm(2), IntImm(4), IntImm(3)), "float16", "cuda")
    assert get_op("matmul").infer_struct_info(
        (lhs, rhs),
        {"transpose_b": True},
    ) == TensorStructInfo((IntImm(2), IntImm(5), IntImm(4)), "float16", "cuda")

    lhs_t = TensorStructInfo((IntImm(2), IntImm(3), IntImm(5)), "float16", "cuda")
    rhs_t = TensorStructInfo((IntImm(2), IntImm(3), IntImm(4)), "float16", "cuda")
    assert get_op("matmul").infer_struct_info(
        (lhs_t, rhs_t),
        {"transpose_a": True},
    ) == TensorStructInfo((IntImm(2), IntImm(5), IntImm(4)), "float16", "cuda")

    x = Var("x", lhs)
    y = Var("y", rhs)
    with pytest.raises(TypeError, match="expects bool"):
        CallOp(StandardOpRef("matmul"), (x, y), result_name="z", attrs={"transpose_b": 1})


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
    gelu = CallOp(StandardOpRef("gelu"), (x,), result_name="y", attrs={"approximate": "none"})
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
