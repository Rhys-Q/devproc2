import pytest

from devproc2.ir import (
    Add,
    Block,
    Call,
    CallDPS,
    CalleeKind,
    CeilDiv,
    Constant,
    FloorDiv,
    Function,
    GE,
    GT,
    IRModule,
    IRVerificationError,
    IntImm,
    LE,
    LT,
    Max,
    Min,
    Mul,
    OpaqueEffect,
    Printer,
    PrimVar,
    PureEffect,
    ReadOnlyEffect,
    Return,
    Sub,
    TensorCreateKind,
    TensorCreateOp,
    TensorStructInfo,
    TupleExpr,
    TupleGetItem,
    Var,
    WriteEffect,
    ceildiv,
    pmax,
    pmin,
    print_module,
    verify,
)


# ---------------------------------------------------------------------------
# PrimExpr unit tests
# ---------------------------------------------------------------------------

def test_intImm_prints():
    p = Printer()
    assert p.print_prim_expr(IntImm(4096)) == "4096"


def test_primvar_prints():
    p = Printer()
    B = PrimVar("B", upper=8)
    assert p.print_prim_expr(B) == "B"


def test_operator_overloads_build_correct_nodes():
    B = PrimVar("B")
    S = PrimVar("S")

    assert B + S == Add(B, S)
    assert B - 1 == Sub(B, IntImm(1))
    assert 2 * S == Mul(IntImm(2), S)
    assert B // 4 == FloorDiv(B, IntImm(4))
    assert (B < S) == LT(B, S)
    assert (B <= 8) == LE(B, IntImm(8))
    assert (S > 0) == GT(S, IntImm(0))
    assert (S >= 1) == GE(S, IntImm(1))
    assert B.eq(S) is not None  # EQ node created


def test_free_function_helpers():
    S = PrimVar("S")
    assert ceildiv(S, 16) == CeilDiv(S, IntImm(16))
    assert pmin(S, 128) == Min(S, IntImm(128))
    assert pmax(S, 1) == Max(S, IntImm(1))


def test_prim_expr_printer_arithmetic():
    S = PrimVar("S", upper=2048)
    p = Printer()
    assert p.print_prim_expr(ceildiv(S, 16)) == "ceildiv(S, 16)"
    assert p.print_prim_expr(S * 4) == "(S * 4)"
    assert p.print_prim_expr(S + 1) == "(S + 1)"
    assert p.print_prim_expr(S - 1) == "(S - 1)"
    assert p.print_prim_expr(S // 2) == "(S // 2)"
    assert p.print_prim_expr(pmin(S, 64)) == "min(S, 64)"
    assert p.print_prim_expr(pmax(S, 1)) == "max(S, 1)"
    assert p.print_prim_expr(S <= 2048) == "(S <= 2048)"


def test_tensor_struct_info_int_coercion():
    """Plain ints in shape are auto-coerced to IntImm."""
    B = PrimVar("B", upper=8)
    si = TensorStructInfo((B, 4096), "float16", "cuda")
    assert si.shape == (B, IntImm(4096))


# ---------------------------------------------------------------------------
# Printer tests
# ---------------------------------------------------------------------------

def test_print_basic_function():
    B = PrimVar("B")
    S = PrimVar("S")
    x_si = TensorStructInfo((B, S, 4096), "float16", "cuda")
    w_si = TensorStructInfo((4096, 4096), "float16", "cuda")
    out_si = TensorStructInfo((B, S, 4096), "float16", "cuda")

    x = Var("x", x_si)
    w = Var("w", w_si)
    y = Var("y")
    z = Var("z")

    block = Block(
        bindings=(
            (y, Call("@matmul", (x, w))),
            (z, Call("@silu", (y,))),
        ),
        body=Return(z),
    )
    fn = Function(params=(x, w), body=block, ret_struct_info=out_si)
    module = IRModule({"main": fn})

    text = print_module(module)
    assert "@main(" in text
    assert "%x: Tensor[(B, S, 4096), float16, cuda]" in text
    assert "%y = @matmul(%x, %w)" in text
    assert "%z = @silu(%y)" in text
    assert "return %z" in text
    assert " -> Tensor[(B, S, 4096), float16, cuda]" in text


def test_print_calldps_no_output():
    k_cache = Var("k_cache")
    v_cache = Var("v_cache")
    k = Var("k")
    v = Var("v")
    pos = Var("pos")

    calldps = CallDPS(
        callee="@kernel.update_kvcache",
        inputs=(k_cache, v_cache, k, v, pos),
        output=None,
        effect=WriteEffect((k_cache, v_cache)),
        callee_kind=CalleeKind.kernel,
    )

    block = Block(bindings=((None, calldps),), body=Return(pos))
    fn = Function(params=(k_cache, v_cache, k, v, pos), body=block)
    module = IRModule({"update": fn})

    text = print_module(module)
    assert "call_dps @kernel.update_kvcache(" in text
    assert "inputs=[%k_cache, %v_cache, %k, %v, %pos]" in text
    assert "output=None" in text
    assert "callee_kind=kernel" in text
    assert "effect=write(%k_cache, %v_cache)" in text


def test_print_calldps_with_output():
    x = Var("x")
    out = Var("out")
    calldps = CallDPS(
        callee="@kernel.relu",
        inputs=(x,),
        output=out,
        effect=WriteEffect((out,)),
        callee_kind=CalleeKind.kernel,
    )
    block = Block(bindings=((out, calldps),), body=Return(out))
    fn = Function(params=(x,), body=block)
    text = print_module(IRModule({"f": fn}))
    assert "call_dps @kernel.relu(" in text
    assert "output=%out" in text
    assert "effect=write(%out)" in text


def test_print_multi_function_separator():
    x = Var("x")
    block = Block(bindings=((Var("y"), Call("@relu", (x,))),), body=Return(Var("y")))
    fn = Function(params=(x,), body=block)
    module = IRModule({"f1": fn, "f2": fn})
    assert "\n\n" in print_module(module)


def test_printer_reuse():
    x = Var("x")
    block = Block(bindings=(), body=Return(x))
    fn = Function(params=(x,), body=block)
    p = Printer()
    t1 = p.print_module(IRModule({"a": fn}))
    t2 = p.print_module(IRModule({"b": fn}))
    assert "@a" in t1 and "@b" not in t1
    assert "@b" in t2 and "@a" not in t2


def test_tuple_ir():
    x = Var("x")
    qkv = Var("qkv")
    q = Var("q")
    block = Block(
        bindings=(
            (qkv, Call("@qkv_proj", (x,))),
            (q, TupleGetItem(qkv, 0)),
        ),
        body=Return(q),
    )
    fn = Function(params=(x,), body=block)
    text = print_module(IRModule({"f": fn}))
    assert "%qkv = @qkv_proj(%x)" in text
    assert "%q = %qkv[0]" in text


def test_tensor_create_op_printer():
    B = PrimVar("B", upper=8)
    buf = Var("buf")
    block = Block(
        bindings=(
            (buf, TensorCreateOp(
                kind=TensorCreateKind.empty,
                shape=(B, 4096),
                dtype="float16",
                device="cuda",
            )),
        ),
        body=Return(buf),
    )
    fn = Function(params=(), body=block)
    text = print_module(IRModule({"f": fn}))
    assert "dp.empty" in text
    assert "B" in text
    assert "4096" in text


# ---------------------------------------------------------------------------
# Verifier tests
# ---------------------------------------------------------------------------

def test_verifier_rejects_alloc_storage():
    x = Var("x")
    y = Var("y")
    block = Block(bindings=((y, Call("@alloc_storage", (x,))),), body=Return(y))
    fn = Function(params=(x,), body=block)
    with pytest.raises(IRVerificationError, match="alloc_storage"):
        verify(IRModule({"bad": fn}))


def test_verifier_rejects_alloc_tensor():
    x = Var("x")
    y = Var("y")
    block = Block(bindings=((y, Call("@alloc_tensor", (x,))),), body=Return(y))
    fn = Function(params=(x,), body=block)
    with pytest.raises(IRVerificationError, match="alloc_tensor"):
        verify(IRModule({"bad": fn}))


def test_verifier_rejects_nested_forbidden():
    x = Var("x")
    y = Var("y")
    z = Var("z")
    block = Block(
        bindings=(
            (y, Call("@alloc_storage", (x,))),
            (z, TupleExpr((y,))),
        ),
        body=Return(z),
    )
    fn = Function(params=(x,), body=block)
    with pytest.raises(IRVerificationError, match="alloc_storage"):
        verify(IRModule({"bad": fn}))


def test_verifier_catches_use_before_def():
    x = Var("x")
    y = Var("y")
    z = Var("z")
    block = Block(bindings=((y, Call("@foo", (z,))),), body=Return(y))
    fn = Function(params=(x,), body=block)
    with pytest.raises(IRVerificationError, match="used before definition"):
        verify(IRModule({"f": fn}))


def test_verifier_catches_double_def():
    x = Var("x")
    y = Var("y")
    block = Block(
        bindings=(
            (y, Call("@foo", (x,))),
            (y, Call("@bar", (x,))),
        ),
        body=Return(y),
    )
    fn = Function(params=(x,), body=block)
    with pytest.raises(IRVerificationError, match="defined more than once"):
        verify(IRModule({"f": fn}))


def test_verifier_write_effect_not_false_positive():
    """WriteEffect.vars are effect metadata — must not trigger use-before-def."""
    k_cache = Var("k_cache")
    v_cache = Var("v_cache")
    k = Var("k")
    v = Var("v")
    pos = Var("pos")
    calldps = CallDPS(
        callee="@kernel.update_kvcache",
        inputs=(k_cache, v_cache, k, v, pos),
        output=None,
        effect=WriteEffect((k_cache, v_cache)),
        callee_kind=CalleeKind.kernel,
    )
    block = Block(bindings=((None, calldps),), body=Return(pos))
    fn = Function(params=(k_cache, v_cache, k, v, pos), body=block)
    verify(IRModule({"f": fn}))  # must not raise


def test_verifier_block_body_must_be_return():
    x = Var("x")
    block = Block(bindings=(), body=x)
    fn = Function(params=(x,), body=block)
    with pytest.raises(IRVerificationError, match="block body must be Return"):
        verify(IRModule({"f": fn}))


def test_verifier_accepts_valid_module():
    x = Var("x", TensorStructInfo((128,), "float16", "cuda"))
    y = Var("y")
    block = Block(bindings=((y, Call("@relu", (x,))),), body=Return(y))
    fn = Function(params=(x,), body=block)
    verify(IRModule({"f": fn}))


# ---------------------------------------------------------------------------
# TensorCreateOp validation tests
# ---------------------------------------------------------------------------

def test_tensor_create_op_empty_like_validation():
    x = Var("x")
    with pytest.raises(ValueError, match="requires 'like'"):
        TensorCreateOp(kind=TensorCreateKind.empty_like, shape=(), dtype="float16", device="cuda")

    B = PrimVar("B")
    with pytest.raises(ValueError, match="must not specify 'shape'"):
        TensorCreateOp(
            kind=TensorCreateKind.empty_like, shape=(B,), dtype="float16", device="cuda", like=x
        )

    with pytest.raises(ValueError, match="must not specify 'like'"):
        TensorCreateOp(
            kind=TensorCreateKind.empty, shape=(IntImm(128),), dtype="float16", device="cuda", like=x
        )

    op = TensorCreateOp(
        kind=TensorCreateKind.empty_like, shape=(), dtype="float16", device="cuda", like=x
    )
    assert op.like is x
