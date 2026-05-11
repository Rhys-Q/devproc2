import pytest

from devproc2.ir import (
    BinOpDim,
    Block,
    Call,
    CallDPS,
    CalleeKind,
    ConstDim,
    Constant,
    Function,
    IRModule,
    IRVerificationError,
    OpaqueEffect,
    Printer,
    PureEffect,
    ReadOnlyEffect,
    Return,
    SymDimRef,
    SymbolicDim,
    TensorCreateKind,
    TensorCreateOp,
    TensorStructInfo,
    TupleExpr,
    TupleGetItem,
    Var,
    WriteEffect,
    print_module,
    verify,
)


def test_print_basic_function():
    B = SymbolicDim("B")
    S = SymbolicDim("S")
    x_si = TensorStructInfo((SymDimRef(B), SymDimRef(S), ConstDim(4096)), "float16", "cuda")
    w_si = TensorStructInfo((ConstDim(4096), ConstDim(4096)), "float16", "cuda")
    out_si = TensorStructInfo((SymDimRef(B), SymDimRef(S), ConstDim(4096)), "float16", "cuda")

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

    block = Block(bindings=((None, calldps),), body=Return(Constant(None)))
    fn = Function(params=(k_cache, v_cache, k, v, pos), body=block)
    module = IRModule({"update": fn})

    text = print_module(module)
    assert "call_dps @kernel.update_kvcache(" in text
    assert "inputs=[%k_cache, %v_cache, %k, %v, %pos]" in text
    assert "output=None" in text
    assert "callee_kind=kernel" in text
    assert "effect=write(%k_cache, %v_cache)" in text


def test_verifier_rejects_alloc_storage():
    x = Var("x")
    y = Var("y")
    block = Block(bindings=((y, Call("@alloc_storage", (x,))),), body=Return(y))
    fn = Function(params=(x,), body=block)
    module = IRModule({"bad": fn})

    with pytest.raises(IRVerificationError, match="alloc_storage"):
        verify(module)


def test_verifier_rejects_alloc_tensor():
    x = Var("x")
    y = Var("y")
    block = Block(bindings=((y, Call("@alloc_tensor", (x,))),), body=Return(y))
    fn = Function(params=(x,), body=block)
    module = IRModule({"bad": fn})

    with pytest.raises(IRVerificationError, match="alloc_tensor"):
        verify(module)


def test_verifier_catches_use_before_def():
    x = Var("x")
    y = Var("y")
    z = Var("z")
    block = Block(
        bindings=((y, Call("@foo", (z,))),),  # z is not defined yet
        body=Return(y),
    )
    fn = Function(params=(x,), body=block)
    module = IRModule({"f": fn})

    with pytest.raises(IRVerificationError, match="used before definition"):
        verify(module)


def test_verifier_catches_double_def():
    x = Var("x")
    y = Var("y")
    block = Block(
        bindings=(
            (y, Call("@foo", (x,))),
            (y, Call("@bar", (x,))),  # y defined twice
        ),
        body=Return(y),
    )
    fn = Function(params=(x,), body=block)
    module = IRModule({"f": fn})

    with pytest.raises(IRVerificationError, match="defined more than once"):
        verify(module)


def test_verifier_accepts_valid_module():
    x = Var("x", TensorStructInfo((ConstDim(128),), "float16", "cuda"))
    y = Var("y")
    block = Block(bindings=((y, Call("@relu", (x,))),), body=Return(y))
    fn = Function(params=(x,), body=block)
    module = IRModule({"f": fn})
    verify(module)  # must not raise


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
    module = IRModule({"f": fn})
    text = print_module(module)
    assert "%qkv = @qkv_proj(%x)" in text
    assert "%q = %qkv[0]" in text


def test_tensor_create_op():
    B = SymbolicDim("B", upper=8)
    x = Var("x")
    buf = Var("buf")
    block = Block(
        bindings=(
            (buf, TensorCreateOp(
                kind=TensorCreateKind.empty,
                shape=(SymDimRef(B), ConstDim(4096)),
                dtype="float16",
                device="cuda",
            )),
        ),
        body=Return(buf),
    )
    fn = Function(params=(x,), body=block)
    module = IRModule({"f": fn})
    text = print_module(module)
    assert "dp.empty" in text
    assert "B" in text
    assert "4096" in text


def test_shape_expr_arithmetic():
    S = SymbolicDim("S", upper=2048)
    expr = BinOpDim("ceildiv", SymDimRef(S), ConstDim(16))
    p = Printer()
    assert p.print_shape_expr(expr) == "ceildiv(S, 16)"


def test_symbolic_dim_in_struct_info():
    B = SymbolicDim("B", upper=8)
    S = SymbolicDim("S", upper=2048)
    si = TensorStructInfo(
        (SymDimRef(B), SymDimRef(S), ConstDim(4096)), "float16", "cuda"
    )
    p = Printer()
    text = p.print_struct_info(si)
    assert "B" in text
    assert "S" in text
    assert "4096" in text
    assert "float16" in text
