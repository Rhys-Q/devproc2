"""M2 IR tests — rewritten for Op/Block/Region architecture."""
import pytest

from devproc2.ir import (
    Add,
    Block,
    CallDPSOp,
    CallOp,
    CeilDiv,
    Constant,
    EffectSummary,
    FloorDiv,
    ForOp,
    Function,
    GE,
    GT,
    IRModule,
    IRStage,
    IRVerificationError,
    IntImm,
    IterArg,
    KernelRef,
    LE,
    LT,
    Max,
    Min,
    Mul,
    Printer,
    PrimVar,
    Range,
    Region,
    ReturnOp,
    StandardOpRef,
    Sub,
    TensorCreateKind,
    TensorCreateOp,
    TensorStructInfo,
    KnownShape,
    TupleGetItemOp,
    TupleOp,
    Value,
    Var,
    YieldOp,
    ceildiv,
    pmax,
    pmin,
    print_module,
    verify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_fn(params: tuple[Var, ...], ops: tuple, name: str = "f") -> IRModule:
    block = Block(args=params, ops=ops)
    region = Region((block,))
    fn = Function(region)
    return IRModule({name: fn})


def std(name: str) -> StandardOpRef:
    return StandardOpRef(name)


def kernel_ref(name: str) -> KernelRef:
    return KernelRef(name)


# ---------------------------------------------------------------------------
# PrimExpr tests
# ---------------------------------------------------------------------------

def test_intImm_prints():
    p = Printer()
    assert p.print_prim_expr(IntImm(4096)) == "4096"


def test_primvar_prints():
    p = Printer()
    B = PrimVar("B", upper=8)
    assert p.print_prim_expr(B) == "B"


def test_prim_var_sym_id_unique():
    a = PrimVar("B")
    b = PrimVar("B")
    c = PrimVar("S")
    assert a.sym_id != b.sym_id
    assert a.sym_id != c.sym_id
    assert a is not b


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
    assert B.eq(S) is not None


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

    matmul_op = CallOp(std("matmul"), args=(x, w), result_name="y")
    y = matmul_op.results[0]

    silu_op = CallOp(std("silu"), args=(y,), result_name="z")
    z = silu_op.results[0]

    block = Block(
        args=(x, w),
        ops=(matmul_op, silu_op, ReturnOp(values=(z,))),
    )
    fn = Function(body=Region((block,)), ret_struct_info=out_si)
    text = print_module(IRModule({"main": fn}))

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

    calldps = CallDPSOp(
        target_ref=kernel_ref("@kernel.update_kvcache"),
        inputs=(k_cache, v_cache, k, v, pos),
        outputs=(),
        effect=EffectSummary.write(k_cache, v_cache),
    )

    block = Block(
        args=(k_cache, v_cache, k, v, pos),
        ops=(calldps, ReturnOp(values=(pos,))),
    )
    fn = Function(Region((block,)))
    text = print_module(IRModule({"update": fn}))

    assert "call_dps @kernel.update_kvcache(" in text
    assert "inputs=[%k_cache, %v_cache, %k, %v, %pos]" in text
    assert "outputs=[]" in text
    assert "effect=write(%k_cache, %v_cache)" in text


def test_print_calldps_with_output():
    x = Var("x")
    out = Var("out")
    calldps = CallDPSOp(
        target_ref=kernel_ref("@kernel.relu"),
        inputs=(x,),
        outputs=(out,),
        effect=EffectSummary.write(out),
    )
    block = Block(args=(x, out), ops=(calldps, ReturnOp(values=(out,))))
    fn = Function(Region((block,)))
    text = print_module(IRModule({"f": fn}))
    assert "call_dps @kernel.relu(" in text
    assert "outputs=[%out]" in text
    assert "effect=write(%out)" in text


def test_print_multi_function_separator():
    x = Var("x")
    call_op = CallOp(std("relu"), args=(x,), result_name="y")
    y = call_op.results[0]
    block = Block(args=(x,), ops=(call_op, ReturnOp((y,))))
    fn = Function(Region((block,)))
    assert "\n\n" in print_module(IRModule({"f1": fn, "f2": fn}))


def test_printer_reuse():
    x = Var("x")
    block = Block(args=(x,), ops=(ReturnOp((x,)),))
    fn = Function(Region((block,)))
    p = Printer()
    t1 = p.print_module(IRModule({"a": fn}))
    t2 = p.print_module(IRModule({"b": fn}))
    assert "@a" in t1 and "@b" not in t1
    assert "@b" in t2 and "@a" not in t2


def test_tuple_ir():
    x = Var("x")

    qkv_op = CallOp(std("qkv_proj"), args=(x,), result_name="qkv")
    qkv = qkv_op.results[0]

    tgi_op = TupleGetItemOp(tup=qkv, index=0, result_name="q")
    q = tgi_op.results[0]

    block = Block(
        args=(x,),
        ops=(qkv_op, tgi_op, ReturnOp((q,))),
    )
    fn = Function(Region((block,)))
    text = print_module(IRModule({"f": fn}))
    assert "%qkv = @qkv_proj(%x)" in text
    assert "%q = %qkv[0]" in text


def test_tensor_create_op_printer():
    B = PrimVar("B", upper=8)

    tc_op = TensorCreateOp(
        result_name="buf",
        kind=TensorCreateKind.empty,
        shape=(B, 4096),
        dtype="float16",
        device="cuda",
    )
    buf = tc_op.results[0]

    block = Block(
        args=(),
        ops=(tc_op, ReturnOp((buf,))),
    )
    fn = Function(Region((block,)))
    text = print_module(IRModule({"f": fn}))
    assert "dp.empty" in text
    assert "B" in text
    assert "4096" in text


# ---------------------------------------------------------------------------
# Verifier tests
# ---------------------------------------------------------------------------

def test_verifier_rejects_alloc_storage():
    from devproc2.ir import AllocStorageOp

    alloc = AllocStorageOp(result_name="s0", size_bytes=1, alignment=256, device="cpu")
    block = Block(args=(), ops=(alloc, ReturnOp((alloc.results[0],))))
    fn = Function(Region((block,)))
    with pytest.raises(IRVerificationError, match="AllocStorageOp"):
        verify(IRModule({"bad": fn}), stage=IRStage.raw)


def test_verifier_rejects_alloc_tensor():
    from devproc2.ir import AllocStorageOp, AllocTensorOp

    alloc = AllocStorageOp(result_name="s0", size_bytes=16, alignment=256, device="cpu")
    tensor = AllocTensorOp(result_name="t0", storage=alloc.results[0], offset=0, shape=(1,), dtype="float32")
    block = Block(args=(), ops=(alloc, tensor, ReturnOp((tensor.results[0],))))
    fn = Function(Region((block,)))
    with pytest.raises(IRVerificationError, match="AllocStorageOp"):
        verify(IRModule({"bad": fn}), stage=IRStage.raw)


def test_verifier_catches_use_before_def():
    x = Var("x")
    z = Var("z")  # never defined — Var as undefined operand
    call_op = CallOp(std("foo"), args=(z,), result_name="y")
    y = call_op.results[0]
    block = Block(args=(x,), ops=(call_op, ReturnOp((y,))))
    fn = Function(Region((block,)))
    with pytest.raises(IRVerificationError, match="used before definition"):
        verify(IRModule({"f": fn}))


def test_verifier_catches_double_def():
    x = Var("x")
    foo_op = CallOp(std("foo"), args=(x,), result_name="y")
    # Use same block arg twice — triggers double-def on block arg level
    block = Block(
        args=(x, x),  # x defined twice as block arg
        ops=(foo_op, ReturnOp((foo_op.results[0],))),
    )
    fn = Function(Region((block,)))
    with pytest.raises(IRVerificationError, match="defined more than once"):
        verify(IRModule({"f": fn}))


def test_verifier_write_effect_not_false_positive():
    """EffectSummary writes are effect metadata and are verified like operands."""
    k_cache = Var("k_cache")
    v_cache = Var("v_cache")
    k = Var("k")
    v = Var("v")
    pos = Var("pos")
    calldps = CallDPSOp(
        target_ref=kernel_ref("@kernel.update_kvcache"),
        inputs=(k_cache, v_cache, k, v, pos),
        outputs=(),
        effect=EffectSummary.write(k_cache, v_cache),
    )
    block = Block(args=(k_cache, v_cache, k, v, pos), ops=(calldps, ReturnOp((pos,))))
    fn = Function(Region((block,)))
    verify(IRModule({"f": fn}))  # must not raise


def test_verifier_block_must_not_be_empty():
    x = Var("x")
    block = Block(args=(x,), ops=())
    fn = Function(Region((block,)))
    with pytest.raises(IRVerificationError, match="must not be empty"):
        verify(IRModule({"f": fn}))


def test_verifier_last_op_must_be_terminator():
    x = Var("x")
    call_op = CallOp(std("relu"), args=(x,), result_name="y")
    block = Block(args=(x,), ops=(call_op,))
    fn = Function(Region((block,)))
    with pytest.raises(IRVerificationError, match="TerminatorOp"):
        verify(IRModule({"f": fn}))


def test_verifier_terminator_not_at_end():
    x = Var("x")
    call_op = CallOp(std("relu"), args=(x,), result_name="y")
    y = call_op.results[0]
    block = Block(
        args=(x,),
        ops=(
            ReturnOp((x,)),
            call_op,
            ReturnOp((y,)),
        ),
    )
    fn = Function(Region((block,)))
    with pytest.raises(IRVerificationError, match="must be the last op"):
        verify(IRModule({"f": fn}))


def test_verifier_accepts_valid_module():
    x = Var("x", TensorStructInfo((128,), "float16", "cuda"))
    call_op = CallOp(std("relu"), args=(x,), result_name="y")
    y = call_op.results[0]
    block = Block(args=(x,), ops=(call_op, ReturnOp((y,))))
    fn = Function(Region((block,)))
    verify(IRModule({"f": fn}))


def test_stage_verifier_allows_tensor_create_before_memory_ir():
    create = TensorCreateOp(
        result_name="buf",
        kind=TensorCreateKind.empty,
        shape=(IntImm(4),),
        dtype="float32",
        device="cpu",
    )
    block = Block(args=(), ops=(create, ReturnOp((create.results[0],))))
    fn = Function(Region((block,)))

    verify(IRModule({"f": fn}), stage=IRStage.raw)


def test_stage_verifier_keeps_tuple_ops_after_dps_lowering():
    x = Var("x")
    y = Var("y")
    tup = TupleOp(result_name="pair", elems=(x, y))
    block = Block(args=(x, y), ops=(tup, ReturnOp((tup.results[0],))))
    fn = Function(Region((block,)))

    verify(IRModule({"f": fn}), stage=IRStage.dps)


def test_known_shape_hash_matches_tuple_equality():
    shape = (IntImm(2), IntImm(3))
    known = KnownShape(shape)

    assert known == shape
    assert {known: "value"}[shape] == "value"


# ---------------------------------------------------------------------------
# TensorCreateOp validation tests
# ---------------------------------------------------------------------------

def test_tensor_create_op_empty_like_validation():
    x = Var("x")
    with pytest.raises(ValueError, match="requires 'like'"):
        TensorCreateOp(result_name="out", kind=TensorCreateKind.empty_like, shape=(), dtype="float16", device="cuda")

    B = PrimVar("B")
    with pytest.raises(ValueError, match="must not specify 'shape'"):
        TensorCreateOp(result_name="out", kind=TensorCreateKind.empty_like, shape=(B,), dtype="float16", device="cuda", like=x)

    with pytest.raises(ValueError, match="must not specify 'like'"):
        TensorCreateOp(result_name="out", kind=TensorCreateKind.empty, shape=(IntImm(128),), dtype="float16", device="cuda", like=x)

    op = TensorCreateOp(result_name="out", kind=TensorCreateKind.empty_like, shape=(), dtype="float16", device="cuda", like=x)
    assert op.like is x
