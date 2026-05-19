"""M3 Control Flow IR tests — rewritten for Op/Block/Region architecture."""
import pytest

from devproc2.ir import (
    AliasAnalysis,
    Block,
    CallDPSOp,
    CallOp,
    Constant,
    EffectSummary,
    ForOp,
    Function,
    IfOp,
    IRModule,
    IRStage,
    IRVerificationError,
    IterArg,
    KernelRef,
    Printer,
    Range,
    Region,
    ReturnOp,
    StandardOpRef,
    AllocStorageOp,
    TensorStructInfo,
    TerminatorOp,
    TupleGetItemOp,
    TupleOp,
    Var,
    YieldOp,
    print_module,
    verify,
)


def _calldps(callee, inputs, writes):
    return CallDPSOp(
        KernelRef(callee),
        inputs=inputs,
        outputs=(),
        effect=EffectSummary.write(*writes),
    )


def std(name: str) -> StandardOpRef:
    return StandardOpRef(name)


def _cf_region(*ops):
    return Region((Block(args=(), ops=ops),))


# ---------------------------------------------------------------------------
# Scenario 1: SSA-result IfOp
# ---------------------------------------------------------------------------

def _make_ssa_if_module():
    x = Var("x"); flag = Var("flag")
    relu_op = CallOp(std("relu"), args=(x,), result_name="v0"); v0 = relu_op.results[0]
    silu_op = CallOp(std("silu"), args=(x,), result_name="v1"); v1 = silu_op.results[0]
    then_region = _cf_region(relu_op, YieldOp((v0,)))
    else_region = _cf_region(silu_op, YieldOp((v1,)))
    if_op = IfOp(cond=flag, then_region=then_region, else_region=else_region, result_names=("y",))
    y = if_op.results[0]
    block = Block(args=(x, flag), ops=(if_op, ReturnOp((y,))))
    return IRModule({"branch_relu_silu": Function(Region((block,)))})


def test_ssa_if_printer():
    text = print_module(_make_ssa_if_module())
    assert "if %flag {" in text
    assert "yield %v0" in text
    assert "} else {" in text
    assert "yield %v1" in text
    assert "%y" in text and "if %flag" in text
    assert "return %y" in text


def test_ssa_if_verifier_passes():
    verify(_make_ssa_if_module())


def test_alias_analysis_resolves_if_result_sources():
    module = _make_ssa_if_module()
    if_op = module.functions["branch_relu_silu"].body.entry_block.ops[0]
    aliases = AliasAnalysis.from_region(module.functions["branch_relu_silu"].body)

    assert aliases.sources(if_op.results[0]) == (
        if_op.then_region.entry_block.ops[-1].values[0],
        if_op.else_region.entry_block.ops[-1].values[0],
    )


def test_alias_analysis_projects_tuple_get_item_through_if_result():
    x0 = Var("x0")
    x1 = Var("x1")
    y0 = Var("y0")
    y1 = Var("y1")
    flag = Var("flag")

    then_tuple = TupleOp("then_pair", (x0, x1))
    else_tuple = TupleOp("else_pair", (y0, y1))
    if_op = IfOp(
        cond=flag,
        then_region=_cf_region(then_tuple, YieldOp((then_tuple.results[0],))),
        else_region=_cf_region(else_tuple, YieldOp((else_tuple.results[0],))),
        result_names=("pair",),
    )
    item = TupleGetItemOp(if_op.results[0], 0, "picked")

    aliases = AliasAnalysis((if_op, then_tuple, else_tuple, item))

    assert aliases.sources(item.results[0]) == (x0, y0)
    assert aliases.resolve_matching(item.results[0], lambda v: v in (x0, y0)) == frozenset({x0, y0})


# ---------------------------------------------------------------------------
# Scenario 2: Effect-only IfOp
# ---------------------------------------------------------------------------

def _make_effect_if_module():
    k_cache = Var("k_cache"); v_cache = Var("v_cache"); cond = Var("cond")
    k = Var("k"); v = Var("v"); pos = Var("pos")
    then_region = _cf_region(
        _calldps("@update_kvcache", (k_cache, v_cache, k, v, pos), (k_cache, v_cache)),
        YieldOp(()))
    else_region = _cf_region(_calldps("@noop", (k_cache,), (k_cache,)), YieldOp(()))
    if_op = IfOp(cond=cond, then_region=then_region, else_region=else_region)
    block = Block(args=(k_cache, v_cache, cond, k, v, pos), ops=(if_op, ReturnOp((k_cache,))))
    return IRModule({"update": Function(Region((block,)))})


def test_effect_if_printer():
    text = print_module(_make_effect_if_module())
    assert "if %cond {" in text
    assert "yield\n" in text
    assert "} else {" in text
    assert "= if" not in text


def test_effect_if_verifier_passes():
    verify(_make_effect_if_module())


# ---------------------------------------------------------------------------
# Scenario 3: Loop-carried ForOp
# ---------------------------------------------------------------------------

def _make_loop_carried_module():
    acc = Var("acc"); x = Var("x"); n = Var("n"); i = Var("i"); acc_iter = Var("acc_iter")
    add_op = CallOp(std("add"), args=(acc_iter, x), result_name="acc_next")
    acc_next = add_op.results[0]
    body_region = _cf_region(add_op, YieldOp((acc_next,)))
    for_op = ForOp(loop_var=i, range_=Range(Constant(0), n, Constant(1)),
                   iter_args=(IterArg(var=acc_iter, init=acc),),
                   body_region=body_region, result_names=("acc_out",))
    acc_out = for_op.results[0]
    block = Block(args=(acc, x, n), ops=(for_op, ReturnOp((acc_out,))))
    return IRModule({"loop_accum": Function(Region((block,)))})


def test_loop_carried_printer():
    text = print_module(_make_loop_carried_module())
    assert "for %i in range(0, %n, 1)" in text
    assert "iter_args(%acc_iter = %acc)" in text
    assert "yield %acc_next" in text
    assert "%acc_out" in text
    assert "return %acc_out" in text


def test_loop_carried_verifier_passes():
    verify(_make_loop_carried_module())


# ---------------------------------------------------------------------------
# Scenario 4: Effect-only ForOp
# ---------------------------------------------------------------------------

def _make_effect_for_module():
    k_cache = Var("k_cache"); v_cache = Var("v_cache"); n = Var("n"); i = Var("i")
    body_region = _cf_region(
        _calldps("@update_kvcache", (k_cache, v_cache, i), (k_cache, v_cache)), YieldOp(()))
    for_op = ForOp(loop_var=i, range_=Range(Constant(0), n, Constant(1)),
                   iter_args=(), body_region=body_region)
    block = Block(args=(k_cache, v_cache, n), ops=(for_op, ReturnOp((k_cache,))))
    return IRModule({"write_loop": Function(Region((block,)))})


def test_effect_for_printer():
    text = print_module(_make_effect_for_module())
    assert "for %i in range(0, %n, 1) {" in text
    assert "iter_args" not in text
    assert "yield\n" in text
    assert "= for" not in text


def test_effect_for_verifier_passes():
    verify(_make_effect_for_module())


# ---------------------------------------------------------------------------
# Scenario 5: Nested For + If
# ---------------------------------------------------------------------------

def _make_nested_module():
    k_cache = Var("k_cache"); v_cache = Var("v_cache"); cond = Var("cond"); n = Var("n"); i = Var("i")
    then_region = _cf_region(_calldps("@write_a", (k_cache, i), (k_cache,)), YieldOp(()))
    else_region = _cf_region(_calldps("@write_b", (v_cache, i), (v_cache,)), YieldOp(()))
    inner_if = IfOp(cond=cond, then_region=then_region, else_region=else_region)
    body_region = _cf_region(inner_if, YieldOp(()))
    for_op = ForOp(loop_var=i, range_=Range(Constant(0), n, Constant(1)),
                   iter_args=(), body_region=body_region)
    block = Block(args=(k_cache, v_cache, cond, n), ops=(for_op, ReturnOp((k_cache,))))
    return IRModule({"nested": Function(Region((block,)))})


def test_nested_cf_printer():
    text = print_module(_make_nested_module())
    assert "for %i in range" in text
    assert "if %cond" in text
    assert "@write_a" in text
    assert "@write_b" in text


def test_nested_cf_verifier_passes():
    verify(_make_nested_module())


# ---------------------------------------------------------------------------
# Verifier error cases
# ---------------------------------------------------------------------------

def test_verifier_function_region_needs_return():
    x = Var("x")
    block = Block(args=(x,), ops=(YieldOp((x,)),))
    with pytest.raises(IRVerificationError, match="ReturnOp"):
        verify(IRModule({"f": Function(Region((block,)))}))


def test_verifier_cf_region_needs_yield():
    x = Var("x"); flag = Var("flag")
    then_region = Region((Block(args=(), ops=(ReturnOp((x,)),)),))
    else_region = _cf_region(YieldOp(()))
    if_op = IfOp(cond=flag, then_region=then_region, else_region=else_region)
    block = Block(args=(x, flag), ops=(if_op, ReturnOp((x,))))
    with pytest.raises(IRVerificationError, match="YieldOp"):
        verify(IRModule({"f": Function(Region((block,)))}))


def test_verifier_if_yield_count_mismatch():
    x = Var("x"); flag = Var("flag")
    relu_op = CallOp(std("relu"), args=(x,), result_name="v0"); v0 = relu_op.results[0]
    silu_op = CallOp(std("silu"), args=(x,), result_name="v1"); v1 = silu_op.results[0]
    gelu_op = CallOp(std("gelu"), args=(x,), result_name="v2"); v2 = gelu_op.results[0]
    then_region = _cf_region(relu_op, YieldOp((v0,)))
    else_region = _cf_region(silu_op, gelu_op, YieldOp((v1, v2)))
    if_op = IfOp(cond=flag, then_region=then_region, else_region=else_region, result_names=("y",))
    y = if_op.results[0]
    block = Block(args=(x, flag), ops=(if_op, ReturnOp((y,))))
    with pytest.raises(IRVerificationError):
        verify(IRModule({"f": Function(Region((block,)))}))


def test_verifier_if_result_requires_else_region():
    x = Var("x"); flag = Var("flag")
    relu_op = CallOp(std("relu"), args=(x,), result_name="v0"); v0 = relu_op.results[0]
    then_region = _cf_region(relu_op, YieldOp((v0,)))
    if_op = IfOp(cond=flag, then_region=then_region, result_names=("y",))
    y = if_op.results[0]
    block = Block(args=(x, flag), ops=(if_op, ReturnOp((y,))))
    with pytest.raises(IRVerificationError, match="requires an else_region"):
        verify(IRModule({"f": Function(Region((block,)))}))


def test_verifier_if_branch_mismatch_no_result():
    x = Var("x"); flag = Var("flag")
    relu_op = CallOp(std("relu"), args=(x,), result_name="v0"); v0 = relu_op.results[0]
    then_region = _cf_region(relu_op, YieldOp((v0,)))
    else_region = _cf_region(YieldOp(()))
    if_op = IfOp(cond=flag, then_region=then_region, else_region=else_region)
    block = Block(args=(x, flag), ops=(if_op, ReturnOp((x,))))
    with pytest.raises(IRVerificationError, match="then_region yields"):
        verify(IRModule({"f": Function(Region((block,)))}))


def test_verifier_for_yield_count_mismatch():
    acc = Var("acc"); x = Var("x"); n = Var("n"); i = Var("i"); acc_iter = Var("acc_iter")
    add_op = CallOp(std("add"), args=(acc_iter, x), result_name="acc_next"); acc_next = add_op.results[0]
    foo_op = CallOp(std("foo"), args=(acc_iter,), result_name="extra"); extra = foo_op.results[0]
    body_region = _cf_region(add_op, foo_op, YieldOp((acc_next, extra)))
    for_op = ForOp(loop_var=i, range_=Range(Constant(0), n, Constant(1)),
                   iter_args=(IterArg(var=acc_iter, init=acc),),
                   body_region=body_region, result_names=("acc_out",))
    acc_out = for_op.results[0]
    block = Block(args=(acc, x, n), ops=(for_op, ReturnOp((acc_out,))))
    with pytest.raises(IRVerificationError, match="iter_args"):
        verify(IRModule({"f": Function(Region((block,)))}))


def test_verifier_for_range_use_before_def():
    acc = Var("acc"); i = Var("i"); undefined_n = Var("undefined_n"); acc_iter = Var("acc_iter")
    add_op = CallOp(std("add"), args=(acc_iter, Constant(1)), result_name="acc_next")
    acc_next = add_op.results[0]
    body_region = _cf_region(add_op, YieldOp((acc_next,)))
    for_op = ForOp(loop_var=i, range_=Range(Constant(0), undefined_n, Constant(1)),
                   iter_args=(IterArg(var=acc_iter, init=acc),),
                   body_region=body_region, result_names=("acc_out",))
    acc_out = for_op.results[0]
    block = Block(args=(acc,), ops=(for_op, ReturnOp((acc_out,))))
    with pytest.raises(IRVerificationError, match="used before definition"):
        verify(IRModule({"f": Function(Region((block,)))}))


def test_m2_regression_alloc_storage():
    alloc = AllocStorageOp(result_name="s0", size_bytes=1, alignment=256, device="cpu")
    block = Block(args=(), ops=(alloc, ReturnOp((alloc.results[0],))))
    with pytest.raises(IRVerificationError, match="AllocStorageOp"):
        verify(IRModule({"bad": Function(Region((block,)))}), stage=IRStage.raw)


def test_yield_is_terminator():
    assert issubclass(YieldOp, TerminatorOp)


def test_return_is_terminator():
    assert issubclass(ReturnOp, TerminatorOp)
