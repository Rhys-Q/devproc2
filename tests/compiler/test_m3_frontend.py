"""M3 frontend DSL tests — rewritten for Op/Block/Region architecture."""
import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.passes.control_flow_normalize import ControlFlowNormalizePass
from devproc2.compiler.passes.control_flow_verify import ControlFlowVerifyPass
from devproc2.ir import (
    Block,
    Function,
    IRModule,
    IRVerificationError,
    Var,
    print_module,
    verify,
)
from devproc2.ir.ops import ForOp, IfOp, YieldOp


@pytest.fixture(autouse=True)
def reset():
    dp.reset_module()
    yield
    dp.reset_module()


# ---------------------------------------------------------------------------
# Scenario 5: elif expansion + loop-carried For
# ---------------------------------------------------------------------------

def test_frontend_decode_step():
    @dp.function
    def decode_step(x, flag, n):
        if flag:
            y = dp.ops.relu(x)
        elif flag > 0:
            y = dp.ops.silu(x)
        else:
            y = dp.ops.gelu(x)
        for i in dp.range(0, n):
            y = dp.ops.layernorm(y)
        return y

    module = dp.get_module()
    assert "decode_step" in module.functions
    verify(module)

    fn = module.functions["decode_step"]
    entry_ops = fn.body.entry_block.ops

    # Find IfOp and ForOp
    if_op = next((op for op in entry_ops if isinstance(op, IfOp)), None)
    for_op = next((op for op in entry_ops if isinstance(op, ForOp)), None)

    assert if_op is not None, "Expected IfOp from if/elif/else"
    assert for_op is not None, "Expected ForOp from for loop"

    # elif is nested If inside else_region
    assert if_op.else_region is not None
    inner_ops = if_op.else_region.entry_block.ops
    inner_if = next((op for op in inner_ops if isinstance(op, IfOp)), None)
    assert inner_if is not None, "elif should produce nested IfOp"

    # For loop has iter_args (y is loop-carried)
    assert len(for_op.iter_args) == 1
    assert for_op.iter_args[0].var.name.startswith("y")


def test_frontend_decode_step_printed_ir():
    @dp.function
    def decode_step(x, flag, n):
        if flag:
            y = dp.ops.relu(x)
        elif flag > 0:
            y = dp.ops.silu(x)
        else:
            y = dp.ops.gelu(x)
        for i in dp.range(0, n):
            y = dp.ops.layernorm(y)
        return y

    text = print_module(dp.get_module())
    assert "if " in text
    assert "@relu" in text
    assert "@silu" in text
    assert "@gelu" in text
    assert "for %i in range" in text
    assert "iter_args" in text
    assert "@layernorm" in text
    assert "return" in text


# ---------------------------------------------------------------------------
# SSA-result If
# ---------------------------------------------------------------------------

def test_frontend_ssa_if():
    @dp.function
    def branch(x, flag):
        if flag:
            y = dp.ops.relu(x)
        else:
            y = dp.ops.silu(x)
        return y

    module = dp.get_module()
    verify(module)
    fn = module.functions["branch"]
    if_op = next(op for op in fn.body.entry_block.ops if isinstance(op, IfOp))
    assert len(if_op.results) == 1


# ---------------------------------------------------------------------------
# Effect-only If
# ---------------------------------------------------------------------------

def test_frontend_effect_only_if():
    @dp.function
    def update(k_cache, v_cache, cond):
        if cond:
            dp.ops.update_kvcache(k_cache, v_cache)
        else:
            dp.ops.noop(k_cache)
        return k_cache

    module = dp.get_module()
    verify(module)
    fn = module.functions["update"]
    if_op = next(op for op in fn.body.entry_block.ops if isinstance(op, IfOp))
    assert if_op.results == ()
    # then branch yields nothing
    then_yield = if_op.then_region.entry_block.ops[-1]
    assert isinstance(then_yield, YieldOp)
    assert then_yield.values == ()


# ---------------------------------------------------------------------------
# Loop-carried For
# ---------------------------------------------------------------------------

def test_frontend_loop_carried_for():
    @dp.function
    def loop_accum(acc, x, n):
        for i in dp.range(0, n):
            acc = dp.ops.add(acc, x)
        return acc

    module = dp.get_module()
    verify(module)
    fn = module.functions["loop_accum"]
    for_op = next(op for op in fn.body.entry_block.ops if isinstance(op, ForOp))
    assert len(for_op.iter_args) == 1
    body_yield = for_op.body_region.entry_block.ops[-1]
    assert isinstance(body_yield, YieldOp)
    assert len(body_yield.values) == 1


# ---------------------------------------------------------------------------
# Effect-only For
# ---------------------------------------------------------------------------

def test_frontend_effect_only_for():
    @dp.function
    def write_loop(k_cache, v_cache, n):
        for i in dp.range(0, n):
            dp.ops.update_kvcache(k_cache, v_cache, i)
        return k_cache

    module = dp.get_module()
    verify(module)
    fn = module.functions["write_loop"]
    for_op = next(op for op in fn.body.entry_block.ops if isinstance(op, ForOp))
    assert for_op.iter_args == ()
    body_yield = for_op.body_region.entry_block.ops[-1]
    assert isinstance(body_yield, YieldOp)
    assert body_yield.values == ()


# ---------------------------------------------------------------------------
# ControlFlowNormalizePass
# ---------------------------------------------------------------------------

def test_normalize_pass_runs_on_module():
    @dp.function
    def f(x, flag):
        if flag:
            y = dp.ops.relu(x)
        else:
            y = dp.ops.silu(x)
        return y

    module = dp.get_module()
    normalized = ControlFlowNormalizePass().run(module)
    verify(normalized)
    assert "f" in normalized.functions


# ---------------------------------------------------------------------------
# ControlFlowVerifyPass
# ---------------------------------------------------------------------------

def test_cf_verify_pass_accepts_valid_module():
    @dp.function
    def g(k_cache, cond, n):
        for i in dp.range(0, n):
            if cond:
                dp.ops.write(k_cache, i)
            else:
                dp.ops.noop(k_cache)
        return k_cache

    module = dp.get_module()
    verify(module)
    ControlFlowVerifyPass().run(module)


# ---------------------------------------------------------------------------
# DSL error cases
# ---------------------------------------------------------------------------

def test_frontend_rejects_while():
    src = """
def bad(x):
    while True:
        x = dp.ops.relu(x)
    return x
"""
    import ast, textwrap
    from devproc2.frontend.dsl import DSLBuilder, DSLError
    tree = ast.parse(textwrap.dedent(src))
    fn_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    with pytest.raises(DSLError, match="while"):
        DSLBuilder().build(fn_def)


def test_frontend_rejects_break():
    src = """
def bad(x, n):
    for i in dp.range(0, n):
        break
    return x
"""
    import ast, textwrap
    from devproc2.frontend.dsl import DSLBuilder, DSLError
    tree = ast.parse(textwrap.dedent(src))
    fn_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    with pytest.raises(DSLError, match="break"):
        DSLBuilder().build(fn_def)
