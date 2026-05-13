"""M6 DPS Lowering MVP tests."""
import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.kernel_select import KernelSelectPass
from devproc2.ir import IRModule, TensorStructInfo, print_module, verify
from devproc2.ir.nodes import IntImm
from devproc2.ir.ops import CallDPSOp, CallOp, IfOp, ReturnOp, TensorCreateOp
from devproc2.ir.prim_expr import PrimVar
from devproc2.kernel.registry import KernelMatchKey, KernelRegistry, KernelSpec


@pytest.fixture(autouse=True)
def reset():
    dp.reset_module()
    yield
    dp.reset_module()


def _make_registry(*specs: KernelSpec) -> KernelRegistry:
    reg = KernelRegistry()
    for s in specs:
        reg.register(s)
    return reg


def _relu_spec(**kwargs) -> KernelSpec:
    defaults = dict(op_name="relu", device="cuda", dtype="float16",
                    kernel_name="kernel.relu_fp16")
    defaults.update(kwargs)
    return KernelSpec(**defaults)


def _layernorm_spec(**kwargs) -> KernelSpec:
    defaults = dict(op_name="layernorm", device="cuda", dtype="float16",
                    kernel_name="kernel.layernorm_fp16")
    defaults.update(kwargs)
    return KernelSpec(**defaults)


# ---------------------------------------------------------------------------
# KernelRegistry tests
# ---------------------------------------------------------------------------

def test_registry_register_and_lookup_exact():
    reg = _make_registry(_relu_spec())
    key = KernelMatchKey("relu", "cuda", "float16")
    spec = reg.lookup(key)
    assert spec is not None
    assert spec.kernel_name == "kernel.relu_fp16"


def test_registry_lookup_no_match():
    reg = _make_registry(_relu_spec())
    key = KernelMatchKey("relu", "cpu", "float16")
    assert reg.lookup(key) is None


def test_registry_lookup_wildcard_dtype():
    spec = KernelSpec(op_name="relu", device="cuda", dtype="float16",
                      kernel_name="kernel.relu_fp16")
    reg = _make_registry(spec)
    # registry dtype="float16", key dtype="*" → should match
    key = KernelMatchKey("relu", "cuda", "*")
    assert reg.lookup(key) is not None


def test_registry_lookup_priority():
    low  = KernelSpec(op_name="relu", device="cuda", dtype="float16",
                      kernel_name="kernel.relu_low",  priority=0)
    high = KernelSpec(op_name="relu", device="cuda", dtype="float16",
                      kernel_name="kernel.relu_high", priority=10)
    reg = _make_registry(low, high)
    key = KernelMatchKey("relu", "cuda", "float16")
    assert reg.lookup(key).kernel_name == "kernel.relu_high"


def test_registry_lookup_match_predicate():
    def only_small(call_op: CallOp) -> bool:
        # Pretend: match only if first arg is named "x_small"
        return False  # never matches in this test

    spec = KernelSpec(op_name="relu", device="cuda", dtype="float16",
                      kernel_name="kernel.relu_special", match=only_small)
    fallback = _relu_spec(kernel_name="kernel.relu_fp16", priority=-1)
    reg = _make_registry(spec, fallback)

    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    fn = module.functions["f"]
    call_op = next(op for op in fn.body.entry_block.ops if isinstance(op, CallOp))
    key = KernelMatchKey("relu", "cuda", "float16")
    # match=False → spec rejected, fallback wins
    result = reg.lookup(key, call_op)
    assert result.kernel_name == "kernel.relu_fp16"


# ---------------------------------------------------------------------------
# KernelSelectPass tests
# ---------------------------------------------------------------------------

def test_kernel_select_finds_annotated_call():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    reg = _make_registry(_relu_spec())
    sel = KernelSelectPass(reg).run(module)

    fn = module.functions["f"]
    call_op = next(op for op in fn.body.entry_block.ops if isinstance(op, CallOp))
    assert id(call_op) in sel
    assert sel[id(call_op)].kernel_name == "kernel.relu_fp16"


def test_kernel_select_skips_unannotated_call():
    @dp.function
    def f(x):
        y = dp.ops.relu(x)
        return y

    # No InferStructInfoPass — struct_info is None
    module = dp.get_module()
    reg = _make_registry(_relu_spec())
    sel = KernelSelectPass(reg).run(module)
    assert len(sel) == 0


def test_kernel_select_no_registry_hit():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    reg = KernelRegistry()  # empty registry
    sel = KernelSelectPass(reg).run(module)
    assert len(sel) == 0


# ---------------------------------------------------------------------------
# DPSLoweringPass tests
# ---------------------------------------------------------------------------

def _lowered_module(*specs):
    module = InferStructInfoPass().run(dp.get_module())
    reg = _make_registry(*specs)
    return DPSLoweringPass(reg).run(module)


def test_dps_lowering_inserts_tensor_create():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered_module(_relu_spec())
    fn = module.functions["f"]
    ops = fn.body.entry_block.ops
    assert isinstance(ops[0], TensorCreateOp)
    assert isinstance(ops[1], CallDPSOp)


def test_dps_lowering_removes_call_op():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered_module(_relu_spec())
    fn = module.functions["f"]
    assert not any(isinstance(op, CallOp) for op in fn.body.entry_block.ops)


def test_dps_lowering_callee_name():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered_module(_relu_spec())
    fn = module.functions["f"]
    dps = next(op for op in fn.body.entry_block.ops if isinstance(op, CallDPSOp))
    assert dps.callee == "kernel.relu_fp16"


def test_dps_lowering_output_is_tensor_create_result():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered_module(_relu_spec())
    fn = module.functions["f"]
    ops = fn.body.entry_block.ops
    create_op = ops[0]
    dps_op    = ops[1]
    assert isinstance(create_op, TensorCreateOp)
    assert isinstance(dps_op, CallDPSOp)
    assert dps_op.output is create_op.results[0]


def test_dps_lowering_return_uses_create_result():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered_module(_relu_spec())
    fn = module.functions["f"]
    ops = fn.body.entry_block.ops
    create_op = ops[0]
    ret = next(op for op in ops if isinstance(op, ReturnOp))
    assert ret.values[0] is create_op.results[0]


def test_dps_lowering_unmatched_call_unchanged():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    # Empty registry → no lowering
    module = InferStructInfoPass().run(dp.get_module())
    module = DPSLoweringPass(KernelRegistry()).run(module)
    fn = module.functions["f"]
    assert any(isinstance(op, CallOp) for op in fn.body.entry_block.ops)
    assert not any(isinstance(op, TensorCreateOp) for op in fn.body.entry_block.ops)


def test_dps_lowering_printed_ir():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    module = DPSLoweringPass(_make_registry(_layernorm_spec())).run(module)
    text = print_module(module)
    assert "dp.empty" in text
    assert "call_dps" in text
    assert "kernel.layernorm_fp16" in text


# ---------------------------------------------------------------------------
# Pipeline and regression tests
# ---------------------------------------------------------------------------

def test_dps_lowering_verifier_passes():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered_module(_relu_spec())
    verify(module)


def test_full_m6_pipeline():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    module = dp.get_module()
    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(_make_registry(_layernorm_spec())).run(module)

    verify(module)

    fn = module.functions["main"]
    ops = fn.body.entry_block.ops
    create_op = next(op for op in ops if isinstance(op, TensorCreateOp))
    dps_op    = next(op for op in ops if isinstance(op, CallDPSOp))

    # Output shape matches input StructInfo
    assert create_op.shape[0] is B  # PrimVar identity preserved
    assert create_op.shape[1] is S
    assert create_op.shape[2] == IntImm(4096)
    assert create_op.dtype == "float16"
    assert create_op.device == "cuda"

    assert dps_op.callee == "kernel.layernorm_fp16"
    assert dps_op.output is create_op.results[0]


def test_m4_regression_no_lowering_without_registry():
    """Empty registry: DPSLoweringPass is a no-op; verifier still passes."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 4096), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = InferStructInfoPass().run(dp.get_module())
    module = DPSLoweringPass(KernelRegistry()).run(module)
    verify(module)
    fn = module.functions["f"]
    assert any(isinstance(op, CallOp) for op in fn.body.entry_block.ops)


def test_dps_lowering_chained_ops():
    """Chain: z = layernorm(relu(x)).  Both ops must be lowered and the
    _sub substitution must correctly wire layernorm's input to relu's buffer."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        z = dp.ops.layernorm(y)
        return z

    reg = _make_registry(_relu_spec(), _layernorm_spec())
    module = InferStructInfoPass().run(dp.get_module())
    module = DPSLoweringPass(reg).run(module)

    verify(module)
    fn = module.functions["f"]
    ops = fn.body.entry_block.ops

    # No CallOps remain; two TensorCreateOps and two CallDPSOps
    assert not any(isinstance(op, CallOp) for op in ops)
    creates = [op for op in ops if isinstance(op, TensorCreateOp)]
    dpss    = [op for op in ops if isinstance(op, CallDPSOp)]
    assert len(creates) == 2
    assert len(dpss) == 2

    relu_create, ln_create = creates
    relu_dps, ln_dps = dpss

    # layernorm's input must be the relu output buffer
    assert relu_dps.output is relu_create.results[0]
    assert ln_dps.output is ln_create.results[0]
    assert ln_dps.inputs[0] is relu_create.results[0]

    # ReturnOp refers to layernorm's buffer
    ret = next(op for op in ops if isinstance(op, ReturnOp))
    assert ret.values[0] is ln_create.results[0]


def test_dps_lowering_inside_if_branch():
    """DPS lowering must work inside an IfOp then/else branch."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"], flag):
        if flag:
            y = dp.ops.relu(x)
        else:
            y = dp.ops.silu(x)
        return y

    silu_spec = KernelSpec(op_name="silu", device="cuda", dtype="float16",
                           kernel_name="kernel.silu_fp16")
    reg = _make_registry(_relu_spec(), silu_spec)
    module = InferStructInfoPass().run(dp.get_module())
    module = DPSLoweringPass(reg).run(module)

    verify(module)
    fn = module.functions["f"]

    if_op = next(op for op in fn.body.entry_block.ops if isinstance(op, IfOp))

    then_ops = if_op.then_region.entry_block.ops
    else_ops = if_op.else_region.entry_block.ops

    assert any(isinstance(op, TensorCreateOp) for op in then_ops)
    assert any(isinstance(op, CallDPSOp) for op in then_ops)
    assert any(isinstance(op, TensorCreateOp) for op in else_ops)
    assert any(isinstance(op, CallDPSOp) for op in else_ops)
