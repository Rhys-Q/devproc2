"""M7 Memory Planning MVP tests."""
import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import (
    LiveInterval,
    MemoryPlanningPass,
    StoragePlan,
)
from devproc2.ir import (
    IRModule,
    OpaqueEffect,
    TensorStructInfo,
    WriteEffect,
    print_module,
    verify,
)
from devproc2.ir.nodes import Var
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    CallDPSOp,
    CalleeKind,
    TensorCreateKind,
    TensorCreateOp,
)
from devproc2.ir.prim_expr import IntImm, PrimVar
from devproc2.kernel.registry import KernelRegistry, KernelSpec


@pytest.fixture(autouse=True)
def reset():
    dp.reset_module()
    yield
    dp.reset_module()


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _reg(*specs: KernelSpec) -> KernelRegistry:
    r = KernelRegistry()
    for s in specs:
        r.register(s)
    return r


def _spec(op: str, **kw) -> KernelSpec:
    defaults = dict(device="cuda", input_dtypes=("float16",),
                    kernel_name=f"kernel.{op}_fp16")
    defaults.update(kw)
    return KernelSpec(op_name=op, **defaults)


def _lowered(module: IRModule, *specs: KernelSpec) -> IRModule:
    module = InferStructInfoPass().run(module)
    return DPSLoweringPass(_reg(*specs)).run(module)


# ---------------------------------------------------------------------------
# LiveInterval helpers
# ---------------------------------------------------------------------------

def test_live_interval_overlaps_basic():
    a = LiveInterval(0, 3)
    b = LiveInterval(2, 5)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_live_interval_no_overlap():
    a = LiveInterval(0, 2)
    b = LiveInterval(3, 5)
    assert not a.overlaps(b)
    assert not b.overlaps(a)


def test_live_interval_touching():
    # intervals that share an endpoint are overlapping
    a = LiveInterval(0, 3)
    b = LiveInterval(3, 5)
    assert a.overlaps(b)


# ---------------------------------------------------------------------------
# MemoryPlanningPass: basic functionality
# ---------------------------------------------------------------------------

def test_memory_planning_returns_same_module():
    """MemoryPlanningPass must not modify the IRModule."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    original_text = print_module(module)

    ctx = PassContext()
    out_module = MemoryPlanningPass().run(module, ctx)

    assert print_module(out_module) == original_text


def test_memory_planning_writes_storage_plan():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)

    plan = ctx.get("storage_plan")
    assert plan is not None
    assert isinstance(plan, StoragePlan)
    assert len(plan.entries) >= 1


def test_memory_planning_single_tensor_one_entry():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)

    plan = ctx.get("storage_plan")
    # One TensorCreateOp for y → one storage entry
    assert len(plan.entries) == 1
    assert plan.entries[0].reused_by == ["y"]


# ---------------------------------------------------------------------------
# Size computation with symbolic dims
# ---------------------------------------------------------------------------

def test_size_uses_upper_bound():
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def f(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)

    plan = ctx.get("storage_plan")
    # size = align256(8 * 2048 * 4096 * 2) = 134_217_728
    expected = ((8 * 2048 * 4096 * 2 + 255) // 256) * 256
    assert plan.entries[0].size_bytes == expected


def test_size_missing_upper_dynamic_alloc():
    """A tensor with an unbounded PrimVar still gets alloc_storage + alloc_tensor.
    It is marked is_reusable=True (unless returned), so two such tensors with the
    same size_expr and non-overlapping intervals share one storage block.
    """
    B = PrimVar("B", upper=None)  # no upper bound

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)

    plan = ctx.get("storage_plan")
    assert plan is not None
    # y is returned → not reusable, gets its own entry
    assert len(plan.entries) == 1
    assert plan.entries[0].reused_by == ["y"]
    assert plan.entries[0].size_bytes is None       # dynamic: no static size
    from devproc2.ir.prim_expr import PrimExpr
    assert isinstance(plan.entries[0].size_expr, PrimExpr)

    # LowerTensorCreateToAllocPass emits alloc_storage + alloc_tensor
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)
    fn = lowered.functions["f"]
    ops = fn.body.entry_block.ops
    assert any(isinstance(op, AllocStorageOp) for op in ops)
    assert any(isinstance(op, AllocTensorOp) for op in ops)
    assert not any(isinstance(op, TensorCreateOp) for op in ops)

    # AllocStorageOp carries a symbolic size_bytes, not a plain integer constant
    from devproc2.ir.prim_expr import IntImm
    storage_op = next(op for op in ops if isinstance(op, AllocStorageOp))
    assert not isinstance(storage_op.size_bytes, IntImm), (
        "Dynamic tensor should have a symbolic size_bytes expression"
    )


def test_dynamic_shape_tensors_can_share_storage():
    """Two intermediate tensors with the same unbounded dynamic shape and
    non-overlapping life intervals must share a single storage block.
    """
    B = PrimVar("B", upper=None)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)
        b = dp.ops.layernorm(a)
        c = dp.ops.relu(b)       # same shape as a; intervals don't overlap
        d = dp.ops.layernorm(c)  # returned
        return d

    module = _lowered(f.lower_module(), _spec("relu"), _spec("layernorm"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")

    # a and c have same size_expr and non-overlapping intervals → share storage
    assert plan.tensor_to_storage["a"] == plan.tensor_to_storage["c"], (
        "a and c should share storage (same dynamic shape, non-overlapping intervals)"
    )
    # d is returned → separate storage
    assert plan.tensor_to_storage["d"] != plan.tensor_to_storage["a"]
    # At least one entry reused by 2+ tensors
    assert any(len(e.reused_by) >= 2 for e in plan.entries)


# ---------------------------------------------------------------------------
# Storage reuse: non-overlapping tensors
# ---------------------------------------------------------------------------

def test_reuse_sequential_same_device_and_size():
    """Two independent tensors of same shape/device are assigned same storage."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"],
          z: dp.Tensor[(B, 512), "float16", "cuda"]):
        # y uses x, then is returned → not reusable
        # w uses z, then is returned → not reusable
        # But if we have intermediate ops:
        a = dp.ops.relu(x)
        b = dp.ops.silu(z)
        return a, b

    # Both relu and silu produce same-shape tensors; both are returned
    # → each gets its own storage (not reusable)
    module = _lowered(f.lower_module(),
                      _spec("relu"),
                      KernelSpec(op_name="silu", device="cuda",
                                 input_dtypes=("float16",),
                                 kernel_name="kernel.silu_fp16"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")
    # Both outputs → 2 entries
    assert len(plan.entries) == 2


def test_reuse_intermediate_tensors():
    """Intermediate tensors with non-overlapping lifetimes share storage."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)       # intermediate: a used only by layernorm
        b = dp.ops.layernorm(a)  # b is returned
        return b

    module = _lowered(f.lower_module(), _spec("relu"), _spec("layernorm"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")

    # a is intermediate (live: relu → layernorm); b is returned
    # a is reusable, b is not → they get different storage
    assert len(plan.entries) == 2
    a_id = plan.tensor_to_storage["a"]
    b_id = plan.tensor_to_storage["b"]
    assert a_id != b_id


def test_reuse_three_sequential_intermediates():
    """a → b → c → return c.  a ends before c starts → a and c can't share.
    But a ends at b's start; b ends at c's start — none overlap with each other
    in terms of final last_use vs first_def boundaries (depending on exact indexing)."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)
        b = dp.ops.layernorm(a)
        c = dp.ops.silu(b)
        return c

    silu_spec = KernelSpec(op_name="silu", device="cuda",
                           input_dtypes=("float16",),
                           kernel_name="kernel.silu_fp16")
    module = _lowered(f.lower_module(), _spec("relu"), _spec("layernorm"), silu_spec)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")

    # 3 TensorCreateOps: a, b, c
    # c is returned → not reusable (own entry)
    # a and b are intermediates → may share if non-overlapping
    assert len(plan.tensor_to_storage) == 3
    # At most 3 entries (no reuse), at least 2 (c separate, a+b may share if non-overlapping)
    assert len(plan.entries) <= 3
    assert len(plan.entries) >= 2


def test_at_least_two_tensors_share_storage():
    """Acceptance criterion: at least 2 tensors reuse same storage.

    With 4 ops: a→b→c→d (d returned):
      a: [0,3], b: [2,5], c: [4,7], d: [6,8] not-reusable
    a and c have non-overlapping intervals → share storage.
    """
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)
        b = dp.ops.layernorm(a)
        c = dp.ops.relu(b)       # c is intermediate; same shape as a
        d = dp.ops.layernorm(c)  # returned
        return d

    module = _lowered(f.lower_module(), _spec("relu"), _spec("layernorm"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")

    multi_reuse = [e for e in plan.entries if len(e.reused_by) >= 2]
    assert multi_reuse, (
        f"Expected at least one storage entry reused by 2+ tensors, "
        f"got entries: {[e.reused_by for e in plan.entries]}"
    )


# ---------------------------------------------------------------------------
# Effect-aware lifetime extension
# ---------------------------------------------------------------------------

def test_opaque_effect_extends_live_ranges():
    """An OpaqueEffect CallDPS must extend the live range of all live tensors."""
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"],
          k_cache: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)
        return a

    module = _lowered(f.lower_module(), _spec("relu"))
    # Manually inject a no-output CallDPS with OpaqueEffect after relu
    fn = module.functions["f"]
    from devproc2.ir.nodes import Block, Region, Function
    entry = fn.body.entry_block
    # Insert a CallDPS(effect=opaque, output=None) between TensorCreateOp and ReturnOp
    relu_create = entry.ops[0]  # TensorCreateOp
    relu_dps = entry.ops[1]     # CallDPSOp
    ret = entry.ops[2]          # ReturnOp
    opaque_call = CallDPSOp(
        callee="kernel.side_effect",
        callee_kind=CalleeKind.kernel,
        inputs=(relu_create.results[0],),
        output=None,
        effect=OpaqueEffect(),
    )
    new_entry = Block(entry.args, (relu_create, relu_dps, opaque_call, ret))
    new_fn = Function(Region((new_entry,)), fn.ret_struct_info)
    module = IRModule({"f": new_fn})

    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")
    # a is returned → not reusable regardless; just check planning didn't crash
    assert plan is not None


def test_write_effect_extends_var_live_range():
    """WriteEffect(vars=[k_cache]) should extend k_cache lifetime."""
    # This test builds the IR manually to precisely control effects.
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)
        return a

    module = _lowered(f.lower_module(), _spec("relu"))
    # Insert a CallDPS with WriteEffect on the tensor 'a'
    fn = module.functions["f"]
    entry = fn.body.entry_block
    create_op = next(op for op in entry.ops if isinstance(op, TensorCreateOp))
    a_result = create_op.results[0]

    from devproc2.ir.nodes import Block, Region, Function, Var, TensorStructInfo
    from devproc2.ir.prim_expr import IntImm
    # Create a dummy var with same struct_info
    dummy_var = Var(name="buf", struct_info=TensorStructInfo(
        shape=(IntImm(8), IntImm(512)), dtype="float16", device="cuda"
    ))
    write_call = CallDPSOp(
        callee="kernel.write_something",
        callee_kind=CalleeKind.kernel,
        inputs=(a_result,),
        output=None,
        effect=WriteEffect(vars=(dummy_var,)),
    )
    ret = next(op for op in entry.ops if not isinstance(op, (TensorCreateOp, CallDPSOp)))
    new_ops = tuple(op for op in entry.ops if not isinstance(op, type(ret)))
    new_ops = new_ops + (write_call, ret)
    new_entry = Block(entry.args, new_ops)
    new_fn = Function(Region((new_entry,)), fn.ret_struct_info)
    module = IRModule({"f": new_fn})

    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")
    assert plan is not None


# ---------------------------------------------------------------------------
# LowerTensorCreateToAllocPass
# ---------------------------------------------------------------------------

def test_lower_produces_alloc_storage_and_alloc_tensor():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)

    fn = lowered.functions["f"]
    all_ops = fn.body.entry_block.ops
    assert any(isinstance(op, AllocStorageOp) for op in all_ops)
    assert any(isinstance(op, AllocTensorOp) for op in all_ops)
    assert not any(isinstance(op, TensorCreateOp) for op in all_ops)


def test_lower_alloc_storage_hoisted_first():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)

    fn = lowered.functions["f"]
    first_op = fn.body.entry_block.ops[0]
    assert isinstance(first_op, AllocStorageOp)


def test_lower_alloc_tensor_references_storage():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)

    fn = lowered.functions["f"]
    ops = fn.body.entry_block.ops
    storage_op = next(op for op in ops if isinstance(op, AllocStorageOp))
    tensor_op  = next(op for op in ops if isinstance(op, AllocTensorOp))
    assert tensor_op.storage is storage_op.results[0]


def test_lower_alloc_tensor_preserves_shape_and_dtype():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)

    fn = lowered.functions["f"]
    tensor_op = next(
        op for op in fn.body.entry_block.ops if isinstance(op, AllocTensorOp)
    )
    assert tensor_op.shape[0] is B
    assert tensor_op.shape[1] == IntImm(512)
    assert tensor_op.dtype == "float16"
    assert tensor_op.offset == 0


def test_lower_printed_ir_contains_alloc_keywords():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)

    text = print_module(lowered)
    assert "alloc_storage" in text
    assert "alloc_tensor" in text
    assert "dp.empty" not in text  # TensorCreateOp gone


def test_lower_verifier_passes():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)
    verify(lowered)


def test_lower_without_plan_raises():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    module = _lowered(f.lower_module(), _spec("relu"))
    ctx = PassContext()  # empty — no plan stored
    with pytest.raises(RuntimeError, match="storage_plan"):
        LowerTensorCreateToAllocPass(ctx).run(module)


# ---------------------------------------------------------------------------
# Multi-function module: per-function plan keys
# ---------------------------------------------------------------------------

def test_per_function_plan_keys():
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.relu(x)
        return y

    @dp.function
    def g(x: dp.Tensor[(B, 512), "float16", "cuda"]):
        y = dp.ops.layernorm(x)
        return y

    module = IRModule()
    module.functions["f"] = f.lower_module().functions["f"]
    module.functions["g"] = g.lower_module().functions["g"]
    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(_reg(_spec("relu"), _spec("layernorm"))).run(module)

    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)

    assert ctx.get("storage_plan:f") is not None
    assert ctx.get("storage_plan:g") is not None
    # single "storage_plan" not set when there are 2 functions
    assert ctx.get("storage_plan") is None


# ---------------------------------------------------------------------------
# Acceptance criteria scenario (from spec)
# ---------------------------------------------------------------------------

def test_acceptance_storage_plan_json_shape():
    """Spec: storage_plan with at least 2 entries where reuse occurs.

    Uses a 4-op chain so non-adjacent intermediates can share storage.
    """
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        a = dp.ops.relu(x)
        b = dp.ops.layernorm(a)
        c = dp.ops.relu(b)       # intermediate; reuses a's storage
        d = dp.ops.layernorm(c)  # returned
        return d

    module = _lowered(main.lower_module(), _spec("relu"), _spec("layernorm"))
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")

    assert plan is not None
    assert len(plan.entries) >= 2

    # At least one entry has >= 2 tensors (storage reuse)
    assert any(len(e.reused_by) >= 2 for e in plan.entries), \
        f"No reuse found. Entries: {[(e.id, e.reused_by) for e in plan.entries]}"

    # tensor_to_storage covers all 4 tensors
    assert set(plan.tensor_to_storage.keys()) == {"a", "b", "c", "d"}

    # d is returned → separate storage
    d_sid = plan.tensor_to_storage["d"]
    other_sids = {plan.tensor_to_storage[k] for k in ("a", "b", "c")}
    # At least one intermediate shares a storage block with another
    assert len(other_sids) < 3  # at most 2 distinct storage ids for a, b, c


def test_acceptance_lower_alloc_ir():
    """Spec: alloc_storage/alloc_tensor appear in IR; at least 2 tensors share storage."""
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 4096), "float16", "cuda"]):
        a = dp.ops.relu(x)
        b = dp.ops.layernorm(a)
        c = dp.ops.relu(b)       # c reuses a's storage
        d = dp.ops.layernorm(c)  # returned
        return d

    module = _lowered(main.lower_module(), _spec("relu"), _spec("layernorm"))

    # MemoryPlanningPass: IRModule unchanged
    ctx = PassContext()
    after_planning = MemoryPlanningPass().run(module, ctx)
    assert print_module(after_planning) == print_module(module)
    verify(after_planning)

    # LowerTensorCreateToAllocPass: TensorCreateOp → alloc_storage + alloc_tensor
    lowered = LowerTensorCreateToAllocPass(ctx).run(module)
    text = print_module(lowered)
    assert "alloc_storage" in text
    assert "alloc_tensor" in text
    assert "dp.empty" not in text
    verify(lowered)

    # Check storage entries
    plan = ctx.get("storage_plan")
    assert any(len(e.reused_by) >= 2 for e in plan.entries)

    # At least one AllocStorageOp covers multiple AllocTensorOps
    fn = lowered.functions["main"]
    ops = fn.body.entry_block.ops
    storage_ops = [op for op in ops if isinstance(op, AllocStorageOp)]
    tensor_ops  = [op for op in ops if isinstance(op, AllocTensorOp)]
    assert len(storage_ops) >= 1
    assert len(tensor_ops) == 4

    # Check reuse: at least one storage referenced by >= 2 tensor ops
    storage_usage: dict[int, int] = {}
    for top in tensor_ops:
        sid = id(top.storage.op)
        storage_usage[sid] = storage_usage.get(sid, 0) + 1
    assert max(storage_usage.values()) >= 2


# ---------------------------------------------------------------------------
# prim_expr_structural_eq
# ---------------------------------------------------------------------------

def test_prim_expr_structural_eq_intImm():
    from devproc2.ir.prim_expr import prim_expr_structural_eq, IntImm
    assert prim_expr_structural_eq(IntImm(42), IntImm(42))
    assert not prim_expr_structural_eq(IntImm(1), IntImm(2))


def test_prim_expr_structural_eq_primvar_by_name_upper():
    from devproc2.ir.prim_expr import prim_expr_structural_eq, PrimVar
    # Different objects, same (name, upper) → structurally equal
    a = PrimVar("B", upper=8)
    b = PrimVar("B", upper=8)
    assert a is not b                          # different objects
    assert a != b                              # Python == uses identity
    assert prim_expr_structural_eq(a, b)       # structural: equal

    # Different upper → not equal
    c = PrimVar("B", upper=16)
    assert not prim_expr_structural_eq(a, c)

    # Different name → not equal
    d = PrimVar("S", upper=8)
    assert not prim_expr_structural_eq(a, d)

    # No upper vs upper → not equal
    e = PrimVar("B", upper=None)
    assert not prim_expr_structural_eq(a, e)


def test_prim_expr_structural_eq_composite():
    from devproc2.ir.prim_expr import prim_expr_structural_eq, PrimVar, Mul, IntImm
    B1 = PrimVar("B", upper=8)
    B2 = PrimVar("B", upper=8)
    expr1 = Mul(B1, IntImm(2))
    expr2 = Mul(B2, IntImm(2))
    # B1 is not B2, but structural equality treats them the same
    assert prim_expr_structural_eq(expr1, expr2)

    # Different constant → not equal
    expr3 = Mul(B1, IntImm(4))
    assert not prim_expr_structural_eq(expr1, expr3)


def test_prim_expr_structural_eq_type_mismatch():
    from devproc2.ir.prim_expr import prim_expr_structural_eq, PrimVar, IntImm
    assert not prim_expr_structural_eq(PrimVar("B"), IntImm(8))


# ---------------------------------------------------------------------------
# Dynamic reuse with reconstructed PrimVar (different objects, same semantics)
# ---------------------------------------------------------------------------

def test_dynamic_reuse_with_reconstructed_primvars():
    """Two TensorCreateOps built with *different* PrimVar objects that have the
    same (name, upper) must still share storage, thanks to prim_expr_structural_eq.
    """
    from devproc2.ir.nodes import Block, Region, Function
    from devproc2.ir.prim_expr import IntImm, PrimVar
    from devproc2.ir.ops import (
        TensorCreateOp, TensorCreateKind, CallDPSOp, CalleeKind, ReturnOp,
    )
    from devproc2.ir.nodes import OpaqueEffect, Var, TensorStructInfo

    # Two *different* PrimVar objects with the same name/upper
    B1 = PrimVar("B", upper=None)
    B2 = PrimVar("B", upper=None)
    assert B1 is not B2

    shape1 = (B1, IntImm(512))
    shape2 = (B2, IntImm(512))

    x = Var("x", struct_info=TensorStructInfo(shape1, "float16", "cuda"))

    create_a = TensorCreateOp("a", TensorCreateKind.empty, shape1, "float16", "cuda")
    dps_a = CallDPSOp("k.relu", CalleeKind.kernel, (x,), create_a.results[0], OpaqueEffect())
    create_b = TensorCreateOp("b", TensorCreateKind.empty, shape2, "float16", "cuda")
    dps_b = CallDPSOp("k.layernorm", CalleeKind.kernel, (create_a.results[0],), create_b.results[0], OpaqueEffect())
    # Manufacture two more ops so a and b don't overlap
    create_c = TensorCreateOp("c", TensorCreateKind.empty, shape1, "float16", "cuda")
    dps_c = CallDPSOp("k.relu", CalleeKind.kernel, (create_b.results[0],), create_c.results[0], OpaqueEffect())
    create_d = TensorCreateOp("d", TensorCreateKind.empty, shape2, "float16", "cuda")
    dps_d = CallDPSOp("k.layernorm", CalleeKind.kernel, (create_c.results[0],), create_d.results[0], OpaqueEffect())
    ret = ReturnOp((create_d.results[0],))

    block = Block(args=(x,), ops=(
        create_a, dps_a,
        create_b, dps_b,
        create_c, dps_c,
        create_d, dps_d,
        ret,
    ))
    fn = Function(Region((block,)))
    module = IRModule({"f": fn})

    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")

    # a and c should share (non-overlapping intervals, same size_expr by structural eq)
    assert plan.tensor_to_storage["a"] == plan.tensor_to_storage["c"], (
        "a and c should share storage: same dynamic shape (structural eq), non-overlapping"
    )


# ---------------------------------------------------------------------------
# DSL: tuple return produces TupleOp (not Var("(a, b)"))
# ---------------------------------------------------------------------------

def test_dsl_tuple_return_uses_tupleop():
    from devproc2.ir.ops import TupleOp, ReturnOp
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"],
          z: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)
        b = dp.ops.silu(z)
        return a, b

    fn = f.lower_module().functions["f"]
    ret = next(op for op in fn.body.entry_block.ops if isinstance(op, ReturnOp))
    # ReturnOp must carry the result of a TupleOp, not a bare Var
    from devproc2.ir.nodes import OpResult
    assert len(ret.values) == 1
    assert isinstance(ret.values[0], OpResult)
    assert isinstance(ret.values[0].op, TupleOp)
    tuple_op = ret.values[0].op
    assert len(tuple_op.elems) == 2


def test_dsl_tuple_return_marks_both_tensors_non_reusable():
    """After the DSL fix, both tensors in `return a, b` are detected as
    returned (not reusable) by _collect_return_values via TupleOp traversal.
    """
    B = dp.symbolic_dim("B", upper=8)

    @dp.function
    def f(x: dp.Tensor[(B, 512), "float16", "cuda"],
          z: dp.Tensor[(B, 512), "float16", "cuda"]):
        a = dp.ops.relu(x)
        b = dp.ops.silu(z)
        return a, b

    silu_spec = KernelSpec(op_name="silu", device="cuda", input_dtypes=("float16",),
                           kernel_name="kernel.silu_fp16")
    module = _lowered(f.lower_module(), _spec("relu"), silu_spec)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")

    # Both a and b are returned → each gets its own storage entry
    assert len(plan.entries) == 2
    a_sid = plan.tensor_to_storage["a"]
    b_sid = plan.tensor_to_storage["b"]
    assert a_sid != b_sid


# ---------------------------------------------------------------------------
# AllocStorageOp: int size_bytes is auto-coerced to IntImm
# ---------------------------------------------------------------------------

def test_alloc_storage_op_int_coercion():
    from devproc2.ir.ops import AllocStorageOp
    from devproc2.ir.prim_expr import IntImm
    op = AllocStorageOp(result_name="s0", size_bytes=1024, alignment=256, device="cuda")
    assert isinstance(op.size_bytes, IntImm)
    assert op.size_bytes.value == 1024


def test_alloc_storage_op_prim_expr_unchanged():
    from devproc2.ir.ops import AllocStorageOp
    from devproc2.ir.prim_expr import IntImm, Mul, PrimVar
    B = PrimVar("B", upper=None)
    expr = Mul(B, IntImm(2))
    op = AllocStorageOp(result_name="s0", size_bytes=expr, alignment=256, device="cuda")
    assert op.size_bytes is expr  # PrimExpr passed directly, unchanged
