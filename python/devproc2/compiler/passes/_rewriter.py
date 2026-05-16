"""Shared IR rewriter base for passes that rebuild ops.

When a pass creates a new Op from an old one, the new Op's results have
different object ids.  Any downstream op that uses the old Op's results would
fail the verifier.  IRRewriter maintains a substitution map that is applied
to Value operands when rebuilding each op.
"""
from __future__ import annotations

from devproc2.ir.nodes import (
    Block, Function, IRModule, Op, OpResult, Region, Value, Var,
)
from devproc2.ir.ops import (
    AllocStorageOp, AllocTensorOp,
    CallDPSOp, CallOp, ForOp, IfOp, IterArg, Range,
    ReturnOp, ShapeAssertOp, TensorCreateOp, TupleGetItemOp, TupleOp, YieldOp,
)


class IRRewriter:
    """Base class for passes that do tree-transforming rewrites.

    Subclasses override `rewrite_op` to transform specific Op types.
    The base `rewrite_block` applies `rewrite_op` to each op and maintains
    a substitution map so that downstream ops automatically see new results.

    Scope note: `_sub` is a single dict shared across nested rewrite_region
    calls.  Inner-region results get registered in the same dict as outer
    results.  This is safe because SSA guarantees that inner results are never
    referenced outside their containing region; the stale entries are simply
    never looked up from outer scopes.
    """

    def __init__(self) -> None:
        # Maps old OpResult → new OpResult for the current block walk.
        # OpResult uses identity equality (eq=False), so direct object keys work.
        self._sub: dict[Value, OpResult] = {}

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def rewrite_module(self, module: IRModule) -> IRModule:
        return IRModule({n: self.rewrite_fn(fn) for n, fn in module.functions.items()})

    def rewrite_fn(self, fn: Function) -> Function:
        self._sub = {}
        new_body = self.rewrite_region(fn.body)
        return Function(new_body, fn.ret_struct_info)

    def rewrite_region(self, region: Region) -> Region:
        return Region(tuple(self.rewrite_block(b) for b in region.blocks))

    def rewrite_block(self, block: Block) -> Block:
        new_ops: list[Op] = []
        for op in block.ops:
            new_op = self.rewrite_op(op)
            # Register old→new result substitution.
            for old_r, new_r in zip(op.results, new_op.results):
                self._sub[old_r] = new_r
            new_ops.append(new_op)
        return Block(block.args, tuple(new_ops))

    def rewrite_op(self, op: Op) -> Op:
        """Override in subclasses.  Default: substitute operands only."""
        return self._subst_op(op)

    # ------------------------------------------------------------------
    # Value substitution helpers
    # ------------------------------------------------------------------

    def sv(self, v: Value) -> Value:
        """Substitute a single Value if it's a known OpResult."""
        if isinstance(v, OpResult):
            return self._sub.get(v, v)
        return v

    def svs(self, vals: tuple) -> tuple:
        return tuple(self.sv(v) for v in vals)

    # ------------------------------------------------------------------
    # Default operand-only rewrite (no structural change)
    # ------------------------------------------------------------------

    def _subst_op(self, op: Op) -> Op:
        """Rebuild op with substituted operands, preserving structure."""
        if isinstance(op, ReturnOp):
            return ReturnOp(values=self.svs(op.values))
        if isinstance(op, YieldOp):
            return YieldOp(values=self.svs(op.values))
        if isinstance(op, CallOp):
            return CallOp(callee=op.callee, args=self.svs(op.args),
                         result_name=op.result_name,
                         result_struct_info=op.result_struct_info,
                         attrs=op.attrs,
                         call_kind=op.call_kind)
        if isinstance(op, CallDPSOp):
            return CallDPSOp(callee=op.callee, callee_kind=op.callee_kind,
                            inputs=self.svs(op.inputs),
                            output=self.sv(op.output) if op.output is not None else None,
                            effect=op.effect,
                            attrs=op.attrs)
        if isinstance(op, TupleOp):
            return TupleOp(result_name=op.result_name, elems=self.svs(op.elems))
        if isinstance(op, TupleGetItemOp):
            return TupleGetItemOp(tup=self.sv(op.tup), index=op.index,
                                  result_name=op.result_name)
        if isinstance(op, IfOp):
            return IfOp(
                cond=self.sv(op.cond),
                then_region=self.rewrite_region(op.then_region),
                else_region=self.rewrite_region(op.else_region) if op.else_region else None,
                result_names=op.result_names,
            )
        if isinstance(op, ForOp):
            new_range = Range(self.sv(op.range_.start), self.sv(op.range_.end),
                             self.sv(op.range_.step))
            new_iter = tuple(IterArg(var=ia.var, init=self.sv(ia.init)) for ia in op.iter_args)
            return ForOp(
                loop_var=op.loop_var, range_=new_range, iter_args=new_iter,
                body_region=self.rewrite_region(op.body_region),
                result_names=op.result_names,
            )
        if isinstance(op, TensorCreateOp):
            return op  # shape is PrimExpr, not Value; no substitution needed
        if isinstance(op, AllocStorageOp):
            return op  # no Value operands
        if isinstance(op, AllocTensorOp):
            return AllocTensorOp(
                result_name=op.result_name,
                storage=self.sv(op.storage),
                offset=op.offset,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, ShapeAssertOp):
            return op  # tensor is a Var (block arg), no substitution needed
        # Unknown op type returned as-is.  Safe only when the op carries no
        # Value-typed operands that might reference substituted OpResults.
        # If a new Op with Value fields is added, add an explicit case above —
        # omitting it will cause "OpResult used before definition" in verify().
        assert not op.results, (
            f"Unhandled Op type with results in IRRewriter._subst_op: "
            f"{type(op).__name__}. Add an explicit case to preserve operand substitution."
        )
        return op
