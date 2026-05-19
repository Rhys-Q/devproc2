"""Shared IR rewriter base for passes that rebuild ops.

When a pass creates a new Op from an old one, the new Op's results have
different object ids.  Any downstream op that uses the old Op's results would
fail the verifier.  IRRewriter maintains a substitution map that is applied
to Value operands when rebuilding each op.
"""
from __future__ import annotations

from devproc2.ir.nodes import (
    AliasInfo,
    Block,
    EffectSummary,
    Function,
    IRModule,
    Op,
    OpResult,
    Region,
    Value,
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
        operands = self.svs(op.operands)
        regions = tuple(self.rewrite_region(region) for region in op.regions)
        return op.replace_operands(
            operands,
            regions=regions,
            effects=self._subst_effect(op.effects),
        )

    def _subst_effect(self, effect: EffectSummary) -> EffectSummary:
        return EffectSummary(
            reads=self.svs(effect.reads),
            writes=self.svs(effect.writes),
            allocates=effect.allocates,
            frees=effect.frees,
            opaque=effect.opaque,
            external_state=effect.external_state,
            alias=(
                AliasInfo(effect.alias.kind, self.sv(effect.alias.source))
                if effect.alias is not None and effect.alias.source is not None
                else effect.alias
            ),
        )
