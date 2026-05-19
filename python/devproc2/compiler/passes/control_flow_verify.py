"""ControlFlowVerifyPass — strict semantic checks after normalization."""
from __future__ import annotations

from devproc2.ir.nodes import Block, Function, IRModule, IRStage, Op, Region
from devproc2.ir.ops import ForOp, IfOp, YieldOp
from devproc2.ir.verifier import IRVerificationError


class ControlFlowVerifyPass:
    input_stage = IRStage.normalized
    output_stage = IRStage.normalized
    required_analysis: tuple[str, ...] = ()
    preserved_analysis: tuple[str, ...] = ()

    def run(self, module: IRModule) -> IRModule:
        for name, fn in module.functions.items():
            self._verify_fn(name, fn)
        return module

    def _verify_fn(self, fn_name: str, fn: Function) -> None:
        outer = {p.name for p in fn.params}
        self._verify_region(fn_name, fn.body, outer)

    def _verify_region(self, fn_name: str, region: Region, outer: set[str]) -> None:
        for block in region.blocks:
            self._verify_block(fn_name, block, outer)

    def _verify_block(self, fn_name: str, block: Block, outer: set[str]) -> None:
        local = set(outer)
        for arg in block.args:
            local.add(arg.name)
        for op in block.ops:
            self._verify_op(fn_name, op, local)

    def _verify_op(self, fn_name: str, op: Op, outer: set[str]) -> None:
        if isinstance(op, IfOp):
            self._verify_if(fn_name, op, outer)
        elif isinstance(op, ForOp):
            self._verify_for(fn_name, op, outer)

    def _verify_if(self, fn_name: str, op: IfOp, outer: set[str]) -> None:
        if not op.results:
            # Effect-only: both branches must yield nothing.
            then_yield = op.then_region.entry_block.ops[-1]
            if isinstance(then_yield, YieldOp) and then_yield.values:
                raise IRVerificationError(
                    f"In @{fn_name}: effect-only IfOp true_branch must yield no values"
                )
            if op.else_region is not None:
                else_yield = op.else_region.entry_block.ops[-1]
                if isinstance(else_yield, YieldOp) and else_yield.values:
                    raise IRVerificationError(
                        f"In @{fn_name}: effect-only IfOp else_branch must yield no values"
                    )
        self._verify_region(fn_name, op.then_region, set(outer))
        if op.else_region is not None:
            self._verify_region(fn_name, op.else_region, set(outer))

    def _verify_for(self, fn_name: str, op: ForOp, outer: set[str]) -> None:
        if op.loop_var.name in outer:
            raise IRVerificationError(
                f"In @{fn_name}: ForOp loop_var '%{op.loop_var.name}' shadows outer variable"
            )
        body_outer = set(outer)
        body_outer.add(op.loop_var.name)
        for ia in op.iter_args:
            body_outer.add(ia.var.name)
        self._verify_region(fn_name, op.body_region, body_outer)
