"""ControlFlowNormalizePass — normalize control flow IR (Op/Block/Region arch)."""
from __future__ import annotations

from devproc2.ir.nodes import IRModule, IRStage
from devproc2.compiler.passes._rewriter import IRRewriter


class ControlFlowNormalizePass(IRRewriter):
    input_stage = IRStage.raw
    output_stage = IRStage.normalized
    required_analysis: tuple[str, ...] = ()
    preserved_analysis: tuple[str, ...] = ()

    def run(self, module: IRModule) -> IRModule:
        return self.rewrite_module(module)
