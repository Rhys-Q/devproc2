"""ControlFlowNormalizePass — normalize control flow IR (Op/Block/Region arch)."""
from __future__ import annotations

from devproc2.ir.nodes import Function, IRModule
from devproc2.compiler.passes._rewriter import IRRewriter


class ControlFlowNormalizePass(IRRewriter):
    def run(self, module: IRModule) -> IRModule:
        return self.rewrite_module(module)


