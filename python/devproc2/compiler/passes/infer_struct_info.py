"""InferStructInfoPass — propagate TensorStructInfo through the IR."""
from __future__ import annotations

from devproc2.ir.nodes import Function, IRModule, Op, OpResult, StructInfo, Var
from devproc2.ir.ops import CallOp, ForOp, IfOp
from devproc2.compiler.passes._rewriter import IRRewriter


class InferStructInfoPass(IRRewriter):
    def __init__(self) -> None:
        super().__init__()
        self._type_env: dict[int, StructInfo] = {}  # id(Value) → StructInfo

    def run(self, module: IRModule) -> IRModule:
        return self.rewrite_module(module)

    def rewrite_fn(self, fn: Function) -> Function:
        self._sub = {}
        self._type_env = {}
        for p in fn.params:
            if p.struct_info is not None:
                self._type_env[id(p)] = p.struct_info
        new_body = self.rewrite_region(fn.body)
        return Function(new_body, fn.ret_struct_info)

    def rewrite_op(self, op: Op) -> Op:
        if isinstance(op, CallOp) and op.result_name:
            # Record existing struct_info.
            if op.results[0].struct_info is not None:
                self._type_env[id(op.results[0])] = op.results[0].struct_info
            # Propagate from first arg if result has none.
            elif op.args:
                first = op.args[0]
                # First, apply substitution (in case first was mapped)
                first = self.sv(first)
                si = self._type_env.get(id(first))
                if si is not None:
                    new_op = CallOp(
                        callee=op.callee,
                        args=self.svs(op.args),
                        result_name=op.result_name,
                        result_struct_info=si,
                    )
                    self._type_env[id(new_op.results[0])] = si
                    return new_op
        return self._subst_op(op)
