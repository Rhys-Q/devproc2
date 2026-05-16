"""InferStructInfoPass — propagate TensorStructInfo through the IR."""
from __future__ import annotations

from devproc2.compiler.passes._rewriter import IRRewriter
from devproc2.ir.nodes import (
    Function,
    IRModule,
    Op,
    OpResult,
    StructInfo,
    TensorStructInfo,
    Value,
    Var,
)
from devproc2.compiler.op import get_op, infer_struct_info
from devproc2.ir.ops import CallOp, ForOp, IfOp, TensorCreateOp


class InferStructInfoPass(IRRewriter):
    def __init__(self) -> None:
        super().__init__()
        self._type_env: dict[Value, StructInfo] = {}

    def run(self, module: IRModule) -> IRModule:
        return self.rewrite_module(module)

    def rewrite_fn(self, fn: Function) -> Function:
        self._sub = {}
        self._type_env = {}
        for p in fn.params:
            if p.struct_info is not None:
                self._type_env[p] = p.struct_info
        new_body = self.rewrite_region(fn.body)
        return Function(new_body, fn.ret_struct_info)

    def rewrite_op(self, op: Op) -> Op:
        if isinstance(op, TensorCreateOp):
            # Derive struct_info from the op's shape/dtype/device declaration.
            si = TensorStructInfo(op.shape, op.dtype, op.device)
            new_op = self._subst_op(op)
            # Stamp struct_info onto the OpResult so downstream ops (e.g.
            # DPSLoweringPass.build_input_dtypes) can read it directly.
            object.__setattr__(new_op.results[0], "struct_info", si)
            self._type_env[new_op.results[0]] = si
            return new_op
        if isinstance(op, CallOp) and op.result_name:
            # Record existing struct_info.
            if op.results[0].struct_info is not None:
                self._type_env[op.results[0]] = op.results[0].struct_info
            elif op.args:
                new_args = self.svs(op.args)
                arg_infos = tuple(self._struct_info_for_value(arg) for arg in new_args)
                op_def = op.op or get_op(op.callee)
                if op_def is not None:
                    si = op_def.infer_struct_info(arg_infos, op.attrs)
                else:
                    si = infer_struct_info(op.callee, arg_infos, op.attrs)
                if si is not None:
                    new_op = CallOp(
                        callee=op.callee,
                        args=new_args,
                        result_name=op.result_name,
                        result_struct_info=si,
                        attrs=op.attrs,
                        call_kind=op.call_kind,
                    )
                    self._type_env[new_op.results[0]] = si
                    return new_op
        return self._subst_op(op)

    def _struct_info_for_value(self, value: Value) -> StructInfo | None:
        value = self.sv(value)
        if isinstance(value, Var):
            return value.struct_info
        if isinstance(value, OpResult):
            if value.struct_info is not None:
                return value.struct_info
            return self._type_env.get(value)
        return None
