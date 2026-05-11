from __future__ import annotations

from io import StringIO
from typing import Optional

from devproc2.ir.nodes import (
    BinOpDim,
    Binding,
    Block,
    Call,
    CallDPS,
    CalleeKind,
    ConstDim,
    Constant,
    Expr,
    EffectInfo,
    Function,
    IRModule,
    OpaqueEffect,
    PureEffect,
    ReadOnlyEffect,
    Return,
    ShapeExpr,
    SymDimRef,
    TensorCreateKind,
    TensorCreateOp,
    TensorStructInfo,
    TupleExpr,
    TupleGetItem,
    Var,
    WriteEffect,
)


class Printer:
    def __init__(self) -> None:
        self._buf = StringIO()

    def print_module(self, module: IRModule) -> str:
        for name, fn in module.functions.items():
            self._print_function(name, fn)
        return self._buf.getvalue()

    def _print_function(self, name: str, fn: Function) -> None:
        params_str = ", ".join(self._param_str(p) for p in fn.params)
        ret_str = ""
        if fn.ret_struct_info is not None:
            ret_str = f" -> {self.print_struct_info(fn.ret_struct_info)}"
        self._buf.write(f"@{name}({params_str}){ret_str} {{\n")
        self._print_block(fn.body, indent="  ")
        self._buf.write("}\n")

    def _param_str(self, v: Var) -> str:
        if v.struct_info is not None:
            return f"%{v.name}: {self.print_struct_info(v.struct_info)}"
        return f"%{v.name}"

    def _print_block(self, block: Block, indent: str = "  ") -> None:
        for binding in block.bindings:
            self._print_binding(binding, indent)
        self._print_body(block.body, indent)

    def _print_body(self, expr: Expr, indent: str) -> None:
        if isinstance(expr, Return):
            self._buf.write(f"{indent}return {self._expr_str(expr.value)}\n")
        else:
            self._buf.write(f"{indent}{self._expr_str(expr)}\n")

    def _print_binding(self, binding: Binding, indent: str) -> None:
        var, expr = binding
        if isinstance(expr, CallDPS):
            self._print_calldps(var, expr, indent)
        elif var is None:
            self._buf.write(f"{indent}{self._expr_str(expr)}\n")
        else:
            self._buf.write(f"{indent}%{var.name} = {self._expr_str(expr)}\n")

    def _print_calldps(self, var: Optional[Var], expr: CallDPS, indent: str) -> None:
        inner = indent + "  "
        inputs_str = "[" + ", ".join(self._expr_str(a) for a in expr.inputs) + "]"
        output_str = f"%{expr.output.name}" if expr.output is not None else "None"
        kind_str = expr.callee_kind.name
        effect_str = self._effect_str(expr.effect)
        self._buf.write(f"{indent}call_dps {expr.callee}(\n")
        self._buf.write(f"{inner}inputs={inputs_str},\n")
        self._buf.write(f"{inner}output={output_str},\n")
        self._buf.write(f"{inner}callee_kind={kind_str},\n")
        self._buf.write(f"{inner}effect={effect_str}\n")
        self._buf.write(f"{indent})\n")

    def _expr_str(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return f"%{expr.name}"
        if isinstance(expr, Constant):
            return repr(expr.value)
        if isinstance(expr, Call):
            args_str = ", ".join(self._expr_str(a) for a in expr.args)
            return f"{expr.callee}({args_str})"
        if isinstance(expr, TupleExpr):
            elems_str = ", ".join(self._expr_str(e) for e in expr.elems)
            return f"({elems_str})"
        if isinstance(expr, TupleGetItem):
            return f"{self._expr_str(expr.tup)}[{expr.index}]"
        if isinstance(expr, TensorCreateOp):
            return self._tensor_create_str(expr)
        if isinstance(expr, Return):
            return f"return {self._expr_str(expr.value)}"
        if isinstance(expr, CallDPS):
            # Inline form (should not normally appear via _expr_str, but as fallback)
            inputs_str = "[" + ", ".join(self._expr_str(a) for a in expr.inputs) + "]"
            output_str = f"%{expr.output.name}" if expr.output is not None else "None"
            return (
                f"call_dps {expr.callee}(inputs={inputs_str}, output={output_str}, "
                f"callee_kind={expr.callee_kind.name}, effect={self._effect_str(expr.effect)})"
            )
        return repr(expr)

    def _tensor_create_str(self, op: TensorCreateOp) -> str:
        kind_map = {
            TensorCreateKind.empty: "dp.empty",
            TensorCreateKind.zeros: "dp.zeros",
            TensorCreateKind.full: "dp.full",
            TensorCreateKind.empty_like: "dp.empty_like",
        }
        name = kind_map[op.kind]
        if op.kind == TensorCreateKind.empty_like:
            like_str = self._expr_str(op.like) if op.like is not None else "None"
            return f"{name}({like_str})"
        shape_str = "(" + ", ".join(self.print_shape_expr(s) for s in op.shape) + ")"
        result = f"{name}(shape={shape_str}, dtype={op.dtype}, device={op.device}"
        if op.kind == TensorCreateKind.full and op.fill_value is not None:
            result += f", fill_value={op.fill_value!r}"
        result += ")"
        return result

    def print_struct_info(self, si: TensorStructInfo) -> str:
        shape_str = ", ".join(self.print_shape_expr(s) for s in si.shape)
        return f"Tensor[({shape_str}), {si.dtype}, {si.device}]"

    def print_shape_expr(self, se: ShapeExpr) -> str:
        if isinstance(se, ConstDim):
            return str(se.value)
        if isinstance(se, SymDimRef):
            return se.dim.name
        if isinstance(se, BinOpDim):
            return f"{se.op}({self.print_shape_expr(se.lhs)}, {self.print_shape_expr(se.rhs)})"
        return repr(se)

    def _effect_str(self, eff: EffectInfo) -> str:
        if isinstance(eff, PureEffect):
            return "pure"
        if isinstance(eff, ReadOnlyEffect):
            return "read_only"
        if isinstance(eff, WriteEffect):
            vars_str = ", ".join(f"%{v.name}" for v in eff.vars)
            return f"write({vars_str})"
        if isinstance(eff, OpaqueEffect):
            return "opaque"
        return repr(eff)


def print_module(module: IRModule) -> str:
    return Printer().print_module(module)
