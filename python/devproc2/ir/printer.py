from __future__ import annotations

from io import StringIO
from typing import Optional

from devproc2.ir.nodes import (
    Binding,
    Block,
    Call,
    CallDPS,
    Constant,
    EffectInfo,
    Expr,
    Function,
    IRModule,
    OpaqueEffect,
    PureEffect,
    ReadOnlyEffect,
    Return,
    StructInfo,
    TensorCreateKind,
    TensorCreateOp,
    TensorStructInfo,
    TupleExpr,
    TupleGetItem,
    Var,
    WriteEffect,
)
from devproc2.ir.prim_expr import (
    Add,
    CeilDiv,
    EQ,
    FloorDiv,
    GE,
    GT,
    IntImm,
    LE,
    LT,
    Max,
    Min,
    Mul,
    PrimExpr,
    PrimVar,
    Sub,
)


class Printer:
    def __init__(self) -> None:
        self._buf = StringIO()

    def print_module(self, module: IRModule) -> str:
        self._buf = StringIO()
        for i, (name, fn) in enumerate(module.functions.items()):
            if i > 0:
                self._buf.write("\n")
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
            self._print_calldps(expr, indent)
        elif var is None:
            self._buf.write(f"{indent}{self._expr_str(expr)}\n")
        else:
            self._buf.write(f"{indent}%{var.name} = {self._expr_str(expr)}\n")

    def _print_calldps(self, expr: CallDPS, indent: str) -> None:
        # For CallDPS the output appears inline as `output=%name` inside the body.
        # There is no separate `%out = ...` prefix regardless of whether output is set.
        inner = indent + "  "
        inputs_str = "[" + ", ".join(self._expr_str(a) for a in expr.inputs) + "]"
        output_str = f"%{expr.output.name}" if expr.output is not None else "None"
        self._buf.write(f"{indent}call_dps {expr.callee}(\n")
        self._buf.write(f"{inner}inputs={inputs_str},\n")
        self._buf.write(f"{inner}output={output_str},\n")
        self._buf.write(f"{inner}callee_kind={expr.callee_kind.name},\n")
        self._buf.write(f"{inner}effect={self._effect_str(expr.effect)}\n")
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
            return "(" + ", ".join(self._expr_str(e) for e in expr.elems) + ")"
        if isinstance(expr, TupleGetItem):
            return f"{self._expr_str(expr.tup)}[{expr.index}]"
        if isinstance(expr, TensorCreateOp):
            return self._tensor_create_str(expr)
        if isinstance(expr, CallDPS):
            # Inline fallback (CallDPS normally printed via _print_calldps at statement level)
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
            return f"{name}({self._expr_str(op.like)})"
        shape_str = "(" + ", ".join(self.print_prim_expr(s) for s in op.shape) + ")"
        result = f"{name}(shape={shape_str}, dtype={op.dtype}, device={op.device}"
        if op.kind == TensorCreateKind.full and op.fill_value is not None:
            result += f", fill_value={op.fill_value!r}"
        result += ")"
        return result

    def print_struct_info(self, si: StructInfo) -> str:
        if isinstance(si, TensorStructInfo):
            shape_str = ", ".join(self.print_prim_expr(s) for s in si.shape)
            return f"Tensor[({shape_str}), {si.dtype}, {si.device}]"
        raise NotImplementedError(f"No printer for StructInfo type: {type(si).__name__}")

    def print_prim_expr(self, e: PrimExpr) -> str:
        if isinstance(e, IntImm):
            return str(e.value)
        if isinstance(e, PrimVar):
            return e.name
        if isinstance(e, Add):
            return f"({self.print_prim_expr(e.lhs)} + {self.print_prim_expr(e.rhs)})"
        if isinstance(e, Sub):
            return f"({self.print_prim_expr(e.lhs)} - {self.print_prim_expr(e.rhs)})"
        if isinstance(e, Mul):
            return f"({self.print_prim_expr(e.lhs)} * {self.print_prim_expr(e.rhs)})"
        if isinstance(e, FloorDiv):
            return f"({self.print_prim_expr(e.lhs)} // {self.print_prim_expr(e.rhs)})"
        if isinstance(e, CeilDiv):
            return f"ceildiv({self.print_prim_expr(e.lhs)}, {self.print_prim_expr(e.rhs)})"
        if isinstance(e, Min):
            return f"min({self.print_prim_expr(e.lhs)}, {self.print_prim_expr(e.rhs)})"
        if isinstance(e, Max):
            return f"max({self.print_prim_expr(e.lhs)}, {self.print_prim_expr(e.rhs)})"
        if isinstance(e, (EQ, LT, LE, GT, GE)):
            op_sym = {EQ: "==", LT: "<", LE: "<=", GT: ">", GE: ">="}[type(e)]
            return f"({self.print_prim_expr(e.lhs)} {op_sym} {self.print_prim_expr(e.rhs)})"
        raise NotImplementedError(f"No printer for PrimExpr type: {type(e).__name__}")

    def _effect_str(self, eff: EffectInfo) -> str:
        if isinstance(eff, PureEffect):
            return "pure"
        if isinstance(eff, ReadOnlyEffect):
            return "read_only"
        if isinstance(eff, WriteEffect):
            return "write(" + ", ".join(f"%{v.name}" for v in eff.vars) + ")"
        if isinstance(eff, OpaqueEffect):
            return "opaque"
        return repr(eff)


def print_module(module: IRModule) -> str:
    return Printer().print_module(module)
