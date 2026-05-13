from __future__ import annotations

from io import StringIO
from typing import Optional

from devproc2.ir.nodes import (
    Block,
    Constant,
    EffectInfo,
    Function,
    IRModule,
    OpaqueEffect,
    Op,
    OpResult,
    PureEffect,
    ReadOnlyEffect,
    Region,
    StructInfo,
    TensorStructInfo,
    TerminatorOp,
    Value,
    Var,
    WriteEffect,
)
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    CallDPSOp,
    CallOp,
    ForOp,
    IfOp,
    IterArg,
    Range,
    ReturnOp,
    ShapeAssertOp,
    TensorCreateKind,
    TensorCreateOp,
    TupleGetItemOp,
    TupleOp,
    YieldOp,
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


def _op_result_name(op: Op, index: int) -> Optional[str]:
    """Return the user-given name for result[index], or None for auto-numbering."""
    if isinstance(op, (CallOp, TensorCreateOp, TupleOp, TupleGetItemOp,
                       AllocStorageOp, AllocTensorOp)):
        name = op.result_name
        return name if name else None
    if isinstance(op, (IfOp, ForOp)):
        if index < len(op.result_names):
            return op.result_names[index]
    return None


class Printer:
    def __init__(self) -> None:
        self._buf: StringIO = StringIO()
        self._result_names: dict[int, str] = {}   # id(OpResult) → display name
        self._auto_idx: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_module(self, module: IRModule) -> str:
        self._buf = StringIO()
        for i, (name, fn) in enumerate(module.functions.items()):
            if i > 0:
                self._buf.write("\n")
            self._print_function(name, fn)
        return self._buf.getvalue()

    # ------------------------------------------------------------------
    # Function
    # ------------------------------------------------------------------

    def _print_function(self, name: str, fn: Function) -> None:
        # Reset name table per function
        self._result_names = {}
        self._auto_idx = 0

        params_str = ", ".join(self._param_str(p) for p in fn.params)
        ret_str = ""
        if fn.ret_struct_info is not None:
            ret_str = f" -> {self.print_struct_info(fn.ret_struct_info)}"
        self._buf.write(f"@{name}({params_str}){ret_str} {{\n")
        entry = fn.body.entry_block
        for op in entry.ops:
            self._print_op(op, indent="  ")
        self._buf.write("}\n")

    def _param_str(self, v: Var) -> str:
        if v.struct_info is not None:
            return f"%{v.name}: {self.print_struct_info(v.struct_info)}"
        return f"%{v.name}"

    # ------------------------------------------------------------------
    # Region / Block (for nested CF regions)
    # ------------------------------------------------------------------

    def _print_region(self, region: Region, indent: str) -> None:
        for block in region.blocks:
            for op in block.ops:
                self._print_op(op, indent)

    # ------------------------------------------------------------------
    # Result name registration
    # ------------------------------------------------------------------

    def _register_op_results(self, op: Op) -> None:
        """Assign display names for all results of op before printing it."""
        for r in op.results:
            name = _op_result_name(op, r.index)
            if not name:
                name = str(self._auto_idx)
                self._auto_idx += 1
            self._result_names[id(r)] = name

    # ------------------------------------------------------------------
    # Op dispatch
    # ------------------------------------------------------------------

    def _print_op(self, op: Op, indent: str) -> None:
        self._register_op_results(op)

        if isinstance(op, ReturnOp):
            if len(op.values) == 1:
                self._buf.write(f"{indent}return {self._value_str(op.values[0])}\n")
            else:
                vals = ", ".join(self._value_str(v) for v in op.values)
                self._buf.write(f"{indent}return ({vals})\n")

        elif isinstance(op, YieldOp):
            if op.values:
                vals = ", ".join(self._value_str(v) for v in op.values)
                self._buf.write(f"{indent}yield {vals}\n")
            else:
                self._buf.write(f"{indent}yield\n")

        elif isinstance(op, CallOp):
            args_str = ", ".join(self._value_str(a) for a in op.args)
            expr = f"{op.callee}({args_str})"
            if op.results:
                rname = self._result_names[id(op.results[0])]
                self._buf.write(f"{indent}%{rname} = {expr}\n")
            else:
                self._buf.write(f"{indent}{expr}\n")

        elif isinstance(op, CallDPSOp):
            self._print_calldps(op, indent)

        elif isinstance(op, TensorCreateOp):
            rname = self._result_names[id(op.results[0])]
            self._buf.write(f"{indent}%{rname} = {self._tensor_create_str(op)}\n")

        elif isinstance(op, TupleOp):
            rname = self._result_names[id(op.results[0])]
            elems = "(" + ", ".join(self._value_str(e) for e in op.elems) + ")"
            self._buf.write(f"{indent}%{rname} = {elems}\n")

        elif isinstance(op, TupleGetItemOp):
            rname = self._result_names[id(op.results[0])]
            self._buf.write(
                f"{indent}%{rname} = {self._value_str(op.tup)}[{op.index}]\n"
            )

        elif isinstance(op, IfOp):
            self._print_if(op, indent)

        elif isinstance(op, ForOp):
            self._print_for(op, indent)

        elif isinstance(op, ShapeAssertOp):
            self._buf.write(
                f"{indent}assert %{op.tensor.name}.shape[{op.dim_idx}] <= {op.upper}\n"
            )

        elif isinstance(op, AllocStorageOp):
            rname = self._result_names[id(op.results[0])]
            self._buf.write(
                f"{indent}%{rname} = alloc_storage("
                f"size={self.print_prim_expr(op.size_bytes)}, "
                f"alignment={op.alignment}, device={op.device})\n"
            )

        elif isinstance(op, AllocTensorOp):
            rname = self._result_names[id(op.results[0])]
            shape_str = "(" + ", ".join(self.print_prim_expr(s) for s in op.shape) + ")"
            self._buf.write(
                f"{indent}%{rname} = alloc_tensor("
                f"{self._value_str(op.storage)}, offset={op.offset}, "
                f"shape={shape_str}, dtype={op.dtype})\n"
            )

        else:
            self._buf.write(f"{indent}{repr(op)}\n")

    # ------------------------------------------------------------------
    # CallDPS
    # ------------------------------------------------------------------

    def _print_calldps(self, op: CallDPSOp, indent: str) -> None:
        inner = indent + "  "
        inputs_str = "[" + ", ".join(self._value_str(a) for a in op.inputs) + "]"
        output_str = self._value_str(op.output) if op.output is not None else "None"
        self._buf.write(f"{indent}call_dps {op.callee}(\n")
        self._buf.write(f"{inner}inputs={inputs_str},\n")
        self._buf.write(f"{inner}output={output_str},\n")
        self._buf.write(f"{inner}callee_kind={op.callee_kind.name},\n")
        self._buf.write(f"{inner}effect={self._effect_str(op.effect)}\n")
        self._buf.write(f"{indent})\n")

    # ------------------------------------------------------------------
    # IfOp
    # ------------------------------------------------------------------

    def _print_if(self, op: IfOp, indent: str) -> None:
        inner = indent + "  "
        if op.results:
            rnames = " ".join(f"%{self._result_names[id(r)]}" for r in op.results)
            prefix = f"{indent}{rnames} = "
        else:
            prefix = indent
        self._buf.write(f"{prefix}if {self._value_str(op.cond)} {{\n")
        self._print_region(op.then_region, inner)
        self._buf.write(f"{indent}}}")
        if op.else_region is not None:
            self._buf.write(" else {\n")
            self._print_region(op.else_region, inner)
            self._buf.write(f"{indent}}}")
        self._buf.write("\n")

    # ------------------------------------------------------------------
    # ForOp
    # ------------------------------------------------------------------

    def _print_for(self, op: ForOp, indent: str) -> None:
        inner = indent + "  "
        if op.results:
            rnames = " ".join(f"%{self._result_names[id(r)]}" for r in op.results)
            prefix = f"{indent}{rnames} = "
        else:
            prefix = indent
        r = op.range_
        range_str = (
            f"range({self._value_str(r.start)}, "
            f"{self._value_str(r.end)}, "
            f"{self._value_str(r.step)})"
        )
        iter_str = ""
        if op.iter_args:
            parts = [f"%{ia.var.name} = {self._value_str(ia.init)}" for ia in op.iter_args]
            iter_str = " iter_args(" + ", ".join(parts) + ")"
        self._buf.write(f"{prefix}for %{op.loop_var.name} in {range_str}{iter_str} {{\n")
        self._print_region(op.body_region, inner)
        self._buf.write(f"{indent}}}\n")

    # ------------------------------------------------------------------
    # TensorCreate
    # ------------------------------------------------------------------

    def _tensor_create_str(self, op: TensorCreateOp) -> str:
        kind_map = {
            TensorCreateKind.empty:      "dp.empty",
            TensorCreateKind.zeros:      "dp.zeros",
            TensorCreateKind.full:       "dp.full",
            TensorCreateKind.empty_like: "dp.empty_like",
        }
        name = kind_map[op.kind]
        if op.kind == TensorCreateKind.empty_like:
            return f"{name}({self._value_str(op.like)})"
        shape_str = "(" + ", ".join(self.print_prim_expr(s) for s in op.shape) + ")"
        result = f"{name}(shape={shape_str}, dtype={op.dtype}, device={op.device}"
        if op.kind == TensorCreateKind.full and op.fill_value is not None:
            result += f", fill_value={op.fill_value!r}"
        result += ")"
        return result

    # ------------------------------------------------------------------
    # Value / StructInfo / PrimExpr / Effect helpers
    # ------------------------------------------------------------------

    def _value_str(self, v: Value) -> str:
        if isinstance(v, Var):
            return f"%{v.name}"
        if isinstance(v, OpResult):
            return f"%{self._result_names[id(v)]}"
        if isinstance(v, Constant):
            return repr(v.value)
        return repr(v)

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
