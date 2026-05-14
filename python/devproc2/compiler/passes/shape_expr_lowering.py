"""ShapeExprLoweringPass — materialize PrimExpr trees into VM builtin calls.

Used as a service class by VMCodegenPass.  At function-entry codegen time,
ShapeExprLoweringPass.setup_fn() extracts PrimVar values from tensor
parameters (via vm.builtin.shape_of + vm.builtin.get_shape_dim) and emits
vm.builtin.assert_le_i64 checks for PrimVars with upper bounds.

After setup_fn, the returned _PrimExprLowerer can materialize any PrimExpr
(including compound expressions like ceildiv(S, 16)) into a VM register.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from devproc2.ir.nodes import Function, TensorStructInfo
from devproc2.ir.prim_expr import (
    Add,
    CeilDiv,
    FloorDiv,
    IntImm,
    Max,
    Min,
    Mul,
    PrimExpr,
    PrimVar,
    Sub,
)
from devproc2.vm.executable import Instruction, Opcode

if TYPE_CHECKING:
    from devproc2.compiler.passes.vm_codegen import _FnCtx

_BINOP_BUILTINS: dict[type, str] = {
    Add:      "vm.builtin.add_i64",
    Sub:      "vm.builtin.sub_i64",
    Mul:      "vm.builtin.mul_i64",
    FloorDiv: "vm.builtin.floordiv_i64",
    CeilDiv:  "vm.builtin.ceildiv_i64",
    Min:      "vm.builtin.min_i64",
    Max:      "vm.builtin.max_i64",
}


class _PrimExprLowerer:
    """Recursively materializes PrimExpr trees into VM register indices."""

    def __init__(self, ctx: _FnCtx) -> None:
        self._ctx = ctx
        self._var_reg: dict[int, int] = {}  # id(PrimVar) → register index

    def bind_var(self, pvar: PrimVar, reg: int) -> None:
        self._var_reg[id(pvar)] = reg

    def materialize(self, expr: PrimExpr) -> int:
        ctx = self._ctx
        if isinstance(expr, IntImm):
            return ctx.reg_for_int(expr.value)
        if isinstance(expr, PrimVar):
            reg = self._var_reg.get(id(expr))
            if reg is None:
                raise RuntimeError(
                    f"PrimVar '{expr.name}' not bound to a VM register; "
                    "ensure ShapeExprLoweringPass.setup_fn ran before materialization"
                )
            return reg
        # Binary operators
        builtin_name = _BINOP_BUILTINS.get(type(expr))
        if builtin_name is None:
            raise NotImplementedError(
                f"_PrimExprLowerer: unsupported PrimExpr node {type(expr).__name__}"
            )
        lhs_reg = self.materialize(expr.lhs)   # type: ignore[union-attr]
        rhs_reg = self.materialize(expr.rhs)   # type: ignore[union-attr]
        result_reg = ctx.alloc_reg()
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=result_reg,
            func_idx=ctx.builtin(builtin_name),
            arg_regs=[lhs_reg, rhs_reg],
        ))
        return result_reg


class ShapeExprLoweringPass:
    """Service class that sets up PrimVar → register bindings at function entry.

    Call setup_fn(fn, ctx) before codegen-ing the function body.  It will:
    1. For each tensor parameter with PrimVar shape dimensions:
       - Emit vm.builtin.shape_of to extract the shape tuple.
       - Emit vm.builtin.get_shape_dim for each unseen PrimVar dimension.
       - Register the dim value in ctx._value_reg (for VMCodegenPass compatibility).
    2. For each PrimVar with an upper bound, emit vm.builtin.assert_le_i64.
    3. Return a _PrimExprLowerer that can materialize any PrimExpr.
    """

    @staticmethod
    def setup_fn(fn: Function, ctx: _FnCtx) -> _PrimExprLowerer:
        lowerer = _PrimExprLowerer(ctx)
        seen: set[int] = set()

        for param in fn.params:
            si = param.struct_info
            if not isinstance(si, TensorStructInfo):
                continue

            # Check whether this param has any unseen PrimVar dims.
            unseen_vars = [
                (idx, dim)
                for idx, dim in enumerate(si.shape)
                if isinstance(dim, PrimVar) and id(dim) not in seen
            ]
            if not unseen_vars:
                continue

            # Extract the shape tuple from this tensor parameter.
            param_reg = ctx.reg_of(param)
            shape_reg = ctx.alloc_reg()
            ctx.emit(Instruction(
                opcode=Opcode.CALL,
                dst_reg=shape_reg,
                func_idx=ctx.builtin("vm.builtin.shape_of"),
                arg_regs=[param_reg],
            ))

            for idx, dim in unseen_vars:
                seen.add(id(dim))
                idx_reg = ctx.reg_for_int(idx)
                dim_reg = ctx.alloc_reg()
                ctx.emit(Instruction(
                    opcode=Opcode.CALL,
                    dst_reg=dim_reg,
                    func_idx=ctx.builtin("vm.builtin.get_shape_dim"),
                    arg_regs=[shape_reg, idx_reg],
                ))
                lowerer.bind_var(dim, dim_reg)
                # Keep VMCodegenPass's existing PrimVar-in-_value_reg path working.
                ctx._value_reg[id(dim)] = dim_reg

                if dim.upper is not None:
                    bound_reg = ctx.reg_for_int(dim.upper)
                    msg = f"{dim.name} exceeds upper bound {dim.upper}"
                    msg_reg = ctx._reg_for_const(msg)
                    ctx.emit(Instruction(
                        opcode=Opcode.CALL,
                        dst_reg=-1,
                        func_idx=ctx.builtin("vm.builtin.assert_le_i64"),
                        arg_regs=[dim_reg, bound_reg, msg_reg],
                    ))

        return lowerer
