"""ShapeConstraintVerifyPass — validate shape assertions against concrete bindings.

Given a dict {dim_name: concrete_int}, checks that no ShapeAssertOp is
violated. Raises RuntimeShapeError on the first violation found.
"""
from __future__ import annotations

from devproc2.ir.nodes import Function, IRModule, TensorStructInfo
from devproc2.ir.ops import ShapeAssertOp
from devproc2.ir.prim_expr import PrimVar


class RuntimeShapeError(Exception):
    pass


class ShapeConstraintVerifyPass:
    def run(self, module: IRModule, bindings: dict[str, int] | None = None) -> IRModule:
        if bindings:
            for name, fn in module.functions.items():
                self._check_fn(name, fn, bindings)
        return module

    def _check_fn(self, fn_name: str, fn: Function, bindings: dict[str, int]) -> None:
        for block in fn.body.blocks:
            for op in block.ops:
                if isinstance(op, ShapeAssertOp):
                    dim_name = self._dim_name(fn, op)
                    if dim_name and dim_name in bindings:
                        val = bindings[dim_name]
                        if val > op.upper:
                            raise RuntimeShapeError(
                                f"In @{fn_name}: shape dim '{dim_name}' = {val} "
                                f"exceeds upper bound {op.upper}"
                            )

    def _dim_name(self, fn: Function, op: ShapeAssertOp) -> str | None:
        for p in fn.params:
            if p.name == op.tensor.name and isinstance(p.struct_info, TensorStructInfo):
                dim = p.struct_info.shape[op.dim_idx]
                if isinstance(dim, PrimVar):
                    return dim.name
        return None
