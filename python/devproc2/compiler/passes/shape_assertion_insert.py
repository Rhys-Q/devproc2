"""ShapeAssertionInsertPass — insert ShapeAssertOp at function entry."""
from __future__ import annotations

from devproc2.ir.nodes import Block, Function, IRModule, IRStage, Region, TensorStructInfo
from devproc2.ir.ops import ShapeAssertOp
from devproc2.ir.prim_expr import PrimVar


class ShapeAssertionInsertPass:
    input_stage = IRStage.inferred
    output_stage = IRStage.inferred
    required_analysis: tuple[str, ...] = ()
    preserved_analysis: tuple[str, ...] = ()

    def run(self, module: IRModule) -> IRModule:
        return IRModule({name: self._insert_fn(fn) for name, fn in module.functions.items()})

    def _insert_fn(self, fn: Function) -> Function:
        assert_ops: list[ShapeAssertOp] = []
        for param in fn.params:
            if isinstance(param.struct_info, TensorStructInfo):
                for idx, dim in enumerate(param.struct_info.shape):
                    if isinstance(dim, PrimVar) and dim.upper is not None:
                        assert_ops.append(ShapeAssertOp(tensor=param, dim_idx=idx, upper=dim.upper))
        if not assert_ops:
            return fn
        entry = fn.body.entry_block
        new_ops = tuple(assert_ops) + entry.ops
        new_block = Block(entry.args, new_ops)
        new_body = Region((new_block,) + fn.body.blocks[1:])
        return Function(new_body, fn.ret_struct_info)
