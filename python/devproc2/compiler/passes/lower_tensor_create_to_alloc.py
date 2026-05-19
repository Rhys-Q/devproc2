"""LowerTensorCreateToAllocPass — replace TensorCreateOp with alloc_storage + alloc_tensor.

Requires: MemoryPlanningPass must have written a StoragePlan to PassContext.

For each function:
  1. Create one AllocStorageOp per StorageEntry (static IntImm or dynamic PrimExpr size).
  2. Hoist all AllocStorageOps to the top of the function entry block.
  3. Replace every TensorCreateOp with an AllocTensorOp referencing its storage.
"""
from __future__ import annotations

from typing import Optional

from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes._rewriter import IRRewriter
from devproc2.compiler.passes.memory_planning import StoragePlan
from devproc2.ir.nodes import (
    Block,
    Function,
    IRModule,
    IRStage,
    Op,
    Region,
    TensorStructInfo,
    shape_values,
)
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    TensorCreateKind,
    TensorCreateOp,
)


class LowerTensorCreateToAllocPass(IRRewriter):
    """Reads StoragePlan from PassContext and rewrites TensorCreateOp."""
    input_stage = IRStage.dps
    output_stage = IRStage.memory
    required_analysis: tuple[str, ...] = ("storage_plan",)
    preserved_analysis: tuple[str, ...] = ()

    def __init__(self, ctx: PassContext) -> None:
        super().__init__()
        self._ctx = ctx
        self._current_fn_name: str = ""
        self._plan: Optional[StoragePlan] = None
        self._alloc_storage_ops: dict[int, AllocStorageOp] = {}

    def run(self, module: IRModule) -> IRModule:
        result: dict[str, Function] = {}
        for name, fn in module.functions.items():
            self._current_fn_name = name
            result[name] = self.rewrite_fn(fn)
        return IRModule(result)

    def rewrite_fn(self, fn: Function) -> Function:
        self._sub = {}
        fn_name = self._current_fn_name
        plan: Optional[StoragePlan] = self._ctx.get(f"storage_plan:{fn_name}")
        if plan is None:
            plan = self._ctx.get("storage_plan")
        if plan is None:
            raise RuntimeError(
                f"LowerTensorCreateToAllocPass: no storage_plan in PassContext "
                f"for function '{fn_name}'.  Run MemoryPlanningPass first."
            )

        # Build one AllocStorageOp per storage entry; size_bytes is a PrimExpr
        # (IntImm for static/bounded shapes, symbolic expr for dynamic shapes).
        alloc_ops: dict[int, AllocStorageOp] = {}
        for entry in plan.entries:
            alloc_ops[entry.id] = AllocStorageOp(
                result_name=f"s{entry.id}",
                size_bytes=entry.size_expr,
                alignment=entry.alignment,
                device=entry.device,
            )

        self._plan = plan
        self._alloc_storage_ops = alloc_ops

        new_body = self.rewrite_region(fn.body)

        # Hoist AllocStorageOps to top of entry block (before all other ops)
        entry = new_body.entry_block
        hoisted = tuple(alloc_ops.values()) + entry.ops
        new_entry = Block(entry.args, hoisted)
        new_body = Region((new_entry,) + new_body.blocks[1:])
        return Function(new_body, fn.ret_struct_info)

    def rewrite_op(self, op: Op) -> Op:
        if isinstance(op, TensorCreateOp):
            return self._lower_tensor_create(op)
        return self._subst_op(op)

    def _lower_tensor_create(self, op: TensorCreateOp) -> AllocTensorOp:
        name = op.result_name
        storage_id = self._plan.tensor_to_storage[name]
        storage_op = self._alloc_storage_ops[storage_id]
        shape, dtype = _resolve_shape_dtype(op)
        new_op = AllocTensorOp(
            result_name=name,
            storage=storage_op.results[0],
            offset=0,
            shape=shape,
            dtype=dtype,
        )
        return new_op


def _resolve_shape_dtype(op: TensorCreateOp):
    if op.kind == TensorCreateKind.empty_like:
        si = op.like.struct_info if op.like is not None else None
        if isinstance(si, TensorStructInfo):
            return shape_values(si.shape), si.dtype
        raise ValueError(
            f"TensorCreateOp(empty_like) '{op.result_name}': "
            "cannot resolve shape; ensure InferStructInfoPass ran first"
        )
    return op.shape, op.dtype
