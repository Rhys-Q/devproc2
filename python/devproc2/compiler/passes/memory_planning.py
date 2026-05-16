"""MemoryPlanningPass — analyse tensor lifetimes and plan storage reuse.

Analysis-only: does NOT modify the IRModule.  Writes a StoragePlan to
PassContext under key "storage_plan" (single-function modules) and
"storage_plan:<fn_name>" for each function.

Internal pipeline (run per function):
  A. TensorCreateAnalyze  — collect TensorCreateOps, mark non-reusable
  B. LifetimeAnalyze      — linearize IR, compute LiveInterval per tensor
  C. StorageSizeAnalyze   — compute max_bytes using UpperBound for PrimVars
  D. StoragePlan (greedy) — assign tensors to storage blocks
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import prod
from typing import Optional

from devproc2.compiler.pass_context import PassContext
from devproc2.utils.dtype import dtype_itemsize
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
    prim_expr_structural_eq,
)
from devproc2.ir.analysis import AliasAnalysis, iter_ops
from devproc2.ir.nodes import (
    Function,
    IRModule,
    IRStage,
    OpResult,
    Region,
    TensorStructInfo,
    Value,
    shape_values,
)
from devproc2.ir.ops import (
    CallDPSOp,
    Op,
    ReturnOp,
    TensorCreateKind,
    TensorCreateOp,
)

_MAX_INT = sys.maxsize


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class LiveInterval:
    first_def: int
    last_use:  int

    def overlaps(self, other: LiveInterval) -> bool:
        return self.first_def <= other.last_use and other.first_def <= self.last_use


@dataclass
class TensorInfo:
    name:        str
    device:      str
    shape:       tuple[PrimExpr, ...]
    dtype:       str
    size_bytes:  Optional[int]  # None when any dim has no upper bound
    size_expr:   PrimExpr       # always set; used as AllocStorageOp.size_bytes
    interval:    LiveInterval
    is_reusable: bool


@dataclass
class StorageEntry:
    id:         int
    device:     str
    size_bytes: Optional[int]   # None for dynamic entries (no reuse comparison)
    size_expr:  PrimExpr        # passed to AllocStorageOp
    alignment:  int = 256
    reused_by:  list[str] = field(default_factory=list)
    _intervals: list[LiveInterval] = field(default_factory=list, repr=False)

    def accepts(self, ti: TensorInfo) -> bool:
        if ti.device != self.device:
            return False
        if self.size_bytes is not None and ti.size_bytes is not None:
            # Both static: entry must be at least as large as the incoming tensor.
            if self.size_bytes < ti.size_bytes:
                return False
        elif self.size_bytes is None and ti.size_bytes is None:
            # Both dynamic: require structurally equal size expressions.
            # prim_expr_structural_eq compares PrimVars by (name, upper) rather
            # than object identity, so this works even if a pass reconstructed
            # PrimVar objects with the same semantic content.
            if not prim_expr_structural_eq(self.size_expr, ti.size_expr):
                return False
        else:
            # Mixed static/dynamic: cannot guarantee compatibility, skip.
            return False
        return all(not ti.interval.overlaps(iv) for iv in self._intervals)


@dataclass
class StoragePlan:
    entries:           list[StorageEntry]
    tensor_to_storage: dict[str, int]  # result_name → entry.id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_upper(expr: PrimExpr) -> int:
    """Evaluate PrimExpr using upper bounds for PrimVars (conservative).

    Raises _NoBound if any PrimVar in the expression has no upper bound.
    """
    if isinstance(expr, IntImm):
        return expr.value
    if isinstance(expr, PrimVar):
        if expr.upper is None:
            raise _NoBound(expr.name)
        return expr.upper
    if isinstance(expr, Add):
        return _eval_upper(expr.lhs) + _eval_upper(expr.rhs)
    if isinstance(expr, Sub):
        return _eval_upper(expr.lhs) - _eval_upper(expr.rhs)
    if isinstance(expr, Mul):
        return _eval_upper(expr.lhs) * _eval_upper(expr.rhs)
    if isinstance(expr, FloorDiv):
        return _eval_upper(expr.lhs) // max(1, _eval_upper(expr.rhs))
    if isinstance(expr, CeilDiv):
        b = max(1, _eval_upper(expr.rhs))
        return (_eval_upper(expr.lhs) + b - 1) // b
    if isinstance(expr, Min):
        return min(_eval_upper(expr.lhs), _eval_upper(expr.rhs))
    if isinstance(expr, Max):
        return max(_eval_upper(expr.lhs), _eval_upper(expr.rhs))
    raise NotImplementedError(f"Cannot evaluate upper bound for {type(expr).__name__}")


class _NoBound(Exception):
    """Raised by _eval_upper when a PrimVar has no upper bound."""


def _align256(n: int) -> int:
    return ((n + 255) // 256) * 256


def _compute_size_bytes(shape: tuple[PrimExpr, ...], dtype: str) -> Optional[int]:
    """Return statically aligned byte size, or None if any dim has no upper bound.

    None signals that the tensor size cannot be determined at compile time;
    the tensor is still lowered to alloc_storage + alloc_tensor but is not
    eligible for storage reuse.
    """
    try:
        nbytes = prod(_eval_upper(d) for d in shape) * dtype_itemsize(dtype)
        return _align256(max(nbytes, 1))
    except _NoBound:
        return None


def _compute_size_expr(shape: tuple[PrimExpr, ...], dtype: str) -> PrimExpr:
    """Build a PrimExpr for the byte size without upper-bound substitution.

    The resulting expression is evaluated at runtime by the VM.  Alignment
    is communicated separately via AllocStorageOp.alignment; the VM handles
    it when calling DeviceAPI::Alloc.
    """
    itemsize = dtype_itemsize(dtype)
    result: PrimExpr = IntImm(itemsize)
    for dim in shape:
        result = Mul(dim, result)
    return result


# ---------------------------------------------------------------------------
# IR traversal utilities
# ---------------------------------------------------------------------------

def _collect_ops_linear(region: Region, out: list[Op]) -> None:
    """DFS linearisation: appends every Op (including nested-region ops)."""
    out.extend(iter_ops(region))


def _collect_return_values(
    all_ops: list[Op],
    create_results: set[OpResult],
    aliases: AliasAnalysis,
) -> set[OpResult]:
    """Return TensorCreateOp result ids that flow into function returns.

    Return values may be wrapped in TupleOp or produced by structured control
    flow.  For example, `return if_result` must keep the tensors yielded from
    the selected IfOp branch alive for the caller.  This function resolves those
    value aliases back to the underlying TensorCreateOp results.
    """
    results: set[OpResult] = set()
    for op in all_ops:
        if isinstance(op, ReturnOp):
            for v in op.values:
                results.update(_resolve_create_results(v, create_results, aliases))
    return results


def _operand_results(op: Op) -> list[OpResult]:
    """Collect all OpResult operands referenced by public op semantics."""
    refs: list[OpResult] = []

    def _add(v: Value) -> None:
        if isinstance(v, OpResult):
            refs.append(v)

    effect = op.effects
    for value in op.operands + effect.reads + effect.writes:
        _add(value)
    return refs


def _resolve_create_results(
    value: Value,
    create_results: set[OpResult],
    aliases: AliasAnalysis,
) -> frozenset[OpResult]:
    return frozenset(
        result
        for result in aliases.resolve_matching(value, lambda v: v in create_results)
        if isinstance(result, OpResult)
    )


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

class MemoryPlanningPass:
    """Analysis-only pass; writes StoragePlan to PassContext."""
    input_stage = IRStage.dps
    output_stage = IRStage.dps
    required_analysis: tuple[str, ...] = ()
    preserved_analysis: tuple[str, ...] = ("storage_plan",)

    def run(self, module: IRModule, ctx: PassContext) -> IRModule:
        for fn_name, fn in module.functions.items():
            plan = self._plan_function(fn)
            ctx.put(f"storage_plan:{fn_name}", plan)
        if len(module.functions) == 1:
            fn_name = next(iter(module.functions))
            ctx.put("storage_plan", ctx.get(f"storage_plan:{fn_name}"))
        return module  # unchanged

    # ------------------------------------------------------------------
    # Per-function planning
    # ------------------------------------------------------------------

    def _plan_function(self, fn: Function) -> StoragePlan:
        # Phase A: collect TensorCreateOps
        all_ops: list[Op] = []
        _collect_ops_linear(fn.body, all_ops)

        create_ops: list[TensorCreateOp] = [
            op for op in all_ops if isinstance(op, TensorCreateOp)
        ]

        _check_unique_tensor_create_names(create_ops)

        # Phase A: identify non-reusable (output) tensors
        create_results = {cop.results[0] for cop in create_ops}
        aliases = AliasAnalysis(all_ops)
        return_result_ids = _collect_return_values(
            all_ops, create_results, aliases
        )

        # Phase B: build op→index map and compute live intervals
        op_index: dict[int, int] = {id(op): i for i, op in enumerate(all_ops)}

        # result_id → defining TensorCreateOp result_name
        result_to_name: dict[OpResult, str] = {}
        for cop in create_ops:
            result_to_name[cop.results[0]] = cop.result_name

        # first_def per tensor (index of TensorCreateOp in linear order)
        first_def: dict[str, int] = {}
        for cop in create_ops:
            first_def[cop.result_name] = op_index[id(cop)]

        # Phase B step 1: explicit last_use from data flow only.
        explicit_last_use: dict[str, int] = {n: first_def[n] for n in first_def}
        for op in all_ops:
            idx = op_index[id(op)]
            for ref in _operand_results(op):
                for result in _resolve_create_results(ref, create_results, aliases):
                    name = result_to_name[result]
                    explicit_last_use[name] = max(explicit_last_use[name], idx)

        # Phase B step 2: apply effect-based extensions
        last_use: dict[str, int] = dict(explicit_last_use)
        for op in all_ops:
            idx = op_index[id(op)]
            if isinstance(op, CallDPSOp):
                if op.effect.reads:
                    for value in op.effect.reads:
                        for result in _resolve_create_results(
                            value, create_results, aliases
                        ):
                            name = result_to_name[result]
                            last_use[name] = max(last_use[name], idx)
                if op.effect.writes:
                    for value in op.effect.writes:
                        for result in _resolve_create_results(
                            value, create_results, aliases
                        ):
                            name = result_to_name[result]
                            last_use[name] = max(last_use[name], idx)
                if op.effect.opaque:
                    # Conservative: extend tensors that are ALREADY live at this point.
                    # "Already live" = defined before AND last-used (explicitly) at or after
                    # this index.  This avoids resurrecting tensors that have already been
                    # consumed, which would destroy all reuse opportunities.
                    for name in last_use:
                        if (first_def[name] <= idx
                                and explicit_last_use[name] >= idx):
                            last_use[name] = max(last_use[name], idx)

        # Phase C: compute sizes; build TensorInfo list
        tensor_infos: list[TensorInfo] = []
        for cop in create_ops:
            name = cop.result_name
            shape, dtype, device = _get_shape_dtype_device(cop, fn)
            size_bytes = _compute_size_bytes(shape, dtype)   # None if unbounded
            size_expr = (IntImm(size_bytes) if size_bytes is not None
                         else _compute_size_expr(shape, dtype))
            is_ret = (cop.results[0] in return_result_ids)
            # Not reusable only if the tensor is returned (must stay alive for
            # the caller).  Dynamic (unbounded) tensors are reusable as long as
            # they are not returned — two dynamic tensors with the same size_expr
            # and non-overlapping intervals can share storage.
            is_reusable = not is_ret
            interval = LiveInterval(first_def[name], last_use[name])
            if not is_ret:
                is_reusable = True
            else:
                is_reusable = False
            tensor_infos.append(TensorInfo(
                name=name,
                device=device,
                shape=shape,
                dtype=dtype,
                size_bytes=size_bytes,
                size_expr=size_expr,
                interval=interval,
                is_reusable=is_reusable,
            ))

        # Phase D: greedy storage assignment
        return _greedy_plan(tensor_infos)


def _get_shape_dtype_device(
    cop: TensorCreateOp, fn: Function
) -> tuple[tuple[PrimExpr, ...], str, str]:
    if cop.kind == TensorCreateKind.empty_like:
        si = cop.like.struct_info if cop.like is not None else None
        if isinstance(si, TensorStructInfo):
            return shape_values(si.shape), si.dtype, si.device
        raise ValueError(
            f"TensorCreateOp(empty_like) '{cop.result_name}': "
            "cannot determine shape; ensure InferStructInfoPass ran first"
        )
    return cop.shape, cop.dtype, cop.device


def _check_unique_tensor_create_names(create_ops: list[TensorCreateOp]) -> None:
    seen: set[str] = set()
    for op in create_ops:
        if op.result_name in seen:
            raise ValueError(
                "MemoryPlanningPass requires unique TensorCreateOp result_name "
                f"within a function, got duplicate {op.result_name!r}"
            )
        seen.add(op.result_name)


def _greedy_plan(tensors: list[TensorInfo]) -> StoragePlan:
    entries: list[StorageEntry] = []
    tensor_to_storage: dict[str, int] = {}
    next_id = 0

    for ti in sorted(tensors, key=lambda t: t.interval.first_def):
        if not ti.is_reusable:
            entry = StorageEntry(
                id=next_id, device=ti.device,
                size_bytes=ti.size_bytes, size_expr=ti.size_expr,
            )
            entry.reused_by.append(ti.name)
            entry._intervals.append(ti.interval)
            entries.append(entry)
            tensor_to_storage[ti.name] = entry.id
            next_id += 1
        else:
            candidate = next((e for e in entries if e.accepts(ti)), None)
            if candidate is not None:
                candidate.reused_by.append(ti.name)
                candidate._intervals.append(ti.interval)
                tensor_to_storage[ti.name] = candidate.id
            else:
                entry = StorageEntry(
                    id=next_id, device=ti.device,
                    size_bytes=ti.size_bytes, size_expr=ti.size_expr,
                )
                entry.reused_by.append(ti.name)
                entry._intervals.append(ti.interval)
                entries.append(entry)
                tensor_to_storage[ti.name] = entry.id
                next_id += 1

    return StoragePlan(entries=entries, tensor_to_storage=tensor_to_storage)
