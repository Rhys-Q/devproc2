"""devproc2 IR Op definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar, Mapping, Optional

from devproc2.ir.attrs import AttrDict
from devproc2.ir.nodes import (
    DialectKind,
    EffectSummary,
    Op,
    OpResult,
    Region,
    StructInfo,
    TerminatorOp,
    Value,
    Var,
)
from devproc2.ir.op_ref import (
    BuiltinOpRef,
    ExternalFuncRef,
    KernelRef,
    PackedFuncRef,
    StandardOpRef,
)
from devproc2.ir.prim_expr import IntImm, PrimExpr


# ---------------------------------------------------------------------------
# Terminator Ops — must be last Op in a Block
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ReturnOp(TerminatorOp):
    """Function return."""
    dialect: ClassVar[DialectKind] = DialectKind.control
    values: tuple[Value, ...]


@dataclass(frozen=True, eq=False)
class YieldOp(TerminatorOp):
    """Region yield.  values=() means effect-only."""
    dialect: ClassVar[DialectKind] = DialectKind.control
    values: tuple[Value, ...]


# ---------------------------------------------------------------------------
# Compute Ops
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class CallOp(Op):
    """High-level call to a standard, builtin, or external operation.

    result_name=""  → no SSA result (effect-only call).
    result_name="y" → produces one OpResult accessible as results[0].
    result_struct_info optionally propagates type info into the OpResult.
    """
    op_ref:             StandardOpRef | BuiltinOpRef | ExternalFuncRef
    args:               tuple[Value, ...]
    result_name:        str                  = ""
    result_struct_info: Optional[StructInfo] = None
    attrs:              AttrDict | Mapping[str, object] = field(default_factory=AttrDict.empty)

    def __post_init__(self) -> None:
        attrs = self.attrs
        if not isinstance(attrs, AttrDict):
            attrs = AttrDict.from_python(attrs)
        op_def = self.op_def
        if op_def is not None:
            attrs = op_def.normalize_attrs(attrs.to_python_dict(), include_defaults=False)
            if isinstance(self.op_ref, StandardOpRef) and self.op_ref.op_def is None:
                object.__setattr__(self, "op_ref", StandardOpRef(self.op_ref.name, op_def))
        object.__setattr__(self, "attrs", attrs)
        if self.result_name:
            object.__setattr__(self, "results", (
                OpResult(op=self, index=0, struct_info=self.result_struct_info),
            ))

    @property
    def dialect(self) -> DialectKind:
        return self.op_ref.dialect

    @property
    def op_def(self):
        if isinstance(self.op_ref, StandardOpRef):
            return self.op_ref.resolve()
        if isinstance(self.op_ref, BuiltinOpRef):
            return self.op_ref.op_def
        return None

    @property
    def symbol_name(self) -> str:
        return self.op_ref.name


@dataclass(frozen=True, eq=False)
class CallDPSOp(Op):
    """Destination-passing-style call.

    outputs=() means effect-only.  DPS ops define no SSA results.
    """
    dialect:    ClassVar[DialectKind] = DialectKind.runtime
    target_ref: KernelRef | PackedFuncRef | BuiltinOpRef
    inputs:     tuple[Value, ...]
    outputs:    tuple[Value, ...]
    effect:     EffectSummary = field(default_factory=EffectSummary.opaque_call)
    attrs:      AttrDict | Mapping[str, object] = field(default_factory=AttrDict.empty)

    def __post_init__(self) -> None:
        if not isinstance(self.attrs, AttrDict):
            object.__setattr__(self, "attrs", AttrDict.from_python(self.attrs))


class TensorCreateKind(Enum):
    empty      = auto()
    zeros      = auto()
    full       = auto()
    empty_like = auto()


@dataclass(frozen=True, eq=False)
class TensorCreateOp(Op):
    """Allocate / create a tensor buffer."""
    dialect:     ClassVar[DialectKind] = DialectKind.memory
    result_name: str
    kind:        TensorCreateKind
    shape:       tuple[PrimExpr, ...]
    dtype:       str
    device:      str
    fill_value:  Optional[object] = None
    like:        Optional[Var]    = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )
        if self.kind == TensorCreateKind.empty_like:
            if self.like is None:
                raise ValueError("TensorCreateOp(empty_like) requires 'like'")
            if self.shape:
                raise ValueError("TensorCreateOp(empty_like) must not specify 'shape'")
        else:
            if self.like is not None:
                raise ValueError(f"TensorCreateOp({self.kind.name}) must not specify 'like'")
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))


@dataclass(frozen=True, eq=False)
class TupleOp(Op):
    """Construct a tuple value from its elements."""
    dialect:     ClassVar[DialectKind] = DialectKind.tensor
    result_name: str
    elems:       tuple[Value, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))


@dataclass(frozen=True, eq=False)
class TupleGetItemOp(Op):
    """Extract element at `index` from a tuple."""
    dialect:     ClassVar[DialectKind] = DialectKind.tensor
    tup:         Value
    index:       int
    result_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))


# ---------------------------------------------------------------------------
# Control-flow Ops
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Range:
    """Loop bounds used by ForOp."""
    start: Value
    end:   Value
    step:  Value


@dataclass(frozen=True)
class IterArg:
    """One loop-carried variable for ForOp."""
    var:  Var    # block arg inside the loop body
    init: Value  # initial value from outer scope


@dataclass(frozen=True, eq=False)
class IfOp(Op):
    """Structured conditional.

    result_names=()          → effect-only: both branches yield no values.
    result_names=("y", ...)  → SSA results: branches yield matching values.
    """
    cond:         Value
    then_region:  Region
    else_region:  Optional[Region] = None
    result_names: tuple[str, ...]  = ()
    dialect: ClassVar[DialectKind] = DialectKind.control

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(
            OpResult(op=self, index=i) for i in range(len(self.result_names))
        ))


@dataclass(frozen=True, eq=False)
class ForOp(Op):
    """Structured loop over a Range.

    result_names=()          → effect-only loop; body yields nothing.
    result_names=("out", ...) → loop-carried; body yields updated values.
    """
    loop_var:     Var
    range_:       Range
    iter_args:    tuple[IterArg, ...]
    body_region:  Region
    result_names: tuple[str, ...] = ()
    dialect: ClassVar[DialectKind] = DialectKind.control

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(
            OpResult(op=self, index=i) for i in range(len(self.result_names))
        ))


# ---------------------------------------------------------------------------
# Shape assertion Op
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AllocStorageOp(Op):
    """Allocate a raw storage buffer.  Hoisted to function entry by LowerTensorCreateToAllocPass.

    size_bytes is a PrimExpr so it supports both static shapes (IntImm) and
    dynamic shapes (symbolic expressions evaluated at runtime).
    """
    dialect:     ClassVar[DialectKind] = DialectKind.memory
    result_name: str
    size_bytes:  PrimExpr   # IntImm for static; symbolic expr for dynamic
    alignment:   int
    device:      str

    def __post_init__(self) -> None:
        if isinstance(self.size_bytes, int):
            object.__setattr__(self, "size_bytes", IntImm(self.size_bytes))
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))


@dataclass(frozen=True, eq=False)
class AllocTensorOp(Op):
    """Create a tensor view over a storage buffer."""
    dialect:     ClassVar[DialectKind] = DialectKind.memory
    result_name: str
    storage:     Value             # OpResult from AllocStorageOp
    offset:      int               # byte offset; always 0 in MVP
    shape:       tuple[PrimExpr, ...]
    dtype:       str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))


# ---------------------------------------------------------------------------
# Shape assertion Op
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ShapeAssertOp(Op):
    """Runtime assertion: tensor.shape[dim_idx] <= upper."""
    dialect: ClassVar[DialectKind] = DialectKind.shape
    tensor:  Var
    dim_idx: int
    upper:   int
