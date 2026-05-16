from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Iterator, Optional, Union

from devproc2.ir.prim_expr import IntImm, PrimExpr


# ---------------------------------------------------------------------------
# Value hierarchy — operands passed to Ops
# ---------------------------------------------------------------------------

class Value:
    """Base class for Op operands (Var, OpResult, or Constant)."""


@dataclass(frozen=True, eq=False)
class Var(Value):
    """Block argument — defined at block entry (function params, iter args)."""
    name: str
    struct_info: Optional[StructInfo] = None


@dataclass(frozen=True)
class Constant(Value):
    """Compile-time scalar or tensor constant (int, float, bool, or None).

    Not an SSA def — inlined directly into Op operands with no name binding.
    For tensor constants use a dedicated ConstantTensorOp (future work).
    """
    value: Union[int, float, bool, None]


@dataclass(frozen=True, eq=False)
class OpResult(Value):
    """SSA value produced by an Op.

    op.results[index] gives this value.  The name for printing is stored
    on the defining Op (result_name / result_names), not here.

    Design note: OpResult holds a strong back-reference to its defining Op,
    and Op.results holds all its OpResults — a deliberate cyclic reference.
    Python's cyclic GC handles this; the cycle is broken when the IR tree
    is no longer reachable.
    """
    op: Op
    index: int
    struct_info: Optional[StructInfo] = None


# ---------------------------------------------------------------------------
# StructInfo — type + shape metadata attached to Vars / OpResults
# ---------------------------------------------------------------------------

class StructInfo:
    """Base class for structural type info (shape + dtype + device)."""


class ShapeInfo:
    """Base class for tensor shape descriptors."""


@dataclass(frozen=True)
class KnownShape(ShapeInfo):
    values: tuple[PrimExpr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.values),
        )

    def __iter__(self) -> Iterator[PrimExpr]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __bool__(self) -> bool:
        return bool(self.values)

    def __add__(self, other):
        return self.values + tuple(other)

    def __radd__(self, other):
        return tuple(other) + self.values

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KnownShape):
            return self.values == other.values
        if isinstance(other, tuple):
            return self.values == other
        return False

    def __hash__(self) -> int:
        return hash(self.values)


@dataclass(frozen=True)
class UnknownShape(ShapeInfo):
    ndim: Optional[int] = None

    def __iter__(self) -> Iterator[PrimExpr]:
        raise TypeError("UnknownShape has no concrete dimensions")

    def __len__(self) -> int:
        if self.ndim is None:
            raise TypeError("UnknownShape rank is unknown")
        return self.ndim

    def __bool__(self) -> bool:
        return self.ndim not in (None, 0)


@dataclass(frozen=True)
class TensorStructInfo(StructInfo):
    shape: ShapeInfo | tuple[PrimExpr, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        if isinstance(self.shape, (KnownShape, UnknownShape)):
            return
        object.__setattr__(self, "shape", KnownShape(tuple(self.shape)))


@dataclass(frozen=True)
class ScalarStructInfo(StructInfo):
    dtype: str


@dataclass(frozen=True)
class ObjectStructInfo(StructInfo):
    type_key: str
    role: Optional[str] = None


@dataclass(frozen=True)
class ShapeStructInfo(StructInfo):
    ndim: Optional[int] = None
    values: ShapeInfo | None = None


@dataclass(frozen=True)
class TupleStructInfo(StructInfo):
    fields: tuple[StructInfo, ...]


@dataclass(frozen=True)
class FuncStructInfo(StructInfo):
    params: tuple[StructInfo, ...]
    ret: StructInfo | None
    effects: "EffectSummary | None" = None


# ---------------------------------------------------------------------------
# Effect / alias model
# ---------------------------------------------------------------------------

class AliasKind(Enum):
    no_alias = "no_alias"
    may_alias = "may_alias"
    must_alias = "must_alias"
    view_of = "view_of"


@dataclass(frozen=True)
class AliasInfo:
    kind: AliasKind
    source: Value | None = None


@dataclass(frozen=True)
class EffectSummary:
    """Side-effect summary attached to runtime and external operations."""

    reads: tuple[Value, ...] = ()
    writes: tuple[Value, ...] = ()
    allocates: bool = False
    frees: bool = False
    opaque: bool = False
    external_state: Optional[str] = None

    @classmethod
    def pure(cls) -> "EffectSummary":
        return cls()

    @classmethod
    def readonly(cls, *values: Value) -> "EffectSummary":
        return cls(reads=tuple(values))

    @classmethod
    def write(cls, *values: Value) -> "EffectSummary":
        return cls(writes=tuple(values))

    @classmethod
    def opaque_call(cls, external_state: str | None = None) -> "EffectSummary":
        return cls(opaque=True, external_state=external_state)

    @property
    def is_pure(self) -> bool:
        return (
            not self.reads
            and not self.writes
            and not self.allocates
            and not self.frees
            and not self.opaque
            and self.external_state is None
        )


# ---------------------------------------------------------------------------
# Dialect / stage model
# ---------------------------------------------------------------------------

class DialectKind(Enum):
    tensor = "tensor"
    shape = "shape"
    memory = "memory"
    runtime = "runtime"
    control = "control"


class IRStage(Enum):
    raw = "RawIR"
    normalized = "NormalizedIR"
    inferred = "InferredIR"
    dps = "DPSIR"
    memory = "MemoryIR"
    vm = "VMIR"


_STAGE_DIALECTS: dict[IRStage, frozenset[DialectKind]] = {
    IRStage.raw: frozenset(
        {
            DialectKind.tensor,
            DialectKind.shape,
            DialectKind.memory,
            DialectKind.control,
            DialectKind.runtime,
        }
    ),
    IRStage.normalized: frozenset(
        {
            DialectKind.tensor,
            DialectKind.shape,
            DialectKind.memory,
            DialectKind.control,
            DialectKind.runtime,
        }
    ),
    IRStage.inferred: frozenset(
        {
            DialectKind.tensor,
            DialectKind.shape,
            DialectKind.memory,
            DialectKind.control,
            DialectKind.runtime,
        }
    ),
    IRStage.dps: frozenset(
        {
            DialectKind.tensor,
            DialectKind.memory,
            DialectKind.shape,
            DialectKind.control,
            DialectKind.runtime,
        }
    ),
    IRStage.memory: frozenset(
        {
            DialectKind.tensor,
            DialectKind.memory,
            DialectKind.shape,
            DialectKind.control,
            DialectKind.runtime,
        }
    ),
    IRStage.vm: frozenset(
        {
            DialectKind.tensor,
            DialectKind.memory,
            DialectKind.shape,
            DialectKind.control,
            DialectKind.runtime,
        }
    ),
}


def allowed_dialects(stage: IRStage) -> frozenset[DialectKind]:
    return _STAGE_DIALECTS[stage]


def shape_values(shape: ShapeInfo | tuple[PrimExpr, ...]) -> tuple[PrimExpr, ...]:
    if isinstance(shape, KnownShape):
        return shape.values
    if isinstance(shape, UnknownShape):
        raise TypeError("UnknownShape has no concrete dimensions")
    return tuple(shape)


# ---------------------------------------------------------------------------
# Op — base class for all IR operations
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Op:
    """Base class for all IR operations.

    results is populated by subclass __post_init__ via object.__setattr__.
    Subclasses store result names in result_name / result_names fields;
    the printer reads those to assign display names.
    """
    dialect: ClassVar[DialectKind] = DialectKind.tensor
    results: tuple[OpResult, ...] = field(default_factory=tuple, init=False)


@dataclass(frozen=True, eq=False)
class TerminatorOp(Op):
    """Marker base class for block-terminating ops (ReturnOp, YieldOp)."""
    dialect: ClassVar[DialectKind] = DialectKind.control


# ---------------------------------------------------------------------------
# Block — linear sequence of Ops
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Block:
    """A basic block: block arguments followed by a sequence of Ops.

    args: SSA values defined at block entry (Var — function params, iter vars).
    ops:  Ordered Op sequence.  The last Op MUST be a TerminatorOp.
    """
    args: tuple[Var, ...]
    ops:  tuple[Op, ...]


# ---------------------------------------------------------------------------
# Region — container of one or more Blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Region:
    """A region holds one or more Blocks.

    For structured control flow (MVP), each region contains exactly one Block.
    """
    blocks: tuple[Block, ...]

    @property
    def entry_block(self) -> Block:
        return self.blocks[0]


# ---------------------------------------------------------------------------
# Function / IRModule
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Function:
    """A named function with a single body Region.

    params is a convenience property derived from the entry block's args —
    there is no separate stored field to avoid redundancy.
    """
    body: Region
    ret_struct_info: Optional[StructInfo] = None

    @property
    def params(self) -> tuple[Var, ...]:
        return self.body.entry_block.args


@dataclass
class IRModule:
    functions: dict[str, Function] = field(default_factory=dict)
