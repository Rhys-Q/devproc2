from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import numpy as np

from devproc2.ir.prim_expr import IntImm, PrimExpr


# ---------------------------------------------------------------------------
# Value hierarchy — operands passed to Ops
# ---------------------------------------------------------------------------

class Value:
    """Base class for Op operands (Var, OpResult, or Constant)."""


@dataclass(frozen=True)
class Var(Value):
    """Block argument — defined at block entry (function params, iter args)."""
    name: str
    struct_info: Optional[StructInfo] = None


@dataclass(frozen=True)
class Constant(Value):
    """Compile-time scalar or tensor constant."""
    value: Union[int, float, bool, None, "np.ndarray"]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TensorStructInfo(StructInfo):
    shape: tuple[PrimExpr, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )


# ---------------------------------------------------------------------------
# EffectInfo — side-effect annotation for CallDPSOp
# ---------------------------------------------------------------------------

class EffectInfo:
    """Base class for effect annotations."""


@dataclass(frozen=True)
class PureEffect(EffectInfo):
    pass


@dataclass(frozen=True)
class ReadOnlyEffect(EffectInfo):
    pass


@dataclass(frozen=True)
class WriteEffect(EffectInfo):
    vars: tuple[Var, ...]


@dataclass(frozen=True)
class OpaqueEffect(EffectInfo):
    pass


# ---------------------------------------------------------------------------
# Op — base class for all IR operations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    """Base class for all IR operations.

    results is populated by subclass __post_init__ via object.__setattr__.
    Subclasses store result names in result_name / result_names fields;
    the printer reads those to assign display names.
    """
    results: tuple[OpResult, ...] = field(default_factory=tuple, init=False)


@dataclass(frozen=True)
class TerminatorOp(Op):
    """Marker base class for block-terminating ops (ReturnOp, YieldOp)."""


# ---------------------------------------------------------------------------
# Block — linear sequence of Ops
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
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
