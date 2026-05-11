from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, TypeAlias, Union

if TYPE_CHECKING:
    import numpy as np

from devproc2.ir.prim_expr import IntImm, PrimExpr


class Expr:
    """Base class for all IR expression nodes."""


class StructInfo:
    """Base class for all struct info types (type + runtime structural info)."""


@dataclass(frozen=True)
class TensorStructInfo(StructInfo):
    shape: tuple[PrimExpr, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        # Coerce bare int literals in shape to IntImm for ergonomic construction.
        object.__setattr__(
            self, "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )


@dataclass(frozen=True)
class Var(Expr):
    """SSA binding variable. Distinct from prim_expr.PrimVar."""
    name: str
    struct_info: Optional[StructInfo] = None


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


@dataclass(frozen=True)
class Constant(Expr):
    # Scalar literal or numpy array (tensor constant loaded from weights).
    value: Union[int, float, bool, None, "np.ndarray"]


class CalleeKind(Enum):
    vm_func = auto()
    builtin = auto()
    packed_func = auto()
    kernel = auto()


@dataclass(frozen=True)
class Call(Expr):
    callee: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class CallDPS(Expr):
    callee: str
    inputs: tuple[Expr, ...]
    output: Optional[Var]
    effect: EffectInfo
    callee_kind: CalleeKind


@dataclass(frozen=True)
class TupleExpr(Expr):
    elems: tuple[Expr, ...]


@dataclass(frozen=True)
class TupleGetItem(Expr):
    tup: Expr
    index: int


class TensorCreateKind(Enum):
    empty = auto()
    zeros = auto()
    full = auto()
    empty_like = auto()


@dataclass(frozen=True)
class TensorCreateOp(Expr):
    kind: TensorCreateKind
    shape: tuple[PrimExpr, ...]  # must be () for empty_like
    dtype: str
    device: str
    fill_value: Optional[object] = None  # only for full
    like: Optional[Var] = None           # only for empty_like

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


@dataclass(frozen=True)
class Return(Expr):
    value: Expr


# var=None means a bare statement (no-output CallDPS with no binding LHS).
Binding: TypeAlias = tuple[Optional[Var], Expr]


@dataclass(frozen=True)
class Block:
    bindings: tuple[Binding, ...]
    body: Expr  # must be Return (enforced by verifier)


@dataclass(frozen=True)
class Function:
    params: tuple[Var, ...]
    body: Block
    ret_struct_info: Optional[StructInfo] = None


@dataclass
class IRModule:
    functions: dict[str, Function] = field(default_factory=dict)
