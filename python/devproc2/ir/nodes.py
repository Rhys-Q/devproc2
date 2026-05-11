from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Expr:
    """Base class for all IR expression nodes."""


class StructInfo:
    """Base class for all struct info types (type + runtime structural info)."""


class ShapeExpr:
    """Base class for shape dimension expressions."""


@dataclass(frozen=True)
class SymbolicDim:
    name: str
    upper: Optional[int] = None


@dataclass(frozen=True)
class ConstDim(ShapeExpr):
    value: int


@dataclass(frozen=True)
class SymDimRef(ShapeExpr):
    dim: SymbolicDim


@dataclass(frozen=True)
class BinOpDim(ShapeExpr):
    """Arithmetic combination of two shape expressions.

    op must be one of: add, sub, mul, floordiv, ceildiv, min, max
    """
    op: str
    lhs: ShapeExpr
    rhs: ShapeExpr


@dataclass(frozen=True)
class TensorStructInfo(StructInfo):
    shape: tuple[ShapeExpr, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class Var(Expr):
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
    value: object


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
    shape: tuple[ShapeExpr, ...]
    dtype: str
    device: str
    fill_value: Optional[object] = None
    like: Optional[Var] = None


@dataclass(frozen=True)
class Return(Expr):
    value: Expr


# A Binding is (Optional[Var], Expr): var=None means a bare statement (no-output CallDPS).
Binding = tuple[Optional[Var], Expr]


@dataclass(frozen=True)
class Block:
    bindings: tuple[Binding, ...]
    body: Expr


@dataclass(frozen=True)
class Function:
    params: tuple[Var, ...]
    body: Block
    ret_struct_info: Optional[StructInfo] = None


@dataclass
class IRModule:
    functions: dict[str, Function] = field(default_factory=dict)
