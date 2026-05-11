"""Primitive scalar expressions for devproc2.

PrimExpr represents integer scalar expressions used for:
  - tensor shape dimensions  (TensorStructInfo.shape)
  - runtime shape values     (vm.builtin.shape_of / get_shape_dim)
  - kernel launch grid       (ceildiv(S, BLOCK) etc.)
  - upper-bound assertions   (assert_le(S, 2048))

Design mirrors TVM tir.PrimExpr: each operator is a distinct class,
no string discriminator.  Python operator overloads let you write
  B * S,  ceildiv(S, 16),  pmin(n, 128)
and get proper AST nodes back.

Naming note: the symbolic variable here is `PrimVar` to avoid clashing
with `devproc2.ir.nodes.Var` (the SSA binding variable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

# Anything accepted where a PrimExpr is expected.
# Plain ints are implicitly lifted to IntImm.
PrimExprLike = Union["PrimExpr", int]


def _to_prim(x: PrimExprLike) -> "PrimExpr":
    if isinstance(x, PrimExpr):
        return x
    if isinstance(x, int):
        return IntImm(x)
    raise TypeError(f"Cannot convert {type(x).__name__!r} to PrimExpr")


class PrimExpr:
    """Base class for all primitive scalar expressions."""

    # ------------------------------------------------------------------ #
    # Arithmetic operator overloads                                        #
    # ------------------------------------------------------------------ #

    def __add__(self, other: PrimExprLike) -> "Add":
        return Add(self, _to_prim(other))

    def __radd__(self, other: PrimExprLike) -> "Add":
        return Add(_to_prim(other), self)

    def __sub__(self, other: PrimExprLike) -> "Sub":
        return Sub(self, _to_prim(other))

    def __rsub__(self, other: PrimExprLike) -> "Sub":
        return Sub(_to_prim(other), self)

    def __mul__(self, other: PrimExprLike) -> "Mul":
        return Mul(self, _to_prim(other))

    def __rmul__(self, other: PrimExprLike) -> "Mul":
        return Mul(_to_prim(other), self)

    def __floordiv__(self, other: PrimExprLike) -> "FloorDiv":
        return FloorDiv(self, _to_prim(other))

    def __rfloordiv__(self, other: PrimExprLike) -> "FloorDiv":
        return FloorDiv(_to_prim(other), self)

    # ------------------------------------------------------------------ #
    # Comparison overloads (return PrimExpr booleans for assertions)      #
    # ------------------------------------------------------------------ #

    def __lt__(self, other: PrimExprLike) -> "LT":
        return LT(self, _to_prim(other))

    def __le__(self, other: PrimExprLike) -> "LE":
        return LE(self, _to_prim(other))

    def __gt__(self, other: PrimExprLike) -> "GT":
        return GT(self, _to_prim(other))

    def __ge__(self, other: PrimExprLike) -> "GE":
        return GE(self, _to_prim(other))

    # == and != are kept as identity checks (Python default) so that
    # PrimExpr nodes can be used in sets/dicts.  Use .eq() for symbolic EQ.
    def eq(self, other: PrimExprLike) -> "EQ":
        return EQ(self, _to_prim(other))


# ------------------------------------------------------------------ #
# Leaf nodes                                                           #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class IntImm(PrimExpr):
    """Integer constant, e.g. IntImm(4096)."""
    value: int


@dataclass(frozen=True)
class PrimVar(PrimExpr):
    """Symbolic integer variable, e.g. PrimVar("B", upper=8).

    Corresponds to tir.Var in TVM.  The `upper` field is devproc2-specific:
    it records the compile-time upper bound used by MemoryPlanningPass and
    runtime shape assertions.
    """
    name: str
    upper: Optional[int] = None


# ------------------------------------------------------------------ #
# Binary arithmetic nodes                                             #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Add(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class Sub(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class Mul(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class FloorDiv(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class CeilDiv(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class Min(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class Max(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


# ------------------------------------------------------------------ #
# Comparison nodes (used for runtime assertions in X2)               #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class EQ(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class LT(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class LE(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class GT(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


@dataclass(frozen=True)
class GE(PrimExpr):
    lhs: PrimExpr
    rhs: PrimExpr


# ------------------------------------------------------------------ #
# Free-function helpers (no Python magic-method equivalent)           #
# ------------------------------------------------------------------ #

def ceildiv(a: PrimExprLike, b: PrimExprLike) -> CeilDiv:
    return CeilDiv(_to_prim(a), _to_prim(b))


def pmin(a: PrimExprLike, b: PrimExprLike) -> Min:
    return Min(_to_prim(a), _to_prim(b))


def pmax(a: PrimExprLike, b: PrimExprLike) -> Max:
    return Max(_to_prim(a), _to_prim(b))
