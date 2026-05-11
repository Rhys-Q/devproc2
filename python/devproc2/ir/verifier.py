from __future__ import annotations

from dataclasses import fields
from typing import Iterator

from devproc2.ir.nodes import (
    BinOpDim,
    Binding,
    Block,
    Call,
    CallDPS,
    Constant,
    Expr,
    Function,
    IRModule,
    Return,
    ShapeExpr,
    SymDimRef,
    TensorCreateOp,
    TupleExpr,
    TupleGetItem,
    Var,
    WriteEffect,
)

_FORBIDDEN_CALLEES = frozenset({"@alloc_storage", "@alloc_tensor"})


class IRVerificationError(Exception):
    pass


class Verifier:
    def verify_module(self, module: IRModule) -> None:
        for name, fn in module.functions.items():
            self._verify_function(name, fn)

    def _verify_function(self, name: str, fn: Function) -> None:
        defined: set[str] = set()
        for p in fn.params:
            if p.name in defined:
                raise IRVerificationError(
                    f"In @{name}: parameter '%{p.name}' defined more than once"
                )
            defined.add(p.name)

        self._verify_block(name, fn.body, defined)

    def _verify_block(self, fn_name: str, block: Block, defined: set[str]) -> None:
        for binding in block.bindings:
            var, expr = binding
            self._check_forbidden(fn_name, expr)
            if isinstance(expr, Return):
                raise IRVerificationError(
                    f"In @{fn_name}: Return node must only appear as block body, not in bindings"
                )
            self._check_refs_defined(fn_name, expr, defined)
            if isinstance(expr, CallDPS):
                if expr.output is None and var is not None:
                    raise IRVerificationError(
                        f"In @{fn_name}: CallDPS with output=None must have var=None in binding"
                    )
                if expr.output is not None and var is None:
                    raise IRVerificationError(
                        f"In @{fn_name}: CallDPS with output var must be bound in binding"
                    )
            if var is not None:
                if var.name in defined:
                    raise IRVerificationError(
                        f"In @{fn_name}: Variable '%{var.name}' defined more than once"
                    )
                defined.add(var.name)

        self._check_forbidden(fn_name, block.body)
        self._check_refs_defined(fn_name, block.body, defined)

    def _check_refs_defined(self, fn_name: str, expr: Expr, defined: set[str]) -> None:
        for ref in _refs_in_expr(expr):
            if ref.name not in defined:
                raise IRVerificationError(
                    f"In @{fn_name}: Variable '%{ref.name}' used before definition"
                )

    def _check_forbidden(self, fn_name: str, expr: Expr) -> None:
        if isinstance(expr, (Call, CallDPS)):
            callee = expr.callee
            if callee in _FORBIDDEN_CALLEES:
                node_name = callee.lstrip("@")
                raise IRVerificationError(
                    f"In @{fn_name}: {node_name} is forbidden in high-level IR "
                    f"(use TensorCreateOp instead)"
                )


def _refs_in_expr(expr: object) -> Iterator[Var]:
    """Depth-first walk collecting all Var nodes referenced in an expression tree."""
    if isinstance(expr, Var):
        yield expr
        return
    if isinstance(expr, (int, float, bool, str, type(None))):
        return
    if isinstance(expr, (list, tuple)):
        for item in expr:
            yield from _refs_in_expr(item)
        return
    # dataclass nodes: recurse into all fields
    try:
        fs = fields(expr)  # type: ignore[arg-type]
    except TypeError:
        return
    for f in fs:
        child = getattr(expr, f.name)
        yield from _refs_in_expr(child)


def verify(module: IRModule) -> None:
    """Verify IR invariants. Raises IRVerificationError on the first violation."""
    Verifier().verify_module(module)
