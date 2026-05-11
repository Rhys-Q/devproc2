from __future__ import annotations

from dataclasses import fields
from typing import Iterator

from devproc2.ir.nodes import (
    Binding,
    Block,
    Call,
    CallDPS,
    EffectInfo,
    Expr,
    Function,
    IRModule,
    Return,
    Var,
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
        # Note: `defined` is mutated in place. M3 If/For branches must copy it
        # before recursing into sub-blocks to avoid cross-branch contamination.
        if not isinstance(block.body, Return):
            raise IRVerificationError(
                f"In @{fn_name}: block body must be Return, "
                f"got {type(block.body).__name__}"
            )
        for binding in block.bindings:
            var, expr = binding
            self._check_forbidden(fn_name, expr)
            if isinstance(expr, Return):
                raise IRVerificationError(
                    f"In @{fn_name}: Return must only appear as block body, not in bindings"
                )
            self._check_refs_defined(fn_name, expr, defined)
            if isinstance(expr, CallDPS):
                if expr.output is None and var is not None:
                    raise IRVerificationError(
                        f"In @{fn_name}: CallDPS with output=None must have var=None in binding"
                    )
                if expr.output is not None and var is None:
                    raise IRVerificationError(
                        f"In @{fn_name}: CallDPS with output set must be bound in binding"
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
        for ref in _value_refs(expr):
            if ref.name not in defined:
                raise IRVerificationError(
                    f"In @{fn_name}: Variable '%{ref.name}' used before definition"
                )

    def _check_forbidden(self, fn_name: str, node: object) -> None:
        if isinstance(node, (int, float, bool, str, type(None))):
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                self._check_forbidden(fn_name, item)
            return
        if isinstance(node, (Call, CallDPS)):
            if node.callee in _FORBIDDEN_CALLEES:
                node_name = node.callee.lstrip("@")
                raise IRVerificationError(
                    f"In @{fn_name}: {node_name} is forbidden in high-level IR "
                    f"(use TensorCreateOp instead)"
                )
        try:
            fs = fields(node)  # type: ignore[arg-type]
        except TypeError:
            return
        for f in fs:
            self._check_forbidden(fn_name, getattr(node, f.name))


def _value_refs(expr: object) -> Iterator[Var]:
    """Depth-first walk collecting IR Var value-uses in an expression tree.

    EffectInfo subtrees are skipped: WriteEffect.vars declares side-effect
    metadata, not SSA value uses.
    """
    if isinstance(expr, Var):
        yield expr
        return
    if isinstance(expr, EffectInfo):
        return
    if isinstance(expr, (int, float, bool, str, type(None))):
        return
    if isinstance(expr, (list, tuple)):
        for item in expr:
            yield from _value_refs(item)
        return
    try:
        fs = fields(expr)  # type: ignore[arg-type]
    except TypeError:
        return
    for f in fs:
        yield from _value_refs(getattr(expr, f.name))


def verify(module: IRModule) -> None:
    """Verify IR invariants. Raises IRVerificationError on the first violation."""
    Verifier().verify_module(module)
