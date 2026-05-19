"""Call emission for Python op functions."""
from __future__ import annotations

from typing import Protocol

from devproc2.compiler.op.schema import OpDef


class Emitter(Protocol):
    def emit_op(self, op: OpDef | str, args: tuple[object, ...], attrs: dict[str, object]):
        ...


_CURRENT_EMITTER: Emitter | None = None


def set_current_emitter(emitter: Emitter | None) -> Emitter | None:
    global _CURRENT_EMITTER
    prev = _CURRENT_EMITTER
    _CURRENT_EMITTER = emitter
    return prev


def get_current_emitter() -> Emitter | None:
    return _CURRENT_EMITTER


def emit(op_func, *args, **attrs):
    emitter = _CURRENT_EMITTER
    if emitter is None:
        raise RuntimeError("devproc2 ops can only run inside GraphBuilder.build")
    op = getattr(op_func, "op_def", op_func)
    return emitter.emit_op(op, args, attrs)


def call(name: str, *args, **attrs):
    emitter = _CURRENT_EMITTER
    if emitter is None:
        raise RuntimeError("devproc2 ops can only run inside GraphBuilder.build")
    return emitter.emit_op(name, args, attrs)


__all__ = [
    "Emitter",
    "call",
    "emit",
    "get_current_emitter",
    "set_current_emitter",
]
