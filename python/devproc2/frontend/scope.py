from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from devproc2.ir.nodes import Value, Var


@dataclass
class ScopeFrame:
    bindings: dict[str, Value] = field(default_factory=dict)


class ScopeStack:
    """Lexical scope stack for the DSL builder.

    Stores Value objects (Var for block args / iter vars, OpResult for op results).
    """

    def __init__(self) -> None:
        self._frames: list[ScopeFrame] = [ScopeFrame()]

    def push(self) -> None:
        self._frames.append(ScopeFrame())

    def pop(self) -> None:
        if len(self._frames) > 1:
            self._frames.pop()

    def define(self, name: str, val: Value) -> None:
        self._frames[-1].bindings[name] = val

    def lookup(self, name: str) -> Optional[Value]:
        for frame in reversed(self._frames):
            if name in frame.bindings:
                return frame.bindings[name]
        return None

    def outer_names(self) -> set[str]:
        """Names visible from all frames."""
        names: set[str] = set()
        for frame in self._frames:
            names.update(frame.bindings)
        return names

    def snapshot(self) -> list[ScopeFrame]:
        return [ScopeFrame(dict(f.bindings)) for f in self._frames]

    def restore(self, snap: list[ScopeFrame]) -> None:
        self._frames = snap
