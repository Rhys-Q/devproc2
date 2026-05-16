"""First-class operation and runtime target references."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from devproc2.ir.nodes import DialectKind

if TYPE_CHECKING:
    from devproc2.compiler.op.schema import OpDef
    from devproc2.kernel.registry import KernelSpec


class ExternalKind(Enum):
    runtime = "runtime"
    user = "user"


class OpRef:
    """Base class for symbolic references carried by IR operations."""

    name: str
    dialect: DialectKind

    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class StandardOpRef(OpRef):
    name: str
    op_def: Optional["OpDef"] = None
    dialect: DialectKind = DialectKind.tensor

    def __post_init__(self) -> None:
        if self.op_def is not None:
            object.__setattr__(self, "dialect", self.op_def.dialect)

    def resolve(self) -> Optional["OpDef"]:
        if self.op_def is not None:
            return self.op_def
        from devproc2.compiler.op.registry import get

        return get(self.name)

    def display_name(self) -> str:
        return f"@{self.name}"


@dataclass(frozen=True)
class BuiltinOpRef(OpRef):
    name: str
    op_def: Optional["OpDef"] = None
    dialect: DialectKind = DialectKind.runtime

    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class ExternalFuncRef(OpRef):
    name: str
    kind: ExternalKind = ExternalKind.runtime
    dialect: DialectKind = DialectKind.runtime

    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class KernelRef(OpRef):
    name: str
    spec: Optional["KernelSpec"] = None
    dialect: DialectKind = DialectKind.runtime

    def display_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class PackedFuncRef(OpRef):
    name: str
    dialect: DialectKind = DialectKind.runtime

    def display_name(self) -> str:
        return self.name


def standard_ref(name: str) -> StandardOpRef:
    from devproc2.compiler.op.registry import get

    return StandardOpRef(name=name, op_def=get(name))


__all__ = [
    "BuiltinOpRef",
    "ExternalFuncRef",
    "ExternalKind",
    "KernelRef",
    "OpRef",
    "PackedFuncRef",
    "StandardOpRef",
    "standard_ref",
]
