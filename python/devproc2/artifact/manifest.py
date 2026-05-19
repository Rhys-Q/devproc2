"""Generic artifact manifest declarations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    path: str | Path
    target_path: str | None = None
    kind: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PackedFuncSpec:
    name: str
    device: str | None = None
    effect: str = "opaque"

    def to_json_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device": self.device,
            "effect": self.effect,
        }


@dataclass(frozen=True)
class PackedBackendRecipe:
    name: str
    kind: str
    sources: tuple[str, ...]
    include_dirs: tuple[str, ...] = ()
    compile_definitions: tuple[str, ...] = ()
    compile_options: tuple[str, ...] = ()
    link_libraries: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    register_symbol: str = "devproc2_register_packed_backend"
    packed_funcs: tuple[PackedFuncSpec, ...] = ()
    library: str | None = None

    def to_table_obj(self, *, target_arch: str) -> dict[str, object]:
        library = self.library
        if library is None and self.kind != "linked_packed_backend":
            library = f"backends/{self.name.replace('.', '_')}.so"
        return {
            "name": self.name,
            "kind": self.kind,
            "library": library,
            "register_symbol": self.register_symbol,
            "target_arch": target_arch,
            "packed_funcs": [func.to_json_obj() for func in self.packed_funcs],
        }


__all__ = [
    "PackedBackendRecipe",
    "PackedFuncSpec",
    "ResourceSpec",
]
