"""Generic compile recipe declarations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from devproc2.nn import Module


RecipeOptions = Mapping[str, Any]


@dataclass(frozen=True)
class EntrypointRecipe:
    """Model-owned description of one compilable entrypoint."""

    name: str
    model_id: str
    build_module: Callable[[RecipeOptions], Module]
    input_specs: Callable[[RecipeOptions], dict[str, object]]
    function_name: str = "main"
    normal_method: str = "forward"
    fast_method: str = "forward_fast"
    model_name: str | None = None
    packed_backends: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CompileRecipe:
    """Collection of entrypoints for one model."""

    model_id: str
    entrypoints: Mapping[str, EntrypointRecipe]

    def entrypoint(self, name: str) -> EntrypointRecipe:
        try:
            return self.entrypoints[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.entrypoints))
            raise KeyError(f"unknown entrypoint {name!r}; available: {available}") from exc


__all__ = [
    "CompileRecipe",
    "EntrypointRecipe",
    "RecipeOptions",
]
