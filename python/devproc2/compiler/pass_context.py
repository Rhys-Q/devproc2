from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PassContext:
    """Key-value store for sharing analysis results between compiler passes."""
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        self._cache[key] = value
