"""Backend provider interface for compiling kernel implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from devproc2.kernel.registry import KernelSpec


@dataclass(frozen=True)
class KernelCompileResult:
    kernel_name: str
    backend: str
    symbol: str
    artifact_kind: str
    data: bytes
    metadata: dict[str, object] = field(default_factory=dict)


class KernelProvider(Protocol):
    backend: str

    def compile(
        self,
        spec: KernelSpec,
        kernel_impl: Any,
        *,
        output_dir: str,
        sm_arch: int,
    ) -> KernelCompileResult:
        ...


class KernelProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, KernelProvider] = {}

    def register(self, provider: KernelProvider) -> None:
        self._providers[provider.backend] = provider

    def get(self, backend: str) -> KernelProvider:
        try:
            return self._providers[backend]
        except KeyError as err:
            raise KeyError(f"no kernel provider registered for backend {backend!r}") from err

    def has(self, backend: str) -> bool:
        return backend in self._providers


_GLOBAL_PROVIDER_REGISTRY = KernelProviderRegistry()


def get_kernel_provider_registry() -> KernelProviderRegistry:
    return _GLOBAL_PROVIDER_REGISTRY
