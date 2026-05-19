"""Backend provider interface for compiling kernel implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import subprocess
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


class CudaSourceProvider:
    """Compile a CUDA C++ source file into a cubin artifact with nvcc."""

    backend = "cuda"

    def compile(
        self,
        spec: KernelSpec,
        kernel_impl: Any,
        *,
        output_dir: str,
        sm_arch: int,
    ) -> KernelCompileResult:
        source_path = spec.source_path or getattr(kernel_impl, "_cuda_source_path", None)
        if not source_path:
            raise ValueError(
                f"CUDA kernel {spec.kernel_name!r} requires source_path or "
                "kernel_impl._cuda_source_path"
            )
        source_path = os.path.abspath(os.fspath(source_path))
        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)

        kernels_dir = os.path.join(output_dir, "kernels")
        os.makedirs(kernels_dir, exist_ok=True)
        cubin_name = spec.kernel_name.removeprefix("kernel.") + ".cubin"
        cubin_path = os.path.join(kernels_dir, cubin_name)

        nvcc = str(spec.compile_options.get("nvcc", os.environ.get("NVCC", "nvcc")))
        cmd = [
            nvcc,
            "--cubin",
            f"-arch=sm_{int(sm_arch)}",
            source_path,
            "-o",
            cubin_path,
        ]
        for inc in spec.include_dirs:
            cmd.extend(["-I", os.path.abspath(os.fspath(inc))])
        cmd.extend(spec.extra_nvcc_flags)

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as err:
            stderr = err.stderr.decode("utf-8", errors="replace")
            stdout = err.stdout.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"nvcc failed for {spec.kernel_name} ({source_path})\n"
                f"command: {' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ) from err

        data = open(cubin_path, "rb").read()
        return KernelCompileResult(
            kernel_name=spec.kernel_name,
            backend=self.backend,
            symbol=spec.symbol or spec.kernel_name.removeprefix("kernel."),
            artifact_kind="cubin",
            data=data,
            metadata={
                "sm_arch": sm_arch,
                "source_path": source_path,
                "cubin_path": cubin_path,
            },
        )


_GLOBAL_PROVIDER_REGISTRY.register(CudaSourceProvider())
