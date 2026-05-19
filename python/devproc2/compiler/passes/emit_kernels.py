"""EmitKernelsPass: write cubin files to the artifact kernels/ directory."""
from __future__ import annotations

import os

from devproc2.kernel.provider import KernelProviderRegistry, get_kernel_provider_registry


class EmitKernelsPass:
    """Write cubin bytes to output_dir/kernels/<name>.cubin.

    Usage::

        EmitKernelsPass().run(
            {"kernel.relu_kernel": b"...cubin...", "kernel.matmul": b"..."},
            output_dir="/tmp/my_model"
        )

    Parameters
    ----------
    kernel_cubins : dict[str, bytes]
        Map from kernel name (e.g. "kernel.relu_fp16") to cubin bytes.
    output_dir : str
        Root artifact directory. Cubin files are written to
        ``output_dir/kernels/<name>.cubin``.
    """

    def __init__(self, provider_registry: KernelProviderRegistry | None = None) -> None:
        self._provider_registry = provider_registry or get_kernel_provider_registry()

    def run(self, kernel_cubins: dict[str, bytes], output_dir: str) -> None:
        kernels_dir = os.path.join(output_dir, "kernels")
        os.makedirs(kernels_dir, exist_ok=True)
        for name, cubin in kernel_cubins.items():
            # Strip "kernel." prefix for filename if present
            fname = name.removeprefix("kernel.")
            path = os.path.join(kernels_dir, f"{fname}.cubin")
            with open(path, "wb") as f:
                f.write(cubin)

    def compile_specs(self, specs, output_dir: str, *, sm_arch: int) -> dict[str, bytes]:
        """Compile KernelSpecs through their backend providers and write cubins."""

        cubins: dict[str, bytes] = {}
        for spec in specs:
            provider = self._provider_registry.get(spec.backend)
            result = provider.compile(
                spec,
                None,
                output_dir=output_dir,
                sm_arch=sm_arch,
            )
            cubins[result.kernel_name] = result.data
        self.run(cubins, output_dir)
        return cubins
