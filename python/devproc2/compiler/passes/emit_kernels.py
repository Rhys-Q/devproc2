"""EmitKernelsPass: write cubin files to the artifact kernels/ directory."""
from __future__ import annotations

import os


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

    def run(self, kernel_cubins: dict[str, bytes], output_dir: str) -> None:
        kernels_dir = os.path.join(output_dir, "kernels")
        os.makedirs(kernels_dir, exist_ok=True)
        for name, cubin in kernel_cubins.items():
            # Strip "kernel." prefix for filename if present
            fname = name.removeprefix("kernel.")
            path = os.path.join(kernels_dir, f"{fname}.cubin")
            with open(path, "wb") as f:
                f.write(cubin)
