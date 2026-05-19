"""TritonAOTCompilePass: AOT-compile a Triton kernel function to cubin bytes.

Requires triton to be installed. Raises ImportError if not available.
"""
from __future__ import annotations

import os
from typing import Any, Optional


class TritonAOTCompilePass:
    """Ahead-of-time compile a @triton.jit function to cubin.

    Usage::

        cubin = TritonAOTCompilePass().run(
            kernel_fn, output_dir, sm_arch=90,
            compile_options={"num_warps": 4, "num_stages": 3},
        )

    Parameters
    ----------
    kernel_fn : callable
        A Triton kernel function decorated with @triton.jit.
    output_dir : str
        Directory to write the <kernel_name>.cubin file.
    sm_arch : int
        Target SM compute capability (e.g. 80 for Ampere, 90 for Hopper).
    signature : dict, optional
        Triton type signature overrides. If None, inferred from kernel_fn.
    compile_options : dict, optional
        Extra options forwarded to triton.compile().

    Returns
    -------
    bytes
        The compiled cubin binary.

    Raises
    ------
    ImportError
        If triton is not installed.
    RuntimeError
        If compilation fails.
    """

    def run(
        self,
        kernel_fn: Any,
        output_dir: str,
        sm_arch: int = 90,
        signature: Optional[dict] = None,
        compile_options: Optional[dict] = None,
    ) -> bytes:
        try:
            import triton
            import triton.compiler as tc
        except ImportError as e:
            raise ImportError(
                "triton is required for TritonAOTCompilePass. "
                "Install with: pip install triton"
            ) from e

        kernel_name = getattr(kernel_fn, "__name__", "kernel")

        # Build compile options
        target = tc.ASTSource(
            fn=kernel_fn,
            signature=signature or {},
        )
        compile_kwargs = dict(compile_options or {})
        compiled = triton.compile(
            target,
            target=tc.GPUTarget("cuda", sm_arch, 32),
            options=compile_kwargs,
        )

        # Extract cubin bytes
        if hasattr(compiled, "asm") and "cubin" in compiled.asm:
            cubin_bytes: bytes = compiled.asm["cubin"]
        elif hasattr(compiled, "cubin"):
            cubin_bytes = compiled.cubin
        else:
            raise RuntimeError(
                f"Triton compile output for '{kernel_name}' has no cubin; "
                f"available keys: {list(getattr(compiled, 'asm', {}).keys())}"
            )

        # Write to output_dir/kernels/<name>.cubin
        kernels_dir = os.path.join(output_dir, "kernels")
        os.makedirs(kernels_dir, exist_ok=True)
        cubin_path = os.path.join(kernels_dir, f"{kernel_name}.cubin")
        with open(cubin_path, "wb") as f:
            f.write(cubin_bytes)

        return cubin_bytes
