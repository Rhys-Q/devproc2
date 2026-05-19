"""Pi0.5 CUDA/HPC op facade."""
from __future__ import annotations

from pathlib import Path

import devproc2 as dp

from devproc2.ir.prim_expr import PrimExpr, ceildiv


PI05_CUDA_SOURCE = Path(__file__).resolve().parent / "cuda" / "pi05_kernels.cu"


def cuda_symbol(name: str) -> str:
    return f"{PI05_CUDA_SOURCE}::{name}"


def cuda_metadata(
    name: str,
    *,
    effect: str = "opaque",
    launch: dp.KernelLaunchSpec | None = None,
    output_indices: int | tuple[int, ...] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "kernel_name": f"kernel.{name}",
        "extra_nvcc_flags": ("--std=c++17",),
        "effect": effect,
    }
    if launch is not None:
        metadata["launch"] = launch
    if output_indices is not None:
        metadata["output_indices"] = output_indices
    return metadata


def grid_1d(n: int | PrimExpr, block: int = 256) -> dp.KernelLaunchSpec:
    grid_x = max(1, (n + block - 1) // block) if isinstance(n, int) else ceildiv(n, block)
    return dp.KernelLaunchSpec(grid=(grid_x, 1, 1), block=(block, 1, 1))


def call_packed_out(
    name: str,
    inputs,
    *,
    output_shape,
    output_dtype: str = "bfloat16",
    output_device: str = "cuda",
):
    out = dp.empty(output_shape, dtype=output_dtype, device=output_device)
    dp.call_dps_packed(name, inputs=[*inputs, out])
    return out


def call_cuda(
    name: str,
    *args,
    effect: str = "opaque",
    launch: dp.KernelLaunchSpec | None = None,
    output_indices: int | tuple[int, ...] | None = None,
):
    dp.cuda_call(
        cuda_symbol(name),
        *args,
        metadata=cuda_metadata(
            name,
            effect=effect,
            launch=launch,
            output_indices=output_indices,
        ),
    )


def bf16_linear(x, weight, *, rows: int, out_features: int, in_features: int):
    return call_packed_out(
        "pi05.cuda.bf16_nn_bf16",
        inputs=[x, weight, rows, out_features, in_features],
        output_shape=(rows, out_features),
    )


def fp8_linear(
    x_fp8,
    weight_fp8,
    *,
    rows: int,
    out_features: int,
    in_features: int,
    x_scale,
    weight_scale,
):
    return call_packed_out(
        "pi05.cuda.fp8_nt_bf16",
        inputs=[
            x_fp8,
            weight_fp8,
            rows,
            out_features,
            in_features,
            x_scale,
            weight_scale,
        ],
        output_shape=(rows, out_features),
    )


def fp8_linear_accum_(
    x_fp8,
    weight_fp8,
    *,
    residual,
    rows: int,
    out_features: int,
    in_features: int,
    x_scale,
    weight_scale,
):
    dp.call_dps_packed(
        "pi05.cuda.fp8_nt_bf16_accum",
        inputs=[
            x_fp8,
            weight_fp8,
            residual,
            rows,
            out_features,
            in_features,
            x_scale,
            weight_scale,
        ],
    )
    return residual


def attention_fa2(
    q,
    k,
    v,
    *,
    rows: int,
    prefix_valid_rows,
    suffix_rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale_bits: int,
):
    return call_packed_out(
        "pi05.cuda.fa2_bf16",
        inputs=[
            q,
            k,
            v,
            rows,
            prefix_valid_rows,
            suffix_rows,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale_bits,
        ],
        output_shape=(rows, num_q_heads, head_dim),
    )


def attention_fa2_batched(
    q,
    k,
    v,
    *,
    batches: int,
    query_rows: int,
    key_rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale_bits: int,
):
    return call_packed_out(
        "pi05.cuda.fa2_bf16_batched",
        inputs=[
            q,
            k,
            v,
            batches,
            query_rows,
            key_rows,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale_bits,
        ],
        output_shape=(batches * query_rows, num_q_heads, head_dim),
    )


def quantize_fp8_static(x, scale, n: int, output_shape):
    out = dp.empty(output_shape, dtype="fp8_e4m3", device="cuda")
    dp.cuda_call(
        cuda_symbol("pi05_quantize_fp8_static_bf16"),
        x,
        scale,
        n,
        out,
        metadata=cuda_metadata("pi05_quantize_fp8_static_bf16", launch=grid_1d(n)),
    )
    return out, scale


def quantize_fp8_maybe_static(x, n: int, output_shape, scale):
    if scale is None:
        return quantize_fp8_dynamic(x, n, output_shape)
    return quantize_fp8_static(x, scale, n, output_shape)


def layer_norm_fp8_maybe_static(x, weight, bias, rows: int, cols: int, eps_bits: int, scale):
    if scale is None:
        normed = dp.empty((rows, cols), dtype="bfloat16", device="cuda")
        dp.cuda_call(
            cuda_symbol("pi05_layer_norm_bf16"),
            x,
            weight,
            bias,
            rows,
            cols,
            eps_bits,
            normed,
            metadata=cuda_metadata(
                "pi05_layer_norm_bf16",
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            ),
        )
        return quantize_fp8_dynamic(normed, rows * cols, (rows, cols))
    out = dp.empty((rows, cols), dtype="fp8_e4m3", device="cuda")
    dp.cuda_call(
        cuda_symbol("pi05_layer_norm_to_fp8_bf16"),
        x,
        weight,
        bias,
        scale,
        rows,
        cols,
        eps_bits,
        out,
        metadata=cuda_metadata(
            "pi05_layer_norm_to_fp8_bf16",
            launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
        ),
    )
    return out, scale


def rms_norm_unit_fp8_maybe_static(x, rows: int, cols: int, eps_bits: int, scale):
    if scale is None:
        normed = dp.empty((rows, cols), dtype="bfloat16", device="cuda")
        dp.cuda_call(
            cuda_symbol("pi05_rms_norm_unit_bf16"),
            x,
            rows,
            cols,
            eps_bits,
            normed,
            metadata=cuda_metadata(
                "pi05_rms_norm_unit_bf16",
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            ),
        )
        return quantize_fp8_dynamic(normed, rows * cols, (rows, cols))
    out = dp.empty((rows, cols), dtype="fp8_e4m3", device="cuda")
    dp.cuda_call(
        cuda_symbol("pi05_rms_norm_unit_to_fp8_bf16"),
        x,
        scale,
        rows,
        cols,
        eps_bits,
        out,
        metadata=cuda_metadata(
            "pi05_rms_norm_unit_to_fp8_bf16",
            launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
        ),
    )
    return out, scale


def quantize_fp8_dynamic(x, n: int, output_shape):
    if n <= 4096:
        out = dp.empty(output_shape, dtype="fp8_e4m3", device="cuda")
        scale = dp.empty((1,), dtype="float32", device="cuda")
        dp.cuda_call(
            cuda_symbol("pi05_quantize_fp8_dynamic_bf16"),
            x,
            n,
            out,
            scale,
            metadata=cuda_metadata(
                "pi05_quantize_fp8_dynamic_bf16",
                launch=dp.KernelLaunchSpec(grid=(1, 1, 1), block=(256, 1, 1)),
            ),
        )
        return out, scale

    partial_blocks = 128
    partial_amax = dp.empty((partial_blocks,), dtype="float32", device="cuda")
    dp.cuda_call(
        cuda_symbol("pi05_reduce_amax_bf16"),
        x,
        n,
        partial_amax,
        metadata=cuda_metadata(
            "pi05_reduce_amax_bf16",
            launch=dp.KernelLaunchSpec(grid=(partial_blocks, 1, 1), block=(256, 1, 1)),
        ),
    )
    scale = dp.empty((1,), dtype="float32", device="cuda")
    dp.cuda_call(
        cuda_symbol("pi05_amax_to_scale"),
        partial_amax,
        partial_blocks,
        scale,
        metadata=cuda_metadata(
            "pi05_amax_to_scale",
            launch=dp.KernelLaunchSpec(grid=(1, 1, 1), block=(256, 1, 1)),
        ),
    )
    out = dp.empty(output_shape, dtype="fp8_e4m3", device="cuda")
    dp.cuda_call(
        cuda_symbol("pi05_quantize_fp8_static_bf16"),
        x,
        scale,
        n,
        out,
        metadata=cuda_metadata("pi05_quantize_fp8_static_bf16", launch=grid_1d(n)),
    )
    return out, scale


__all__ = [
    "PI05_CUDA_SOURCE",
    "attention_fa2",
    "attention_fa2_batched",
    "bf16_linear",
    "call_cuda",
    "call_packed_out",
    "cuda_metadata",
    "cuda_symbol",
    "fp8_linear",
    "fp8_linear_accum_",
    "grid_1d",
    "layer_norm_fp8_maybe_static",
    "quantize_fp8_dynamic",
    "quantize_fp8_maybe_static",
    "quantize_fp8_static",
    "rms_norm_unit_fp8_maybe_static",
]
