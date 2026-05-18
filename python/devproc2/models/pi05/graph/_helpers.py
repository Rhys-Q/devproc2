"""Shared helpers for Pi0.5 model fragments."""
from __future__ import annotations

import struct

import devproc2 as dp

from .. import ops as pi05_ops


def _static_dim(value, axis: int) -> int:
    ir_value = getattr(value, "value", value)
    si = getattr(ir_value, "struct_info", None)
    shape = getattr(si, "shape", None)
    if shape is None or axis >= len(shape):
        raise ValueError("forward_fast requires static tensor rank information")
    dim = shape[axis]
    return int(getattr(dim, "value", dim))


def _grid_1d(n, block: int = 256) -> dp.KernelLaunchSpec:
    return pi05_ops.grid_1d(n, block)


def _view_bf16_row(table, row: int, width: int):
    return dp.select(table, axis=0, index=row)


def _fp8_linear_ref(x, weight, *, out_features: int):
    return dp.matmul(x, weight, transpose_b=True, out_dtype="bfloat16")


def _qkv_views(qkv, rows: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    q_raw, k_raw, v_raw = dp.split(qkv, (q_dim, kv_dim, kv_dim), axis=1)
    q = dp.reshape(q_raw, (rows, num_q_heads, head_dim))
    k = dp.reshape(k_raw, (rows, num_kv_heads, head_dim))
    v = dp.reshape(v_raw, (rows, num_kv_heads, head_dim))
    return q, k, v


def _add_bias_if_present(x, bias):
    return dp.add(x, bias) if bias is not None else x


def _quantize_fp8_maybe_static(x, n: int, output_shape, scale):
    return pi05_ops.quantize_fp8_maybe_static(x, n, output_shape, scale)


def _layer_norm_fp8_maybe_static(x, weight, bias, rows: int, cols: int, eps_bits: int, scale):
    return pi05_ops.layer_norm_fp8_maybe_static(
        x,
        weight,
        bias,
        rows,
        cols,
        eps_bits,
        scale,
    )


def _rms_norm_unit_fp8_maybe_static(x, rows: int, cols: int, eps_bits: int, scale):
    return pi05_ops.rms_norm_unit_fp8_maybe_static(x, rows, cols, eps_bits, scale)


def _quantize_fp8_dynamic_parallel(x, n: int, output_shape):
    return pi05_ops.quantize_fp8_dynamic(x, n, output_shape)


def _f32_to_i64_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<f", float(value)) + b"\x00\x00\x00\x00", "little")




__all__ = [
    "_add_bias_if_present",
    "_f32_to_i64_bits",
    "_fp8_linear_ref",
    "_grid_1d",
    "_layer_norm_fp8_maybe_static",
    "_qkv_views",
    "_quantize_fp8_dynamic_parallel",
    "_quantize_fp8_maybe_static",
    "_rms_norm_unit_fp8_maybe_static",
    "_static_dim",
    "_view_bf16_row",
]
