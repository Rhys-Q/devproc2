"""Numpy-only FP8 deployment helpers."""
from __future__ import annotations

from typing import Iterable

import numpy as np


FP8_E4M3_MAX = 448.0
MIN_SCALE = 1.0e-12


def common_fp8_scale(
    amax_values: Iterable[float | int | np.ndarray],
    *,
    max_value: float = FP8_E4M3_MAX,
    min_scale: float = MIN_SCALE,
) -> float:
    amax = 0.0
    for value in amax_values:
        if isinstance(value, np.ndarray):
            current = float(np.max(np.abs(value))) if value.size else 0.0
        else:
            current = abs(float(value))
        amax = max(amax, current)
    return max(amax / float(max_value), float(min_scale))


def quantize_e4m3_reference(values, scale: float) -> np.ndarray:
    """Reference deploy quantizer returning saturated uint8 payload bytes.

    This helper intentionally avoids framework runtimes. It is a deterministic
    manifest/requantization utility, not a hardware FP8 arithmetic emulator.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    arr = np.asarray(values, dtype=np.float32)
    clipped = np.clip(arr / float(scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return clipped.astype(np.int16).astype(np.uint8)


__all__ = [
    "FP8_E4M3_MAX",
    "MIN_SCALE",
    "common_fp8_scale",
    "quantize_e4m3_reference",
]
