"""Dtype utilities shared across compiler and runtime layers."""
from __future__ import annotations


DTYPE_BYTES: dict[str, int] = {
    "float16":  2,
    "bfloat16": 2,
    "float32":  4,
    "float64":  8,
    "int8":     1,
    "int16":    2,
    "int32":    4,
    "int64":    8,
    "bool":     1,
}


def dtype_itemsize(dtype: str) -> int:
    """Return the byte size of a single element for the given dtype string.

    Raises ValueError for unknown dtypes.
    """
    size = DTYPE_BYTES.get(dtype)
    if size is None:
        raise ValueError(f"Unknown dtype '{dtype}'")
    return size
