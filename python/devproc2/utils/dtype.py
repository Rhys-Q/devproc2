"""Dtype and device utilities shared across compiler and runtime layers."""
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

# DLPack DLDataType encoding: dtype_str → (code, bits, lanes)
DTYPE_DL_ENCODING: dict[str, tuple[int, int, int]] = {
    "bool":     (6,  8,  1),   # kDLBool
    "int8":     (0,  8,  1),   # kDLInt
    "int16":    (0,  16, 1),
    "int32":    (0,  32, 1),
    "int64":    (0,  64, 1),
    "uint8":    (1,  8,  1),   # kDLUInt
    "uint16":   (1,  16, 1),
    "uint32":   (1,  32, 1),
    "uint64":   (1,  64, 1),
    "float16":  (2,  16, 1),   # kDLFloat
    "float32":  (2,  32, 1),
    "float64":  (2,  64, 1),
    "bfloat16": (4,  16, 1),   # kDLBfloat
}

# DLPack DLDeviceType encoding: device_name → type_int
DEVICE_TYPE_ENCODING: dict[str, int] = {
    "cpu":    1,   # kDLCPU
    "cuda":   2,   # kDLCUDA
    "cuda_host": 3, # kDLCUDAHost
    "vulkan": 7,   # kDLVulkan
    "metal":  8,   # kDLMetal
    "rocm":   10,  # kDLROCM
}


def dtype_itemsize(dtype: str) -> int:
    size = DTYPE_BYTES.get(dtype)
    if size is None:
        raise ValueError(f"Unknown dtype '{dtype}'")
    return size


def parse_dtype(dtype_str: str) -> tuple[int, int, int]:
    """Parse dtype string to DLPack (code, bits, lanes)."""
    result = DTYPE_DL_ENCODING.get(dtype_str.lower())
    if result is None:
        raise ValueError(f"Unknown dtype: {dtype_str!r}")
    return result


def parse_device(device_str: str) -> tuple[int, int]:
    """Parse device string to DLPack (device_type_int, device_id_int).

    Accepts "cpu", "cuda", "cuda:1", etc.
    """
    parts = device_str.split(":")
    dev_name = parts[0].lower()
    dev_type = DEVICE_TYPE_ENCODING.get(dev_name)
    if dev_type is None:
        raise ValueError(f"Unknown device type: {dev_name!r}")
    dev_id = int(parts[1]) if len(parts) > 1 else 0
    return dev_type, dev_id
