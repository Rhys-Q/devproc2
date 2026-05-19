"""Framework-level quantization protocol helpers."""
from __future__ import annotations

from devproc2.quantization.fp8 import common_fp8_scale, quantize_e4m3_reference
from devproc2.quantization.manifest import (
    CalibrationManifest,
    FusionComponentSpec,
    FusionGroupSpec,
    QuantTensorSpec,
    QuantizationManifest,
)

__all__ = [
    "CalibrationManifest",
    "FusionComponentSpec",
    "FusionGroupSpec",
    "QuantTensorSpec",
    "QuantizationManifest",
    "common_fp8_scale",
    "quantize_e4m3_reference",
]
