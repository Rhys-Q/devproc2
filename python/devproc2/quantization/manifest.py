"""Stable quantization manifest schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuantTensorSpec:
    name: str
    dtype: str
    scale: float | None = None
    amax: float | None = None
    axis: int | None = None
    group_size: int | None = None
    source: str | None = None
    quantizer: str | None = None

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationManifest:
    dataset: str | None = None
    sample_count: int | None = None
    sample_hash: str | None = None
    producer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusionComponentSpec:
    source_name: str
    logical_name: str
    amax: float | None = None
    scale: float | None = None

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusionGroupSpec:
    name: str
    output_name: str
    components: tuple[FusionComponentSpec, ...]
    scale_policy: str = "max_component_amax"
    requantize_from: str = "source_fp"
    layout: str | None = None

    def to_json_obj(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["components"] = [component.to_json_obj() for component in self.components]
        return payload


@dataclass(frozen=True)
class QuantizationManifest:
    format: str = "devproc2.quantization.manifest"
    format_version: int = 1
    tensors: tuple[QuantTensorSpec, ...] = ()
    calibration: CalibrationManifest | None = None
    fusion_groups: tuple[FusionGroupSpec, ...] = ()
    producer: str | None = None
    producer_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "format_version": self.format_version,
            "tensors": [tensor.to_json_obj() for tensor in self.tensors],
            "calibration": (
                None if self.calibration is None else self.calibration.to_json_obj()
            ),
            "fusion_groups": [group.to_json_obj() for group in self.fusion_groups],
            "producer": self.producer,
            "producer_version": self.producer_version,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "CalibrationManifest",
    "FusionComponentSpec",
    "FusionGroupSpec",
    "QuantTensorSpec",
    "QuantizationManifest",
]
