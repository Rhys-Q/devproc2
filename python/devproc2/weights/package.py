"""Generic devproc2 weight package schema and writer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


BF16 = "bfloat16"
FP16 = "float16"
FP32 = "float32"
FP8_E4M3 = "fp8_e4m3"


@dataclass(frozen=True)
class QuantSpec:
    scheme: str
    storage_dtype: str
    compute_dtype: str
    scale_name: str | None
    zero_point_name: str | None = None
    group_size: int | None = None
    axis: int | None = None
    packed_layout: str | None = None


@dataclass(frozen=True)
class WeightEntry:
    name: str
    kind: Literal["weight", "constant_tensor", "scale"]
    shape: tuple[int, ...]
    dtype: str
    layout: str
    offset: int
    nbytes: int
    alignment: int = 256
    transform: str | None = None
    tied_to: str | None = None
    quant: QuantSpec | None = None

    def to_weight_map_obj(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["quant"] = None if self.quant is None else asdict(self.quant)
        payload.pop("offset")
        payload.pop("nbytes")
        payload.pop("alignment")
        return payload

    def to_index_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "alignment": self.alignment,
        }


class WeightPackageWriter:
    """Write a devproc2 self-contained weight package."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model: str,
        precision: str = BF16,
        alignment: int = 256,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model = model
        self.precision = precision
        self.alignment = int(alignment)
        self._data = bytearray()
        self.entries: list[WeightEntry] = []

    def add_tensor(
        self,
        name: str,
        tensor: Any,
        *,
        dtype: str | None = None,
        kind: Literal["weight", "constant_tensor", "scale"] = "weight",
        layout: str = "row_major",
        transform: str | None = None,
        tied_to: str | None = None,
        quant: QuantSpec | None = None,
    ) -> WeightEntry:
        raw, shape, resolved_dtype = tensor_to_bytes(tensor, dtype)
        offset = self._align(len(self._data))
        if offset > len(self._data):
            self._data.extend(b"\x00" * (offset - len(self._data)))
        self._data.extend(raw)
        entry = WeightEntry(
            name=name,
            kind=kind,
            shape=tuple(int(s) for s in shape),
            dtype=resolved_dtype,
            layout=layout,
            offset=offset,
            nbytes=len(raw),
            alignment=self.alignment,
            transform=transform,
            tied_to=tied_to,
            quant=quant,
        )
        self.entries.append(entry)
        return entry

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "weights.bin").write_bytes(bytes(self._data))
        write_json(self.output_dir / "manifest.json", {
            "format": "devproc2.weights",
            "format_version": 1,
            "model": self.model,
            "precision": self.precision,
            "data_file": "weights.bin",
            "index_file": "weights.index.json",
            "weight_map_file": "weight_map.json",
        })
        write_json(self.output_dir / "weights.index.json", {
            "format_version": 1,
            "data_file": "weights.bin",
            "entries": [entry.to_index_obj() for entry in self.entries],
        })
        write_json(self.output_dir / "weight_map.json", {
            "format_version": 1,
            "weights": [entry.to_weight_map_obj() for entry in self.entries],
        })
        write_json(self.output_dir / "quantization.json", {
            "format_version": 1,
            "entries": [
                {
                    "name": entry.name,
                    "shape": list(entry.shape),
                    "dtype": entry.dtype,
                    "quant": asdict(entry.quant),
                }
                for entry in self.entries
                if entry.quant is not None
            ],
        })

    def _align(self, value: int) -> int:
        rem = value % self.alignment
        return value if rem == 0 else value + (self.alignment - rem)


def read_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_package(path: str | Path) -> None:
    package_dir = Path(path)
    manifest = read_manifest(package_dir / "manifest.json")
    for key in ("data_file", "index_file", "weight_map_file"):
        value = manifest.get(key)
        if not isinstance(value, str) or not (package_dir / value).exists():
            raise FileNotFoundError(package_dir / str(value))


def tensor_to_bytes(tensor: Any, dtype: str | None) -> tuple[bytes, tuple[int, ...], str]:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(tensor, torch.Tensor):
        t = tensor.detach().contiguous()
        resolved = dtype or _torch_dtype_name(t)
        if resolved == BF16:
            raw = t.to(torch.bfloat16).view(torch.uint16).cpu().numpy().tobytes()
        elif resolved == FP16:
            raw = t.to(torch.float16).cpu().numpy().tobytes()
        elif resolved == FP32:
            raw = t.to(torch.float32).cpu().numpy().tobytes()
        elif resolved == FP8_E4M3:
            if t.dtype == torch.float8_e4m3fn:
                raw = t.view(torch.uint8).cpu().numpy().tobytes()
            else:
                raw = t.to(torch.uint8).cpu().numpy().tobytes()
        else:
            raw = t.cpu().numpy().tobytes()
        return raw, tuple(int(s) for s in t.shape), resolved

    arr = np.asarray(tensor)
    resolved = dtype or str(arr.dtype)
    if resolved == BF16:
        if arr.dtype != np.uint16:
            raise TypeError("numpy bfloat16 payload must be provided as uint16 bit pattern")
        raw = np.ascontiguousarray(arr).tobytes()
    elif resolved == FP8_E4M3:
        raw = np.ascontiguousarray(arr.astype(np.uint8, copy=False)).tobytes()
    else:
        raw = np.ascontiguousarray(arr).tobytes()
    return raw, tuple(int(s) for s in arr.shape), resolved


def _torch_dtype_name(tensor: Any) -> str:
    import torch

    if tensor.dtype == torch.bfloat16:
        return BF16
    if tensor.dtype == torch.float16:
        return FP16
    if tensor.dtype == torch.float32:
        return FP32
    if tensor.dtype == torch.float8_e4m3fn:
        return FP8_E4M3
    return str(tensor.dtype).removeprefix("torch.")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "BF16",
    "FP16",
    "FP32",
    "FP8_E4M3",
    "QuantSpec",
    "WeightEntry",
    "WeightPackageWriter",
    "read_manifest",
    "tensor_to_bytes",
    "validate_package",
    "write_json",
]
