"""Pi0.5 deploy weight specs and package writer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from devproc2.quantization import (
    FusionComponentSpec,
    FusionGroupSpec,
    QuantTensorSpec,
    QuantizationManifest,
)


BF16 = "bfloat16"
FP16 = "float16"
FP32 = "float32"
FP8_E4M3 = "fp8_e4m3"
PI05_MODEL_NAME = "open" + "pi0.5"

VIS_L = 27
VIS_D = 1152
VIS_H = 4304
ENC_L = 18
ENC_D = 2048
ENC_H = 16384
DEC_L = 18
DEC_D = 1024
DEC_H = 4096
ACTION_DIM = 32
ACTION_HORIZON = 50
NUM_STEPS_DEFAULT = 10


def pi05_act_scale_name(logical_name: str, layer_idx: int | None = None) -> str:
    suffix = f"_{layer_idx}" if layer_idx is not None else ""
    return f"act_scale.{logical_name}{suffix}"


def pi05_fp8_scale_name(logical_name: str, layer_idx: int | None = None) -> str:
    suffix = f"_{layer_idx}" if layer_idx is not None else ""
    return f"fp8.{logical_name}{suffix}.scale"


def pi05_fp8_weight_name(logical_name: str, layer_idx: int | None = None) -> str:
    suffix = f"_{layer_idx}" if layer_idx is not None else ""
    return f"fp8.{logical_name}{suffix}.weight"


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


def select_fp8_layout(hardware: str | None = None, fp8_layout: str | None = None) -> str:
    if fp8_layout is not None:
        if fp8_layout not in ("kn", "nk"):
            raise ValueError(f"fp8_layout must be 'kn' or 'nk', got {fp8_layout!r}")
        return fp8_layout
    if hardware == "rtx_sm89":
        return "nk"
    if hardware == "rtx_sm120":
        return "kn"
    return "kn"


def pi05_deploy_quantization_manifest(*, fp8_layout: str = "nk") -> QuantizationManifest:
    tensors: list[QuantTensorSpec] = []
    groups: list[FusionGroupSpec] = []

    def add_tensor(logical_name: str, layer_idx: int | None = None) -> None:
        suffix = f"_{layer_idx}" if layer_idx is not None else ""
        source = f"{logical_name}{suffix}"
        tensors.append(
            QuantTensorSpec(
                name=pi05_fp8_weight_name(logical_name, layer_idx),
                dtype=FP8_E4M3,
                source=source,
                quantizer="fp8_e4m3_per_tensor",
            )
        )

    def add_qkv_group(prefix: str, layer_idx: int) -> None:
        add_tensor(f"{prefix}_attn_qkv_w", layer_idx)
        groups.append(
            FusionGroupSpec(
                name=f"{prefix}_attn_qkv_w_{layer_idx}",
                output_name=pi05_fp8_weight_name(f"{prefix}_attn_qkv_w", layer_idx),
                components=(
                    FusionComponentSpec(f"{prefix}_attn_q_w_{layer_idx}", "q"),
                    FusionComponentSpec(f"{prefix}_attn_k_w_{layer_idx}", "k"),
                    FusionComponentSpec(f"{prefix}_attn_v_w_{layer_idx}", "v"),
                ),
                layout=fp8_layout,
            )
        )

    def add_gate_up_group(prefix: str, layer_idx: int) -> None:
        add_tensor(f"{prefix}_ffn_gate_up_w", layer_idx)
        groups.append(
            FusionGroupSpec(
                name=f"{prefix}_ffn_gate_up_w_{layer_idx}",
                output_name=pi05_fp8_weight_name(f"{prefix}_ffn_gate_up_w", layer_idx),
                components=(
                    FusionComponentSpec(f"{prefix}_ffn_gate_w_{layer_idx}", "gate"),
                    FusionComponentSpec(f"{prefix}_ffn_up_w_{layer_idx}", "up"),
                ),
                layout=fp8_layout,
            )
        )

    for layer_idx in range(VIS_L):
        add_qkv_group("vision", layer_idx)
        add_tensor("vision_attn_o_w", layer_idx)
        add_tensor("vision_ffn_up_w", layer_idx)
        add_tensor("vision_ffn_down_w", layer_idx)
    add_tensor("vision_projector_w")

    for layer_idx in range(ENC_L):
        add_qkv_group("encoder", layer_idx)
        add_tensor("encoder_attn_o_w", layer_idx)
        add_gate_up_group("encoder", layer_idx)
        add_tensor("encoder_ffn_down_w", layer_idx)

    for layer_idx in range(DEC_L):
        add_qkv_group("decoder", layer_idx)
        add_tensor("decoder_attn_o_w", layer_idx)
        add_gate_up_group("decoder", layer_idx)
        add_tensor("decoder_ffn_down_w", layer_idx)

    return QuantizationManifest(
        tensors=tuple(tensors),
        fusion_groups=tuple(groups),
        metadata={"fp8_layout": fp8_layout},
    )


class WeightPackageWriter:
    """Write a devproc2 self-contained weight package."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model: str = PI05_MODEL_NAME,
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
        raw, shape, resolved_dtype = _tensor_to_bytes(tensor, dtype)
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
        _write_json(self.output_dir / "manifest.json", {
            "format": "devproc2.weights",
            "format_version": 1,
            "model": self.model,
            "precision": self.precision,
            "data_file": "weights.bin",
            "index_file": "weights.index.json",
            "weight_map_file": "weight_map.json",
        })
        _write_json(self.output_dir / "weights.index.json", {
            "format_version": 1,
            "data_file": "weights.bin",
            "entries": [entry.to_index_obj() for entry in self.entries],
        })
        _write_json(self.output_dir / "weight_map.json", {
            "format_version": 1,
            "weights": [entry.to_weight_map_obj() for entry in self.entries],
        })
        _write_json(self.output_dir / "quantization.json", {
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


def _tensor_to_bytes(tensor: Any, dtype: str | None) -> tuple[bytes, tuple[int, ...], str]:
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
