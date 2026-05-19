"""Pi0.5 deploy weight specs and package writer."""
from __future__ import annotations

from pathlib import Path


from devproc2.quantization import (
    FusionComponentSpec,
    FusionGroupSpec,
    QuantTensorSpec,
    QuantizationManifest,
)
from devproc2.weights import (
    BF16,
    FP16,
    FP32,
    FP8_E4M3,
    QuantSpec,
    WeightEntry,
    WeightPackageWriter as _WeightPackageWriter,
)


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


class WeightPackageWriter(_WeightPackageWriter):
    """Pi0.5 convenience wrapper over the generic weight package writer."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model: str = PI05_MODEL_NAME,
        precision: str = BF16,
        alignment: int = 256,
    ) -> None:
        super().__init__(
            output_dir,
            model=model,
            precision=precision,
            alignment=alignment,
        )
