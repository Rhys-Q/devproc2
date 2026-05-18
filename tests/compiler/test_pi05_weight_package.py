from __future__ import annotations

import json

import numpy as np

from devproc2.quantization import (
    CalibrationManifest,
    FusionComponentSpec,
    FusionGroupSpec,
    QuantTensorSpec,
    QuantizationManifest,
    common_fp8_scale,
    quantize_e4m3_reference,
)
from devproc2.models.pi05.weights import (
    BF16,
    FP8_E4M3,
    QuantSpec,
    WeightPackageWriter,
    pi05_deploy_quantization_manifest,
    pi05_fp8_scale_name,
    pi05_fp8_weight_name,
    select_fp8_layout,
)


def test_weight_package_writer_alignment_and_metadata(tmp_path):
    writer = WeightPackageWriter(tmp_path, precision="bf16+fp8")
    writer.add_tensor(
        "bf16.weight",
        np.arange(6, dtype=np.uint16).reshape(2, 3),
        dtype=BF16,
    )
    writer.add_tensor(
        "fp8.weight.scale",
        np.asarray([0.125], dtype=np.float32),
        dtype="float32",
        kind="scale",
        layout="scalar",
    )
    writer.add_tensor(
        "fp8.weight",
        np.arange(6, dtype=np.uint8).reshape(3, 2),
        dtype=FP8_E4M3,
        layout="nk",
        quant=QuantSpec(
            scheme="fp8_e4m3_per_tensor",
            storage_dtype=FP8_E4M3,
            compute_dtype=BF16,
            scale_name="fp8.weight.scale",
            packed_layout="nk",
        ),
    )
    writer.write()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    index = json.loads((tmp_path / "weights.index.json").read_text())
    weight_map = json.loads((tmp_path / "weight_map.json").read_text())
    quant = json.loads((tmp_path / "quantization.json").read_text())

    assert manifest["format"] == "devproc2.weights"
    assert (tmp_path / "weights.bin").exists()
    offsets = [entry["offset"] for entry in index["entries"]]
    assert offsets == [0, 256, 512]
    assert weight_map["weights"][2]["quant"]["scheme"] == "fp8_e4m3_per_tensor"
    assert weight_map["weights"][2]["quant"]["packed_layout"] == "nk"
    assert quant["entries"][0]["name"] == "fp8.weight"


def test_select_fp8_layout_defaults_to_sm89_nk():
    assert select_fp8_layout("rtx_sm89") == "nk"
    assert select_fp8_layout("rtx_sm120") == "kn"
    assert select_fp8_layout(fp8_layout="nk") == "nk"


def test_pi05_deploy_quantization_manifest_declares_fused_fp8_specs():
    manifest = pi05_deploy_quantization_manifest(fp8_layout="nk")
    payload = manifest.to_json_obj()
    groups = {group["name"]: group for group in payload["fusion_groups"]}
    tensors = {tensor["name"]: tensor for tensor in payload["tensors"]}

    assert pi05_fp8_scale_name("decoder_attn_qkv_w", 0) == "fp8.decoder_attn_qkv_w_0.scale"
    assert pi05_fp8_weight_name("decoder_attn_qkv_w", 0) in tensors
    group = groups["decoder_attn_qkv_w_0"]
    assert group["output_name"] == "fp8.decoder_attn_qkv_w_0.weight"
    assert [component["logical_name"] for component in group["components"]] == ["q", "k", "v"]
    assert group["scale_policy"] == "max_component_amax"
    assert group["requantize_from"] == "source_fp"


def test_quantization_manifest_and_fp8_requant_helpers_are_framework_only():
    scale = common_fp8_scale([np.asarray([1.0, -224.0], dtype=np.float32), 448.0])
    payload = quantize_e4m3_reference(np.asarray([-999.0, 0.0, 999.0], dtype=np.float32), scale)
    manifest = QuantizationManifest(
        tensors=(
            QuantTensorSpec(
                name="fp8.weight",
                dtype=FP8_E4M3,
                scale=scale,
                amax=448.0,
                source="linear.weight",
            ),
        ),
        calibration=CalibrationManifest(dataset="unit", sample_count=1, sample_hash="abc"),
        fusion_groups=(
            FusionGroupSpec(
                name="qkv",
                output_name="fp8.qkv.weight",
                components=(
                    FusionComponentSpec("q.weight", "q", amax=1.0),
                    FusionComponentSpec("k.weight", "k", amax=2.0),
                ),
                layout="nk",
            ),
        ),
    )

    assert scale == 1.0
    assert payload.dtype == np.uint8
    assert manifest.to_json_obj()["fusion_groups"][0]["scale_policy"] == "max_component_amax"
