from __future__ import annotations

import json

import numpy as np

from devproc2.models.pi05.weights import (
    BF16,
    FP8_E4M3,
    QuantSpec,
    WeightPackageWriter,
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
