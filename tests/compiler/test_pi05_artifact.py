import json

import numpy as np

from devproc2.pi05 import Pi05ArtifactSummary, prepare_pi05_artifact, pi05_kernel_specs
from devproc2.pi05.weights import QuantSpec, WeightPackageWriter


def test_pi05_kernel_specs_are_concrete_cuda_specs():
    specs = pi05_kernel_specs(sm_arch=89)

    assert specs
    assert all(spec.backend == "cuda" for spec in specs)
    assert all(spec.source_path and spec.source_path.endswith("pi05_kernels.cu") for spec in specs)
    assert all(spec.cubin_path and spec.cubin_path.startswith("kernels/") for spec in specs)
    assert {spec.sm_arches for spec in specs} == {(89,)}


def test_prepare_pi05_artifact_copies_weights_metadata_and_tokenizer(tmp_path):
    weights_dir = tmp_path / "weights_pkg"
    writer = WeightPackageWriter(weights_dir, precision="fp8")
    writer.add_tensor("fp8.test.scale", np.array([1.0], dtype=np.float32), kind="scale")
    writer.add_tensor(
        "fp8.test.weight",
        np.arange(8, dtype=np.uint8),
        dtype="fp8_e4m3",
        quant=QuantSpec(
            scheme="fp8_e4m3_per_tensor",
            storage_dtype="fp8_e4m3",
            compute_dtype="bfloat16",
            scale_name="fp8.test.scale",
            packed_layout="nk",
        ),
    )
    writer.write()
    (weights_dir / "convert_report.json").write_text(json.dumps({"fp8_layout": "nk"}))

    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fake-tokenizer")
    artifact_dir = tmp_path / "artifact"

    summary = prepare_pi05_artifact(
        weight_package_dir=weights_dir,
        artifact_dir=artifact_dir,
        tokenizer_model_path=tokenizer,
        sm_arch=89,
        compile_kernels=False,
    )

    assert isinstance(summary, Pi05ArtifactSummary)
    assert summary.weights_entries == 2
    assert summary.fp8_layout == "nk"
    assert (artifact_dir / "weights" / "weights.bin").exists()
    assert (artifact_dir / "weights" / "weights.index.json").exists()
    assert (artifact_dir / "metadata" / "weight_map.json").exists()
    assert (artifact_dir / "metadata" / "quantization.json").exists()
    assert (artifact_dir / "metadata" / "pi05_kernel_catalog.json").exists()
    assert (artifact_dir / "resources" / "tokenizer.model").read_bytes() == b"fake-tokenizer"
    assert not (artifact_dir / "metadata" / "kernel_table.json").exists()

    pi05_manifest = json.loads((artifact_dir / "metadata" / "pi05_artifact.json").read_text())
    assert pi05_manifest["weights"]["precision"] == "fp8"
    assert pi05_manifest["weights"]["entries"] == 2
    assert pi05_manifest["weights"]["fp8_layout"] == "nk"
    assert pi05_manifest["tokenizer"] == "resources/tokenizer.model"
    assert pi05_manifest["kernels"]["compiled"] is False
