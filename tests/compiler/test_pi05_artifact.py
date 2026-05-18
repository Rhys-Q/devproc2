import json

import numpy as np

from devproc2.models.pi05 import Pi05ArtifactSummary, prepare_pi05_artifact
from devproc2.models.pi05.weights import QuantSpec, WeightPackageWriter


def _write_weight_package(path):
    writer = WeightPackageWriter(path, precision="fp8")
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
    (path / "convert_report.json").write_text(json.dumps({"fp8_layout": "nk"}))
    return path


def test_prepare_pi05_artifact_copies_weights_metadata_and_tokenizer(tmp_path):
    weights_dir = _write_weight_package(tmp_path / "weights_pkg")

    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fake-tokenizer")
    artifact_dir = tmp_path / "artifact"
    (artifact_dir / "metadata").mkdir(parents=True)
    (artifact_dir / "metadata" / "pi05_kernel_catalog.json").write_text("[]")

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
    assert not (artifact_dir / "metadata" / "pi05_kernel_catalog.json").exists()
    assert (artifact_dir / "resources" / "tokenizer.model").read_bytes() == b"fake-tokenizer"
    assert not (artifact_dir / "metadata" / "kernel_table.json").exists()

    pi05_manifest = json.loads((artifact_dir / "metadata" / "pi05_artifact.json").read_text())
    assert pi05_manifest["weights"]["precision"] == "fp8"
    assert pi05_manifest["weights"]["entries"] == 2
    assert pi05_manifest["weights"]["fp8_layout"] == "nk"
    assert pi05_manifest["tokenizer"] == "resources/tokenizer.model"
    assert pi05_manifest["kernels"]["compiled"] is False
    assert pi05_manifest["kernels"]["count"] == 0
    assert pi05_manifest["kernels"]["table"] is None


def test_prepare_pi05_artifact_uses_emitted_kernel_table(tmp_path):
    weights_dir = _write_weight_package(tmp_path / "weights_pkg")
    artifact_dir = tmp_path / "artifact"
    metadata_dir = artifact_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    source = tmp_path / "pi05_kernels.cu"
    source.write_text("__global__ void pi05_noop() {}\n")
    kernel_table = [
        {
            "name": "kernel.pi05_noop",
            "kind": "kernel",
            "backend": "cuda",
            "op": "cuda.pi05_noop",
            "symbol": "pi05_noop",
            "source": str(source),
            "launch": {
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "shared_memory_bytes": 0,
            },
        },
    ]
    (metadata_dir / "kernel_table.json").write_text(json.dumps(kernel_table))

    summary = prepare_pi05_artifact(
        weight_package_dir=weights_dir,
        artifact_dir=artifact_dir,
        tokenizer_model_path=None,
        sm_arch=89,
        compile_kernels=False,
    )

    assert summary.kernels == 1
    assert json.loads((metadata_dir / "kernel_table.json").read_text()) == kernel_table
    pi05_manifest = json.loads((metadata_dir / "pi05_artifact.json").read_text())
    assert pi05_manifest["kernels"]["count"] == 1
    assert pi05_manifest["kernels"]["table"] == "metadata/kernel_table.json"
