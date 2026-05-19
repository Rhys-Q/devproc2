import json

import numpy as np

from devproc2.artifact import (
    ArtifactBuildSummary,
    PackedBackendRecipe,
    PackedFuncSpec,
    ResourceSpec,
    prepare_artifact,
)
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

    summary = prepare_artifact(
        model_id="openpi0.5",
        entrypoint="sample_tokens",
        weight_package_dir=weights_dir,
        artifact_dir=artifact_dir,
        resources=(
            ResourceSpec(
                name="tokenizer",
                path=tokenizer,
                target_path="resources/tokenizer.model",
                metadata={"tokenizer_model": "paligemma"},
            ),
        ),
        sm_arch=89,
        compile_kernels=False,
    )

    assert isinstance(summary, ArtifactBuildSummary)
    assert summary.weights_entries == 2
    assert summary.fp8_layout == "nk"
    assert (artifact_dir / "weights" / "weights.bin").exists()
    assert (artifact_dir / "weights" / "weights.index.json").exists()
    assert (artifact_dir / "metadata" / "weight_map.json").exists()
    assert (artifact_dir / "metadata" / "quantization.json").exists()
    assert (artifact_dir / "resources" / "tokenizer.model").read_bytes() == b"fake-tokenizer"
    assert not (artifact_dir / "metadata" / "kernel_table.json").exists()

    manifest = json.loads((artifact_dir / "metadata" / "artifact.json").read_text())
    assert manifest["format"] == "devproc2.artifact"
    assert manifest["model_id"] == "openpi0.5"
    assert manifest["entrypoint"] == "sample_tokens"
    assert manifest["weights"]["precision"] == "fp8"
    assert manifest["weights"]["entries"] == 2
    assert manifest["weights"]["fp8_layout"] == "nk"
    assert manifest["resources"][0]["path"] == "resources/tokenizer.model"
    assert manifest["kernels"]["compiled"] is False
    assert manifest["kernels"]["count"] == 0
    assert manifest["kernels"]["table"] is None


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

    summary = prepare_artifact(
        model_id="openpi0.5",
        entrypoint="sample_tokens",
        weight_package_dir=weights_dir,
        artifact_dir=artifact_dir,
        sm_arch=89,
        compile_kernels=False,
    )

    assert summary.kernels == 1
    assert json.loads((metadata_dir / "kernel_table.json").read_text()) == kernel_table
    manifest = json.loads((metadata_dir / "artifact.json").read_text())
    assert manifest["kernels"]["count"] == 1
    assert manifest["kernels"]["table"] == "metadata/kernel_table.json"


def test_prepare_artifact_writes_packed_backend_table(tmp_path):
    weights_dir = _write_weight_package(tmp_path / "weights_pkg")
    artifact_dir = tmp_path / "artifact"
    backend = PackedBackendRecipe(
        name="test.backend",
        kind="linked_packed_backend",
        sources=("src/backend.cc",),
        register_symbol="devproc2_register_test_backend",
        packed_funcs=(PackedFuncSpec("test.backend.echo", device="cuda"),),
    )

    prepare_artifact(
        model_id="unit",
        entrypoint="main",
        weight_package_dir=weights_dir,
        artifact_dir=artifact_dir,
        sm_arch=89,
        compile_kernels=False,
        packed_backends=(backend,),
    )

    table = json.loads((artifact_dir / "metadata" / "packed_backend_table.json").read_text())
    assert table[0]["kind"] == "linked_packed_backend"
    assert table[0]["library"] is None
    assert table[0]["packed_funcs"][0]["name"] == "test.backend.echo"
    manifest = json.loads((artifact_dir / "metadata" / "artifact.json").read_text())
    assert manifest["packed_backends"] == table


def test_prepare_artifact_installs_compiled_packed_backend(tmp_path):
    weights_dir = _write_weight_package(tmp_path / "weights_pkg")
    artifact_dir = tmp_path / "artifact"
    backend_dir = tmp_path / "backend_build"
    backend_dir.mkdir()
    backend_lib = backend_dir / "libdevproc2_test_backend_backend.so"
    backend_lib.write_bytes(b"fake-shared-library")
    backend = PackedBackendRecipe(
        name="test.backend",
        kind="compiled_packed_backend",
        sources=("src/backend.cc",),
        register_symbol="devproc2_register_test_backend",
        packed_funcs=(PackedFuncSpec("test.backend.echo", device="cuda"),),
    )

    prepare_artifact(
        model_id="unit",
        entrypoint="main",
        weight_package_dir=weights_dir,
        artifact_dir=artifact_dir,
        sm_arch=89,
        compile_kernels=False,
        backend_library_dirs=(backend_dir,),
        packed_backends=(backend,),
    )

    assert (artifact_dir / "backends" / "test_backend.so").read_bytes() == b"fake-shared-library"
    table = json.loads((artifact_dir / "metadata" / "packed_backend_table.json").read_text())
    assert table[0]["kind"] == "compiled_packed_backend"
    assert table[0]["library"] == "backends/test_backend.so"
