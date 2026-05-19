from __future__ import annotations

import json
import subprocess
import sys

import devproc2 as dp
import devproc2.nn as nn
import numpy as np
import pytest
from devproc2.artifact import PackedBackendRecipe, prepare_artifact
from devproc2.build import build
from devproc2.export.recipe import EntrypointRecipe
from devproc2.nn.specs import Parameter
from devproc2.weights import WeightPackageWriter


class TinyWeightedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Parameter((2, 3), "float32", name="tiny.weight")

    def forward(self, x):
        return dp.add(x, self.weight)


class TinyStaticScaleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = Parameter((1,), "float32", role="constant_tensor", name="act_scale.foo")

    def forward(self, x):
        return dp.multiply(x, self.scale)


def _tiny_recipe(module_cls=TinyWeightedModule) -> EntrypointRecipe:
    return EntrypointRecipe(
        name="main",
        model_id="tiny",
        build_module=lambda opts: module_cls(),
        input_specs=lambda opts: {"x": nn.TensorSpec((2, 3), "float32")},
    )


def _write_weight_package(path, *, include_weight: bool = True, include_act_scale: bool = False):
    writer = WeightPackageWriter(path, model="tiny", precision="float32")
    if include_weight:
        writer.add_tensor("tiny.weight", np.ones((2, 3), dtype=np.float32), dtype="float32")
    if include_act_scale:
        writer.add_tensor(
            "act_scale.foo",
            np.asarray([1.0], dtype=np.float32),
            dtype="float32",
            kind="constant_tensor",
            layout="scalar",
        )
    writer.write()
    return path


def test_build_api_writes_build_config_and_weight_binding_metadata(tmp_path):
    weights = _write_weight_package(tmp_path / "weights")
    (weights / "convert_report.json").write_text(
        json.dumps(
            {
                "source": {"type": "unit", "path": "/tmp/tiny.safetensors"},
                "target_hardware": "unit_cpu",
                "shape_profile": "tiny_profile",
                "action_horizon": 5,
                "num_steps": 2,
            }
        )
    )
    artifact = tmp_path / "artifact"

    summary = build(
        recipe=_tiny_recipe(),
        weights=weights,
        artifact_dir=artifact,
        compile_mode="normal",
        compile_kernels=False,
        build_backends="never",
    )

    assert summary.weight_validation is not None
    assert summary.weight_validation.bound_weights == 1
    assert (artifact / "metadata" / "build.json").exists()
    assert (artifact / "metadata" / "config.json").exists()
    binding = json.loads((artifact / "metadata" / "weight_binding.json").read_text())
    assert binding["bound_weights"] == 1
    assert binding["package_precision"] == "float32"
    assert binding["source_checkpoint"] == "/tmp/tiny.safetensors"
    assert binding["target_hardware"] == "unit_cpu"
    assert binding["shape_profile"] == "tiny_profile"
    assert binding["action_horizon"] == 5
    assert binding["num_steps"] == 2
    assert binding["bindings"][0]["name"] == "tiny.weight"
    manifest = json.loads((artifact / "metadata" / "artifact.json").read_text())
    assert manifest["metadata"]["build"] == "metadata/build.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "devproc2.inspect",
            str(artifact),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0
    assert "devproc2.artifact" in result.stdout


def test_build_weight_validation_reports_missing_tensor(tmp_path):
    weights = _write_weight_package(tmp_path / "weights", include_weight=False)

    with pytest.raises(ValueError, match="missing required weights: tiny.weight"):
        build(
            recipe=_tiny_recipe(),
            weights=weights,
            artifact_dir=tmp_path / "artifact",
            compile_mode="normal",
            compile_kernels=False,
            build_backends="never",
        )


def test_static_activation_scale_build_fails_when_package_declares_missing(tmp_path):
    weights = _write_weight_package(tmp_path / "weights", include_weight=False)
    (weights / "convert_report.json").write_text(
        json.dumps(
            {
                "activation_scales": "missing",
                "supports_static_act_scales": False,
            }
        )
    )

    with pytest.raises(ValueError, match="static activation scales requested"):
        build(
            recipe=_tiny_recipe(TinyStaticScaleModule),
            weights=weights,
            artifact_dir=tmp_path / "artifact",
            options={"use_static_act_scales": True},
            compile_mode="normal",
            compile_kernels=False,
            build_backends="never",
        )


def test_artifact_builder_does_not_implicitly_build_compiled_backend(tmp_path):
    weights = _write_weight_package(tmp_path / "weights")
    backend_build_dir = tmp_path / "cmake-build"
    backend_build_dir.mkdir()
    (backend_build_dir / "CMakeCache.txt").write_text("fake")
    backend = PackedBackendRecipe(
        name="test.backend",
        kind="compiled_packed_backend",
        sources=("src/backend.cc",),
    )

    with pytest.raises(FileNotFoundError, match="devproc2 build --build-backends auto"):
        prepare_artifact(
            model_id="tiny",
            entrypoint="main",
            artifact_dir=tmp_path / "artifact",
            weight_package_dir=weights,
            packed_backends=(backend,),
            backend_build_dir=backend_build_dir,
            compile_kernels=False,
        )
