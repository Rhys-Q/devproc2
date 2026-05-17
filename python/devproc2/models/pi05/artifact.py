"""Pi0.5 artifact resource packaging."""
from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from devproc2.kernel.provider import CudaSourceProvider
from devproc2.models.pi05.kernels import pi05_kernel_specs


_DEFAULT_TOKENIZER = Path("/root/autodl-tmp/openpi/outputs/pi05_torch_infer/tokenizer.model")


@dataclass(frozen=True)
class Pi05ArtifactSummary:
    artifact_dir: Path
    weights_entries: int
    kernels: int
    tokenizer: str | None
    fp8_layout: str | None


def prepare_pi05_artifact(
    *,
    weight_package_dir: str | Path,
    artifact_dir: str | Path,
    tokenizer_model_path: str | Path | None = _DEFAULT_TOKENIZER,
    sm_arch: int = 89,
    compile_kernels: bool = True,
    nvcc: str | None = None,
) -> Pi05ArtifactSummary:
    """Install Pi0.5 runtime resources into a devproc2 artifact directory.

    This is intentionally separate from VM bytecode emission: the compiler can
    emit ``executable.vm``/``abi.json`` first, then this function makes the
    artifact self-contained by adding weights, tokenizer resources, quant
    metadata and CUDA kernel cubins/catalog metadata.
    """

    weight_package_dir = Path(weight_package_dir)
    artifact_dir = Path(artifact_dir)
    metadata_dir = artifact_dir / "metadata"
    weights_dir = artifact_dir / "weights"
    resources_dir = artifact_dir / "resources"
    kernels_dir = artifact_dir / "kernels"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    resources_dir.mkdir(exist_ok=True)
    kernels_dir.mkdir(exist_ok=True)

    manifest = _read_json_required(weight_package_dir / "manifest.json")
    index = _read_json_required(weight_package_dir / manifest.get("index_file", "weights.index.json"))
    weight_map_name = manifest.get("weight_map_file", "weight_map.json")
    quant_path = weight_package_dir / "quantization.json"
    report_path = weight_package_dir / "convert_report.json"

    data_file = str(manifest.get("data_file", "weights.bin"))
    _copy_required(weight_package_dir / data_file, weights_dir / "weights.bin")
    _copy_required(weight_package_dir / manifest.get("index_file", "weights.index.json"),
                   weights_dir / "weights.index.json")
    _copy_required(weight_package_dir / weight_map_name, metadata_dir / "weight_map.json")
    if quant_path.exists():
        shutil.copy2(quant_path, metadata_dir / "quantization.json")
    if report_path.exists():
        shutil.copy2(report_path, metadata_dir / "convert_report.json")

    tokenizer_rel: str | None = None
    if tokenizer_model_path is not None:
        tokenizer_src = Path(tokenizer_model_path)
        if tokenizer_src.exists():
            _copy_required(tokenizer_src, resources_dir / "tokenizer.model")
            tokenizer_rel = "resources/tokenizer.model"
            _write_json(metadata_dir / "tokenizer.json", {
                "kind": "sentencepiece",
                "model": "paligemma",
                "path": tokenizer_rel,
                "sha256": _sha256(resources_dir / "tokenizer.model"),
            })

    specs = pi05_kernel_specs(sm_arch=sm_arch)
    kernel_catalog = [spec.to_json_obj() for spec in specs]
    _write_json(metadata_dir / "pi05_kernel_catalog.json", kernel_catalog)

    if compile_kernels:
        _compile_pi05_kernel_cubins(specs, artifact_dir, sm_arch=sm_arch, nvcc=nvcc)
        _merge_kernel_table(metadata_dir / "kernel_table.json", kernel_catalog)

    weights_entries = len(index.get("entries", []))
    report = _read_json(report_path)
    fp8_layout = report.get("fp8_layout") if isinstance(report, dict) else None
    summary = Pi05ArtifactSummary(
        artifact_dir=artifact_dir,
        weights_entries=weights_entries,
        kernels=len(specs),
        tokenizer=tokenizer_rel,
        fp8_layout=fp8_layout,
    )

    _write_json(metadata_dir / "pi05_artifact.json", {
        "format": "devproc2.models.pi05.artifact",
        "format_version": 1,
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "openpi0.5",
        "target": "cuda",
        "sm_arch": sm_arch,
        "weights": {
            "path": "weights",
            "entries": weights_entries,
            "precision": manifest.get("precision"),
            "sha256": _sha256(weights_dir / "weights.bin"),
            "index": "weights/weights.index.json",
            "weight_map": "metadata/weight_map.json",
            "quantization": (
                "metadata/quantization.json"
                if (metadata_dir / "quantization.json").exists()
                else None
            ),
            "fp8_layout": fp8_layout,
        },
        "tokenizer": tokenizer_rel,
        "kernels": {
            "count": len(specs),
            "compiled": compile_kernels,
            "table": (
                "metadata/kernel_table.json"
                if compile_kernels
                else "metadata/pi05_kernel_catalog.json"
            ),
        },
    })
    return summary


def _compile_pi05_kernel_cubins(
    specs,
    artifact_dir: Path,
    *,
    sm_arch: int,
    nvcc: str | None,
) -> None:
    if not specs:
        return
    first = specs[0]
    if nvcc:
        first = first.__class__(
            **{
                **first.__dict__,
                "compile_options": {**dict(first.compile_options), "nvcc": nvcc},
            }
        )
    result = CudaSourceProvider().compile(
        first,
        None,
        output_dir=str(artifact_dir),
        sm_arch=sm_arch,
    )
    cubin_bytes = result.data
    for spec in specs[1:]:
        path = artifact_dir / str(spec.cubin_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cubin_bytes)


def _merge_kernel_table(path: Path, entries: list[dict[str, Any]]) -> None:
    existing = _read_json(path)
    table: list[dict[str, Any]]
    if isinstance(existing, list):
        table = list(existing)
    else:
        table = []
    by_name = {str(entry.get("name")): entry for entry in table if isinstance(entry, dict)}
    for entry in entries:
        by_name[str(entry["name"])] = entry
    _write_json(path, list(by_name.values()))


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
