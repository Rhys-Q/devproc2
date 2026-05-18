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

from devproc2.kernel import KernelLaunchSpec, KernelSpec
from devproc2.kernel.provider import CudaSourceProvider


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

    stale_catalog = metadata_dir / "pi05_kernel_catalog.json"
    if stale_catalog.exists():
        stale_catalog.unlink()

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

    specs = _load_cuda_kernel_specs(metadata_dir / "kernel_table.json", sm_arch=sm_arch)

    if compile_kernels:
        if not specs:
            raise FileNotFoundError(
                metadata_dir / "kernel_table.json",
            )
        _compile_pi05_kernel_cubins(specs, artifact_dir, sm_arch=sm_arch, nvcc=nvcc)

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
        "format": "devproc2.artifact.pi05",
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
            "table": "metadata/kernel_table.json" if specs else None,
        },
    })
    return summary


def _load_cuda_kernel_specs(path: Path, *, sm_arch: int) -> list[KernelSpec]:
    table = _read_json(path)
    if not isinstance(table, list):
        return []
    specs: list[KernelSpec] = []
    for entry in table:
        if not isinstance(entry, dict) or entry.get("backend") != "cuda":
            continue
        source = entry.get("source")
        if not source:
            continue
        launch_obj = entry.get("launch")
        launch = _launch_from_json(launch_obj) if isinstance(launch_obj, dict) else KernelLaunchSpec()
        specs.append(
            KernelSpec(
                op_name=str(entry.get("op", f"cuda.{entry.get('symbol', '')}")),
                device="cuda",
                input_dtypes=(),
                kernel_name=str(entry["name"]),
                backend="cuda",
                output_dtype=entry.get("output_dtype"),
                symbol=str(entry.get("symbol", entry["name"])).removeprefix("kernel."),
                sm_arches=(sm_arch,),
                launch=launch,
                source_path=str(source),
                extra_nvcc_flags=("--std=c++17",),
            )
        )
    return specs


def _launch_from_json(data: dict[str, Any]) -> KernelLaunchSpec:
    grid = data.get("grid", (1, 1, 1))
    if not _launch_tuple_is_plain_ints(grid):
        grid = (1, 1, 1)
    return KernelLaunchSpec(
        grid=tuple(grid),
        block=tuple(data.get("block", (256, 1, 1))),
        shared_memory_bytes=int(data.get("shared_memory_bytes", 0)),
    )


def _launch_tuple_is_plain_ints(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, int) for item in value)
    )


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
