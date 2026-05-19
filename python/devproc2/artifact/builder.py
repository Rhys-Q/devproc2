"""Generic artifact resource packaging."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devproc2.artifact.manifest import PackedBackendRecipe, ResourceSpec
from devproc2.kernel import KernelLaunchSpec, KernelSpec
from devproc2.kernel.provider import CudaSourceProvider


@dataclass(frozen=True)
class ArtifactBuildSummary:
    artifact_dir: Path
    weights_entries: int
    kernels: int
    resources: tuple[str, ...]
    fp8_layout: str | None
    packed_backends: tuple[str, ...] = ()

    @property
    def tokenizer(self) -> str | None:
        for resource in self.resources:
            if resource.endswith("tokenizer.model"):
                return resource
        return None


def prepare_artifact(
    *,
    model_id: str,
    entrypoint: str,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    resources: tuple[ResourceSpec, ...] = (),
    metadata: Mapping[str, Any] | None = None,
    packed_backends: tuple[PackedBackendRecipe, ...] = (),
    target: str = "cuda",
    sm_arch: int = 89,
    compile_kernels: bool = True,
    compile_backends: bool = True,
    nvcc: str | None = None,
    backend_build_dir: str | Path | None = None,
    backend_library_dirs: tuple[str | Path, ...] = (),
) -> ArtifactBuildSummary:
    """Install runtime resources and write the generic artifact manifest."""

    artifact_dir = Path(artifact_dir)
    metadata_dir = artifact_dir / "metadata"
    weights_dir = artifact_dir / "weights"
    resources_dir = artifact_dir / "resources"
    kernels_dir = artifact_dir / "kernels"
    backends_dir = artifact_dir / "backends"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    resources_dir.mkdir(exist_ok=True)
    kernels_dir.mkdir(exist_ok=True)
    backends_dir.mkdir(exist_ok=True)

    weights_manifest: dict[str, Any] | None = None
    weights_entries = 0
    fp8_layout: str | None = None
    if weight_package_dir is not None:
        weights_manifest, weights_entries, fp8_layout = _install_weight_package(
            Path(weight_package_dir),
            weights_dir,
            metadata_dir,
        )

    resource_entries = _install_resources(resources, artifact_dir, metadata_dir)
    specs = _load_cuda_kernel_specs(metadata_dir / "kernel_table.json", sm_arch=sm_arch)
    if compile_kernels:
        if not specs:
            raise FileNotFoundError(metadata_dir / "kernel_table.json")
        _compile_kernel_cubins(specs, artifact_dir, sm_arch=sm_arch, nvcc=nvcc)

    backend_entries = _install_packed_backends(
        packed_backends,
        artifact_dir,
        sm_arch=sm_arch,
        compile_backends=compile_backends,
        backend_build_dir=Path(backend_build_dir) if backend_build_dir is not None else None,
        backend_library_dirs=tuple(Path(path) for path in backend_library_dirs),
    )
    if backend_entries:
        _write_json(metadata_dir / "packed_backend_table.json", backend_entries)

    manifest = {
        "format": "devproc2.artifact",
        "format_version": 1,
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "entrypoint": entrypoint,
        "target": {"kind": target, "arch": f"sm{sm_arch}"},
        "executable": "executable.vm",
        "abi": "abi.json",
        "weights": _weights_manifest_obj(
            weights_manifest,
            weights_entries=weights_entries,
            fp8_layout=fp8_layout,
            artifact_dir=artifact_dir,
        ),
        "resources": resource_entries,
        "kernels": {
            "count": len(specs),
            "compiled": compile_kernels,
            "table": "metadata/kernel_table.json" if specs else None,
        },
        "packed_backends": backend_entries,
        "metadata": dict(metadata or {}),
    }
    _write_json(metadata_dir / "artifact.json", manifest)

    return ArtifactBuildSummary(
        artifact_dir=artifact_dir,
        weights_entries=weights_entries,
        kernels=len(specs),
        resources=tuple(str(entry["path"]) for entry in resource_entries),
        fp8_layout=fp8_layout,
        packed_backends=tuple(str(entry["name"]) for entry in backend_entries),
    )


def _install_weight_package(
    weight_package_dir: Path,
    weights_dir: Path,
    metadata_dir: Path,
) -> tuple[dict[str, Any], int, str | None]:
    manifest = _read_json_required(weight_package_dir / "manifest.json")
    index_name = str(manifest.get("index_file", "weights.index.json"))
    index = _read_json_required(weight_package_dir / index_name)
    weight_map_name = str(manifest.get("weight_map_file", "weight_map.json"))
    quant_path = weight_package_dir / "quantization.json"
    report_path = weight_package_dir / "convert_report.json"

    data_file = str(manifest.get("data_file", "weights.bin"))
    _copy_required(weight_package_dir / data_file, weights_dir / "weights.bin")
    _copy_required(weight_package_dir / index_name, weights_dir / "weights.index.json")
    _copy_required(weight_package_dir / weight_map_name, metadata_dir / "weight_map.json")
    if quant_path.exists():
        shutil.copy2(quant_path, metadata_dir / "quantization.json")
    if report_path.exists():
        shutil.copy2(report_path, metadata_dir / "convert_report.json")

    report = _read_json(report_path)
    fp8_layout = report.get("fp8_layout") if isinstance(report, dict) else None
    return manifest, len(index.get("entries", [])), fp8_layout


def _weights_manifest_obj(
    manifest: dict[str, Any] | None,
    *,
    weights_entries: int,
    fp8_layout: str | None,
    artifact_dir: Path,
) -> dict[str, object] | None:
    if manifest is None:
        return None
    weights_bin = artifact_dir / "weights" / "weights.bin"
    quantization = artifact_dir / "metadata" / "quantization.json"
    return {
        "path": "weights",
        "entries": weights_entries,
        "precision": manifest.get("precision"),
        "sha256": _sha256(weights_bin),
        "index": "weights/weights.index.json",
        "weight_map": "metadata/weight_map.json",
        "quantization": "metadata/quantization.json" if quantization.exists() else None,
        "fp8_layout": fp8_layout,
    }


def _install_resources(
    resources: tuple[ResourceSpec, ...],
    artifact_dir: Path,
    metadata_dir: Path,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for spec in resources:
        src = Path(spec.path)
        target_path = spec.target_path or f"resources/{src.name}"
        dst = artifact_dir / target_path
        _copy_required(src, dst)
        entry: dict[str, object] = {
            "name": spec.name,
            "kind": spec.kind,
            "path": target_path,
            "sha256": _sha256(dst),
        }
        entry.update(spec.metadata)
        entries.append(entry)
        _write_json(metadata_dir / f"resource_{spec.name}.json", entry)
    return entries


def _install_packed_backends(
    packed_backends: tuple[PackedBackendRecipe, ...],
    artifact_dir: Path,
    *,
    sm_arch: int,
    compile_backends: bool,
    backend_build_dir: Path | None,
    backend_library_dirs: tuple[Path, ...],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for backend in packed_backends:
        entry = backend.to_table_obj(target_arch=f"sm{sm_arch}")
        if backend.kind == "compiled_packed_backend" and compile_backends:
            library = entry.get("library")
            if not isinstance(library, str) or not library:
                raise ValueError(
                    f"compiled packed backend {backend.name!r} must declare a library"
                )
            library_src = _resolve_compiled_backend_library(
                backend,
                backend_build_dir=backend_build_dir,
                backend_library_dirs=backend_library_dirs,
            )
            _copy_required(library_src, artifact_dir / library)
        entries.append(entry)
    return entries


def _resolve_compiled_backend_library(
    backend: PackedBackendRecipe,
    *,
    backend_build_dir: Path | None,
    backend_library_dirs: tuple[Path, ...],
) -> Path:
    sanitized = _sanitize_backend_name(backend.name)
    env_name = f"DEVPROC2_PACKED_BACKEND_{sanitized.upper()}_SO"
    env_path = os.environ.get(env_name)
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"{env_name} points to missing file: {path}")
        return path

    target = f"devproc2_{sanitized}_backend"
    library_names = _backend_library_names(target)
    search_dirs = list(backend_library_dirs)
    for search_dir in search_dirs:
        found = _find_library(search_dir, library_names)
        if found is not None:
            return found

    if backend_build_dir is not None:
        found = _find_library(backend_build_dir, library_names)
        if found is not None:
            return found

    searched_dirs = [*search_dirs]
    if backend_build_dir is not None:
        searched_dirs.append(backend_build_dir)
    searched = ", ".join(str(path) for path in searched_dirs)
    raise FileNotFoundError(
        f"Could not locate compiled packed backend {backend.name!r}. "
        f"Set {env_name}, pass --backend-library-dir, or run "
        f"devproc2 build --build-backends auto. Searched: {searched}"
    )


def _sanitize_backend_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _backend_library_names(target: str) -> tuple[str, ...]:
    return (
        f"lib{target}.so",
        f"{target}.so",
        f"lib{target}.dylib",
        f"{target}.dll",
    )


def _find_library(root: Path, names: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    matches: list[Path] = []
    if root.is_file() and root.name in names:
        matches.append(root)
    if root.is_dir():
        for name in names:
            matches.extend(root.rglob(name))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


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


def _compile_kernel_cubins(
    specs: list[KernelSpec],
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
    if src.resolve() == dst.resolve():
        return
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


__all__ = [
    "ArtifactBuildSummary",
    "prepare_artifact",
]
