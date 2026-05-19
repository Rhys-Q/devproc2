"""Canonical product build API and CLI for devproc2 model artifacts."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devproc2.artifact.manifest import PackedBackendRecipe, ResourceSpec
from devproc2.export.pipeline import (
    CompileResult,
    ExportSummary,
    compile_entrypoint,
    emit_compile_result,
)
from devproc2.export.recipe import CompileRecipe, EntrypointRecipe
from devproc2.ir.nodes import TensorStructInfo

BuildBackendMode = str


@dataclasses.dataclass(frozen=True)
class BackendBuildEntry:
    name: str
    kind: str
    mode: str
    target_arch: str
    library: str | None = None
    cache_key: str | None = None
    cache_dir: str | None = None
    cmake_build_dir: str | None = None
    built: bool = False
    source: str | None = None


@dataclasses.dataclass(frozen=True)
class BackendBuildSummary:
    mode: str
    entries: tuple[BackendBuildEntry, ...]

    @property
    def library_dirs(self) -> tuple[Path, ...]:
        dirs: list[Path] = []
        for entry in self.entries:
            if entry.library is not None:
                dirs.append(Path(entry.library).parent)
        return tuple(dirs)

    def to_json_obj(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "entries": [dataclasses.asdict(entry) for entry in self.entries],
        }


@dataclasses.dataclass(frozen=True)
class WeightValidationSummary:
    weight_package_dir: Path
    required_weights: int
    bound_weights: int
    fp8_layout: str | None
    activation_scales: str
    package_precision: str | None
    source_checkpoint: str | None
    target_hardware: str | None
    shape_profile: str | None
    action_horizon: int | None
    num_steps: int | None
    supports_static_act_scales: bool
    bindings: tuple[dict[str, object], ...]

    def to_json_obj(self) -> dict[str, object]:
        return {
            "weight_package_dir": str(self.weight_package_dir),
            "required_weights": self.required_weights,
            "bound_weights": self.bound_weights,
            "fp8_layout": self.fp8_layout,
            "activation_scales": self.activation_scales,
            "package_precision": self.package_precision,
            "source_checkpoint": self.source_checkpoint,
            "target_hardware": self.target_hardware,
            "shape_profile": self.shape_profile,
            "action_horizon": self.action_horizon,
            "num_steps": self.num_steps,
            "supports_static_act_scales": self.supports_static_act_scales,
            "bindings": list(self.bindings),
        }


@dataclasses.dataclass(frozen=True)
class BuildSummary:
    artifact_dir: Path
    model_id: str
    entrypoint: str
    profile: str | None
    target: str
    sm_arch: int
    activation_scales: str
    export: ExportSummary
    weight_validation: WeightValidationSummary | None
    backend_build: BackendBuildSummary

    def to_json_obj(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "artifact_dir": str(self.artifact_dir),
            "model_id": self.model_id,
            "entrypoint": self.entrypoint,
            "profile": self.profile,
            "target": self.target,
            "sm_arch": self.sm_arch,
            "activation_scales": self.activation_scales,
            "export": self.export.to_json_obj(),
            "backend_build": self.backend_build.to_json_obj(),
        }
        if self.weight_validation is not None:
            payload["weight_validation"] = self.weight_validation.to_json_obj()
        return payload


def build(
    *,
    recipe: CompileRecipe | EntrypointRecipe | None = None,
    model: str | None = None,
    entry: str | None = None,
    weights: str | Path | None = None,
    weight_package_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    out: str | Path | None = None,
    config: object | None = None,
    profile: str | None = None,
    resources: Mapping[str, str | Path] | None = None,
    target: str = "cuda",
    sm_arch: int = 89,
    build_backends: BuildBackendMode = "auto",
    backend_cache_dir: str | Path = "build/model-backends",
    backend_library_dirs: tuple[str | Path, ...] = (),
    backend_cmake_build_dir: str | Path | None = None,
    options: Mapping[str, Any] | None = None,
    activation_scales: str = "auto",
    compile_mode: str | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
) -> BuildSummary:
    """Build a self-contained model artifact.

    This is the product-facing wrapper over the generic compile/emit pipeline.
    It makes weight validation and model-owned backend handling explicit before
    the artifact packager copies resources into the final artifact directory.
    """

    artifact_path = Path(artifact_dir or out or "")
    if not str(artifact_path):
        raise ValueError("artifact_dir or out is required")
    weight_path = Path(weight_package_dir or weights) if (weight_package_dir or weights) else None

    model_ctx = _resolve_model_context(
        model,
        recipe=recipe,
        entry=entry,
        profile=profile,
        config=config,
    )
    entrypoint = model_ctx["entrypoint"]
    resolved_profile = model_ctx["profile"]
    resolved_config = model_ctx["config"]

    opts = _config_to_options(resolved_config)
    opts.update(dict(options or {}))
    if compile_mode is None:
        compile_mode = str(opts.get("compile_mode", "fast"))
    opts["compile_mode"] = compile_mode

    activation_mode = _resolve_activation_scale_mode(
        activation_scales,
        weight_path,
        explicit=opts.get("use_static_act_scales"),
    )
    opts["use_static_act_scales"] = activation_mode == "static"

    compile_result = compile_entrypoint(
        entrypoint,
        options=opts,
        sm_arch=sm_arch,
        compile_mode=compile_mode,
    )
    export_summary = emit_compile_result(
        artifact_path,
        compile_result,
        model_name=entrypoint.model_name or f"{entrypoint.model_id}-{entrypoint.name}",
        target=target,
        target_arch=f"sm{sm_arch}",
    )

    weight_validation = None
    if weight_path is not None:
        weight_validation = validate_weight_package(
            compile_result,
            weight_path,
            expected_model_id=entrypoint.model_id,
            options=opts,
            expected_fp8_layout=_config_fp8_layout(resolved_config),
            activation_scales=activation_mode,
            profile=resolved_profile,
        )

    backend_summary = build_packed_backends(
        entrypoint.packed_backends,
        model_id=entrypoint.model_id,
        sm_arch=sm_arch,
        mode=build_backends,
        backend_cache_dir=backend_cache_dir,
        backend_library_dirs=tuple(Path(path) for path in backend_library_dirs),
        cmake_build_dir=Path(backend_cmake_build_dir) if backend_cmake_build_dir else None,
    )
    all_backend_dirs = (
        tuple(Path(path) for path in backend_library_dirs)
        + backend_summary.library_dirs
    )

    resource_specs = _resource_specs(resources or {}, model_id=entrypoint.model_id)
    package_metadata = {
        "build": "metadata/build.json",
        "config": "metadata/config.json",
        "weight_binding": "metadata/weight_binding.json" if weight_validation else None,
        "backend_build": "metadata/backend_build.json",
    }
    from devproc2.artifact.builder import prepare_artifact

    resource_summary = prepare_artifact(
        model_id=entrypoint.model_id,
        entrypoint=entrypoint.name,
        artifact_dir=artifact_path,
        weight_package_dir=weight_path,
        resources=tuple(resource_specs),
        metadata=package_metadata,
        packed_backends=entrypoint.packed_backends,
        target=target,
        sm_arch=sm_arch,
        compile_kernels=compile_kernels,
        compile_backends=True,
        nvcc=nvcc,
        backend_build_dir=Path(backend_cmake_build_dir) if backend_cmake_build_dir else None,
        backend_library_dirs=all_backend_dirs,
    )
    export_summary = ExportSummary(
        artifact_dir=export_summary.artifact_dir,
        function_name=export_summary.function_name,
        num_user_inputs=export_summary.num_user_inputs,
        num_weight_params=export_summary.num_weight_params,
        vm_functions=export_summary.vm_functions,
        instructions=export_summary.instructions,
        storage_bytes=export_summary.storage_bytes,
        resource_summary=resource_summary,
    )

    summary = BuildSummary(
        artifact_dir=artifact_path,
        model_id=entrypoint.model_id,
        entrypoint=entrypoint.name,
        profile=resolved_profile,
        target=target,
        sm_arch=sm_arch,
        activation_scales=activation_mode,
        export=export_summary,
        weight_validation=weight_validation,
        backend_build=backend_summary,
    )
    _write_build_metadata(
        artifact_path,
        summary=summary,
        config=resolved_config,
        options=opts,
        resources=resources or {},
        build_backends=build_backends,
        backend_cache_dir=Path(backend_cache_dir),
        compile_kernels=compile_kernels,
    )
    return summary


def build_packed_backends(
    packed_backends: tuple[PackedBackendRecipe, ...],
    *,
    model_id: str,
    sm_arch: int,
    mode: BuildBackendMode,
    backend_cache_dir: str | Path,
    backend_library_dirs: tuple[Path, ...] = (),
    cmake_build_dir: Path | None = None,
) -> BackendBuildSummary:
    if mode not in {"auto", "never", "force"}:
        raise ValueError("build_backends must be one of: auto, never, force")

    entries: list[BackendBuildEntry] = []
    cache_root = Path(backend_cache_dir)
    for backend in packed_backends:
        if backend.kind != "compiled_packed_backend":
            entries.append(
                BackendBuildEntry(
                    name=backend.name,
                    kind=backend.kind,
                    mode=mode,
                    target_arch=f"sm{sm_arch}",
                    built=False,
                )
            )
            continue

        cache_key = _backend_cache_key(backend, model_id=model_id, sm_arch=sm_arch)
        cache_dir = cache_root / cache_key
        library_names = _backend_library_names(_backend_target_name(backend))
        cached = None if mode == "force" else _find_library(cache_dir, library_names)
        source = "cache" if cached is not None else None
        built = False

        if cached is None:
            cached = _find_backend_in_dirs(backend, backend_library_dirs)
            source = "library_dir" if cached is not None else None

        if cached is None and mode == "never":
            raise FileNotFoundError(_backend_disabled_error(backend, sm_arch))

        if cached is None:
            cached = _build_backend_with_cmake(
                backend,
                model_id=model_id,
                sm_arch=sm_arch,
                cache_dir=cache_dir,
                cmake_build_dir=cmake_build_dir,
            )
            source = "cmake"
            built = True

        entries.append(
            BackendBuildEntry(
                name=backend.name,
                kind=backend.kind,
                mode=mode,
                target_arch=f"sm{sm_arch}",
                library=str(cached),
                cache_key=cache_key,
                cache_dir=str(cache_dir),
                cmake_build_dir=str(cmake_build_dir or (cache_dir / "cmake")),
                built=built,
                source=source,
            )
        )
    return BackendBuildSummary(mode=mode, entries=tuple(entries))


def validate_weight_package(
    result: CompileResult,
    weight_package_dir: str | Path,
    *,
    expected_model_id: str,
    options: Mapping[str, Any],
    expected_fp8_layout: str | None,
    activation_scales: str,
    profile: str | None,
) -> WeightValidationSummary:
    package_dir = Path(weight_package_dir)
    manifest = _read_json_required(package_dir / "manifest.json")
    weight_map_path = package_dir / str(manifest.get("weight_map_file", "weight_map.json"))
    weight_map = _read_json_required(weight_map_path)
    entries = {
        str(entry.get("name")): entry
        for entry in weight_map.get("weights", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    report = _read_json(package_dir / "convert_report.json")
    package_model = manifest.get("model")
    if package_model is not None and package_model != expected_model_id:
        raise ValueError(
            f"weight package model {package_model!r} does not match build model "
            f"{expected_model_id!r}"
        )

    report_layout = report.get("fp8_layout") if isinstance(report, dict) else None
    if expected_fp8_layout and report_layout and report_layout != expected_fp8_layout:
        raise ValueError(
            f"weight package fp8_layout={report_layout!r} does not match config "
            f"fp8_layout={expected_fp8_layout!r}"
        )
    _validate_report_int(report, "action_horizon", options.get("action_horizon"))
    _validate_report_int(report, "num_steps", options.get("num_steps"))
    if profile and isinstance(report, dict) and report.get("shape_profile") not in (None, profile):
        raise ValueError(
            f"weight package shape_profile={report.get('shape_profile')!r} does not "
            f"match build profile={profile!r}"
        )

    has_act_scales = any(name.startswith("act_scale.") for name in entries)
    supports_static = (
        bool(report.get("supports_static_act_scales"))
        if isinstance(report, dict) and "supports_static_act_scales" in report
        else has_act_scales
    )
    if activation_scales == "static" and not supports_static:
        raise ValueError(
            "static activation scales requested, but the weight package does not "
            "declare supports_static_act_scales=true and has no act_scale.* entries"
        )

    required = _required_weight_params(result)
    bindings: list[dict[str, object]] = []
    missing: list[str] = []
    mismatches: list[str] = []
    uses_fp8 = False
    for param in required:
        name = param.name
        entry = entries.get(name)
        if entry is None:
            missing.append(name)
            continue
        param_si = param.struct_info
        binding: dict[str, object] = {"name": name}
        if isinstance(param_si, TensorStructInfo):
            expected_shape = _shape_to_int_list(param_si.shape)
            expected_dtype = param_si.dtype
            actual_shape = entry.get("shape")
            actual_dtype = entry.get("dtype")
            binding.update(
                {
                    "expected_shape": expected_shape,
                    "expected_dtype": expected_dtype,
                    "package_shape": actual_shape,
                    "package_dtype": actual_dtype,
                    "package_layout": entry.get("layout"),
                    "package_kind": entry.get("kind"),
                }
            )
            if expected_shape is not None and list(actual_shape or []) != expected_shape:
                mismatches.append(
                    f"{name}: shape expected {expected_shape}, package has {actual_shape}"
                )
            if actual_dtype != expected_dtype:
                mismatches.append(
                    f"{name}: dtype expected {expected_dtype}, package has {actual_dtype}"
                )
            if expected_dtype == "fp8_e4m3":
                uses_fp8 = True
                quant = entry.get("quant")
                if not isinstance(quant, dict):
                    mismatches.append(f"{name}: FP8 tensor is missing quant metadata")
                else:
                    scale_name = quant.get("scale_name")
                    binding["scale_name"] = scale_name
                    if scale_name not in entries:
                        mismatches.append(f"{name}: missing declared FP8 scale {scale_name!r}")
                    packed_layout = quant.get("packed_layout")
                    if expected_fp8_layout and packed_layout != expected_fp8_layout:
                        mismatches.append(
                            f"{name}: packed_layout expected {expected_fp8_layout}, "
                            f"package has {packed_layout}"
                        )
        bindings.append(binding)

    if uses_fp8 and "fp8" not in str(manifest.get("precision", "")):
        mismatches.append(
            f"graph requires FP8 weights but package precision is {manifest.get('precision')!r}"
        )
    if activation_scales == "static":
        missing_act = [name for name in missing if name.startswith("act_scale.")]
        if missing_act:
            preview = ", ".join(missing_act[:8])
            raise ValueError(
                "static activation graph requires act_scale.* tensors missing from "
                f"weight package: {preview}"
            )
    if missing:
        preview = ", ".join(missing[:16])
        raise ValueError(f"weight package is missing required weights: {preview}")
    if mismatches:
        preview = "\n".join(mismatches[:16])
        raise ValueError(f"weight package does not match graph ABI:\n{preview}")

    return WeightValidationSummary(
        weight_package_dir=package_dir,
        required_weights=len(required),
        bound_weights=len(bindings),
        fp8_layout=report_layout or expected_fp8_layout,
        activation_scales=activation_scales,
        package_precision=_string_or_none(manifest.get("precision")),
        source_checkpoint=_source_checkpoint_from_report(report),
        target_hardware=_report_string(report, manifest, "target_hardware"),
        shape_profile=_report_string(report, manifest, "shape_profile"),
        action_horizon=_report_int(report, manifest, "action_horizon"),
        num_steps=_report_int(report, manifest, "num_steps"),
        supports_static_act_scales=bool(supports_static),
        bindings=tuple(bindings),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a devproc2 model artifact.")
    parser.add_argument("--model", default="pi05", help="registered model id, e.g. pi05")
    parser.add_argument("--recipe", default=None, help="debug/internal import.path:object recipe")
    parser.add_argument("--entry", default=None)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--target", default="cuda")
    parser.add_argument("--sm-arch", type=int, default=89)
    parser.add_argument(
        "--build-backends",
        choices=("auto", "never", "force"),
        default="auto",
    )
    parser.add_argument("--backend-cache-dir", type=Path, default=Path("build/model-backends"))
    parser.add_argument("--backend-library-dir", type=Path, action="append", default=[])
    parser.add_argument("--backend-cmake-build-dir", type=Path, default=None)
    parser.add_argument("--resource", action="append", default=[], help="name=/path")
    parser.add_argument("--option", action="append", default=[], help="name=value debug option")
    parser.add_argument(
        "--activation-scales",
        choices=("auto", "static", "dynamic"),
        default="auto",
    )
    parser.add_argument("--compile-mode", choices=("fast", "normal"), default=None)
    parser.add_argument("--no-compile-kernels", action="store_true")
    parser.add_argument("--nvcc", default=None)
    args = parser.parse_args(argv)

    recipe = _load_recipe(args.recipe, entry=args.entry) if args.recipe else None
    summary = build(
        recipe=recipe,
        model=None if recipe else args.model,
        entry=args.entry,
        weights=args.weights,
        artifact_dir=args.out,
        profile=args.profile,
        target=args.target,
        sm_arch=args.sm_arch,
        build_backends=args.build_backends,
        backend_cache_dir=args.backend_cache_dir,
        backend_library_dirs=tuple(args.backend_library_dir),
        backend_cmake_build_dir=args.backend_cmake_build_dir,
        resources=_parse_key_values(args.resource),
        options=_parse_options(args.option),
        activation_scales=args.activation_scales,
        compile_mode=args.compile_mode,
        compile_kernels=not args.no_compile_kernels,
        nvcc=args.nvcc,
    )
    print(json.dumps(summary.to_json_obj(), indent=2, sort_keys=True))


def _resolve_model_context(
    model: str | None,
    *,
    recipe: CompileRecipe | EntrypointRecipe | None,
    entry: str | None,
    profile: str | None,
    config: object | None,
) -> dict[str, Any]:
    if recipe is None:
        model_key = (model or "pi05").lower()
        if model_key not in {"pi05", "openpi0.5", "openpi05"}:
            raise KeyError(f"unknown model {model!r}")
        from devproc2.models.pi05.config import PI05Config
        from devproc2.models.pi05.model import PI05_MODEL

        recipe = PI05_MODEL
        if config is None:
            resolved_profile = profile or PI05Config.default_profile()
            config = PI05Config.for_profile(resolved_profile)
        else:
            resolved_profile = profile
    else:
        resolved_profile = profile

    if isinstance(recipe, CompileRecipe):
        if entry is None:
            entry_name = _config_entry(config) or "sample_tokens"
        else:
            entry_name = entry
        entrypoint = recipe.entrypoint(entry_name)
    elif isinstance(recipe, EntrypointRecipe):
        entrypoint = recipe
    else:
        raise TypeError("recipe must be a CompileRecipe or EntrypointRecipe")
    return {"entrypoint": entrypoint, "profile": resolved_profile, "config": config}


def _config_entry(config: object | None) -> str | None:
    entrypoint = getattr(config, "entrypoint", None)
    name = getattr(entrypoint, "name", None)
    return str(name) if name else None


def _config_to_options(config: object | None) -> dict[str, Any]:
    if config is None:
        return {}
    to_options = getattr(config, "to_options", None)
    if callable(to_options):
        return dict(to_options())
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def _config_fp8_layout(config: object | None) -> str | None:
    layout = getattr(config, "layout", None)
    value = getattr(layout, "fp8_layout", None)
    return str(value) if value else None


def _resource_specs(resources: Mapping[str, str | Path], *, model_id: str) -> list[ResourceSpec]:
    specs: list[ResourceSpec] = []
    for name, path in resources.items():
        metadata: dict[str, object] = {}
        target_path = None
        if model_id == "openpi0.5" and name == "tokenizer":
            target_path = "resources/tokenizer.model"
            metadata = {
                "tokenizer_kind": "sentencepiece",
                "tokenizer_model": "paligemma",
            }
        specs.append(ResourceSpec(name=name, path=path, target_path=target_path, metadata=metadata))
    return specs


def _resolve_activation_scale_mode(
    requested: str,
    weight_path: Path | None,
    *,
    explicit: object,
) -> str:
    if requested not in {"auto", "static", "dynamic"}:
        raise ValueError("activation_scales must be one of: auto, static, dynamic")
    if requested in {"static", "dynamic"}:
        return requested
    if explicit is not None:
        return "static" if bool(explicit) else "dynamic"
    if weight_path is None:
        return "dynamic"
    report = _read_json(weight_path / "convert_report.json")
    if isinstance(report, dict) and report.get("supports_static_act_scales"):
        return "static"
    weight_map = _read_weight_map(weight_path)
    if any(name.startswith("act_scale.") for name in weight_map):
        return "static"
    return "dynamic"


def _required_weight_params(result: CompileResult):
    fn = result.module.functions.get("main")
    if fn is None:
        return ()
    return tuple(fn.params[result.num_user_inputs:])


def _shape_to_int_list(shape: object) -> list[int] | None:
    values = getattr(shape, "values", shape)
    out: list[int] = []
    for dim in values:
        value = getattr(dim, "value", None)
        if value is None:
            return None
        out.append(int(value))
    return out


def _validate_report_int(report: object, key: str, expected: object) -> None:
    if not isinstance(report, dict) or key not in report or expected is None:
        return
    if int(report[key]) != int(expected):
        raise ValueError(
            f"weight package {key}={report[key]!r} does not match build {key}={expected!r}"
        )


def _report_string(report: object, manifest: Mapping[str, Any], key: str) -> str | None:
    value: object = None
    if isinstance(report, dict):
        value = report.get(key)
    if value is None:
        value = manifest.get(key)
    return _string_or_none(value)


def _report_int(report: object, manifest: Mapping[str, Any], key: str) -> int | None:
    value: object = None
    if isinstance(report, dict):
        value = report.get(key)
    if value is None:
        value = manifest.get(key)
    if value is None:
        return None
    return int(value)


def _source_checkpoint_from_report(report: object) -> str | None:
    if not isinstance(report, dict):
        return None
    source = report.get("source")
    if isinstance(source, dict):
        return _string_or_none(source.get("path"))
    return _string_or_none(source)


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_build_metadata(
    artifact_dir: Path,
    *,
    summary: BuildSummary,
    config: object | None,
    options: Mapping[str, Any],
    resources: Mapping[str, str | Path],
    build_backends: str,
    backend_cache_dir: Path,
    compile_kernels: bool,
) -> None:
    metadata_dir = artifact_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    build_payload = {
        "format": "devproc2.build",
        "format_version": 1,
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": summary.model_id,
        "entrypoint": summary.entrypoint,
        "profile": summary.profile,
        "target": summary.target,
        "sm_arch": summary.sm_arch,
        "activation_scales": summary.activation_scales,
        "compile_mode": options.get("compile_mode"),
        "compile_kernels": compile_kernels,
        "build_backends": build_backends,
        "backend_cache_dir": str(backend_cache_dir),
        "resources": {name: str(path) for name, path in resources.items()},
    }
    _write_json(metadata_dir / "build.json", build_payload)
    _write_json(metadata_dir / "backend_build.json", summary.backend_build.to_json_obj())
    config_payload = _config_to_json_obj(config)
    config_payload["resolved_options"] = dict(options)
    _write_json(metadata_dir / "config.json", config_payload)
    if summary.weight_validation is not None:
        _write_json(
            metadata_dir / "weight_binding.json",
            summary.weight_validation.to_json_obj(),
        )


def _config_to_json_obj(config: object | None) -> dict[str, object]:
    if config is None:
        return {}
    to_json = getattr(config, "to_json_obj", None)
    if callable(to_json):
        return dict(to_json())
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return {"repr": repr(config)}


def _backend_cache_key(backend: PackedBackendRecipe, *, model_id: str, sm_arch: int) -> str:
    payload = {
        "model_id": model_id,
        "backend": backend.name,
        "kind": backend.kind,
        "target_arch": f"sm{sm_arch}",
        "sources": [_source_digest(path) for path in backend.sources],
        "include_dirs": list(backend.include_dirs),
        "compile_definitions": list(backend.compile_definitions),
        "compile_options": list(backend.compile_options),
        "link_libraries": list(backend.link_libraries),
        "targets": list(backend.targets),
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _source_digest(path: str) -> dict[str, object]:
    repo = _repo_root()
    source = Path(path)
    if not source.is_absolute():
        source = repo / source
    payload: dict[str, object] = {"path": path}
    if source.exists() and source.is_file():
        payload["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        payload["missing"] = True
    return payload


def _backend_target_name(backend: PackedBackendRecipe) -> str:
    sanitized = _sanitize_backend_name(backend.name)
    return f"devproc2_{sanitized}_backend"


def _backend_library_names(target: str) -> tuple[str, ...]:
    return (
        f"lib{target}.so",
        f"{target}.so",
        f"lib{target}.dylib",
        f"{target}.dll",
    )


def _find_backend_in_dirs(backend: PackedBackendRecipe, dirs: tuple[Path, ...]) -> Path | None:
    env_name = f"DEVPROC2_PACKED_BACKEND_{_sanitize_backend_name(backend.name).upper()}_SO"
    env_path = os.environ.get(env_name)
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"{env_name} points to missing file: {path}")
        return path
    return _find_library_in_dirs(dirs, _backend_library_names(_backend_target_name(backend)))


def _find_library_in_dirs(dirs: tuple[Path, ...], names: tuple[str, ...]) -> Path | None:
    for directory in dirs:
        found = _find_library(directory, names)
        if found is not None:
            return found
    return None


def _find_library(root: Path, names: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    matches: list[Path] = []
    if root.is_file() and root.name in names:
        matches.append(root)
    elif root.is_dir():
        for name in names:
            matches.extend(root.rglob(name))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _build_backend_with_cmake(
    backend: PackedBackendRecipe,
    *,
    model_id: str,
    sm_arch: int,
    cache_dir: Path,
    cmake_build_dir: Path | None,
) -> Path:
    if model_id != "openpi0.5" or backend.name != "pi05.cuda":
        raise RuntimeError(
            f"no backend build recipe is registered for {model_id}:{backend.name}"
        )
    build_dir = cmake_build_dir or (cache_dir / "cmake")
    lib_dir = cache_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    target = _backend_target_name(backend)
    source_dir = _repo_root() / "python" / "devproc2" / "models" / "pi05" / "cuda"
    configure = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DDEVPROC2_WITH_CUDA=ON",
        "-DDEVPROC2_WITH_CUTLASS=ON",
        "-DDEVPROC2_BUILD_TESTS=OFF",
        f"-DDEVPROC2_REPO_ROOT={_repo_root()}",
        f"-DCMAKE_CUDA_ARCHITECTURES={sm_arch}",
    ]
    cache_path = build_dir / "CMakeCache.txt"
    if cache_path.exists():
        cache_home = _cmake_cache_home(cache_path)
        if cache_home is not None and cache_home.resolve() != source_dir.resolve():
            raise RuntimeError(
                f"backend CMake build dir {build_dir} was configured for "
                f"{cache_home}; Pi0.5 backend now builds from {source_dir}. "
                "Use a fresh --backend-cmake-build-dir or remove the old cache."
            )
    else:
        _run_command(configure, "configure model backend")
    jobs = str(max(1, min(os.cpu_count() or 2, 8)))
    _run_command(
        ["cmake", "--build", str(build_dir), "--target", target, "-j", jobs],
        f"build model backend {backend.name}",
    )
    built = _find_library(build_dir, _backend_library_names(target))
    if built is None:
        raise FileNotFoundError(f"CMake target {target} did not produce a shared library")
    cached = lib_dir / built.name
    shutil.copy2(built, cached)
    return cached


def _run_command(command: list[str], description: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"cmake is required to {description}") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        raise RuntimeError(f"Failed to {description}:\n{output[-4000:]}") from exc


def _cmake_cache_home(cache_path: Path) -> Path | None:
    for line in cache_path.read_text(errors="ignore").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            return Path(line.split("=", 1)[1])
    return None


def _backend_disabled_error(backend: PackedBackendRecipe, sm_arch: int) -> str:
    return (
        f"Pi0.5 entry requires model backend {backend.name} for target sm{sm_arch}.\n"
        "Backend build is disabled by --build-backends never.\n"
        "Pass --build-backends auto, --build-backends force, or "
        "--backend-library-dir <dir>."
    )


def _sanitize_backend_name(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _read_weight_map(package_dir: Path) -> dict[str, dict[str, object]]:
    manifest = _read_json_required(package_dir / "manifest.json")
    weight_map = _read_json_required(package_dir / str(manifest.get("weight_map_file", "weight_map.json")))
    return {
        str(entry["name"]): entry
        for entry in weight_map.get("weights", [])
        if isinstance(entry, dict) and "name" in entry
    }


def _read_json(path: Path) -> object:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_recipe(spec: str, *, entry: str | None) -> EntrypointRecipe:
    if ":" not in spec:
        raise ValueError("--recipe must use import.path:object syntax")
    module_name, object_name = spec.split(":", 1)
    obj = getattr(importlib.import_module(module_name), object_name)
    if isinstance(obj, EntrypointRecipe):
        return obj
    if isinstance(obj, CompileRecipe):
        if entry is None:
            raise ValueError("--entry is required when --recipe points to a CompileRecipe")
        return obj.entrypoint(entry)
    raise TypeError(f"{spec!r} did not resolve to an EntrypointRecipe or CompileRecipe")


def _parse_key_values(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected name=value, got {item!r}")
        name, value = item.split("=", 1)
        parsed[name] = value
    return parsed


def _parse_options(items: list[str]) -> dict[str, Any]:
    return {name: _parse_scalar(value) for name, value in _parse_key_values(items).items()}


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
