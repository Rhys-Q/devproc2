"""Generic recipe-based export CLI."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from devproc2.export.pipeline import export_artifact
from devproc2.export.recipe import CompileRecipe, EntrypointRecipe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a devproc2 artifact from a recipe.")
    parser.add_argument("--recipe", required=True, help="import.path:object")
    parser.add_argument("--entry", default=None, help="entrypoint name when --recipe is a CompileRecipe")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--weight-package-dir", type=Path, default=None)
    parser.add_argument("--resource", action="append", default=[], help="name=/path/to/resource")
    parser.add_argument("--option", action="append", default=[], help="name=value recipe option")
    parser.add_argument("--sm-arch", type=int, default=89)
    parser.add_argument("--compile-mode", choices=("fast", "normal"), default="fast")
    parser.add_argument("--no-compile-kernels", action="store_true")
    parser.add_argument("--no-compile-backends", action="store_true")
    parser.add_argument("--backend-build-dir", type=Path, default=None)
    parser.add_argument(
        "--backend-library-dir",
        type=Path,
        action="append",
        default=[],
        help="directory containing prebuilt packed backend shared libraries",
    )
    parser.add_argument("--nvcc", default=None)
    args = parser.parse_args(argv)

    recipe = _load_recipe(args.recipe, entry=args.entry)
    resources = _parse_key_values(args.resource)
    options = _parse_options(args.option)
    options["compile_mode"] = args.compile_mode
    summary = export_artifact(
        recipe,
        artifact_dir=args.artifact_dir,
        resources=resources,
        weight_package_dir=args.weight_package_dir,
        options=options,
        sm_arch=args.sm_arch,
        compile_mode=args.compile_mode,
        compile_kernels=not args.no_compile_kernels,
        compile_backends=not args.no_compile_backends,
        nvcc=args.nvcc,
        backend_build_dir=args.backend_build_dir,
        backend_library_dirs=tuple(args.backend_library_dir),
    )
    print(json.dumps(summary.to_json_obj(), indent=2, sort_keys=True))


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
