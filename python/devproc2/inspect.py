"""Inspect devproc2 artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_inspect(artifact_dir: str | Path) -> int:
    root = Path(artifact_dir)
    if not root.is_dir():
        print(f"error: '{root}' is not a directory", file=sys.stderr)
        return 1

    abi_path = root / "abi.json"
    manifest_path = _manifest_path(root)
    fn_table_path = root / "metadata" / "function_table.json"
    vm_path = root / "executable.vm"

    print(f"devproc2 artifact: {root.resolve()}")
    if manifest_path is not None:
        _print_section("Manifest", _read_json(manifest_path))
    else:
        print("\n[metadata/artifact.json or manifest.json not found]")

    if abi_path.exists():
        abi = _read_json(abi_path)
        _print_section(
            "ABI",
            {
                "devproc_abi_version": abi.get("devproc_abi_version"),
                "target": abi.get("target"),
                "target_arch": abi.get("target_arch"),
                "inputs": abi.get("inputs", []),
                "outputs": abi.get("outputs", []),
                "shape_constraints": abi.get("shape_constraints", {}),
                "required_packed_funcs": abi.get("required_packed_funcs", []),
            },
        )
    else:
        print("\n[abi.json not found]")

    if fn_table_path.exists():
        _print_section("Function Table", _read_json(fn_table_path))
    else:
        print("\n[metadata/function_table.json not found]")

    if vm_path.exists():
        print(f"\nexecutable.vm: {vm_path.stat().st_size} bytes")
    else:
        print("\n[executable.vm not found]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a devproc2 artifact.")
    parser.add_argument("artifact_dir")
    args = parser.parse_args(argv)
    return cmd_inspect(args.artifact_dir)


def _manifest_path(root: Path) -> Path | None:
    for candidate in (root / "metadata" / "artifact.json", root / "manifest.json"):
        if candidate.exists():
            return candidate
    return None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _print_section(title: str, data: object) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

