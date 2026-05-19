#!/usr/bin/env python3
"""devproc2 CLI — inspect and manage compiled artifacts.

Usage:
    python devproc_cli.py inspect <artifact_dir>

Commands:
    inspect <dir>   Pretty-print artifact summary (ABI, manifest, function table).

Examples:
    python devproc_cli.py inspect build/kvcache_demo/
"""
from __future__ import annotations

import json
import os
import sys


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _print_section(title: str, data: object) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(json.dumps(data, indent=2))


def cmd_inspect(artifact_dir: str) -> int:
    """Inspect an artifact directory and print its contents."""
    if not os.path.isdir(artifact_dir):
        print(f"error: '{artifact_dir}' is not a directory", file=sys.stderr)
        return 1

    abi_path      = os.path.join(artifact_dir, "abi.json")
    artifact_manifest_path = os.path.join(artifact_dir, "metadata", "artifact.json")
    legacy_manifest_path = os.path.join(artifact_dir, "manifest.json")
    manifest_path = (
        artifact_manifest_path
        if os.path.exists(artifact_manifest_path)
        else legacy_manifest_path
    )
    fn_table_path = os.path.join(artifact_dir, "metadata", "function_table.json")
    vm_path       = os.path.join(artifact_dir, "executable.vm")

    print(f"devproc2 artifact: {os.path.abspath(artifact_dir)}")

    # Manifest
    if os.path.exists(manifest_path):
        manifest = _read_json(manifest_path)
        _print_section("Manifest", manifest)
    else:
        print("\n[metadata/artifact.json or manifest.json not found]")

    # ABI
    if os.path.exists(abi_path):
        abi = _read_json(abi_path)
        summary = {
            "devproc_abi_version":  abi.get("devproc_abi_version"),
            "target":               abi.get("target"),
            "target_arch":          abi.get("target_arch"),
            "inputs":               abi.get("inputs", []),
            "outputs":              abi.get("outputs", []),
            "shape_constraints":    abi.get("shape_constraints", {}),
            "required_packed_funcs": abi.get("required_packed_funcs", []),
        }
        _print_section("ABI", summary)
    else:
        print("\n[abi.json not found]")

    # Function table
    if os.path.exists(fn_table_path):
        fn_table = _read_json(fn_table_path)
        _print_section("Function Table", fn_table)
    else:
        print("\n[metadata/function_table.json not found]")

    # Executable size
    if os.path.exists(vm_path):
        size = os.path.getsize(vm_path)
        print(f"\nexecutable.vm: {size} bytes")
    else:
        print("\n[executable.vm not found]")

    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1

    cmd = sys.argv[1]
    if cmd == "inspect":
        if len(sys.argv) < 3:
            print("error: 'inspect' requires an artifact directory argument",
                  file=sys.stderr)
            return 1
        return cmd_inspect(sys.argv[2])

    print(f"error: unknown command '{cmd}'", file=sys.stderr)
    print("Available commands: inspect", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
