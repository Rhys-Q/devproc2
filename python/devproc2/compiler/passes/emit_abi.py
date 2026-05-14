"""EmitABIPass — generate ABI/manifest/metadata JSON files for a compiled artifact."""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
from typing import Any, Optional

from devproc2.compiler.pass_context import PassContext
from devproc2.ir.nodes import (
    Function,
    IRModule,
    TensorStructInfo,
)
from devproc2.ir.prim_expr import PrimVar
from devproc2.vm.executable import CalleeKind, Executable

_ABI_VERSION = "0.1"
_ARTIFACT_VERSION = "0.1.0"


def _shape_dim_to_json(dim) -> Any:
    """Convert a PrimExpr shape dimension to a JSON-compatible value (int or str)."""
    from devproc2.ir.prim_expr import IntImm
    if isinstance(dim, IntImm):
        return dim.value
    if isinstance(dim, PrimVar):
        return dim.name
    return str(dim)


def _collect_prim_vars(struct_info) -> dict[str, Optional[int]]:
    """Walk a StructInfo tree and collect all PrimVar name → upper mappings."""
    result: dict[str, Optional[int]] = {}
    if isinstance(struct_info, TensorStructInfo):
        for dim in struct_info.shape:
            _collect_prim_vars_from_expr(dim, result)
    return result


def _collect_prim_vars_from_expr(expr, out: dict[str, Optional[int]]) -> None:
    from devproc2.ir.prim_expr import (
        Add, CeilDiv, FloorDiv, IntImm, Max, Min, Mul, Sub,
    )
    if isinstance(expr, IntImm):
        return
    if isinstance(expr, PrimVar):
        if expr.name not in out:
            out[expr.name] = expr.upper
        return
    for cls in (Add, Sub, Mul, FloorDiv, CeilDiv, Min, Max):
        if isinstance(expr, cls):
            _collect_prim_vars_from_expr(expr.lhs, out)  # type: ignore[attr-defined]
            _collect_prim_vars_from_expr(expr.rhs, out)  # type: ignore[attr-defined]
            return


def _tensor_struct_info_to_dict(si: TensorStructInfo) -> dict[str, Any]:
    return {
        "dtype": si.dtype,
        "shape": [_shape_dim_to_json(d) for d in si.shape],
        "device": si.device,
    }


def _function_entry_to_dict(fe) -> dict[str, Any]:
    return {
        "name": fe.name,
        "kind": fe.kind.name,
        "instr_offset": fe.instr_offset,
        "instr_count": fe.instr_count,
        "num_regs": fe.num_regs,
        "num_args": fe.num_args,
    }


class EmitABIPass:
    """Generate abi.json, manifest.json, and metadata/*.json for an artifact.

    Reads ABI info from:
      - IRModule.functions["main"] — input/output types and shape constraints
      - Executable.function_table — function/kernel/packed_func tables
      - PassContext["storage_plan"] — memory planning results
    """

    def run(
        self,
        module: IRModule,
        exe: Executable,
        ctx: PassContext,
        output_dir: str,
        model_name: str = "model",
        target: str = "cpu",
        target_arch: str = "",
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "kernels"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "constants"), exist_ok=True)

        fn = module.functions.get("main")
        inputs_json, outputs_json, shape_constraints = self._extract_abi_from_fn(fn)
        required_packed_funcs = self._extract_required_packed_funcs(exe)

        abi = {
            "devproc_abi_version": _ABI_VERSION,
            "vm_bytecode_version": _ABI_VERSION,
            "kernel_calling_convention": "dps_kernel_v1",
            "packed_func_calling_convention": "dps_packed_v1",
            "target": target,
            "target_arch": target_arch,
            "inputs": inputs_json,
            "outputs": outputs_json,
            "shape_constraints": shape_constraints,
            "required_packed_funcs": required_packed_funcs,
        }
        self._write_json(output_dir, "abi.json", abi)

        manifest = {
            "name": model_name,
            "version": _ARTIFACT_VERSION,
            "build_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": target,
            "target_arch": target_arch,
        }
        self._write_json(output_dir, "manifest.json", manifest)

        meta = os.path.join(output_dir, "metadata")
        function_table = [_function_entry_to_dict(fe) for fe in exe.function_table]
        self._write_json(meta, "function_table.json", function_table)
        self._write_json(meta, "kernel_table.json", [
            _function_entry_to_dict(fe)
            for fe in exe.function_table
            if fe.kind == CalleeKind.kernel
        ])
        self._write_json(meta, "packed_func_table.json", [
            _function_entry_to_dict(fe)
            for fe in exe.function_table
            if fe.kind == CalleeKind.packed_func
        ])
        self._write_json(meta, "shape_constraints.json", shape_constraints)

        plan = ctx.get("storage_plan")
        if plan is not None:
            plan_json = self._storage_plan_to_json(plan)
        else:
            plan_json = []
        self._write_json(meta, "storage_plan.json", plan_json)

    # ------------------------------------------------------------------

    def _extract_abi_from_fn(self, fn: Optional[Function]):
        inputs_json = []
        outputs_json = []
        shape_constraints: dict[str, Any] = {}

        if fn is None:
            return inputs_json, outputs_json, shape_constraints

        for param in fn.params:
            si = param.struct_info
            if isinstance(si, TensorStructInfo):
                entry = {"name": param.name}
                entry.update(_tensor_struct_info_to_dict(si))
                inputs_json.append(entry)
                shape_constraints.update(self._constraints_from_struct_info(si))

        ret = fn.ret_struct_info
        if isinstance(ret, TensorStructInfo):
            outputs_json.append(_tensor_struct_info_to_dict(ret))
            shape_constraints.update(self._constraints_from_struct_info(ret))

        return inputs_json, outputs_json, shape_constraints

    def _constraints_from_struct_info(self, si: TensorStructInfo) -> dict[str, Any]:
        prim_vars: dict[str, Optional[int]] = {}
        _collect_prim_vars(si)
        for dim in si.shape:
            _collect_prim_vars_from_expr(dim, prim_vars)
        result = {}
        for name, upper in prim_vars.items():
            if upper is not None:
                result[name] = {"upper": upper}
        return result

    def _extract_required_packed_funcs(self, exe: Executable) -> list[str]:
        return [
            fe.name
            for fe in exe.function_table
            if fe.kind == CalleeKind.packed_func
        ]

    def _storage_plan_to_json(self, plan) -> list[dict[str, Any]]:
        result = []
        for entry in plan.entries:
            result.append({
                "id": entry.id,
                "device": entry.device,
                "size_bytes": entry.size_bytes,
                "alignment": entry.alignment,
                "reused_by": list(entry.reused_by),
            })
        return result

    def _write_json(self, directory: str, filename: str, data: Any) -> None:
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
