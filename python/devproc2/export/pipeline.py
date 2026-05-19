"""Generic compile, emit, and artifact export pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import ast
import inspect
from pathlib import Path
import textwrap
from typing import Any, Mapping

import devproc2 as dp
import devproc2.frontend.dsl as dsl

from devproc2.artifact.builder import ArtifactBuildSummary, prepare_artifact
from devproc2.artifact.manifest import PackedBackendRecipe, ResourceSpec
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_abi import EmitABIPass
from devproc2.compiler.passes.emit_executable import EmitExecutablePass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.export.recipe import EntrypointRecipe
from devproc2.ir.nodes import Function, IRModule
from devproc2.ir.ops import CallDPSOp, CudaCallOp, ReturnOp
from devproc2.nn import GraphBuilder, Module, ModuleList
from devproc2.vm.executable import Executable


CompileMode = str


@dataclass(frozen=True)
class CompileResult:
    module: IRModule
    lowered_module: IRModule
    executable: Executable
    context: PassContext
    num_user_inputs: int


@dataclass(frozen=True)
class ExportSummary:
    artifact_dir: Path
    function_name: str
    num_user_inputs: int
    num_weight_params: int
    vm_functions: int
    instructions: int
    storage_bytes: int
    resource_summary: ArtifactBuildSummary | None = None

    def to_json_obj(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_dir": str(self.artifact_dir),
            "function_name": self.function_name,
            "num_user_inputs": self.num_user_inputs,
            "num_weight_params": self.num_weight_params,
            "vm_functions": self.vm_functions,
            "instructions": self.instructions,
            "storage_bytes": self.storage_bytes,
        }
        if self.resource_summary is not None:
            payload["weights_entries"] = self.resource_summary.weights_entries
            payload["kernels"] = self.resource_summary.kernels
            payload["resources"] = list(self.resource_summary.resources)
            payload["fp8_layout"] = self.resource_summary.fp8_layout
            payload["packed_backends"] = list(self.resource_summary.packed_backends)
        return payload


def normalize_compile_mode(compile_mode: CompileMode) -> str:
    if compile_mode not in ("fast", "normal"):
        raise ValueError("compile_mode must be 'fast' or 'normal'")
    return str(compile_mode)


def build_graph(
    entrypoint: EntrypointRecipe,
    *,
    options: Mapping[str, Any] | None = None,
    compile_mode: CompileMode | None = None,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    opts = dict(options or {})
    if reset_dsl:
        dp.reset_module()
    mode = normalize_compile_mode(str(compile_mode or opts.get("compile_mode", "fast")))
    module = entrypoint.build_module(opts)
    input_specs = entrypoint.input_specs(opts)
    function_name = str(opts.get("function_name", entrypoint.function_name))
    normal = str(opts.get("normal_method", entrypoint.normal_method))
    fast = str(opts.get("fast_method", entrypoint.fast_method))
    method = _select_method(module, mode, normal=normal, fast=fast)
    ir_module = GraphBuilder().build(method, input_specs)
    if mode == "normal":
        assert_normal_ir_has_no_backend_ops(ir_module)
    method_name = getattr(method, "__name__", "main")
    fn = ir_module.functions[method_name]
    if function_name != method_name:
        ir_module = IRModule({function_name: fn})
    return ir_module, len(input_specs)


def compile_entrypoint(
    entrypoint: EntrypointRecipe,
    *,
    options: Mapping[str, Any] | None = None,
    sm_arch: int = 89,
    compile_mode: CompileMode | None = None,
) -> CompileResult:
    mode = normalize_compile_mode(
        str(compile_mode or (options or {}).get("compile_mode", "fast"))
    )
    module, num_user_inputs = build_graph(
        entrypoint,
        options=options,
        compile_mode=mode,
    )
    return compile_ir_module(
        module,
        num_user_inputs,
        sm_arch=sm_arch,
        compile_mode=mode,
    )


def compile_ir_module(
    module: IRModule,
    num_user_inputs: int,
    *,
    sm_arch: int,
    compile_mode: CompileMode,
) -> CompileResult:
    mode = normalize_compile_mode(compile_mode)
    module = InferStructInfoPass().run(module)
    module = stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(
        dsl.get_kernel_registry(),
        sm_arch=sm_arch,
        reference_fallback=(mode == "normal"),
    ).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return CompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def emit_entrypoint(
    entrypoint: EntrypointRecipe,
    output_dir: str | Path,
    *,
    options: Mapping[str, Any] | None = None,
    model_name: str | None = None,
    target: str = "cuda",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    compile_mode: CompileMode | None = None,
) -> ExportSummary:
    result = compile_entrypoint(
        entrypoint,
        options=options,
        sm_arch=sm_arch,
        compile_mode=compile_mode,
    )
    resolved_model_name = (
        model_name
        or entrypoint.model_name
        or f"{entrypoint.model_id}-{entrypoint.name}"
    )
    return emit_compile_result(
        Path(output_dir),
        result,
        model_name=resolved_model_name,
        target=target,
        target_arch=target_arch,
    )


def emit_compile_result(
    output_dir: Path,
    result: CompileResult,
    *,
    model_name: str,
    target: str = "cuda",
    target_arch: str,
) -> ExportSummary:
    EmitExecutablePass().run(result.executable, str(output_dir))
    EmitABIPass().run(
        result.module,
        result.executable,
        result.context,
        str(output_dir),
        model_name=model_name,
        target=target,
        target_arch=target_arch,
    )
    main_fn = result.executable.function_table[-1]
    storage_plan = result.context.get("storage_plan")
    storage_bytes = 0
    if storage_plan is not None:
        storage_bytes = sum(int(entry.size_bytes) for entry in storage_plan.entries)
    return ExportSummary(
        artifact_dir=output_dir,
        function_name=main_fn.name,
        num_user_inputs=result.num_user_inputs,
        num_weight_params=max(0, main_fn.num_args - result.num_user_inputs),
        vm_functions=len(result.executable.function_table),
        instructions=len(result.executable.instructions),
        storage_bytes=storage_bytes,
    )


def export_artifact(
    entrypoint: EntrypointRecipe,
    *,
    artifact_dir: str | Path,
    resources: Mapping[str, str | Path] | None = None,
    resource_specs: tuple[ResourceSpec, ...] = (),
    weight_package_dir: str | Path | None = None,
    packed_backends: tuple[PackedBackendRecipe, ...] = (),
    metadata: Mapping[str, Any] | None = None,
    compile_kernels: bool = True,
    compile_backends: bool = True,
    nvcc: str | None = None,
    backend_build_dir: str | Path | None = None,
    backend_library_dirs: tuple[str | Path, ...] = (),
    sm_arch: int = 89,
    options: Mapping[str, Any] | None = None,
    compile_mode: CompileMode | None = None,
) -> ExportSummary:
    summary = emit_entrypoint(
        entrypoint,
        artifact_dir,
        options=options,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        compile_mode=compile_mode,
    )
    specs = list(resource_specs)
    for name, src in (resources or {}).items():
        if name == "weight_package_dir":
            if weight_package_dir is None:
                weight_package_dir = src
            continue
        specs.append(ResourceSpec(name=name, path=src))
    resolved_backends = packed_backends or entrypoint.packed_backends
    resource_summary = prepare_artifact(
        model_id=entrypoint.model_id,
        entrypoint=entrypoint.name,
        artifact_dir=artifact_dir,
        weight_package_dir=weight_package_dir,
        resources=tuple(specs),
        metadata=dict(metadata or {}),
        packed_backends=resolved_backends,
        sm_arch=sm_arch,
        compile_kernels=compile_kernels,
        compile_backends=compile_backends,
        nvcc=nvcc,
        backend_build_dir=backend_build_dir,
        backend_library_dirs=backend_library_dirs,
    )
    return ExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def assert_normal_ir_has_no_backend_ops(module: IRModule) -> None:
    for fn_name, fn in module.functions.items():
        for op in fn.body.entry_block.ops:
            if isinstance(op, CudaCallOp):
                raise RuntimeError(f"{fn_name}: normal path emitted cuda_call")
            if isinstance(op, CallDPSOp):
                raise RuntimeError(f"{fn_name}: normal path emitted DPS/packed call")


def stamp_single_return_struct_info(module: IRModule) -> IRModule:
    functions: dict[str, Function] = {}
    for name, fn in module.functions.items():
        ret_si = fn.ret_struct_info
        term = fn.body.entry_block.ops[-1]
        if ret_si is None and isinstance(term, ReturnOp) and len(term.values) == 1:
            ret_si = getattr(term.values[0], "struct_info", None)
        functions[name] = Function(fn.body, ret_si)
    return IRModule(functions)


def _select_method(
    module: Module,
    compile_mode: str,
    *,
    normal: str,
    fast: str,
):
    _validate_module_contract(module)
    if compile_mode == "fast" and hasattr(module, fast):
        return getattr(module, fast)
    return getattr(module, normal)


def _validate_module_contract(root: Module) -> None:
    for path, module in root.named_modules():
        if isinstance(module, ModuleList):
            continue
        forward = type(module).__dict__.get("forward")
        if forward is None or forward is Module.forward:
            name = path or type(module).__name__
            raise RuntimeError(f"{name}: modules must implement forward()")
        if _method_source_mentions(forward, "forward_fast"):
            name = path or type(module).__name__
            raise RuntimeError(f"{name}.forward() must not call forward_fast()")


def _method_source_mentions(method, attr_name: str) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(method))
    except (OSError, TypeError):
        return False
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Attribute) and node.attr == attr_name
        for node in ast.walk(tree)
    )


__all__ = [
    "CompileMode",
    "CompileResult",
    "ExportSummary",
    "assert_normal_ir_has_no_backend_ops",
    "build_graph",
    "compile_entrypoint",
    "compile_ir_module",
    "emit_compile_result",
    "emit_entrypoint",
    "export_artifact",
    "normalize_compile_mode",
    "stamp_single_return_struct_info",
]
