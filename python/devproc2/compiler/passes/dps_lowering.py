"""DPSLoweringPass — rewrite CallOp → TensorCreateOp + CallDPSOp.

Must run after InferStructInfoPass so that CallOp results carry TensorStructInfo.

For each matched CallOp:
  1. Insert TensorCreateOp(empty) for the output buffer (same result_name).
  2. Replace the CallOp with a CallDPSOp whose output is the new buffer.
  3. Register the old result → new buffer result in _sub so downstream ops
     (ReturnOp, YieldOp, other CallOps) automatically reference the buffer.

Shape scalars (%M, %N, %K) in inputs are NOT inserted here; that is the job
of ShapeExprLoweringPass (M7/X2). For M6, inputs = original op arguments.

Effect writes the destination buffer explicitly; opaque effects are reserved
for calls whose behavior cannot be modeled more precisely.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from devproc2.ir.nodes import (
    Block,
    EffectSummary,
    IRModule,
    IRStage,
    TensorStructInfo,
)
from devproc2.ir.op_ref import KernelRef, PackedFuncRef, StandardOpRef
from devproc2.ir.ops import (
    CallDPSOp,
    CallOp,
    CudaCallOp,
    TensorCreateKind,
    TensorCreateOp,
)
from devproc2.compiler.op import LoweringKind
from devproc2.kernel.registry import (
    KernelMatchKey,
    KernelLaunchSpec,
    KernelParamSpec,
    KernelRegistry,
    KernelSpec,
    build_input_dtypes,
    prim_expr_to_json_obj,
)
from devproc2.compiler.passes._rewriter import IRRewriter


class DPSLoweringPass(IRRewriter):
    """Lowers matched CallOps to TensorCreateOp + CallDPSOp pairs.

    sm_arch: target SM compute capability passed to the registry lookup.
             None = skip SM filter.
    """
    input_stage = IRStage.inferred
    output_stage = IRStage.dps
    required_analysis: tuple[str, ...] = ()
    preserved_analysis: tuple[str, ...] = ()

    def __init__(
        self,
        registry: KernelRegistry,
        sm_arch: Optional[int] = None,
        *,
        reference_fallback: bool = False,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._sm_arch = sm_arch
        self._reference_fallback = bool(reference_fallback)

    def run(self, module: IRModule) -> IRModule:
        return self.rewrite_module(module)

    def rewrite_block(self, block: Block) -> Block:
        new_ops = []
        for op in block.ops:
            if isinstance(op, CudaCallOp):
                dps_op = self._lower_cuda_call(op)
                new_ops.append(dps_op)
                continue
            if isinstance(op, CallOp) and op.results:
                op_def = op.op_def if isinstance(op.op_ref, StandardOpRef) else None
                if op_def is None:
                    new_op = self._subst_op(op)
                    for old_r, new_r in zip(op.results, new_op.results):
                        self._sub[old_r] = new_r
                    new_ops.append(new_op)
                    continue
                if op_def.lowering.kind != LoweringKind.kernel:
                    new_op = self._subst_op(op)
                    for old_r, new_r in zip(op.results, new_op.results):
                        self._sub[old_r] = new_r
                    new_ops.append(new_op)
                    continue
                si = op.results[0].struct_info
                kernel = self._lookup(op, si)
                if kernel is not None:
                    create_op = TensorCreateOp(
                        result_name=op.result_name,
                        kind=TensorCreateKind.empty,
                        shape=si.shape,
                        dtype=si.dtype,
                        device=si.device,
                    )
                    # Redirect downstream uses of the old CallOp result.
                    self._sub[op.results[0]] = create_op.results[0]
                    dps_op = CallDPSOp(
                        target_ref=KernelRef(kernel.kernel_name, kernel),
                        inputs=self.svs(op.args),
                        outputs=(create_op.results[0],),
                        effect=EffectSummary.write(create_op.results[0]),
                        attrs=op.attrs,
                    )
                    new_ops.append(create_op)
                    new_ops.append(dps_op)
                    continue
                if self._reference_fallback:
                    create_op = TensorCreateOp(
                        result_name=op.result_name,
                        kind=TensorCreateKind.empty,
                        shape=si.shape,
                        dtype=si.dtype,
                        device=si.device,
                    )
                    self._sub[op.results[0]] = create_op.results[0]
                    dps_op = CallDPSOp(
                        target_ref=PackedFuncRef(f"runtime.reference.{op.op_ref.name}"),
                        inputs=self.svs(op.args),
                        outputs=(create_op.results[0],),
                        effect=EffectSummary.write(create_op.results[0]),
                        attrs=op.attrs,
                    )
                    new_ops.append(create_op)
                    new_ops.append(dps_op)
                    continue
            # Default path: substitute operands, register result mapping.
            new_op = self._subst_op(op)
            for old_r, new_r in zip(op.results, new_op.results):
                self._sub[old_r] = new_r
            new_ops.append(new_op)
        return Block(block.args, tuple(new_ops))

    def _lower_cuda_call(self, op: CudaCallOp) -> CallDPSOp:
        op = self._subst_op(op)
        assert isinstance(op, CudaCallOp)
        spec = self._cuda_kernel_spec(op)
        return CallDPSOp(
            target_ref=KernelRef(spec.kernel_name, spec),
            inputs=op.args,
            outputs=(),
            effect=op.effect,
            attrs=op.attrs,
        )

    def _cuda_kernel_spec(self, op: CudaCallOp) -> KernelSpec:
        kernel_name = op.kernel_name or _auto_cuda_kernel_name(op)
        launch = op.launch if isinstance(op.launch, KernelLaunchSpec) else KernelLaunchSpec()
        sm_arches = op.sm_arches or ((self._sm_arch,) if self._sm_arch is not None else ())
        return KernelSpec(
            op_name=f"cuda.{op.symbol}",
            device=_cuda_call_device(op),
            input_dtypes=op.input_dtypes or _cuda_call_input_dtypes(op),
            kernel_name=kernel_name,
            backend="cuda",
            output_dtype=op.output_dtype or _cuda_call_output_dtype(op),
            symbol=op.symbol,
            sm_arches=sm_arches,
            launch=launch,
            params=_cuda_call_params(op),
            source_path=op.source_path,
            include_dirs=op.include_dirs,
            extra_nvcc_flags=op.extra_nvcc_flags,
            compile_options=op.compile_options,
        )

    def _lookup(self, op: CallOp, si: object) -> KernelSpec | None:
        if not isinstance(si, TensorStructInfo):
            return None
        key = KernelMatchKey(
            op_name=op.op_ref.name,
            device=si.device,
            input_dtypes=build_input_dtypes(op.args),
        )
        return self._registry.lookup(key, self._sm_arch, op)


def _auto_cuda_kernel_name(op: CudaCallOp) -> str:
    payload = {
        "source_path": op.source_path,
        "symbol": op.symbol,
        "launch": (
            op.launch.to_json_obj()
            if isinstance(op.launch, KernelLaunchSpec)
            else repr(op.launch)
        ),
        "attrs": op.attrs.to_python_dict(),
        "sm_arches": list(op.sm_arches),
        "include_dirs": list(op.include_dirs),
        "extra_nvcc_flags": list(op.extra_nvcc_flags),
        "compile_options": prim_expr_to_json_obj(dict(op.compile_options)),
        "input_dtypes": list(build_input_dtypes(op.args)),
        "explicit_input_dtypes": list(op.input_dtypes),
        "output_dtype": op.output_dtype,
        "output_indices": list(op.output_indices),
        "params": [
            p.to_json_obj() if hasattr(p, "to_json_obj") else repr(p)
            for p in op.params
        ],
    }
    blob = json.dumps(
        prim_expr_to_json_obj(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    digest = hashlib.sha1(blob).hexdigest()[:12]
    safe_symbol = re.sub(r"[^0-9A-Za-z_]+", "_", op.symbol).strip("_") or "cuda"
    return f"kernel.cuda.{safe_symbol}.{digest}"


def _cuda_call_device(op: CudaCallOp) -> str:
    for value in op.outputs + op.args:
        si = getattr(value, "struct_info", None)
        device = getattr(si, "device", None)
        if device is not None:
            return str(device)
    return "cuda"


def _cuda_call_output_dtype(op: CudaCallOp) -> str | None:
    if len(op.output_indices) != 1:
        return None
    idx = op.output_indices[0]
    if idx < 0 or idx >= len(op.args):
        return None
    si = getattr(op.args[idx], "struct_info", None)
    return getattr(si, "dtype", None)


def _cuda_call_input_dtypes(op: CudaCallOp) -> tuple[str, ...]:
    outputs = set(op.output_indices)
    return build_input_dtypes(
        tuple(value for i, value in enumerate(op.args) if i not in outputs)
    )


def _cuda_call_params(op: CudaCallOp) -> tuple[KernelParamSpec, ...]:
    if op.params:
        return tuple(op.params)
    params: list[KernelParamSpec] = []
    outputs = set(op.output_indices)
    for i, value in enumerate(op.args):
        kind, dtype = _param_kind_dtype(value)
        name = op.param_names[i] if i < len(op.param_names) else getattr(value, "name", f"arg{i}")
        params.append(
            KernelParamSpec(
                name=name,
                kind=kind,
                dtype=dtype,
                source="output" if i in outputs else "input",
                index=i,
            )
        )
    return tuple(params)


def _param_kind_dtype(value: object) -> tuple[str, str | None]:
    from devproc2.ir.nodes import Constant, ScalarStructInfo

    si = getattr(value, "struct_info", None)
    if isinstance(si, TensorStructInfo):
        return "tensor", si.dtype
    if isinstance(si, ScalarStructInfo):
        return "scalar", si.dtype
    if isinstance(value, Constant):
        if isinstance(value.value, bool):
            return "scalar", "bool"
        if isinstance(value.value, int):
            return "scalar", "int64"
        if isinstance(value.value, float):
            return "scalar", "float64"
    return "tensor", None
