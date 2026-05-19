"""KernelSelectPass — pure analysis that maps each matchable CallOp to a KernelSpec."""
from __future__ import annotations

from typing import Optional

from devproc2.ir.nodes import IRModule, IRStage, Region, TensorStructInfo
from devproc2.compiler.op import LoweringKind
from devproc2.ir.op_ref import StandardOpRef
from devproc2.ir.ops import CallOp
from devproc2.kernel.registry import (
    KernelMatchKey,
    KernelRegistry,
    KernelSpec,
    build_input_dtypes,
)


class KernelSelectPass:
    """Traverse the module and return {CallOp: KernelSpec} for every matchable call.

    A call is matchable iff its first result carries a TensorStructInfo (i.e.
    InferStructInfoPass has already run) and the registry has a matching entry.
    This pass does not modify the IR.

    CallOp uses identity equality (eq=False), so object keys are unambiguous.

    sm_arch: target SM compute capability (e.g. 80, 90).  None = skip SM filter.
    """
    input_stage = IRStage.inferred
    output_stage = IRStage.inferred
    required_analysis: tuple[str, ...] = ()
    preserved_analysis: tuple[str, ...] = ()

    def __init__(self, registry: KernelRegistry, sm_arch: Optional[int] = None) -> None:
        self._registry = registry
        self._sm_arch = sm_arch

    def run(self, module: IRModule) -> dict[CallOp, KernelSpec]:
        result: dict[CallOp, KernelSpec] = {}
        for fn in module.functions.values():
            self._select_region(fn.body, result)
        return result

    def _select_region(self, region: Region, result: dict[CallOp, KernelSpec]) -> None:
        for block in region.blocks:
            for op in block.ops:
                if isinstance(op, CallOp) and op.results:
                    op_def = op.op_def if isinstance(op.op_ref, StandardOpRef) else None
                    if op_def is None:
                        continue
                    if op_def.lowering.kind != LoweringKind.kernel:
                        continue
                    si = op.results[0].struct_info
                    if isinstance(si, TensorStructInfo):
                        key = KernelMatchKey(
                            op_name=op.op_ref.name,
                            device=si.device,
                            input_dtypes=build_input_dtypes(op.args),
                        )
                        spec = self._registry.lookup(key, self._sm_arch, op)
                        if spec is not None:
                            result[op] = spec
                # Recurse into nested regions (IfOp, ForOp).
                # These attribute names cover all current region-bearing ops;
                # add new names here if a future Op introduces a region field
                # with a different attribute name.
                for attr in ("then_region", "else_region", "body_region"):
                    sub = getattr(op, attr, None)
                    if sub is not None:
                        self._select_region(sub, result)
