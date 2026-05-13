"""KernelSelectPass — pure analysis that maps each matchable CallOp to a KernelSpec."""
from __future__ import annotations

from devproc2.ir.nodes import IRModule, Region, TensorStructInfo
from devproc2.ir.ops import CallOp
from devproc2.kernel.registry import KernelMatchKey, KernelRegistry, KernelSpec


class KernelSelectPass:
    """Traverse the module and return {id(CallOp): KernelSpec} for every matchable call.

    A call is matchable iff its first result carries a TensorStructInfo (i.e.
    InferStructInfoPass has already run) and the registry has an entry for it.
    This pass does not modify the IR.
    """

    def __init__(self, registry: KernelRegistry) -> None:
        self._registry = registry

    def run(self, module: IRModule) -> dict[int, KernelSpec]:
        result: dict[int, KernelSpec] = {}
        for fn in module.functions.values():
            self._select_region(fn.body, result)
        return result

    def _select_region(self, region: Region, result: dict[int, KernelSpec]) -> None:
        for block in region.blocks:
            for op in block.ops:
                if isinstance(op, CallOp) and op.results:
                    si = op.results[0].struct_info
                    if isinstance(si, TensorStructInfo):
                        key = KernelMatchKey(
                            op_name=op.callee.lstrip("@"),
                            device=si.device,
                            dtype=si.dtype,
                        )
                        spec = self._registry.lookup(key, op)
                        if spec is not None:
                            result[id(op)] = spec
                # Recurse into nested regions (IfOp, ForOp).
                # These attribute names cover all current region-bearing ops;
                # add new names here if a future Op introduces a region field
                # with a different attribute name.
                for attr in ("then_region", "else_region", "body_region"):
                    sub = getattr(op, attr, None)
                    if sub is not None:
                        self._select_region(sub, result)
