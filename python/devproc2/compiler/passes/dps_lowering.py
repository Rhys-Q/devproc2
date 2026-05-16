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

from typing import Optional

from devproc2.ir.nodes import (
    Block,
    EffectSummary,
    IRModule,
    IRStage,
    TensorStructInfo,
)
from devproc2.ir.op_ref import KernelRef, StandardOpRef
from devproc2.ir.ops import (
    CallDPSOp,
    CallOp,
    TensorCreateKind,
    TensorCreateOp,
)
from devproc2.compiler.op import LoweringKind
from devproc2.kernel.registry import (
    KernelMatchKey,
    KernelRegistry,
    KernelSpec,
    build_input_dtypes,
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

    def __init__(self, registry: KernelRegistry, sm_arch: Optional[int] = None) -> None:
        super().__init__()
        self._registry = registry
        self._sm_arch = sm_arch

    def run(self, module: IRModule) -> IRModule:
        return self.rewrite_module(module)

    def rewrite_block(self, block: Block) -> Block:
        new_ops = []
        for op in block.ops:
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
            # Default path: substitute operands, register result mapping.
            new_op = self._subst_op(op)
            for old_r, new_r in zip(op.results, new_op.results):
                self._sub[old_r] = new_r
            new_ops.append(new_op)
        return Block(block.args, tuple(new_ops))

    def _lookup(self, op: CallOp, si: object) -> KernelSpec | None:
        if not isinstance(si, TensorStructInfo):
            return None
        key = KernelMatchKey(
            op_name=op.op_ref.name,
            device=si.device,
            input_dtypes=build_input_dtypes(op.args),
        )
        return self._registry.lookup(key, self._sm_arch, op)
