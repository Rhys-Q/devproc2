"""DPSLoweringPass — rewrite CallOp → TensorCreateOp + CallDPSOp.

Must run after InferStructInfoPass so that CallOp results carry TensorStructInfo.

For each matched CallOp:
  1. Insert TensorCreateOp(empty) for the output buffer (same result_name).
  2. Replace the CallOp with a CallDPSOp whose output is the new buffer.
  3. Register the old result → new buffer result in _sub so downstream ops
     (ReturnOp, YieldOp, other CallOps) automatically reference the buffer.

Shape scalars (%M, %N, %K) in inputs are NOT inserted here; that is the job
of ShapeExprLoweringPass (M7/X2). For M6, inputs = original op arguments.

Effect is set to OpaqueEffect() for all lowered kernels. M5 will refine this
to WriteEffect once the effect system is implemented.
"""
from __future__ import annotations

from typing import Optional

from devproc2.ir.nodes import (
    Block,
    IRModule,
    OpaqueEffect,
    TensorStructInfo,
)
from devproc2.ir.ops import (
    CallDPSOp,
    CalleeKind,
    CallOp,
    TensorCreateKind,
    TensorCreateOp,
)
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
                        callee=kernel.kernel_name,
                        callee_kind=CalleeKind.kernel,
                        inputs=self.svs(op.args),
                        output=create_op.results[0],
                        effect=OpaqueEffect(),
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
            op_name=op.callee.lstrip("@"),
            device=si.device,
            input_dtypes=build_input_dtypes(op.args),
        )
        return self._registry.lookup(key, self._sm_arch, op)
