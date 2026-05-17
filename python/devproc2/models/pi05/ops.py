"""Pi0.5 fast-path standard op declarations.

These ops are not used by default model forward methods. They are explicit
fast-path hooks emitted by forward_fast and lowered to hand-written CUDA
kernels through the KernelRegistry.
"""
from __future__ import annotations

from typing import Optional

from devproc2.compiler.op.emit import emit
from devproc2.compiler.op.registry import register_op
from devproc2.compiler.op.schema import InferContext, Input, OpPatternKind
from devproc2.ir.nodes import StructInfo


def _same_as_x(ctx: InferContext) -> Optional[StructInfo]:
    return ctx.arg(0)


@register_op(
    name="pi05.encoder_ffn_fp8_fused",
    inputs=(
        Input("x"),
        Input("gate_up_w_fp8"),
        Input("gate_up_w_scale"),
        Input("down_w_fp8"),
        Input("down_w_scale"),
        Input("act0_scale"),
        Input("act1_scale"),
    ),
    infer=_same_as_x,
    pattern=OpPatternKind.opaque,
)
def encoder_ffn_fp8_fused(
    x,
    gate_up_w_fp8,
    gate_up_w_scale,
    down_w_fp8,
    down_w_scale,
    act0_scale,
    act1_scale,
):
    return emit(
        encoder_ffn_fp8_fused,
        x,
        gate_up_w_fp8,
        gate_up_w_scale,
        down_w_fp8,
        down_w_scale,
        act0_scale,
        act1_scale,
    )


@register_op(
    name="pi05.decoder_ffn_fp8_fused",
    inputs=(
        Input("x"),
        Input("style"),
        Input("gate_up_w_fp8"),
        Input("gate_up_w_scale"),
        Input("down_w_fp8"),
        Input("down_w_scale"),
        Input("act0_scale"),
        Input("act1_scale"),
    ),
    infer=_same_as_x,
    pattern=OpPatternKind.opaque,
)
def decoder_ffn_fp8_fused(
    x,
    style,
    gate_up_w_fp8,
    gate_up_w_scale,
    down_w_fp8,
    down_w_scale,
    act0_scale,
    act1_scale,
):
    return emit(
        decoder_ffn_fp8_fused,
        x,
        style,
        gate_up_w_fp8,
        gate_up_w_scale,
        down_w_fp8,
        down_w_scale,
        act0_scale,
        act1_scale,
    )


__all__ = [
    "decoder_ffn_fp8_fused",
    "encoder_ffn_fp8_fused",
]
