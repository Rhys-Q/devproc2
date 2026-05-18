"""Neural-network op declarations."""
from __future__ import annotations

from devproc2.compiler.op.emit import emit
from devproc2.compiler.op.registry import register_op
from devproc2.compiler.op.schema import Attr, AttrType, InferContext, Input, OpPatternKind


def _same_as_q(ctx: InferContext):
    return ctx.arg(0)


@register_op(
    inputs=(Input("q"), Input("k"), Input("v")),
    attrs=(
        Attr("scale", AttrType.float(), default=1.0),
        Attr("causal", AttrType.bool(), default=False),
    ),
    infer=_same_as_q,
    pattern=OpPatternKind.opaque,
)
def attention(q, k, v, scale: float = 1.0, causal: bool = False):
    return emit(attention, q, k, v, scale=scale, causal=causal)


@register_op(
    inputs=(Input("x"), Input("weight"), Input("bias")),
    attrs=(
        Attr("axes", AttrType.array(AttrType.int()), default=(-1,)),
        Attr("epsilon", AttrType.float(), default=1e-6),
        Attr("center", AttrType.bool(), default=True),
        Attr("scale", AttrType.bool(), default=True),
    ),
    pattern=OpPatternKind.injective,
)
def layer_norm(
    x,
    weight,
    bias,
    axes=(-1,),
    epsilon: float = 1e-6,
    *,
    eps: float | None = None,
    normalized_shape=(),
    center: bool = True,
    scale: bool = True,
):
    if eps is not None:
        epsilon = eps
    if normalized_shape and axes == (-1,):
        axes = tuple(range(-len(tuple(normalized_shape)), 0))
    return emit(
        layer_norm,
        x,
        weight,
        bias,
        axes=tuple(axes),
        epsilon=epsilon,
        center=center,
        scale=scale,
    )


@register_op(
    inputs=(Input("x"), Input("weight")),
    attrs=(
        Attr("axes", AttrType.array(AttrType.int()), default=(-1,)),
        Attr("epsilon", AttrType.float(), default=1e-6),
    ),
    pattern=OpPatternKind.injective,
)
def rms_norm(x, weight, axes=(-1,), epsilon: float = 1e-6, *, eps: float | None = None):
    if eps is not None:
        epsilon = eps
    return emit(rms_norm, x, weight, axes=tuple(axes), epsilon=epsilon)


@register_op(
    inputs=(Input("x"),),
    attrs=(
        Attr("axes", AttrType.array(AttrType.int()), default=(-1,)),
        Attr("epsilon", AttrType.float(), default=1e-6),
    ),
    infer=_same_as_q,
    pattern=OpPatternKind.injective,
)
def rms_norm_unit(x, axes=(-1,), epsilon: float = 1e-6, *, eps: float | None = None):
    if eps is not None:
        epsilon = eps
    return emit(rms_norm_unit, x, axes=tuple(axes), epsilon=epsilon)


@register_op(
    inputs=(Input("x"), Input("weight"), Input("cond")),
    attrs=(
        Attr("axes", AttrType.array(AttrType.int()), default=(-1,)),
        Attr("epsilon", AttrType.float(), default=1e-6),
    ),
    pattern=OpPatternKind.injective,
)
def adarms_norm(x, weight, cond, axes=(-1,), epsilon: float = 1e-6, *, eps: float | None = None):
    if eps is not None:
        epsilon = eps
    return emit(adarms_norm, x, weight, cond, axes=tuple(axes), epsilon=epsilon)


__all__ = [
    "adarms_norm",
    "attention",
    "layer_norm",
    "rms_norm",
    "rms_norm_unit",
]
