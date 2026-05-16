"""Neural-network op declarations."""
from __future__ import annotations

from devproc2.compiler.op.emit import emit
from devproc2.compiler.op.registry import register_op
from devproc2.compiler.op.schema import Attr, Input, OpPatternKind


@register_op(
    inputs=(Input("x"), Input("weight"), Input("bias")),
    attrs=(
        Attr("axes", "array[int]", default=(-1,)),
        Attr("epsilon", "float", default=1e-6),
        Attr("center", "bool", default=True),
        Attr("scale", "bool", default=True),
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
        Attr("axes", "array[int]", default=(-1,)),
        Attr("epsilon", "float", default=1e-6),
    ),
    pattern=OpPatternKind.injective,
)
def rms_norm(x, weight, axes=(-1,), epsilon: float = 1e-6, *, eps: float | None = None):
    if eps is not None:
        epsilon = eps
    return emit(rms_norm, x, weight, axes=tuple(axes), epsilon=epsilon)


@register_op(
    inputs=(Input("x"), Input("weight"), Input("cond")),
    attrs=(
        Attr("axes", "array[int]", default=(-1,)),
        Attr("epsilon", "float", default=1e-6),
    ),
    pattern=OpPatternKind.injective,
)
def adarms_norm(x, weight, cond, axes=(-1,), epsilon: float = 1e-6, *, eps: float | None = None):
    if eps is not None:
        epsilon = eps
    return emit(adarms_norm, x, weight, cond, axes=tuple(axes), epsilon=epsilon)


__all__ = [
    "adarms_norm",
    "layer_norm",
    "rms_norm",
]
