"""Tensor op declarations."""
from __future__ import annotations

from typing import Optional

from devproc2.compiler.op.emit import emit
from devproc2.compiler.op.infer import broadcast as infer_broadcast
from devproc2.compiler.op.infer import cat as infer_cat
from devproc2.compiler.op.infer import comparison as infer_comparison
from devproc2.compiler.op.infer import embedding as infer_embedding
from devproc2.compiler.op.infer import matmul as infer_matmul
from devproc2.compiler.op.infer import permute_dims as infer_permute_dims
from devproc2.compiler.op.infer import reshape as infer_reshape
from devproc2.compiler.op.infer import same_as_first
from devproc2.compiler.op.registry import register_op
from devproc2.compiler.op.schema import Attr, AttrType, Input, LoweringPolicy, OpPatternKind


@register_op(inputs=("x",), pattern=OpPatternKind.elementwise)
def relu(x):
    return emit(relu, x)


@register_op(inputs=("x",), pattern=OpPatternKind.elementwise)
def silu(x):
    return emit(silu, x)


@register_op(inputs=("lhs", "rhs"), infer=infer_broadcast, pattern=OpPatternKind.broadcast)
def add(lhs, rhs):
    return emit(add, lhs, rhs)


@register_op(inputs=("lhs", "rhs"), infer=infer_broadcast, pattern=OpPatternKind.broadcast)
def multiply(lhs, rhs):
    return emit(multiply, lhs, rhs)


@register_op(
    inputs=("x",),
    attrs=(Attr("approximate", AttrType.string(), default="none"),),
    pattern=OpPatternKind.elementwise,
)
def gelu(x, approximate: str = "none"):
    return emit(gelu, x, approximate=approximate)


@register_op(
    inputs=("indices", "weight"),
    attrs=(
        Attr("padding_idx", AttrType.optional(AttrType.int()), default=None),
    ),
    outputs=("embeddings",),
    infer=infer_embedding,
    pattern=OpPatternKind.injective,
)
def embedding(indices, weight, padding_idx: Optional[int] = None):
    return emit(embedding, indices, weight, padding_idx=padding_idx)


@register_op(
    name="permute_dims",
    inputs=("x",),
    attrs=(Attr("axes", AttrType.optional(AttrType.array(AttrType.int())), default=None),),
    infer=infer_permute_dims,
    pattern=OpPatternKind.injective,
)
def permute_dims(x, axes: Optional[tuple[int, ...]] = None):
    return emit(permute_dims, x, axes=None if axes is None else tuple(axes))


def transpose(x, axes: Optional[tuple[int, ...]] = None):
    """Compatibility alias for the standard permute_dims op."""
    return permute_dims(x, axes=axes)


@register_op(
    inputs=("x",),
    attrs=(Attr("shape", AttrType.shape(), required=True),),
    infer=infer_reshape,
    pattern=OpPatternKind.injective,
)
def reshape(x, shape):
    if isinstance(shape, (tuple, list)):
        normalized_shape = tuple(shape)
    else:
        normalized_shape = (shape,)
    return emit(reshape, x, shape=normalized_shape)


@register_op(
    inputs=(Input("inputs", variadic=True),),
    attrs=(Attr("axis", AttrType.int(), default=0),),
    infer=infer_cat,
    pattern=OpPatternKind.injective,
)
def cat(inputs, axis: int = 0):
    return emit(cat, *tuple(inputs), axis=axis)


@register_op(
    inputs=("a", "b"),
    attrs=(
        Attr("out_dtype", AttrType.optional(AttrType.dtype()), default=None),
        Attr("transpose_a", AttrType.bool(), default=False),
        Attr("transpose_b", AttrType.bool(), default=False),
    ),
    infer=infer_matmul,
    pattern=OpPatternKind.out_ewise_fusable,
)
def matmul(
    a,
    b,
    out_dtype: Optional[str] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
):
    return emit(
        matmul,
        a,
        b,
        out_dtype=out_dtype,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


@register_op(
    name="identity",
    inputs=("x",),
    infer=same_as_first,
    pattern=OpPatternKind.injective,
    lowering=LoweringPolicy.none(),
)
def identity(x):
    return emit(identity, x)


def _register_comparison(name: str):
    @register_op(
        name=name,
        inputs=("lhs", "rhs"),
        infer=infer_comparison,
        pattern=OpPatternKind.broadcast,
        lowering=LoweringPolicy.none(),
    )
    def _cmp(lhs, rhs):
        return emit(_cmp, lhs, rhs)

    _cmp.__name__ = name
    return _cmp


greater = _register_comparison("__gt__")
greater_equal = _register_comparison("__ge__")
less = _register_comparison("__lt__")
less_equal = _register_comparison("__le__")
equal = _register_comparison("__eq__")
not_equal = _register_comparison("__ne__")


__all__ = [
    "add",
    "cat",
    "embedding",
    "equal",
    "gelu",
    "greater",
    "greater_equal",
    "identity",
    "less",
    "less_equal",
    "matmul",
    "multiply",
    "not_equal",
    "permute_dims",
    "relu",
    "reshape",
    "silu",
    "transpose",
]
