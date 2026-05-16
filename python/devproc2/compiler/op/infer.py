"""Shared struct-info inference helpers for standard ops."""
from __future__ import annotations

from typing import Optional

from devproc2.ir.nodes import ScalarStructInfo, StructInfo, TensorStructInfo
from devproc2.compiler.op.schema import InferContext
from devproc2.ir.prim_expr import IntImm, PrimExpr, prim_expr_structural_eq


def same_as_first(ctx: InferContext) -> Optional[StructInfo]:
    return ctx.arg(0)


def broadcast(ctx: InferContext) -> Optional[StructInfo]:
    lhs = ctx.arg(0)
    rhs = ctx.arg(1)
    lhs_tensor = tensor(lhs)
    rhs_tensor = tensor(rhs)
    if lhs_tensor is not None and rhs_tensor is not None:
        shape = broadcast_shape(lhs_tensor.shape, rhs_tensor.shape)
        if shape is None:
            return None
        _check_same_device(lhs_tensor, rhs_tensor)
        return TensorStructInfo(shape, lhs_tensor.dtype, lhs_tensor.device)
    if lhs_tensor is not None:
        return lhs_tensor
    if rhs_tensor is not None:
        return rhs_tensor
    if isinstance(lhs, ScalarStructInfo) and isinstance(rhs, ScalarStructInfo):
        return lhs
    return None


def comparison(ctx: InferContext) -> Optional[StructInfo]:
    out = broadcast(ctx)
    if isinstance(out, TensorStructInfo):
        return TensorStructInfo(out.shape, "bool", out.device)
    if isinstance(out, ScalarStructInfo):
        return ScalarStructInfo("bool")
    return None


def embedding(ctx: InferContext) -> Optional[StructInfo]:
    indices = tensor(ctx.arg(0))
    weight = tensor(ctx.arg(1))
    if indices is None or weight is None:
        return None
    if indices.dtype not in ("int32", "int64"):
        raise ValueError(f"embedding: indices must have int32/int64 dtype, got {indices.dtype}")
    if len(weight.shape) != 2:
        raise ValueError(f"embedding: weight must be rank-2, got rank {len(weight.shape)}")
    return TensorStructInfo(
        indices.shape + (weight.shape[1],),
        weight.dtype,
        weight.device,
    )


def permute_dims(ctx: InferContext) -> Optional[StructInfo]:
    info = tensor(ctx.arg(0))
    if info is None:
        return None
    axes = normalize_axes(ctx.attrs["axes"], len(info.shape))
    shape = tuple(info.shape[axis] for axis in axes)
    return info if shape == info.shape else TensorStructInfo(shape, info.dtype, info.device)


def matmul(ctx: InferContext) -> Optional[StructInfo]:
    lhs = tensor(ctx.arg(0))
    rhs = tensor(ctx.arg(1))
    if lhs is None or rhs is None:
        return None
    _check_same_device(lhs, rhs)
    if not lhs.shape or not rhs.shape:
        raise ValueError("matmul: operands must not be scalar tensors")

    lhs_shape = lhs.shape
    rhs_shape = rhs.shape
    lhs_was_1d = len(lhs_shape) == 1
    rhs_was_1d = len(rhs_shape) == 1
    if lhs_was_1d:
        lhs_shape = (IntImm(1),) + lhs_shape
    if rhs_was_1d:
        rhs_shape = rhs_shape + (IntImm(1),)

    if not dims_may_equal(lhs_shape[-1], rhs_shape[-2]):
        raise ValueError(
            f"matmul: reduction dims must match, got {lhs_shape[-1]!r} and {rhs_shape[-2]!r}"
        )

    batch_shape = broadcast_shape(lhs_shape[:-2], rhs_shape[:-2])
    if batch_shape is None:
        return None
    out_dtype = ctx.attrs["out_dtype"]
    dtype = out_dtype if isinstance(out_dtype, str) else lhs.dtype
    out_shape = batch_shape
    if not lhs_was_1d:
        out_shape += (lhs_shape[-2],)
    if not rhs_was_1d:
        out_shape += (rhs_shape[-1],)
    return TensorStructInfo(out_shape, dtype, lhs.device)


def tensor(info: Optional[StructInfo]) -> Optional[TensorStructInfo]:
    return info if isinstance(info, TensorStructInfo) else None


def normalize_dim(dim: object, rank: int) -> int:
    if not isinstance(dim, int):
        raise ValueError(f"axis must be int, got {type(dim).__name__}")
    if dim < 0:
        dim += rank
    if not 0 <= dim < rank:
        raise ValueError(f"axis {dim} is out of range for rank {rank}")
    return dim


def normalize_axes(axes: object, rank: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(reversed(range(rank)))
    if not isinstance(axes, (tuple, list)):
        raise ValueError(f"axes must be a tuple/list of int or None, got {type(axes).__name__}")
    if len(axes) != rank:
        raise ValueError(f"axes length must equal rank {rank}, got {len(axes)}")
    normalized = tuple(normalize_dim(axis, rank) for axis in axes)
    if len(set(normalized)) != rank:
        raise ValueError(f"axes must be a permutation, got {axes!r}")
    return normalized


def broadcast_shape(
    lhs: tuple[PrimExpr, ...],
    rhs: tuple[PrimExpr, ...],
) -> Optional[tuple[PrimExpr, ...]]:
    out = []
    for i in range(1, max(len(lhs), len(rhs)) + 1):
        ldim = lhs[-i] if i <= len(lhs) else IntImm(1)
        rdim = rhs[-i] if i <= len(rhs) else IntImm(1)
        if is_one(ldim):
            out.append(rdim)
        elif is_one(rdim) or dims_equal(ldim, rdim):
            out.append(ldim)
        elif isinstance(ldim, IntImm) and isinstance(rdim, IntImm):
            raise ValueError(f"broadcast: incompatible dimensions {ldim.value} and {rdim.value}")
        else:
            return None
    return tuple(reversed(out))


def broadcast_prefix(
    lhs: tuple[PrimExpr, ...],
    rhs: tuple[PrimExpr, ...],
) -> Optional[tuple[PrimExpr, ...]]:
    return broadcast_shape(lhs, rhs)


def is_one(dim: PrimExpr) -> bool:
    return isinstance(dim, IntImm) and dim.value == 1


def dims_equal(lhs: PrimExpr, rhs: PrimExpr) -> bool:
    return prim_expr_structural_eq(lhs, rhs)


def dims_may_equal(lhs: PrimExpr, rhs: PrimExpr) -> bool:
    if dims_equal(lhs, rhs):
        return True
    if isinstance(lhs, IntImm) and isinstance(rhs, IntImm):
        return lhs.value == rhs.value
    return True


def _check_same_device(lhs: TensorStructInfo, rhs: TensorStructInfo) -> None:
    if lhs.device != rhs.device:
        raise ValueError(f"device mismatch: {lhs.device!r} vs {rhs.device!r}")


__all__ = [
    "broadcast",
    "broadcast_prefix",
    "broadcast_shape",
    "comparison",
    "dims_equal",
    "dims_may_equal",
    "embedding",
    "is_one",
    "matmul",
    "normalize_dim",
    "normalize_axes",
    "permute_dims",
    "same_as_first",
    "tensor",
]
