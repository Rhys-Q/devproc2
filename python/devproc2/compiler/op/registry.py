"""Registry for standard IR ops."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from devproc2.ir.nodes import StructInfo
from devproc2.compiler.op.schema import (
    Attr,
    InferContext,
    InferFn,
    Input,
    LoweringKind,
    NormalizeFn,
    OpDef,
    OpPatternKind,
    Output,
    PurityKind,
    ValidateFn,
)


_OPS: dict[str, OpDef] = {}


def register(op: OpDef) -> OpDef:
    if op.name in _OPS:
        raise ValueError(f"op {op.name!r} is already registered")
    _OPS[op.name] = op
    return op


def register_op(
    *,
    name: str | None = None,
    inputs: tuple[str | Input, ...],
    attrs: tuple[Attr, ...] = (),
    outputs: tuple[str | Output, ...] = ("y",),
    infer: InferFn | None = None,
    normalize: NormalizeFn | None = None,
    validate: ValidateFn | None = None,
    purity: PurityKind = PurityKind.pure,
    pattern: OpPatternKind = OpPatternKind.opaque,
    lowering_kind: LoweringKind = LoweringKind.kernel,
):
    def decorator(fn):
        op = register(OpDef(
            name=name or fn.__name__,
            inputs=tuple(_input(item) for item in inputs),
            attrs=attrs,
            outputs=tuple(_output(item) for item in outputs),
            infer=infer or _same_as_first,
            normalize=normalize,
            validate=validate,
            purity=purity,
            pattern=pattern,
            lowering_kind=lowering_kind,
        ))
        fn.op_def = op
        return fn

    return decorator


def get(name_or_callee: str) -> Optional[OpDef]:
    return _OPS.get(name_or_callee.lstrip("@"))


def require(name_or_callee: str) -> OpDef:
    op = get(name_or_callee)
    if op is None:
        raise KeyError(f"unknown standard op {name_or_callee!r}")
    return op


def normalize_attrs(
    name_or_callee: str,
    attrs: Optional[Mapping[str, object]] = None,
    *,
    include_defaults: bool = True,
) -> dict[str, object]:
    op = require(name_or_callee)
    return op.normalize_attrs(attrs, include_defaults=include_defaults)


def validate_call(
    name_or_callee: str,
    args: tuple[object, ...],
    attrs: Optional[Mapping[str, object]] = None,
) -> None:
    require(name_or_callee).validate_call(args, attrs)


def infer_struct_info(
    name_or_callee: str,
    args: tuple[Optional[StructInfo], ...],
    attrs: Optional[Mapping[str, object]] = None,
) -> Optional[StructInfo]:
    op = get(name_or_callee)
    return None if op is None else op.infer_struct_info(args, attrs)


get_op = get


def _input(value: str | Input) -> Input:
    return Input(value) if isinstance(value, str) else value


def _output(value: str | Output) -> Output:
    return Output(value) if isinstance(value, str) else value


def _same_as_first(ctx: InferContext) -> Optional[StructInfo]:
    return ctx.arg(0)


__all__ = [
    "get",
    "get_op",
    "infer_struct_info",
    "normalize_attrs",
    "register",
    "register_op",
    "require",
    "validate_call",
]
