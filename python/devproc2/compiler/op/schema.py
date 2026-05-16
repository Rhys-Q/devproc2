"""Standard op schema objects.

The op registry is the single source of truth for standard tensor ops.  IR
calls reference an op by callee name, while this module describes the stable
schema, attr normalization, inference, validation, and high-level lowering
metadata associated with that callee.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from devproc2.ir.nodes import StructInfo, TensorStructInfo


@dataclass(frozen=True)
class Input:
    name: str
    kind: str = "tensor"
    optional: bool = False


@dataclass(frozen=True)
class Attr:
    name: str
    type_name: str
    default: object = None
    required: bool = False


@dataclass(frozen=True)
class Output:
    name: str
    kind: str = "tensor"


class OpPatternKind(Enum):
    """Fusion/analysis pattern, modeled after TVM Relax's op pattern tags."""

    elementwise = "elementwise"
    broadcast = "broadcast"
    injective = "injective"
    reduction = "reduction"
    out_ewise_fusable = "out_ewise_fusable"
    tuple = "tuple"
    opaque = "opaque"


class PurityKind(Enum):
    pure = "pure"
    readonly = "readonly"
    impure = "impure"


class LoweringKind(Enum):
    none = "none"
    kernel = "kernel"
    builtin = "builtin"
    external = "external"


@dataclass(frozen=True)
class InferContext:
    args: tuple[Optional[StructInfo], ...]
    attrs: Mapping[str, object]

    def arg(self, index: int) -> Optional[StructInfo]:
        return self.args[index] if index < len(self.args) else None


InferFn = Callable[[InferContext], Optional[StructInfo]]
NormalizeFn = Callable[["CallLike"], "CallLike"]
ValidateFn = Callable[["CallLike"], None]


class CallLike:
    """Protocol-like base used only for lightweight type hints.

    Keeping this as a concrete class avoids importing ``CallOp`` from the IR
    layer, which would create an unnecessary dependency cycle.
    """

    callee: str
    args: tuple[object, ...]
    attrs: Mapping[str, object]


@dataclass(frozen=True)
class OpDef:
    name: str
    inputs: tuple[Input, ...]
    attrs: tuple[Attr, ...]
    outputs: tuple[Output, ...]
    infer: InferFn
    normalize: Optional[NormalizeFn] = None
    validate: Optional[ValidateFn] = None
    purity: PurityKind = PurityKind.pure
    pattern: OpPatternKind = OpPatternKind.opaque
    lowering_kind: LoweringKind = LoweringKind.kernel

    @property
    def callee(self) -> str:
        return f"@{self.name}"

    def normalize_attrs(
        self,
        attrs: Optional[Mapping[str, object]] = None,
        *,
        include_defaults: bool = True,
    ) -> dict[str, object]:
        provided = dict(attrs or {})
        declared = {attr.name: attr for attr in self.attrs}
        unknown = set(provided) - set(declared)
        if unknown:
            raise ValueError(f"{self.name}: unknown attrs: {', '.join(sorted(unknown))}")

        normalized = {}
        for attr in self.attrs:
            if attr.name in provided:
                normalized[attr.name] = _normalize_attr_value(self.name, attr, provided[attr.name])
            elif attr.required:
                raise ValueError(f"{self.name}: missing required attr {attr.name!r}")
            elif include_defaults:
                normalized[attr.name] = _normalize_attr_value(self.name, attr, attr.default)
        return normalized

    def validate_call(
        self,
        args: tuple[object, ...],
        attrs: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._validate_num_inputs(args)
        self.normalize_attrs(attrs, include_defaults=True)
        if self.validate is not None:
            self.validate(_SimpleCall(self.callee, args, attrs or {}))

    def infer_struct_info(
        self,
        args: tuple[Optional[StructInfo], ...],
        attrs: Optional[Mapping[str, object]] = None,
    ) -> Optional[StructInfo]:
        return self.infer(InferContext(args, self.normalize_attrs(attrs)))

    def _validate_num_inputs(self, args: tuple[object, ...]) -> None:
        min_inputs = sum(1 for inp in self.inputs if not inp.optional)
        max_inputs = len(self.inputs)
        if not (min_inputs <= len(args) <= max_inputs):
            if min_inputs == max_inputs:
                expected = str(max_inputs)
            else:
                expected = f"{min_inputs}..{max_inputs}"
            raise ValueError(f"{self.name}: expected {expected} inputs, got {len(args)}")


StandardOp = OpDef
InputDef = Input
AttrDef = Attr
OutputDef = Output


@dataclass(frozen=True)
class _SimpleCall(CallLike):
    callee: str
    args: tuple[object, ...]
    attrs: Mapping[str, object]


def _normalize_attr_value(op_name: str, attr: Attr, value: object) -> object:
    type_name = attr.type_name.strip()
    if " | " in type_name:
        errors = []
        for part in type_name.split("|"):
            candidate = Attr(attr.name, part.strip(), attr.default, attr.required)
            try:
                return _normalize_attr_value(op_name, candidate, value)
            except TypeError as err:
                errors.append(str(err))
        raise TypeError(
            f"{op_name}: attr {attr.name!r} expects {attr.type_name}, "
            f"got {type(value).__name__}"
        )
    if type_name.startswith("optional[") and type_name.endswith("]"):
        if value is None:
            return None
        candidate = Attr(attr.name, type_name[len("optional["):-1], attr.default, attr.required)
        return _normalize_attr_value(op_name, candidate, value)
    if type_name == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if _matches_type(type_name, value):
        if type_name == "array[int]" and isinstance(value, list):
            return tuple(value)
        if type_name == "array[str]" and isinstance(value, list):
            return tuple(value)
        return value
    raise TypeError(
        f"{op_name}: attr {attr.name!r} expects {attr.type_name}, "
        f"got {type(value).__name__}"
    )


def _matches_type(type_name: str, value: object) -> bool:
    if " | " in type_name:
        return any(_matches_type(part.strip(), value) for part in type_name.split("|"))
    if type_name.startswith("optional[") and type_name.endswith("]"):
        return value is None or _matches_type(type_name[len("optional["):-1], value)
    if type_name in ("None", "none"):
        return value is None
    if type_name in ("int", "Integer"):
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name in ("str", "string", "dtype"):
        return isinstance(value, str)
    if type_name == "shape":
        return isinstance(value, tuple)
    if type_name == "array[int]":
        return isinstance(value, (tuple, list)) and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        )
    if type_name == "array[str]":
        return isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value)
    if type_name == "tensor":
        return isinstance(value, TensorStructInfo)
    if type_name == "object":
        return True
    return True


__all__ = [
    "Attr",
    "AttrDef",
    "CallLike",
    "InferContext",
    "InferFn",
    "Input",
    "InputDef",
    "LoweringKind",
    "NormalizeFn",
    "OpDef",
    "OpPatternKind",
    "Output",
    "OutputDef",
    "PurityKind",
    "StandardOp",
    "ValidateFn",
]
