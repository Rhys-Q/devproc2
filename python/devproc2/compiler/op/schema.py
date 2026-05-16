"""Standard op schema objects.

The op registry is the single source of truth for standard tensor ops. IR
calls reference an op through StandardOpRef, while this module describes the
stable schema, attr normalization, inference, validation, and lowering policy.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from devproc2.ir.attrs import AttrDict, AttrType, wrap_attr_value
from devproc2.ir.nodes import DialectKind
from devproc2.ir.nodes import StructInfo


@dataclass(frozen=True)
class Input:
    name: str
    kind: str = "tensor"
    optional: bool = False
    variadic: bool = False


@dataclass(frozen=True)
class Attr:
    name: str
    type: AttrType
    default: object = None
    required: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            object.__setattr__(self, "type", AttrType.parse(self.type))

    @property
    def type_name(self) -> str:
        return self.type.describe()


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
class LoweringPolicy:
    kind: LoweringKind
    target: str | None = None

    @staticmethod
    def none() -> "LoweringPolicy":
        return LoweringPolicy(LoweringKind.none)

    @staticmethod
    def kernel() -> "LoweringPolicy":
        return LoweringPolicy(LoweringKind.kernel)

    @staticmethod
    def builtin(target: str | None = None) -> "LoweringPolicy":
        return LoweringPolicy(LoweringKind.builtin, target)

    @staticmethod
    def external(target: str | None = None) -> "LoweringPolicy":
        return LoweringPolicy(LoweringKind.external, target)


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

    name: str
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
    dialect: DialectKind = DialectKind.tensor
    lowering: LoweringPolicy = LoweringPolicy.kernel()

    def normalize_attrs(
        self,
        attrs: Optional[Mapping[str, object]] = None,
        *,
        include_defaults: bool = True,
    ) -> AttrDict:
        if isinstance(attrs, AttrDict):
            provided = attrs.to_python_dict()
        else:
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
        return AttrDict(normalized)

    def validate_call(
        self,
        args: tuple[object, ...],
        attrs: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._validate_num_inputs(args)
        normalized = self.normalize_attrs(attrs, include_defaults=True)
        if self.validate is not None:
            self.validate(_SimpleCall(self.name, args, normalized.to_python_dict()))

    def infer_struct_info(
        self,
        args: tuple[Optional[StructInfo], ...],
        attrs: Optional[Mapping[str, object]] = None,
    ) -> Optional[StructInfo]:
        return self.infer(InferContext(args, self.normalize_attrs(attrs).to_python_dict()))

    def _validate_num_inputs(self, args: tuple[object, ...]) -> None:
        variadic_inputs = [i for i, inp in enumerate(self.inputs) if inp.variadic]
        if variadic_inputs:
            if len(variadic_inputs) != 1 or variadic_inputs[0] != len(self.inputs) - 1:
                raise ValueError(f"{self.name}: variadic input must be the final input")
            variadic = self.inputs[variadic_inputs[0]]
            fixed_inputs = self.inputs[: variadic_inputs[0]]
            min_inputs = sum(1 for inp in fixed_inputs if not inp.optional)
            if not variadic.optional:
                min_inputs += 1
            if len(args) < min_inputs:
                raise ValueError(
                    f"{self.name}: expected at least {min_inputs} inputs, got {len(args)}"
                )
            return
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
    name: str
    args: tuple[object, ...]
    attrs: Mapping[str, object]


def _normalize_attr_value(op_name: str, attr: Attr, value: object):
    try:
        return wrap_attr_value(value, attr.type)
    except TypeError as err:
        raise TypeError(
            f"{op_name}: attr {attr.name!r} expects {attr.type_name}, "
            f"got {type(value).__name__}"
        ) from err


__all__ = [
    "Attr",
    "AttrDef",
    "CallLike",
    "InferContext",
    "InferFn",
    "Input",
    "InputDef",
    "LoweringKind",
    "LoweringPolicy",
    "NormalizeFn",
    "OpDef",
    "OpPatternKind",
    "Output",
    "OutputDef",
    "PurityKind",
    "StandardOp",
    "ValidateFn",
]
