"""Typed compile-time attributes for IR operations."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Optional

from devproc2.ir.prim_expr import IntImm, PrimExpr


@dataclass(frozen=True)
class AttrType:
    kind: str
    element: Optional["AttrType"] = None
    options: tuple["AttrType", ...] = ()

    @staticmethod
    def int() -> "AttrType":
        return AttrType("int")

    @staticmethod
    def float() -> "AttrType":
        return AttrType("float")

    @staticmethod
    def bool() -> "AttrType":
        return AttrType("bool")

    @staticmethod
    def string() -> "AttrType":
        return AttrType("string")

    @staticmethod
    def dtype() -> "AttrType":
        return AttrType("dtype")

    @staticmethod
    def device() -> "AttrType":
        return AttrType("device")

    @staticmethod
    def shape() -> "AttrType":
        return AttrType("shape")

    @staticmethod
    def prim_expr() -> "AttrType":
        return AttrType("prim_expr")

    @staticmethod
    def none() -> "AttrType":
        return AttrType("none")

    @staticmethod
    def object() -> "AttrType":
        return AttrType("object")

    @staticmethod
    def array(element: "AttrType") -> "AttrType":
        return AttrType("array", element=element)

    @staticmethod
    def optional(inner: "AttrType") -> "AttrType":
        return AttrType("union", options=(inner, AttrType.none()))

    @staticmethod
    def union(*options: "AttrType") -> "AttrType":
        return AttrType("union", options=tuple(options))

    @staticmethod
    def parse(spec: str) -> "AttrType":
        spec = spec.strip()
        if " | " in spec:
            return AttrType.union(*(AttrType.parse(part) for part in spec.split("|")))
        if spec.startswith("optional[") and spec.endswith("]"):
            return AttrType.optional(AttrType.parse(spec[len("optional[") : -1]))
        if spec.startswith("array[") and spec.endswith("]"):
            return AttrType.array(AttrType.parse(spec[len("array[") : -1]))
        table = {
            "int": AttrType.int(),
            "Integer": AttrType.int(),
            "float": AttrType.float(),
            "bool": AttrType.bool(),
            "str": AttrType.string(),
            "string": AttrType.string(),
            "dtype": AttrType.dtype(),
            "device": AttrType.device(),
            "shape": AttrType.shape(),
            "prim_expr": AttrType.prim_expr(),
            "None": AttrType.none(),
            "none": AttrType.none(),
            "object": AttrType.object(),
            "tensor": AttrType.object(),
        }
        return table.get(spec, AttrType.object())

    def describe(self) -> str:
        if self.kind == "array" and self.element is not None:
            return f"array[{self.element.describe()}]"
        if self.kind == "union":
            return " | ".join(option.describe() for option in self.options)
        if self.kind == "none":
            return "None"
        return self.kind


class AttrValue:
    def to_python(self) -> object:
        raise NotImplementedError

    def to_json_obj(self) -> object:
        return self.to_python()


@dataclass(frozen=True)
class IntAttr(AttrValue):
    value: int

    def to_python(self) -> int:
        return self.value


@dataclass(frozen=True)
class FloatAttr(AttrValue):
    value: float

    def to_python(self) -> float:
        return self.value


@dataclass(frozen=True)
class BoolAttr(AttrValue):
    value: bool

    def to_python(self) -> bool:
        return self.value


@dataclass(frozen=True)
class StringAttr(AttrValue):
    value: str

    def to_python(self) -> str:
        return self.value


@dataclass(frozen=True)
class DTypeAttr(StringAttr):
    pass


@dataclass(frozen=True)
class DeviceAttr(StringAttr):
    pass


@dataclass(frozen=True)
class PrimExprAttr(AttrValue):
    value: PrimExpr

    def __post_init__(self) -> None:
        if isinstance(self.value, int):
            object.__setattr__(self, "value", IntImm(self.value))

    def to_python(self) -> PrimExpr:
        return self.value

    def to_json_obj(self) -> object:
        return _prim_expr_to_json_obj(self.value)


@dataclass(frozen=True)
class ShapeAttr(AttrValue):
    values: tuple[PrimExpr, ...]

    def __post_init__(self) -> None:
        normalized = []
        for value in self.values:
            if isinstance(value, bool):
                raise TypeError("shape dimensions must be int or PrimExpr")
            if isinstance(value, int):
                normalized.append(IntImm(value))
            elif isinstance(value, PrimExpr):
                normalized.append(value)
            else:
                raise TypeError("shape dimensions must be int or PrimExpr")
        object.__setattr__(self, "values", tuple(normalized))

    def to_python(self) -> tuple[PrimExpr, ...]:
        return self.values

    def to_json_obj(self) -> list[object]:
        return [_prim_expr_to_json_obj(value) for value in self.values]


@dataclass(frozen=True)
class ArrayAttr(AttrValue):
    values: tuple[AttrValue, ...]

    def to_python(self) -> tuple[object, ...]:
        return tuple(value.to_python() for value in self.values)

    def to_json_obj(self) -> list[object]:
        return [value.to_json_obj() for value in self.values]


@dataclass(frozen=True)
class DictAttr(AttrValue):
    values: Mapping[str, AttrValue]

    def to_python(self) -> dict[str, object]:
        return {key: value.to_python() for key, value in self.values.items()}

    def to_json_obj(self) -> dict[str, object]:
        return {key: value.to_json_obj() for key, value in self.values.items()}


@dataclass(frozen=True)
class NoneAttr(AttrValue):
    def to_python(self) -> None:
        return None


@dataclass(frozen=True)
class ObjectAttr(AttrValue):
    value: object

    def to_python(self) -> object:
        return self.value


@dataclass(frozen=True)
class AttrDict(Mapping[str, AttrValue]):
    values: Mapping[str, AttrValue]

    @staticmethod
    def empty() -> "AttrDict":
        return AttrDict({})

    @staticmethod
    def from_python(values: Mapping[str, object] | None = None) -> "AttrDict":
        return AttrDict({key: wrap_attr_value(value) for key, value in (values or {}).items()})

    def __getitem__(self, key: str) -> AttrValue:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get_py(self, key: str, default: object = None) -> object:
        if key not in self.values:
            return default
        return self.values[key].to_python()

    def to_python_dict(self) -> dict[str, object]:
        return {key: value.to_python() for key, value in self.values.items()}

    def to_json_obj(self) -> dict[str, object]:
        return {key: value.to_json_obj() for key, value in self.values.items()}

    def __bool__(self) -> bool:
        return bool(self.values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AttrDict):
            return dict(self.values) == dict(other.values)
        if isinstance(other, Mapping):
            return self.to_python_dict() == dict(other)
        return False


def wrap_attr_value(value: object, attr_type: AttrType | None = None) -> AttrValue:
    if isinstance(value, AttrValue):
        return value

    if attr_type is not None:
        if attr_type.kind == "union":
            errors: list[Exception] = []
            for option in attr_type.options:
                try:
                    return wrap_attr_value(value, option)
                except TypeError as err:
                    errors.append(err)
            raise TypeError(f"expects {attr_type.describe()}, got {type(value).__name__}")
        if attr_type.kind == "none":
            if value is None:
                return NoneAttr()
            raise TypeError(f"expects None, got {type(value).__name__}")
        if attr_type.kind == "array":
            if not isinstance(value, (tuple, list)):
                raise TypeError(f"expects {attr_type.describe()}, got {type(value).__name__}")
            elem_type = attr_type.element or AttrType.object()
            return ArrayAttr(tuple(wrap_attr_value(item, elem_type) for item in value))
        if attr_type.kind == "shape":
            if not isinstance(value, (tuple, list)):
                raise TypeError(f"expects shape, got {type(value).__name__}")
            return ShapeAttr(tuple(value))
        if attr_type.kind == "prim_expr":
            if isinstance(value, (PrimExpr, int)):
                return PrimExprAttr(value)
            raise TypeError(f"expects prim_expr, got {type(value).__name__}")
        if attr_type.kind == "dtype":
            if isinstance(value, str):
                return DTypeAttr(value)
            raise TypeError(f"expects dtype, got {type(value).__name__}")
        if attr_type.kind == "device":
            if isinstance(value, str):
                return DeviceAttr(value)
            raise TypeError(f"expects device, got {type(value).__name__}")
        if attr_type.kind == "string":
            if isinstance(value, str):
                return StringAttr(value)
            raise TypeError(f"expects string, got {type(value).__name__}")
        if attr_type.kind == "int":
            if isinstance(value, int) and not isinstance(value, bool):
                return IntAttr(value)
            raise TypeError(f"expects int, got {type(value).__name__}")
        if attr_type.kind == "float":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return FloatAttr(float(value))
            raise TypeError(f"expects float, got {type(value).__name__}")
        if attr_type.kind == "bool":
            if isinstance(value, bool):
                return BoolAttr(value)
            raise TypeError(f"expects bool, got {type(value).__name__}")
        return ObjectAttr(value)

    if value is None:
        return NoneAttr()
    if isinstance(value, bool):
        return BoolAttr(value)
    if isinstance(value, int):
        return IntAttr(value)
    if isinstance(value, float):
        return FloatAttr(value)
    if isinstance(value, str):
        return StringAttr(value)
    if isinstance(value, PrimExpr):
        return PrimExprAttr(value)
    if isinstance(value, (tuple, list)):
        return ArrayAttr(tuple(wrap_attr_value(item) for item in value))
    if isinstance(value, Mapping):
        return DictAttr({str(key): wrap_attr_value(val) for key, val in value.items()})
    return ObjectAttr(value)


def _prim_expr_to_json_obj(value: PrimExpr) -> object:
    if isinstance(value, IntImm):
        return value.value
    if hasattr(value, "name"):
        payload: dict[str, object] = {
            "kind": type(value).__name__,
            "name": getattr(value, "name"),
        }
        upper = getattr(value, "upper", None)
        if upper is not None:
            payload["upper"] = upper
        return payload
    if hasattr(value, "lhs") and hasattr(value, "rhs"):
        return {
            "kind": type(value).__name__,
            "lhs": _prim_expr_to_json_obj(getattr(value, "lhs")),
            "rhs": _prim_expr_to_json_obj(getattr(value, "rhs")),
        }
    return {"kind": type(value).__name__, "repr": repr(value)}


__all__ = [
    "ArrayAttr",
    "AttrDict",
    "AttrType",
    "AttrValue",
    "BoolAttr",
    "DTypeAttr",
    "DeviceAttr",
    "DictAttr",
    "FloatAttr",
    "IntAttr",
    "NoneAttr",
    "ObjectAttr",
    "PrimExprAttr",
    "ShapeAttr",
    "StringAttr",
    "wrap_attr_value",
]
