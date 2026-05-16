"""Specs and parameters for the nn frontend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from devproc2.ir.nodes import ObjectStructInfo, ScalarStructInfo, TensorStructInfo
from devproc2.ir.prim_expr import IntImm, PrimExpr


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[PrimExpr, ...]
    dtype: str
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )

    @property
    def struct_info(self) -> TensorStructInfo:
        return TensorStructInfo(self.shape, self.dtype, self.device)


@dataclass(frozen=True)
class ScalarSpec:
    dtype: str

    @property
    def struct_info(self) -> ScalarStructInfo:
        return ScalarStructInfo(self.dtype)


@dataclass(frozen=True)
class ObjectSpec:
    type_key: str
    role: Optional[str] = None

    @property
    def struct_info(self) -> ObjectStructInfo:
        return ObjectStructInfo(self.type_key, self.role)


@dataclass(frozen=True)
class Parameter:
    shape: tuple[PrimExpr, ...]
    dtype: str
    device: str = "cuda"
    layout: str = "row_major"
    role: Literal["weight", "constant_tensor"] = "weight"
    name: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )

    @property
    def struct_info(self) -> TensorStructInfo:
        return TensorStructInfo(self.shape, self.dtype, self.device)


def with_parameter_name(parameter: Parameter, name: str) -> Parameter:
    if parameter.name == name:
        return parameter
    return Parameter(
        shape=parameter.shape,
        dtype=parameter.dtype,
        device=parameter.device,
        layout=parameter.layout,
        role=parameter.role,
        name=name,
    )


__all__ = [
    "ObjectSpec",
    "Parameter",
    "ScalarSpec",
    "TensorSpec",
    "with_parameter_name",
]
