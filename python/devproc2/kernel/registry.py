"""Backend-neutral kernel registry for standard op CUDA implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, TYPE_CHECKING

from devproc2.ir.attrs import AttrDict
from devproc2.ir.prim_expr import (
    Add,
    CeilDiv,
    FloorDiv,
    IntImm,
    Max,
    Min,
    Mul,
    PrimExpr,
    PrimVar,
    Sub,
)

if TYPE_CHECKING:
    from devproc2.ir.ops import CallOp


LaunchExpr = int | str | PrimExpr
BackendName = str


def _tuple3(values: tuple[object, ...] | list[object], field_name: str) -> tuple:
    values = tuple(values)
    if len(values) != 3:
        raise ValueError(f"{field_name} must have exactly 3 values, got {len(values)}")
    return values


def prim_expr_to_json_obj(value: object) -> object:
    """Return a stable JSON shape for launch/constraint primitive expressions."""
    if isinstance(value, IntImm):
        return value.value
    if isinstance(value, PrimVar):
        payload: dict[str, object] = {"kind": "var", "name": value.name}
        if value.upper is not None:
            payload["upper"] = value.upper
        return payload
    for cls, name in (
        (Add, "add"),
        (Sub, "sub"),
        (Mul, "mul"),
        (FloorDiv, "floordiv"),
        (CeilDiv, "ceildiv"),
        (Min, "min"),
        (Max, "max"),
    ):
        if isinstance(value, cls):
            return {
                "kind": name,
                "lhs": prim_expr_to_json_obj(value.lhs),  # type: ignore[attr-defined]
                "rhs": prim_expr_to_json_obj(value.rhs),  # type: ignore[attr-defined]
            }
    if isinstance(value, tuple):
        return [prim_expr_to_json_obj(v) for v in value]
    if isinstance(value, list):
        return [prim_expr_to_json_obj(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): prim_expr_to_json_obj(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class AttrConstraint:
    """Constraint on a normalized standard-op attribute."""

    values: tuple[object, ...]

    @staticmethod
    def eq(value: object) -> "AttrConstraint":
        return AttrConstraint((value,))

    @staticmethod
    def one_of(*values: object) -> "AttrConstraint":
        return AttrConstraint(tuple(values))

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def matches(self, value: object) -> bool:
        return value in self.values

    def to_json_obj(self) -> dict[str, object]:
        key = "eq" if len(self.values) == 1 else "one_of"
        value = self.values[0] if len(self.values) == 1 else list(self.values)
        return {key: prim_expr_to_json_obj(value)}


@dataclass(frozen=True)
class KernelLaunchSpec:
    """Runtime launch metadata, separate from kernel ABI params."""

    grid: tuple[LaunchExpr, LaunchExpr, LaunchExpr] = (1, 1, 1)
    block: tuple[int, int, int] = (256, 1, 1)
    shared_memory_bytes: int = 0
    cluster: tuple[int, int, int] = (1, 1, 1)
    cooperative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", _tuple3(self.grid, "launch.grid"))
        object.__setattr__(
            self,
            "block",
            tuple(int(v) for v in _tuple3(self.block, "launch.block")),
        )
        object.__setattr__(
            self,
            "cluster",
            tuple(int(v) for v in _tuple3(self.cluster, "launch.cluster")),
        )
        object.__setattr__(self, "shared_memory_bytes", int(self.shared_memory_bytes))

    def to_json_obj(self) -> dict[str, object]:
        return {
            "grid": [prim_expr_to_json_obj(v) for v in self.grid],
            "block": list(self.block),
            "shared_memory_bytes": self.shared_memory_bytes,
            "cluster": list(self.cluster),
            "cooperative": self.cooperative,
        }


@dataclass(frozen=True)
class KernelParamSpec:
    """One entry in the explicit kernel runtime ABI."""

    name: str
    kind: str
    dtype: str | None = None
    source: str | None = None
    index: int | None = None
    optional: bool = False
    constexpr: bool = False

    def to_json_obj(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name, "kind": self.kind}
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        if self.source is not None:
            payload["source"] = self.source
        if self.index is not None:
            payload["index"] = self.index
        if self.optional:
            payload["optional"] = True
        if self.constexpr:
            payload["constexpr"] = True
        return payload


def derive_kernel_params(inputs: tuple, outputs: tuple) -> tuple[KernelParamSpec, ...]:
    """Derive a conservative ABI for a CallDPSOp.

    Tensor values become tensor pointer params. Scalar constants and values
    with ScalarStructInfo become by-value scalar params. Unknown values keep
    the historical tensor fallback for compatibility.
    """
    params = []
    for i, value in enumerate(inputs):
        kind, dtype = _param_kind_dtype(value)
        params.append(KernelParamSpec(
            name=getattr(value, "name", f"input{i}"),
            kind=kind,
            dtype=dtype,
            source="input",
            index=i,
        ))
    for i, value in enumerate(outputs):
        kind, dtype = _param_kind_dtype(value)
        params.append(KernelParamSpec(
            name=getattr(value, "name", f"output{i}"),
            kind=kind,
            dtype=dtype,
            source="output",
            index=i,
        ))
    return tuple(params)


def _param_kind_dtype(value: object) -> tuple[str, str | None]:
    from devproc2.ir.nodes import Constant, ScalarStructInfo, TensorStructInfo

    si = getattr(value, "struct_info", None)
    if isinstance(si, TensorStructInfo):
        return "tensor", si.dtype
    if isinstance(si, ScalarStructInfo):
        return "scalar", si.dtype
    if isinstance(value, Constant):
        if isinstance(value.value, bool):
            return "scalar", "bool"
        if isinstance(value.value, int):
            return "scalar", "int64"
        if isinstance(value.value, float):
            return "scalar", "float64"
    return "tensor", None


@dataclass(frozen=True)
class KernelMatchKey:
    """Exact lookup key derived from a CallOp."""

    op_name: str
    device: str
    input_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class KernelSpec:
    """Concrete implementation of a standard op for a target/backend."""

    op_name: str
    device: str
    input_dtypes: tuple[str, ...]
    kernel_name: str
    backend: BackendName = "triton"
    output_dtype: str | None = None
    symbol: str | None = None
    sm_arches: tuple[int, ...] = ()
    priority: int = 0
    attr_constraints: Mapping[str, AttrConstraint] = field(default_factory=dict)
    layout_constraints: tuple[str, ...] = ()
    shape_constraints: tuple[object, ...] = ()
    launch: KernelLaunchSpec = field(default_factory=KernelLaunchSpec)
    params: tuple[KernelParamSpec, ...] = ()
    cubin_path: str | None = None
    ptx_path: str | None = None
    source_path: str | None = None
    include_dirs: tuple[str, ...] = ()
    extra_nvcc_flags: tuple[str, ...] = ()
    compile_options: Mapping[str, object] = field(default_factory=dict)
    match: Optional[Callable[["CallOp"], bool]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dtypes", tuple(self.input_dtypes))
        object.__setattr__(self, "sm_arches", tuple(self.sm_arches))
        object.__setattr__(self, "attr_constraints", dict(self.attr_constraints))
        object.__setattr__(self, "layout_constraints", tuple(self.layout_constraints))
        object.__setattr__(self, "shape_constraints", tuple(self.shape_constraints))
        object.__setattr__(self, "params", tuple(self.params))
        object.__setattr__(self, "include_dirs", tuple(self.include_dirs))
        object.__setattr__(self, "extra_nvcc_flags", tuple(self.extra_nvcc_flags))
        object.__setattr__(self, "compile_options", dict(self.compile_options))
        if self.symbol is None:
            object.__setattr__(self, "symbol", self.kernel_name.removeprefix("kernel."))
        if self.cubin_path is None and self.backend in {"triton", "cutedsl", "cuda"}:
            path = f"kernels/{self.kernel_name.removeprefix('kernel.')}.cubin"
            object.__setattr__(self, "cubin_path", path)

    def with_params(self, params: tuple[KernelParamSpec, ...]) -> "KernelSpec":
        return KernelSpec(
            op_name=self.op_name,
            device=self.device,
            input_dtypes=self.input_dtypes,
            kernel_name=self.kernel_name,
            backend=self.backend,
            output_dtype=self.output_dtype,
            symbol=self.symbol,
            sm_arches=self.sm_arches,
            priority=self.priority,
            attr_constraints=self.attr_constraints,
            layout_constraints=self.layout_constraints,
            shape_constraints=self.shape_constraints,
            launch=self.launch,
            params=params,
            cubin_path=self.cubin_path,
            ptx_path=self.ptx_path,
            source_path=self.source_path,
            include_dirs=self.include_dirs,
            extra_nvcc_flags=self.extra_nvcc_flags,
            compile_options=self.compile_options,
            match=self.match,
        )

    def with_launch(self, launch: KernelLaunchSpec) -> "KernelSpec":
        return KernelSpec(
            op_name=self.op_name,
            device=self.device,
            input_dtypes=self.input_dtypes,
            kernel_name=self.kernel_name,
            backend=self.backend,
            output_dtype=self.output_dtype,
            symbol=self.symbol,
            sm_arches=self.sm_arches,
            priority=self.priority,
            attr_constraints=self.attr_constraints,
            layout_constraints=self.layout_constraints,
            shape_constraints=self.shape_constraints,
            launch=launch,
            params=self.params,
            cubin_path=self.cubin_path,
            ptx_path=self.ptx_path,
            source_path=self.source_path,
            include_dirs=self.include_dirs,
            extra_nvcc_flags=self.extra_nvcc_flags,
            compile_options=self.compile_options,
            match=self.match,
        )

    def to_json_obj(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.kernel_name,
            "kind": "kernel",
            "backend": self.backend,
            "op": self.op_name,
            "symbol": self.symbol or self.kernel_name.removeprefix("kernel."),
            "launch": self.launch.to_json_obj(),
            "grid": [prim_expr_to_json_obj(v) for v in self.launch.grid],
            "block": list(self.launch.block),
            "shared_memory_bytes": self.launch.shared_memory_bytes,
            "params": [p.to_json_obj() for p in self.params],
            "constraints": {
                "sm_arches": list(self.sm_arches),
                "attrs": {
                    name: constraint.to_json_obj()
                    for name, constraint in self.attr_constraints.items()
                },
                "layouts": list(self.layout_constraints),
                "shapes": [prim_expr_to_json_obj(v) for v in self.shape_constraints],
            },
        }
        if self.output_dtype is not None:
            payload["output_dtype"] = self.output_dtype
        if self.cubin_path is not None:
            payload["cubin"] = self.cubin_path
        if self.ptx_path is not None:
            payload["ptx"] = self.ptx_path
        if self.source_path is not None:
            payload["source"] = self.source_path
        return payload


def build_input_dtypes(args: tuple) -> tuple[str, ...]:
    """Extract dtype from each arg's struct_info; '' for non-tensor args."""
    result = []
    for arg in args:
        si = getattr(arg, "struct_info", None)
        result.append(getattr(si, "dtype", "") if si is not None else "")
    return tuple(result)


_DictKey = tuple[str, str, tuple[str, ...]]


class KernelRegistry:
    """Two-level kernel registry: exact key, then structured filters."""

    def __init__(self) -> None:
        self._specs: dict[_DictKey, list[KernelSpec]] = {}

    def register(self, spec: KernelSpec) -> None:
        key: _DictKey = (spec.op_name, spec.device, spec.input_dtypes)
        bucket = self._specs.setdefault(key, [])
        bucket.append(spec)
        bucket.sort(key=lambda s: s.priority, reverse=True)

    def lookup(
        self,
        key: KernelMatchKey,
        sm_arch: Optional[int] = None,
        call_op: Optional["CallOp"] = None,
    ) -> Optional[KernelSpec]:
        dict_key: _DictKey = (key.op_name, key.device, key.input_dtypes)
        for spec in self._specs.get(dict_key, []):
            if spec.sm_arches and sm_arch is not None and sm_arch not in spec.sm_arches:
                continue
            if call_op is not None:
                if not _matches_attr_constraints(spec, call_op):
                    continue
                if not _matches_layout_constraints(spec, call_op):
                    continue
                if spec.match is not None and not spec.match(call_op):
                    continue
            return spec
        return None

    def get_by_kernel_name(self, kernel_name: str) -> Optional[KernelSpec]:
        for bucket in self._specs.values():
            for spec in bucket:
                if spec.kernel_name == kernel_name:
                    return spec
        return None


def _normalized_call_attrs(call_op: "CallOp") -> Mapping[str, object]:
    attrs = getattr(call_op, "attrs", {})
    provided = attrs.to_python_dict() if isinstance(attrs, AttrDict) else dict(attrs)
    op_def = getattr(call_op, "op_def", None)
    if op_def is not None:
        return op_def.normalize_attrs(provided, include_defaults=True).to_python_dict()
    return provided


def _matches_attr_constraints(spec: KernelSpec, call_op: "CallOp") -> bool:
    if not spec.attr_constraints:
        return True
    attrs = _normalized_call_attrs(call_op)
    return all(
        name in attrs and constraint.matches(attrs[name])
        for name, constraint in spec.attr_constraints.items()
    )


def _value_layout(value: object) -> str:
    si = getattr(value, "struct_info", None)
    for attr in ("layout", "layout_kind", "memory_layout"):
        layout = getattr(si, attr, None)
        if layout is not None:
            return str(layout)
    return "contiguous"


def _matches_layout_constraints(spec: KernelSpec, call_op: "CallOp") -> bool:
    constraints = spec.layout_constraints
    if not constraints:
        return True
    tensor_args = [
        arg
        for arg in getattr(call_op, "args", ())
        if getattr(getattr(arg, "struct_info", None), "shape", None) is not None
    ]
    if len(constraints) == 1:
        return bool(tensor_args) and all(_value_layout(arg) == constraints[0] for arg in tensor_args)
    if len(constraints) != len(tensor_args):
        return False
    return all(_value_layout(arg) == required for arg, required in zip(tensor_args, constraints))
