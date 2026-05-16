"""Tracing graph builder for nn.Module frontends."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Optional

from devproc2.compiler.op import (
    StandardOp,
    get_op,
    set_current_emitter,
)
from devproc2.ir.op_ref import ExternalFuncRef, StandardOpRef
from devproc2.ir.nodes import (
    Block,
    Constant,
    Function,
    IRModule,
    Region,
    StructInfo,
    TensorStructInfo,
    Value,
    Var,
)
from devproc2.ir.ops import CallOp, ReturnOp, make_call_op

from devproc2.nn.module import Module
from devproc2.nn.specs import ObjectSpec, Parameter, ScalarSpec, TensorSpec


class GraphBuilder:
    def __init__(self) -> None:
        self._ops: list[CallOp] = []
        self._counter = 0
        self._param_values: dict[int, TraceValue] = {}

    def build(
        self,
        model_method,
        input_specs: Mapping[str, object] | Sequence[object],
    ) -> IRModule:
        self._ops = []
        self._counter = 0
        self._param_values = {}

        root = getattr(model_method, "__self__", None)
        if isinstance(root, Module):
            root._assign_parameter_names()
            root._set_tracing_recursive(True)

        try:
            params = self._make_input_vars(model_method, input_specs)
            args = [TraceValue(v, self) for v in params]
            with _TraceContext(self):
                result = model_method(*args)
        finally:
            if isinstance(root, Module):
                root._set_tracing_recursive(False)

        ret = _unwrap_trace_value(result)
        block = Block(
            args=tuple(params) + tuple(self._parameter_vars()),
            ops=tuple(self._ops) + (ReturnOp((ret,)),),
        )
        fn_name = getattr(model_method, "__name__", "main")
        return IRModule({fn_name: Function(Region((block,)))})

    def emit_op(
        self,
        op: StandardOp | str,
        args: tuple[object, ...],
        attrs: dict[str, object],
    ) -> TraceValue:
        ir_args = tuple(_unwrap_trace_value(arg) for arg in args)
        if isinstance(op, StandardOp):
            op_name = op.name
            attrs_dict = op.normalize_attrs(attrs)
            result_si = op.infer_struct_info(
                tuple(_struct_info_for_value(arg) for arg in ir_args),
                attrs_dict,
            )
            op_ref = StandardOpRef(op_name, op)
        else:
            op_name = op
            op_def = get_op(op_name)
            if op_def is not None:
                attrs_dict = op_def.normalize_attrs(attrs)
                result_si = op_def.infer_struct_info(
                    tuple(_struct_info_for_value(arg) for arg in ir_args),
                    attrs_dict,
                )
                op_ref = StandardOpRef(op_name, op_def)
            else:
                attrs_dict = attrs or {}
                result_si = None
                op_ref = ExternalFuncRef(op_name)
        call = make_call_op(
            op_ref=op_ref,
            args=ir_args,
            result_name=self._fresh(op_name),
            result_struct_info=result_si,
            attrs=attrs_dict,
        )
        self._ops.append(call)
        return TraceValue(call.results[0], self)

    def parameter_value(self, parameter: Parameter) -> TraceValue:
        if parameter.name is None:
            raise RuntimeError("Parameter is not attached to a Module path")
        key = id(parameter)
        value = self._param_values.get(key)
        if value is None:
            var = Var(parameter.name, parameter.struct_info)
            value = TraceValue(var, self)
            self._param_values[key] = value
        return value

    def _parameter_vars(self) -> Iterator[Var]:
        for value in self._param_values.values():
            if isinstance(value.value, Var):
                yield value.value

    def _fresh(self, base: str) -> str:
        name = f"{base}_{self._counter}"
        self._counter += 1
        return name

    def _make_input_vars(
        self,
        model_method,
        input_specs: Mapping[str, object] | Sequence[object],
    ) -> list[Var]:
        if isinstance(input_specs, Mapping):
            return [
                Var(name, _spec_struct_info(spec))
                for name, spec in input_specs.items()
            ]
        return [
            Var(f"arg{i}", _spec_struct_info(spec))
            for i, spec in enumerate(input_specs)
        ]


class TraceValue:
    def __init__(self, value: Value, builder: GraphBuilder) -> None:
        self.value = value
        self._builder = builder


class _TraceContext:
    def __init__(self, builder: GraphBuilder) -> None:
        self._builder = builder
        self._prev: Optional[GraphBuilder] = None
        self._prev_emitter = None

    def __enter__(self):
        global _CURRENT_BUILDER
        self._prev = _CURRENT_BUILDER
        self._prev_emitter = set_current_emitter(self._builder)
        _CURRENT_BUILDER = self._builder
        return self

    def __exit__(self, exc_type, exc, tb):
        global _CURRENT_BUILDER
        _CURRENT_BUILDER = self._prev
        set_current_emitter(self._prev_emitter)


def unwrap_trace_value(value: object) -> Value:
    return _unwrap_trace_value(value)


def _unwrap_trace_value(value: object) -> Value:
    if isinstance(value, TraceValue):
        return value.value
    if isinstance(value, Parameter):
        if _CURRENT_BUILDER is None:
            raise RuntimeError("Parameter values are only available inside GraphBuilder.build")
        return _CURRENT_BUILDER.parameter_value(value).value
    if isinstance(value, Value):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return Constant(value)
    raise TypeError(f"expected trace value, got {type(value).__name__}")


def _struct_info_for_value(value: Value) -> Optional[StructInfo]:
    return getattr(value, "struct_info", None)


def _spec_struct_info(spec: object) -> Optional[StructInfo]:
    if isinstance(spec, StructInfo):
        return spec
    if isinstance(spec, (TensorSpec, ScalarSpec, ObjectSpec)):
        return spec.struct_info
    if spec is None:
        return None
    raise TypeError(
        "expected TensorSpec, ScalarSpec, ObjectSpec, or StructInfo, "
        f"got {type(spec).__name__}"
    )


def _tensor_struct_info(value: Value) -> Optional[TensorStructInfo]:
    si = getattr(value, "struct_info", None)
    return si if isinstance(si, TensorStructInfo) else None


_CURRENT_BUILDER: Optional[GraphBuilder] = None


__all__ = [
    "GraphBuilder",
    "TraceValue",
    "unwrap_trace_value",
]
