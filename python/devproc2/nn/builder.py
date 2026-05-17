"""Tracing graph builder for nn.Module frontends."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Optional

from devproc2.compiler.op import (
    StandardOp,
    get_op,
    set_current_emitter,
)
from devproc2.ir.op_ref import ExternalFuncRef, KernelRef, PackedFuncRef, StandardOpRef
from devproc2.ir.nodes import (
    Block,
    Constant,
    EffectSummary,
    Function,
    IRModule,
    Op,
    Region,
    StructInfo,
    TensorStructInfo,
    Value,
    Var,
)
from devproc2.ir.ops import (
    CallDPSOp,
    CallOp,
    ReturnOp,
    TensorCreateKind,
    TensorCreateOp,
    TensorViewOp,
    TupleOp,
    make_call_op,
)

from devproc2.nn.module import Module
from devproc2.nn.specs import ObjectSpec, Parameter, ScalarSpec, TensorSpec


class GraphBuilder:
    def __init__(self) -> None:
        self._ops: list[Op] = []
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

        ret = self._materialize_return_value(result)
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

    def emit_dps_kernel(
        self,
        name: str,
        *,
        inputs: object | None = None,
        launch: object | None = None,
        output_like: object | None = None,
        output_shape: object | None = None,
        output_dtype: str | None = None,
        output_device: str | None = None,
        output_spec: object | None = None,
        output_specs: object | None = None,
        effect: str = "opaque",
    ) -> TraceValue | tuple[TraceValue, ...]:
        ir_inputs = tuple(_unwrap_trace_value(arg) for arg in _as_sequence(inputs))
        if output_specs is not None:
            if any(v is not None for v in (output_like, output_shape, output_dtype, output_device, output_spec)):
                raise ValueError("output_specs cannot be combined with single-output overrides")
            sis = tuple(
                _resolve_output_struct_info(
                    ir_inputs,
                    output_like=None,
                    output_shape=None,
                    output_dtype=None,
                    output_device=None,
                    output_spec=spec,
                )
                for spec in _as_sequence(output_specs)
            )
            if not sis:
                raise ValueError("output_specs must contain at least one output spec")
        else:
            has_output = (
                output_like is not None
                or output_shape is not None
                or output_dtype is not None
                or output_device is not None
                or output_spec is not None
            )
            kernel_name = name if name.startswith("kernel.") else f"kernel.{name}"
            from devproc2.frontend.dsl import get_kernel_registry

            spec = get_kernel_registry().get_by_kernel_name(kernel_name)
            if spec is not None and launch is not None:
                spec = spec.with_launch(launch)
            if not has_output:
                self._ops.append(
                    CallDPSOp(
                        target_ref=KernelRef(kernel_name, spec),
                        inputs=ir_inputs,
                        outputs=(),
                        effect=EffectSummary.opaque_call(kernel_name),
                    )
                )
                return None
            sis = (
                _resolve_output_struct_info(
                    ir_inputs,
                    output_like=output_like,
                    output_shape=output_shape,
                    output_dtype=output_dtype,
                    output_device=output_device,
                    output_spec=output_spec,
                ),
            )

        kernel_name = name if name.startswith("kernel.") else f"kernel.{name}"
        from devproc2.frontend.dsl import get_kernel_registry

        spec = get_kernel_registry().get_by_kernel_name(kernel_name)
        if spec is not None and launch is not None:
            spec = spec.with_launch(launch)
        creates = tuple(
            TensorCreateOp(
                result_name=self._fresh(name.replace(".", "_")),
                kind=TensorCreateKind.empty,
                shape=si.shape,
                dtype=si.dtype,
                device=si.device,
            )
            for si in sis
        )
        dps = CallDPSOp(
            target_ref=KernelRef(kernel_name, spec),
            inputs=ir_inputs,
            outputs=tuple(create.results[0] for create in creates),
            effect=_dps_effect(effect, tuple(create.results[0] for create in creates), kernel_name),
        )
        self._ops.extend(creates)
        self._ops.append(dps)
        values = tuple(TraceValue(create.results[0], self) for create in creates)
        return values[0] if len(values) == 1 else values

    def emit_dps_packed(
        self,
        name: str,
        *,
        inputs: object | None = None,
        output_like: object | None = None,
        output_shape: object | None = None,
        output_dtype: str | None = None,
        output_device: str | None = None,
        output_spec: object | None = None,
        effect: str = "opaque",
    ):
        ir_inputs = tuple(_unwrap_trace_value(arg) for arg in _as_sequence(inputs))
        has_output = (
            output_like is not None
            or output_shape is not None
            or output_dtype is not None
            or output_device is not None
            or output_spec is not None
        )
        if not has_output:
            dps = CallDPSOp(
                target_ref=PackedFuncRef(name),
                inputs=ir_inputs,
                outputs=(),
                effect=EffectSummary.opaque_call(name),
            )
            self._ops.append(dps)
            return None

        si = _resolve_output_struct_info(
            ir_inputs,
            output_like=output_like,
            output_shape=output_shape,
            output_dtype=output_dtype,
            output_device=output_device,
            output_spec=output_spec,
        )
        create = TensorCreateOp(
            result_name=self._fresh(name.replace(".", "_")),
            kind=TensorCreateKind.empty,
            shape=si.shape,
            dtype=si.dtype,
            device=si.device,
        )
        dps = CallDPSOp(
            target_ref=PackedFuncRef(name),
            inputs=ir_inputs,
            outputs=(create.results[0],),
            effect=_dps_effect(effect, (create.results[0],), name),
        )
        self._ops.append(create)
        self._ops.append(dps)
        return TraceValue(create.results[0], self)

    def emit_empty(
        self,
        shape: object,
        *,
        dtype: str = "float32",
        device: str = "cpu",
    ) -> TraceValue:
        if not isinstance(shape, tuple):
            shape = tuple(shape) if isinstance(shape, list) else (shape,)
        create = TensorCreateOp(
            result_name=self._fresh("empty"),
            kind=TensorCreateKind.empty,
            shape=tuple(shape),
            dtype=dtype,
            device=device,
        )
        self._ops.append(create)
        return TraceValue(create.results[0], self)

    def emit_tensor_view(
        self,
        base: object,
        byte_offset: object,
        shape: object,
        *,
        dtype: str | None = None,
        device: str | None = None,
        byte_stride: int = 1,
        base_offset: int = 0,
    ) -> TraceValue:
        ir_base = _unwrap_trace_value(base)
        ir_offset = _unwrap_trace_value(byte_offset)
        if not isinstance(shape, tuple):
            shape = tuple(shape) if isinstance(shape, list) else (shape,)
        base_si = _tensor_struct_info(ir_base)
        view = TensorViewOp(
            result_name=self._fresh("view"),
            base=ir_base,
            byte_offset=ir_offset,
            shape=tuple(shape),
            dtype=dtype or (base_si.dtype if base_si is not None else None),
            device=device or (base_si.device if base_si is not None else None),
            byte_stride=byte_stride,
            base_offset=base_offset,
        )
        self._ops.append(view)
        return TraceValue(view.results[0], self)

    def _materialize_return_value(self, result: object) -> Value:
        if isinstance(result, (tuple, list)):
            elems = tuple(self._materialize_return_value(item) for item in result)
            tuple_op = TupleOp(result_name=self._fresh("tuple"), elems=elems)
            self._ops.append(tuple_op)
            return tuple_op.results[0]
        return _unwrap_trace_value(result)

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


def _as_sequence(value: object | None) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _resolve_output_struct_info(
    ir_inputs: tuple[Value, ...],
    *,
    output_like: object | None,
    output_shape: object | None,
    output_dtype: str | None,
    output_device: str | None,
    output_spec: object | None,
) -> TensorStructInfo:
    if output_spec is not None:
        if isinstance(output_spec, TensorStructInfo):
            return output_spec
        spec_si = getattr(output_spec, "struct_info", None)
        if isinstance(spec_si, TensorStructInfo):
            return spec_si
        raise TypeError("output_spec must be TensorSpec or TensorStructInfo")

    like_value = _unwrap_trace_value(output_like) if output_like is not None else None
    if like_value is None:
        if not ir_inputs:
            raise ValueError("DPS calls with outputs require inputs, output_like, or output_spec")
        like_value = ir_inputs[0]
    like_si = _tensor_struct_info(like_value)
    if like_si is None:
        raise TypeError("DPS output_like must have TensorStructInfo")

    shape = output_shape if output_shape is not None else like_si.shape
    if not isinstance(shape, tuple):
        shape = tuple(shape) if isinstance(shape, list) else (shape,)
    return TensorStructInfo(
        tuple(shape),
        output_dtype or like_si.dtype,
        output_device or like_si.device,
    )


def _dps_effect(effect: str, outputs: tuple[Value, ...], target: str) -> EffectSummary:
    if effect == "pure":
        return EffectSummary.write(*outputs)
    return EffectSummary(writes=outputs, opaque=True, external_state=target)


_CURRENT_BUILDER: Optional[GraphBuilder] = None


__all__ = [
    "GraphBuilder",
    "TraceValue",
    "unwrap_trace_value",
]
