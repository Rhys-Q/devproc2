"""devproc2 IR Op definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar, Mapping, Optional

from devproc2.ir.attrs import AttrDict
from devproc2.ir.nodes import (
    AliasInfo,
    AliasKind,
    DialectKind,
    EffectSummary,
    Op,
    OpResult,
    Region,
    StructInfo,
    TensorStructInfo,
    TerminatorOp,
    Value,
    Var,
)
from devproc2.ir.op_ref import (
    BuiltinOpRef,
    ExternalFuncRef,
    KernelRef,
    PackedFuncRef,
    StandardOpRef,
)
from devproc2.ir.prim_expr import IntImm, PrimExpr


# ---------------------------------------------------------------------------
# Terminator Ops — must be last Op in a Block
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ReturnOp(TerminatorOp):
    """Function return."""
    dialect: ClassVar[DialectKind] = DialectKind.control
    values: tuple[Value, ...]

    @property
    def operands(self) -> tuple[Value, ...]:
        return self.values

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "ReturnOp":
        return ReturnOp(values=operands)


@dataclass(frozen=True, eq=False)
class YieldOp(TerminatorOp):
    """Region yield.  values=() means effect-only."""
    dialect: ClassVar[DialectKind] = DialectKind.control
    values: tuple[Value, ...]

    @property
    def operands(self) -> tuple[Value, ...]:
        return self.values

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "YieldOp":
        return YieldOp(values=operands)


# ---------------------------------------------------------------------------
# Compute Ops
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class CallOp(Op):
    """High-level call to a standard, builtin, or external operation.

    result_name=""  → no SSA result (effect-only call).
    result_name="y" → produces one OpResult accessible as results[0].
    result_struct_info optionally propagates type info into the OpResult.
    """
    op_ref:             StandardOpRef | BuiltinOpRef | ExternalFuncRef
    args:               tuple[Value, ...]
    result_name:        str                  = ""
    result_struct_info: Optional[StructInfo] = None
    attrs:              AttrDict | Mapping[str, object] = field(default_factory=AttrDict.empty)

    def __post_init__(self) -> None:
        attrs = self.attrs
        if not isinstance(attrs, AttrDict):
            attrs = AttrDict.from_python(attrs)
        op_def = self.op_def
        if op_def is not None:
            attrs = op_def.normalize_attrs(attrs.to_python_dict(), include_defaults=False)
            if isinstance(self.op_ref, StandardOpRef) and self.op_ref.op_def is None:
                object.__setattr__(self, "op_ref", StandardOpRef(self.op_ref.name, op_def))
        object.__setattr__(self, "attrs", attrs)
        if self.result_name:
            object.__setattr__(self, "results", (
                OpResult(op=self, index=0, struct_info=self.result_struct_info),
            ))

    @property
    def dialect(self) -> DialectKind:
        return self.op_ref.dialect

    @property
    def op_def(self):
        if isinstance(self.op_ref, StandardOpRef):
            return self.op_ref.resolve()
        if isinstance(self.op_ref, BuiltinOpRef):
            return self.op_ref.op_def
        return None

    @property
    def symbol_name(self) -> str:
        return self.op_ref.name

    @property
    def operands(self) -> tuple[Value, ...]:
        return self.args

    @property
    def effects(self) -> EffectSummary:
        if isinstance(self.op_ref, ExternalFuncRef):
            return EffectSummary.opaque_call(self.op_ref.name)
        if isinstance(self.op_ref, BuiltinOpRef) and self.op_ref.op_def is None:
            return EffectSummary.opaque_call(self.op_ref.name)
        op_def = self.op_def
        purity = getattr(getattr(op_def, "purity", None), "value", None)
        if purity == "readonly":
            return EffectSummary.readonly(*self.args)
        if purity == "impure":
            return EffectSummary.opaque_call(self.op_ref.name)
        return EffectSummary.pure()

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "CallOp":
        return make_call_op(
            op_ref=self.op_ref,
            args=operands,
            result_name=self.result_name,
            result_struct_info=self.result_struct_info,
            attrs=self.attrs,
        )


@dataclass(frozen=True, eq=False)
class StandardCallOp(CallOp):
    """Call to a registered high-level standard op."""

    op_ref: StandardOpRef

    def __post_init__(self) -> None:
        if not isinstance(self.op_ref, StandardOpRef):
            raise TypeError("StandardCallOp requires StandardOpRef")
        super().__post_init__()


@dataclass(frozen=True, eq=False)
class BuiltinCallOp(CallOp):
    """Call to a VM/runtime builtin op."""

    op_ref: BuiltinOpRef

    def __post_init__(self) -> None:
        if not isinstance(self.op_ref, BuiltinOpRef):
            raise TypeError("BuiltinCallOp requires BuiltinOpRef")
        super().__post_init__()


@dataclass(frozen=True, eq=False)
class ExternalCallOp(CallOp):
    """Call to an opaque external function.

    External calls default to an opaque effect via CallOp.effects.
    """

    op_ref: ExternalFuncRef

    def __post_init__(self) -> None:
        if not isinstance(self.op_ref, ExternalFuncRef):
            raise TypeError("ExternalCallOp requires ExternalFuncRef")
        super().__post_init__()


def make_call_op(
    op_ref: StandardOpRef | BuiltinOpRef | ExternalFuncRef,
    args: tuple[Value, ...],
    *,
    result_name: str = "",
    result_struct_info: Optional[StructInfo] = None,
    attrs: AttrDict | Mapping[str, object] | None = None,
) -> CallOp:
    kwargs = {
        "op_ref": op_ref,
        "args": args,
        "result_name": result_name,
        "result_struct_info": result_struct_info,
        "attrs": AttrDict.empty() if attrs is None else attrs,
    }
    if isinstance(op_ref, StandardOpRef):
        return StandardCallOp(**kwargs)
    if isinstance(op_ref, BuiltinOpRef):
        return BuiltinCallOp(**kwargs)
    if isinstance(op_ref, ExternalFuncRef):
        return ExternalCallOp(**kwargs)
    raise TypeError(f"unsupported op_ref for CallOp: {type(op_ref).__name__}")


@dataclass(frozen=True, eq=False)
class CallDPSOp(Op):
    """Destination-passing-style call.

    outputs=() means effect-only.  DPS ops define no SSA results.
    """
    dialect:    ClassVar[DialectKind] = DialectKind.runtime
    target_ref: KernelRef | PackedFuncRef | BuiltinOpRef
    inputs:     tuple[Value, ...]
    outputs:    tuple[Value, ...]
    effect:     EffectSummary = field(default_factory=EffectSummary.opaque_call)
    attrs:      AttrDict | Mapping[str, object] = field(default_factory=AttrDict.empty)

    def __post_init__(self) -> None:
        if not isinstance(self.attrs, AttrDict):
            object.__setattr__(self, "attrs", AttrDict.from_python(self.attrs))
        missing_writes = tuple(v for v in self.outputs if v not in self.effect.writes)
        if missing_writes:
            object.__setattr__(
                self,
                "effect",
                EffectSummary(
                    reads=self.effect.reads,
                    writes=self.effect.writes + missing_writes,
                    allocates=self.effect.allocates,
                    frees=self.effect.frees,
                    opaque=self.effect.opaque,
                    external_state=self.effect.external_state,
                    alias=self.effect.alias,
                ),
            )

    @property
    def op_ref(self) -> KernelRef | PackedFuncRef | BuiltinOpRef:
        return self.target_ref

    @property
    def operands(self) -> tuple[Value, ...]:
        return self.inputs + self.outputs

    @property
    def effects(self) -> EffectSummary:
        return self.effect

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "CallDPSOp":
        n_inputs = len(self.inputs)
        return CallDPSOp(
            target_ref=self.target_ref,
            inputs=operands[:n_inputs],
            outputs=operands[n_inputs:],
            effect=effects if effects is not None else self.effect,
            attrs=self.attrs,
        )


@dataclass(frozen=True, eq=False)
class CudaCallOp(Op):
    """Unregistered CUDA source-symbol custom call.

    ``args`` preserve the exact CUDA kernel ABI order written in Python.
    ``output_indices`` identifies which args are destinations for effect and
    alias analysis, without moving them to the end of the VM argument list.
    Lowering turns this op into ``CallDPSOp(KernelRef(KernelSpec(...)))``.
    """

    dialect: ClassVar[DialectKind] = DialectKind.runtime
    source_path: str
    symbol: str
    args: tuple[Value, ...]
    output_indices: tuple[int, ...] = ()
    launch: object | None = None
    attrs: AttrDict | Mapping[str, object] = field(default_factory=AttrDict.empty)
    sm_arches: tuple[int, ...] = ()
    include_dirs: tuple[str, ...] = ()
    extra_nvcc_flags: tuple[str, ...] = ()
    compile_options: Mapping[str, object] = field(default_factory=dict)
    params: tuple[object, ...] = ()
    input_dtypes: tuple[str, ...] = ()
    output_dtype: str | None = None
    kernel_name: str | None = None
    effect: EffectSummary | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attrs, AttrDict):
            object.__setattr__(self, "attrs", AttrDict.from_python(self.attrs))
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "output_indices", tuple(int(i) for i in self.output_indices))
        object.__setattr__(self, "sm_arches", tuple(int(v) for v in self.sm_arches))
        object.__setattr__(self, "include_dirs", tuple(self.include_dirs))
        object.__setattr__(self, "extra_nvcc_flags", tuple(self.extra_nvcc_flags))
        object.__setattr__(self, "compile_options", dict(self.compile_options))
        object.__setattr__(self, "params", tuple(self.params))
        object.__setattr__(self, "input_dtypes", tuple(str(v) for v in self.input_dtypes))
        n_args = len(self.args)
        bad = [i for i in self.output_indices if i < 0 or i >= n_args]
        if bad:
            raise ValueError(f"CudaCallOp output_indices out of range: {bad}")
        if self.effect is None:
            object.__setattr__(
                self,
                "effect",
                EffectSummary(
                    writes=tuple(self.args[i] for i in self.output_indices),
                    opaque=True,
                    external_state=self.symbol,
                ),
            )

    @property
    def operands(self) -> tuple[Value, ...]:
        return self.args

    @property
    def effects(self) -> EffectSummary:
        assert self.effect is not None
        return self.effect

    @property
    def outputs(self) -> tuple[Value, ...]:
        return tuple(self.args[i] for i in self.output_indices)

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "CudaCallOp":
        return CudaCallOp(
            source_path=self.source_path,
            symbol=self.symbol,
            args=operands,
            output_indices=self.output_indices,
            launch=self.launch,
            attrs=self.attrs,
            sm_arches=self.sm_arches,
            include_dirs=self.include_dirs,
            extra_nvcc_flags=self.extra_nvcc_flags,
            compile_options=self.compile_options,
            params=self.params,
            input_dtypes=self.input_dtypes,
            output_dtype=self.output_dtype,
            kernel_name=self.kernel_name,
            effect=effects if effects is not None else self.effect,
        )


class TensorCreateKind(Enum):
    empty      = auto()
    zeros      = auto()
    full       = auto()
    empty_like = auto()


@dataclass(frozen=True, eq=False)
class TensorCreateOp(Op):
    """Allocate / create a tensor buffer."""
    dialect:     ClassVar[DialectKind] = DialectKind.memory
    result_name: str
    kind:        TensorCreateKind
    shape:       tuple[PrimExpr, ...]
    dtype:       str
    device:      str
    fill_value:  Optional[object] = None
    like:        Optional[Value]  = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )
        if self.kind == TensorCreateKind.empty_like:
            if self.like is None:
                raise ValueError("TensorCreateOp(empty_like) requires 'like'")
            if self.shape:
                raise ValueError("TensorCreateOp(empty_like) must not specify 'shape'")
        else:
            if self.like is not None:
                raise ValueError(f"TensorCreateOp({self.kind.name}) must not specify 'like'")
        if self.kind == TensorCreateKind.empty_like:
            si = getattr(self.like, "struct_info", None)
        else:
            si = TensorStructInfo(self.shape, self.dtype, self.device)
        object.__setattr__(self, "results", (OpResult(op=self, index=0, struct_info=si),))

    @property
    def operands(self) -> tuple[Value, ...]:
        return () if self.like is None else (self.like,)

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "TensorCreateOp":
        like = operands[0] if operands else None
        return TensorCreateOp(
            result_name=self.result_name,
            kind=self.kind,
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            fill_value=self.fill_value,
            like=like,
        )


@dataclass(frozen=True, eq=False)
class TensorViewOp(Op):
    """Create a tensor view over an existing tensor without allocating storage.

    ``byte_offset`` is a scalar byte index.  ``byte_stride`` and
    ``base_offset`` let loops pass an index value while codegen materializes
    ``base_offset + byte_offset * byte_stride`` before calling the VM builtin.
    """
    dialect:     ClassVar[DialectKind] = DialectKind.memory
    result_name: str
    base:        Value
    byte_offset: Value
    shape:       tuple[PrimExpr, ...]
    dtype:       Optional[str] = None
    device:      Optional[str] = None
    byte_stride: int = 1
    base_offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )
        dtype = self.dtype
        device = self.device
        base_si = getattr(self.base, "struct_info", None)
        if isinstance(base_si, TensorStructInfo):
            dtype = dtype or base_si.dtype
            device = device or base_si.device
        si = (
            TensorStructInfo(self.shape, dtype, device)
            if dtype is not None and device is not None
            else None
        )
        object.__setattr__(self, "results", (OpResult(op=self, index=0, struct_info=si),))

    @property
    def operands(self) -> tuple[Value, ...]:
        return (self.base, self.byte_offset)

    @property
    def effects(self) -> EffectSummary:
        return EffectSummary(
            reads=(self.base,),
            alias=AliasInfo(AliasKind.view_of, source=self.base),
        )

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "TensorViewOp":
        return TensorViewOp(
            result_name=self.result_name,
            base=operands[0],
            byte_offset=operands[1],
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            byte_stride=self.byte_stride,
            base_offset=self.base_offset,
        )


@dataclass(frozen=True, eq=False)
class TupleOp(Op):
    """Construct a tuple value from its elements."""
    dialect:     ClassVar[DialectKind] = DialectKind.tensor
    result_name: str
    elems:       tuple[Value, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))

    @property
    def operands(self) -> tuple[Value, ...]:
        return self.elems

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "TupleOp":
        return TupleOp(result_name=self.result_name, elems=operands)


@dataclass(frozen=True, eq=False)
class TupleGetItemOp(Op):
    """Extract element at `index` from a tuple."""
    dialect:     ClassVar[DialectKind] = DialectKind.tensor
    tup:         Value
    index:       int
    result_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))

    @property
    def operands(self) -> tuple[Value, ...]:
        return (self.tup,)

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "TupleGetItemOp":
        return TupleGetItemOp(
            tup=operands[0],
            index=self.index,
            result_name=self.result_name,
        )


# ---------------------------------------------------------------------------
# Control-flow Ops
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Range:
    """Loop bounds used by ForOp."""
    start: Value
    end:   Value
    step:  Value


@dataclass(frozen=True)
class IterArg:
    """One loop-carried variable for ForOp."""
    var:  Var    # block arg inside the loop body
    init: Value  # initial value from outer scope


@dataclass(frozen=True, eq=False)
class IfOp(Op):
    """Structured conditional.

    result_names=()          → effect-only: both branches yield no values.
    result_names=("y", ...)  → SSA results: branches yield matching values.
    """
    cond:         Value
    then_region:  Region
    else_region:  Optional[Region] = None
    result_names: tuple[str, ...]  = ()
    dialect: ClassVar[DialectKind] = DialectKind.control

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(
            OpResult(op=self, index=i) for i in range(len(self.result_names))
        ))

    @property
    def operands(self) -> tuple[Value, ...]:
        return (self.cond,)

    @property
    def regions(self) -> tuple[Region, ...]:
        if self.else_region is None:
            return (self.then_region,)
        return (self.then_region, self.else_region)

    @property
    def effects(self) -> EffectSummary:
        return _regions_effects(self.regions)

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "IfOp":
        new_regions = self.regions if regions is None else regions
        return IfOp(
            cond=operands[0],
            then_region=new_regions[0],
            else_region=new_regions[1] if len(new_regions) > 1 else None,
            result_names=self.result_names,
        )


@dataclass(frozen=True, eq=False)
class ForOp(Op):
    """Structured loop over a Range.

    result_names=()          → effect-only loop; body yields nothing.
    result_names=("out", ...) → loop-carried; body yields updated values.
    """
    loop_var:     Var
    range_:       Range
    iter_args:    tuple[IterArg, ...]
    body_region:  Region
    result_names: tuple[str, ...] = ()
    dialect: ClassVar[DialectKind] = DialectKind.control

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(
            OpResult(op=self, index=i) for i in range(len(self.result_names))
        ))

    @property
    def operands(self) -> tuple[Value, ...]:
        return (
            self.range_.start,
            self.range_.end,
            self.range_.step,
            *(ia.init for ia in self.iter_args),
        )

    @property
    def regions(self) -> tuple[Region, ...]:
        return (self.body_region,)

    @property
    def effects(self) -> EffectSummary:
        return _regions_effects(self.regions)

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "ForOp":
        n_range = 3
        new_range = Range(
            start=operands[0],
            end=operands[1],
            step=operands[2],
        )
        new_iter_args = tuple(
            IterArg(var=ia.var, init=operands[n_range + i])
            for i, ia in enumerate(self.iter_args)
        )
        new_regions = self.regions if regions is None else regions
        return ForOp(
            loop_var=self.loop_var,
            range_=new_range,
            iter_args=new_iter_args,
            body_region=new_regions[0],
            result_names=self.result_names,
        )


# ---------------------------------------------------------------------------
# Shape assertion Op
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AllocStorageOp(Op):
    """Allocate a raw storage buffer.  Hoisted to function entry by LowerTensorCreateToAllocPass.

    size_bytes is a PrimExpr so it supports both static shapes (IntImm) and
    dynamic shapes (symbolic expressions evaluated at runtime).
    """
    dialect:     ClassVar[DialectKind] = DialectKind.memory
    result_name: str
    size_bytes:  PrimExpr   # IntImm for static; symbolic expr for dynamic
    alignment:   int
    device:      str

    def __post_init__(self) -> None:
        if isinstance(self.size_bytes, int):
            object.__setattr__(self, "size_bytes", IntImm(self.size_bytes))
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))

    @property
    def effects(self) -> EffectSummary:
        return EffectSummary(allocates=True)


@dataclass(frozen=True, eq=False)
class AllocTensorOp(Op):
    """Create a tensor view over a storage buffer."""
    dialect:     ClassVar[DialectKind] = DialectKind.memory
    result_name: str
    storage:     Value             # OpResult from AllocStorageOp
    offset:      int               # byte offset; always 0 in MVP
    shape:       tuple[PrimExpr, ...]
    dtype:       str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape",
            tuple(IntImm(s) if isinstance(s, int) else s for s in self.shape),
        )
        object.__setattr__(self, "results", (OpResult(op=self, index=0),))

    @property
    def operands(self) -> tuple[Value, ...]:
        return (self.storage,)

    @property
    def effects(self) -> EffectSummary:
        return EffectSummary(alias=AliasInfo(AliasKind.view_of, source=self.storage))

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "AllocTensorOp":
        return AllocTensorOp(
            result_name=self.result_name,
            storage=operands[0],
            offset=self.offset,
            shape=self.shape,
            dtype=self.dtype,
        )


# ---------------------------------------------------------------------------
# Shape assertion Op
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ShapeAssertOp(Op):
    """Runtime assertion: tensor.shape[dim_idx] <= upper."""
    dialect: ClassVar[DialectKind] = DialectKind.shape
    tensor:  Var
    dim_idx: int
    upper:   int

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, Var):
            raise TypeError("ShapeAssertOp.tensor must be a BlockArg/Var")

    @property
    def operands(self) -> tuple[Value, ...]:
        return (self.tensor,)

    @property
    def effects(self) -> EffectSummary:
        return EffectSummary.readonly(self.tensor)

    def replace_operands(
        self,
        operands: tuple[Value, ...],
        *,
        regions: tuple[Region, ...] | None = None,
        effects: EffectSummary | None = None,
    ) -> "ShapeAssertOp":
        return ShapeAssertOp(
            tensor=operands[0],
            dim_idx=self.dim_idx,
            upper=self.upper,
        )


def _regions_effects(regions: tuple[Region, ...]) -> EffectSummary:
    reads: list[Value] = []
    writes: list[Value] = []
    allocates = False
    frees = False
    opaque = False
    external_state: str | None = None

    for region in regions:
        for block in region.blocks:
            for op in block.ops:
                effect = op.effects
                reads.extend(effect.reads)
                writes.extend(effect.writes)
                allocates = allocates or effect.allocates
                frees = frees or effect.frees
                opaque = opaque or effect.opaque
                external_state = external_state or effect.external_state

    return EffectSummary(
        reads=tuple(reads),
        writes=tuple(writes),
        allocates=allocates,
        frees=frees,
        opaque=opaque,
        external_state=external_state,
    )
