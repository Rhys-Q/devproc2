"""VMCodegenPass: memory-explicit IR → Executable VM bytecode."""
from __future__ import annotations

from typing import Any, Optional

from devproc2.ir.nodes import (
    Block,
    Function,
    IRModule,
    Op,
    Value,
    Var,
    Constant,
    OpResult,
)
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    CallDPSOp,
    CalleeKind as IRCalleeKind,
    ForOp,
    IfOp,
    IterArg,
    ReturnOp,
    ShapeAssertOp,
    TupleGetItemOp,
    TupleOp,
    YieldOp,
)
from devproc2.ir.prim_expr import IntImm, PrimExpr, PrimVar
from devproc2.vm.executable import (
    CalleeKind,
    ConstInit,
    Executable,
    FunctionEntry,
    Instruction,
    Opcode,
)


# ---------------------------------------------------------------------------
# Device / dtype encoding helpers
# ---------------------------------------------------------------------------

_DEVICE_TYPE_MAP: dict[str, int] = {
    "cpu":    1,  # kDLCPU
    "cuda":   2,  # kDLCUDA
    "metal":  8,  # kDLMetal
    "vulkan": 7,  # kDLVulkan
    "rocm":   10, # kDLROCM
}

_DTYPE_MAP: dict[str, tuple[int, int, int]] = {
    # (code, bits, lanes)
    "bool":     (6,  8,  1),   # kDLBool
    "int8":     (0,  8,  1),   # kDLInt
    "int16":    (0,  16, 1),
    "int32":    (0,  32, 1),
    "int64":    (0,  64, 1),
    "uint8":    (1,  8,  1),   # kDLUInt
    "uint16":   (1,  16, 1),
    "uint32":   (1,  32, 1),
    "uint64":   (1,  64, 1),
    "float16":  (2,  16, 1),   # kDLFloat
    "float32":  (2,  32, 1),
    "float64":  (2,  64, 1),
    "bfloat16": (4,  16, 1),   # kDLBfloat
}


def _parse_device(device_str: str) -> tuple[int, int]:
    """Parse "cpu", "cuda", "cuda:0" → (device_type_int, device_id_int)."""
    parts = device_str.split(":")
    dev_name = parts[0].lower()
    dev_type = _DEVICE_TYPE_MAP.get(dev_name)
    if dev_type is None:
        raise ValueError(f"Unknown device type: {dev_name!r}")
    dev_id = int(parts[1]) if len(parts) > 1 else 0
    return dev_type, dev_id


def _parse_dtype(dtype_str: str) -> tuple[int, int, int]:
    """Parse "float16" → (code, bits, lanes)."""
    result = _DTYPE_MAP.get(dtype_str.lower())
    if result is None:
        raise ValueError(f"Unknown dtype: {dtype_str!r}")
    return result


def _ir_callee_kind(kind: IRCalleeKind) -> CalleeKind:
    return {
        IRCalleeKind.vm_func:     CalleeKind.vm_func,
        IRCalleeKind.builtin:     CalleeKind.builtin,
        IRCalleeKind.packed_func: CalleeKind.packed_func,
        IRCalleeKind.kernel:      CalleeKind.kernel,
    }[kind]


# ---------------------------------------------------------------------------
# Per-function codegen context
# ---------------------------------------------------------------------------

class _FnCtx:
    """Mutable state for codegen of a single function."""

    def __init__(self, exec_: Executable) -> None:
        self._exec = exec_
        # id(Value) → register index
        self._value_reg: dict[int, int] = {}
        # Pending instructions for this function
        self.instrs: list[Instruction] = []
        # Next available register index
        self.next_reg: int = 0
        # const_inits: filled into FunctionEntry
        self.const_inits: list[ConstInit] = []

    # ---- register allocation -----------------------------------------------

    def alloc_reg(self) -> int:
        r = self.next_reg
        self.next_reg += 1
        return r

    def bind(self, value: Value, reg: int) -> None:
        self._value_reg[id(value)] = reg

    def reg_of(self, value: Value) -> int:
        """Return register index for an IR Value (Var, OpResult, or Constant)."""
        vid = id(value)
        if vid in self._value_reg:
            return self._value_reg[vid]
        if isinstance(value, Constant):
            # Inline scalar constants via const_init
            reg = self._reg_for_const(value.value)
            # Don't cache: each Constant may be a different Python object with same val
            return reg
        raise KeyError(f"Unbound value {value!r}")

    def _reg_for_const(self, val: Any) -> int:
        """Allocate a new register pre-loaded with val."""
        const_idx = self._intern_const(val)
        reg = self.alloc_reg()
        self.const_inits.append(ConstInit(reg, const_idx))
        return reg

    def reg_for_int(self, val: int) -> int:
        """Shortcut: allocate a register pre-loaded with an integer constant."""
        return self._reg_for_const(int(val))

    # ---- constant pool -----------------------------------------------------

    def _intern_const(self, val: Any) -> int:
        """Add val to Executable.constants (dedup by type+value), return index."""
        consts = self._exec.constants
        for i, c in enumerate(consts):
            if c == val and type(c) is type(val):
                return i
        consts.append(val)
        return len(consts) - 1

    # ---- instruction emission ----------------------------------------------

    def emit(self, instr: Instruction) -> int:
        """Append instruction; return its index within this function's instrs."""
        idx = len(self.instrs)
        self.instrs.append(instr)
        return idx

    def emit_placeholder(self, opcode: Opcode) -> int:
        """Emit a placeholder instruction for later backpatching."""
        return self.emit(Instruction(opcode))

    def pc(self) -> int:
        """Current instruction count (= index of the *next* instruction)."""
        return len(self.instrs)

    # ---- function table management -----------------------------------------

    def intern_func(self, name: str, kind: CalleeKind) -> int:
        """Get or create a FunctionEntry for an external callee. Return func_idx."""
        for i, fe in enumerate(self._exec.function_table):
            if fe.name == name:
                return i
        fe = FunctionEntry(
            name=name, kind=kind,
            instr_offset=-1, instr_count=0,
            num_regs=0, num_args=0,
        )
        self._exec.function_table.append(fe)
        return len(self._exec.function_table) - 1

    def builtin(self, name: str) -> int:
        return self.intern_func(name, CalleeKind.builtin)

    def identity_builtin(self) -> int:
        return self.builtin("vm.builtin.identity")


# ---------------------------------------------------------------------------
# VMCodegenPass
# ---------------------------------------------------------------------------

class VMCodegenPass:
    """Lower a memory-explicit IRModule to an Executable.

    The input module must have been processed by:
      InferStructInfoPass → DPSLoweringPass → MemoryPlanningPass
      → LowerTensorCreateToAllocPass

    All AllocStorageOp.size_bytes must be IntImm (static shapes only in M8).
    """

    def run(self, module: IRModule) -> Executable:
        exec_ = Executable()
        for fn_name, fn in module.functions.items():
            self._codegen_fn(fn_name, fn, exec_)
        return exec_

    # ---- function-level codegen --------------------------------------------

    def _codegen_fn(self, name: str, fn: Function, exec_: Executable) -> None:
        ctx = _FnCtx(exec_)

        # Assign registers 0..n-1 to function parameters
        for i, param in enumerate(fn.params):
            ctx.bind(param, i)
            ctx.next_reg = i + 1

        instr_base = len(exec_.instructions)
        self._codegen_block(fn.body.entry_block, ctx)

        fe = FunctionEntry(
            name=name,
            kind=CalleeKind.vm_func,
            instr_offset=instr_base,
            instr_count=len(ctx.instrs),
            num_regs=ctx.next_reg,
            num_args=len(fn.params),
            const_inits=ctx.const_inits,
        )
        exec_.function_table.append(fe)
        exec_.instructions.extend(ctx.instrs)

    # ---- block-level codegen -----------------------------------------------

    def _codegen_block(self, block: Block, ctx: _FnCtx) -> None:
        for op in block.ops:
            self._codegen_op(op, ctx)

    # ---- op dispatch -------------------------------------------------------

    def _codegen_op(self, op: Op, ctx: _FnCtx) -> None:
        if isinstance(op, AllocStorageOp):
            self._lower_alloc_storage(op, ctx)
        elif isinstance(op, AllocTensorOp):
            self._lower_alloc_tensor(op, ctx)
        elif isinstance(op, CallDPSOp):
            self._lower_calldps(op, ctx)
        elif isinstance(op, ShapeAssertOp):
            self._lower_shape_assert(op, ctx)
        elif isinstance(op, TupleOp):
            self._lower_tuple(op, ctx)
        elif isinstance(op, TupleGetItemOp):
            self._lower_tuple_get_item(op, ctx)
        elif isinstance(op, IfOp):
            self._lower_if(op, ctx)
        elif isinstance(op, ForOp):
            self._lower_for(op, ctx)
        elif isinstance(op, ReturnOp):
            self._lower_return(op, ctx)
        elif isinstance(op, YieldOp):
            pass  # handled by parent IfOp/ForOp
        else:
            raise NotImplementedError(f"VMCodegenPass: unsupported op {type(op).__name__}")

    # ---- AllocStorageOp ----------------------------------------------------

    def _lower_alloc_storage(self, op: AllocStorageOp, ctx: _FnCtx) -> None:
        if not isinstance(op.size_bytes, IntImm):
            raise NotImplementedError(
                f"Dynamic shape in AllocStorageOp '{op.result_name}' is not supported "
                "in M8 (requires X2)."
            )
        size_reg   = ctx.reg_for_int(op.size_bytes.value)
        align_reg  = ctx.reg_for_int(op.alignment)
        dev_type, dev_id = _parse_device(op.device)
        dtype_reg  = ctx.reg_for_int(dev_type)
        devid_reg  = ctx.reg_for_int(dev_id)

        result_reg = ctx.alloc_reg()
        ctx.bind(op.results[0], result_reg)
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=result_reg,
            func_idx=ctx.builtin("vm.builtin.alloc_storage"),
            arg_regs=[size_reg, align_reg, dtype_reg, devid_reg],
        ))

    # ---- AllocTensorOp -----------------------------------------------------

    def _lower_alloc_tensor(self, op: AllocTensorOp, ctx: _FnCtx) -> None:
        storage_reg = ctx.reg_of(op.storage)
        offset_reg  = ctx.reg_for_int(op.offset)

        # Emit make_shape(d0, d1, ...) for the shape dims
        shape_regs: list[int] = []
        for dim in op.shape:
            if isinstance(dim, IntImm):
                shape_regs.append(ctx.reg_for_int(dim.value))
            elif isinstance(dim, PrimVar):
                reg = ctx._value_reg.get(id(dim))
                if reg is None:
                    raise NotImplementedError(
                        f"PrimVar shape dim '{dim.name}' not in register; "
                        "dynamic shapes require X2."
                    )
                shape_regs.append(reg)
            else:
                raise NotImplementedError(
                    f"Non-IntImm/PrimVar shape dim in AllocTensorOp: {type(dim).__name__}"
                )

        shape_reg = ctx.alloc_reg()
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=shape_reg,
            func_idx=ctx.builtin("vm.builtin.make_shape"),
            arg_regs=shape_regs,
        ))

        # dtype: (code, bits, lanes)
        dtype_code, dtype_bits, dtype_lanes = _parse_dtype(op.dtype)
        code_reg  = ctx.reg_for_int(dtype_code)
        bits_reg  = ctx.reg_for_int(dtype_bits)
        lanes_reg = ctx.reg_for_int(dtype_lanes)

        result_reg = ctx.alloc_reg()
        ctx.bind(op.results[0], result_reg)
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=result_reg,
            func_idx=ctx.builtin("vm.builtin.alloc_tensor"),
            arg_regs=[storage_reg, offset_reg, shape_reg,
                      code_reg, bits_reg, lanes_reg],
        ))

    # ---- CallDPSOp ---------------------------------------------------------

    def _lower_calldps(self, op: CallDPSOp, ctx: _FnCtx) -> None:
        arg_regs = [ctx.reg_of(v) for v in op.inputs]
        if op.output is not None:
            arg_regs.append(ctx.reg_of(op.output))
        func_idx = ctx.intern_func(op.callee, _ir_callee_kind(op.callee_kind))
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=-1,  # DPS ops produce no SSA result
            func_idx=func_idx,
            arg_regs=arg_regs,
        ))

    # ---- ShapeAssertOp -----------------------------------------------------

    def _lower_shape_assert(self, op: ShapeAssertOp, ctx: _FnCtx) -> None:
        tensor_reg = ctx.reg_of(op.tensor)
        dim_reg    = ctx.reg_for_int(op.dim_idx)
        upper_reg  = ctx.reg_for_int(op.upper)
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=-1,
            func_idx=ctx.builtin("vm.builtin.shape_assert"),
            arg_regs=[tensor_reg, dim_reg, upper_reg],
        ))

    # ---- TupleOp -----------------------------------------------------------

    def _lower_tuple(self, op: TupleOp, ctx: _FnCtx) -> None:
        elem_regs  = [ctx.reg_of(e) for e in op.elems]
        result_reg = ctx.alloc_reg()
        ctx.bind(op.results[0], result_reg)
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=result_reg,
            func_idx=ctx.builtin("vm.builtin.make_tuple"),
            arg_regs=elem_regs,
        ))

    # ---- TupleGetItemOp ----------------------------------------------------

    def _lower_tuple_get_item(self, op: TupleGetItemOp, ctx: _FnCtx) -> None:
        tup_reg    = ctx.reg_of(op.tup)
        idx_reg    = ctx.reg_for_int(op.index)
        result_reg = ctx.alloc_reg()
        ctx.bind(op.results[0], result_reg)
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=result_reg,
            func_idx=ctx.builtin("vm.builtin.tuple_get_item"),
            arg_regs=[tup_reg, idx_reg],
        ))

    # ---- IfOp --------------------------------------------------------------

    def _lower_if(self, op: IfOp, ctx: _FnCtx) -> None:
        cond_reg = ctx.reg_of(op.cond)

        # Pre-allocate result registers (before either branch is emitted)
        result_regs: list[int] = []
        for r in op.results:
            reg = ctx.alloc_reg()
            ctx.bind(r, reg)
            result_regs.append(reg)

        # Emit IF with placeholder false_offset
        if_pc = ctx.emit_placeholder(Opcode.IF)
        ctx.instrs[if_pc].cond_reg    = cond_reg
        ctx.instrs[if_pc].true_offset = 1  # then-branch starts immediately after IF

        # ---- then-branch ---------------------------------------------------
        then_block = op.then_region.entry_block
        # Codegen all ops except the terminating YieldOp
        for sub_op in then_block.ops:
            if isinstance(sub_op, YieldOp):
                # Copy yielded values into result_regs via identity
                self._emit_identity_copies(sub_op.values, result_regs, ctx)
            else:
                self._codegen_op(sub_op, ctx)

        has_else = op.else_region is not None

        if has_else:
            # Emit GOTO (placeholder) to skip else-branch
            goto_pc = ctx.emit_placeholder(Opcode.GOTO)

        else_pc = ctx.pc()
        # Backpatch IF.false_offset
        ctx.instrs[if_pc].false_offset = else_pc - if_pc

        if has_else:
            # ---- else-branch -----------------------------------------------
            else_block = op.else_region.entry_block
            for sub_op in else_block.ops:
                if isinstance(sub_op, YieldOp):
                    self._emit_identity_copies(sub_op.values, result_regs, ctx)
                else:
                    self._codegen_op(sub_op, ctx)

            after_pc = ctx.pc()
            # Backpatch GOTO
            ctx.instrs[goto_pc].offset = after_pc - goto_pc

    def _emit_identity_copies(
        self,
        values: tuple[Value, ...],
        result_regs: list[int],
        ctx: _FnCtx,
    ) -> None:
        """Emit vm.builtin.identity calls to copy yielded values → result_regs."""
        identity_idx = ctx.identity_builtin()
        for res_reg, val in zip(result_regs, values):
            src_reg = ctx.reg_of(val)
            ctx.emit(Instruction(
                opcode=Opcode.CALL,
                dst_reg=res_reg,
                func_idx=identity_idx,
                arg_regs=[src_reg],
            ))

    # ---- ForOp -------------------------------------------------------------

    def _lower_for(self, op: ForOp, ctx: _FnCtx) -> None:
        identity_idx = ctx.identity_builtin()

        # Resolve range bounds into registers
        start_reg = self._materialize_value(op.range_.start, ctx)
        end_reg   = self._materialize_value(op.range_.end,   ctx)
        step_reg  = self._materialize_value(op.range_.step,  ctx)

        # Allocate loop variable register and copy start value in
        i_reg = ctx.alloc_reg()
        ctx.bind(op.loop_var, i_reg)
        ctx.emit(Instruction(
            opcode=Opcode.CALL, dst_reg=i_reg,
            func_idx=identity_idx, arg_regs=[start_reg],
        ))

        # Initialize iter_arg registers from their init values
        iter_regs: list[int] = []
        for ia in op.iter_args:
            reg = ctx.alloc_reg()
            ctx.bind(ia.var, reg)
            iter_regs.append(reg)
            init_reg = ctx.reg_of(ia.init)
            ctx.emit(Instruction(
                opcode=Opcode.CALL, dst_reg=reg,
                func_idx=identity_idx, arg_regs=[init_reg],
            ))

        # Loop header: condition check (i < end)
        loop_header_pc = ctx.pc()
        cond_reg = ctx.alloc_reg()
        ctx.emit(Instruction(
            opcode=Opcode.CALL, dst_reg=cond_reg,
            func_idx=ctx.builtin("vm.builtin.lt_i64"),
            arg_regs=[i_reg, end_reg],
        ))

        # IF (loop condition): true → enter body, false → exit loop
        if_pc = ctx.emit_placeholder(Opcode.IF)
        ctx.instrs[if_pc].cond_reg    = cond_reg
        ctx.instrs[if_pc].true_offset = 1  # body starts immediately after IF

        # Codegen loop body
        body_block = op.body_region.entry_block
        body_yield: Optional[YieldOp] = None
        for sub_op in body_block.ops:
            if isinstance(sub_op, YieldOp):
                body_yield = sub_op
            else:
                self._codegen_op(sub_op, ctx)

        # After body: update iter_arg registers from yielded values
        if body_yield is not None:
            for reg, val in zip(iter_regs, body_yield.values):
                src = ctx.reg_of(val)
                ctx.emit(Instruction(
                    opcode=Opcode.CALL, dst_reg=reg,
                    func_idx=identity_idx, arg_regs=[src],
                ))

        # Increment loop variable: i = i + step
        ctx.emit(Instruction(
            opcode=Opcode.CALL, dst_reg=i_reg,
            func_idx=ctx.builtin("vm.builtin.add_i64"),
            arg_regs=[i_reg, step_reg],
        ))

        # GOTO back to loop header (negative offset)
        goto_pc = ctx.pc()
        ctx.emit(Instruction(opcode=Opcode.GOTO, offset=loop_header_pc - goto_pc))

        # After-loop: allocate ForOp result registers, copy from iter_regs
        after_loop_pc = ctx.pc()
        # Backpatch IF false_offset
        ctx.instrs[if_pc].false_offset = after_loop_pc - if_pc

        for i, res in enumerate(op.results):
            res_reg = ctx.alloc_reg()
            ctx.bind(res, res_reg)
            ctx.emit(Instruction(
                opcode=Opcode.CALL, dst_reg=res_reg,
                func_idx=identity_idx,
                arg_regs=[iter_regs[i]],
            ))

    # ---- ReturnOp ----------------------------------------------------------

    def _lower_return(self, op: ReturnOp, ctx: _FnCtx) -> None:
        if len(op.values) == 0:
            ctx.emit(Instruction(opcode=Opcode.RET, src_reg=-1))
        elif len(op.values) == 1:
            src_reg = ctx.reg_of(op.values[0])
            ctx.emit(Instruction(opcode=Opcode.RET, src_reg=src_reg))
        else:
            raise NotImplementedError(
                "ReturnOp with multiple values must use TupleOp first"
            )

    # ---- helpers -----------------------------------------------------------

    def _materialize_value(self, value: Value, ctx: _FnCtx) -> int:
        """Return register holding `value`; for Constant, allocate a const reg."""
        if isinstance(value, Constant):
            return ctx.reg_for_int(int(value.value))
        return ctx.reg_of(value)
