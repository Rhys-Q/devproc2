"""VMCodegenPass: memory-explicit IR → Executable VM bytecode."""
from __future__ import annotations

from typing import Any, Optional

from devproc2.ir.nodes import (
    Block,
    Function,
    IRModule,
    IRStage,
    Op,
    Value,
    Constant,
)
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    CallDPSOp,
    ForOp,
    IfOp,
    ReturnOp,
    ShapeAssertOp,
    TupleGetItemOp,
    TupleOp,
    YieldOp,
)
from devproc2.ir.op_ref import BuiltinOpRef, KernelRef, PackedFuncRef
from devproc2.ir.prim_expr import IntImm, PrimVar
from devproc2.compiler.passes.shape_expr_lowering import ShapeExprLoweringPass, _PrimExprLowerer
from devproc2.vm.executable import (
    CalleeKind,
    ConstInit,
    Executable,
    FunctionEntry,
    Instruction,
    Opcode,
)
from devproc2.utils.dtype import parse_dtype, parse_device


def _target_callee_kind(ref) -> CalleeKind:
    if isinstance(ref, KernelRef):
        return CalleeKind.kernel
    if isinstance(ref, PackedFuncRef):
        return CalleeKind.packed_func
    if isinstance(ref, BuiltinOpRef):
        return CalleeKind.builtin
    raise TypeError(f"unsupported CallDPS target ref {type(ref).__name__}")


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
        # Set by ShapeExprLoweringPass.setup_fn at function-entry codegen time
        self.prim_lowerer: Optional[_PrimExprLowerer] = None

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

    Parameters
    ----------
    kernel_specs : dict[str, KernelSpec], optional
        Map from kernel callee name (e.g. "kernel.relu_fp16") to its
        KernelSpec. When provided, grid dims from spec.grid_fn are emitted
        as constant args appended to the kernel CALL instruction.
    """
    input_stage = IRStage.memory
    output_stage = IRStage.vm
    required_analysis: tuple[str, ...] = ()
    preserved_analysis: tuple[str, ...] = ()

    def __init__(self, kernel_specs=None) -> None:
        self._kernel_specs: dict = kernel_specs or {}

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

        # Setup PrimVar extraction + assert_le_i64 prologue for dynamic shapes.
        ctx.prim_lowerer = ShapeExprLoweringPass.setup_fn(fn, ctx)

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
        if isinstance(op.size_bytes, IntImm):
            size_reg = ctx.reg_for_int(op.size_bytes.value)
        else:
            size_reg = ctx.prim_lowerer.materialize(op.size_bytes)
        align_reg  = ctx.reg_for_int(op.alignment)
        dev_type, dev_id = parse_device(op.device)
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
                    raise RuntimeError(
                        f"PrimVar shape dim '{dim.name}' not in register; "
                        "ensure ShapeExprLoweringPass.setup_fn ran first."
                    )
                shape_regs.append(reg)
            else:
                shape_regs.append(ctx.prim_lowerer.materialize(dim))

        shape_reg = ctx.alloc_reg()
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=shape_reg,
            func_idx=ctx.builtin("vm.builtin.make_shape"),
            arg_regs=shape_regs,
        ))

        # dtype: (code, bits, lanes)
        dtype_code, dtype_bits, dtype_lanes = parse_dtype(op.dtype)
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
        for output in op.outputs:
            arg_regs.append(ctx.reg_of(output))
        # For kernel callees: emit static grid dims (gx, gy, gz) as extra args.
        if isinstance(op.target_ref, KernelRef):
            spec = op.target_ref.spec or self._kernel_specs.get(op.target_ref.name)
            if spec is not None and spec.grid_fn is not None:
                grid = self._compute_grid(spec.grid_fn, op.inputs)
                for g in grid:
                    arg_regs.append(ctx.reg_for_int(int(g)))
        func_idx = ctx.intern_func(op.target_ref.name, _target_callee_kind(op.target_ref))
        ctx.emit(Instruction(
            opcode=Opcode.CALL,
            dst_reg=-1,  # DPS ops produce no SSA result
            func_idx=func_idx,
            arg_regs=arg_regs,
        ))

    @staticmethod
    def _compute_grid(grid_fn, inputs: tuple) -> tuple[int, int, int]:
        """Compute static grid dims from input shapes when possible.

        Extracts concrete shapes from each input's struct_info.  If all dims
        are IntImm (static), passes ``[(d0,d1,...), ...]`` to grid_fn.
        Falls back to no-arg call for backward compatibility or when shapes
        contain dynamic PrimVars.
        """
        shapes = []
        all_static = True
        for v in inputs:
            si = getattr(v, "struct_info", None)
            if si is not None and hasattr(si, "shape"):
                dims = []
                for d in si.shape:
                    if isinstance(d, IntImm):
                        dims.append(d.value)
                    else:
                        all_static = False
                        break
                shapes.append(tuple(dims))
            else:
                all_static = False
                break

        if all_static:
            try:
                return tuple(grid_fn(shapes))
            except TypeError:
                pass  # fall through to no-arg call
        return tuple(grid_fn())

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
        if op.results and op.else_region is None:
            raise ValueError(
                "IfOp with results must have an else_region; "
                "otherwise false-branch result registers would be uninitialized"
            )

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
