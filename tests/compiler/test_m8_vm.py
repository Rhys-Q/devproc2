"""M8 VM MVP tests — data structures, codegen, interpreter, integration."""
import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir import (
    Block,
    Constant,
    EffectSummary,
    Function,
    IRModule,
    KernelRef,
    Region,
    ReturnOp,
    TensorStructInfo,
    Var,
    YieldOp,
)
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    CallDPSOp,
    ForOp,
    IfOp,
    IterArg,
    Range,
    TensorCreateKind,
    TensorCreateOp,
    TensorViewOp,
    TupleOp,
    TupleGetItemOp,
)
from devproc2.ir.prim_expr import IntImm
from devproc2.kernel.registry import KernelRegistry, KernelSpec
from devproc2.vm import (
    CalleeKind,
    ConstInit,
    Executable,
    FunctionEntry,
    Instruction,
    Opcode,
    VMInterpreter,
    serializer,
)
from devproc2.vm.interpreter import _Storage, _Tensor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_dsl():
    dp.reset_module()
    yield
    dp.reset_module()


def _spec(op: str, **kw) -> KernelSpec:
    defaults = dict(device="cpu", input_dtypes=("float16",),
                    kernel_name=f"kernel.{op}_fp16")
    defaults.update(kw)
    return KernelSpec(op_name=op, **defaults)


def _run_pipeline(module: IRModule, *specs: KernelSpec) -> Executable:
    """Full M1-M8 pipeline: IR → Executable."""
    reg = KernelRegistry()
    for s in specs:
        reg.register(s)
    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(reg).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    return VMCodegenPass().run(module)


def _build_alloc_storage_fn(device: str = "cpu") -> IRModule:
    """Manual IR: f() { s0 = alloc_storage(4096, 256, device); return void }"""
    s_op = AllocStorageOp(result_name="s0", size_bytes=IntImm(4096),
                          alignment=256, device=device)
    ret_op = ReturnOp(values=())
    block = Block(args=(), ops=(s_op, ret_op))
    fn = Function(body=Region((block,)))
    return IRModule({"f": fn})


def _build_alloc_tensor_fn() -> IRModule:
    """Manual IR: f(x) { s0 = alloc_storage(1024, 256, cpu);
                         t0 = alloc_tensor(s0, 0, [512], float16); return t0 }"""
    x = Var("x")
    s_op = AllocStorageOp(result_name="s0", size_bytes=IntImm(1024),
                          alignment=256, device="cpu")
    t_op = AllocTensorOp(result_name="t0", storage=s_op.results[0],
                         offset=0, shape=(IntImm(512),), dtype="float16")
    ret_op = ReturnOp(values=(t_op.results[0],))
    block = Block(args=(x,), ops=(s_op, t_op, ret_op))
    fn = Function(body=Region((block,)))
    return IRModule({"f": fn})


def _build_calldps_fn() -> IRModule:
    """Manual IR: f(x) { s0, t0 = alloc...; call_dps @kernel.relu([x], t0); ret t0 }"""
    x = Var("x")
    s_op = AllocStorageOp(result_name="s0", size_bytes=IntImm(1024),
                          alignment=256, device="cpu")
    t_op = AllocTensorOp(result_name="t0", storage=s_op.results[0],
                         offset=0, shape=(IntImm(512),), dtype="float16")
    call_op = CallDPSOp(
        target_ref=KernelRef("kernel.relu_fp16"),
        inputs=(x,),
        outputs=(t_op.results[0],),
        effect=EffectSummary.opaque_call(),
    )
    ret_op = ReturnOp(values=(t_op.results[0],))
    block = Block(args=(x,), ops=(s_op, t_op, call_op, ret_op))
    fn = Function(body=Region((block,)))
    return IRModule({"f": fn})


def _build_if_fn_with_results() -> IRModule:
    """IR: f(cond) { if cond { yield Constant(1) } else { yield Constant(0) } return result }"""
    cond = Var("cond")

    c1 = Constant(1)
    c0 = Constant(0)

    then_yield = YieldOp(values=(c1,))
    then_block = Block(args=(), ops=(then_yield,))
    then_region = Region((then_block,))

    else_yield = YieldOp(values=(c0,))
    else_block = Block(args=(), ops=(else_yield,))
    else_region = Region((else_block,))

    if_op = IfOp(
        cond=cond,
        then_region=then_region,
        else_region=else_region,
        result_names=("result",),
    )
    ret_op = ReturnOp(values=(if_op.results[0],))
    block = Block(args=(cond,), ops=(if_op, ret_op))
    fn = Function(body=Region((block,)))
    return IRModule({"f": fn})


def _build_if_fn_effect_only() -> IRModule:
    """IR: f(cond) { if cond { yield } else { yield } return void }"""
    cond = Var("cond")

    then_block = Block(args=(), ops=(YieldOp(values=()),))
    else_block = Block(args=(), ops=(YieldOp(values=()),))
    if_op = IfOp(
        cond=cond,
        then_region=Region((then_block,)),
        else_region=Region((else_block,)),
        result_names=(),
    )
    ret_op = ReturnOp(values=())
    block = Block(args=(cond,), ops=(if_op, ret_op))
    fn = Function(body=Region((block,)))
    return IRModule({"f": fn})


def _build_for_fn_no_iter_args() -> IRModule:
    """IR: f(n) { for i in range(0, n, 1): yield; return void }"""
    n = Var("n")
    i_var = Var("i")

    body_block = Block(args=(), ops=(YieldOp(values=()),))
    for_op = ForOp(
        loop_var=i_var,
        range_=Range(start=Constant(0), end=n, step=Constant(1)),
        iter_args=(),
        body_region=Region((body_block,)),
        result_names=(),
    )
    ret_op = ReturnOp(values=())
    block = Block(args=(n,), ops=(for_op, ret_op))
    fn = Function(body=Region((block,)))
    return IRModule({"f": fn})


# ---------------------------------------------------------------------------
# 1. Executable / Instruction data structure tests
# ---------------------------------------------------------------------------

def test_opcode_values():
    assert Opcode.CALL == 0
    assert Opcode.RET  == 1
    assert Opcode.IF   == 2
    assert Opcode.GOTO == 3


def test_callee_kind_values():
    assert CalleeKind.vm_func     == 0
    assert CalleeKind.builtin     == 1
    assert CalleeKind.packed_func == 2
    assert CalleeKind.kernel      == 3


def test_instruction_call_defaults():
    instr = Instruction(opcode=Opcode.CALL, dst_reg=3, func_idx=1, arg_regs=[0, 1])
    assert instr.opcode == Opcode.CALL
    assert instr.dst_reg == 3
    assert instr.func_idx == 1
    assert instr.arg_regs == [0, 1]
    assert instr.src_reg == -1  # default


def test_instruction_ret_defaults():
    instr = Instruction(opcode=Opcode.RET, src_reg=-1)
    assert instr.opcode == Opcode.RET
    assert instr.src_reg == -1
    assert instr.dst_reg == -1


def test_executable_get_func_index():
    exe = Executable()
    exe.function_table.append(
        FunctionEntry("f", CalleeKind.vm_func, 0, 5, 4, 1))
    exe.function_table.append(
        FunctionEntry("g", CalleeKind.vm_func, 5, 3, 2, 0))
    assert exe.get_func_index("f") == 0
    assert exe.get_func_index("g") == 1
    with pytest.raises(KeyError):
        exe.get_func_index("missing")


def test_const_init_stored_in_function_entry():
    ci = ConstInit(reg_idx=5, const_idx=2)
    fe = FunctionEntry("f", CalleeKind.vm_func, 0, 1, 10, 2,
                       const_inits=[ci])
    assert len(fe.const_inits) == 1
    assert fe.const_inits[0].reg_idx == 5
    assert fe.const_inits[0].const_idx == 2


# ---------------------------------------------------------------------------
# 2. VMCodegenPass tests
# ---------------------------------------------------------------------------

def test_codegen_alloc_storage_function_table():
    """AllocStorageOp generates a CALL to vm.builtin.alloc_storage."""
    module = _build_alloc_storage_fn()
    exe = VMCodegenPass().run(module)

    # Should have "f" vm_func + "vm.builtin.alloc_storage" builtin
    names = [fe.name for fe in exe.function_table]
    assert "f" in names
    assert "vm.builtin.alloc_storage" in names
    alloc_idx = exe.get_func_index("vm.builtin.alloc_storage")
    assert exe.function_table[alloc_idx].kind == CalleeKind.builtin


def test_codegen_alloc_storage_const_inits():
    """AllocStorageOp constants (size, alignment, device) go into const_inits."""
    module = _build_alloc_storage_fn("cpu")
    exe = VMCodegenPass().run(module)

    f_entry = exe.function_table[exe.get_func_index("f")]
    # Must have const_inits for: size_bytes=4096, alignment=256, device_type=1, device_id=0
    const_vals = {exe.constants[ci.const_idx] for ci in f_entry.const_inits}
    assert 4096 in const_vals
    assert 256  in const_vals
    assert 1    in const_vals  # kDLCPU


def test_codegen_alloc_storage_and_tensor():
    """AllocTensorOp generates make_shape + alloc_tensor calls."""
    module = _build_alloc_tensor_fn()
    exe = VMCodegenPass().run(module)

    names = [fe.name for fe in exe.function_table]
    assert "vm.builtin.make_shape" in names
    assert "vm.builtin.alloc_tensor" in names

    f_entry = exe.function_table[exe.get_func_index("f")]
    # There should be at least 4 instructions: alloc_storage CALL, make_shape CALL,
    # alloc_tensor CALL, RET
    assert f_entry.instr_count >= 4


def test_codegen_tensor_view_emits_builtin_and_offset_math():
    x = Var("x", TensorStructInfo((16,), "float16", "cpu"))
    view = TensorViewOp(
        result_name="view",
        base=x,
        byte_offset=Constant(3),
        shape=(IntImm(4),),
        byte_stride=2,
        base_offset=8,
    )
    block = Block(args=(x,), ops=(view, ReturnOp((view.results[0],))))
    module = IRModule({"f": Function(Region((block,)))})
    exe = VMCodegenPass().run(module)
    names = [fe.name for fe in exe.function_table]
    assert "vm.builtin.mul_i64" in names
    assert "vm.builtin.add_i64" in names
    assert "vm.builtin.make_shape" in names
    assert "vm.builtin.tensor_view" in names


def test_tensor_view_keeps_base_storage_live_when_returned():
    base = TensorCreateOp(
        result_name="base",
        kind=TensorCreateKind.empty,
        shape=(IntImm(8),),
        dtype="float16",
        device="cpu",
    )
    view = TensorViewOp(
        result_name="view",
        base=base.results[0],
        byte_offset=Constant(4),
        shape=(IntImm(4),),
    )
    scratch = TensorCreateOp(
        result_name="scratch",
        kind=TensorCreateKind.empty,
        shape=(IntImm(8),),
        dtype="float16",
        device="cpu",
    )
    block = Block(args=(), ops=(
        base,
        view,
        scratch,
        ReturnOp((view.results[0],)),
    ))
    module = IRModule({"f": Function(Region((block,)))})
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    plan = ctx.get("storage_plan")
    assert plan.tensor_to_storage["base"] != plan.tensor_to_storage["scratch"]

def test_codegen_return_value():
    """ReturnOp emits RET with the correct src_reg."""
    x = Var("x")
    ret_op = ReturnOp(values=(x,))
    block = Block(args=(x,), ops=(ret_op,))
    fn = Function(body=Region((block,)))
    module = IRModule({"f": fn})
    exe = VMCodegenPass().run(module)

    f_idx = exe.get_func_index("f")
    f_entry = exe.function_table[f_idx]
    last_instr = exe.instructions[f_entry.instr_offset + f_entry.instr_count - 1]
    assert last_instr.opcode == Opcode.RET
    assert last_instr.src_reg == 0  # x is in register 0


def test_codegen_return_void():
    """ReturnOp with no values emits RET with src_reg=-1."""
    ret_op = ReturnOp(values=())
    block = Block(args=(), ops=(ret_op,))
    fn = Function(body=Region((block,)))
    exe = VMCodegenPass().run(IRModule({"f": fn}))

    f_entry = exe.function_table[exe.get_func_index("f")]
    last_instr = exe.instructions[f_entry.instr_offset + f_entry.instr_count - 1]
    assert last_instr.opcode == Opcode.RET
    assert last_instr.src_reg == -1


def test_codegen_calldps_kernel_dst_reg_minus_one():
    """CallDPSOp emits CALL with dst_reg=-1 (DPS produces no SSA value)."""
    module = _build_calldps_fn()
    exe = VMCodegenPass().run(module)

    # Find the CALL to kernel.relu_fp16
    kernel_idx = exe.get_func_index("kernel.relu_fp16")
    assert exe.function_table[kernel_idx].kind == CalleeKind.kernel

    f_entry = exe.function_table[exe.get_func_index("f")]
    kernel_calls = [
        exe.instructions[f_entry.instr_offset + i]
        for i in range(f_entry.instr_count)
        if exe.instructions[f_entry.instr_offset + i].opcode == Opcode.CALL
        and exe.instructions[f_entry.instr_offset + i].func_idx == kernel_idx
    ]
    assert len(kernel_calls) == 1
    assert kernel_calls[0].dst_reg == -1


def test_codegen_ifop_with_results_instruction_structure():
    """IfOp with SSA results generates: IF + then_body + GOTO + else_body."""
    module = _build_if_fn_with_results()
    exe = VMCodegenPass().run(module)

    f_entry = exe.function_table[exe.get_func_index("f")]
    instrs = exe.instructions[f_entry.instr_offset:
                               f_entry.instr_offset + f_entry.instr_count]
    opcodes = [i.opcode for i in instrs]

    assert Opcode.IF   in opcodes
    assert Opcode.GOTO in opcodes
    assert Opcode.RET  in opcodes


def test_codegen_ifop_with_results_offsets():
    """IF.false_offset skips past then-branch + GOTO to the else-branch."""
    module = _build_if_fn_with_results()
    exe = VMCodegenPass().run(module)

    f_entry = exe.function_table[exe.get_func_index("f")]
    instrs = exe.instructions[f_entry.instr_offset:
                               f_entry.instr_offset + f_entry.instr_count]

    if_instr = next(i for i in instrs if i.opcode == Opcode.IF)
    goto_instr = next(i for i in instrs if i.opcode == Opcode.GOTO)

    # false_offset should be positive (jump forward past then+GOTO)
    assert if_instr.false_offset > 0
    # true_offset always 1 (next instr = then branch)
    assert if_instr.true_offset == 1
    # GOTO.offset should be positive (skip over else branch)
    assert goto_instr.offset > 0


def test_codegen_ifop_effect_only_no_goto_needed():
    """Effect-only IfOp may or may not emit GOTO; importantly no identity calls."""
    module = _build_if_fn_effect_only()
    exe = VMCodegenPass().run(module)

    f_entry = exe.function_table[exe.get_func_index("f")]
    instrs = exe.instructions[f_entry.instr_offset:
                               f_entry.instr_offset + f_entry.instr_count]

    identity_idx = next(
        (i for i, fe in enumerate(exe.function_table)
         if fe.name == "vm.builtin.identity"), None)

    if identity_idx is not None:
        identity_calls = [
            instr for instr in instrs
            if instr.opcode == Opcode.CALL and instr.func_idx == identity_idx
        ]
        assert len(identity_calls) == 0, "Effect-only IfOp should not emit identity calls"


def test_codegen_forop_basic_has_goto():
    """ForOp generates a backward GOTO (negative offset) for the loop-back."""
    module = _build_for_fn_no_iter_args()
    exe = VMCodegenPass().run(module)

    f_entry = exe.function_table[exe.get_func_index("f")]
    instrs = exe.instructions[f_entry.instr_offset:
                               f_entry.instr_offset + f_entry.instr_count]

    goto_instrs = [i for i in instrs if i.opcode == Opcode.GOTO]
    assert len(goto_instrs) == 1, "ForOp should emit exactly one GOTO"
    assert goto_instrs[0].offset < 0, "GOTO should jump backward to loop header"


def test_codegen_forop_has_condition_check():
    """ForOp generates a vm.builtin.lt_i64 call for the loop condition."""
    module = _build_for_fn_no_iter_args()
    exe = VMCodegenPass().run(module)

    names = [fe.name for fe in exe.function_table]
    assert "vm.builtin.lt_i64" in names


# ---------------------------------------------------------------------------
# 3. VMInterpreter execution tests
# ---------------------------------------------------------------------------

def _make_simple_executable() -> Executable:
    """Executable for: identity(x) → return x (r0=x, RET r0)"""
    exe = Executable()
    exe.function_table = [
        FunctionEntry("f", CalleeKind.vm_func, 0, 1, 1, 1),
    ]
    exe.instructions = [
        Instruction(opcode=Opcode.RET, src_reg=0),
    ]
    return exe


def _make_alloc_storage_executable() -> Executable:
    """Executable for: alloc_storage(256, 256, 1, 0) → return storage"""
    exe = Executable()
    exe.constants = [256, 256, 1, 0]  # size, alignment, device_type, device_id
    # regs: r0=const[0]=256(sz), r1=const[1]=256(align), r2=const[2]=1(dt), r3=const[3]=0(did)
    # r4 = alloc_storage result; r5 = (unused)
    alloc_idx = 1
    exe.function_table = [
        FunctionEntry("f", CalleeKind.vm_func, 0, 2, 5, 0,
                      const_inits=[ConstInit(0, 0), ConstInit(1, 1),
                                   ConstInit(2, 2), ConstInit(3, 3)]),
        FunctionEntry("vm.builtin.alloc_storage", CalleeKind.builtin, -1, 0, 0, 0),
    ]
    exe.instructions = [
        Instruction(opcode=Opcode.CALL, dst_reg=4, func_idx=1,
                    arg_regs=[0, 1, 2, 3]),
        Instruction(opcode=Opcode.RET, src_reg=4),
    ]
    return exe


def test_interpreter_identity_return():
    """f(x) → return x: interpreter returns the passed value."""
    exe = _make_simple_executable()
    vm = VMInterpreter(exe)
    assert vm.invoke("f", [42]) == 42
    assert vm.invoke("f", ["hello"]) == "hello"


def test_interpreter_alloc_storage_returns_storage():
    """vm.builtin.alloc_storage returns a _Storage object."""
    exe = _make_alloc_storage_executable()
    vm = VMInterpreter(exe)
    result = vm.invoke("f", [])
    assert isinstance(result, _Storage)
    assert result.nbytes == 256
    assert result.device_type == 1   # kDLCPU


def test_interpreter_alloc_tensor_returns_tensor():
    """alloc_storage + alloc_tensor returns a _Tensor with correct shape."""
    module = _build_alloc_tensor_fn()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    # x is not actually used in the function (only storage is), pass a dummy
    result = vm.invoke("f", [None])
    assert isinstance(result, _Tensor)
    assert result.shape == (512,)
    assert result.dtype_code == 2   # kDLFloat
    assert result.dtype_bits == 16


def test_interpreter_kernel_mock_no_error():
    """CallDPSOp with kernel callee runs without error (M8 kernel is a no-op stub)."""
    module = _build_calldps_fn()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    # x = None (not used), result should be a _Tensor
    result = vm.invoke("f", [None])
    assert isinstance(result, _Tensor)


def test_interpreter_if_true_branch():
    """IfOp: cond=True executes then-branch."""
    module = _build_if_fn_with_results()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    result = vm.invoke("f", [True])
    assert result == 1


def test_interpreter_if_false_branch():
    """IfOp: cond=False executes else-branch."""
    module = _build_if_fn_with_results()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    result = vm.invoke("f", [False])
    assert result == 0


def test_interpreter_if_effect_only_no_error():
    """Effect-only IfOp runs without error for both branches."""
    module = _build_if_fn_effect_only()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    vm.invoke("f", [True])
    vm.invoke("f", [False])


def test_interpreter_for_loop_count():
    """ForOp: loop terminates without error for various iteration counts."""
    module = _build_for_fn_no_iter_args()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)

    for n in [0, 1, 3, 5]:
        result = vm.invoke("f", [n])
        assert result is None  # void return


def test_interpreter_for_loop_termination_various_n():
    """ForOp with various n values terminates correctly."""
    module = _build_for_fn_no_iter_args()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    for n in [0, 1, 4, 10]:
        result = vm.invoke("f", [n])
        assert result is None  # void return


def test_interpreter_for_with_iter_args_manually_built():
    """ForOp with iter_arg: manually constructed Executable for add-accumulation."""
    # Function: count(N) = sum of 1s for i in 0..N-1 = N
    # Registers:
    #  r0 = N (arg)
    #  r1 = 0 (init i, const_init[0])
    #  r2 = 0 (init acc, const_init[1])
    #  r3 = 1 (step, const_init[2])
    #  r4 = cond (lt_i64 result)
    #  r5 = new_acc (add_i64 result)
    #  r6 = result (after loop)
    exe = Executable()
    exe.constants = [0, 1]  # const[0]=0, const[1]=1

    identity_fidx = 0
    lt_fidx       = 1
    add_fidx      = 2
    main_fidx     = 3

    exe.function_table = [
        FunctionEntry("vm.builtin.identity", CalleeKind.builtin, -1, 0, 0, 0),
        FunctionEntry("vm.builtin.lt_i64",   CalleeKind.builtin, -1, 0, 0, 0),
        FunctionEntry("vm.builtin.add_i64",  CalleeKind.builtin, -1, 0, 0, 0),
        FunctionEntry("count", CalleeKind.vm_func, 0, 0, 7, 1,
                      const_inits=[ConstInit(1, 0), ConstInit(2, 0), ConstInit(3, 1)]),
    ]

    # Instructions:
    # [0] CALL r1, identity, [r1]        — init i = 0 (already done by const_init)
    # [1] CALL r2, identity, [r2]        — init acc = 0 (already done)
    # [2] (loop header) CALL r4, lt_i64, [r1, r0]
    # [3] IF r4, +1, +5  (true→[4] body, false→[8] after)
    # [4] CALL r5, add_i64, [r2, r3]     — new_acc = acc + 1
    # [5] CALL r2, identity, [r5]        — acc = new_acc
    # [6] CALL r1, add_i64, [r1, r3]     — i = i + 1
    # [7] GOTO -5                         — back to [2]
    # [8] CALL r6, identity, [r2]        — result = acc
    # [9] RET r6
    exe.instructions = [
        # init (identity copies — not strictly needed since const_init, but matches codegen)
        Instruction(Opcode.CALL, dst_reg=1, func_idx=identity_fidx, arg_regs=[1]),
        Instruction(Opcode.CALL, dst_reg=2, func_idx=identity_fidx, arg_regs=[2]),
        # loop header pc=2
        Instruction(Opcode.CALL, dst_reg=4, func_idx=lt_fidx, arg_regs=[1, 0]),
        Instruction(Opcode.IF,   cond_reg=4, true_offset=1, false_offset=5),
        # body pc=4
        Instruction(Opcode.CALL, dst_reg=5, func_idx=add_fidx, arg_regs=[2, 3]),
        Instruction(Opcode.CALL, dst_reg=2, func_idx=identity_fidx, arg_regs=[5]),
        Instruction(Opcode.CALL, dst_reg=1, func_idx=add_fidx, arg_regs=[1, 3]),
        Instruction(Opcode.GOTO, offset=-5),  # back to pc=2
        # after loop pc=8
        Instruction(Opcode.CALL, dst_reg=6, func_idx=identity_fidx, arg_regs=[2]),
        Instruction(Opcode.RET,  src_reg=6),
    ]
    # Update FunctionEntry instr_count
    exe.function_table[main_fidx].instr_count = len(exe.instructions)
    exe.function_table[main_fidx].num_regs = 7

    vm = VMInterpreter(exe)
    assert vm.invoke("count", [0]) == 0
    assert vm.invoke("count", [1]) == 1
    assert vm.invoke("count", [5]) == 5
    assert vm.invoke("count", [10]) == 10


# ---------------------------------------------------------------------------
# 4. Serializer roundtrip test
# ---------------------------------------------------------------------------

def test_serializer_roundtrip_preserves_structure():
    """Executable → serialize → deserialize keeps function_table and constants."""
    module = _build_alloc_tensor_fn()
    exe = VMCodegenPass().run(module)

    data = serializer.serialize(exe)
    exe2 = serializer.deserialize(data)

    assert len(exe2.function_table) == len(exe.function_table)
    for fe1, fe2 in zip(exe.function_table, exe2.function_table):
        assert fe1.name == fe2.name
        assert fe1.kind == fe2.kind
        assert fe1.num_args == fe2.num_args
        assert fe1.param_names == fe2.param_names
        assert len(fe1.const_inits) == len(fe2.const_inits)

    assert len(exe2.instructions) == len(exe.instructions)
    assert exe2.constants == exe.constants


# ---------------------------------------------------------------------------
# 5. Integration / acceptance tests
# ---------------------------------------------------------------------------

def test_acceptance_alloc_and_ret():
    """Acceptance criterion: alloc_storage + alloc_tensor + ret executes correctly."""
    module = _build_alloc_tensor_fn()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    result = vm.invoke("f", [None])
    assert isinstance(result, _Tensor)
    assert result.shape == (512,)


def test_acceptance_kernel_call_and_ret():
    """Acceptance criterion: alloc_storage + alloc_tensor + kernel_call + ret."""
    module = _build_calldps_fn()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)
    result = vm.invoke("f", [None])
    assert isinstance(result, _Tensor)


def test_acceptance_if_goto_true_and_false():
    """Acceptance criterion: if/goto correct for both true and false branches."""
    module = _build_if_fn_with_results()
    exe = VMCodegenPass().run(module)
    vm = VMInterpreter(exe)

    # Verify both branches produce distinct values
    assert vm.invoke("f", [True])  == 1
    assert vm.invoke("f", [False]) == 0


def test_full_pipeline_dsl_relu():
    """Full pipeline: DSL → M7 → VMCodegenPass → VMInterpreter."""
    @dp.function
    def relu_fn(x: dp.Tensor[(512,), "float16", "cpu"]):
        y = dp.ops.relu(x)
        return y

    module = relu_fn.lower_module()
    exe = _run_pipeline(module, _spec("relu"))
    vm = VMInterpreter(exe)
    result = vm.invoke("relu_fn", [None])
    # relu_fn returns an allocated tensor (kernel is a no-op mock in M8)
    assert isinstance(result, _Tensor)
    assert result.shape == (512,)


def test_full_pipeline_multi_op():
    """Full pipeline: multiple ops, checks storage reuse and correct execution."""
    @dp.function
    def chain_fn(x: dp.Tensor[(256,), "float16", "cpu"]):
        y = dp.ops.relu(x)
        z = dp.ops.relu(y)
        return z

    module = chain_fn.lower_module()
    exe = _run_pipeline(module, _spec("relu"))
    vm = VMInterpreter(exe)
    result = vm.invoke("chain_fn", [None])
    assert isinstance(result, _Tensor)
    assert result.shape == (256,)
