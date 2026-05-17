"""devproc2 Python DSL frontend."""
from __future__ import annotations

import ast
import builtins as _builtins
import inspect
import textwrap
from typing import Optional

from devproc2.ir.nodes import (
    Block,
    Constant,
    EffectSummary,
    Function,
    IRModule,
    Op,
    Region,
    TensorStructInfo,
    Value,
    Var,
)
from devproc2.ir.op_ref import ExternalFuncRef, KernelRef, PackedFuncRef, StandardOpRef
from devproc2.ir.ops import (
    CallDPSOp,
    ForOp,
    IfOp,
    IterArg,
    Range,
    ReturnOp,
    TensorCreateKind,
    TensorCreateOp,
    TensorViewOp,
    TupleOp,
    YieldOp,
    make_call_op,
)
from devproc2.ir.prim_expr import IntImm, PrimExpr, PrimVar
from devproc2.compiler.op import get_op
from devproc2.frontend.scope import ScopeStack
from devproc2.kernel.registry import (
    KernelLaunchSpec,
    KernelParamSpec,
    KernelRegistry,
    KernelSpec,
)


class DSLError(Exception):
    pass


# ---------------------------------------------------------------------------
# Symbolic dim / Tensor annotation helpers
# ---------------------------------------------------------------------------

def symbolic_dim(name: str, upper: Optional[int] = None) -> PrimVar:
    """Create a symbolic shape dimension: PrimVar("B", upper=8)."""
    return PrimVar(name, upper=upper)


class Tensor:
    """dp.Tensor[(B, S, 4096), "float16", "cuda"] → TensorStructInfo."""

    def __class_getitem__(cls, args):
        if not isinstance(args, tuple) or len(args) not in (2, 3):
            raise TypeError("dp.Tensor requires (shape, dtype) or (shape, dtype, device)")
        if len(args) == 2:
            shape_tpl, dtype = args
            device = "cuda"
        else:
            shape_tpl, dtype, device = args
        if not isinstance(shape_tpl, tuple):
            shape_tpl = (shape_tpl,)
        return TensorStructInfo(shape_tpl, dtype, device)


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_decorated_fns: list = []           # functions decorated with @dp.function
_kernel_registry: KernelRegistry = KernelRegistry()


def get_kernel_registry() -> KernelRegistry:
    return _kernel_registry


def reset_module() -> None:
    """Clear all decorated functions and kernel registry."""
    global _decorated_fns, _kernel_registry
    _decorated_fns = []
    _kernel_registry = KernelRegistry()


# ---------------------------------------------------------------------------
# Runtime stubs
# ---------------------------------------------------------------------------

class _RangeStub:
    def __call__(self, start, end, step=1):
        return _builtins.range(int(start), int(end), int(step))


class _OpsStub:
    def __getattr__(self, name: str):
        def _call(*args):
            pass
        return _call


range = _RangeStub()
ops = _OpsStub()


def empty(shape, dtype: str = "float32", device: str = "cpu"):
    """Runtime stub — real tensor creation happens via DSL AST compilation."""
    from devproc2.compiler.op.emit import get_current_emitter

    emitter = get_current_emitter()
    if emitter is not None and hasattr(emitter, "emit_empty"):
        return emitter.emit_empty(shape, dtype=dtype, device=device)
    return None


def call_dps_packed(
    name: str,
    inputs=None,
    output=None,
    effect: str = "opaque",
    output_like=None,
    output_shape=None,
    output_dtype: str | None = None,
    output_device: str | None = None,
    output_spec=None,
):
    """Runtime stub — real op emission happens via DSL AST compilation."""
    from devproc2.compiler.op.emit import get_current_emitter

    emitter = get_current_emitter()
    if emitter is not None and hasattr(emitter, "emit_dps_packed"):
        return emitter.emit_dps_packed(
            name,
            inputs=inputs,
            output_like=output_like if output_like is not None else output,
            output_shape=output_shape,
            output_dtype=output_dtype,
            output_device=output_device,
            output_spec=output_spec,
            effect=effect,
        )
    return None


def call_dps_kernel(
    name: str,
    inputs=None,
    output=None,
    effect: str = "opaque",
    launch: KernelLaunchSpec | None = None,
    output_like=None,
    output_shape=None,
    output_dtype: str | None = None,
    output_device: str | None = None,
    output_spec=None,
    output_specs=None,
):
    """Runtime stub — real kernel DPS emission happens via DSL AST compilation."""
    from devproc2.compiler.op.emit import get_current_emitter

    emitter = get_current_emitter()
    if emitter is not None and hasattr(emitter, "emit_dps_kernel"):
        return emitter.emit_dps_kernel(
            name,
            inputs=inputs,
            launch=launch,
            output_like=output_like if output_like is not None else output,
            output_shape=output_shape,
            output_dtype=output_dtype,
            output_device=output_device,
            output_spec=output_spec,
            output_specs=output_specs,
            effect=effect,
        )
    return None


def cuda_call(
    source_symbol: str,
    *args,
    attrs: dict | None = None,
    metadata: dict | None = None,
):
    """Runtime stub for unregistered CUDA source-symbol custom calls."""
    from devproc2.compiler.op.emit import get_current_emitter

    emitter = get_current_emitter()
    if emitter is not None and hasattr(emitter, "emit_cuda_call"):
        return emitter.emit_cuda_call(
            source_symbol,
            args=args,
            attrs=attrs,
            metadata=metadata,
        )
    return None


def tensor_view(
    base,
    byte_offset,
    shape,
    *,
    dtype: str | None = None,
    device: str | None = None,
    byte_stride: int = 1,
    base_offset: int = 0,
):
    """Runtime stub — emits a no-copy tensor view inside tracing/DSL builds."""
    from devproc2.compiler.op.emit import get_current_emitter

    emitter = get_current_emitter()
    if emitter is not None and hasattr(emitter, "emit_tensor_view"):
        return emitter.emit_tensor_view(
            base,
            byte_offset,
            shape,
            dtype=dtype,
            device=device,
            byte_stride=byte_stride,
            base_offset=base_offset,
        )
    return None


def kernel(
    *,
    op: str,
    backend: str,
    device: str = "cuda",
    dtype: str = "",
    dtypes: list | tuple | None = None,
    output_dtype: str | None = None,
    symbol: str | None = None,
    sm_arches=(),
    priority: int = 0,
    attr_constraints=None,
    layout_constraints=(),
    shape_constraints=(),
    launch: KernelLaunchSpec | None = None,
    params: tuple[KernelParamSpec, ...] = (),
    cubin_path: str | None = None,
    ptx_path: str | None = None,
    source_path: str | None = None,
    include_dirs=(),
    extra_nvcc_flags=(),
    compile_options: dict | None = None,
):
    """Decorator to register a kernel implementation.

    Parameters
    ----------
    op : str
        High-level operator name (e.g. "relu", "matmul").  Matched against
        ``dp.ops.relu(x)`` → ``CallOp(StandardOpRef("relu"))`` during DPS lowering.
    backend : str
        Compiler backend: "triton" | "cutedsl" | "cuda" | "python" | "llvm".
    device : str
        Target device: "cuda", "cpu", etc.
    dtype : str
        Convenience shorthand for single-input homogeneous kernels (e.g. relu).
        When set, ``input_dtypes`` = ``(dtype,)``.  Mutually exclusive with
        ``dtypes``.
    dtypes : list[str]
        Explicit per-input dtype list for multi-input kernels (e.g. matmul
        needs ``["float16", "float16"]``).  Takes precedence over ``dtype``.
    launch : KernelLaunchSpec
        Explicit runtime launch metadata. Dynamic entries can use PrimExpr.
    sm_arches : tuple[int]
        Supported SM compute capabilities, e.g. ``(80, 90)``.  Empty = any SM.
    params : tuple[KernelParamSpec, ...]
        Explicit kernel ABI parameters. Empty means artifact emission derives
        a conservative inputs+outputs tensor ABI from the selected CallDPSOp.
    compile_options : dict
        Backend-specific compile options for the provider.

    Example::

        @dp.kernel(
            op="relu",
            backend="triton",
            device="cuda",
            dtype="float16",
            launch=KernelLaunchSpec(grid=(128, 1, 1), block=(256, 1, 1)),
        )
        def relu_kernel(x, out):
            ...  # Triton kernel body

        @dp.kernel(op="matmul", backend="cutedsl", device="cuda",
                   dtypes=["float16", "float16"])
        def matmul_kernel(a, b, out):
            ...
    """
    # Resolve input_dtypes: explicit dtypes list takes precedence over dtype shorthand
    if dtypes is not None:
        resolved_dtypes = tuple(dtypes)
    elif dtype:
        resolved_dtypes = (dtype,)
    else:
        raise ValueError(
            f"@dp.kernel(op={op!r}): either 'dtype' or 'dtypes' must be provided"
        )

    def decorator(fn):
        spec = KernelSpec(
            op_name=op,
            device=device,
            input_dtypes=resolved_dtypes,
            kernel_name=f"kernel.{fn.__name__}",
            backend=backend,
            output_dtype=output_dtype,
            symbol=symbol,
            sm_arches=sm_arches,
            priority=priority,
            attr_constraints=attr_constraints or {},
            layout_constraints=tuple(layout_constraints),
            shape_constraints=tuple(shape_constraints),
            launch=launch or KernelLaunchSpec(),
            params=tuple(params),
            cubin_path=cubin_path,
            ptx_path=ptx_path,
            source_path=source_path,
            include_dirs=tuple(include_dirs),
            extra_nvcc_flags=tuple(extra_nvcc_flags),
            compile_options=compile_options or {},
        )
        _kernel_registry.register(spec)
        fn._kernel_spec = spec
        return fn
    return decorator


# ---------------------------------------------------------------------------
# @dp.function decorator
# ---------------------------------------------------------------------------

def function(fn):
    """Decorator: parse the function's AST and store IR on ``fn._dp_ir``.

    Does NOT mutate any global state.  Call ``fn.lower_module()`` to get an
    IRModule containing just this function.

    Usage::

        @dp.function
        def my_func(x: dp.Tensor[(4,), "float32", "cpu"]):
            return dp.ops.relu(x)

        module = my_func.lower_module()
    """
    annotations = {k: v for k, v in fn.__annotations__.items() if k != "return"}
    src = inspect.getsource(fn)
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    fn_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    builder = DSLBuilder(annotations=annotations, globals_dict=fn.__globals__)
    ir_name, ir_func = builder.build(fn_def)
    fn._dp_ir = (ir_name, ir_func)

    def lower_module():
        mod = IRModule()
        mod.functions[ir_name] = ir_func
        return mod

    fn.lower_module = lower_module
    _decorated_fns.append(fn)
    return fn


# ---------------------------------------------------------------------------
# DSL → IR builder
# ---------------------------------------------------------------------------

class DSLBuilder:
    def __init__(self, annotations: Optional[dict] = None,
                 globals_dict: Optional[dict] = None) -> None:
        self.scope = ScopeStack()
        self._counter = 0
        self._annotations = annotations or {}
        self._globals = globals_dict or {}

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def build(self, fn_def: ast.FunctionDef) -> tuple[str, Function]:
        params = [
            Var(a.arg, struct_info=self._annotations.get(a.arg))
            for a in fn_def.args.args
        ]
        self.scope.push()
        for p in params:
            self.scope.define(p.name, p)
        ops_list, terminator = self._build_stmts(fn_def.body)
        self.scope.pop()
        all_ops = tuple(ops_list) + (terminator,)
        block = Block(args=tuple(params), ops=all_ops)
        region = Region((block,))
        return fn_def.name, Function(region)

    # ------------------------------------------------------------------
    # Statement list → (ops list, terminator)
    # ------------------------------------------------------------------

    def _build_stmts(self, stmts: list[ast.stmt]) -> tuple[list[Op], ReturnOp]:
        ops_list: list[Op] = []
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                pre, ret_val = self._materialize_value(stmt.value)
                ops_list.extend(pre)
                return ops_list, ReturnOp(values=(ret_val,))
            ops_list.extend(self._build_stmt(stmt))
        raise DSLError("function body must end with 'return'")

    # ------------------------------------------------------------------
    # Single statement → list of Ops
    # ------------------------------------------------------------------

    def _build_stmt(self, stmt: ast.stmt) -> list[Op]:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise DSLError("Only simple single-target assignment supported")
            name = stmt.targets[0].id
            val_expr = stmt.value

            if isinstance(val_expr, ast.Call):
                callee_str = ast.unparse(val_expr.func)
                if callee_str in ("dp.empty", "empty"):
                    result_name = self._pick_name(name)
                    create_op = self._build_empty(val_expr, result_name)
                    self.scope.define(name, create_op.results[0])
                    return [create_op]
                if callee_str in ("dp.tensor_view", "tensor_view"):
                    result_name = self._pick_name(name)
                    view_op = self._build_tensor_view(val_expr, result_name)
                    self.scope.define(name, view_op.results[0])
                    return [view_op]
                callee = self._extract_callee(val_expr.func)
                pre_ops, args = self._materialize_args(val_expr.args)
                result_name = self._pick_name(name)
                call_op = make_call_op(
                    op_ref=self._op_ref_for_callee(callee, produces_result=True),
                    args=args,
                    result_name=result_name,
                )
                self.scope.define(name, call_op.results[0])
                return pre_ops + [call_op]
            elif isinstance(val_expr, ast.If):
                return self._build_if(val_expr)
            else:
                pre_ops, val = self._materialize_value(val_expr)
                result_name = self._pick_name(name)
                call_op = make_call_op(
                    op_ref=StandardOpRef("identity", get_op("identity")),
                    args=(val,),
                    result_name=result_name,
                )
                self.scope.define(name, call_op.results[0])
                return pre_ops + [call_op]

        if isinstance(stmt, ast.Expr):
            val_expr = stmt.value
            if isinstance(val_expr, ast.Call):
                callee_str = ast.unparse(val_expr.func)
                if "call_dps_packed" in callee_str:
                    return [self._build_call_dps_packed(val_expr)]
                if "call_dps_kernel" in callee_str:
                    return [self._build_call_dps_kernel(val_expr)]
                callee = self._extract_callee(val_expr.func)
                pre_ops, args = self._materialize_args(val_expr.args)
                return pre_ops + [
                    make_call_op(
                        op_ref=self._op_ref_for_callee(callee, produces_result=False),
                        args=args,
                    )
                ]
            return []

        if isinstance(stmt, ast.If):
            return self._build_if(stmt)

        if isinstance(stmt, ast.For):
            return self._build_for(stmt)

        if isinstance(stmt, ast.While):
            raise DSLError("'while' is not supported in devproc2 DSL")
        if isinstance(stmt, (ast.Break, ast.Continue)):
            raise DSLError("'break'/'continue' are not supported in devproc2 DSL")
        if isinstance(stmt, ast.Return):
            raise DSLError("'return' inside if/for is not supported")

        raise DSLError(f"Unsupported statement: {type(stmt).__name__}")

    # ------------------------------------------------------------------
    # If / elif → nested IfOp
    # ------------------------------------------------------------------

    def _build_if(self, node: ast.If) -> list[Op]:
        pre_ops, cond = self._materialize_value(node.test)
        true_assigned = set(self._collect_assigned(node.body))
        false_assigned = set(self._collect_assigned(node.orelse)) if node.orelse else set()
        shared = sorted(true_assigned & false_assigned)

        if shared:
            snap = self.scope.snapshot()

            self.scope.push()
            true_ops = self._build_branch_ops(node.body)
            true_yield_vals = tuple(self.scope.lookup(n) or Var(n) for n in shared)
            self.scope.restore(snap)

            self.scope.push()
            false_ops = self._build_branch_ops(node.orelse)
            false_yield_vals = tuple(self.scope.lookup(n) or Var(n) for n in shared)
            self.scope.restore(snap)

            result_names = tuple(self._pick_name(n) for n in shared)

            then_region = _make_cf_region(true_ops, YieldOp(true_yield_vals))
            else_region = _make_cf_region(false_ops, YieldOp(false_yield_vals))
            if_op = IfOp(
                cond=cond,
                then_region=then_region,
                else_region=else_region,
                result_names=result_names,
            )
            for python_name, result in zip(shared, if_op.results):
                self.scope.define(python_name, result)
            return pre_ops + [if_op]
        else:
            snap = self.scope.snapshot()

            self.scope.push()
            true_ops = self._build_branch_ops(node.body)
            self.scope.restore(snap)

            else_region: Optional[Region] = None
            if node.orelse:
                self.scope.push()
                false_ops = self._build_branch_ops(node.orelse)
                self.scope.restore(snap)
                else_region = _make_cf_region(false_ops, YieldOp(()))

            then_region = _make_cf_region(true_ops, YieldOp(()))
            return pre_ops + [IfOp(cond=cond, then_region=then_region, else_region=else_region)]

    def _build_branch_ops(self, stmts: list[ast.stmt]) -> list[Op]:
        ops: list[Op] = []
        for stmt in stmts:
            ops.extend(self._build_stmt(stmt))
        return ops

    # ------------------------------------------------------------------
    # For loop → ForOp
    # ------------------------------------------------------------------

    def _build_for(self, node: ast.For) -> list[Op]:
        if not isinstance(node.target, ast.Name):
            raise DSLError("For loop variable must be a simple name")
        range_ = self._build_range(node.iter)
        loop_var = Var(node.target.id)

        body_assigned = set(self._collect_assigned(node.body))
        outer_names = self.scope.outer_names()
        carried = sorted(body_assigned & outer_names)

        snap = self.scope.snapshot()
        self.scope.push()
        self.scope.define(loop_var.name, loop_var)

        iter_args: list[IterArg] = []
        for name in carried:
            outer_val = None
            for frame in reversed(snap):
                if name in frame.bindings:
                    outer_val = frame.bindings[name]
                    break
            iter_v = Var(self._fresh(f"{name}_iter"))
            iter_args.append(IterArg(var=iter_v, init=outer_val or Var(name)))
            self.scope.define(name, iter_v)

        body_ops = self._build_branch_ops(node.body)
        yield_vals = tuple(self.scope.lookup(n) or Var(n) for n in carried)
        self.scope.restore(snap)

        body_region = _make_cf_region(body_ops, YieldOp(yield_vals))

        if carried:
            result_names = tuple(self._pick_name(n) for n in carried)
            for_op = ForOp(
                loop_var=loop_var,
                range_=range_,
                iter_args=tuple(iter_args),
                body_region=body_region,
                result_names=result_names,
            )
            for python_name, result in zip(carried, for_op.results):
                self.scope.define(python_name, result)
            return [for_op]
        else:
            return [ForOp(loop_var=loop_var, range_=range_, iter_args=(), body_region=body_region)]

    # ------------------------------------------------------------------
    # Range
    # ------------------------------------------------------------------

    def _build_range(self, node: ast.expr) -> Range:
        if not isinstance(node, ast.Call):
            raise DSLError("For iter must be dp.range(...)")
        _, args = self._materialize_args(node.args)
        if len(args) < 2:
            raise DSLError("dp.range requires at least 2 arguments")
        return Range(start=args[0], end=args[1], step=args[2] if len(args) >= 3 else Constant(1))

    # ------------------------------------------------------------------
    # call_dps_packed
    # ------------------------------------------------------------------

    def _build_call_dps_packed(self, node: ast.Call) -> CallDPSOp:
        callee, inputs, output = self._parse_dps_call(node, "call_dps_packed")
        return CallDPSOp(
            target_ref=PackedFuncRef(callee),
            inputs=inputs,
            outputs=() if output is None else (output,),
            effect=EffectSummary.opaque_call(),
        )

    def _build_call_dps_kernel(self, node: ast.Call) -> CallDPSOp:
        callee, inputs, output = self._parse_dps_call(node, "call_dps_kernel")
        kernel_name = callee if callee.startswith("kernel.") else f"kernel.{callee}"
        spec = _kernel_registry.get_by_kernel_name(kernel_name)
        return CallDPSOp(
            target_ref=KernelRef(kernel_name, spec),
            inputs=inputs,
            outputs=() if output is None else (output,),
            effect=EffectSummary.opaque_call(),
        )

    def _parse_dps_call(
        self,
        node: ast.Call,
        api_name: str,
    ) -> tuple[str, tuple[Value, ...], Optional[Value]]:
        if not node.args:
            raise DSLError(f"{api_name} requires function name as first arg")
        name_expr = node.args[0]
        if not isinstance(name_expr, ast.Constant) or not isinstance(name_expr.value, str):
            raise DSLError(f"{api_name}: first arg must be a string literal")
        callee = name_expr.value
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        inputs: tuple[Value, ...] = ()
        if "inputs" in kwargs:
            inp_node = kwargs["inputs"]
            if isinstance(inp_node, ast.List):
                inputs = tuple(self._build_value(e) for e in inp_node.elts)
        output: Optional[Value] = None
        if "output" in kwargs:
            out_node = kwargs["output"]
            if not (isinstance(out_node, ast.Constant) and out_node.value is None):
                out_val = self._build_value(kwargs["output"])
                output = out_val
        return callee, inputs, output

    # ------------------------------------------------------------------
    # dp.empty()
    # ------------------------------------------------------------------

    def _build_empty(self, node: ast.Call, result_name: str) -> TensorCreateOp:
        if not node.args:
            raise DSLError("dp.empty requires shape as first argument")
        shape_node = node.args[0]
        if isinstance(shape_node, ast.Tuple):
            shape = tuple(self._build_prim_expr(e) for e in shape_node.elts)
        else:
            shape = (self._build_prim_expr(shape_node),)
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        dtype = kwargs["dtype"].value if "dtype" in kwargs and isinstance(kwargs["dtype"], ast.Constant) else "float32"
        device = kwargs["device"].value if "device" in kwargs and isinstance(kwargs["device"], ast.Constant) else "cpu"
        return TensorCreateOp(
            result_name=result_name,
            kind=TensorCreateKind.empty,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    def _build_tensor_view(self, node: ast.Call, result_name: str) -> TensorViewOp:
        if len(node.args) < 3:
            raise DSLError("dp.tensor_view requires base, byte_offset, and shape")
        base = self._build_value(node.args[0])
        byte_offset = self._build_value(node.args[1])
        shape_node = node.args[2]
        if isinstance(shape_node, ast.Tuple):
            shape = tuple(self._build_prim_expr(e) for e in shape_node.elts)
        else:
            shape = (self._build_prim_expr(shape_node),)
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        dtype = (
            kwargs["dtype"].value
            if "dtype" in kwargs and isinstance(kwargs["dtype"], ast.Constant)
            else None
        )
        device = (
            kwargs["device"].value
            if "device" in kwargs and isinstance(kwargs["device"], ast.Constant)
            else None
        )
        byte_stride = (
            int(kwargs["byte_stride"].value)
            if "byte_stride" in kwargs and isinstance(kwargs["byte_stride"], ast.Constant)
            else 1
        )
        base_offset = (
            int(kwargs["base_offset"].value)
            if "base_offset" in kwargs and isinstance(kwargs["base_offset"], ast.Constant)
            else 0
        )
        return TensorViewOp(
            result_name=result_name,
            base=base,
            byte_offset=byte_offset,
            shape=shape,
            dtype=dtype,
            device=device,
            byte_stride=byte_stride,
            base_offset=base_offset,
        )

    def _build_prim_expr(self, node: ast.expr) -> PrimExpr:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return IntImm(node.value)
        if isinstance(node, ast.Name):
            v = self.scope.lookup(node.id)
            if isinstance(v, PrimVar):
                return v
            # Resolve named integer constants from the caller's module globals
            gval = self._globals.get(node.id)
            if isinstance(gval, int):
                return IntImm(gval)
            return PrimVar(node.id)
        raise DSLError(f"Unsupported shape expression: {ast.unparse(node)}")

    # ------------------------------------------------------------------
    # Value materialization
    # ------------------------------------------------------------------

    def _materialize_value(self, node: ast.expr) -> tuple[list[Op], Value]:
        if isinstance(node, ast.Tuple):
            all_pre: list[Op] = []
            elems: list[Value] = []
            for elt in node.elts:
                pre, val = self._materialize_value(elt)
                all_pre.extend(pre)
                elems.append(val)
            tuple_op = TupleOp(result_name=self._fresh("tuple"), elems=tuple(elems))
            return all_pre + [tuple_op], tuple_op.results[0]
        if isinstance(node, ast.Name):
            v = self.scope.lookup(node.id)
            return [], (v if v is not None else Var(node.id))
        if isinstance(node, ast.Constant):
            return [], Constant(node.value)
        if isinstance(node, ast.Call):
            callee_str = ast.unparse(node.func)
            if callee_str in ("dp.tensor_view", "tensor_view"):
                tmp_name = self._fresh("view")
                view_op = self._build_tensor_view(node, tmp_name)
                return [view_op], view_op.results[0]
            callee = self._extract_callee(node.func)
            pre_ops, args = self._materialize_args(node.args)
            tmp_name = self._fresh("tmp")
            call_op = make_call_op(
                op_ref=self._op_ref_for_callee(callee, produces_result=True),
                args=args,
                result_name=tmp_name,
            )
            return pre_ops + [call_op], call_op.results[0]
        if isinstance(node, ast.Compare):
            lhs_pre, lhs = self._materialize_value(node.left)
            if len(node.ops) == 1 and len(node.comparators) == 1:
                op_name = {
                    ast.Gt: "__gt__", ast.GtE: "__ge__",
                    ast.Lt: "__lt__", ast.LtE: "__le__",
                    ast.Eq: "__eq__", ast.NotEq: "__ne__",
                }.get(type(node.ops[0]), "__cmp__")
                rhs_pre, rhs = self._materialize_value(node.comparators[0])
                cmp_name = self._fresh("cmp")
                cmp_op = make_call_op(
                    op_ref=StandardOpRef(op_name, get_op(op_name)),
                    args=(lhs, rhs),
                    result_name=cmp_name,
                )
                return lhs_pre + rhs_pre + [cmp_op], cmp_op.results[0]
            raise DSLError(f"Complex comparison not supported: {ast.dump(node)}")
        return [], Var(ast.unparse(node))

    def _materialize_args(self, arg_nodes: list[ast.expr]) -> tuple[list[Op], tuple[Value, ...]]:
        all_pre: list[Op] = []
        values: list[Value] = []
        for node in arg_nodes:
            pre, val = self._materialize_value(node)
            all_pre.extend(pre)
            values.append(val)
        return all_pre, tuple(values)

    # ------------------------------------------------------------------
    # Value builder (simple, no op emission)
    # ------------------------------------------------------------------

    def _build_value(self, node: ast.expr) -> Value:
        if isinstance(node, ast.Name):
            v = self.scope.lookup(node.id)
            return v if v is not None else Var(node.id)
        if isinstance(node, ast.Constant):
            return Constant(node.value)
        return Var(ast.unparse(node))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_callee(self, node: ast.expr) -> str:
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node)

    def _op_ref_for_callee(self, callee: str, *, produces_result: bool):
        op_def = get_op(callee)
        if op_def is not None:
            return StandardOpRef(callee, op_def)
        if produces_result:
            return StandardOpRef(callee)
        return ExternalFuncRef(callee)

    def _pick_name(self, python_name: str) -> str:
        """Return a unique SSA name for python_name (fresh if already in scope)."""
        if self.scope.lookup(python_name) is not None:
            return self._fresh(python_name)
        return python_name

    def _fresh(self, base: str) -> str:
        name = f"{base}_{self._counter}"
        self._counter += 1
        return name

    def _collect_assigned(self, stmts: list[ast.stmt]) -> list[str]:
        names: list[str] = []
        for s in stmts:
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
            elif isinstance(s, ast.If):
                names.extend(self._collect_assigned(s.body))
                names.extend(self._collect_assigned(s.orelse))
            elif isinstance(s, ast.For):
                names.extend(self._collect_assigned(s.body))
        return names


# ---------------------------------------------------------------------------
# Helper: build a single-block CF region
# ---------------------------------------------------------------------------

def _make_cf_region(ops: list[Op], terminator: YieldOp) -> Region:
    return Region((Block(args=(), ops=tuple(ops) + (terminator,)),))
