"""devproc2 Python DSL frontend."""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Optional

from devproc2.ir.nodes import (
    Block,
    Constant,
    Function,
    IRModule,
    OpaqueEffect,
    Op,
    OpResult,
    Region,
    TensorStructInfo,
    Value,
    Var,
)
from devproc2.ir.ops import (
    CallDPSOp,
    CalleeKind,
    CallOp,
    ForOp,
    IfOp,
    IterArg,
    Range,
    ReturnOp,
    TupleGetItemOp,
    TupleOp,
    YieldOp,
)
from devproc2.ir.prim_expr import PrimVar
from devproc2.frontend.scope import ScopeStack


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
        if not isinstance(args, tuple) or len(args) < 2:
            raise TypeError("dp.Tensor requires (shape, dtype) or (shape, dtype, device)")
        if len(args) == 2:
            shape_tpl, dtype = args
            device = "cuda"
        else:
            shape_tpl, dtype, device = args[0], args[1], args[2]
        if not isinstance(shape_tpl, tuple):
            shape_tpl = (shape_tpl,)
        return TensorStructInfo(shape_tpl, dtype, device)


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_module: IRModule = IRModule()


def get_module() -> IRModule:
    return _module


def reset_module() -> None:
    global _module
    _module = IRModule()


# ---------------------------------------------------------------------------
# Runtime stubs
# ---------------------------------------------------------------------------

class _RangeStub:
    def __call__(self, start, end, step=1):
        return builtins_range(int(start), int(end), int(step))


import builtins
builtins_range = builtins.range

class _OpsStub:
    def __getattr__(self, name: str):
        def _call(*args):
            pass
        return _call


range = _RangeStub()
ops = _OpsStub()


def call_dps_packed(name: str, inputs=None, output=None, effect: str = "opaque"):
    pass


# ---------------------------------------------------------------------------
# @dp.function decorator
# ---------------------------------------------------------------------------

def function(fn):
    annotations = {k: v for k, v in fn.__annotations__.items() if k != "return"}
    src = inspect.getsource(fn)
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    fn_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    builder = DSLBuilder(annotations=annotations)
    ir_name, ir_func = builder.build(fn_def)
    _module.functions[ir_name] = ir_func
    return fn


# ---------------------------------------------------------------------------
# DSL → IR builder
# ---------------------------------------------------------------------------

class DSLBuilder:
    def __init__(self, annotations: Optional[dict] = None) -> None:
        self.scope = ScopeStack()
        self._counter = 0
        self._annotations = annotations or {}

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
                callee = self._extract_callee(val_expr.func)
                pre_ops, args = self._materialize_args(val_expr.args)
                result_name = self._pick_name(name)
                call_op = CallOp(callee=callee, args=args, result_name=result_name)
                self.scope.define(name, call_op.results[0])
                return pre_ops + [call_op]
            elif isinstance(val_expr, ast.If):
                return self._build_if(val_expr)
            else:
                pre_ops, val = self._materialize_value(val_expr)
                result_name = self._pick_name(name)
                call_op = CallOp(callee="@identity", args=(val,), result_name=result_name)
                self.scope.define(name, call_op.results[0])
                return pre_ops + [call_op]

        if isinstance(stmt, ast.Expr):
            val_expr = stmt.value
            if isinstance(val_expr, ast.Call):
                callee_str = ast.unparse(val_expr.func)
                if "call_dps_packed" in callee_str:
                    return [self._build_call_dps_packed(val_expr)]
                callee = self._extract_callee(val_expr.func)
                pre_ops, args = self._materialize_args(val_expr.args)
                return pre_ops + [CallOp(callee=callee, args=args)]
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
        if not node.args:
            raise DSLError("call_dps_packed requires function name as first arg")
        name_expr = node.args[0]
        if not isinstance(name_expr, ast.Constant) or not isinstance(name_expr.value, str):
            raise DSLError("call_dps_packed: first arg must be a string literal")
        callee = name_expr.value
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        inputs: tuple[Value, ...] = ()
        if "inputs" in kwargs:
            inp_node = kwargs["inputs"]
            if isinstance(inp_node, ast.List):
                inputs = tuple(self._build_value(e) for e in inp_node.elts)
        output: Optional[Var] = None
        if "output" in kwargs:
            out_val = self._build_value(kwargs["output"])
            if isinstance(out_val, Var):
                output = out_val
        return CallDPSOp(
            callee=callee,
            callee_kind=CalleeKind.packed_func,
            inputs=inputs,
            output=output,
            effect=OpaqueEffect(),
        )

    # ------------------------------------------------------------------
    # Value materialization
    # ------------------------------------------------------------------

    def _materialize_value(self, node: ast.expr) -> tuple[list[Op], Value]:
        if isinstance(node, ast.Name):
            v = self.scope.lookup(node.id)
            return [], (v if v is not None else Var(node.id))
        if isinstance(node, ast.Constant):
            return [], Constant(node.value)
        if isinstance(node, ast.Call):
            callee = self._extract_callee(node.func)
            pre_ops, args = self._materialize_args(node.args)
            tmp_name = self._fresh("tmp")
            call_op = CallOp(callee=callee, args=args, result_name=tmp_name)
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
                cmp_op = CallOp(callee=f"@{op_name}", args=(lhs, rhs), result_name=cmp_name)
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
            return f"@{node.attr}"
        if isinstance(node, ast.Name):
            return f"@{node.id}"
        return f"@{ast.unparse(node)}"

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
