from __future__ import annotations

from typing import Iterator, Optional

from devproc2.ir.nodes import (
    Block,
    Constant,
    Function,
    IRModule,
    Op,
    OpResult,
    Region,
    TerminatorOp,
    Value,
    Var,
    WriteEffect,
)
from devproc2.ir.ops import (
    CallDPSOp,
    CallOp,
    ForOp,
    IfOp,
    ReturnOp,
    ShapeAssertOp,
    TensorCreateOp,
    TupleGetItemOp,
    TupleOp,
    YieldOp,
)

_FORBIDDEN_CALLEES = frozenset({"@alloc_storage", "@alloc_tensor"})


class IRVerificationError(Exception):
    pass


class Verifier:
    def verify_module(self, module: IRModule) -> None:
        for name, fn in module.functions.items():
            self._verify_function(name, fn)

    # ------------------------------------------------------------------
    # Function
    # ------------------------------------------------------------------

    def _verify_function(self, name: str, fn: Function) -> None:
        seen: set[str] = set()
        for p in fn.params:
            if p.name in seen:
                raise IRVerificationError(
                    f"In @{name}: parameter '%{p.name}' defined more than once"
                )
            seen.add(p.name)
        self._verify_region(name, fn.body, set(), set(), expected_terminator=ReturnOp)

    # ------------------------------------------------------------------
    # Region / Block
    # ------------------------------------------------------------------

    def _verify_region(
        self,
        fn_name: str,
        region: Region,
        defined_names: set[str],
        defined_results: set[int],
        expected_terminator: type[TerminatorOp] = YieldOp,
    ) -> None:
        for block in region.blocks:
            self._verify_block(fn_name, block, defined_names, defined_results, expected_terminator)

    def _verify_block(
        self,
        fn_name: str,
        block: Block,
        defined_names: set[str],
        defined_results: set[int],
        expected_terminator: type[TerminatorOp],
    ) -> None:
        if not block.ops:
            raise IRVerificationError(
                f"In @{fn_name}: block must not be empty (needs at least a terminator)"
            )
        for arg in block.args:
            if arg.name in defined_names:
                raise IRVerificationError(
                    f"In @{fn_name}: Variable '%{arg.name}' defined more than once"
                )
            defined_names.add(arg.name)

        for i, op in enumerate(block.ops):
            is_last = i == len(block.ops) - 1

            if isinstance(op, TerminatorOp):
                if not is_last:
                    raise IRVerificationError(
                        f"In @{fn_name}: TerminatorOp {type(op).__name__} must be "
                        f"the last op in a block, but appears at position {i}"
                    )
                if not isinstance(op, expected_terminator):
                    raise IRVerificationError(
                        f"In @{fn_name}: block expects {expected_terminator.__name__} "
                        f"as terminator, got {type(op).__name__}"
                    )
            else:
                if is_last:
                    raise IRVerificationError(
                        f"In @{fn_name}: last op in block must be a TerminatorOp, "
                        f"got {type(op).__name__}"
                    )

            self._verify_op(fn_name, op, defined_names, defined_results)

            for result in op.results:
                if id(result) in defined_results:
                    raise IRVerificationError(
                        f"In @{fn_name}: OpResult defined more than once"
                    )
                defined_results.add(id(result))

    # ------------------------------------------------------------------
    # Op-level verification
    # ------------------------------------------------------------------

    def _chk_value(
        self,
        fn_name: str,
        v: Value,
        defined_names: set[str],
        defined_results: set[int],
    ) -> None:
        """Assert that v (Var or OpResult) has been defined in the current scope.

        id(OpResult) is used as identity: within one IR tree construction
        all OpResult objects are alive, so id() is stable and unique.
        """
        if isinstance(v, Var):
            if v.name not in defined_names:
                raise IRVerificationError(
                    f"In @{fn_name}: Variable '%{v.name}' used before definition"
                )
        elif isinstance(v, OpResult):
            if id(v) not in defined_results:
                raise IRVerificationError(
                    f"In @{fn_name}: OpResult used before definition"
                )

    def _verify_op(
        self,
        fn_name: str,
        op: Op,
        defined_names: set[str],
        defined_results: set[int],
    ) -> None:
        self._check_forbidden(fn_name, op)

        def chk(v: Value) -> None:
            self._chk_value(fn_name, v, defined_names, defined_results)

        if isinstance(op, CallOp):
            for v in _value_refs(op.args):
                chk(v)

        elif isinstance(op, CallDPSOp):
            for v in _value_refs(op.inputs):
                chk(v)
            if op.output is not None and op.output.name not in defined_names:
                raise IRVerificationError(
                    f"In @{fn_name}: CallDPS output '%{op.output.name}' used before definition"
                )

        elif isinstance(op, TupleOp):
            for v in _value_refs(op.elems):
                chk(v)

        elif isinstance(op, TupleGetItemOp):
            chk(op.tup)

        elif isinstance(op, ReturnOp):
            for v in _value_refs(op.values):
                chk(v)

        elif isinstance(op, YieldOp):
            for v in _value_refs(op.values):
                chk(v)

        elif isinstance(op, IfOp):
            self._verify_if_op(fn_name, op, defined_names, defined_results)

        elif isinstance(op, ForOp):
            self._verify_for_op(fn_name, op, defined_names, defined_results)

        elif isinstance(op, ShapeAssertOp):
            if op.tensor.name not in defined_names:
                raise IRVerificationError(
                    f"In @{fn_name}: ShapeAssert tensor '%{op.tensor.name}' used before definition"
                )

    # ------------------------------------------------------------------
    # IfOp
    # ------------------------------------------------------------------

    def _verify_if_op(
        self,
        fn_name: str,
        op: IfOp,
        defined_names: set[str],
        defined_results: set[int],
    ) -> None:
        def chk(v: Value) -> None:
            self._chk_value(fn_name, v, defined_names, defined_results)

        chk(op.cond)
        self._verify_region(fn_name, op.then_region, set(defined_names), set(defined_results), YieldOp)
        if op.else_region is not None:
            self._verify_region(fn_name, op.else_region, set(defined_names), set(defined_results), YieldOp)

        then_yield = _region_terminator(op.then_region)
        assert isinstance(then_yield, YieldOp)
        n = len(then_yield.values)

        if op.results and len(op.results) != n:
            raise IRVerificationError(
                f"In @{fn_name}: IfOp has {len(op.results)} results but "
                f"then_region yields {n} values"
            )
        if op.else_region is not None:
            else_yield = _region_terminator(op.else_region)
            assert isinstance(else_yield, YieldOp)
            if len(else_yield.values) != n:
                raise IRVerificationError(
                    f"In @{fn_name}: IfOp then_region yields {n} values "
                    f"but else_region yields {len(else_yield.values)}"
                )

    # ------------------------------------------------------------------
    # ForOp
    # ------------------------------------------------------------------

    def _verify_for_op(
        self,
        fn_name: str,
        op: ForOp,
        defined_names: set[str],
        defined_results: set[int],
    ) -> None:
        def chk(v: Value) -> None:
            self._chk_value(fn_name, v, defined_names, defined_results)

        for v in _value_refs((op.range_.start, op.range_.end, op.range_.step)):
            chk(v)
        for ia in op.iter_args:
            chk(ia.init)

        n = len(op.iter_args)
        body_yield = _region_terminator(op.body_region)
        assert isinstance(body_yield, YieldOp)
        if len(body_yield.values) != n:
            raise IRVerificationError(
                f"In @{fn_name}: ForOp has {n} iter_args but body yields "
                f"{len(body_yield.values)} values"
            )
        if op.results and len(op.results) != n:
            raise IRVerificationError(
                f"In @{fn_name}: ForOp has {len(op.results)} results but {n} iter_args"
            )

        body_names = set(defined_names)
        body_results = set(defined_results)
        body_names.add(op.loop_var.name)
        for ia in op.iter_args:
            if ia.var.name in body_names:
                raise IRVerificationError(
                    f"In @{fn_name}: ForOp iter_arg '%{ia.var.name}' shadows existing var"
                )
            body_names.add(ia.var.name)
        self._verify_region(fn_name, op.body_region, body_names, body_results, YieldOp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_forbidden(self, fn_name: str, op: Op) -> None:
        callee: Optional[str] = None
        if isinstance(op, CallOp):
            callee = op.callee
        elif isinstance(op, CallDPSOp):
            callee = op.callee
        if callee and callee in _FORBIDDEN_CALLEES:
            node_name = callee.lstrip("@")
            raise IRVerificationError(
                f"In @{fn_name}: {node_name} is forbidden in high-level IR "
                f"(use TensorCreateOp instead)"
            )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _value_refs(vals: tuple) -> Iterator[Value]:
    for v in vals:
        if isinstance(v, (Var, OpResult)):
            yield v


def _region_terminator(region: Region) -> TerminatorOp:
    block = region.entry_block
    assert block.ops
    last = block.ops[-1]
    assert isinstance(last, TerminatorOp)
    return last


def verify(module: IRModule) -> None:
    Verifier().verify_module(module)
