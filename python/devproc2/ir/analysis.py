"""Shared semantic analyses over the lightweight IR."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from devproc2.ir.nodes import Op, OpResult, Region, Value
from devproc2.ir.ops import ForOp, IfOp, TupleGetItemOp, TupleOp, YieldOp


def iter_ops(region: Region) -> Iterator[Op]:
    """Yield every op in a region tree in deterministic DFS order."""
    for block in region.blocks:
        for op in block.ops:
            yield op
            for sub_region in op.regions:
                yield from iter_ops(sub_region)


class AliasAnalysis:
    """Transitive forwarding analysis for SSA values.

    This captures IR-level value forwarding, not physical storage aliasing:
    tuple construction/extraction and structured-control-flow yields can forward
    tensor objects to another SSA result.  Memory planning, escape analysis, and
    later DCE/CSE passes can consume the same source of truth.
    """

    def __init__(self, ops: Iterable[Op]) -> None:
        self.forward_edges: dict[OpResult, tuple[Value, ...]] = {}
        self._build(tuple(ops))

    @classmethod
    def from_region(cls, region: Region) -> "AliasAnalysis":
        return cls(iter_ops(region))

    def sources(self, value: Value) -> tuple[Value, ...]:
        if isinstance(value, OpResult):
            return self.forward_edges.get(value, ())
        return ()

    def resolve_matching(
        self,
        value: Value,
        predicate: Callable[[Value], bool],
    ) -> frozenset[Value]:
        """Resolve ``value`` through forwarding edges and return matching leaves."""
        return frozenset(self._resolve_matching(value, predicate, set()))

    def _build(self, ops: tuple[Op, ...]) -> None:
        for op in ops:
            if isinstance(op, TupleOp):
                self.forward_edges[op.results[0]] = op.elems

            elif isinstance(op, TupleGetItemOp):
                self.forward_edges[op.results[0]] = self._project_tuple_item(
                    op.tup, op.index, set()
                )

            elif isinstance(op, IfOp):
                branch_yields = [_region_yield(op.then_region)]
                if op.else_region is not None:
                    branch_yields.append(_region_yield(op.else_region))
                for result in op.results:
                    self.forward_edges[result] = tuple(
                        y.values[result.index]
                        for y in branch_yields
                        if result.index < len(y.values)
                    )

            elif isinstance(op, ForOp):
                body_yield = _region_yield(op.body_region)
                for result in op.results:
                    values: list[Value] = []
                    if result.index < len(op.iter_args):
                        values.append(op.iter_args[result.index].init)
                    if result.index < len(body_yield.values):
                        values.append(body_yield.values[result.index])
                    self.forward_edges[result] = tuple(values)

    def _project_tuple_item(
        self,
        value: Value,
        index: int,
        visiting: set[int],
    ) -> tuple[Value, ...]:
        """Return possible sources for ``value[index]`` through forwarding edges."""
        if not isinstance(value, OpResult):
            return ()

        value_id = id(value)
        if value_id in visiting:
            return ()
        visiting.add(value_id)

        if isinstance(value.op, TupleOp):
            if 0 <= index < len(value.op.elems):
                visiting.remove(value_id)
                return (value.op.elems[index],)
            visiting.remove(value_id)
            return ()

        projected: list[Value] = []
        for source in self.forward_edges.get(value, ()):
            projected.extend(self._project_tuple_item(source, index, visiting))
        visiting.remove(value_id)
        return tuple(projected)

    def _resolve_matching(
        self,
        value: Value,
        predicate: Callable[[Value], bool],
        visiting: set[int],
    ) -> set[Value]:
        if predicate(value):
            return {value}
        if not isinstance(value, OpResult):
            return set()

        value_id = id(value)
        if value_id in visiting:
            return set()
        visiting.add(value_id)

        resolved: set[Value] = set()
        for source in self.forward_edges.get(value, ()):
            resolved.update(self._resolve_matching(source, predicate, visiting))
        visiting.remove(value_id)
        return resolved


def _region_yield(region: Region) -> YieldOp:
    term = region.entry_block.ops[-1]
    if not isinstance(term, YieldOp):
        raise ValueError(
            f"expected YieldOp terminator in region, got {type(term).__name__}"
        )
    return term


__all__ = ["AliasAnalysis", "iter_ops"]
