"""Kernel registry — two-level dispatch for op → KernelSpec matching.

Dispatch pipeline:
  Level 1 — exact dict lookup on (op_name, device, input_dtypes).  O(1).
             input_dtypes is a tuple with one entry per CallOp arg;
             non-tensor args contribute "" (empty string).
  Level 2 — linear scan over candidates sorted by priority (descending):
             a) SM arch filter: if spec.sm_arches is non-empty and sm_arch
                is provided, skip specs that don't include the target SM.
             b) Custom predicate: spec.match(call_op) for shape/attr checks.

launch_rule and attrs fields on CallOp are reserved for M11.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from devproc2.ir.ops import CallOp


@dataclass(frozen=True)
class KernelMatchKey:
    """Exact lookup key derived from a CallOp.

    op_name:      callee name without leading "@"
    device:       "cuda", "cpu", etc.
    input_dtypes: one dtype string per positional arg; "" for non-tensor args
    """
    op_name:      str
    device:       str
    input_dtypes: tuple[str, ...]


@dataclass
class KernelSpec:
    """Concrete kernel descriptor registered in KernelRegistry.

    op_name / device / input_dtypes must be canonical (exact, no wildcards).

    sm_arches: SM compute capabilities this kernel supports, e.g. (80, 90).
               Empty tuple means the kernel runs on any SM.
    match:     Optional predicate for shape/attr/custom second-level filtering.
               Receives the CallOp; return False to skip this spec.
               When call_op is None at lookup time, the predicate is skipped
               and the spec is treated as matching.
    """
    op_name:      str
    device:       str
    input_dtypes: tuple[str, ...]
    kernel_name:  str               # callee in CallDPSOp, e.g. "kernel.relu_fp16"
    sm_arches:    tuple[int, ...] = ()   # () = any SM
    priority:     int = 0
    match:        Optional[Callable[["CallOp"], bool]] = None


def build_input_dtypes(args: tuple) -> tuple[str, ...]:
    """Extract dtype from each arg's struct_info; '' for non-tensor args.

    Uses duck-typing (getattr) to avoid circular imports with the IR layer.
    """
    result = []
    for arg in args:
        si = getattr(arg, "struct_info", None)
        result.append(getattr(si, "dtype", "") if si is not None else "")
    return tuple(result)


# Internal dict key type alias
_DictKey = tuple[str, str, tuple[str, ...]]  # (op_name, device, input_dtypes)


class KernelRegistry:
    """Per-instance registry. Not thread-safe: concurrent register/lookup
    calls require external synchronisation.

    Specs are pre-sorted by priority (descending) on register() so that
    lookup() is a simple linear scan over the second-level candidates.
    """

    def __init__(self) -> None:
        self._specs: dict[_DictKey, list[KernelSpec]] = {}

    def register(self, spec: KernelSpec) -> None:
        key: _DictKey = (spec.op_name, spec.device, spec.input_dtypes)
        bucket = self._specs.setdefault(key, [])
        bucket.append(spec)
        bucket.sort(key=lambda s: s.priority, reverse=True)

    def lookup(
        self,
        key: KernelMatchKey,
        sm_arch: Optional[int] = None,
        call_op: Optional["CallOp"] = None,
    ) -> Optional[KernelSpec]:
        """Return the highest-priority KernelSpec that passes all filters.

        SM filter:  if spec.sm_arches is non-empty and sm_arch is given,
                    the spec is skipped unless sm_arch is in spec.sm_arches.
        Predicate:  if spec.match is non-None and call_op is given,
                    the spec is skipped unless spec.match(call_op) is True.
        """
        dict_key: _DictKey = (key.op_name, key.device, key.input_dtypes)
        for spec in self._specs.get(dict_key, []):
            if spec.sm_arches and sm_arch is not None:
                if sm_arch not in spec.sm_arches:
                    continue
            if spec.match is not None and call_op is not None:
                if not spec.match(call_op):
                    continue
            return spec
        return None
