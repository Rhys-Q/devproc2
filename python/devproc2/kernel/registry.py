"""Kernel registry — maps op_name + device + dtype to KernelSpec.

Dispatch pipeline (from docs/design/kernel_register.md):
  1. op_name filter
  2. device filter  ("*" on either side is a wildcard)
  3. dtype filter   ("*" on either side is a wildcard)
  4. match(call_op) predicate if provided
  5. highest priority wins

launch_rule and attrs fields are reserved for M11 (Triton grid computation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from devproc2.ir.ops import CallOp


@dataclass(frozen=True)
class KernelMatchKey:
    """Lookup key built from a CallOp's op_name, device, and dtype.

    Use "*" for device or dtype to match any registered spec.
    """
    op_name: str   # "layernorm", "relu", etc. — no leading "@"
    device:  str   # "cuda", "cpu", or "*"
    dtype:   str   # "float16", "float32", or "*"


@dataclass
class KernelSpec:
    """Describes a concrete kernel implementation for one op variant.

    op_name / device / dtype must be canonical (no wildcards).
    kernel_name becomes the callee in CallDPSOp after DPS lowering.
    match, if provided, is a predicate that can inspect the CallOp for
    attrs or shape constraints (used in M11 for Triton specializations).
    """
    op_name:     str
    device:      str
    dtype:       str
    kernel_name: str                                    # e.g. "kernel.relu_fp16"
    priority:    int = 0
    match:       Optional[Callable[["CallOp"], bool]] = None


def _matches(spec: KernelSpec, key: KernelMatchKey) -> bool:
    def compat(a: str, b: str) -> bool:
        return a == "*" or b == "*" or a == b

    return (
        compat(spec.op_name, key.op_name)
        and compat(spec.device, key.device)
        and compat(spec.dtype, key.dtype)
    )


class KernelRegistry:
    """Per-instance kernel registry. Not thread-safe: concurrent register/lookup
    calls require external synchronisation.

    Specs are stored per op_name and sorted by priority (descending)
    on every register() call so lookup() is a simple linear scan.
    """

    def __init__(self) -> None:
        self._specs: dict[str, list[KernelSpec]] = {}

    def register(self, spec: KernelSpec) -> None:
        bucket = self._specs.setdefault(spec.op_name, [])
        bucket.append(spec)
        bucket.sort(key=lambda s: s.priority, reverse=True)

    def lookup(
        self,
        key: KernelMatchKey,
        call_op: Optional["CallOp"] = None,
    ) -> Optional[KernelSpec]:
        """Return the highest-priority KernelSpec for key, or None.

        If call_op is None and a spec has a match predicate, the predicate is
        skipped and the spec is treated as matching. Pass the actual CallOp
        whenever available so that predicates can inspect the call.
        """
        candidates = self._specs.get(key.op_name, []) + self._specs.get("*", [])
        for spec in candidates:
            if not _matches(spec, key):
                continue
            if spec.match is not None and call_op is not None:
                if not spec.match(call_op):
                    continue
            return spec
        return None
