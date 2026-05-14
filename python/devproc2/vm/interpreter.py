"""Pure-Python VMInterpreter for testing and prototyping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .executable import CalleeKind, Executable, Opcode


# ---------------------------------------------------------------------------
# Lightweight runtime objects (Python-side only, for testing without C++)
# ---------------------------------------------------------------------------

@dataclass
class _Storage:
    """Mock storage buffer."""
    data:        bytearray
    device_type: int
    device_id:   int

    @property
    def nbytes(self) -> int:
        return len(self.data)


@dataclass
class _Tensor:
    """Mock tensor backed by a _Storage."""
    storage:     _Storage
    offset:      int
    shape:       tuple[int, ...]
    dtype_code:  int   # DLPack code
    dtype_bits:  int
    dtype_lanes: int

    def __repr__(self) -> str:
        return (f"_Tensor(shape={self.shape}, "
                f"dtype=({self.dtype_code},{self.dtype_bits},{self.dtype_lanes}))")


# ---------------------------------------------------------------------------
# Builtin implementations
# ---------------------------------------------------------------------------

def _builtin_alloc_storage(args: list) -> _Storage:
    size_bytes, alignment, device_type, device_id = \
        int(args[0]), int(args[1]), int(args[2]), int(args[3])
    return _Storage(bytearray(size_bytes), device_type, device_id)


def _builtin_alloc_tensor(args: list) -> _Tensor:
    storage, offset, shape = args[0], int(args[1]), args[2]
    dtype_code, dtype_bits, dtype_lanes = int(args[3]), int(args[4]), int(args[5])
    if not isinstance(shape, tuple):
        shape = tuple(int(d) for d in shape)
    return _Tensor(storage, offset, tuple(int(d) for d in shape),
                   dtype_code, dtype_bits, dtype_lanes)


def _builtin_make_shape(args: list) -> tuple:
    return tuple(int(d) for d in args)


def _builtin_make_tuple(args: list) -> tuple:
    return tuple(args)


def _builtin_tuple_get_item(args: list) -> Any:
    return args[0][int(args[1])]


def _builtin_identity(args: list) -> Any:
    return args[0]


def _builtin_lt_i64(args: list) -> bool:
    return bool(int(args[0]) < int(args[1]))


def _builtin_add_i64(args: list) -> int:
    return int(args[0]) + int(args[1])


def _builtin_shape_assert(args: list) -> None:
    tensor, dim_idx, upper = args[0], int(args[1]), int(args[2])
    actual = tensor.shape[dim_idx]
    if actual > upper:
        raise RuntimeError(
            f"RuntimeShapeError: dim {dim_idx} = {actual} exceeds upper bound {upper}"
        )
    return None


_DEFAULT_BUILTINS: dict[str, Callable[[list], Any]] = {
    "vm.builtin.alloc_storage":  _builtin_alloc_storage,
    "vm.builtin.alloc_tensor":   _builtin_alloc_tensor,
    "vm.builtin.make_shape":     _builtin_make_shape,
    "vm.builtin.make_tuple":     _builtin_make_tuple,
    "vm.builtin.tuple_get_item": _builtin_tuple_get_item,
    "vm.builtin.identity":       _builtin_identity,
    "vm.builtin.lt_i64":         _builtin_lt_i64,
    "vm.builtin.add_i64":        _builtin_add_i64,
    "vm.builtin.shape_assert":   _builtin_shape_assert,
}


# ---------------------------------------------------------------------------
# VM Interpreter
# ---------------------------------------------------------------------------

class VMInterpreter:
    """Execute an Executable object.

    Maintains a flat register file that grows as frames are pushed and
    shrinks as they are popped.  Constants are pre-loaded into each
    function's register slice when the frame is set up.
    """

    def __init__(self, executable: Executable) -> None:
        self._exec = executable
        # Extra builtins / packed_funcs registered at runtime
        self._extra: dict[str, Callable[[list], Any]] = {}
        # Kernel mock registry: name → callable(args) → result
        self._kernels: dict[str, Callable[[list], Any]] = {}

    def register_packed_func(self, name: str, fn: Callable[[list], Any]) -> None:
        """Register a packed_func callable (for testing)."""
        self._extra[name] = fn

    def register_kernel(self, name: str, fn: Callable[[list], Any]) -> None:
        """Register a kernel mock (for testing)."""
        self._kernels[name] = fn

    def invoke(self, func_name: str, args: list) -> Any:
        """Execute func_name with the given argument list, return the result."""
        exec_ = self._exec
        func_idx = exec_.get_func_index(func_name)

        regs: list[Any] = []
        # (func_idx, pc, reg_base)
        frames: list[tuple[int, int, int]] = []
        # Return-value routing: (dst_reg, caller_reg_base) pushed for each vm_func CALL
        return_slots: list[tuple[int, int]] = []

        def push_frame(fidx: int, call_args: list) -> None:
            fe = exec_.function_table[fidx]
            base = len(regs)
            regs.extend([None] * fe.num_regs)
            # Copy args
            for i, a in enumerate(call_args):
                regs[base + i] = a
            # Apply const_inits
            for ci in fe.const_inits:
                regs[base + ci.reg_idx] = exec_.constants[ci.const_idx]
            frames.append((fidx, 0, base))

        push_frame(func_idx, args)

        while frames:
            fidx, pc, base = frames[-1]
            fe = exec_.function_table[fidx]
            instr = exec_.instructions[fe.instr_offset + pc]

            if instr.opcode == Opcode.CALL:
                callee = exec_.function_table[instr.func_idx]
                call_args = [regs[base + r] for r in instr.arg_regs]

                if callee.kind == CalleeKind.vm_func:
                    # Advance caller pc before pushing new frame
                    frames[-1] = (fidx, pc + 1, base)
                    return_slots.append((instr.dst_reg, base))
                    push_frame(instr.func_idx, call_args)
                    continue  # don't ++pc again
                else:
                    result = self._dispatch_external(callee, call_args)
                    if instr.dst_reg >= 0:
                        regs[base + instr.dst_reg] = result
                # fall through to ++pc

            elif instr.opcode == Opcode.RET:
                result = regs[base + instr.src_reg] if instr.src_reg >= 0 else None
                # Shrink register file back to caller's extent
                del regs[base:]
                frames.pop()
                if frames:
                    # Write return value into caller's dst_reg
                    dst_reg, caller_base = return_slots.pop()
                    if dst_reg >= 0:
                        regs[caller_base + dst_reg] = result
                    # pc was already advanced when vm_func CALL was processed
                    continue
                else:
                    # Top-level return
                    return result

            elif instr.opcode == Opcode.IF:
                cond = bool(regs[base + instr.cond_reg])
                jump = instr.true_offset if cond else instr.false_offset
                frames[-1] = (fidx, pc + jump, base)
                continue  # no ++pc

            elif instr.opcode == Opcode.GOTO:
                frames[-1] = (fidx, pc + instr.offset, base)
                continue  # no ++pc

            # Advance pc
            frames[-1] = (fidx, pc + 1, base)

        return None  # should not normally reach here

    def _dispatch_external(self, callee, call_args: list) -> Any:
        name = callee.name
        if callee.kind == CalleeKind.builtin:
            fn = _DEFAULT_BUILTINS.get(name) or self._extra.get(name)
            if fn is None:
                raise RuntimeError(f"Unknown builtin: {name!r}")
            return fn(call_args)

        if callee.kind == CalleeKind.packed_func:
            fn = self._extra.get(name)
            if fn is None:
                raise RuntimeError(f"PackedFunc '{name}' not registered")
            return fn(call_args)

        if callee.kind == CalleeKind.kernel:
            fn = self._kernels.get(name)
            if fn is not None:
                return fn(call_args)
            return None  # M8: kernel is a no-op stub

        raise RuntimeError(f"Unknown callee kind: {callee.kind}")
