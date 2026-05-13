"""VM data structures: Opcode, Instruction, FunctionEntry, Executable."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Opcode(IntEnum):
    CALL = 0
    RET  = 1
    IF   = 2
    GOTO = 3


class CalleeKind(IntEnum):
    vm_func     = 0
    builtin     = 1
    packed_func = 2
    kernel      = 3


@dataclass
class Instruction:
    """One VM instruction. Fields unused by the opcode are ignored."""
    opcode: Opcode

    # CALL operands
    dst_reg:  int = -1        # -1 = no return value
    func_idx: int = 0
    arg_regs: list[int] = field(default_factory=list)

    # RET operands
    src_reg: int = -1         # -1 = void return

    # IF operands (pc-relative, no ++pc after execute)
    cond_reg:     int = 0
    true_offset:  int = 0
    false_offset: int = 0

    # GOTO operands (pc-relative, no ++pc after execute)
    offset: int = 0


@dataclass
class ConstInit:
    """Pre-loads constants[const_idx] into regs[reg_idx] at frame setup."""
    reg_idx:   int
    const_idx: int


@dataclass
class FunctionEntry:
    """Entry in Executable.function_table."""
    name:         str
    kind:         CalleeKind
    instr_offset: int          # index into Executable.instructions
    instr_count:  int
    num_regs:     int          # total registers for this function
    num_args:     int          # first num_args registers receive call args
    const_inits:  list[ConstInit] = field(default_factory=list)


@dataclass
class Executable:
    """Immutable VM program: function table + flat instruction array + constants."""
    function_table: list[FunctionEntry] = field(default_factory=list)
    instructions:   list[Instruction]   = field(default_factory=list)
    constants:      list[Any]           = field(default_factory=list)
    # constants: Python scalars (int, float, bool, None) or tuples for shapes

    def get_func_index(self, name: str) -> int:
        for i, fe in enumerate(self.function_table):
            if fe.name == name:
                return i
        raise KeyError(f"Function '{name}' not found in Executable")
