"""Executable ↔ binary serialization."""
from __future__ import annotations

import struct
from .executable import (
    CalleeKind,
    ConstInit,
    Executable,
    FunctionEntry,
    Instruction,
    Opcode,
)

_MAGIC = b"DV2E"
_VERSION = 1

_TAG_NULL  = 0
_TAG_INT   = 1
_TAG_FLOAT = 2
_TAG_BOOL  = 3
_TAG_STR   = 4


def serialize(exe: Executable) -> bytes:
    buf = bytearray()
    buf += _MAGIC
    buf += struct.pack("<III", _VERSION, len(exe.function_table), len(exe.instructions))
    buf += struct.pack("<I", len(exe.constants))

    for fe in exe.function_table:
        name_b = fe.name.encode()
        buf += struct.pack("<I", len(name_b)) + name_b
        buf += struct.pack("<Biiiii", int(fe.kind),
                           fe.instr_offset, fe.instr_count, fe.num_regs,
                           fe.num_args, len(fe.const_inits))
        for ci in fe.const_inits:
            buf += struct.pack("<ii", ci.reg_idx, ci.const_idx)

    for instr in exe.instructions:
        buf += struct.pack(
            "<BiiiiiiiI",
            int(instr.opcode),
            instr.dst_reg, instr.func_idx,
            instr.src_reg,
            instr.cond_reg, instr.true_offset, instr.false_offset,
            instr.offset,
            len(instr.arg_regs),
        )
        for r in instr.arg_regs:
            buf += struct.pack("<i", r)

    for c in exe.constants:
        if c is None:
            buf += struct.pack("<B8x", _TAG_NULL)
        elif isinstance(c, bool):
            buf += struct.pack("<Bq", _TAG_BOOL, int(c))
        elif isinstance(c, int):
            buf += struct.pack("<Bq", _TAG_INT, c)
        elif isinstance(c, float):
            buf += struct.pack("<Bd", _TAG_FLOAT, c)
        elif isinstance(c, str):
            s_b = c.encode()
            buf += struct.pack("<BI", _TAG_STR, len(s_b)) + s_b
        else:
            raise TypeError(f"Cannot serialize constant: {c!r}")

    return bytes(buf)


def deserialize(data: bytes) -> Executable:
    pos = 0

    def read(fmt: str):
        nonlocal pos
        result = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return result

    magic = data[:4]; pos = 4
    if magic != _MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    (version, num_funcs, num_instrs) = read("<III")
    if version != _VERSION:
        raise ValueError(f"Version mismatch: expected {_VERSION}, got {version}")
    (num_consts,) = read("<I")

    function_table = []
    for _ in range(num_funcs):
        (nlen,) = read("<I")
        name = data[pos:pos+nlen].decode(); pos += nlen
        (kind_b, ioff, icnt, nregs, nargs, n_ci) = read("<Biiiii")
        const_inits = [ConstInit(*read("<ii")) for _ in range(n_ci)]
        function_table.append(FunctionEntry(
            name=name, kind=CalleeKind(kind_b),
            instr_offset=ioff, instr_count=icnt,
            num_regs=nregs, num_args=nargs,
            const_inits=const_inits,
        ))

    instructions = []
    for _ in range(num_instrs):
        (op, dst, fidx, src, cond, to, fo, off, nargs) = read("<BiiiiiiiI")
        arg_regs = list(read(f"<{nargs}i")) if nargs else []
        instructions.append(Instruction(
            opcode=Opcode(op),
            dst_reg=dst, func_idx=fidx,
            src_reg=src,
            cond_reg=cond, true_offset=to, false_offset=fo,
            offset=off,
            arg_regs=arg_regs,
        ))

    constants = []
    for _ in range(num_consts):
        (tag,) = read("<B")
        if tag == _TAG_NULL:
            read("8x"); constants.append(None)
        elif tag == _TAG_INT:
            (v,) = read("<q"); constants.append(v)
        elif tag == _TAG_FLOAT:
            (v,) = read("<d"); constants.append(v)
        elif tag == _TAG_BOOL:
            (v,) = read("<q"); constants.append(bool(v))
        elif tag == _TAG_STR:
            (slen,) = read("<I")
            s = data[pos:pos+slen].decode(); pos += slen
            constants.append(s)
        else:
            raise ValueError(f"Unknown constant tag: {tag}")

    return Executable(
        function_table=function_table,
        instructions=instructions,
        constants=constants,
    )
