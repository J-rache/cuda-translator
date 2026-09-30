"""Reference interpreter for House IR (debug oracle, NOT the final proof).

Executes translated kernels on the CPU with an explicit work-item loop so
translated logic can be checked independently of any GPU. The acceptance
oracle remains an independent CPU reference of the ORIGINAL algorithm
compared against real device execution (see scripts/verify_numeric.py);
this interpreter exists to isolate frontend/backend faults.

Semantics mirror the OpenCL backend: unsigned storage math with explicit
reinterpretation for signed ops, per-instruction interpretation widths.
"""
from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional, Tuple

from .ir import Instruction, Kernel

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def _s32(v: int) -> int:
    v &= MASK32
    return v - 0x100000000 if v & 0x80000000 else v


def _u32(v: int) -> int:
    return v & MASK32


def _s64(v: int) -> int:
    v &= MASK64
    return v - 0x10000000000000000 if v & 0x8000000000000000 else v


def _u64(v: int) -> int:
    return v & MASK64


def _f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


_CMP = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
    "ltu": lambda a, b: _u32(a) < _u32(b), "leu": lambda a, b: _u32(a) <= _u32(b),
    "gtu": lambda a, b: _u32(a) > _u32(b), "geu": lambda a, b: _u32(a) >= _u32(b),
    "equ": lambda a, b: _u32(a) == _u32(b), "neu": lambda a, b: _u32(a) != _u32(b),
}


class WorkItemState:
    """One work-item's register file + program counter."""

    def __init__(self, kernel: Kernel, gid: Tuple[int, int, int],
                 lid: Tuple[int, int, int], ls: Tuple[int, int, int]):
        self.regs: Dict[str, Any] = {name: 0 for name in kernel.registers}
        self.pred_inv: Dict[str, bool] = {}
        self.pc = 0
        self.done = False
        self.gid, self.lid, self.ls = gid, lid, ls
        self.spec = {
            "local_id_x": lid[0], "local_id_y": lid[1], "local_id_z": lid[2],
            "group_id_x": gid[0], "group_id_y": gid[1], "group_id_z": gid[2],
            "local_size_x": ls[0], "local_size_y": ls[1], "local_size_z": ls[2],
        }


def _mem_width_bytes(width: str) -> int:
    return {"f32": 4, "f64": 8, "u32": 4, "s32": 4, "b32": 4,
            "u64": 8, "s64": 8, "b64": 8}[width]


def _load_mem(buf: bytearray, byte_addr: int, width: str) -> Any:
    n = _mem_width_bytes(width)
    raw = bytes(buf[byte_addr:byte_addr + n])
    if width == "f32":
        return struct.unpack("<f", raw)[0]
    if width == "f64":
        return struct.unpack("<d", raw)[0]
    if width in ("u32", "b32"):
        return struct.unpack("<I", raw)[0]
    if width == "s32":
        return struct.unpack("<i", raw)[0]
    if width in ("u64", "b64"):
        return struct.unpack("<Q", raw)[0]
    if width == "s64":
        return struct.unpack("<q", raw)[0]
    raise ValueError(f"width {width}")


def _store_mem(buf: bytearray, byte_addr: int, width: str, value: Any) -> None:
    n = _mem_width_bytes(width)
    if width == "f32":
        raw = struct.pack("<f", value)
    elif width == "f64":
        raw = struct.pack("<d", value)
    elif width in ("u32", "b32"):
        raw = struct.pack("<I", value & MASK32)
    elif width == "s32":
        raw = struct.pack("<i", _s32(int(value)))
    elif width in ("u64", "b64"):
        raw = struct.pack("<Q", value & MASK64)
    elif width == "s64":
        raw = struct.pack("<q", _s64(int(value)))
    else:
        raise ValueError(f"width {width}")
    buf[byte_addr:byte_addr + n] = raw


def run_kernel(kernel: Kernel, args: Dict[str, Any], global_size: Tuple[int, int, int],
               local_size: Tuple[int, int, int], buffers: Dict[str, bytearray],
               max_steps: int = 10_000_000) -> None:
    """Run one kernel over the NDRange. args: scalar params; buffers: param->mem.

    Buffers are mutated in place (outputs). Raises on unsupported ops
    (fail closed, mirroring the backend).
    """
    def _dim3(t):
        t = tuple(t) + (1,) * (3 - len(t))
        return t[:3]

    gs = _dim3(global_size)
    ls = _dim3(local_size)
    for d in range(3):
        if ls[d] == 0 or gs[d] % ls[d] != 0:
            raise ValueError(
                f"illegal NDRange: global_size[{d}]={gs[d]} not divisible by "
                f"local_size[{d}]={ls[d]} (OpenCL requirement)")
    n_groups = (gs[0] // ls[0], gs[1] // ls[1], gs[2] // ls[2])
    for gz in range(n_groups[2]):
        for gy in range(n_groups[1]):
            for gx in range(n_groups[0]):
                for lz in range(ls[2]):
                    for ly in range(ls[1]):
                        for lx in range(ls[0]):
                            _run_one(kernel, args, (gx, gy, gz), (lx, ly, lz),
                                     ls, buffers, max_steps)


def _run_one(kernel: Kernel, args: Dict[str, Any], gid, lid, ls,
             buffers: Dict[str, bytearray], max_steps: int) -> None:
    st = WorkItemState(kernel, gid, lid, ls)
    steps = 0
    while not st.done:
        if st.pc >= len(kernel.body):
            return
        if steps > max_steps:
            raise RuntimeError(f"kernel {kernel.name}: step limit exceeded")
        steps += 1
        inst = kernel.body[st.pc]
        st.pc += 1
        if inst.pred is not None:
            pv = st.regs.get(inst.pred.name, 0)
            if (not pv) if not inst.pred_inv else bool(pv):
                continue
        _exec(kernel, inst, st, args, buffers)


def _exec(kernel: Kernel, inst: Instruction, st: WorkItemState,
          args: Dict[str, Any], buffers: Dict[str, bytearray]) -> None:
    op = inst.op
    if op == "unsupported":
        raise RuntimeError(
            f"kernel {kernel.name}: unsupported opcode at PTX line {inst.line}")

    def reg(name: str):
        return st.regs[name]

    def setreg(name: str, v):
        st.regs[name] = v

    def srcval(o):
        if o.kind == "imm":
            return int(o.name) if "." not in o.name and "e" not in o.name \
                else float(o.name)
        if o.kind == "special":
            return st.spec[o.name]
        if o.kind == "neg":
            return -srcval_o(o.name)
        if o.kind == "not":
            return 0 if srcval_o(o.name) else 1
        return reg(o.name)

    def srcval_o(name):  # inner operand of neg/not
        return reg(name) if name in st.regs else int(name, 0)

    if op == "ret":
        st.done = True
        return
    if op == "bra":
        target = inst.label
        taken = True
        if inst.pred is not None:
            pv = bool(reg(inst.pred.name))
            taken = (not pv) if inst.pred_inv else pv
        if taken:
            st.pc = kernel.labels[target]
        return
    if op == "ld_param":
        pname = inst.srcs[0].name.split("[")[0]
        setreg(inst.dests[0].name, args.get(pname, 0))
        return
    if op == "mov":
        setreg(inst.dests[0].name, srcval(inst.srcs[0]))
        return
    if op == "cvta_global":
        setreg(inst.dests[0].name, srcval(inst.srcs[0]))
        return
    if op in ("ld_global", "st_global"):
        a = inst.addr
        if a is None or a.base_param is None:
            raise RuntimeError(f"kernel {kernel.name}: unresolved address")
        width = (inst.width or "b32")
        byte = a.const + (a.scale or 0) * (reg(a.index.name) if a.index else 0)
        if a.index is not None and width in ("f32", "f64"):
            pass
        buf = buffers[a.base_param]
        if op == "ld_global":
            setreg(inst.dests[0].name, _load_mem(buf, byte, width))
        else:
            _store_mem(buf, byte, width, srcval(inst.srcs[0]))
        return

    if op == "setp":
        cmp_op = inst.mods[0] if inst.mods else "eq"
        width = inst.width or "s32"
        a, b = srcval(inst.srcs[0]), srcval(inst.srcs[1])
        if width in ("f32",):
            a, b = _f32(a), _f32(b)
        elif width in ("s32",):
            a, b = _s32(a), _s32(b)
        elif width in ("s64",):
            a, b = _s64(a), _s64(b)
        elif width in ("u32", "b32"):
            a, b = _u32(a), _u32(b)
        elif width in ("u64", "b64"):
            a, b = _u64(a), _u64(b)
        fn = _CMP.get(cmp_op)
        if fn is None:
            raise RuntimeError(f"setp {cmp_op} unsupported")
        setreg(inst.dests[0].name, 1 if fn(a, b) else 0)
        return

    if op in ("add", "sub", "mul", "mad", "fma"):
        width = inst.width or "b64"
        vals = [srcval(s) for s in inst.srcs]
        if op == "fma":
            r = _f32(_f32(vals[0]) * _f32(vals[1]) + _f32(vals[2])) \
                if width == "f32" else vals[0] * vals[1] + vals[2]
            setreg(inst.dests[0].name, r)
            return
        if op == "mul" and "wide" in inst.mods:
            a = _s32(vals[0]) if width.startswith("s") else _u32(vals[0])
            b = _s32(vals[1]) if width.startswith("s") else _u32(vals[1])
            setreg(inst.dests[0].name, _s64(a * b) if width.startswith("s")
                   else _u64(a * b))
            return
        if op == "mad":
            if width in ("s32", "u32", "b32"):
                r = (_u32(vals[0]) * _u32(vals[1]) + _u32(vals[2])) & MASK32
                setreg(inst.dests[0].name, _s32(r) if width == "s32" else r)
            else:
                r = (_u64(vals[0]) * _u64(vals[1]) + _u64(vals[2])) & MASK64
                setreg(inst.dests[0].name, _s64(r) if width == "s64" else r)
            return
        if width in ("f32", "f64"):
            a, b = _f32(vals[0]), _f32(vals[1]) if width == "f32" else (vals[0], vals[1])
            r = {("add", "f32"): lambda: _f32(a + b),
                 ("sub", "f32"): lambda: _f32(a - b),
                 ("mul", "f32"): lambda: _f32(a * b)}.get((op, width))
            if r is None:
                r = (a + b) if op == "add" else (a - b) if op == "sub" else a * b
            setreg(inst.dests[0].name, r())
            return
        if width in ("s32",):
            a, b = _s32(vals[0]), _s32(vals[1])
            r = (a + b) if op == "add" else (a - b) if op == "sub" else a * b
            setreg(inst.dests[0].name, _s32(r))
            return
        if width in ("u32", "b32"):
            a, b = _u32(vals[0]), _u32(vals[1])
            r = (a + b) if op == "add" else (a - b) if op == "sub" else a * b
            setreg(inst.dests[0].name, _u32(r))
            return
        if width in ("s64",):
            a, b = _s64(vals[0]), _s64(vals[1])
            r = (a + b) if op == "add" else (a - b) if op == "sub" else a * b
            setreg(inst.dests[0].name, _s64(r))
            return
        a, b = _u64(vals[0]), _u64(vals[1])
        r = (a + b) if op == "add" else (a - b) if op == "sub" else a * b
        setreg(inst.dests[0].name, _u64(r))
        return

    if op == "cvt":
        width = inst.width or "b32"
        v = srcval(inst.srcs[0])
        if width == "f32":
            setreg(inst.dests[0].name, _f32(v))
        elif width == "f64":
            setreg(inst.dests[0].name, float(v))
        elif width in ("s32",):
            setreg(inst.dests[0].name, _s32(int(v)))
        elif width in ("u32", "b32"):
            setreg(inst.dests[0].name, _u32(int(v)))
        else:
            setreg(inst.dests[0].name, int(v))
        return

    raise RuntimeError(f"kernel {kernel.name}: interpreter opcode {op} not implemented")
