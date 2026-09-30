"""OpenCL C backend: lower House IR to OpenCL C kernel source.

Semantic rules (see ir.py):

* Storage vs interpretation: registers are emitted as their declared storage
  type; every use casts to the instruction's interpretation width.
* Explicit wraparound: signed .lo/.hi arithmetic computes in unsigned storage
  and reinterprets (never C signed overflow, which is UB):
      mad.lo.s32 -> (int)((uint)a*(uint)b + (uint)c)
      add.s64    -> (long)((ulong)a + (ulong)b)
  mul.wide.s32 sign-extends: (long)((long)(int)a * (long)(int)b).
* Memory: loads/stores address the __global parameter buffer DIRECTLY through
  the resolved Address (base param + scale*index + const). Pointer params are
  emitted as typed element pointers so offset math scales once, by element.
* Fail closed: TranslationAbort for reachable unsupported ops and for control
  flow the structured lowering cannot prove (IR rule 1).
* Control flow: only the NVIDIA guard idiom is lowered
  (setp / @p bra END / body / END: ret), using a real goto at the label's
  body position.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ._meta import TranslationError
from .ir import (
    Address,
    Instruction,
    Kernel,
    Operand,
    reachable_unsupported,
)

STORAGE_C = {
    "b8": "uchar", "u8": "uchar", "s8": "char",
    "b16": "ushort", "u16": "ushort", "s16": "short",
    "b32": "uint", "u32": "uint", "s32": "int",
    "b64": "ulong", "u64": "ulong", "s64": "long",
    "f32": "float", "f64": "double", "pred": "uchar",
}

INTERP_C = {
    "u32": "uint", "s32": "int", "b32": "uint",
    "u64": "ulong", "s64": "long", "b64": "ulong",
    "f32": "float", "f64": "double",
    "u16": "ushort", "s16": "short", "b16": "ushort",
    "u8": "uchar", "s8": "char", "b8": "uchar",
    "pred": "uchar",
}

SPECIAL_OCL = {
    "local_id_x": "get_local_id(0)",
    "local_id_y": "get_local_id(1)",
    "local_id_z": "get_local_id(2)",
    "group_id_x": "get_group_id(0)",
    "group_id_y": "get_group_id(1)",
    "group_id_z": "get_group_id(2)",
    "local_size_x": "get_local_size(0)",
    "local_size_y": "get_local_size(1)",
    "local_size_z": "get_local_size(2)",
    "num_groups_x": "get_num_groups(0)",
    "num_groups_y": "get_num_groups(1)",
    "num_groups_z": "get_num_groups(2)",
}

_CMP_C = {
    "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    "equ": "==", "neu": "!=", "ltu": "<", "leu": "<=",
    "gtu": ">", "geu": ">=",
}

_MEM_TYPES = ("f32", "f64", "u32", "s32", "b32", "u64", "s64", "b64")


class TranslationAbort(TranslationError):
    """Raised when the backend cannot prove a lowering (fail closed)."""


def _fail(kernel: Kernel, inst, reason: str):
    line = inst.line if inst else 0
    op = inst.op if inst else ""
    loc = f" (opcode {op}, PTX line {line})" if inst else ""
    raise TranslationAbort(f"kernel {kernel.name}: {reason}{loc}",
                           kernel=kernel.name, opcode=op, line=line or 0)


def _storage(kernel: Kernel, op: Operand) -> str:
    dt = op.dtype or kernel.reg_dtype(op.name) or "b64"
    return STORAGE_C.get(dt, "ulong")


def _interp(kernel: Kernel, inst: Instruction) -> str:
    return INTERP_C.get(inst.width or "b64", "ulong")


def _cast(t: str, expr: str) -> str:
    return f"(({t})({expr}))"


class _Emitter:
    def __init__(self, kernel: Kernel):
        self.k = kernel
        self.elem_of: Dict[str, str] = {}  # param name -> element C type

    def cname(self, reg: str) -> str:
        return "r_" + re.sub(r"[^A-Za-z0-9]", "_", reg)

    def pname(self, param: str) -> str:
        return "p_" + re.sub(r"[^A-Za-z0-9]", "_", param)

    def label(self, lab: str) -> str:
        return "L" + re.sub(r"[^A-Za-z0-9]", "_", lab)

    def special(self, name: str) -> str:
        return SPECIAL_OCL.get(name, name)

    def assign(self, reg: str, storage: str, expr: str) -> str:
        return f"{self.cname(reg)} = ({storage})({expr});"

    # ---- operand rendering
    def val(self, inst: Instruction, o: Operand, want: Optional[str] = None) -> str:
        """C expression for operand, cast to interpretation width."""
        want = want or _interp(self.k, inst)
        if o.kind == "imm":
            return _cast(want, o.name)
        if o.kind == "special":
            return _cast(want, self.special(o.name))
        if o.kind in ("neg", "not"):
            inner = Operand(kind="reg", name=o.name, dtype=o.dtype)
            u = "-" if o.kind == "neg" else "!"
            return _cast(want, u + self.val(inst, inner, want))
        if o.kind == "mem":
            _fail(self.k, inst, f"unresolved memory operand {o.name!r}")
        if o.kind in ("label", "sym"):
            _fail(self.k, inst, f"unexpected symbolic operand {o.name!r}")
        return _cast(want, self.cname(o.name))

    def elem_ptr(self, a: Address, const: bool = True) -> str:
        """Pointer to the addressed element via BYTE arithmetic.

        Address.scale is a BYTE stride, so the typed param pointer is cast to
        __global uchar* (1-byte steps) before adding scale*index+const, then
        cast back to the element type. Doing p + r*scale directly on a typed
        pointer would scale TWICE (C pointer arithmetic multiplies by sizeof
        element), producing 16-byte strides for float buffers.
        """
        if a.base_param is None:
            _fail(self.k, None, f"unresolved address (raw_reg={a.raw_reg!r})")
        etype = self.elem_of.get(a.base_param, "uchar")
        base = self.pname(a.base_param)  # the __global param pointer itself
        if a.index is None:
            off = str(a.const)
        else:
            off = f"(size_t)({self.cname(a.index.name)}) * ({a.scale})"
            if a.const:
                off += f" + ({a.const})"
        kw = "const " if const else ""
        return (f"((__global {etype} {kw}*)"
                f"(((__global uchar {kw}*)({base})) + ({off})))")

    # ---- statements
    def stmt(self, inst: Instruction) -> List[str]:
        op = inst.op
        if op == "unsupported":
            _fail(self.k, inst, "reachable unsupported instruction")
        if op == "mov":
            d, s = inst.dests[0], inst.srcs[0]
            return [self.assign(d.name, _storage(self.k, d), self.val(inst, s))]
        if op == "ld_param":
            d, s = inst.dests[0], inst.srcs[0]
            pname = (s.name.split("[")[0] if s.kind == "mem" else s.name)
            return [self.assign(d.name, _storage(self.k, d),
                                f"({self.pname(pname)})")]
        if op == "ld_global":
            d = inst.dests[0]
            a = inst.addr
            if a is None:
                _fail(self.k, inst, "ld_global without resolved address")
            dt = inst.width or "b32"
            if dt not in _MEM_TYPES:
                _fail(self.k, inst, f"ld_global width {dt} not lowered")
            return [self.assign(d.name, _storage(self.k, d),
                                f"*{self.elem_ptr(a)}")]
        if op == "st_global":
            a = inst.addr
            if a is None:
                _fail(self.k, inst, "st_global without resolved address")
            dt = inst.width or "b32"
            if dt not in _MEM_TYPES:
                _fail(self.k, inst, f"st_global width {dt} not lowered")
            etype = STORAGE_C[dt]
            v = self.val(inst, inst.srcs[0], etype)
            return [f"*{self.elem_ptr(a, const=False)} = {v};"]
        if op in ("add", "sub", "mul", "mad", "fma"):
            return self._arith(inst)
        if op == "setp":
            return [self._setp(inst)]
        if op == "cvta_global":
            d, s = inst.dests[0], inst.srcs[0]
            return [self.assign(d.name, _storage(self.k, d),
                                self.val(inst, s, "ulong"))]
        if op == "cvt":
            d, s = inst.dests[0], inst.srcs[0]
            dt = INTERP_C.get(inst.width or "b32", "uint")
            return [self.assign(d.name, _storage(self.k, d),
                                _cast(dt, self.val(inst, s)))]
        if op in ("and", "or", "xor", "shl", "shr", "min", "max", "selp",
                  "not", "neg", "abs"):
            return self._bitwise(inst)
        if op == "ret":
            return ["return;"]
        if op == "bra":
            return []  # lowered at program level
        _fail(self.k, inst, f"opcode {op} has no lowering")
        return []

    def _setp(self, inst: Instruction) -> str:
        cmp_op = inst.mods[0] if inst.mods else "eq"
        cc = _CMP_C.get(cmp_op)
        if cc is None:
            _fail(self.k, inst, f"setp comparison {cmp_op!r} not lowered")
        d, a, b = inst.dests[0], inst.srcs[0], inst.srcs[1]
        it = _interp(self.k, inst)
        if it not in ("int", "uint", "long", "ulong", "float", "double"):
            it = "int"
        if cmp_op.endswith("u") and it in ("int", "long"):
            uit = "uint" if it == "int" else "ulong"
            a_v, b_v = self.val(inst, a, uit), self.val(inst, b, uit)
        else:
            a_v, b_v = self.val(inst, a, it), self.val(inst, b, it)
        return self.assign(d.name, "uchar", f"({a_v} {cc} {b_v}) ? 1 : 0")

    def _arith(self, inst: Instruction) -> List[str]:
        k, op = self.k, inst.op
        d = inst.dests[0]
        dst = _storage(k, d)
        width = inst.width or "b64"

        if op == "fma":
            ft = "float" if width == "f32" else "double"
            a, b, c = (self.val(inst, s, ft) for s in inst.srcs)
            return [self.assign(d.name, dst, f"fma({a}, {b}, {c})")]

        if op == "mul" and "wide" in inst.mods:
            signed = width.startswith("s")
            wide_t = "long" if signed else "ulong"
            narrow = "int" if signed else "uint"
            a, b = inst.srcs
            av = self.val(inst, a, narrow)
            bv = self.val(inst, b, narrow)
            return [self.assign(d.name, dst, f"({wide_t}){av} * ({wide_t}){bv}")]

        if op == "mad" and "wide" in inst.mods:
            _fail(k, inst, "mad.wide not lowered")

        if op == "mad":
            a, b, c = inst.srcs
            if width in ("s32", "u32", "b32"):
                ut = "uint"
                st = "int" if width == "s32" else "uint"
                av, bv, cv = (self.val(inst, s, ut) for s in (a, b, c))
                expr = f"({ut}){av} * ({ut}){bv} + ({ut}){cv}"
                return [self.assign(d.name, dst, _cast(st, expr))]
            ut = "ulong"
            av, bv, cv = (self.val(inst, s, ut) for s in (a, b, c))
            return [self.assign(d.name, dst, f"{av} * {bv} + {cv}")]

        sym = {"add": "+", "sub": "-", "mul": "*"}[op]
        it = _interp(k, inst)
        if it in ("int", "long"):
            ut = "uint" if it == "int" else "ulong"
            av = self.val(inst, inst.srcs[0], ut)
            bv = self.val(inst, inst.srcs[1], ut)
            return [self.assign(d.name, dst, _cast(it, f"{av} {sym} {bv}"))]
        av = self.val(inst, inst.srcs[0], it)
        bv = self.val(inst, inst.srcs[1], it)
        return [self.assign(d.name, dst, f"{av} {sym} {bv}")]

    def _bitwise(self, inst: Instruction) -> List[str]:
        k, op = self.k, inst.op
        d = inst.dests[0]
        dst = _storage(k, d)
        it = _interp(k, inst)
        if it not in ("uint", "int", "ulong", "long", "uchar", "char",
                      "ushort", "short"):
            it = "uint"
        if op in ("and", "or", "xor") and len(inst.srcs) == 2:
            sym = {"and": "&", "or": "|", "xor": "^"}[op]
            a, b = inst.srcs
            return [self.assign(d.name, dst,
                                f"{self.val(inst, a, it)} {sym} {self.val(inst, b, it)}")]
        if op in ("shl", "shr") and len(inst.srcs) == 2:
            sym = "<<" if op == "shl" else ">>"
            a, b = inst.srcs
            av, bv = self.val(inst, a, it), self.val(inst, b, it)
            return [self.assign(d.name, dst, _cast(it, f"{av} {sym} {bv}"))]
        if op in ("min", "max") and len(inst.srcs) == 2:
            a, b = inst.srcs
            return [self.assign(d.name, dst,
                                f"{op}({self.val(inst, a, it)}, {self.val(inst, b, it)})")]
        if op == "selp" and len(inst.srcs) == 3:
            c, a, b = inst.srcs
            return [self.assign(d.name, dst,
                                f"({self.val(inst, c, 'uchar')} ? "
                                f"{self.val(inst, a, it)} : {self.val(inst, b, it)})")]
        if op == "not" and len(inst.srcs) == 1:
            return [self.assign(d.name, dst, f"~{self.val(inst, inst.srcs[0], it)}")]
        if op == "neg" and len(inst.srcs) == 1:
            return [self.assign(d.name, dst, f"-{self.val(inst, inst.srcs[0], it)}")]
        if op == "abs" and len(inst.srcs) == 1:
            return [self.assign(d.name, dst, f"abs({self.val(inst, inst.srcs[0], it)})")]
        _fail(k, inst, f"{op} shape not lowered")
        return []


def _prove_structured(kernel: Kernel) -> None:
    """Only lower the NVIDIA guard idiom; everything else fails closed."""
    if not kernel.body:
        return
    for idx, inst in enumerate(kernel.body):
        if inst.op != "bra":
            continue
        if inst.pred is None:
            _fail(kernel, inst, "unconditional bra is outside the proven guard idiom")
        target = inst.label
        if target not in kernel.labels:
            _fail(kernel, inst, f"bra to unknown label {target!r}")
        t_idx = kernel.labels[target]
        if t_idx <= idx:
            _fail(kernel, inst, "backward branch (loop) is outside the proven guard idiom")
        for j in range(t_idx, len(kernel.body)):
            if kernel.body[j].op != "ret":
                _fail(kernel, kernel.body[j],
                      "guard target label is followed by non-ret code")


def emit_kernel(kernel: Kernel, fail_closed: bool = True) -> str:
    """Emit one OpenCL C kernel. Fail closed by default (IR rule 1)."""
    if fail_closed:
        bad = reachable_unsupported(kernel)
        if bad:
            inst = bad[0]
            raise TranslationAbort(
                f"kernel {kernel.name}: reachable unsupported instruction "
                f"(opcode {inst.op}, PTX line {inst.line})",
                kernel=kernel.name, opcode=inst.op, line=inst.line or 0)
        _prove_structured(kernel)

    em = _Emitter(kernel)
    lines: List[str] = [f"__kernel void {kernel.name}("]
    args: List[str] = []
    for p in kernel.params:
        if p.is_pointer:
            ets = set()
            for inst in kernel.body:
                if inst.addr and inst.addr.base_param == p.name:
                    w = inst.width or "b32"
                    if w in _MEM_TYPES:
                        ets.add(STORAGE_C[w])
            et = ets.pop() if len(ets) == 1 else "uchar"
            em.elem_of[p.name] = et
            args.append(f"    __global {et}* restrict {em.pname(p.name)}")
        else:
            args.append(f"    {STORAGE_C.get(p.dtype, 'ulong')} {em.pname(p.name)}")
    lines.append(",\n".join(args))
    lines.append(") {")
    for name, dt in sorted(kernel.registers.items()):
        init = "0.0f" if dt == "f32" else ("0.0" if dt == "f64" else "0")
        lines.append(f"    {STORAGE_C.get(dt, 'ulong')} {em.cname(name)} = {init};")
    lines.append("    // body")
    label_at = {idx: lab for lab, idx in kernel.labels.items()}
    for idx, inst in enumerate(kernel.body):
        if idx in label_at:
            lines.append(f"    {em.label(label_at[idx])}:;")
        if inst.op == "bra":
            if inst.pred is not None:
                lines.append(f"    if ({'!' if inst.pred_inv else ''}{em.cname(inst.pred.name)}) "
                             f"goto {em.label(inst.label)};")
            else:
                lines.append(f"    goto {em.label(inst.label)};")
            continue
        for s in em.stmt(inst):
            lines.append("    " + s)
    lines.append("}")
    return "\n".join(lines)


def emit_program(result) -> Tuple[str, Dict[str, str]]:
    """Emit all translatable kernels. Returns (header, {name: source})."""
    header_lines = [
        "// Generated by cuda-translator (OpenCL C backend)",
        "// House IR lowering of NVIDIA PTX; see docs/ground-truth.md",
        "",
    ]
    sources: Dict[str, str] = {}
    for k in result.kernels:
        try:
            sources[k.name] = emit_kernel(k, fail_closed=True)
        except TranslationAbort as e:
            header_lines.append(f"// kernel {k.name}: NOT TRANSLATED: {e}")
    return "\n".join(header_lines), sources
