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
    "dynamic_shared_size": "__house_dynamic_shared_size",
}

_CMP_C = {
    "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    "equ": "==", "neu": "!=", "ltu": "<", "leu": "<=",
    "gtu": ">", "geu": ">=",
}

_MEM_TYPES = (
    "f32", "f64",
    "u8", "s8", "b8",
    "u16", "s16", "b16",
    "u32", "s32", "b32",
    "u64", "s64", "b64",
)


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
            if o.name in self.k.registers:
                inner_kind = "reg"
            elif o.name in SPECIAL_OCL:
                inner_kind = "special"
            else:
                inner_kind = "imm"
            inner = Operand(
                kind=inner_kind,
                name=o.name,
                dtype=o.dtype,
            )
            u = "-" if o.kind == "neg" else "!"
            return _cast(
                want,
                u + self.val(inst, inner, want),
            )
        if o.kind == "mem":
            _fail(self.k, inst, f"unresolved memory operand {o.name!r}")
        if o.kind == "sym" and o.name in self.k.dynamic_shared_symbols:
            # CUDA aliases all unsized extern shared symbols to the
            # launch-provided dynamic shared-memory base.
            return _cast(want, "0UL")
        if o.kind in ("label", "sym"):
            _fail(self.k, inst, f"unexpected symbolic operand {o.name!r}")
        return _cast(want, self.cname(o.name))

    def elem_ptr(self, a: Address, width: str, const: bool = True) -> str:
        """Pointer to the addressed element via BYTE arithmetic.

        Address.scale is a BYTE stride, so the typed param pointer is cast to
        __global uchar* (1-byte steps) before adding scale*index+const, then
        cast back to the element type. Doing p + r*scale directly on a typed
        pointer would scale TWICE (C pointer arithmetic multiplies by sizeof
        element), producing 16-byte strides for float buffers.
        """
        if a.base_param is None:
            _fail(self.k, None, f"unresolved address (raw_reg={a.raw_reg!r})")
        etype = STORAGE_C.get(width)
        if etype is None:
            _fail(self.k, None, f"global-memory width {width} not lowered")
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

    def shared_elem_ptr(self, a: Address, width: str) -> str:
        """Pointer into launch-provided dynamic shared memory by byte offset."""
        if a.space != "shared" or a.raw_reg is None:
            _fail(self.k, None, "unresolved shared-memory address")
        etype = STORAGE_C.get(width)
        if etype is None:
            _fail(self.k, None, f"shared-memory width {width} not lowered")
        off = f"(size_t)({self.cname(a.raw_reg)})"
        if a.const:
            off += f" + ({a.const})"
        return (
            f"((__local {etype} *)"
            f"(((__local uchar *)(__house_dynamic_shared)) + ({off})))"
        )

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
                                f"*{self.elem_ptr(a, dt)}")]
        if op == "st_global":
            a = inst.addr
            if a is None:
                _fail(self.k, inst, "st_global without resolved address")
            dt = inst.width or "b32"
            if dt not in _MEM_TYPES:
                _fail(self.k, inst, f"st_global width {dt} not lowered")
            etype = STORAGE_C[dt]
            v = self.val(inst, inst.srcs[0], etype)
            return [f"*{self.elem_ptr(a, dt, const=False)} = {v};"]
        if op == "ld_shared":
            d = inst.dests[0]
            a = inst.addr
            if a is None:
                _fail(self.k, inst, "ld_shared without resolved address")
            dt = inst.width or "b32"
            if dt not in _MEM_TYPES:
                _fail(self.k, inst, f"ld_shared width {dt} not lowered")
            return [
                self.assign(
                    d.name,
                    _storage(self.k, d),
                    f"*{self.shared_elem_ptr(a, dt)}",
                )
            ]
        if op == "st_shared":
            a = inst.addr
            if a is None:
                _fail(self.k, inst, "st_shared without resolved address")
            dt = inst.width or "b32"
            if dt not in _MEM_TYPES:
                _fail(self.k, inst, f"st_shared width {dt} not lowered")
            etype = STORAGE_C[dt]
            v = self.val(inst, inst.srcs[0], etype)
            return [
                f"*{self.shared_elem_ptr(a, dt)} = {v};"
            ]
        if op == "barrier":
            if "sync" not in inst.mods:
                _fail(self.k, inst, "only bar.sync is lowered")
            if (
                len(inst.srcs) != 1
                or inst.srcs[0].kind != "imm"
                or inst.srcs[0].name != "0"
            ):
                _fail(self.k, inst, "only bar.sync 0 is lowered")
            return [
                "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);"
            ]
        if op in ("add", "sub", "mul", "div", "mad", "fma"):
            return self._arith(inst)
        if op == "setp":
            return [self._setp(inst)]
        if op == "cvta_global":
            d, s = inst.dests[0], inst.srcs[0]
            return [self.assign(d.name, _storage(self.k, d),
                                self.val(inst, s, "ulong"))]
        if op == "cvt":
            d, src = inst.dests[0], inst.srcs[0]
            dst_type = INTERP_C.get(
                inst.width or "b32",
                "uint",
            )
            source_width = None
            for mod in reversed(inst.mods):
                if mod in INTERP_C:
                    source_width = mod
                    break
            if source_width is None:
                _fail(
                    self.k,
                    inst,
                    "cvt source type is not explicit",
                )
            src_type = INTERP_C[source_width]
            src_value = self.val(
                inst,
                src,
                src_type,
            )
            return [
                self.assign(
                    d.name,
                    _storage(self.k, d),
                    _cast(dst_type, src_value),
                )
            ]
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
        if it not in (
            "char", "uchar", "short", "ushort",
            "int", "uint", "long", "ulong", "float", "double",
        ):
            it = "int"
        if (
            it in ("float", "double")
            and cmp_op in ("equ", "neu", "ltu", "leu", "gtu", "geu")
        ):
            base_cmp = {
                "equ": "==",
                "neu": "!=",
                "ltu": "<",
                "leu": "<=",
                "gtu": ">",
                "geu": ">=",
            }[cmp_op]
            a_v = self.val(inst, a, it)
            b_v = self.val(inst, b, it)
            expr = (
                f"(isnan({a_v}) || isnan({b_v}) || "
                f"({a_v} {base_cmp} {b_v}))"
            )
            return self.assign(
                d.name,
                "uchar",
                f"{expr} ? 1 : 0",
            )
        if cmp_op.endswith("u") and it in ("char", "short", "int", "long"):
            uit = {
                "char": "uchar",
                "short": "ushort",
                "int": "uint",
                "long": "ulong",
            }[it]
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

        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
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
            # PTX: selp d, a, b, p => d = p ? a : b
            a, b, c = inst.srcs
            return [self.assign(d.name, dst,
                                f"({self.val(inst, c, 'uchar')} ? "
                                f"{self.val(inst, a, it)} : {self.val(inst, b, it)})")]
        if op == "not" and len(inst.srcs) == 1:
            if (inst.width or "") == "pred":
                return [
                    self.assign(
                        d.name,
                        dst,
                        f"!{self.val(inst, inst.srcs[0], 'uchar')}",
                    )
                ]
            return [self.assign(d.name, dst, f"~{self.val(inst, inst.srcs[0], it)}")]
        if op == "neg" and len(inst.srcs) == 1:
            raw_it = _interp(k, inst)
            if raw_it in ("float", "double"):
                it = raw_it
            return [self.assign(d.name, dst, f"-{self.val(inst, inst.srcs[0], it)}")]
        if op == "abs" and len(inst.srcs) == 1:
            raw_it = _interp(k, inst)
            if raw_it in ("float", "double"):
                it = raw_it
                return [
                    self.assign(
                        d.name,
                        dst,
                        f"fabs({self.val(inst, inst.srcs[0], it)})",
                    )
                ]
            return [self.assign(d.name, dst, f"abs({self.val(inst, inst.srcs[0], it)})")]
        _fail(k, inst, f"{op} shape not lowered")
        return []


def _prove_direct_cfg(kernel: Kernel) -> None:
    """Prove direct PTX branches are representable as OpenCL C gotos.

    Registers are declared at kernel scope before every label, so forward and
    backward direct branches do not cross declarations. The PTX frontend
    still fails closed on unsupported/indirect control-flow opcodes.
    """
    for inst in kernel.body:
        if inst.op != "bra":
            continue
        if not inst.label or inst.label not in kernel.labels:
            _fail(kernel, inst, f"bra to unknown label {inst.label!r}")


def emit_kernel(kernel: Kernel, fail_closed: bool = True) -> str:
    """Emit one OpenCL C kernel. Fail closed by default (IR rule 1)."""
    if fail_closed:
        # Prove branch targets before walking the CFG. Otherwise an unknown
        # unconditional target can escape reachable_unsupported() as a raw
        # ValueError instead of the public fail-closed TranslationError contract.
        _prove_direct_cfg(kernel)
        bad = reachable_unsupported(kernel)
        if bad:
            inst = bad[0]
            raise TranslationAbort(
                f"kernel {kernel.name}: reachable unsupported instruction "
                f"(opcode {inst.op}, PTX line {inst.line})",
                kernel=kernel.name, opcode=inst.op, line=inst.line or 0)

    em = _Emitter(kernel)
    uses_fp64 = (
        any(dt == "f64" for dt in kernel.registers.values())
        or any(p.dtype == "f64" for p in kernel.params)
        or any(inst.width == "f64" for inst in kernel.body)
    )
    lines: List[str] = []
    if uses_fp64:
        lines.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable")
    # PTX distinguishes explicit fma from separate mul/add instructions.
    # Prevent the OpenCL compiler from silently contracting the latter.
    lines.append("#pragma OPENCL FP_CONTRACT OFF")
    lines.append(f"__kernel void {kernel.name}(")
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
    if kernel.dynamic_shared_symbols:
        args.append("    __local uchar* __house_dynamic_shared")
        args.append("    ulong __house_dynamic_shared_size")
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
