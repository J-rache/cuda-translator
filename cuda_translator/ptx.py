"""PTX frontend: parse real NVIDIA-generated PTX into House IR.

Grounded against fixtures emitted by nvcc 12.9.86 and NVRTC 12.9 in
examples/artifacts (see docs/ground-truth.md). Implements the four House IR
rules directly:

1. Fail closed: unknown opcodes become op="unsupported" and carry the PTX
   source line; ir.reachable_unsupported() gates translation.
2. Storage vs interpretation: register declarations keep their DECLARED type
   (b32/b64/f32/...); per-instruction semantics live in Instruction.width.
3. No silent C signedness: mad.lo/mul.wide keep their mods so backends emit
   explicit unsigned-storage math (see opencl.py).
4. Address provenance resolved HERE into Address{space, base_param, index,
   scale, const} via an abstract walk of cvta/ld_param/mul.wide/add chains.
"""
from __future__ import annotations

import re
import struct as _struct
from typing import Any, Dict, List, Optional, Tuple

from ._meta import InputError
from .ir import (
    SPECIAL_REGS,
    Address,
    Instruction,
    Kernel,
    KernelParam,
    Operand,
    TranslationResult,
)

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<float>0[fF][0-9A-Fa-f]{8}|0[dD][0-9A-Fa-f]{16}|\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+)
  | (?P<hex>0[xX][0-9A-Fa-f]+)
  | (?P<num>\d+)
  | (?P<directive>\.[A-Za-z_][\w.]*)
  | (?P<ident>[$%A-Za-z_][\w$.%]*(?:<\d+>)?)
  | (?P<punct>[(){}\[\],;:@!=+\-*/<>&|^~])
    """,
    re.VERBOSE,
)


def tokenize_with_lines(text: str) -> List[Tuple[str, int]]:
    """Tokenize PTX, returning (token, 1-based source line) pairs."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    out: List[Tuple[str, int]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        cut = line.find("//")
        code = line if cut < 0 else line[:cut]
        pos = 0
        while pos < len(code):
            m = _TOKEN_RE.match(code, pos)
            if not m:
                raise InputError(f"PTX tokenize error, line {lineno}: {code[pos:pos+40]!r}")
            pos = m.end()
            if m.lastgroup == "ws":
                continue
            out.append((m.group(), lineno))
    return out


def tokenize(text: str) -> List[str]:
    """Tokenize PTX text (tokens only; see tokenize_with_lines for lines)."""
    return [t for t, _ in tokenize_with_lines(text)]


# --- opcode normalization -------------------------------------------------

# PTX base opcode -> House op. None = recognized but unsupported (diagnostic).
OP_MAP: Dict[str, Optional[str]] = {
    "mov": "mov",
    "ld": "ld",
    "st": "st",
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "mad": "mad",
    "fma": "fma",
    "div": "div",
    "bar": "barrier",
    "setp": "setp",
    "bra": "bra",
    "ret": "ret",
    "cvt": "cvt",
    "cvta": "cvta_global",
    "and": "and",
    "or": "or",
    "xor": "xor",
    "not": "not",
    "shl": "shl",
    "shr": "shr",
    "selp": "selp",
    "min": "min",
    "max": "max",
    "abs": "abs",
    "neg": "neg",
    "sin": "sin_approx",
    "cos": "cos_approx",
    "sqrt": "sqrt",
    "ex2": "ex2_approx",
    "lg2": "lg2_approx",
}

CMP_OPS = {"lt", "gt", "le", "ge", "eq", "ne",
           "ltu", "gtu", "leu", "geu", "equ", "neu", "num", "nan"}

SCALAR_TYPES = {"u32", "s32", "u64", "s64", "b32", "b64", "f32", "f64",
                "u16", "s16", "b16", "u8", "s8", "pred"}

STATE_SPACES = {"param", "global", "local", "shared", "const"}
ROUNDING = {"rn", "rz", "rp", "rm"}


class _Parser:
    def __init__(self, toks_with_lines: List[Tuple[str, int]], source_name: str):
        self.tl = toks_with_lines
        self.toks = [t for t, _ in toks_with_lines]
        self.lines = [n for _, n in toks_with_lines]
        self.i = 0
        self.source_name = source_name
        self.diagnostics: List[Dict[str, Any]] = []
        self.ptx_version: Optional[Tuple[int, int]] = None
        self.target: Optional[str] = None
        self.address_size: Optional[int] = None
        self.dynamic_shared_symbols: List[str] = []

    # -- token helpers
    def peek(self, ahead: int = 0) -> Optional[str]:
        j = self.i + ahead
        return self.toks[j] if j < len(self.toks) else None

    def line(self) -> int:
        return self.lines[self.i] if self.i < len(self.lines) else -1

    def next(self) -> str:
        tok = self.peek()
        if tok is None:
            raise InputError(f"{self.source_name}: unexpected end of PTX")
        self.i += 1
        return tok

    def expect(self, want: str) -> str:
        tok = self.next()
        if tok != want:
            raise InputError(
                f"{self.source_name}: expected {want!r}, got {tok!r} (near token {self.i})"
            )
        return tok

    def diag(self, severity: str, message: str, **extra: Any) -> None:
        d = {"severity": severity, "message": message,
             "source": self.source_name, "line": self.line()}
        d.update(extra)
        self.diagnostics.append(d)

    # -- module grammar
    def parse_module(self) -> List[Kernel]:
        kernels: List[Kernel] = []
        while self.peek() is not None:
            tok = self.peek() or ""
            if tok == ".version":
                self.next()
                a, _, b = self.next().partition(".")
                self.ptx_version = (int(a), int(b or 0))
                self._end_statement()
            elif tok == ".target":
                self.next()
                self.target = self.next()
                self._end_statement()
            elif tok == ".address_size":
                self.next()
                self.address_size = int(self.next())
                self._end_statement()
            elif tok in (".visible", ".entry", ".func"):
                kernels.append(self._parse_kernel())
            elif tok in (".file", ".section", ".loc"):
                self.next()
                self._end_statement()
            elif tok == ".extern":
                self._parse_extern_directive()
            else:
                self.next()
                self.diag("warning", f"skipped top-level directive {tok!r}")
                self._end_statement()
        return kernels

    def _parse_extern_directive(self) -> None:
        """Capture unsized .extern .shared declarations.

        Numba emits dynamic shared memory as:
          .extern .shared .align 8 .b8 SYMBOL[];
        CUDA gives all such unsized extern shared symbols the launch-provided
        dynamic shared-memory base. Other extern forms remain inspection-only.
        """
        start_line = self.line()
        self.expect(".extern")
        tokens: List[str] = []
        while self.peek() is not None and self.peek() != ";":
            # Do not consume the next source line if a malformed directive is
            # missing its semicolon.
            if self.i > 0 and self.lines[self.i] > start_line:
                break
            tokens.append(self.next())
        if self.peek() == ";":
            self.next()

        if ".shared" not in tokens:
            self.diag("warning", "skipped top-level directive '.extern'")
            return

        symbol: Optional[str] = None
        unsized = False
        for idx, tok in enumerate(tokens):
            if re.fullmatch(r"[$A-Za-z_][\w$.]*", tok):
                # .shared/.align/.b8 are directives and cannot match because
                # they begin with '.', so the last identifier is the symbol.
                symbol = tok
            if tok == "[" and idx + 1 < len(tokens) and tokens[idx + 1] == "]":
                unsized = True

        if not symbol or not unsized:
            self.diag(
                "error",
                "only unsized extern shared declarations are supported",
                opcode=".extern.shared",
            )
            return

        if symbol not in self.dynamic_shared_symbols:
            self.dynamic_shared_symbols.append(symbol)

    def _end_statement(self) -> None:
        # PTX top-level directives end at newline OR ';'. Stop when the next
        # token is on a later line than the LAST CONSUMED token (newline
        # terminated the statement) so semicolon-less directives cannot
        # swallow the following line (e.g. the kernel header).
        while self.peek() is not None:
            if self.peek() == ";":
                self.next()
                return
            if self.i > 0 and self.i < len(self.lines):
                cur = self.lines[self.i]
                prev = self.lines[self.i - 1]
                if cur > prev:
                    return
            self.next()

    def _parse_kernel(self) -> Kernel:
        if self.peek() == ".visible":
            self.next()
        kind = self.next()  # .entry or .func
        if kind not in (".entry", ".func"):
            raise InputError(f"expected .entry/.func, got {kind!r}")
        name = self.next()
        kernel = Kernel(
            name=name,
            source="ptx",
            dynamic_shared_symbols=list(self.dynamic_shared_symbols),
        )
        if kind == ".func":
            self.diag("warning",
                      ".func (device function) parsed for inspection only; "
                      "only .entry kernels are translated")
        if self.peek() == "(":
            self.next()
            while self.peek() != ")":
                self._parse_param(kernel)
            self.expect(")")
        self.expect("{")
        self._parse_body(kernel)
        return kernel

    def _parse_param(self, kernel: Kernel) -> None:
        self.expect(".param")
        dtype: Optional[str] = None
        name: Optional[str] = None
        while self.peek() not in (")", ",", ".param", None):
            tok = self.peek()
            if tok.startswith(".") and tok.lstrip(".") in SCALAR_TYPES:
                dtype = self.next().lstrip(".")
            elif tok == ".align":
                self.next()
                self.next()
            else:
                name = self.next()
                if self.peek() == "[":  # .param .align 4 .b8 name[12] form
                    self.next()
                    dim = self.next()
                    self.expect("]")
                    name = f"{name}[{dim}]"
        kernel.params.append(KernelParam(name=name or f"param{len(kernel.params)}",
                                         dtype=dtype or "b64"))
        if self.peek() == ",":
            self.next()

    def _parse_body(self, kernel: Kernel) -> None:
        pending_pred: Optional[Operand] = None
        pending_inv = False
        while True:
            tok = self.peek()
            if tok is None:
                raise InputError(f"{self.source_name}: unterminated kernel {kernel.name!r}")
            if tok == "}":
                self.next()
                return
            if tok == ";":
                self.next()
                continue
            if tok == "@":
                self.next()
                pending_inv = False
                if self.peek() == "!":
                    self.next()
                    pending_inv = True
                reg = self.next()
                pending_pred = Operand(kind="reg", name=reg, dtype="pred")
                continue
            if tok.startswith("."):
                self._parse_body_directive(kernel)
                continue
            if tok.startswith("$") and self.peek(1) == ":":
                label = self.next()
                self.next()  # ':'
                kernel.labels[label] = len(kernel.body)
                continue
            inst = self._parse_instruction(kernel)
            inst.pred = pending_pred
            inst.pred_inv = pending_inv
            kernel.body.append(inst)
            pending_pred, pending_inv = None, False

    def _parse_body_directive(self, kernel: Kernel) -> None:
        tok = self.next()
        if tok in (".reg", ".param"):
            dtype = None
            names: List[str] = []
            while self.peek() not in (None, ";"):
                t = self.next()
                if t.startswith(".") and t.lstrip(".") in SCALAR_TYPES:
                    dtype = t.lstrip(".")
                elif t == ",":
                    continue
                else:
                    names.append(t)
            if self.peek() == ";":
                self.next()
            for name in names:
                m = re.fullmatch(r"([%$A-Za-z_][\w$.%]*)<(\d+)>", name)
                if m:
                    base, count = m.group(1), int(m.group(2))
                    for k in range(count):
                        kernel.registers[f"{base}{k}"] = dtype or "b32"
                else:
                    kernel.registers[name] = dtype or "b32"
        elif tok in (".maxnreg", ".reqntid", ".minnctapersm", ".maxntid"):
            self._end_statement()
        else:
            self.diag("warning", f"skipped body directive {tok!r} in {kernel.name}")
            self._end_statement()

    # -- instructions
    def _parse_instruction(self, kernel: Kernel) -> Instruction:
        stmt_line = self.line()
        opcode_tok = self.next()
        parts = opcode_tok.split(".")
        base, mods = parts[0], parts[1:]

        if base not in OP_MAP:
            self.diag("error", f"unknown PTX opcode {opcode_tok!r}", opcode=opcode_tok)
            house_op = "unsupported"
        else:
            house_op = OP_MAP[base]
            if house_op is None:
                self.diag("error", f"PTX opcode {opcode_tok!r} recognized but unsupported",
                          opcode=opcode_tok)
                house_op = "unsupported"

        width: Optional[str] = None
        cmp_op: Optional[str] = None
        space: Optional[str] = None
        rounding: Optional[str] = None
        kept_mods: List[str] = []

        for m in mods:
            if m in SCALAR_TYPES and width is None:
                width = m
            elif m in CMP_OPS and base == "setp":
                cmp_op = m
            elif m in STATE_SPACES and base in ("ld", "st"):
                space = m
            elif m in ROUNDING:
                rounding = m
            else:
                kept_mods.append(m)  # lo, hi, wide, to, FTZ, sat, v2/v4 ...

        inst = Instruction(op=house_op, width=width, line=stmt_line)
        if cmp_op:
            inst.mods.append(cmp_op)
        if rounding:
            inst.mods.append(rounding)
        inst.mods.extend(kept_mods)

        if base in ("ld", "st"):
            if house_op == "ld" and space == "param":
                inst.op = "ld_param"
            elif house_op == "ld" and space == "global":
                inst.op = "ld_global"
            elif house_op == "st" and space == "global":
                inst.op = "st_global"
            elif house_op == "ld" and space == "shared":
                inst.op = "ld_shared"
            elif house_op == "st" and space == "shared":
                inst.op = "st_shared"
            elif house_op == "ld" and space is None:
                # PTX generic-address load. Address-space provenance is
                # resolved after the full instruction stream is parsed.
                inst.op = "ld_generic"
            elif house_op == "st" and space is None:
                inst.op = "st_generic"
            elif house_op in ("ld", "st"):
                self.diag("error",
                          f"{space or 'default'} memory space unsupported",
                          opcode=opcode_tok)
                inst.op = "unsupported"
        if "to" in kept_mods:  # cvta.to.global
            kept_mods.remove("to")

        operands: List[Operand] = []
        while True:
            while self.peek() == ",":
                self.next()
            if self.peek() in (None, ";", "}"):
                break
            operands.append(self._parse_operand(kernel))
        if self.peek() == ";":
            self.next()

        self._assign_operands(inst, operands, opcode_tok)
        return inst

    def _assign_operands(self, inst: Instruction, operands: List[Operand],
                         opcode_tok: str) -> None:
        op = inst.op
        if op in ("st_global", "st_shared", "st_generic"):
            if operands and operands[0].kind == "mem":
                inst.dests = [operands[0]]
                inst.srcs = operands[1:]
            else:
                self.diag("error", "st without memory destination", opcode=opcode_tok)
                inst.dests, inst.srcs = [], operands
            return
        if op == "barrier":
            inst.srcs = operands
            return
        if op == "bra":
            if operands and operands[0].kind in ("label", "sym"):
                inst.label = operands[0].name
            else:
                self.diag("error", "bra without label target", opcode=opcode_tok)
            return
        if op == "ret":
            return
        if operands and operands[0].kind == "mem":
            inst.srcs = operands
            return
        if not operands:
            return
        inst.dests = [operands[0]]
        inst.srcs = operands[1:]

    def _parse_operand(self, kernel: Kernel) -> Operand:
        tok = self.peek()
        if tok is None:
            raise InputError(f"{self.source_name}: unexpected EOF in operand list")
        if tok == "[":
            self.next()
            base = self.next()
            offset = 0
            if self.peek() in ("+", "-"):
                sign = 1 if self.next() == "+" else -1
                offset = sign * int(self.next())
            self.expect("]")
            if offset == 0:
                return Operand(kind="mem", name=base)
            return Operand(kind="mem", name=f"{base}{offset:+d}")
        if tok in ("-", "+", "!"):
            unary = self.next()
            inner = self._parse_operand(kernel)
            if unary == "-":
                return Operand(kind="neg", name=inner.name, dtype=inner.dtype)
            if unary == "!":
                return Operand(kind="not", name=inner.name, dtype=inner.dtype)
            return inner
        self.next()
        if re.fullmatch(r"0[fF][0-9A-Fa-f]{8}", tok):
            bits = int(tok[2:], 16)
            val = _struct.unpack("<f", _struct.pack("<I", bits))[0]
            return Operand(kind="imm", name=repr(val))
        if re.fullmatch(r"0[dD][0-9A-Fa-f]{16}", tok):
            bits = int(tok[2:], 16)
            val = _struct.unpack("<d", _struct.pack("<Q", bits))[0]
            return Operand(kind="imm", name=repr(val))
        if re.fullmatch(r"\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+", tok):
            return Operand(kind="imm", name=repr(float(tok)))
        if re.fullmatch(r"0[xX][0-9A-Fa-f]+", tok):
            return Operand(kind="imm", name=str(int(tok, 16)))
        if re.fullmatch(r"\d+", tok):
            return Operand(kind="imm", name=tok)
        if tok.startswith("%"):
            canon = SPECIAL_REGS.get(tok)
            if canon:
                return Operand(kind="special", name=canon)
            return Operand(kind="reg", name=tok, dtype=kernel.registers.get(tok))
        if tok.startswith("$"):
            return Operand(kind="label", name=tok)
        return Operand(kind="sym", name=tok)


# --- address provenance (IR rule 4) ---------------------------------------

def resolve_addresses(kernel: Kernel) -> int:
    """Fill inst.addr for ld_global/st_global from cvta/ld_param/mul.wide/add chains.

    Abstract environment per register:
      base[r]    = param name providing the global pointer
      off[r]     = (index_reg, scale) byte-offset expression scale*index
      addr[r]    = (param, index_reg, scale, const)
    Returns number of addresses fully resolved (base_param + index known).
    """
    base: Dict[str, str] = {}
    off: Dict[str, Tuple[Operand, int]] = {}
    addr: Dict[str, Tuple[str, Operand, int, int]] = {}
    resolved = 0

    def param_of(name: str) -> Optional[str]:
        return base.get(name)

    for inst in kernel.body:
        op = inst.op
        dest = inst.dests[0].name if inst.dests else None
        if op == "ld_param" and dest and inst.srcs:
            src = inst.srcs[0]
            pname = src.name
            if pname.endswith("]") and "[" in pname:
                pname = pname.split("[")[0]
            base[dest] = pname
        elif op == "cvta_global" and dest and inst.srcs:
            # cvta.to.global %rd, %rd_src : provenance carries through
            src = inst.srcs[0]
            if src.name in addr:
                addr[dest] = addr[src.name]
            elif src.name in base:
                base[dest] = base[src.name]
            elif src.name in off:
                off[dest] = off[src.name]
        elif op == "mul" and "wide" in inst.mods and dest and len(inst.srcs) == 2:
            a, b = inst.srcs
            if b.kind == "imm":
                off[dest] = (a, int(b.name))
            elif a.kind == "imm":
                off[dest] = (b, int(a.name))
        elif op == "add" and dest and len(inst.srcs) == 2:
            a, b = inst.srcs
            a_addr, b_addr = addr.get(a.name), addr.get(b.name)
            if a_addr and b.kind == "imm":
                addr[dest] = (a_addr[0], a_addr[1], a_addr[2], a_addr[3] + int(b.name))
            elif b_addr and a.kind == "imm":
                addr[dest] = (b_addr[0], b_addr[1], b_addr[2], b_addr[3] + int(a.name))
            elif a_addr and b.name in off:
                idx, scale = off[b.name]
                addr[dest] = (a_addr[0], idx, scale, a_addr[3])
            elif b_addr and a.name in off:
                idx, scale = off[a.name]
                addr[dest] = (b_addr[0], idx, scale, b_addr[3])
            elif a.name in base and b.name in off:
                idx, scale = off[b.name]
                addr[dest] = (base[a.name], idx, scale, 0)
            elif b.name in base and a.name in off:
                idx, scale = off[a.name]
                addr[dest] = (base[b.name], idx, scale, 0)
            elif a.name in base and b.kind == "imm":
                base[dest] = base[a.name]  # pointer + const stays a pointer
            elif b.name in base and a.kind == "imm":
                base[dest] = base[b.name]
            elif a.name in base and b.name in base:
                pass  # two pointers: not resolvable here, leave raw
            elif (
                a.name in base
                and b.kind in ("reg", "special")
            ):
                # Generic CUDA/NVVM byte addressing: a proven global base
                # pointer plus an arbitrary integer byte-offset register.
                # More-specific scaled-offset cases above take precedence.
                addr[dest] = (
                    base[a.name],
                    b,
                    1,
                    0,
                )
            elif (
                b.name in base
                and a.kind in ("reg", "special")
            ):
                addr[dest] = (
                    base[b.name],
                    a,
                    1,
                    0,
                )
        elif op in ("ld_shared", "st_shared") and inst.addr is None:
            mem = inst.dests[0] if op == "st_shared" else (
                inst.srcs[0] if inst.srcs else None)
            if mem is not None and mem.kind == "mem":
                reg = mem.name
                const = 0
                m = re.fullmatch(r"(%[\w$.%]+)([+-]\d+)", reg)
                if m:
                    reg, const = m.group(1), int(m.group(2))
                inst.addr = Address(
                    space="shared",
                    raw_reg=reg,
                    const=const,
                )
                resolved += 1
        elif op in (
            "ld_global", "st_global", "ld_generic", "st_generic"
        ) and inst.addr is None:
            is_store = op in ("st_global", "st_generic")
            mem = inst.dests[0] if is_store else (
                inst.srcs[0] if inst.srcs else None)
            if mem is not None and mem.kind == "mem":
                reg = mem.name
                const = 0
                m = re.fullmatch(r"(%[\w$.%]+)([+-]\d+)", reg)
                if m:
                    reg, const = m.group(1), int(m.group(2))
                if reg in addr:
                    p, idx, scale, c = addr[reg]
                    inst.addr = Address(space="global", base_param=p,
                                        index=idx, scale=scale, const=const + c)
                    if op == "ld_generic":
                        inst.op = "ld_global"
                    elif op == "st_generic":
                        inst.op = "st_global"
                    resolved += 1
                elif reg in base:
                    inst.addr = Address(space="global", base_param=base[reg],
                                        raw_reg=reg)
                    if op == "ld_generic":
                        inst.op = "ld_global"
                    elif op == "st_generic":
                        inst.op = "st_global"
                    resolved += 1
                elif reg in off:
                    idx, scale = off[reg]
                    inst.addr = Address(space="global", index=idx, scale=scale,
                                        const=const)
                else:
                    inst.addr = Address(space="global", raw_reg=reg, const=const)
        elif op == "mov" and dest and len(inst.srcs) == 1:
            src = inst.srcs[0]
            if src.kind in ("reg", "special") or src.kind == "imm":
                if src.name in addr:
                    addr[dest] = addr[src.name]
                elif src.name in base:
                    base[dest] = base[src.name]
                elif src.name in off:
                    off[dest] = off[src.name]
    return resolved


def _resolve_param_pointeriness(kernels: List[Kernel]) -> None:
    """Resolve generic memory and mark proven global pointer parameters."""
    for kernel in kernels:
        resolve_addresses(kernel)
        for inst in kernel.body:
            if inst.op in ("ld_generic", "st_generic"):
                inst.comment = (
                    "generic memory address space could not be proven"
                )
                inst.op = "unsupported"
        pointed: Dict[str, int] = {}
        for inst in kernel.body:
            a = inst.addr
            if a is not None and a.base_param:
                pointed[a.base_param] = pointed.get(a.base_param, 0) + 1
        for p in kernel.params:
            clean = p.name.split("[")[0]
            if clean in pointed:
                p.is_pointer = True
                p.name = clean


def parse_ptx(text: str, source_name: str = "<ptx>") -> TranslationResult:
    """Parse PTX text into a TranslationResult (House IR)."""
    parser = _Parser(tokenize_with_lines(text), source_name)
    kernels = parser.parse_module()
    for k in kernels:
        k.source_lines = text.splitlines()
    _resolve_param_pointeriness(kernels)
    return TranslationResult(
        kernels=kernels,
        diagnostics=parser.diagnostics,
        source_kind="ptx",
        notes={
            "ptx_version": parser.ptx_version,
            "target": parser.target,
            "address_size": parser.address_size,
        },
    )


def ir_text(result: TranslationResult) -> str:
    """Render House IR to readable text (diffs, receipts)."""
    out: List[str] = []
    for k in result.kernels:
        params = ", ".join(
            f".param .{p.dtype} {p.name}" + ("  // ptr" if p.is_pointer else "")
            for p in k.params)
        out.append(f".entry {k.name}({params}) {{")
        for name, dt in sorted(k.registers.items()):
            out.append(f"  .reg .{dt} {name};  // storage")
        for inst in k.body:
            line_note = f"  // PTX:{inst.line}" if inst.line else ""
            out.append("  " + str(inst) + ";" + line_note)
        out.append("}")
    return "\n".join(out)
