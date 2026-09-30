"""House GPU IR: the neutral intermediate representation of cuda-translator.

Frontends (PTX) normalize program text into this IR; backends (OpenCL C) and
the reference interpreter consume it. Register names are preserved exactly as
they appear in PTX so IR text can be diffed mechanically against source text.

Core design rules (do not weaken without changing the proof story):

1. FAIL CLOSED. Reachable `unsupported` instructions abort translation with
   the exact opcode, kernel, and PTX source line (see reachable_unsupported).
   Inspection/analysis may continue past them; translation may not.

2. STORAGE vs INTERPRETATION. A PTX register's declared type (.reg .b32) is a
   storage width only. The semantic signedness/float-ness comes from each
   instruction's suffix (mad.lo.s32, add.f32, ...). Kernel.registers maps
   register name -> DECLARED storage type; Instruction.width is the
   per-instruction INTERPRETATION. Backends emit a storage variable and cast
   per use.

3. EXPLICIT WRAPAROUND. .lo/.hi arithmetic is modulo 2^N, never C signed
   overflow: compute in unsigned storage, reinterpret at the end
   (mad.lo.s32 -> (int32_t)((uint32_t)a*(uint32_t)b + (uint32_t)c)).
   mul.wide.s32 sign-extends both operands to 64-bit signed first.

4. ADDRESS PROVENANCE IS IR. Loads/stores carry an Address{space, base_param,
   index, scale, const} resolved by the frontend, instead of an opaque
   pointer register every backend must re-derive. Unresolved addresses keep
   raw_reg as a documented fallback, never a silent guess.

The control flow stays explicit in the IR (bra/labels); backends lower only
the structured subset they prove (e.g. the NVIDIA guard idiom
setp / @p bra END / body / END: ret).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# PTX special registers -> canonical House IR names (OpenCL C work-item dims).
SPECIAL_REGS: Dict[str, str] = {
    "%tid.x": "local_id_x",
    "%tid.y": "local_id_y",
    "%tid.z": "local_id_z",
    "%ctaid.x": "group_id_x",
    "%ctaid.y": "group_id_y",
    "%ctaid.z": "group_id_z",
    "%ntid.x": "local_size_x",
    "%ntid.y": "local_size_y",
    "%ntid.z": "local_size_z",
    "%nctaid.x": "num_groups_x",
    "%nctaid.y": "num_groups_y",
    "%nctaid.z": "num_groups_z",
}


@dataclass
class Operand:
    """A scalar operand: register, special register, or immediate.

    kind: "reg" | "special" | "imm" | "label" | "sym" | "mem" | "neg" | "not"
    For "neg"/"not", name holds the inner operand's text.
    """

    kind: str
    name: str
    dtype: Optional[str] = None  # declared storage type of a register operand

    def __str__(self) -> str:  # pragma: no cover - debug aid
        if self.kind == "imm":
            return str(self.name)
        return self.name


@dataclass
class Address:
    """Resolved memory provenance for ld/st (IR rule 4)."""

    space: str  # "global"
    base_param: Optional[str] = None  # kernel parameter owning the buffer
    index: Optional[Operand] = None  # element/byte index register
    scale: int = 1  # bytes per index unit (4 for f32 arrays)
    const: int = 0  # constant byte offset added to scale*index
    raw_reg: Optional[str] = None  # fallback: unresolved pointer register


@dataclass
class Instruction:
    """One IR instruction.

    op: House opcode. width: per-instruction INTERPRETATION suffix
    ("s32" in mad.lo.s32) — distinct from register storage (IR rule 2).
    mods: flags such as "rn", "wide", "lo", "hi", and setp comparisons.
    line: 1-based PTX source line for fail-closed diagnostics (IR rule 1).
    addr: resolved Address for ld/st (IR rule 4).
    """

    op: str
    dests: List[Operand] = field(default_factory=list)
    srcs: List[Operand] = field(default_factory=list)
    pred: Optional[Operand] = None  # guarding predicate (execute if True)
    pred_inv: bool = False  # True for @!p
    width: Optional[str] = None
    mods: List[str] = field(default_factory=list)
    label: Optional[str] = None  # target for bra
    addr: Optional[Address] = None
    line: Optional[int] = None
    comment: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - debug aid
        parts = []
        if self.pred is not None:
            parts.append(("@!" if self.pred_inv else "@") + self.pred.name)
        parts.append(self.op)
        if self.width:
            parts.append("." + self.width)
        for m in self.mods:
            parts.append("." + m)
        if self.dests:
            parts.append(", ".join(str(d) for d in self.dests) + ",")
        if self.srcs:
            parts.append(", ".join(str(s) for s in self.srcs))
        if self.label is not None:
            parts.append(self.label)
        if self.addr is not None:
            parts.append(f"[{self.addr_summary()}]")
        return " ".join(parts)

    def addr_summary(self) -> str:  # pragma: no cover - debug aid
        a = self.addr
        if a is None:
            return "?"
        if a.base_param is None:
            return f"{a.space}:[{a.raw_reg}]"
        idx = ""
        if a.index is not None:
            idx = f" + {a.scale}*{a.index.name}"
        if a.const:
            idx += f" + {a.const}"
        return f"{a.space}:{a.base_param}{idx}"


@dataclass
class KernelParam:
    name: str
    dtype: str  # House scalar type: f32, f64, u32, s32, u64, b64 (pointer) ...
    is_pointer: bool = False


@dataclass
class Kernel:
    name: str
    params: List[KernelParam] = field(default_factory=list)
    body: List[Instruction] = field(default_factory=list)
    labels: Dict[str, int] = field(default_factory=dict)  # label -> body index
    registers: Dict[str, str] = field(default_factory=dict)  # name -> DECLARED storage type
    source: str = "ptx"  # provenance: "ptx" | "ptx-nvrtc" | synthetic...
    source_lines: List[str] = field(default_factory=list)  # original PTX text

    def reg_dtype(self, name: str) -> Optional[str]:
        """Declared STORAGE type of a register (not its per-use interpretation)."""
        return self.registers.get(name)

    def label_line(self, label: str) -> Optional[int]:
        idx = self.labels.get(label)
        if idx is None or idx >= len(self.body):
            return None
        return self.body[idx].line


def _successors(kernel: Kernel, idx: int) -> List[int]:
    inst = kernel.body[idx]
    if inst.op == "ret":
        return []
    if inst.op == "bra":
        targets = []
        if inst.label in kernel.labels:
            targets.append(kernel.labels[inst.label])
        if inst.pred is not None and idx + 1 < len(kernel.body):
            targets.append(idx + 1)  # conditional branch: fallthrough too
        if inst.pred is None and not targets:
            raise ValueError(
                f"kernel {kernel.name}: unconditional bra with unknown target "
                f"{inst.label!r} at PTX line {inst.line}"
            )
        return targets
    return [idx + 1] if idx + 1 < len(kernel.body) else []


def reachable_unsupported(kernel: Kernel) -> List[Instruction]:
    """IR rule 1: unsupported instructions reachable from entry, fail closed.

    Walks the CFG from body[0] through bra/ret edges and returns every
    `unsupported` instruction that can actually execute. Empty list = kernel
    is inside the supported envelope for translation.
    """
    if not kernel.body:
        return []
    seen = {0}
    work = [0]
    bad: List[Instruction] = []
    while work:
        idx = work.pop()
        inst = kernel.body[idx]
        if inst.op == "unsupported" and inst not in bad:
            bad.append(inst)
        for nxt in _successors(kernel, idx):
            if nxt not in seen:
                seen.add(nxt)
                work.append(nxt)
    return bad


@dataclass
class TranslationResult:
    """What a frontend produces for one input."""

    kernels: List[Kernel]
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    source_kind: str = "ptx"
    notes: Dict[str, Any] = field(default_factory=dict)
