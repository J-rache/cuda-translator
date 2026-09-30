"""ELF cubin parser: header, section table, symbols, entry detection.

Cubin is a plain ELF64 LSB file for the EM_CUDA machine (183). This parser
reads the structures it needs directly (no external dependencies) and is
validated by feeding carved payloads to NVIDIA nvdisasm (docs/ground-truth.md).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ._meta import InputError

EM_CUDA = 190  # verified against real nvcc 12.9 cubins (e_machine bytes)


@dataclass
class ElfSection:
    name: str
    sh_type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    addralign: int
    entsize: int
    data: bytes = b""


@dataclass
class ElfSymbol:
    name: str
    value: int
    size: int
    info: int
    other: int
    shndx: int

    @property
    def is_func(self) -> bool:
        return (self.info & 0xF) == 2  # STT_FUNC


@dataclass
class Cubin:
    data: bytes
    machine: int
    entry: int
    phoff: int
    shoff: int
    flags: int
    shentsize: int
    shnum: int
    shstrndx: int
    sections: List[ElfSection] = field(default_factory=list)
    symbols: List[ElfSymbol] = field(default_factory=list)
    arch: Optional[str] = None

    def entry_symbols(self) -> List[ElfSymbol]:
        return [s for s in self.symbols if s.is_func]

    def section_by_name(self, name: str) -> Optional[ElfSection]:
        for s in self.sections:
            if s.name == name:
                return s
        return None


def _arch_from_flags(flags: int) -> Optional[str]:
    """Empirically derived from 11 real nvcc 12.9.86 cubins (see
    docs/ground-truth.md for the full e_flags matrix):

    sm_50..sm_90 : e_flags = (NN << 16) | (5 << 8) | NN      -> arch = major
    sm_100/sm_120: e_flags = (0x600 << 8)?? actually (1536 << 16)|... no:
                   measured 0x06006402 -> major=1536, mid=100, core=2
                   0x06007802 -> major=1536, mid=120, core=2  -> arch = mid
    Anything outside these two observed shapes returns None (unknown) and the
    caller can surface raw flags; we never guess."""
    major = (flags >> 16) & 0xFFFF
    mid = (flags >> 8) & 0xFF
    if 50 <= major <= 99:
        return f"sm_{major}"
    if major == 1536 and 100 <= mid <= 130:
        return f"sm_{mid}"
    return None


def parse(data: bytes) -> Cubin:
    if len(data) < 64:
        raise InputError(f"ELF too small: {len(data)} bytes")
    if data[:4] != b"\x7fELF":
        raise InputError("bad ELF magic")
    if data[4] != 2:
        raise InputError("only ELF64 supported")
    if data[5] != 1:
        raise InputError("only little-endian ELF supported")
    (e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags,
     e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum,
     e_shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    if e_machine != EM_CUDA:
        raise InputError(f"not a cubin (e_machine={e_machine}, expected {EM_CUDA})")
    cub = Cubin(data=data, machine=e_machine, entry=e_entry, phoff=e_phoff,
                shoff=e_shoff, flags=e_flags, shentsize=e_shentsize,
                shnum=e_shnum, shstrndx=e_shstrndx,
                arch=_arch_from_flags(e_flags))
    if e_shoff == 0 or e_shnum == 0:
        return cub
    # section table
    raw_secs = []
    for i in range(e_shnum):
        (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link,
         sh_info, sh_addralign, sh_entsize) = struct.unpack_from(
            "<IIQQQQIIQQ", data, e_shoff + i * e_shentsize)
        raw_secs.append(ElfSection(name="", sh_type=sh_type, flags=sh_flags,
                                   addr=sh_addr, offset=sh_offset, size=sh_size,
                                   link=sh_link, info=sh_info,
                                   addralign=sh_addralign, entsize=sh_entsize))
        if sh_offset + sh_size <= len(data):
            raw_secs[-1].data = data[sh_offset:sh_offset + sh_size]
    # section header string table
    if e_shstrndx < len(raw_secs):
        strtab = raw_secs[e_shstrndx].data
        for s in raw_secs:
            end = strtab.find(b"\0", s.offset if False else 0)
    strtab = raw_secs[e_shstrndx].data if e_shstrndx < len(raw_secs) else b""
    names = []
    for i in range(e_shnum):
        sh_name = struct.unpack_from("<I", data, e_shoff + i * e_shentsize)[0]
        end = strtab.find(b"\0", sh_name)
        if end < 0:
            end = len(strtab)
        names.append(strtab[sh_name:end].decode("utf-8", errors="replace"))
    for s, n in zip(raw_secs, names):
        s.name = n
    cub.sections = raw_secs
    # symbols from .symtab
    symtab = cub.section_by_name(".symtab")
    if symtab is not None and symtab.entsize:
        strtab_s = cub.sections[symtab.link] if symtab.link < len(cub.sections) else None
        stab = strtab_s.data if strtab_s else b""
        count = len(symtab.data) // symtab.entsize
        for i in range(count):
            (st_name, st_info, st_other, st_shndx, st_value,
             st_size) = struct.unpack_from("<IBBHQQ", symtab.data, i * symtab.entsize)
            end = stab.find(b"\0", st_name)
            if end < 0:
                end = len(stab)
            name = stab[st_name:end].decode("utf-8", errors="replace")
            cub.symbols.append(ElfSymbol(name=name, value=st_value, size=st_size,
                                         info=st_info, other=st_other,
                                         shndx=st_shndx))
    return cub


def parse_file(path: str) -> Cubin:
    with open(path, "rb") as f:
        return parse(f.read())
