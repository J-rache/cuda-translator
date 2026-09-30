"""PE/COFF scanner: locate embedded CUDA fatbins in host executables.

A CUDA host binary (built by nvcc with the runtime API) embeds one or more
fatbin images, referenced by __cuda_fatbin_ctor code and named with the
section they are placed in (.nv_fatbin on Windows/MSVC, also common:
.nvFatBinSegment). This scanner:

1. reads PE headers and section table to learn where raw sections live,
2. scans candidate sections for the fatbin magic 0xBA55ED50,
3. validates the candidate as a fatbin (self-consistent sizes) and extracts
   the exact byte range.

Extracted bytes are then parsed by fatbin.parse — the same bytes the CUDA
runtime would JIT.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

from ._meta import InputError
from . import fatbin

FATBIN_MAGIC = 0xBA55ED50


@dataclass
class FatbinHit:
    offset: int  # file offset where the fatbin starts
    size: int  # exact byte size (outer header + size field)
    source_section: str
    data: bytes

    def parse(self) -> fatbin.Fatbin:
        return fatbin.parse(self.data)


@dataclass
class PEScanResult:
    is_pe: bool
    arch: Optional[str] = None
    sections: List[str] = field(default_factory=list)
    hits: List[FatbinHit] = field(default_factory=list)


def _valid_fatbin_at(data: bytes, off: int) -> Optional[int]:
    """Return the fatbin size if a self-consistent fatbin starts at off."""
    if off + 16 > len(data):
        return None
    magic, version, header_size, size = struct.unpack_from("<IHHQ", data, off)
    if magic != FATBIN_MAGIC or version != 1 or header_size != 16:
        return None
    total = header_size + size
    if total < 16 + 64 or off + total > len(data):
        return None
    # entries must tile the declared size exactly (consistency check)
    pos = header_size
    tiles = 0
    while pos + 8 <= total:
        kind, ever, ehsize, esize = struct.unpack_from("<HHII", data, off + pos)
        if ehsize < 56 or ehsize > 256:
            return None
        if pos + ehsize + esize > total:
            return None
        pos += ehsize + esize
        tiles += 1
        if pos == total:
            return total
    return None


def _pe_arch(machine: int) -> Optional[str]:
    return {0x8664: "x86-64", 0x14C: "x86", 0xAA64: "arm64"}.get(machine)


def scan(data: bytes) -> PEScanResult:
    """Scan PE/COFF bytes for embedded fatbins."""
    res = PEScanResult(is_pe=False)
    if len(data) < 64 or data[:2] != b"MZ":
        return res
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew + 24 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
        return res
    res.is_pe = True
    machine = struct.unpack_from("<H", data, e_lfanew + 4)[0]
    res.arch = _pe_arch(machine)
    num_sections = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    sec_off = e_lfanew + 24 + opt_size
    sections: List[tuple] = []
    for i in range(num_sections):
        base = sec_off + i * 40
        if base + 40 > len(data):
            break
        name = data[base:base + 8].rstrip(b"\0").decode("ascii", errors="replace")
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, base + 8)
        sections.append((name, rawptr, rawsize))
        res.sections.append(name)
    # scan section raw data (fatbin images live in initialized sections)
    for name, ptr, size in sections:
        if ptr == 0 or size == 0 or ptr + size > len(data):
            continue
        blob = data[ptr:ptr + size]
        pos = blob.find(struct.pack("<I", FATBIN_MAGIC))
        while pos >= 0:
            abs_off = ptr + pos
            fsize = _valid_fatbin_at(data, abs_off)
            if fsize:
                res.hits.append(FatbinHit(offset=abs_off, size=fsize,
                                          source_section=name,
                                          data=data[abs_off:abs_off + fsize]))
            pos = blob.find(struct.pack("<I", FATBIN_MAGIC), pos + 1)
    return res


def scan_file(path: str) -> PEScanResult:
    with open(path, "rb") as f:
        return scan(f.read())
