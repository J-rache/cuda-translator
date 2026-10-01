"""CUDA fatbin container parser with NVIDIA-compatible PTX decompression.

Outer header (16 bytes, little-endian) — grounded on nvcc 12.9.86 artifacts:
    u32 magic      = 0xBA55ED50
    u16 version    = 1
    u16 headerSize = 16
    u64 size       = bytes AFTER this header (total = headerSize + size)

Entry header (64 bytes for ELF entries, 80 for PTX entries, little-endian):
    u16 kind          1 = PTX, 2 = ELF (cubin)
    u16 version       0x0101 = 1.1
    u32 headerSize    64 or 80 (80 for compressed-capable PTX entries)
    u64 size          payload size on disk (includes compression padding)
    u32 compressedSize  0 when uncompressed; else compressed byte count
    u32 unknown2      64 on PTX entries (address size), 0 on ELF entries
    u16 minor, u16 major    PTX ISA version (e.g. 8.8) on PTX entries
    u32 arch          sm arch number (e.g. 75)
    u32 objNameOffset, u32 objNameLen
    u64 flags         bit 0x1 = 64-bit, 0x2 = debug, 0x10 = linux,
                      0x2000 = compressed
    u64 zero
    u64 decompressedSize  only meaningful when compressed

PTX decompression implements the documented NVIDIA LZ variant (Apache-2.0
reference: n-eiling/cuda-fatbin-decompression, (c) 2023 Niklas Eiling):
literal-run / match-run token stream with 12-bit back-references.
The decoder is verified byte-for-byte against NVIDIA-generated artifacts:
compressed payload from a -fatbin build decompressed here must equal the
payload of the same source built with -no-compress (see docs/ground-truth.md).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

from ._meta import (
    InputError, MAX_INPUT_BYTES, MAX_DECOMPRESSED_ENTRY_BYTES,
    MAX_FATBIN_ENTRIES, ensure_input_size,
)

FATBIN_MAGIC = 0xBA55ED50
KIND_PTX = 1
KIND_ELF = 2
FLAG_64BIT = 0x1
FLAG_DEBUG = 0x2
FLAG_LINUX = 0x10
FLAG_COMPRESS = 0x2000


@dataclass
class FatEntry:
    kind: int  # 1=PTX, 2=ELF
    version: Tuple = (1, 1)
    arch: int = 0
    flags: int = 0
    compressed_size: int = 0
    decompressed_size: int = 0
    payload: bytes = b""
    name: Optional[str] = None
    offset: int = 0  # offset of entry header within the fatbin
    compressed: bool = False

    @property
    def is_ptx(self) -> bool:
        return self.kind == KIND_PTX

    @property
    def is_elf(self) -> bool:
        return self.kind == KIND_ELF

    def text(self) -> str:
        """Decoded PTX text (PTX entries only); strips alignment NUL padding."""
        if not self.is_ptx:
            raise InputError("entry is not PTX")
        return self.payload.rstrip(b"\x00").decode("utf-8", errors="replace")


@dataclass
class Fatbin:
    entries: List[FatEntry] = field(default_factory=list)
    total_size: int = 0
    size_field: int = 0

    def ptx_entries(self) -> List[FatEntry]:
        return [e for e in self.entries if e.is_ptx]

    def elf_entries(self) -> List[FatEntry]:
        return [e for e in self.entries if e.is_elf]


def _decompress(input_: bytes, expected_size: int) -> bytes:
    """NVIDIA fatbin LZ decoder with explicit allocation/stream bounds."""
    if expected_size < 0 or expected_size > MAX_DECOMPRESSED_ENTRY_BYTES:
        raise InputError(
            f"fatbin decompressed size {expected_size} exceeds "
            f"{MAX_DECOMPRESSED_ENTRY_BYTES}-byte limit"
        )
    if expected_size == 0:
        return b""

    out = bytearray(expected_size + 64)
    ipos, opos = 0, 0
    isize = len(input_)
    while ipos < isize:
        if opos > expected_size:
            raise InputError("fatbin decompression overflow")

        nclen = (input_[ipos] & 0xF0) >> 4
        clen = 4 + (input_[ipos] & 0x0F)
        if nclen == 0x0F:
            while True:
                ipos += 1
                if ipos >= isize:
                    raise InputError("fatbin decompression: truncated literal length")
                nclen += input_[ipos]
                if input_[ipos] != 0xFF:
                    break
        ipos += 1
        if ipos + nclen > isize:
            raise InputError("fatbin decompression: literal run past end")
        if opos + nclen > expected_size:
            raise InputError("fatbin decompression: literal run exceeds declared size")
        out[opos:opos + nclen] = input_[ipos:ipos + nclen]
        ipos += nclen
        opos += nclen
        if ipos >= isize or opos >= expected_size:
            break

        if ipos + 2 > isize:
            raise InputError("fatbin decompression: truncated back-reference")
        back = input_[ipos] + (input_[ipos + 1] << 8)
        ipos += 2
        if back <= 0 or back > opos:
            raise InputError("fatbin decompression: invalid back-reference")
        if clen == 0x0F + 4:
            while True:
                if ipos >= isize:
                    raise InputError("fatbin decompression: truncated match length")
                extra = input_[ipos]
                clen += extra
                ipos += 1
                if extra != 0xFF:
                    break
        if opos + clen > expected_size:
            raise InputError("fatbin decompression: match exceeds declared size")
        if clen <= back:
            out[opos:opos + clen] = out[opos - back:opos - back + clen]
        else:
            for i in range(clen):
                out[opos + i] = out[opos + i - back]
        opos += clen

    if opos != expected_size:
        raise InputError(
            f"fatbin decompression size mismatch: expected {expected_size}, got {opos}"
        )
    return bytes(out[:opos])


def parse(data: bytes, strict: bool = False) -> Fatbin:
    """Parse a fatbin. strict=True raises when the magic is wrong."""
    ensure_input_size(data, "fatbin")
    if len(data) < 16:
        raise InputError(f"fatbin too small: {len(data)} bytes")
    magic, version, header_size, size = struct.unpack_from("<IHHQ", data, 0)
    if magic != FATBIN_MAGIC:
        raise InputError(f"bad fatbin magic 0x{magic:08x} (expected 0x{FATBIN_MAGIC:08x})")
    if strict and version != 1:
        raise InputError(f"unsupported fatbin version {version}")
    if header_size < 16 or header_size > len(data):
        raise InputError(f"invalid fatbin header size {header_size}")
    fb = Fatbin(total_size=len(data), size_field=size)
    off = header_size
    end = header_size + size
    if end > len(data):
        raise InputError(
            f"fatbin declared size {end} exceeds container length {len(data)}"
        )
    entry_count = 0
    while off + 12 <= end:
        entry_count += 1
        if entry_count > MAX_FATBIN_ENTRIES:
            raise InputError(
                f"fatbin has more than {MAX_FATBIN_ENTRIES} entries"
            )
        kind, ever, ehsize, esize = struct.unpack_from("<HHII", data, off)
        if ehsize < 56 or off + ehsize + esize > len(data):
            # last entry may be padded to 8; allow exact end
            if off + ehsize + esize != len(data):
                raise InputError(
                    f"fatbin entry at 0x{off:x}: header {ehsize}, size {esize} "
                    f"exceeds container")
        minor = major = 0
        comp_size = deco_size = 0
        unknown2 = 0
        flags = 0
        if ehsize >= 64:
            comp_size, unknown2, minor, major, arch = struct.unpack_from(
                "<IIHHI", data, off + 16)
            flags = struct.unpack_from("<Q", data, off + 40)[0]
        if ehsize >= 80:
            deco_size = struct.unpack_from("<Q", data, off + 56)[0]
        payload = data[off + ehsize: off + ehsize + esize]
        compressed = bool(flags & FLAG_COMPRESS) and kind == KIND_PTX and comp_size > 0
        if compressed:
            if comp_size > esize:
                raise InputError(
                    f"fatbin entry at 0x{off:x}: compressed size {comp_size} "
                    f"exceeds payload size {esize}"
                )
            # compressed stream lives in the first comp_size bytes (rest is pad)
            plain = _decompress(payload[:comp_size], deco_size)
            # byte-exact check vs NVIDIA uncompressed payload happens in tests;
            # here we keep the trailing zero-padding behavior of nvcc
            pad = (-len(plain)) % 8
            payload = plain + b"\0" * pad
        name = None
        if ehsize >= 40:
            name_off, name_len = struct.unpack_from("<II", data, off + 32)
            if name_off and name_len:
                raw = data[off + name_off: off + name_off + name_len]
                name = raw.rstrip(b"\0").decode("utf-8", errors="replace")
        fb.entries.append(FatEntry(
            kind=kind, version=((ever >> 8) & 0xFF, ever & 0xFF) if ever else (1, 1),
            arch=arch, flags=flags, compressed_size=comp_size if compressed else 0,
            decompressed_size=deco_size, payload=payload, name=name,
            offset=off, compressed=compressed))
        off += ehsize + esize
    if off != end:
        raise InputError(
            f"fatbin entries do not tile declared size: stopped at {off}, end {end}"
        )
    return fb


def parse_file(path: str) -> Fatbin:
    with open(path, "rb") as f:
        return parse(f.read())
