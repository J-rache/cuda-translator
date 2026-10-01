"""Pipeline facade: analyze (lossless) and translate (fail closed) inputs.

analyze_input: parse any supported input, return full diagnostic report.
translate_input: parse + emit OpenCL C; raises TranslationError on any
reachable unsupported instruction or unprovable control flow (IR rule 1).

Supported input kinds (auto-detected):
  "ptx"     — PTX text
  "fatbin"  — CUDA fatbin container (PTX entries are analyzed)
  "cubin"   — ELF cubin (structure report; SASS is out of scope by design)
  "pe"      — PE/COFF host binary (embedded fatbins are extracted and parsed)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import fatbin as _fatbin, elf as _elf, pe as _pe
from ._meta import InputError, ensure_input_size
from .ir import TranslationResult
from .opencl import emit_program, emit_kernel
from .ptx import parse_ptx, ir_text
from .ir import reachable_unsupported
from .opencl import TranslationAbort
from ._meta import TranslationError


def detect_input(data: bytes) -> str:
    if data[:2] == b"MZ":
        return "pe"
    if data[:4] == b"\x7fELF":
        return "cubin"
    if len(data) >= 16:
        magic, version, hs, size = None, None, None, None
        import struct
        if len(data) >= 16:
            magic, version, hs, size = struct.unpack_from("<IHHQ", data, 0)
        if magic == 0xBA55ED50:
            return "fatbin"
    head = data[:4096]
    # PTX may open with a comment banner, so scan past it; require a PTX-only
    # directive to avoid false positives on random text.
    if (b".version" in head and (b".target" in head or b".entry" in head)) \
            or b".visible .entry" in head or b".entry" in head:
        return "ptx"
    raise InputError("unrecognized input: not PTX, cubin, fatbin, or PE")


def analyze_input(data: bytes, name: str = "<input>") -> Dict[str, Any]:
    """Lossless analysis: parse, report, never raises for weird-but-parseable
    content (parse errors raise InputError)."""
    ensure_input_size(data, name)
    kind = detect_input(data)
    report: Dict[str, Any] = {"input": name, "kind": kind, "diagnostics": []}
    if kind == "ptx":
        res = parse_ptx(data.decode("utf-8", errors="replace"), name)
        report.update(_report_from_translation_result(res))
    elif kind == "fatbin":
        fb = _fatbin.parse(data)
        entries = []
        for e in fb.entries:
            ed = {
                "kind": "ptx" if e.is_ptx else "elf",
                "arch": f"sm_{e.arch}" if e.arch else None,
                "ptx_isa": f"{e.version[0]}.{e.version[1]}" if e.is_ptx else None,
                "flags": e.flags,
                "compressed": e.compressed,
                "payload_bytes": len(e.payload),
                "offset": e.offset,
            }
            if e.is_ptx:
                try:
                    res = parse_ptx(e.text(), name)
                    ed["kernels"] = [k.name for k in res.kernels]
                    ed["diagnostics"] = res.diagnostics
                    ver = res.notes.get("ptx_version")
                    if ver:
                        ed["ptx_isa"] = f"{ver[0]}.{ver[1]}"
                except InputError as ex:
                    ed["diagnostics"] = [{"severity": "error", "message": str(ex)}]
            entries.append(ed)
        report["entries"] = entries
        report["diagnostics"] = []
    elif kind == "cubin":
        cub = _elf.parse(data)
        report.update({
            "arch": cub.arch,
            "sections": [s.name for s in cub.sections],
            "entry_points": [s.name for s in cub.entry_symbols()],
        })
    elif kind == "pe":
        scan = _pe.scan(data)
        report.update({
            "pe_arch": scan.arch,
            "sections": scan.sections,
            "embedded_fatbins": [
                {"offset": h.offset, "size": h.size,
                 "source_section": h.source_section}
                for h in scan.hits
            ],
        })
        # descend into embedded fatbins
        fat_reports = []
        for h in scan.hits:
            fb = h.parse()
            for e in fb.entries:
                ed = {"source_section": h.source_section,
                      "kind": "ptx" if e.is_ptx else "elf",
                      "arch": f"sm_{e.arch}" if e.arch else None}
                if e.is_ptx:
                    try:
                        r = parse_ptx(e.text(), name)
                        ed["kernels"] = [k.name for r2 in [r] for k in r2.kernels]
                        ed["diagnostics"] = r.diagnostics
                    except InputError as ex:
                        ed["diagnostics"] = [{"severity": "error", "message": str(ex)}]
                fat_reports.append(ed)
        report["embedded"] = fat_reports
    return report


def _report_from_translation_result(res: TranslationResult) -> Dict[str, Any]:
    kernels = []
    for k in res.kernels:
        bad = reachable_unsupported(k)
        kernels.append({
            "name": k.name,
            "params": [{"name": p.name, "dtype": p.dtype,
                        "pointer": p.is_pointer} for p in k.params],
            "instructions": len(k.body),
            "translatable": not bad,
            "reachable_unsupported": [
                {"op": b.op, "line": b.line} for b in bad],
        })
    return {
        "ptx": {
            "version": res.notes.get("ptx_version"),
            "target": res.notes.get("target"),
            "address_size": res.notes.get("address_size"),
        },
        "kernels": kernels,
        "diagnostics": res.diagnostics,
        "ir": ir_text(res),
    }


def translate_input(data: bytes, name: str = "<input>",
                    kernel: Optional[str] = None) -> Dict[str, Any]:
    """Fail-closed translation to OpenCL C. Raises TranslationError."""
    ensure_input_size(data, name)
    kind = detect_input(data)
    if kind == "pe":
        scan = _pe.scan(data)
        if not scan.hits:
            raise InputError("no embedded fatbin found in PE image")
        data = scan.hits[0].data
        kind = "fatbin"
    if kind == "fatbin":
        fb = _fatbin.parse(data)
        ptx_entries = fb.ptx_entries()
        if not ptx_entries:
            raise InputError("fatbin has no PTX entries")
        text = "\n".join(e.text() for e in ptx_entries)
        data = text.encode("utf-8")  # feed the extracted PTX, not the container
        kind = "ptx"
    if kind != "ptx":
        raise InputError(f"kind {kind} not translatable (only PTX/fatbin/PE)")
    res = parse_ptx(data.decode("utf-8", errors="replace"), name)
    header, sources = emit_program(res)
    not_translated = [n for n in header.splitlines() if "NOT TRANSLATED" in n]
    return {
        "input": name,
        "kind": kind,
        "opencl_header": header,
        "kernels": sources,
        "not_translated": not_translated,
        "ir": ir_text(res),
        "diagnostics": res.diagnostics,
    }


def translate_ptx_text(text: str, name: str = "<ptx>",
                       kernel: Optional[str] = None) -> Dict[str, Any]:
    return translate_input(text.encode("utf-8"), name, kernel)


def translate_input_strict(data: bytes, name: str = "<input>") -> Dict[str, Any]:
    """Strict variant for API/MCP surfaces: ANY kernel refusing translation
    raises TranslationError (HTTP 422 / MCP isError) instead of returning a
    partial success that could be mistaken for complete."""
    import re as _re
    result = translate_input(data, name)
    if result["not_translated"]:
        first = result["not_translated"][0]
        km = _re.search(r"kernel (\S+):", first)
        om = _re.search(r"opcode ([^,\s]+), PTX line (\d+)", first)
        kname = km.group(1) if km else ""
        opcode = om.group(1) if om else ""
        line = int(om.group(2)) if om else 0
        raise TranslationError(
            f"fail-closed: {len(result['not_translated'])} kernel(s) refused; "
            f"{first}", kernel=kname, opcode=opcode, line=line)
    return result


def translate_ptx_text_strict(text: str, name: str = "<ptx>") -> Dict[str, Any]:
    return translate_input_strict(text.encode("utf-8"), name)
