"""cuda-translator command line interface.

Subcommands:
  analyze <file>            structural analysis of PTX/cubin/fatbin/PE
  translate <file> [-o dir] fail-closed translation to OpenCL C (+ IR dump)
  verify [--ptx file]       run the numeric acceptance seam (device or honest no-device)
  serve [--port N]          start the REST API
  mcp                       serve the MCP stdio server
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from ._meta import VERSION, DEFAULT_REST_PORT, InputError, TranslationError


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def cmd_analyze(args) -> int:
    from .pipeline import analyze_input
    data = _read(args.file)
    try:
        report = analyze_input(data, os.path.basename(args.file))
    except InputError as e:
        print(json.dumps({"error": str(e)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0


def _kernel_output_filename(name: str) -> str:
    """Stable Windows-safe filename; kernel symbol in source is unchanged."""
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    if len(clean) <= 96:
        return clean + ".cl"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return clean[:64] + "-" + digest + ".cl"


def cmd_translate(args) -> int:
    from .pipeline import translate_input_strict
    data = _read(args.file)
    try:
        result = translate_input_strict(data, os.path.basename(args.file))
    except (InputError, TranslationError) as e:
        print(json.dumps({"error": str(e)}, indent=2))
        return 2
    payload = {
        "input": result["input"],
        "kernels": result["kernels"],
        "not_translated": result["not_translated"],
    }
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload["written"] = []
        for name, src in result["kernels"].items():
            path = os.path.join(args.out, _kernel_output_filename(name))
            with open(path, "w", encoding="utf-8") as f:
                f.write(result["opencl_header"] + "\n" + src + "\n")
            payload.setdefault("written", []).append(path)
        ir_path = os.path.join(args.out, "house_ir.txt")
        with open(ir_path, "w", encoding="utf-8") as f:
            f.write(result["ir"] + "\n")
        payload["written"].append(ir_path)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_verify(args) -> int:
    from .verify import run_verification
    verdict = run_verification(ptx_path=args.ptx)
    print(json.dumps(verdict, indent=2, default=str))
    if verdict.get("verdict") == "PASS":
        return 0
    if verdict.get("verdict") == "NO-DEVICE":
        return 3
    return 1


def cmd_serve(args) -> int:
    from .rest import serve
    serve(args.host, args.port)
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import serve_stdio
    serve_stdio()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cuda-translator",
        description="Translate CUDA binaries (PTX/fatbin/cubin/PE) into "
                    "portable OpenCL C and inspection artifacts.")
    p.add_argument("--version", action="version", version=f"cuda-translator {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="structural analysis of a CUDA artifact")
    a.add_argument("file")
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("translate", help="fail-closed translation to OpenCL C")
    t.add_argument("file")
    t.add_argument("-o", "--out", default=None, help="write kernels as .cl files")
    t.set_defaults(func=cmd_translate)

    v = sub.add_parser("verify", help="run the numeric acceptance seam")
    v.add_argument("--ptx", default=None,
                   help="PTX fixture (default: bundled nvcc-generated vector_add)")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("serve", help="start the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_REST_PORT)
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("mcp", help="run the MCP stdio server")
    m.set_defaults(func=cmd_mcp)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
