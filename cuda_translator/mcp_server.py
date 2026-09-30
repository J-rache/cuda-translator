"""MCP stdio server for cuda-translator (JSON-RPC 2.0, MCP 2024-11-05).

Transport: newline-delimited JSON-RPC on stdin/stdout (the MCP stdio
convention). Implements initialize, tools/list, tools/call, and ping.
Tools:
    analyze(input_b64)         structural report for a CUDA artifact
    translate(input_b64)       fail-closed OpenCL C translation
    translate_text(ptx)        translate PTX text directly
    verify_numeric()           run the numeric acceptance seam
    list_tools()               self-description helper for humans
Binary payloads use base64 (MCP tools exchange JSON).
"""
from __future__ import annotations

import base64
import json
import sys
from typing import Any, Dict

from ._meta import VERSION, SERVER_NAME, MCP_PROTOCOL_VERSION
from ._meta import InputError, TranslationError

TOOLS = [
    {
        "name": "analyze",
        "description": "Structural analysis of a CUDA artifact (PTX text, "
                       "cubin, fatbin, or PE host binary). Provide the bytes "
                       "base64-encoded in 'input_b64'.",
        "inputSchema": {
            "type": "object",
            "properties": {"input_b64": {"type": "string"}},
            "required": ["input_b64"],
        },
    },
    {
        "name": "translate",
        "description": "Fail-closed translation of a CUDA artifact to OpenCL "
                       "C kernels (House IR). Provide bytes base64-encoded "
                       "in 'input_b64'.",
        "inputSchema": {
            "type": "object",
            "properties": {"input_b64": {"type": "string"}},
            "required": ["input_b64"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate PTX source text (field 'ptx') to OpenCL C. "
                       "Fails closed on unsupported instructions or control flow.",
        "inputSchema": {
            "type": "object",
            "properties": {"ptx": {"type": "string"}},
            "required": ["ptx"],
        },
    },
    {
        "name": "verify_numeric",
        "description": "Run the numeric acceptance seam: nvcc PTX -> House IR "
                       "-> OpenCL C -> OpenCL device -> exact-oracle compare. "
                       "Returns PASS / FAIL / NO-DEVICE honestly.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "analyze":
        data = base64.b64decode(args["input_b64"])
        from .pipeline import analyze_input
        return analyze_input(data, "mcp-upload")
    if name == "translate":
        data = base64.b64decode(args["input_b64"])
        from .pipeline import translate_input_strict
        return translate_input_strict(data, "mcp-upload")
    if name == "translate_text":
        from .pipeline import translate_ptx_text_strict
        return translate_ptx_text_strict(args["ptx"], "mcp.ptx")
    if name == "verify_numeric":
        from .verify import run_verification
        return run_verification()
    raise InputError(f"unknown tool {name!r}")


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    method = msg.get("method", "")
    msg_id = msg.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": VERSION},
        }}
    if method == "notifications/initialized" or method.startswith("notifications/"):
        return {}  # notifications get no response
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        try:
            result = _tool_call(name, args)
            payload = json.dumps(result, default=str)
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": payload}],
                "isError": False,
            }}
        except InputError as e:
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                "isError": True,
            }}
        except TranslationError as e:
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": json.dumps({
                    "error": "fail-closed translation abort", "detail": str(e),
                    "kernel": e.kernel, "opcode": e.opcode, "ptx_line": e.line})}],
                "isError": True,
            }}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}}
    if msg_id is None:
        return {}
    return {"jsonrpc": "2.0", "id": msg_id, "error": {
        "code": -32601, "message": f"method not found: {method}"}}


def serve_stdio() -> None:
    """Main loop: read newline-delimited JSON-RPC from stdin, write to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp = {"jsonrpc": "2.0", "id": None, "error": {
                "code": -32700, "message": f"parse error: {e}"}}
        else:
            resp = handle(msg)
        if resp:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve_stdio()
