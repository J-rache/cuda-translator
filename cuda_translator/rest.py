"""REST API for cuda-translator (stdlib http.server, no dependencies).

Endpoints:
    GET  /api/health            -> {status, version, opencl_device}
    POST /api/analyze           -> structural report     (body: raw artifact)
    POST /api/translate         -> OpenCL C translation  (body: raw artifact)
    POST /api/translate-text    -> {ptx: "..."} JSON, same as translate
    POST /api/verify            -> numeric acceptance-seam verdict

Errors: 400 for bad input (InputError), 422 for fail-closed translation
(TranslationError), 500 only for genuine server faults.
"""
from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ._meta import VERSION, InputError, TranslationError, SERVER_NAME


class _Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVER_NAME}/{VERSION}"

    # -- helpers
    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise InputError("empty request body")
        return self.rfile.read(length)

    def log_message(self, fmt, *args):  # quiet by default
        pass

    # -- routes
    def do_GET(self):
        if self.path == "/api/health":
            device = _opencl_device_summary()
            self._send(200, {"status": "ok", "service": SERVER_NAME,
                             "version": VERSION, "opencl_device": device})
            return
        self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):
        try:
            if self.path == "/api/analyze":
                self._send(200, _analyze(self._read_body()))
            elif self.path == "/api/translate":
                self._send(200, _translate(self._read_body()))
            elif self.path == "/api/translate-text":
                obj = json.loads(self._read_body().decode("utf-8"))
                if not isinstance(obj.get("ptx"), str):
                    raise InputError('body must be JSON {"ptx": "..."}')
                from .pipeline import translate_ptx_text_strict
                self._send(200, translate_ptx_text_strict(obj["ptx"], "rest.ptx"))
            elif self.path == "/api/verify":
                from .verify import run_verification
                self._send(200, run_verification())
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except InputError as e:
            self._send(400, {"error": str(e)})
        except TranslationError as e:
            self._send(422, {"error": "fail-closed translation abort",
                             "detail": str(e), "kernel": e.kernel,
                             "opcode": e.opcode, "ptx_line": e.line})
        except Exception as e:  # genuine server fault
            self._send(500, {"error": str(e),
                             "trace": traceback.format_exc(limit=5)})


def _analyze(data: bytes):
    from .pipeline import analyze_input
    return analyze_input(data, "rest-upload")


def _translate(data: bytes):
    from .pipeline import translate_input_strict
    return translate_input_strict(data, "rest-upload")


def _opencl_device_summary() -> str:
    try:
        from . import opencl_runner
        cl, info, _, _ = opencl_runner.first_device()
        if cl is None:
            return "none"
        return f"{info.platform} / {info.device}"
    except Exception:
        return "probe-failed"


def serve(host: str = "127.0.0.1", port: int = 8377) -> None:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"{SERVER_NAME} REST API listening on http://{host}:{port}")
    print("  GET  /api/health")
    print("  POST /api/analyze        (raw artifact bytes)")
    print("  POST /api/translate      (raw artifact bytes)")
    print("  POST /api/translate-text (JSON {\"ptx\": ...})")
    print("  POST /api/verify")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
