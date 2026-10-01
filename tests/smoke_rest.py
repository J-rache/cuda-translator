"""Installed-package REST smoke using loopback only."""
from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import cuda_translator  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))

GOOD_PTX = (
    resources.files("cuda_translator")
    .joinpath("fixtures", "vector_add_sm75.ptx")
    .read_bytes()
)

ATOMIC_PTX = b""".version 7.8
.target sm_52
.address_size 64
.visible .entry atomic_test(.param .u64 p)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<3>;
    ld.param.u64 %rd1, [p];
    cvta.to.global.u64 %rd2, %rd1;
    atom.global.add.u32 %r1, [%rd2], 1;
    ret;
}
"""


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def request(port: int, method: str, path: str, body: bytes = b""):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/octet-stream"} if body else {}
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data.decode("utf-8"))


def main() -> int:
    port = free_port()
    code = (
        "from cuda_translator.rest import serve;"
        f"serve('127.0.0.1',{port})"
    )
    p = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(100):
            if p.poll() is not None:
                raise RuntimeError(p.stderr.read())
            try:
                status, payload = request(port, "GET", "/api/health")
                if status == 200:
                    break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.05)
        else:
            raise RuntimeError("REST server did not become ready")

        assert payload["status"] == "ok"
        assert payload["version"]

        status, payload = request(port, "POST", "/api/analyze", GOOD_PTX)
        assert status == 200
        assert payload["kind"] == "ptx"

        status, payload = request(port, "POST", "/api/translate", GOOD_PTX)
        assert status == 200
        assert set(payload["kernels"]) == {"vector_add", "saxpy"}

        status, payload = request(port, "POST", "/api/translate", ATOMIC_PTX)
        assert status == 422
        assert payload["error"] == "fail-closed translation abort"
        assert payload["kernel"] == "atomic_test"
        assert payload["ptx_line"] > 0

        print("REST smoke: ALL OK")
        return 0
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    raise SystemExit(main())
