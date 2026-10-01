"""Installed-package MCP smoke.

Exercises initialize, tools/list, health/device provenance, good translation,
fail-closed translation, analysis, malformed base64, and the hardware seam.
PASS and NO-DEVICE are both honest hardware outcomes; FAIL is never accepted.
"""
from __future__ import annotations

import base64
import json
import queue as queue_mod
import subprocess
import sys
import threading
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
    .read_text(encoding="utf-8")
)

ATOMIC_PTX = r"""
.version 7.8
.target sm_52
.address_size 64
.visible .entry atomic_test(
    .param .u64 p
)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<3>;
    ld.param.u64 %rd1, [p];
    cvta.to.global.u64 %rd2, %rd1;
    atom.global.add.u32 %r1, [%rd2], 1;
    ret;
}
"""


def main() -> int:
    p = subprocess.Popen(
        [sys.executable, "-m", "cuda_translator.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    q: "queue_mod.Queue" = queue_mod.Queue()

    def reader():
        for line in p.stdout:
            q.put(line.strip())
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    def rpc(obj, timeout=120):
        p.stdin.write(json.dumps(obj) + "\n")
        p.stdin.flush()
        if "id" not in obj or obj.get("id") is None:
            return None
        remaining = timeout
        while remaining > 0:
            try:
                line = q.get(timeout=1)
            except queue_mod.Empty:
                remaining -= 1
                continue
            if line is None:
                err = p.stderr.read()
                raise RuntimeError(f"MCP server closed stdout: {err}")
            msg = json.loads(line)
            if msg.get("id") == obj["id"]:
                return msg
        raise TimeoutError(f"no response to id={obj.get('id')}")

    try:
        r = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        info = r["result"]["serverInfo"]
        assert info["name"] == "cuda-translator"
        rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

        r = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = {t["name"] for t in r["result"]["tools"]}
        assert names >= {
            "health", "device_info", "analyze", "translate",
            "translate_text", "verify_numeric",
        }

        r = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "health", "arguments": {},
        }})
        payload = json.loads(r["result"]["content"][0]["text"])
        assert payload["status"] in {"ok", "degraded"}

        r = rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "device_info", "arguments": {},
        }})
        payload = json.loads(r["result"]["content"][0]["text"])
        assert payload["status"] in {"ok", "no-device"}

        r = rpc({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
            "name": "translate_text", "arguments": {"ptx": ATOMIC_PTX},
        }})
        payload = json.loads(r["result"]["content"][0]["text"])
        assert r["result"]["isError"] is True
        assert payload["error"] == "fail-closed translation abort"

        r = rpc({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
            "name": "translate_text", "arguments": {"ptx": GOOD_PTX},
        }})
        payload = json.loads(r["result"]["content"][0]["text"])
        assert r["result"]["isError"] is False
        assert set(payload["kernels"]) == {"vector_add", "saxpy"}

        r = rpc({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
            "name": "analyze", "arguments": {
                "input_b64": base64.b64encode(GOOD_PTX.encode("utf-8")).decode("ascii")
            },
        }})
        payload = json.loads(r["result"]["content"][0]["text"])
        assert payload["kind"] == "ptx"
        assert {k["name"] for k in payload["kernels"]} == {"vector_add", "saxpy"}

        r = rpc({"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {
            "name": "analyze", "arguments": {"input_b64": "%%%not-base64%%%"},
        }})
        assert r["result"]["isError"] is True

        r = rpc({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {
            "name": "verify_numeric", "arguments": {},
        }}, timeout=180)
        payload = json.loads(r["result"]["content"][0]["text"])
        assert r["result"]["isError"] is False
        assert payload["verdict"] in {"PASS", "NO-DEVICE"}
        if payload["verdict"] == "NO-DEVICE":
            assert payload.get("device") in {None, "none", "no-device"}

        print("MCP smoke: ALL OK | hardware verdict:", payload["verdict"])
        return 0
    finally:
        if p.stdin:
            try:
                p.stdin.close()
            except OSError:
                pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    raise SystemExit(main())
