"""MCP stdio smoke client: exercises initialize, tools/list, tools/call
(health/device provenance, translate_text good + fail-closed, analyze fatbin,
verify_numeric).

Notification-aware: notifications produce no response, so the client tracks
pending request ids and reads with a timeout instead of blocking forever.
Exit 0 = all smokes passed.
"""
import base64
import json
import subprocess
import sys
import threading
import queue as queue_mod
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)


def main() -> int:
    p = subprocess.Popen(
        [sys.executable, "cuda-translator.py", "mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

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
            return None  # notification: no response by protocol
        deadline_hi = timeout
        while deadline_hi > 0:
            try:
                line = q.get(timeout=1)
            except queue_mod.Empty:
                deadline_hi -= 1
                continue
            if line is None:
                raise RuntimeError("MCP server closed stdout")
            msg = json.loads(line)
            if msg.get("id") == obj["id"]:
                return msg
            # keep draining responses to older requests if any
        raise TimeoutError(f"no response to id={obj.get('id')}")

    ok = True

    r = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    info = r["result"]["serverInfo"]
    print("initialize ->", info, "| proto:", r["result"]["protocolVersion"])
    assert info["name"] == "cuda-translator"
    rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

    r = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in r["result"]["tools"]]
    print("tools:", names)
    assert set(names) >= {"health", "device_info", "analyze", "translate",
                          "translate_text", "verify_numeric"}

    r = rpc({"jsonrpc": "2.0", "id": 30, "method": "tools/call", "params": {
        "name": "health", "arguments": {}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    print("health ->", payload)
    assert payload["status"] == "ok"

    r = rpc({"jsonrpc": "2.0", "id": 31, "method": "tools/call", "params": {
        "name": "device_info", "arguments": {}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    print("device_info ->", payload)
    assert payload["status"] in {"ok", "no-device"}

    hist = open("examples/artifacts/histogram_sm75.ptx", "rb").read().decode()
    r = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "translate_text", "arguments": {"ptx": hist}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    print("histogram -> isError:", r["result"]["isError"],
          "|", payload.get("detail", "")[:70])
    assert r["result"]["isError"] is True

    good = open("examples/artifacts/vector_add_sm75.ptx", "rb").read().decode()
    r = rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
        "name": "translate_text", "arguments": {"ptx": good}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    print("vector_add -> isError:", r["result"]["isError"],
          "| kernels:", sorted(payload["kernels"]))
    assert r["result"]["isError"] is False and set(payload["kernels"]) == \
        {"vector_add", "saxpy"}

    fat = open("examples/artifacts/vector_add_sm75.fatbin", "rb").read()
    r = rpc({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
        "name": "analyze", "arguments": {
            "input_b64": base64.b64encode(fat).decode()}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    kinds = [(e["kind"], e.get("arch")) for e in payload["entries"]]
    print("analyze(fatbin) -> kind:", payload["kind"], "| entries:", kinds)
    assert payload["kind"] == "fatbin" and ("elf", "sm_75") in kinds

    r = rpc({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
        "name": "verify_numeric", "arguments": {}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    print("verify_numeric -> isError:", r["result"]["isError"],
          "| verdict:", payload["verdict"], "| device:", payload["device"])
    assert payload["verdict"] == "PASS"

    p.stdin.close()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    print("MCP smoke: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
