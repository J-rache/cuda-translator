"""Numeric acceptance-seam verification (package API).

Runs: nvcc PTX -> House IR -> OpenCL C -> OpenCL device -> exact-oracle
comparison. Returns a verdict dict; exit codes are chosen by callers.
"""
from __future__ import annotations

import os
import random
import struct
from typing import Any, Dict

from . import ptx
from .ir import reachable_unsupported
from .numeric_oracles import fma_rn_f32, f32_add_rn, f32_mul_rn, f32_bits, bits_f32
from .opencl import emit_program

N = 1024


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _default_ptx_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "examples", "artifacts", "vector_add_sm75.ptx")


def build_references(xs, ys, alpha, n):
    add = [bits_f32(f32_add_rn(f32_bits(xs[i]), f32_bits(ys[i]))) for i in range(n)]
    fused = [bits_f32(fma_rn_f32(f32_bits(xs[i]), f32_bits(alpha), f32_bits(ys[i])))
             for i in range(n)]
    unfused = [bits_f32(f32_add_rn(f32_mul_rn(f32_bits(xs[i]), f32_bits(alpha)),
                                   f32_bits(ys[i]))) for i in range(n)]
    return add, fused, unfused


def run_verification(ptx_path: str = None, n: int = N,
                     seed: int = 2027) -> Dict[str, Any]:
    """Execute the full acceptance seam end-to-end.

    Honesty contract: if no OpenCL device exists the verdict says so
    (verdict 'NO-DEVICE') instead of faking a pass.
    """
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(here, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(here, "scripts"))
    import opencl_runner

    ptx_path = ptx_path or _default_ptx_path()
    ptx_text = open(ptx_path, "rb").read().decode()
    res = ptx.parse_ptx(ptx_text, os.path.basename(ptx_path))
    bad = {k.name: reachable_unsupported(k) for k in res.kernels}
    if any(bad.values()):
        return {"verdict": "FAIL", "reason": "reachable unsupported ops",
                "ir_unsupported_reachable": {k: len(v) for k, v in bad.items() if v}}
    header, sources = emit_program(res)
    if set(sources) != {"vector_add", "saxpy"}:
        return {"verdict": "FAIL", "reason": f"translation incomplete: {sorted(sources)}"}

    random.seed(seed)
    xs = [f32(random.uniform(-1e4, 1e4)) for _ in range(n)]
    ys = [f32(random.uniform(-1e4, 1e4)) for _ in range(n)]
    alpha = f32(-3.75)
    add_ref, fused_ref, unfused_ref = build_references(xs, ys, alpha, n)
    a_buf = struct.pack(f"<{n}f", *xs)
    b_buf = struct.pack(f"<{n}f", *ys)

    report: Dict[str, Any] = {
        "pipeline": "nvcc PTX -> House IR -> OpenCL C",
        "kernels": sorted(sources),
        "oracle": "exact integer f32 oracles (numeric_oracles.py)",
        "ir_unsupported_reachable": {k: len(v) for k, v in bad.items()},
        "device": None,
        "results": {},
    }

    cl, dinfo, plat, dev = opencl_runner.first_device()
    if cl is None:
        report["device"] = "none"
        report["results"] = {"vector_add": {"status": "no-device"},
                             "saxpy": {"status": "no-device"}}
        report["verdict"] = "NO-DEVICE"
        return report

    ctx, q = opencl_runner.make_context_and_queue(cl, plat, dev)
    report["device"] = (f"{dinfo.platform} / {dinfo.device} "
                        f"(driver {dinfo.driver!r}, {dinfo.compute_units} CUs)")

    outs_add = opencl_runner.run_ndrange(
        cl, ctx, q, source=sources["vector_add"], kernel_name="vector_add",
        args=[a_buf, b_buf, bytes(n * 4), ("s32", n)],
        out_indices=[2], global_size=(n,), dev=dev)
    got_add = struct.unpack(f"<{n}f", outs_add["2"])
    exact_add = sum(1 for g, w in zip(got_add, add_ref) if g == w)
    report["results"]["vector_add"] = {"status": "ran", "bit_exact": exact_add,
                                       "of": n, "pass": exact_add == n}

    outs_sax = opencl_runner.run_ndrange(
        cl, ctx, q, source=sources["saxpy"], kernel_name="saxpy",
        args=[("f32", alpha), a_buf, b_buf, bytes(n * 4), ("s32", n)],
        out_indices=[3], global_size=(n,), dev=dev)
    got_sax = struct.unpack(f"<{n}f", outs_sax["3"])
    ex_f = sum(1 for g, w in zip(got_sax, fused_ref) if g == w)
    ex_u = sum(1 for g, w in zip(got_sax, unfused_ref) if g == w)
    report["results"]["saxpy"] = {
        "status": "ran", "bit_exact_vs_fused": ex_f,
        "bit_exact_vs_unfused": ex_u, "of": n,
        "note": "default nvcc build contracts to fma.rn.f32",
        "pass": ex_f == n,
    }

    ok = report["results"]["vector_add"]["pass"] and report["results"]["saxpy"]["pass"]
    report["verdict"] = "PASS" if ok else "FAIL"
    return report
