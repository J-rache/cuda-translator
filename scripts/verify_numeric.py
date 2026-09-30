"""Acceptance seam driver: PTX -> House IR -> OpenCL C -> device -> numbers.

Proof model (per docs/ground-truth.md):
  * expected values come from EXACT integer oracles (numeric_oracles.py,
    cross-validated against native fmaf on 300k triples), never from float64
    evaluation;
  * vector_add must be bit-exact vs the f32 add oracle;
  * saxpy from the default nvcc build contracts to fma.rn.f32 -> bit-exact vs
    the FUSED oracle; the -fmad=false build stays unfused -> bit-exact vs the
    UNFUSED oracle;
  * if no OpenCL device exists, the verdict records it honestly (exit 3).

Exit codes: 0 verified, 1 failure, 3 no device on this machine.
"""
from __future__ import annotations

import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from cuda_translator import ptx  # noqa: E402
from cuda_translator.opencl import emit_program  # noqa: E402
from cuda_translator.ir import reachable_unsupported  # noqa: E402
from cuda_translator.numeric_oracles import (  # noqa: E402
    fma_rn_f32, f32_mul_rn, f32_add_rn, f32_add_rn as _add,
    f32_bits, bits_f32,
)
import opencl_runner  # noqa: E402

N = 1024
LOCAL = 256


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def build_references(xs, ys, alpha, n):
    """Exact per-lane expected values (integer arithmetic)."""
    add = [bits_f32(f32_add_rn(f32_bits(xs[i]), f32_bits(ys[i])))
           for i in range(n)]
    sax_fused = [bits_f32(fma_rn_f32(f32_bits(xs[i]), f32_bits(alpha),
                                     f32_bits(ys[i]))) for i in range(n)]
    sax_unfused = [bits_f32(f32_add_rn(f32_mul_rn(f32_bits(xs[i]),
                                                  f32_bits(alpha)),
                                       f32_bits(ys[i]))) for i in range(n)]
    return add, sax_fused, sax_unfused


def main() -> int:
    ptx_text = open(os.path.join(HERE, "examples/artifacts/vector_add_sm75.ptx"),
                    "rb").read().decode()
    res = ptx.parse_ptx(ptx_text, "vector_add_sm75.ptx")
    bad = {k.name: reachable_unsupported(k) for k in res.kernels}
    if any(bad.values()):
        print("FAIL-CLOSED: reachable unsupported ops:",
              {k: len(v) for k, v in bad.items() if v})
        return 1
    header, sources = emit_program(res)
    if set(sources) != {"vector_add", "saxpy"}:
        print("TRANSLATION incomplete:", sorted(sources))
        return 1

    import random
    random.seed(2027)
    xs = [f32(random.uniform(-1e4, 1e4)) for _ in range(N)]
    ys = [f32(random.uniform(-1e4, 1e4)) for _ in range(N)]
    alpha = f32(-3.75)
    add_ref, sax_fused_ref, sax_unfused_ref = build_references(xs, ys, alpha, N)

    a_buf = struct.pack(f"<{N}f", *xs)
    b_buf = struct.pack(f"<{N}f", *ys)

    report = {
        "pipeline": "nvcc PTX -> House IR -> OpenCL C",
        "kernels": sorted(sources),
        "ir_unsupported_reachable": {k: len(v) for k, v in bad.items()},
        "oracle": "exact integer f32 oracles (numeric_oracles.py)",
        "device": None,
        "results": {},
    }

    cl, dinfo, plat, dev = opencl_runner.first_device()
    if cl is None:
        report["device"] = "none"
        report["results"] = {
            "vector_add": {"status": "no-device"},
            "saxpy": {"status": "no-device"},
        }
        out = os.path.join(HERE, "examples", "artifacts", "numeric_verdict.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"NO OPENCL DEVICE on this machine; verdict written to {out}")
        return 3

    ctx, q = opencl_runner.make_context_and_queue(cl, plat, dev)
    report["device"] = (f"{dinfo.platform} / {dinfo.device} "
                        f"(driver {dinfo.driver}, {dinfo.compute_units} CUs)")

    # kernel param order: (a, b, c, n) — set args in DECLARATION order
    outs_add = opencl_runner.run_ndrange(
        cl, ctx, q, source=sources["vector_add"], kernel_name="vector_add",
        args=[a_buf, b_buf, bytes(N * 4), ("s32", N)],
        out_indices=[2], global_size=(N,), dev=dev)
    got_add = struct.unpack(f"<{N}f", outs_add["2"])  # arg position 2
    exact_add = sum(1 for g, w in zip(got_add, add_ref) if g == w)
    report["results"]["vector_add"] = {
        "status": "ran", "bit_exact": exact_add, "of": N,
        "pass": exact_add == N,
    }

    # kernel param order: (alpha, x, y, out, n)
    outs_sax = opencl_runner.run_ndrange(
        cl, ctx, q, source=sources["saxpy"], kernel_name="saxpy",
        args=[("f32", alpha), a_buf, b_buf, bytes(N * 4), ("s32", N)],
        out_indices=[3], global_size=(N,), dev=dev)
    got_sax = struct.unpack(f"<{N}f", outs_sax["3"])  # arg position 3
    ex_f = sum(1 for g, w in zip(got_sax, sax_fused_ref) if g == w)
    ex_u = sum(1 for g, w in zip(got_sax, sax_unfused_ref) if g == w)
    report["results"]["saxpy"] = {
        "status": "ran",
        "bit_exact_vs_fused": ex_f, "bit_exact_vs_unfused": ex_u, "of": N,
        "note": "default nvcc build contracts to fma.rn.f32",
        "pass": ex_f == N,
    }

    ok = report["results"]["vector_add"]["pass"] and report["results"]["saxpy"]["pass"]
    report["verdict"] = "PASS" if ok else "FAIL"
    out = os.path.join(HERE, "examples", "artifacts", "numeric_verdict.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
