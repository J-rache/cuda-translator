"""Reproduce and execute the public synthetic stress workload.

Exit codes:
  0 = translated and device outputs exactly match the independent CPU oracle
  1 = translation/device execution occurred but proof failed
  3 = no usable OpenCL device (reported honestly, never converted to PASS)
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cuda_translator import opencl_runner, pipeline  # noqa: E402
from cuda_translator.numeric_oracles import (  # noqa: E402
    bits_f32,
    f32_add_rn,
    f32_bits,
    fma_rn_f32,
)

FIXTURE = ROOT / "examples" / "artifacts" / "public_stress_sm52.ptx"
EXPECTED_FIXTURE_SHA256 = "925cbc50d8357f77373889a9c301f0297be91d7caf5c24a364908bc307aa7951"
N = 64
LOCAL = 32


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_oracle():
    inputs_f32 = [
        struct.unpack("<f", struct.pack("<f", ((i - 31) * 0.375 + (i % 7) * 0.03125)))[0]
        for i in range(N)
    ]
    inputs_i8 = [((i * 17 + 3) % 255) - 127 for i in range(N)]
    inputs_i16 = [((i * 997 + 111) % 60001) - 30000 for i in range(N)]

    two = f32_bits(2.0)
    one = f32_bits(1.0)
    zero = f32_bits(0.0)

    out_f32_bits = []
    out_f64 = []
    out_i32 = []
    for x, a, b in zip(inputs_f32, inputs_i8, inputs_i16):
        fused = fma_rn_f32(f32_bits(x), two, one)
        acc = zero
        for _ in range(3):
            acc = f32_add_rn(acc, fused)
        out_f32_bits.append(acc)
        out_f64.append(float(bits_f32(fused)) / 2.0)
        out_i32.append(a + b)

    in_f32 = struct.pack(f"<{N}f", *inputs_f32)
    in_i8 = struct.pack(f"<{N}b", *inputs_i8)
    in_i16 = struct.pack(f"<{N}h", *inputs_i16)
    want_f32 = struct.pack(f"<{N}I", *out_f32_bits)
    want_f64 = struct.pack(f"<{N}d", *out_f64)
    want_i32 = struct.pack(f"<{N}i", *out_i32)
    return in_f32, in_i8, in_i16, want_f32, want_f64, want_i32


def emit(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    fixture = FIXTURE.read_bytes()
    fixture_sha = sha256(fixture)
    if fixture_sha != EXPECTED_FIXTURE_SHA256:
        emit({
            "verdict": "FAIL",
            "reason": "fixture hash mismatch",
            "expected_fixture_sha256": EXPECTED_FIXTURE_SHA256,
            "actual_fixture_sha256": fixture_sha,
        })
        return 1

    translated = pipeline.translate_input_strict(fixture, FIXTURE.name)
    source = translated["kernels"].get("public_stress")
    if not source:
        emit({
            "verdict": "FAIL",
            "reason": "public_stress kernel was not emitted",
            "fixture_sha256": fixture_sha,
        })
        return 1

    cl, info, plat, dev = opencl_runner.first_device()
    if cl is None or info is None or plat is None or dev is None:
        emit({
            "verdict": "NO-DEVICE",
            "fixture_sha256": fixture_sha,
            "translation_sha256": sha256(source.encode("utf-8")),
            "device": None,
        })
        return 3

    in_f32, in_i8, in_i16, want_f32, want_f64, want_i32 = build_oracle()
    out_f32 = bytes(N * 4)
    out_f64 = bytes(N * 8)
    out_i32 = bytes(N * 4)

    ctx = q = None
    try:
        ctx, q = opencl_runner.make_context_and_queue(cl, plat, dev)
        got = opencl_runner.run_ndrange(
            cl,
            ctx,
            q,
            source,
            "public_stress",
            [
                in_f32,
                in_i8,
                in_i16,
                out_f32,
                out_f64,
                out_i32,
                ("u32", N),
                ("local", LOCAL * 4),
                ("u64", LOCAL * 4),
            ],
            out_indices=[3, 4, 5],
            global_size=(N,),
            dev=dev,
            local_size=(LOCAL,),
        )
    except Exception as exc:
        emit({
            "verdict": "FAIL",
            "fixture_sha256": fixture_sha,
            "device": {
                "platform": info.platform,
                "device": info.device,
                "driver": info.driver,
            },
            "error": str(exc),
        })
        return 1
    finally:
        if q:
            cl.lib.clReleaseCommandQueue(q)
        if ctx:
            cl.lib.clReleaseContext(ctx)

    comparisons = {
        "out_f32": got["3"] == want_f32,
        "out_f64": got["4"] == want_f64,
        "out_i32": got["5"] == want_i32,
    }
    payload = {
        "verdict": "PASS" if all(comparisons.values()) else "FAIL",
        "fixture_sha256": fixture_sha,
        "fixture_bytes": len(fixture),
        "translation_sha256": sha256(source.encode("utf-8")),
        "device": {
            "platform": info.platform,
            "device": info.device,
            "driver": info.driver,
            "compute_units": info.compute_units,
        },
        "geometry": {"global": [N], "local": [LOCAL], "dynamic_shared_bytes": LOCAL * 4},
        "comparisons": comparisons,
        "output_sha256": {
            "out_f32": sha256(got["3"]),
            "out_f64": sha256(got["4"]),
            "out_i32": sha256(got["5"]),
        },
        "oracle_sha256": {
            "out_f32": sha256(want_f32),
            "out_f64": sha256(want_f64),
            "out_i32": sha256(want_i32),
        },
    }
    emit(payload)
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
