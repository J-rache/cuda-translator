# Public reproducible stress workload

The repository includes a deterministic synthetic PTX workload so an unrelated user can reproduce a nontrivial translation case without access to private House Field source.

This is a **reproducibility complement**, not a claim that the synthetic workload is equivalent to House Field or to an NVIDIA-compiler conformance suite.

## Files

- Generator: tools/gen_public_stress.py
- Checked-in fixture: examples/artifacts/public_stress_sm52.ptx
- Device/oracle verifier: scripts/verify_public_stress.py

The checked-in fixture is 2,084 bytes with SHA-256:

925cbc50d8357f77373889a9c301f0297be91d7caf5c24a364908bc307aa7951

## Regenerate and check

    python tools/gen_public_stress.py
    python tools/gen_public_stress.py --check

The second command fails if the checked-in fixture differs byte-for-byte from the generator output.

## What it exercises

The single public_stress kernel intentionally combines:

- multiple global pointer parameters;
- mixed-width global accesses (f32, f64, s8, s16, s32);
- signed 8/16-bit interpretation and conversion;
- mul.wide byte-address formation;
- mad.lo.u32;
- explicit fma.rn.f32;
- div.rn.f64;
- dynamic extern shared memory;
- typed shared-memory store/load;
- bar.sync 0;
- a forward bounds branch;
- a backward loop;
- deterministic output mutation.

No new translator semantic was added to make this fixture pass.

## Translate

    python cuda-translator.py analyze examples/artifacts/public_stress_sm52.ptx
    python cuda-translator.py translate examples/artifacts/public_stress_sm52.ptx -o build/public-stress

Strict translation emits exactly one kernel and no refusal.

## Independent comparison

    python scripts/verify_public_stress.py

Exit contract:

- 0 — device execution completed and every output buffer matched the CPU oracle byte-for-byte;
- 1 — translation/execution/proof failure;
- 3 — no usable OpenCL device. This is **NO-DEVICE**, not PASS.

The CPU oracle independently constructs expected outputs. f32 FMA and repeated f32 additions use the repository's integer-arithmetic IEEE-754 oracles rather than host-language fused arithmetic.

## Recorded AMD acceptance

On 2026-10-01 the public fixture executed on:

- Platform: AMD Accelerated Parallel Processing
- Device: gfx90c
- Driver: 3584.0 (PAL,HSAIL)
- Reported compute units: 8
- Global/local geometry: 64 / 32
- Dynamic shared memory: 128 bytes

Result: **PASS**. Device output SHA-256 equaled oracle SHA-256 for all outputs:

| Output | SHA-256 |
|---|---|
| f32 | 1999ef349a2639a57adcf1fb9ed185eb4f1e840764f1211a5d63a47fe9576f12 |
| f64 | 648165a5aa0d8469bae8be3ad44b42ddadcddad533e706a835b67b98728ffd94 |
| i32 | 69c925d20ad23f32b956e2f949589d924200866a4aed28acefc1fe5623f62ae0 |

The emitted OpenCL kernel source for that fixture had SHA-256:

a97e38e4f9cb7371771abdaedf47f0ee5ca54a12dbbc4d30ac27e0876e59f3a1

Hardware receipts remain device-specific evidence. Hosted CI is allowed to report NO-DEVICE and must never relabel that condition as GPU verification.
