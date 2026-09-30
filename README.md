# cuda-translator

A free-standing app that translates CUDA binaries into artifacts any other
application can use — and proves the translation numerically on real,
non-NVIDIA GPUs.

**Status: v0.1.0 pre-release.** This is a deliberately bounded translator,
not a drop-in replacement for the CUDA driver or a general SASS decompiler.
Unsupported or unprovable semantics fail closed instead of emitting an
approximate translation.

```
CUDA container (PTX / fatbin / cubin / PE host binary)
        ↓  frontends (container parsers + PTX parser)
   House GPU IR  (neutral, fail-closed, address provenance explicit)
        ↓  backend
   OpenCL C
        ↓  ctypes ICD runner (no dependencies)
   GPU execution  →  bit-exact compare vs exact integer oracles
```

## What is proven (machine truth, not narrative)

All fixtures are real NVIDIA compiler output; provenance and oracle commands
are in [docs/ground-truth.md](docs/ground-truth.md).

| Seam | Proof |
|---|---|
| fatbin parser + custom LZ decoder | compressed PTX entry decompresses **byte-for-byte** equal to NVIDIA's own `-no-compress` fatbin of the same source |
| fatbin ELF entry | byte-identical to the standalone `nvcc -cubin` file (sha256 match) |
| PE scanner | finds both embedded fatbins in a real nvcc-built host exe; the 4928-byte one is byte-identical to `nvcc -fatbin` output |
| ELF cubin parser | `e_machine`/`e_flags` read from real bytes; `sm_50…sm_90` cubins all decode `sm_NN` + both entry symbols |
| PTX frontend | zero diagnostics on nvcc **and** NVRTC output (two independent NVIDIA compilers); pointer params and per-load address provenance fully resolved |
| OpenCL backend | output **compiled and executed by two independent non-NVIDIA OpenCL stacks**: Intel Iris Xe (96 reported OpenCL CUs) and AMD gfx90c (8 reported OpenCL CUs) |
| Numeric acceptance | `vector_add` **1024/1024 bit-exact**; `saxpy` **1024/1024 bit-exact vs the fused-FMA oracle** (nvcc emits `fma.rn.f32`), and provably *not* unfused |
| FMA oracles | exact integer-arithmetic `fma.rn.f32` matches native `fmaf` on 300k random triples; fused≠unfused on ~25% of triples (real discriminating inputs) |
| Differential semantics | same source, `-fmad=false` build: 0 `fma` ops in PTX, interpreter matches the **unfused** oracle 1024/1024 — the tool preserves each binary's own float contract |
| Fail-closed | `atomicAdd` kernel refused with exact opcode + PTX line; REST returns **HTTP 422**, MCP returns `isError: true`; nothing wrong-but-compiling is ever emitted |
| Data-dependent CFG | `bin_classify` (nested guards from `&` of comparisons) parses and translates cleanly |
| Real Numba/NVVM stress kernel | 452 KB / 6,785-line House Field PTX with dynamic shared memory, barriers, FP64 division and loops compiles on AMD `gfx90c`; captured NVIDIA CUDA launches replay **byte-identically for every input, scratch and output buffer** at 9, 128 and 256 nodes ([receipt](docs/house-field-crossvendor-20260929.md)) |

## Install

No third-party packages required. Python 3.10+.

```bat
pip install .
cuda-translator --help
```

or run from the checkout:

```bat
python cuda-translator.py --help
```

## CLI

```bat
:: structural analysis (PTX, cubin, fatbin, or PE)
python cuda-translator.py analyze examples\artifacts\vector_add_sm75.fatbin

:: fail-closed translation to OpenCL C (writes .cl files + House IR dump)
python cuda-translator.py translate examples\artifacts\vector_add_sm75.ptx -o build\cl

:: numeric acceptance seam (PASS / FAIL / NO-DEVICE - honestly)
python cuda-translator.py verify

:: REST API on http://127.0.0.1:8377
python cuda-translator.py serve

:: MCP stdio server (JSON-RPC 2.0)
python cuda-translator.py mcp
```

## REST API

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/api/health` | – | service + detected OpenCL device |
| POST | `/api/analyze` | raw artifact bytes | structural report |
| POST | `/api/translate` | raw artifact bytes | **422** on fail-closed |
| POST | `/api/translate-text` | `{"ptx": "..."}` | **422** on fail-closed |
| POST | `/api/verify` | – | runs the full acceptance seam |

## MCP server

`python cuda-translator.py mcp` speaks MCP 2024-11-05 over stdio:

* `health` — `{}` cheap service/device readiness
* `device_info` — `{}` detected OpenCL target provenance
* `analyze` — `{"input_b64": ...}` structural report
* `translate` — `{"input_b64": ...}` OpenCL C (fail-closed → `isError`)
* `translate_text` — `{"ptx": ...}`
* `verify_numeric` — `{}` → PASS / FAIL / NO-DEVICE

## Library

```python
from cuda_translator import analyze_input, translate_input, translate_ptx_text

report = analyze_input(open("kernel.fatbin", "rb").read())
out = translate_input(open("kernel.ptx", "rb").read())
print(out["kernels"].keys())     # translatable kernels
print(out["not_translated"])     # refused kernels, with reasons
```

## Security

The REST server has no authentication or TLS and binds to loopback by default.
Do not expose it directly to an untrusted network. CUDA/PTX inputs can cause
compiler and GPU work; for hostile inputs, run the translator with least
privilege and isolation. See [SECURITY.md](SECURITY.md).

## Honest limits (current, deliberate)

* **SASS is not decompiled.** cubin analysis is structural (sections,
  symbols, arch); translation rides the PTX entry, exactly like the CUDA
  driver's JIT when PTX is present. A fatbin with no PTX cannot be
  translated and says so.
* **Control flow:** direct PTX branches to known labels, including backward
  loop edges, lower directly to OpenCL C `goto` with all House registers
  declared at kernel scope. Indirect/unresolved control flow still fails
  closed.
* **Shared-memory subset:** unsized `.extern .shared`, typed `ld/st.shared`,
  `%dynamic_smem_size`, and `bar.sync 0` are proven. Other barrier shapes
  and unsupported memory instructions fail closed.
* **Device launch limits are explicit:** the runner queries the compiled
  kernel's `CL_KERNEL_WORK_GROUP_SIZE` and rejects an oversized local group
  before execution with the requested size and device/kernel limit.
* **e_flags arch decoding** is empirically verified against 11 real
  nvcc 12.9 cubins (sm_50–sm_90 in one layout, sm_100/sm_120 in another;
  full matrix in docs/ground-truth.md) — not claimed as NVIDIA's formal
  universal encoding.
* `verify` is honest about hardware: **NO-DEVICE** verdict if no OpenCL
  runtime is present — never a fabricated pass. Device execution has been
  verified bit-exact on both Intel and AMD OpenCL runtimes.

## Layout

```
cuda_translator/      the app: ptx, ir, opencl, opencl_runner, fatbin,
                      elf, pe, interpreter, numeric_oracles, pipeline,
                      verify, cli, rest, mcp_server
scripts/              checkout compatibility wrapper + verify_numeric driver
tests/                unittest/regression suite + MCP smoke client
tools/                NVRTC PTX generator (second compiler path)
examples/             .cu sources + NVIDIA-generated proof artifacts
docs/ground-truth.md  how every fixture was made + oracle commands
docs/house-field-crossvendor-20260929.md
                      large real Numba/NVVM NVIDIA->AMD replay receipt
LICENSE               MIT license
SECURITY.md            security boundary and reporting guidance
THIRD_PARTY_NOTICES.md fixture/reference attributions
```

## License and notices

The project is MIT licensed. See [LICENSE](LICENSE). Third-party references,
fixture provenance, and NVIDIA-tooling notices are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
