# Ground truth: fixtures, provenance, and oracle checks

Every structural claim in this repo traces to a byte in `examples/artifacts/`
or a command below. Nothing is from memory; two claims below were initially
wrong from memory and were corrected from the bytes (EM_CUDA=190 not 183;
outer fatbin size excludes the 16-byte header).

## Toolchain provenance

* NVIDIA CUDA 12.9 redist archives (developer.download.nvidia.com,
  redistrib_12.9.1.json):
  * `cuda_nvcc-windows-x86_64-12.9.86-archive` → nvcc, ptxas
  * `cuda_cuobjdump-windows-x86_64-12.9.82-archive` → cuobjdump
  * `cuda_nvdisasm-windows-x86_64-12.9.88-archive` → nvdisasm
  * `cuda_nvrtc-windows-x86_64-12.9.86-archive` → NVRTC (second PTX path)
  * `cuda_cudart-windows-x86_64-12.9.79-archive` → headers/libs for host exes
* Host MSVC: VS 18 BuildTools, cl 14.50.35717. nvcc 12.9 rejects that
  version officially, so every nvcc command below used
  `-allow-unsupported-compiler`. **This is a fixture-build accommodation,
  not a claim that CUDA 12.9 supports MSVC 14.50.**
* OpenCL devices used for the numeric acceptance seam:
  * Intel(R) Iris(R) Xe Graphics via Intel(R) OpenCL Graphics, driver
    `32.0.101.7082`, 96 reported OpenCL compute units.
  * AMD `gfx90c` via AMD Accelerated Parallel Processing, driver
    `3584.0 (PAL,HSAIL)`, 8 reported OpenCL compute units.
  Both are cross-vendor execution targets for NVIDIA-produced PTX.

## Fixture generation (exact commands)

```bat
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;%PATH%
set NVCC=.toolchain\redist\cuda_nvcc-windows-x86_64-12.9.86-archive\bin\nvcc.exe
set CUDART=.toolchain\redist\cuda_cudart-windows-x86_64-12.9.79-archive
set CUDA_FLAGS=-allow-unsupported-compiler -I%CUDART%\include

:: canonical kernel pair
%NVCC% %CUDA_FLAGS% -arch=sm_75 -ptx   examples\vector_add.cu    -o examples\artifacts\vector_add_sm75.ptx
%NVCC% %CUDA_FLAGS% -arch=sm_75 -cubin examples\vector_add.cu    -o examples\artifacts\vector_add_sm75.cubin
%NVCC% %CUDA_FLAGS% -arch=sm_75 -fatbin examples\vector_add.cu   -o examples\artifacts\vector_add_sm75.fatbin
%NVCC% %CUDA_FLAGS% -arch=sm_75 -fatbin -no-compress examples\vector_add.cu -o examples\artifacts\vector_add_sm75_nocompress.fatbin
%NVCC% %CUDA_FLAGS% -arch=sm_75 -fmad=false -ptx examples\vector_add.cu -o examples\artifacts\vector_add_sm75_nofmad.ptx
%NVCC% %CUDA_FLAGS% -arch=sm_75 -L%CUDART%\lib\x64 examples\vector_add_app.cu -o examples\artifacts\vector_add_host.exe

:: fail-closed + data-dependent-CFG fixture
%NVCC% %CUDA_FLAGS% -arch=sm_75 -ptx examples\histogram.cu -o examples\artifacts\histogram_sm75.ptx

:: multi-arch e_flags proof
for %A in (50 60 61 70 75 80 86 89 90) do %NVCC% %CUDA_FLAGS% -arch=sm_%A -cubin examples\vector_add.cu -o examples\artifacts\vector_add_sm%A.cubin

:: NVRTC second path
python tools\gen_nvrtc_ptx.py   →  examples\artifacts\vector_add_sm75_nvrtc.ptx
```

## Oracle checks (NVIDIA tools as the judge)

```
cuobjdump --list-elf vector_add_sm75.fatbin
  ELF file    1: CUDA-vector_add_sm75.1.sm_75.cubin
cuobjdump --list-ptx vector_add_sm75.fatbin
  PTX file    1: CUDA-vector_add_sm75.1.sm_75.ptx
cuobjdump -xelf all  → carved cubin sha256 == standalone nvcc -cubin sha256
                       (d3275879…35905, byte-identical)
nvdisasm -c vector_add_sm75.cubin → real sm_75 SASS disassembly
```

Our parser vs NVIDIA, machine-checked in `tests/test_all.py`:

* fatbin compressed PTX payload (696 B) decompressed by our decoder ==
  NVIDIA `-no-compress` fatbin PTX payload (1928 B incl. pad) — byte-exact
* fatbin ELF payload == standalone cubin — byte-exact
* PE-extracted fatbin (host exe, `.nv_fatb`) == `nvcc -fatbin` output — byte-exact

## Header facts (read from the artifacts, not assumed)

* Outer fatbin header is 16 bytes LE: `u32 magic=0xBA55ED50, u16 version=1,
  u16 headerSize=16, u64 size`; `headerSize + size == file size`. The size
  field **excludes** the header (4928 = 16 + 4912).
* Entry headers: 64 bytes (ELF) / 80 bytes (PTX), LE:
  `u16 kind(1=PTX,2=ELF), u16 version(0x0101), u32 headerSize, u64 size,
  u32 compressedSize, u32 unknown2(=64 on PTX), u16 minor, u16 major,
  u32 arch, u32 objNameOffset, u32 objNameLen, u64 flags, u64 zero,
  u64 decompressedSize`. Flags bit 0x2000 = compressed (observed 0x2041 on
  the compressed PTX entry, 0x41 on ELF entries).
* The 80-byte header's minor/major (8,8) matched `.version 8.8` in the PTX.
* Cubin ELF: `e_machine = 190` (EM_CUDA; 183 was a wrong memory claim),
  `e_flags` shapes measured across 11 real cubins:

  | arch | e_flags | major | mid | core |
  |---|---|---|---|---|
  | sm_50 | 0x00320532 | 50 | 5 | 50 |
  | sm_60 | 0x003C053C | 60 | 5 | 60 |
  | sm_61 | 0x003D053D | 61 | 5 | 61 |
  | sm_70 | 0x00460546 | 70 | 5 | 70 |
  | sm_75 | 0x004B054B | 75 | 5 | 75 |
  | sm_80 | 0x00500550 | 80 | 5 | 80 |
  | sm_86 | 0x00560556 | 86 | 5 | 86 |
  | sm_89 | 0x00590559 | 89 | 5 | 89 |
  | sm_90 | 0x005A055A | 90 | 5 | 90 |
  | sm_100 | 0x06006402 | 1536 | 100 | 2 |
  | sm_120 | 0x06007802 | 1536 | 120 | 2 |

  Decoder rule (empirical, `elf._arch_from_flags`): `50 <= major <= 99 →
  sm_{major}`; `major == 1536 and 100 <= mid <= 130 → sm_{mid}`; anything
  else → None (unknown, never guessed). All 11 decode correctly.

## Numeric oracles

`cuda_translator/numeric_oracles.py` implements IEEE-754 `fma.rn.f32`,
`mul.rn.f32`, `add.rn` exactly (integer arithmetic on bit patterns, single
final rounding). Validation: **0 mismatches vs native `fmaf` (ucrtbase) on
300,000 random finite triples.** Fused vs unfused results differ on ~25% of
random triples, so FMA-semantics tests are discriminating, not vacuous.

Acceptance seam (`python cuda-translator.py verify`):
nvcc PTX → House IR → OpenCL C → independent OpenCL device → compare against
exact oracles. The same translated fixtures were executed on both Intel Iris
Xe and AMD gfx90c: vector_add 1024/1024 bit-exact; saxpy 1024/1024 bit-exact
vs the fused oracle (and not vs unfused) on each device. The `-fmad=false`
build shows 0 `fma` ops in PTX and its interpreter/translation match the
unfused oracle 1024/1024 — semantics follow the binary, not a hardcoded model.

Packaging acceptance was also run from fresh virtual environments after
`pip install .`: the installed CLI can analyze and verify without checkout-only
imports; installed MCP completes initialize/tools/verify on Intel; installed
REST health and `/api/verify` work on Intel, while installed verification and
REST health also pass on AMD. The bundled verification PTX and ctypes OpenCL
runner are package data/code, not dependencies on the source tree.

## Device metadata (correction receipt)

An earlier revision queried device info with `CL_DRIVER_VERSION = 0x1027`,
decoded the returned 4 bytes (`01 00 00 00`) as a string, and briefly
worked around the resulting junk by content-classifying platform strings.
Both halves were wrong in our code, not the ICD: 0x1027 is
`CL_DEVICE_AVAILABLE` (a boolean — the bytes are simply `true`), and the
platform observations matched the canonical layout exactly
(0x0901 version "OpenCL 3.0", 0x0902 name, 0x0903 vendor, 0x0904
extensions). The runner now uses the canonical Khronos constants only,
with `CL_DRIVER_VERSION = 0x102D`. No heuristics remain.

## Interpreter scope honesty

The House IR interpreter is a debug oracle (it shares IR semantics with the
backend by construction). The *final* numeric judge is the independent
exact-oracle compare against real device execution above. The interpreter
rejects non-uniform launch geometry (global not divisible by local) instead
of silently floor-dividing the range.


## Large generated-kernel cross-vendor acceptance

The small checked-in fixtures are complemented by a real Numba/NVVM
application-kernel stress test. The source PTX is intentionally not bundled
with the public fixture set.

The exact artifact is 452,030 bytes / 6,785 lines, targets `sm_52`, and has
SHA-256
`df1ca5f1c631b11c767f42f7fd3f1329739d1a2c13e8bfebc78f84a4581d3f30`.
It exercises dynamic extern shared memory, `bar.sync 0`, FP64
round-to-nearest division, mixed-width global accesses, narrow signed
comparison semantics and direct backward-loop control flow.

A captured real NVIDIA GTX 970 CUDA launch (2 blocks x 9 threads, 432 bytes
dynamic shared memory, 16 RK4 slices) was reconstructed from the 180 PTX
parameters and replayed through the translated OpenCL C on AMD `gfx90c`.
Every input, mutated scratch buffer and formal output was byte-identical;
all floating comparisons had max absolute difference 0.0.

Full failure-to-proof history, hardware details and the narrow signed
`ld.s8` / `setp.s16` bug found by the differential run are recorded in
[house-field-crossvendor-20260929.md](house-field-crossvendor-20260929.md).

Scale follow-up on the same AMD device preserved byte identity at 128 nodes
(degree 8, 6,144 B dynamic shared) and 256 nodes (degree 12, 12,288 B dynamic
shared). A 384-node / degree-14 CUDA capture requires a 384-thread work-group;
the AMD device and compiled OpenCL kernel both report a 256-work-item maximum,
so the runner rejects that launch before execution with an explicit
capability error. This boundary and the canonical kernel work-group constants
are recorded in the cross-vendor receipt.
