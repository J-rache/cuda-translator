# Cross-vendor House Field acceptance — 2026-09-29

This receipt records a real application-kernel stress test of cuda-translator.
It is deliberately separate from the small bundled fixtures: the source
artifact is generated PTX from a working Numba-CUDA application and is not
checked into this repository.

## Source artifact

* Producer: NVIDIA NVVM via Numba CUDA, CUDA 11.8.
* Target: `sm_52`.
* Kernel: House Field paired observed/counterfactual RK4 integration kernel.
* PTX size: **452,030 bytes / 6,785 lines**.
* SHA-256:
  `df1ca5f1c631b11c767f42f7fd3f1329739d1a2c13e8bfebc78f84a4581d3f30`.
* PTX entry parameters: **180**.
* The application-level launch has 30 arguments; each 1-D Numba array expands
  to the standard seven-field descriptor
  `meminfo, parent, nitems, itemsize, data, shape[0], stride[0]`.
  The translator independently proved the 25 data-pointer positions:
  `4, 11, 23, ... 177`.

The kernel exercises substantially more of PTX than the small repo fixtures:
dynamic extern shared memory, `bar.sync 0`, FP64 round-to-nearest division,
mixed-width global accesses, signed 8-bit loads into wider bit-storage
registers, direct forward/backward branches, large data-dependent loops, and
many resolved global-buffer address chains.

## Initial fail-closed result

Before this work, analysis produced **171 errors** but only three semantic
families:

| gap | occurrences |
|---|---:|
| shared-memory load/store | 120 |
| `div.rn.f64` | 38 |
| `bar.sync` | 13 |

The translator did not emit a pretending-to-work kernel. It failed closed.

Support added from that evidence includes:

* unsized `.extern .shared` / launch-provided dynamic local memory;
* `ld/st.shared` with access width taken from the individual PTX instruction;
* `%dynamic_smem_size`;
* `bar.sync 0` -> OpenCL work-group barrier with local + global fences;
* `div.rn.f64`;
* direct known-label PTX CFG, including backward loop edges;
* generic byte-offset provenance from a proven global base pointer;
* mixed-width accesses through one global buffer;
* correct narrow signed interpretation for `setp.s8/s16`-family behavior.

Unknown opcodes, unresolved pointers, unknown branch labels and unsupported
barrier shapes remain fail-closed.

## Real hardware

NVIDIA source/oracle execution:

* GPU: NVIDIA GeForce GTX 970 (VIGIL).
* CUDA target: `sm_52`.
* Launch: **2 blocks x 9 threads**, **432 bytes dynamic shared memory**.
* Workload: a nontrivial learned nine-node House Field topology.
* Integration: **16 RK4 slices**, actual and zero-source counterfactual
  trajectories in the two blocks.

Translated execution:

* Platform: AMD Accelerated Parallel Processing.
* Device: `gfx90c`.
* Driver: `3584.0 (PAL,HSAIL)`.
* Compute units: 8.
* Generated OpenCL C: about **1.326 MB**.
* AMD OpenCL compiler accepted the complete generated kernel.

## Differential debugging gates

A captured real CUDA launch was replayed through the translated kernel.
Before final acceptance, four one-slice feature gates were used to localize
semantic differences:

1. full path;
2. convergence disabled;
3. convergence + physical edge conductance disabled;
4. pure leak (source also zero).

The last two became byte-identical first, proving ABI reconstruction,
dynamic shared memory, barriers, RK4, source/leak, and convergence scratch
were not the remaining fault.

The final discrepancy was the physical-edge sign path:

```ptx
ld.s8       %rs1, [...]
setp.gt.s16 %p13, %rs1, 0
```

`%rs1` is declared `.b16`. Raw storage of `-1` is therefore `0xffff`,
but `setp.s16` must reinterpret that storage as signed short. The backend
previously widened unrecognized narrow comparison types to `int`, turning
`0xffff` into positive 65535. Preserving `char/short` interpretation fixed
the physical-edge direction error. A regression test now pins this exact
case.

## Final acceptance

After the fix:

* translator unit suite: **32/32 PASS**;
* exact 452 KB PTX analysis: **0 errors** (two non-semantic `.common`
  inspection warnings);
* strict translation: **PASS**;
* AMD OpenCL compilation of the generated ~1.326 MB kernel: **PASS**;
* all four one-slice differential gates: **every array byte-identical**;
* full 16-slice replay: **every input, scratch and output array
  byte-identical between NVIDIA CUDA and AMD OpenCL**.

The mutated arrays include:

* directed flows;
* receiver amount;
* receiver total;
* output activation;
* output integral;
* output counterfactual.

For every one of them, `max_abs = 0.0`.

The final three formal outputs were:

```text
activation
[0.60558286 0.57917672 0.55055561 0.52996960 0.49496101
 0.47754070 0.44618481 0.41701587 0.39901282]

integral
[0.41748867 0.44452517 0.45839981 0.48767091 0.48025549
 0.49065379 0.53251986 0.56611546 0.62237084]

counterfactual
[0.11597217 0.14123455 0.14790058 0.18166760 0.15054304
 0.15887474 0.20432258 0.23908596 0.31585650]
```

AMD bytes matched the captured NVIDIA CUDA bytes exactly.

## What this proves — and what it does not

This proves that the translator can preserve the semantics of this specific
large, real Numba/NVVM PTX kernel across NVIDIA CUDA and AMD OpenCL,
including its dynamic shared memory, barriers, control flow and FP64
execution.

It does **not** claim arbitrary CUDA/PTX compatibility. Unsupported
instructions and unproved shapes must continue to fail closed. SASS-only
cubin translation remains out of scope.
