# PTX support and proof matrix

This matrix is deliberately stricter than an opcode list. A PTX spelling can be recognized by the frontend without being translatable, and a lowering can exist without having independent cross-vendor proof.

Status meanings:

- **PROVEN** — implemented and exercised by an independent numeric/device proof or by the documented real NVIDIA-artifact/cross-vendor corpus.
- **IMPLEMENTED / LIMITED PROOF** — lowering exists and has focused regression coverage, but the public proof is narrower than the full PTX semantic family.
- **FAIL-CLOSED** — recognized or encountered but deliberately refused.
- **OUT OF SCOPE** — not a translation target for this project.

| Semantic family | Status | Current proof / boundary |
|---|---|---|
| mov / special registers | PROVEN | NVIDIA-generated vector/SAXPY corpus plus Intel/AMD device execution; public stress uses %tid, %ctaid, %ntid. |
| ld/st.global | PROVEN | Real NVIDIA fixtures, byte-addressing regressions, Intel/AMD execution, House Field replay, public stress. |
| ld/st.shared | PROVEN | House Field AMD replay and public stress AMD execution. |
| add | PROVEN | Real device corpus and public stress backward-loop execution. |
| sub | IMPLEMENTED / LIMITED PROOF | Backend lowering exists; no separate public cross-vendor subtraction acceptance receipt. |
| integer wrap semantics (.lo) | IMPLEMENTED / LIMITED PROOF | Explicit unsigned-storage lowering and focused regressions; the whole PTX integer family is not claimed. |
| mul | IMPLEMENTED / LIMITED PROOF | Lowering exists; coverage is narrower than the full PTX modifier space. |
| mul.wide | PROVEN | Used for real/global address formation and executed by the public stress workload on AMD. |
| mad | PROVEN | Public stress uses mad.lo.u32 for global-id formation and passes AMD oracle comparison. |
| mad.wide | FAIL-CLOSED | Backend explicitly refuses it. |
| fma.rn.f32 | PROVEN | Exact integer FMA oracle, fused-vs-unfused discrimination, Intel/AMD execution. |
| div.rn.f64 | PROVEN | House Field AMD receipt and public stress AMD execution. This does not claim every PTX division modifier/type. |
| integer setp | PROVEN | Signed-width regressions, House Field replay, public stress bounds/loop predicates. |
| float/unordered setp | IMPLEMENTED / LIMITED PROOF | Focused NaN/unordered lowering regression; narrower device proof. |
| cvt | IMPLEMENTED / LIMITED PROOF | Width/signedness regressions plus public stress s8/s16/f32→f64 cases; not every rounding/type combination is claimed. |
| cvta.to.global | PROVEN | Real NVIDIA fixtures and both public device workloads. |
| bitwise and/or/xor/not | IMPLEMENTED / LIMITED PROOF | Lowering exists; selected real-kernel/regression coverage only. |
| shifts shl/shr | IMPLEMENTED / LIMITED PROOF | Lowering exists; selected coverage only. |
| selp | IMPLEMENTED / LIMITED PROOF | Explicit operand-order regression; no dedicated cross-vendor numeric receipt. |
| min/max | IMPLEMENTED / LIMITED PROOF | Lowering exists; proof is narrower than the PTX family. |
| abs/neg | IMPLEMENTED / LIMITED PROOF | Typed lowering regressions; proof is narrower than the PTX family. |
| direct forward branches | PROVEN | Data-dependent CFG regression, House Field, and public stress. |
| direct backward loops | PROVEN | House Field and public stress AMD execution. |
| unresolved/indirect branch target | FAIL-CLOSED | Explicit unknown-label regression; public APIs preserve fail-closed error structure. |
| dynamic extern shared memory | PROVEN | House Field AMD replay and public stress with launch-supplied local memory. |
| bar.sync 0 | PROVEN | House Field and public stress AMD execution. |
| other barrier shapes | FAIL-CLOSED | Backend accepts only bar.sync 0; explicit regression pins refusal. |
| atomics | FAIL-CLOSED | atomicAdd/atomic PTX is deliberately refused; CLI/REST/MCP smoke the refusal path. |
| approximate sin/cos/ex2/lg2 | FAIL-CLOSED | Frontend recognizes them as approximate House ops; backend has no lowering. Regression pins this contract. |
| sqrt | FAIL-CLOSED | Recognized by the frontend but no OpenCL backend lowering is claimed. |
| warp vote | FAIL-CLOSED | No proven lowering. |
| shuffle | FAIL-CLOSED | No proven lowering. |
| textures/surfaces | FAIL-CLOSED | No proven lowering. |
| async copy | FAIL-CLOSED | No proven lowering. |
| cooperative groups | FAIL-CLOSED | No proven lowering. |
| SASS-only cubin translation | OUT OF SCOPE | Cubin structure/architecture inspection is supported; SASS decompilation is not. A cubin without PTX cannot be translated. |

## Evidence surfaces

The strongest existing application receipt remains [House Field cross-vendor replay](house-field-crossvendor-20260929.md). It is a real Numba/NVVM application proof, not a public regeneration corpus.

The repository-local reproducibility complement is [the public synthetic stress workload](public-stress.md). It is intentionally synthetic, deterministic, regenerable from this repository, and separately executed against an independent CPU oracle.

This matrix should be updated from parser + IR + backend + tests + device receipts together. Do not promote a row merely because a spelling appears in the frontend opcode map.
