# Changelog

## v0.1.1 — pre-release hardening

This release hardens the public v0.1.0 translator without widening its supported semantic envelope.

- Added GitHub Actions coverage for Python 3.10–3.12 plus a fresh-installed-package job.
- Added a deterministic public synthetic stress workload, generator, independent CPU oracle, and AMD gfx90c byte-exact acceptance receipt.
- Added an explicit PTX support/proof matrix.
- Made CLI translation use the same strict fail-closed contract as REST and MCP.
- Preserved kernel/opcode/PTX-line metadata through strict API failures.
- Added regressions for approximate transcendental refusal, unresolved branches/pointers, unsupported barrier shapes, and SASS-only translation refusal.
- Added 64 MiB public input/request and decompressed-fatbin bounds, malformed compressed-entry checks, and a bounded fatbin entry count.
- Made MCP and REST smoke tests runnable against a fresh installed package and truthful about NO-DEVICE.

Known limitations remain deliberate: no SASS decompilation; unsupported PTX families remain refused; approximate PTX transcendental operations remain fail-closed.

## v0.1.0

Initial public pre-release tag. The v0.1.0 tag remains attached to its original public commit and is not rewritten by v0.1.1 hardening.
