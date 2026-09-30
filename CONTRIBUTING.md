# Contributing

Thanks for helping improve cuda-translator.

The project is deliberately fail-closed: an unsupported or unprovable CUDA/PTX
construct must be rejected explicitly rather than translated approximately.
Changes that widen the supported subset should include a focused regression
test and, where practical, a real compiler/device fixture or other independent
oracle.

## Development

Python 3.10+ is required. The core package has no third-party runtime
dependencies.

```bat
python -m unittest discover -s tests -p "test*.py" -v
python tests\smoke_mcp.py
```

On a machine with a working OpenCL runtime, also run:

```bat
python cuda-translator.py verify
```

A hardware-free test run may legitimately report that the device acceptance
seam cannot run; do not convert that into a fabricated pass.

Before submitting changes, keep `README.md` and `docs/ground-truth.md`
aligned with what has actually been proved. Do not add generated toolchains,
credentials, local machine paths, or temporary acceptance artifacts to Git.

## Pull requests

Keep changes narrow and explain the semantic behavior being added or fixed.
For parser/backend changes, include the first unsupported opcode or control-flow
case in tests when relevant. For packaging/API/MCP changes, prove that the
installed package works outside the source checkout.
