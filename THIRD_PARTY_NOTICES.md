# Third-party and fixture notices

## Fatbin decompression reference

`cuda_translator/fatbin.py` is informed by the documented NVIDIA fatbin LZ
variant in **n-eiling/cuda-fatbin-decompression**, copyright 2023 Niklas Eiling,
published under the Apache License 2.0:

https://github.com/n-eiling/cuda-fatbin-decompression

The project keeps that attribution in the source module. The implementation in
this repository is verified independently against NVIDIA-generated compressed
and uncompressed fixtures.

## NVIDIA-generated fixtures

Files under `examples/artifacts/` and the bundled verification PTX were
generated from the example sources in this repository using NVIDIA CUDA tools.
The exact tool versions and generation commands are recorded in
`docs/ground-truth.md`.

No CUDA toolkit executables, DLLs, headers, redistributable archives, or driver
packages are checked into this repository. NVIDIA, CUDA, NVVM, NVRTC, and
related names are trademarks or product names of NVIDIA Corporation; this
project is not affiliated with or endorsed by NVIDIA.

The generated fixtures are included as interoperability and regression-test
artifacts. Users redistributing them in other contexts should review the
license terms that apply to their CUDA toolchain and environment.
