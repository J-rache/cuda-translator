"""Generate PTX from CUDA kernel C via NVRTC (independent second path vs nvcc).

Uses the real NVIDIA NVRTC DLL shipped in .toolchain/nvrtc-wheel via ctypes.
Output: examples/artifacts/vector_add_sm75_nvrtc.ptx
"""
import ctypes
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NVRTC_DIR = os.path.join(HERE, ".toolchain", "redist", "cuda_nvrtc-windows-x86_64-12.9.86-archive", "bin")
NVRTC = os.path.join(NVRTC_DIR, "nvrtc64_120_0.dll")
OUT = os.path.join(HERE, "examples", "artifacts", "vector_add_sm75_nvrtc.ptx")

KERNEL_SRC = r"""
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void saxpy(float alpha, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}
"""


def check(lib, rc, what):
    if rc != 0:
        try:
            log_size = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(None, ctypes.byref(log_size))
        except Exception:
            pass
        raise RuntimeError(f"NVRTC {what} failed rc={rc}")


def main():
    # NVRTC resolves nvrtc-builtins64_129.dll via plain LoadLibrary, which uses
    # the PATH-based search order -> make its directory visible first.
    os.environ["PATH"] = NVRTC_DIR + os.pathsep + os.environ.get("PATH", "")
    os.chdir(NVRTC_DIR)
    lib = ctypes.CDLL(NVRTC)
    prog = ctypes.c_void_p()
    src = KERNEL_SRC.encode("utf-8")
    name = b"vector_add_nvrtc.cu"
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src, name, 0, None, None)
    check(lib, rc, "CreateProgram")

    opts = (ctypes.c_char_p * 4)(b"--gpu-architecture=compute_75", b"-default-device", b"-I", b".")
    rc = lib.nvrtcCompileProgram(prog, 2, opts)
    if rc != 0:
        log_size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        buf = ctypes.create_string_buffer(log_size.value)
        lib.nvrtcGetProgramLog(prog, buf)
        print(buf.value.decode("utf-8", "replace"), file=sys.stderr)
        raise SystemExit(f"NVRTC compile failed rc={rc}")

    ptx_size = ctypes.c_size_t()
    check(lib, lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)), "GetPTXSize")
    ptx_buf = ctypes.create_string_buffer(ptx_size.value)
    check(lib, lib.nvrtcGetPTX(prog, ptx_buf), "GetPTX")
    check(lib, lib.nvrtcDestroyProgram(ctypes.byref(prog)), "DestroyProgram")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(ptx_buf.raw[: ptx_size.value - 1])  # drop NUL
    print(f"wrote {OUT} ({ptx_size.value - 1} bytes)")


if __name__ == "__main__":
    main()
