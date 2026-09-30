"""OpenCL runner: compile and execute translated kernels on a real OpenCL device.

Uses the system OpenCL ICD loader (OpenCL.dll on Windows, libOpenCL.so on
Linux) through ctypes — no third-party dependencies. If no ICD/device is
present, returns an honest "no device" status instead of pretending.

Exit contract used by verify_numeric.py:
    0  executed and numerically verified
    1  executed but mismatch / compile failure
    3  no OpenCL runtime or device on this machine (skipped, honestly)
"""
from __future__ import annotations

import ctypes
import json
import struct
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

CL_SUCCESS = 0
CL_PLATFORM_NOT_FOUND_KHR = -1001
CL_MEM_READ_WRITE = 0x1
CL_MEM_COPY_HOST_PTR = 0x10
# Canonical clGetKernelWorkGroupInfo constants:
CL_KERNEL_WORK_GROUP_SIZE = 0x11B0
CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1
CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3
CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4
CL_PROGRAM_BUILD_LOG = 0x1183
# Canonical Khronos constants (cl_platform.h):
CL_PLATFORM_PROFILE = 0x0900
CL_PLATFORM_VERSION = 0x0901
CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_PLATFORM_EXTENSIONS = 0x0904
CL_DEVICE_TYPE = 0x1000
CL_DEVICE_TYPE_CPU = 1 << 1
CL_DEVICE_TYPE_GPU = 1 << 2
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D  # NOTE: 0x1027 is CL_DEVICE_AVAILABLE (a bool!)
CL_DEVICE_AVAILABLE = 0x1027
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032
CL_TRUE = 1


@dataclass
class DeviceInfo:
    platform: str
    device: str
    driver: str
    compute_units: int
    vendor: str = ""
    device_type: int = 0
    available: bool = True
    extensions: str = ""
    double_fp_config: int = 0

    @property
    def supports_fp64(self) -> bool:
        return bool(self.double_fp_config) or (
            "cl_khr_fp64" in self.extensions.split()
        )


@dataclass
class RunResult:
    status: str  # "ran" | "no-device"
    device: Optional[DeviceInfo] = None
    kernel_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _load_opencl() -> Optional[ctypes.CDLL]:
    import os
    candidates = ["OpenCL.dll", "libOpenCL.so", "libOpenCL.so.1"]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class _CL:
    """Minimal ctypes binding for the subset of OpenCL we need."""

    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        c = lib
        # 64-bit correctness: every function returning a pointer/handle MUST
        # declare restype=c_void_p, else ctypes truncates to 32-bit c_int and
        # the truncated handle crashes the ICD (observed on Intel OpenCL).
        for fn in ("clCreateContext", "clCreateCommandQueue", "clCreateBuffer",
                   "clCreateProgramWithSource", "clCreateKernel"):
            getattr(c, fn).restype = ctypes.c_void_p
        c.clGetPlatformIDs.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
        c.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
        c.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        c.clGetPlatformInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        c.clCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        c.clCreateCommandQueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_int)]
        c.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        c.clCreateProgramWithSource.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int)]
        c.clBuildProgram.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
        c.clGetProgramBuildInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        c.clCreateKernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        c.clGetKernelWorkGroupInfo.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ]
        c.clSetKernelArg.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
        c.clEnqueueNDRangeKernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        c.clEnqueueReadBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        c.clEnqueueWriteBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        c.clFinish.argtypes = [ctypes.c_void_p]
        for fn in ("clReleaseMemObject", "clReleaseKernel", "clReleaseProgram",
                   "clReleaseCommandQueue", "clReleaseContext"):
            getattr(c, fn).argtypes = [ctypes.c_void_p]

    def _chk(self, rc: int, what: str) -> None:
        if rc != CL_SUCCESS:
            raise RuntimeError(f"OpenCL {what} failed with {rc}")

    def info_string(self, obj: int, which: int, kind: str = "device") -> str:
        n = ctypes.c_size_t(0)
        if kind == "device":
            self._chk(self.lib.clGetDeviceInfo(obj, which, 0, None, ctypes.byref(n)), "GetDeviceInfo size")
        else:
            self._chk(self.lib.clGetPlatformInfo(obj, which, 0, None, ctypes.byref(n)), "GetPlatformInfo size")
        buf = ctypes.create_string_buffer(n.value + 1)
        if kind == "device":
            self._chk(self.lib.clGetDeviceInfo(obj, which, n.value, buf, None), "GetDeviceInfo")
        else:
            self._chk(self.lib.clGetPlatformInfo(obj, which, n.value, buf, None), "GetPlatformInfo")
        return buf.value.decode("utf-8", "replace")


def _device_scalar(cl: _CL, dev: int, which: int, ctype):
    value = ctype()
    cl._chk(
        cl.lib.clGetDeviceInfo(
            dev,
            which,
            ctypes.sizeof(value),
            ctypes.byref(value),
            None,
        ),
        f"GetDeviceInfo[{hex(which)}]",
    )
    return value.value


def enumerate_devices() -> Tuple[
    Optional[_CL],
    List[Tuple[DeviceInfo, int, int]],
]:
    """Enumerate every OpenCL device with portable qualification metadata."""
    lib = _load_opencl()
    if lib is None:
        return None, []

    cl = _CL(lib)
    nplat = ctypes.c_uint(0)
    rc = lib.clGetPlatformIDs(0, None, ctypes.byref(nplat))
    if rc == CL_PLATFORM_NOT_FOUND_KHR or (
        rc == CL_SUCCESS and nplat.value == 0
    ):
        return None, []
    cl._chk(rc, "GetPlatformIDs")

    plats = (ctypes.c_void_p * nplat.value)()
    cl._chk(
        lib.clGetPlatformIDs(nplat.value, plats, None),
        "GetPlatformIDs",
    )

    records: List[Tuple[DeviceInfo, int, int]] = []
    for p_handle in plats:
        p = int(p_handle)
        ndev = ctypes.c_uint(0)
        rc = lib.clGetDeviceIDs(
            p,
            CL_DEVICE_TYPE_ALL,
            0,
            None,
            ctypes.byref(ndev),
        )
        if rc != CL_SUCCESS or ndev.value == 0:
            continue

        devs = (ctypes.c_void_p * ndev.value)()
        cl._chk(
            lib.clGetDeviceIDs(
                p,
                CL_DEVICE_TYPE_ALL,
                ndev.value,
                devs,
                None,
            ),
            "GetDeviceIDs",
        )

        platform_name = cl.info_string(
            p,
            CL_PLATFORM_NAME,
            "platform",
        )
        for d_handle in devs:
            d = int(d_handle)
            info = DeviceInfo(
                platform=platform_name,
                device=cl.info_string(d, CL_DEVICE_NAME),
                driver=cl.info_string(d, CL_DRIVER_VERSION),
                compute_units=int(
                    _device_scalar(
                        cl,
                        d,
                        CL_DEVICE_MAX_COMPUTE_UNITS,
                        ctypes.c_uint,
                    )
                ),
                vendor=cl.info_string(d, CL_DEVICE_VENDOR),
                device_type=int(
                    _device_scalar(
                        cl,
                        d,
                        CL_DEVICE_TYPE,
                        ctypes.c_ulonglong,
                    )
                ),
                available=bool(
                    _device_scalar(
                        cl,
                        d,
                        CL_DEVICE_AVAILABLE,
                        ctypes.c_uint,
                    )
                ),
                extensions=cl.info_string(
                    d,
                    CL_DEVICE_EXTENSIONS,
                ),
                double_fp_config=int(
                    _device_scalar(
                        cl,
                        d,
                        CL_DEVICE_DOUBLE_FP_CONFIG,
                        ctypes.c_ulonglong,
                    )
                ),
            )
            records.append((info, p, d))
    return cl, records


def _select_device_record(
    records: List[Tuple[DeviceInfo, int, int]],
    *,
    prefer_gpu: bool,
    require_fp64: bool,
):
    eligible = [
        record
        for record in records
        if record[0].available
        and (
            not require_fp64
            or record[0].supports_fp64
        )
    ]
    if prefer_gpu:
        eligible.sort(
            key=lambda record: (
                0
                if record[0].device_type & CL_DEVICE_TYPE_GPU
                else 1
            )
        )
    return eligible[0] if eligible else None


def first_device(
    prefer_gpu: bool = True,
    require_fp64: bool = False,
) -> Tuple[
    Optional[_CL],
    Optional[DeviceInfo],
    Optional[int],
    Optional[int],
]:
    """Return the first available device satisfying requested capabilities.

    With prefer_gpu=True, eligible GPUs sort before non-GPU devices. When
    require_fp64=True, devices lacking double-precision support are skipped
    before context creation or kernel compilation. This lets callers fail
    closed without surfacing a megabyte-scale compiler log for an impossible
    float64 kernel.
    """
    cl, records = enumerate_devices()
    if cl is None:
        return None, None, None, None

    chosen = _select_device_record(
        records,
        prefer_gpu=prefer_gpu,
        require_fp64=require_fp64,
    )
    if chosen is None:
        return None, None, None, None

    info, platform_id, device_id = chosen
    return cl, info, platform_id, device_id


def compile_kernel(cl: _CL, ctx: int, dev: int, source: str) -> int:
    src = source.encode("utf-8")
    srct = ctypes.c_char_p(src)
    lens = ctypes.c_size_t(len(src))
    err = ctypes.c_int(0)
    prog = cl.lib.clCreateProgramWithSource(ctx, 1, ctypes.byref(srct), ctypes.byref(lens), ctypes.byref(err))
    cl._chk(err.value, "CreateProgramWithSource")
    rc = cl.lib.clBuildProgram(prog, 1, ctypes.byref(ctypes.c_void_p(dev)), b"", None, None)
    if rc != CL_SUCCESS:
        n = ctypes.c_size_t(0)
        cl.lib.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value + 1)
        cl.lib.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n.value, buf, None)
        raise RuntimeError(f"clBuildProgram failed ({rc}): {buf.value.decode('utf-8', 'replace')}")
    return prog


_SCALAR_CTYPES = {
    "f32": ctypes.c_float, "f64": ctypes.c_double,
    "u32": ctypes.c_uint32, "s32": ctypes.c_int32, "b32": ctypes.c_uint32,
    "u64": ctypes.c_uint64, "s64": ctypes.c_int64, "b64": ctypes.c_uint64,
    "u16": ctypes.c_uint16, "s16": ctypes.c_int16,
    "u8": ctypes.c_ubyte, "s8": ctypes.c_byte,
}


def make_context_and_queue(cl: _CL, plat: int, dev: int):
    """Create (context, command queue) for a device.

    If queue creation fails after a context has been created, release that
    context before propagating the original failure.
    """
    import ctypes
    err = ctypes.c_int(0)
    dref = ctypes.c_void_p(dev)
    ctx = cl.lib.clCreateContext(
        None,
        1,
        ctypes.byref(dref),
        None,
        None,
        ctypes.byref(err),
    )
    cl._chk(err.value, "CreateContext")
    try:
        q = cl.lib.clCreateCommandQueue(
            ctx,
            dev,
            0,
            ctypes.byref(err),
        )
        cl._chk(err.value, "CreateCommandQueue")
    except Exception:
        cl.lib.clReleaseContext(ctx)
        raise
    return ctx, q


def kernel_work_group_limit(cl: _CL, kernel: int, dev: int) -> int:
    """Return the compiled kernel's maximum total local work-group size."""
    value = ctypes.c_size_t(0)
    cl._chk(
        cl.lib.clGetKernelWorkGroupInfo(
            kernel,
            dev,
            CL_KERNEL_WORK_GROUP_SIZE,
            ctypes.sizeof(value),
            ctypes.byref(value),
            None,
        ),
        "GetKernelWorkGroupInfo[WORK_GROUP_SIZE]",
    )
    return int(value.value)


def _validate_launch_geometry(
    global_size: Tuple[int, ...],
    local_size: Optional[Tuple[int, ...]],
    kernel_limit: Optional[int] = None,
) -> None:
    if local_size is None:
        return
    if len(local_size) != len(global_size):
        raise ValueError("local_size dimensionality must match global_size")
    total = 1
    for g, l in zip(global_size, local_size):
        if l <= 0 or g % l != 0:
            raise ValueError(
                f"global size {g} must be divisible by local size {l}"
            )
        total *= l
    if kernel_limit is not None and total > kernel_limit:
        raise ValueError(
            f"requested local work-group size {total} exceeds compiled "
            f"kernel/device limit {kernel_limit}"
        )


def run_ndrange(cl: _CL, ctx: int, q: int, source: str,
                kernel_name: str, args: List[Any],
                out_indices: List[int],
                global_size: Tuple[int, ...], dev: Optional[int] = None,
                local_size: Optional[Tuple[int, ...]] = None) -> Dict[str, bytes]:
    """Compile + set args (in kernel DECLARATION order) + run + read back.

    args: ordered list. Each item is bytes (pointer buffer, becomes a
    cl_mem arg), (dtype, value) tuple (scalar arg), or ("local", size_bytes)
    for a dynamic __local argument. The ORDER MUST MATCH the kernel parameter
    declaration order exactly.
    out_indices: positions in `args` (of the bytes items) to read back.
    Returns {"<arg position>": bytes}.
    """
    if dev is None:
        raise ValueError("run_ndrange needs the device id (for clBuildProgram)")
    prog = compile_kernel(cl, ctx, dev, source)
    err = ctypes.c_int(0)
    kern = cl.lib.clCreateKernel(prog, kernel_name.encode(), ctypes.byref(err))
    cl._chk(err.value, f"CreateKernel {kernel_name}")

    try:
        launch_limit = kernel_work_group_limit(cl, kern, dev)
        _validate_launch_geometry(global_size, local_size, launch_limit)
    except Exception:
        cl.lib.clReleaseKernel(kern)
        cl.lib.clReleaseProgram(prog)
        raise

    mems: Dict[int, int] = {}
    for idx, item in enumerate(args):
        if isinstance(item, (bytes, bytearray)):
            buf = bytes(item)
            err = ctypes.c_int(0)
            mem = cl.lib.clCreateBuffer(ctx, CL_MEM_READ_WRITE, len(buf), None,
                                        ctypes.byref(err))
            cl._chk(err.value, f"CreateBuffer[arg{idx}]")
            hold = ctypes.create_string_buffer(buf, len(buf))
            cl._chk(cl.lib.clEnqueueWriteBuffer(q, mem, CL_TRUE, 0, len(buf),
                                                hold, 0, None, None),
                    f"WriteBuffer[arg{idx}]")
            mems[idx] = mem
            memref = ctypes.c_void_p(mem)
            cl._chk(cl.lib.clSetKernelArg(kern, idx, ctypes.sizeof(ctypes.c_void_p),
                                          ctypes.byref(memref)), f"SetKernelArg[{idx}]")
        else:
            dtype, value = item
            if dtype == "local":
                size = int(value)
                cl._chk(
                    cl.lib.clSetKernelArg(
                        kern,
                        idx,
                        size,
                        None,
                    ),
                    f"SetKernelArg[{idx}] local",
                )
            else:
                t = _SCALAR_CTYPES[dtype]
                v = t(value)
                cl._chk(cl.lib.clSetKernelArg(kern, idx, ctypes.sizeof(t),
                                              ctypes.byref(v)), f"SetKernelArg[{idx}]")

    gdim = len(global_size)
    gsz = (ctypes.c_size_t * gdim)(*global_size)
    lsz = None
    if local_size is not None:
        lsz = (ctypes.c_size_t * gdim)(*local_size)
    rc = cl.lib.clEnqueueNDRangeKernel(q, kern, gdim, None, gsz, lsz, 0, None, None)
    cl._chk(rc, "EnqueueNDRangeKernel")
    cl._chk(cl.lib.clFinish(q), "Finish")

    outs: Dict[str, bytes] = {}
    for i in out_indices:
        size = len(bytes(args[i]))
        out = ctypes.create_string_buffer(size)
        cl._chk(cl.lib.clEnqueueReadBuffer(q, mems[i], CL_TRUE, 0, size, out, 0, None, None), f"ReadBuffer[{i}]")
        outs[str(i)] = out.raw[:size]
    for m in mems.values():
        cl.lib.clReleaseMemObject(m)
    cl.lib.clReleaseKernel(kern)
    cl.lib.clReleaseProgram(prog)
    return outs
