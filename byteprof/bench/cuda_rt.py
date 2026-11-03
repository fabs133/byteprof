"""Minimal NVRTC + CUDA driver API wrapper (ctypes) for benchmark kernels.

No MSVC toolchain is needed: CUDA C++ is compiled at runtime with NVRTC straight to a cubin, loaded into
torch's current context, and launched with cuLaunchKernel / cuLaunchCooperativeKernel on torch's stream
(so kernels can be captured into torch CUDA graphs). Lives in byteprof/bench/, separate from the decode path.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import torch

CUDA_HOME = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"))
_nvrtc = None
_cu = None

ATTR_MAX_DYN_SMEM = 8  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
ATTR_COOP = 95  # CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
ATTR_SM_COUNT = 16
ATTR_MAX_SMEM_OPTIN = 97


class CudaError(RuntimeError):
    pass


def _libs():
    global _nvrtc, _cu
    if _nvrtc is None:
        dll = None
        for cand in (CUDA_HOME / "bin" / "nvrtc64_120_0.dll", Path(torch.__file__).parent / "lib" / "nvrtc64_120_0.dll"):
            if cand.exists():
                dll = cand
                break
        if dll is None:
            raise CudaError("nvrtc64_120_0.dll not found")
        os.add_dll_directory(str(dll.parent))
        _nvrtc = C.CDLL(str(dll))
        _cu = C.WinDLL("nvcuda.dll")
    return _nvrtc, _cu


def _chk_cu(r: int, what: str) -> None:
    if r != 0:
        _, cu = _libs()
        s = C.c_char_p()
        cu.cuGetErrorString(r, C.byref(s))
        raise CudaError(f"{what}: {r} {s.value.decode() if s.value else ''}")


def compile_cubin(src: str, name: str = "k.cu", arch: str = "sm_86", opts: tuple[str, ...] = ()) -> bytes:
    nv, _ = _libs()
    prog = C.c_void_p()
    r = nv.nvrtcCreateProgram(C.byref(prog), src.encode(), name.encode(), 0, None, None)
    if r:
        raise CudaError(f"nvrtcCreateProgram {r}")
    allopts = [f"--gpu-architecture={arch}", f"-I{CUDA_HOME / 'include'}", "--std=c++17", "-default-device", *opts]
    arr = (C.c_char_p * len(allopts))(*[o.encode() for o in allopts])
    r = nv.nvrtcCompileProgram(prog, len(allopts), arr)
    n = C.c_size_t()
    nv.nvrtcGetProgramLogSize(prog, C.byref(n))
    log = C.create_string_buffer(n.value)
    nv.nvrtcGetProgramLog(prog, log)
    if r:
        raise CudaError(f"nvrtc compile failed ({r}):\n{log.value.decode(errors='replace')}")
    nv.nvrtcGetCUBINSize(prog, C.byref(n))
    buf = C.create_string_buffer(n.value)
    if nv.nvrtcGetCUBIN(prog, buf):
        raise CudaError("nvrtcGetCUBIN failed")
    nv.nvrtcDestroyProgram(C.byref(prog))
    return buf.raw


class Module:
    """A loaded cubin. Needs a CUDA context current on this thread (torch.cuda.init + any tensor op)."""

    def __init__(self, src: str, opts: tuple[str, ...] = ()):
        torch.zeros(1, device="cuda")
        _, cu = _libs()
        self.cubin = compile_cubin(src, opts=opts)
        self.mod = C.c_void_p()
        _chk_cu(cu.cuModuleLoadData(C.byref(self.mod), self.cubin), "cuModuleLoadData")
        self._fn: dict[str, C.c_void_p] = {}

    def func(self, name: str, dyn_smem: int = 0) -> C.c_void_p:
        _, cu = _libs()
        f = self._fn.get(name)
        if f is None:
            f = C.c_void_p()
            _chk_cu(cu.cuModuleGetFunction(C.byref(f), self.mod, name.encode()), f"cuModuleGetFunction {name}")
            self._fn[name] = f
        if dyn_smem > 48 * 1024:
            _chk_cu(cu.cuFuncSetAttribute(f, ATTR_MAX_DYN_SMEM, dyn_smem), "cuFuncSetAttribute")
        return f

    def global_ptr(self, name: str) -> tuple[int, int]:
        _, cu = _libs()
        p, n = C.c_uint64(), C.c_size_t()
        _chk_cu(cu.cuModuleGetGlobal_v2(C.byref(p), C.byref(n), self.mod, name.encode()), f"cuModuleGetGlobal {name}")
        return p.value, n.value


def _pack(args):
    holders = []
    for a in args:
        if isinstance(a, torch.Tensor):
            holders.append(C.c_uint64(a.data_ptr()))
        elif isinstance(a, bool):
            holders.append(C.c_int(int(a)))
        elif isinstance(a, int):
            holders.append(C.c_int(a))
        elif isinstance(a, float):
            holders.append(C.c_float(a))
        elif isinstance(a, C._SimpleCData):
            holders.append(a)
        else:
            raise TypeError(type(a))
    ptrs = (C.c_void_p * len(holders))(*[C.cast(C.byref(h), C.c_void_p) for h in holders])
    return holders, ptrs


def launch(f, grid: int, block: int, args: list, smem: int = 0, cooperative: bool = False) -> None:
    """Launch on torch's current stream (graph-capturable)."""
    _, cu = _libs()
    holders, ptrs = _pack(args)
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    fn = cu.cuLaunchCooperativeKernel if cooperative else cu.cuLaunchKernel
    if cooperative:
        r = fn(f, grid, 1, 1, block, 1, 1, smem, stream, ptrs)
    else:
        r = fn(f, grid, 1, 1, block, 1, 1, smem, stream, ptrs, None)
    _chk_cu(r, "cuLaunchKernel")


def occupancy(f, block: int, smem: int) -> int:
    _, cu = _libs()
    n = C.c_int()
    _chk_cu(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(n), f, block, C.c_size_t(smem)), "occupancy")
    return n.value


def device_attr(attr: int) -> int:
    torch.zeros(1, device="cuda")
    _, cu = _libs()
    v = C.c_int()
    dev = C.c_int()
    _chk_cu(cu.cuCtxGetDevice(C.byref(dev)), "cuCtxGetDevice")
    _chk_cu(cu.cuDeviceGetAttribute(C.byref(v), attr, dev), "cuDeviceGetAttribute")
    return v.value
