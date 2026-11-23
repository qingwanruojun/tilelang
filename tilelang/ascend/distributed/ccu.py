"""Register and launch programs from tilelang.ascend.language.ccu on a Team stream."""

import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path

import torch
import torch.distributed as dist

from tilelang.ascend.language.ccu import _lower


@lru_cache(maxsize=1)
def _backend():
    from torch.utils.cpp_extension import load

    root = Path(os.environ["ASCEND_HOME_PATH"])
    directories = ["include", "include/hccl", "include/hcomm", "include/hcomm/ccu", "pkg_inc", "pkg_inc/hcomm", "pkg_inc/hcomm/ccu"]
    path = load(
        name="tilelang_ccu_host",
        sources=[str(Path(__file__).with_name("ccu_host.cpp"))],
        extra_include_paths=[str(root / d) for d in directories],
        extra_ldflags=[f"-L{root}/lib64", "-Wl,--no-as-needed", "-lhcomm", "-lhccl", "-lascendcl"],
        is_python_module=False,
    )
    api = ct.CDLL(path)
    words = ct.POINTER(ct.c_uint64)
    api.tl_ccu_channels.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, words, words]
    api.tl_ccu_register.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32, words, words, ct.c_uint32, ct.c_char_p, words]
    api.tl_ccu_args.argtypes = [ct.c_uint64] * 4 + [words]
    api.tl_ccu_launch.argtypes = [ct.c_uint64, ct.c_uint64, words]
    return api


def _check(function, *args):
    status = function(*args)
    if status:
        raise RuntimeError(f"{function.__name__} returned {status}")


def launch_program(fn, src, dst, team):
    def specialize():
        if not team._comm or torch.npu.current_stream(team.device) != team.stream:
            raise ValueError("CCU kernel needs an open team on its submission stream")
        if src.device != team.device or dst.device != team.device or src.dtype != dst.dtype:
            raise ValueError("Expected two tensors on the team device with identical dtype")
        if not src.is_contiguous() or not dst.is_contiguous():
            raise ValueError("CCU kernels require contiguous input and output tensors")
        if max(src.data_ptr(), dst.data_ptr()) < min(src.data_ptr() + src.nbytes, dst.data_ptr() + dst.nbytes):
            raise ValueError("CCU input and output storage must be disjoint")
        return _lower(fn, team.size, src.numel(), dst.numel(), src.element_size())

    digest, programs = team._step(specialize)
    spec = (digest, src.numel(), dst.numel(), str(src.dtype))
    specs = [None] * team.size
    dist.all_gather_object(specs, spec, group=team.group)
    if any(other != spec for other in specs):
        raise ValueError("All team members must submit the same specialized CCU program and tensor signature")

    api = team._step(_backend)
    args = (ct.c_uint64 * 4)()
    team._step(lambda: _check(api.tl_ccu_args, src.data_ptr(), dst.data_ptr(), src.nbytes, dst.nbytes, args))
    transport = "tilelang_ccu"
    if transport not in team._collectives:
        comm = team._new_comm(6)  # HCCL CCU_SCHED
        channels = (ct.c_uint64 * team.size)()
        thread = ct.c_uint64()
        try:
            team._step(lambda: _check(api.tl_ccu_channels, comm, team.stream.npu_stream, team.rank, team.size, ct.byref(thread), channels))
        except Exception:
            team._call("HcclCommDestroy", comm)
            raise
        team._collectives[transport] = comm
        team._programs[transport] = thread, channels
    thread, channels = team._programs[transport]
    key = (transport, digest)
    if key not in team._programs:
        code = programs[team.rank]
        words = (ct.c_uint64 * (len(code) * 5))(*(word for instruction in code for word in instruction))
        handle = ct.c_uint64()
        team._step(
            lambda: _check(
                api.tl_ccu_register,
                team._collectives[transport],
                team.rank,
                team.size,
                channels,
                words,
                len(code),
                ("tl_" + digest[:32]).encode(),
                ct.byref(handle),
            )
        )
        team._programs[key] = handle
    src.record_stream(team.stream)
    dst.record_stream(team.stream)
    handle = team._programs[key]
    team._step(lambda: _check(api.tl_ccu_launch, thread, handle, args))
    return dst
