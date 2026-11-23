"""Build the framework-owned AICPU executor and submit rank-specialized transfer plans."""

import ctypes as ct
import fcntl
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist

from tilelang.ascend.language.aicpu import _lower


class _Arguments(ct.Structure):
    _fields_ = [
        ("queue", ct.c_uint64),
        ("host_on_device", ct.c_uint64),
        ("rank", ct.c_uint32),
        ("size", ct.c_uint32),
        ("peers", ct.c_void_p),
        ("ops", ct.c_void_p),
        ("count", ct.c_uint64),
        ("src", ct.c_void_p),
        ("dst", ct.c_void_p),
        ("src_bytes", ct.c_uint64),
        ("dst_bytes", ct.c_uint64),
        ("output_offset", ct.c_uint64),
    ]


@lru_cache(maxsize=1)
def _backend():
    root = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    source = Path(__file__).resolve().with_name("aicpu_backend")
    digest = sha256(str((root, source)).encode())
    for path in sorted(source.iterdir()):
        if path.is_file():
            digest.update(path.read_bytes())
    digest.update(subprocess.check_output([os.environ.get("CXX", "c++"), "--version"]))
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tilelang" / "aicpu" / digest.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "libtilelang_aicpu.so"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache / "complete").exists() or not library.exists():
            subprocess.run(["cmake", "-S", str(source), "-B", str(cache), f"-DASCEND_CANN_PACKAGE_PATH={root}"], check=True)
            subprocess.run(["cmake", "--build", str(cache), "-j2"], check=True)
            (cache / "complete").touch()
    api = ct.CDLL(str(library))
    words = ct.POINTER(ct.c_uint64)
    api.tl_aicpu_prepare.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, words, words]
    api.tl_aicpu_load.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    api.tl_aicpu_unload.argtypes = [ct.c_void_p]
    api.tl_aicpu_launch.argtypes = [ct.c_void_p, ct.POINTER(_Arguments), ct.c_uint64, ct.c_uint64, ct.c_void_p]
    return api


def _check(function, *args):
    rc = function(*args)
    if rc:
        raise RuntimeError(f"{function.__name__} returned {rc}")


def launch_program(fn, src, dst, team):
    def specialize():
        if not team._comm or torch.npu.current_stream(team.device) != team.stream:
            raise ValueError("Submit on the open team's bound NPU stream")
        if src.device != team.device or dst.device != team.device or src.dtype != dst.dtype:
            raise ValueError("Expected tensors on the team device with matching dtype")
        if not src.is_contiguous() or not dst.is_contiguous():
            raise ValueError("AICPU communication requires contiguous tensors")
        if max(src.data_ptr(), dst.data_ptr()) < min(src.data_ptr() + src.nbytes, dst.data_ptr() + dst.nbytes):
            raise ValueError("Input and output storage must be disjoint")
        return _lower(fn, team.size, src.numel(), dst.numel(), src.element_size())

    digest, programs = team._step(specialize)
    spec = digest, src.numel(), dst.numel(), str(src.dtype)
    specs = [None] * team.size
    dist.all_gather_object(specs, spec, group=team.group)
    if any(other != spec for other in specs):
        raise ValueError("All ranks must submit the same specialized AICPU program and tensor signature")

    api = team._step(_backend)
    transport = "tilelang_aicpu"
    if transport not in team._collectives:
        config = Path(os.environ["ASCEND_HOME_PATH"]) / "opp/vendors/cust/aicpu/config/libtilelang_aicpu_kernel.json"
        binary, function = ct.c_void_p(), ct.c_void_p()

        def load():
            if not config.is_file():
                raise RuntimeError(
                    f"Framework AICPU executor is not installed: {config}. Deploy aicpu_tilelang_transfer once for this CANN installation."
                )
            _check(api.tl_aicpu_load, os.fsencode(config), ct.byref(binary), ct.byref(function))

        comm = None

        def prepare():
            queues = (ct.c_uint64 * 4)()
            peers = (ct.c_uint64 * (3 * team.size))()
            _check(api.tl_aicpu_prepare, comm, team.stream.npu_stream, team.rank, team.size, queues, peers)
            return queues, min(peers[2::3]), torch.tensor(list(peers), dtype=torch.uint64, device=team.device)

        try:
            team._step(load)
            comm = team._new_comm(2)  # HCCL_OP_EXPANSION_MODE_AICPU_TS
            context = team._step(prepare)
        except Exception:
            if binary:
                _check(api.tl_aicpu_unload, binary)
            if comm:
                team._call("HcclCommDestroy", comm)
            raise
        team._programs[transport] = (*context, function)
        team._aicpu_binary = (api, binary)
        team._collectives[transport] = comm

    queues, capacity, peers, function = team._programs[transport]
    output_offset = (src.nbytes + 127) // 128 * 128

    def bind():
        if output_offset + dst.nbytes > capacity:
            raise ValueError("Input and output staging areas exceed the AICPU communication window")
        key = (transport, digest)
        if key not in team._programs:
            team._programs[key] = torch.tensor(programs[team.rank], dtype=torch.uint64, device=team.device)
        return team._programs[key]

    code = team._step(bind)
    args = _Arguments(
        queues[0],
        queues[1],
        team.rank,
        team.size,
        peers.data_ptr(),
        code.data_ptr(),
        len(programs[team.rank]),
        src.data_ptr(),
        dst.data_ptr(),
        src.nbytes,
        dst.nbytes,
        output_offset,
    )
    src.record_stream(team.stream)
    dst.record_stream(team.stream)
    team._step(lambda: _check(api.tl_aicpu_launch, function, ct.byref(args), queues[2], queues[3], team.stream.npu_stream))
    return dst
