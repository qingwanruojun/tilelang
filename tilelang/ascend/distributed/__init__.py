"""Ascend 950 AIV communication over HCCL symmetric windows."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path
import torch
import torch.distributed as dist


@lru_cache(maxsize=1)
def _native():
    import torch_npu
    from torch.utils.cpp_extension import load

    root = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    npu = Path(torch_npu.__file__).resolve().parent
    library = load(
        name="tilelang_ain_host",
        sources=[str(Path(__file__).with_name("ain_host.cpp"))],
        extra_include_paths=[str(root / "include"), str(root / "include/hccl"), str(npu / "include")],
        extra_ldflags=[f"-L{root}/lib64", "-Wl,--no-as-needed", "-lhcomm", "-lhccl", "-lc_sec", f"-L{npu}/lib", "-ltorch_npu"],
        is_python_module=False,
    )
    api = ct.CDLL(library)
    api.tl_ain_root_size.restype = ct.c_size_t
    return api


class Window:
    """Registered GM storage. Ordinary tensor views retain the local allocation."""

    def __init__(self, team, local, handle):
        self.team = team
        self._local = local
        self._base = local.data_ptr()
        self._nbytes = local.nbytes
        self._handle = handle

    @property
    def local(self):
        """Local tensor view. Keep the backing storage fixed while registered."""
        return self._local.view_as(self._local)


class Team:
    """Collectively managed group. A team has one submission stream and one channel per peer."""

    @classmethod
    def from_group(cls, group=None, *, transport="ain"):
        return cls(dist.group.WORLD if group is None else group, transport)

    def __init__(self, group, transport):
        self.transport = transport
        self._channels = None
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        if self.size < 2 or self.rank < 0:
            raise ValueError("AIN requires membership in a group of at least two ranks")
        transports = [None] * self.size
        dist.all_gather_object(transports, transport, group=group)
        if transport not in ("ain", "channel") or any(t != transport for t in transports):
            raise ValueError(f"Expected a common 'ain' or 'channel' transport, got {transports}")
        self.device = torch.device("npu", torch.npu.current_device())
        self.stream = torch.npu.current_stream(self.device)
        self.windows = []
        self._registration_failed = False
        self._ctx = None
        self._table = None
        self._team = ct.c_void_p()
        self._comm = ct.c_void_p()
        self._api = self._step(_native)
        self._collectives = {}
        self._programs = {}
        self._aicpu_binary = None
        self._comm = self._new_comm(0)

    def _new_comm(self, expansion):
        root = ct.create_string_buffer(self._api.tl_ain_root_size())
        self._step(lambda: self._call("HcclGetRootInfo", ct.byref(root)) if self.rank == 0 else None)
        payload = [root.raw if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=dist.get_global_rank(self.group, 0), group=self.group)
        root.raw = payload[0]
        comm = ct.c_void_p()
        self._step(lambda: self._call("tl_ain_comm", ct.byref(root), self.size, self.rank, expansion, ct.byref(comm)))
        return comm

    def _call(self, name, *args):
        rc = getattr(self._api, name)(*args)
        if rc:
            raise RuntimeError(f"{name} returned {rc} on group rank {self.rank}")

    def _step(self, fn):
        try:
            result = fn()
            error = None
        except Exception as exc:
            result, error = None, f"{type(exc).__name__}: {exc}"
        errors = [None] * self.size
        dist.all_gather_object(errors, error, group=self.group)
        if any(errors):
            raise RuntimeError(f"Ascend communication failed across group ranks: {errors}")
        return result

    def alloc_window(self, shape, dtype):
        """Collectively allocate and register matching contiguous tensors before prepare or the first launch."""
        if self._ctx is not None or not self._comm or self._registration_failed:
            raise RuntimeError("Allocate windows on an open team before preparing its AIV context")
        shape = tuple(shape)
        dtype = getattr(torch, dtype, None) if isinstance(dtype, str) else dtype
        specs = [None] * self.size
        dist.all_gather_object(specs, (shape, str(dtype)), group=self.group)
        if any(spec != specs[0] for spec in specs) or not isinstance(dtype, torch.dtype) or not shape or min(shape) <= 0:
            raise ValueError(f"Invalid or mismatched window specifications: {specs}")
        handle = ct.c_void_p()
        if self.transport == "ain":
            local = self._step(lambda: torch.ops.tilelang_ain.alloc(shape, dtype, self.device))
        else:
            local = self._step(lambda: torch.empty(shape, dtype=dtype, device=self.device))
        index = len(self.windows)
        window = Window(self, local, handle)
        # Keep each allocation alive even if registration succeeds on only some ranks.
        self.windows.append(window)
        self._registration_failed = True
        if self.transport == "ain":
            self._step(
                lambda: self._call(
                    "HcclCommSymWinRegister", self._comm, ct.c_void_p(local.data_ptr()), ct.c_uint64(local.nbytes), ct.byref(handle), 1
                )
            )
        else:
            self._step(
                lambda: self._call(
                    "tl_ain_register", self._comm, ct.c_void_p(local.data_ptr()), ct.c_uint64(local.nbytes), index, ct.byref(handle)
                )
            )
        self._registration_failed = False
        return window

    def prepare(self):
        """Collectively bind all AIV windows and channels before the first kernel."""
        if not self._comm or not self.windows or self._registration_failed:
            raise RuntimeError("An open team with registered windows is required")
        if torch.npu.current_stream(self.device) != self.stream:
            raise ValueError("Prepare and launch on the stream on which this team was created")
        if self._ctx is None:
            # A failed setup may have acquired resources on a subset of ranks.
            # Keep them for close() and prevent duplicate acquisitions on retry.
            self._registration_failed = True
            count = len(self.windows)
            channels_ptr = 0
            if self.transport == "ain":
                self._step(lambda: self._call("tl_ain_team", self._comm, self.size, self.rank, ct.byref(self._team)))
                records = [(w._base, w._handle.value, w._nbytes) for w in self.windows]
            else:
                handles = (ct.c_void_p * count)(*[w._handle.value for w in self.windows])
                channels = (ct.c_uint64 * self.size)()
                addresses = (ct.c_uint64 * (self.size * count))()
                self._step(lambda: self._call("tl_ain_channels", self._comm, self.size, self.rank, handles, count, channels, addresses))
                self._channels = torch.tensor(list(channels), dtype=torch.uint64, device=self.device)
                channels_ptr = self._channels.data_ptr()
                records = [
                    (w._base, addresses[peer * count + i], w._nbytes) for peer in range(self.size) for i, w in enumerate(self.windows)
                ]
            self._table = torch.tensor(records, dtype=torch.uint64, device=self.device)
            self._ctx = torch.tensor(
                [self._team.value or 0, self.rank, self.size, count, self._table.data_ptr(), channels_ptr, 0],
                dtype=torch.int64,
                device=self.device,
            )

            self._registration_failed = False

    def launch(self, kernel, *args):
        """Launch on the team's stream. Window arguments are bound to their registered tensor views."""
        if any(isinstance(a, Window) and a.team is not self for a in args):
            raise ValueError("Window belongs to a different team")
        attrs = getattr(getattr(kernel, "prim_func", None), "attrs", None)
        if self.transport != "ain" and attrs and attrs.get("tl.ascend_ain_sync", 0):
            raise ValueError("AIN signal and team_sync operations require transport='ain'")
        if self.transport != "channel" and attrs and attrs.get("tl.ascend_jetty", 0):
            raise ValueError("T.jetty requires Team.from_group(transport='channel')")
        self.prepare()
        return kernel(self._ctx, *(a.local if isinstance(a, Window) else a for a in args))

    def barrier(self):
        """Wait for work already submitted on the team stream, then rendezvous with all peers."""

        def wait_local():
            self.stream.synchronize()
            if self._ctx is not None:
                status = int(self._ctx[6].item())
                if status:
                    raise RuntimeError(f"AIV status {status} on rank {self.rank}")

        self._step(wait_local)

    def close(self):
        if not self._comm:
            return
        # Teardown must still deregister a completed kernel's windows when it
        # reported a peer/index error. Stream faults propagate before release.
        self._step(self.stream.synchronize)
        statuses = [None] * self.size
        dist.all_gather_object(statuses, int(self._ctx[6].item()) if self._ctx is not None else 0, group=self.group)
        if self._team:
            self._call("HcclTeamDestroy", self._team)
            self._team = ct.c_void_p()
        if self.transport == "ain":
            while self.windows:
                if self.windows[-1]._handle:
                    self._call("HcclCommSymWinDeregister", self.windows[-1]._handle)
                self.windows.pop()
        if self._aicpu_binary is not None:
            from .aicpu import _check

            api, binary = self._aicpu_binary
            self._step(lambda: _check(api.tl_aicpu_unload, binary))
            self._aicpu_binary = None
        for engine in list(self._collectives):
            self._call("HcclCommDestroy", self._collectives[engine])
            del self._collectives[engine]
        self._call("HcclCommDestroy", self._comm)
        self.windows.clear()
        self._programs.clear()
        self._channels = None
        self._comm = ct.c_void_p()
        self._ctx = self._table = None
        if any(statuses):
            raise RuntimeError(f"AIV completion status by group rank: {statuses}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class NativeAllGather:
    """Native asc-comm host entry for an author-provided AICPU or CCU algorithm."""

    library: str
    symbol: str = "AllGatherCustom"

    def _load(self):
        path = Path(self.library).resolve(strict=True)
        digest = sha256(path.read_bytes()).hexdigest()
        library = ct.CDLL(str(path))
        entry = getattr(library, self.symbol)
        entry.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.c_int, ct.c_void_p, ct.c_void_p]
        entry.restype = ct.c_int
        return digest, entry


def all_gather_into(dst, src, *, team, engine, program: NativeAllGather | None = None):
    """Gather contiguous local tensors in group rank order on the team's stream.

    engine selects the HCCL communicator expansion policy (AICPU_TS or CCU_SCHED).
    With program=None HCCL chooses its built-in algorithm. A NativeAllGather
    supplies the host launch entry of an engine-specific algorithm on the same stream.
    """
    configs = {"aicpu": 2, "ccu": 6}
    types = {torch.float16: 3, torch.float32: 4, torch.int32: 2, torch.bfloat16: 11}
    symbol = program.symbol if isinstance(program, NativeAllGather) else None
    spec = (engine, symbol, tuple(src.shape), str(src.dtype), tuple(dst.shape), str(dst.dtype))
    specs = [None] * team.size
    dist.all_gather_object(specs, spec, group=team.group)
    if any(item != spec for item in specs):
        raise ValueError(f"AllGather arguments differ across ranks: {specs}")

    def validate():
        if program is not None and (not isinstance(program, NativeAllGather) or not isinstance(symbol, str) or not symbol):
            raise ValueError("Expected a NativeAllGather with a nonempty entry symbol")
        if not team._comm or engine not in configs or src.dtype not in types or src.dtype != dst.dtype:
            raise ValueError("Expected an open team, a supported dtype and an 'aicpu' or 'ccu' policy")
        if src.device != team.device or dst.device != team.device or not src.is_contiguous() or not dst.is_contiguous():
            raise ValueError("AllGather needs two contiguous tensors on the team device")
        if dst.numel() != team.size * src.numel() or torch.npu.current_stream(team.device) != team.stream:
            raise ValueError("Incorrect destination size or submission stream")
        if max(src.data_ptr(), dst.data_ptr()) < min(src.data_ptr() + src.nbytes, dst.data_ptr() + dst.nbytes):
            raise ValueError("AllGather input and output must have disjoint storage")

    team._step(validate)
    if not src.numel():
        return dst
    key = engine
    if program is not None:
        # All members participate even on a cache hit (paths are node-local).
        native = team._step(lambda: team._programs.get(program) or program._load())
        digest, entry = native
        digests = [None] * team.size
        dist.all_gather_object(digests, digest, group=team.group)
        if len(set(digests)) != 1:
            raise ValueError("Native AllGather program binaries differ across ranks")
        team._programs[program] = native
        key = (engine, digest, program.symbol)
    if key not in team._collectives:
        team._collectives[key] = team._new_comm(configs[engine])
    src.record_stream(team.stream)
    dst.record_stream(team.stream)
    args = (
        ct.c_void_p(src.data_ptr()),
        ct.c_void_p(dst.data_ptr()),
        ct.c_uint64(src.numel()),
        types[src.dtype],
        team._collectives[key],
        ct.c_void_p(team.stream.npu_stream),
    )
    if program is None:
        team._call("HcclAllGather", *args)
    else:

        def submit():
            status = entry(*args)
            if status:
                raise RuntimeError(f"{program.symbol} returned {status} on group rank {team.rank}")

        team._step(submit)
    return dst
