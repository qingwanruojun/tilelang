"""Static, rank-specialized communication kernels for the Ascend 950 CCU."""

from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
import json
from operator import index


@dataclass(frozen=True)
class _Region:
    buffer: object
    start: int
    stop: int


class _Buffer:
    def __init__(self, size):
        self.size = size

    def __getitem__(self, region):
        if not isinstance(region, slice) or region.step not in (None, 1):
            raise ValueError("CCU requires a contiguous one-dimensional slice")
        start = 0 if region.start is None else index(region.start)
        stop = self.size if region.stop is None else index(region.stop)
        if not 0 <= start <= stop <= self.size:
            raise ValueError("CCU slice exceeds the tensor")
        return _Region(self, start, stop)


class _Builder:
    def __init__(self, rank, size, src, dst, itemsize):
        self.rank, self.size = rank, size
        self._src, self._dst, self._itemsize = src, dst, itemsize
        self.code = []
        self.pending = 0

    def copy(self, src, dst, *, peer):
        """Issue a copy from local src into peer's dst. Offsets are in elements."""
        peer = index(peer)
        if not 0 <= peer < self.size:
            raise ValueError("Invalid destination rank")
        if not isinstance(src, _Region) or not isinstance(dst, _Region) or src.buffer is not self._src or dst.buffer is not self._dst:
            raise ValueError("copy expects an input slice and an output slice")
        count = src.stop - src.start
        if count != dst.stop - dst.start:
            raise ValueError("Source and destination slice sizes differ")
        if count == 0:
            return
        nbytes = count * self._itemsize
        if nbytes > 256 * 1024 * 1024:
            raise ValueError("Split transfers larger than 256 MiB")
        if self.pending == 16:
            raise ValueError("Use wait_all before issuing more than 16 outstanding copies")
        self.code.append((0, peer, src.start * self._itemsize, dst.start * self._itemsize, nbytes))
        self.pending += 1

    def wait_all(self):
        """Wait for copies issued by this rank since the previous wait."""
        if self.pending:
            self.code.append((1, 0, 0, 0, 0))
            self.pending = 0

    def barrier(self):
        """Rendezvous with all ranks after their previously issued copies complete."""
        if self.pending:
            raise ValueError("Call wait_all before barrier")
        self.code.append((2, 0, 0, 0, 0))


def _lower(fn, size, src_count, dst_count, itemsize):
    src, dst = _Buffer(src_count), _Buffer(dst_count)
    phases, writes, programs = [], {}, []
    for rank in range(size):
        builder = _Builder(rank, size, src, dst, itemsize)
        fn(builder, src, dst)
        if not builder.code or builder.code[-1][0] != 2:
            raise ValueError("Every CCU rank must finish with a barrier")
        phase = 0
        for op, peer, _, offset, length in builder.code:
            if op == 0:
                writes.setdefault((phase, peer), []).append((offset, offset + length))
            elif op == 2:
                phase += 1
        phases.append(phase)
        programs.append(builder.code)
    if len(set(phases)) != 1:
        raise ValueError("All ranks must reach the same number of barriers")
    for regions in writes.values():
        regions.sort()
        if any(first[1] > second[0] for first, second in zip(regions, regions[1:])):
            raise ValueError("Multiple copies write overlapping output ranges in one barrier phase")
    digest = sha256(json.dumps(programs, separators=(",", ":")).encode()).hexdigest()
    return digest, programs


def kernel(fn):
    """Define a CCU program. Python control flow specializes on group rank, size and tensor extents."""

    @wraps(fn)
    def launch(src, dst, *, team):
        from tilelang.ascend.distributed.ccu import launch_program

        return launch_program(fn, src, dst, team)

    return launch
