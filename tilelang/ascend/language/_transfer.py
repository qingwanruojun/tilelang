"""Common contiguous regions and cross-rank phase checks for communication programs."""

from dataclasses import dataclass
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
            raise ValueError("Expected a contiguous one-dimensional slice")
        start = 0 if region.start is None else index(region.start)
        stop = self.size if region.stop is None else index(region.stop)
        if not 0 <= start <= stop <= self.size:
            raise ValueError("Communication slice exceeds the tensor")
        return _Region(self, start, stop)


def copy_instruction(src, dst, peer, size, input_buffer, output_buffer, itemsize):
    peer = index(peer)
    if not 0 <= peer < size:
        raise ValueError("Invalid destination rank")
    if not isinstance(src, _Region) or not isinstance(dst, _Region) or src.buffer is not input_buffer or dst.buffer is not output_buffer:
        raise ValueError("copy expects an input slice and an output slice")
    count = src.stop - src.start
    if count != dst.stop - dst.start:
        raise ValueError("Source and destination slice sizes differ")
    if count == 0:
        return None
    nbytes = count * itemsize
    if nbytes > 256 * 1024 * 1024:
        raise ValueError("Split transfers larger than 256 MiB")
    return 0, peer, src.start * itemsize, dst.start * itemsize, nbytes


def lower(fn, size, src_count, dst_count, itemsize, builder_type, barrier_opcode, engine):
    src, dst = _Buffer(src_count), _Buffer(dst_count)
    phases, writes, programs = [], {}, []
    for rank in range(size):
        builder = builder_type(rank, size, src, dst, itemsize)
        fn(builder, src, dst)
        if not builder.code or builder.code[-1][0] != barrier_opcode:
            raise ValueError(f"Every {engine} rank must finish with a barrier")
        phase = 0
        for op, peer, _, offset, length in builder.code:
            if op == 0:
                writes.setdefault((phase, peer), []).append((offset, offset + length))
            elif op == barrier_opcode:
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
