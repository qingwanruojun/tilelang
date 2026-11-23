"""Rank-specialized Python communication programs for an Ascend AICPU_TS queue."""

from functools import wraps

from ._transfer import copy_instruction, lower


class _Builder:
    def __init__(self, rank, size, src, dst, itemsize):
        self.rank, self.size = rank, size
        self._src, self._dst, self._itemsize = src, dst, itemsize
        self.code = []

    def copy(self, src, dst, *, peer):
        """Enqueue a local or peer write on this rank's ordered communication queue."""
        instruction = copy_instruction(src, dst, peer, self.size, self._src, self._dst, self._itemsize)
        if instruction is not None:
            self.code.append(instruction)

    def barrier(self):
        """Complete the transfer phase and rendezvous with all team members."""
        self.code.append((1, 0, 0, 0, 0))


def _lower(fn, size, src_count, dst_count, itemsize):
    return lower(fn, size, src_count, dst_count, itemsize, _Builder, 1, "AICPU")


def kernel(fn):
    """Build a transfer program for one AICPU queue per rank. Finish each program with a barrier."""

    @wraps(fn)
    def launch(src, dst, *, team):
        from tilelang.ascend.distributed.aicpu import launch_program

        return launch_program(fn, src, dst, team)

    return launch
