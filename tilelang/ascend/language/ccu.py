"""Static, rank-specialized communication kernels for the Ascend 950 CCU."""

from functools import wraps

from ._transfer import copy_instruction, lower


class _Builder:
    def __init__(self, rank, size, src, dst, itemsize):
        self.rank, self.size = rank, size
        self._src, self._dst, self._itemsize = src, dst, itemsize
        self.code = []
        self.pending = 0

    def copy(self, src, dst, *, peer):
        """Issue a copy from local src into peer's dst. Offsets are in elements."""
        instruction = copy_instruction(src, dst, peer, self.size, self._src, self._dst, self._itemsize)
        if instruction is None:
            return
        if self.pending == 16:
            raise ValueError("Use wait_all before issuing more than 16 outstanding copies")
        self.code.append(instruction)
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
    return lower(fn, size, src_count, dst_count, itemsize, _Builder, 2, "CCU")


def kernel(fn):
    """Define a CCU program. Python control flow specializes on group rank, size and tensor extents."""

    @wraps(fn)
    def launch(src, dst, *, team):
        from tilelang.ascend.distributed.ccu import launch_program

        return launch_program(fn, src, dst, team)

    return launch
