"""Rank specialization, phase ordering and region validation for CCU kernels."""

import pytest
from tilelang.ascend.language.ccu import _lower


def test_rank_specialized_ring_and_empty_input():
    def ring(c, src, dst):
        c.copy(src[:], dst[1 : src.size + 1], peer=(c.rank + 1) % c.size)
        c.wait_all()
        c.barrier()

    _, programs = _lower(ring, 3, 5, 6, 4)
    assert [p[0] for p in programs] == [(0, 1, 0, 4, 20), (0, 2, 0, 4, 20), (0, 0, 0, 4, 20)]
    _, empty = _lower(ring, 3, 0, 1, 4)
    assert all(p == [(2, 0, 0, 0, 0)] for p in empty)


def test_detect_write_conflicts_and_distinguish_phases():
    def overlap(c, src, dst):
        c.copy(src[:], dst[:], peer=0)
        c.wait_all()
        c.barrier()

    with pytest.raises(ValueError, match="overlapping"):
        _lower(overlap, 2, 4, 4, 4)

    def ordered(c, src, dst):
        for producer in range(c.size):
            if c.rank == producer:
                c.copy(src[:], dst[:], peer=0)
                c.wait_all()
            c.barrier()

    _lower(ordered, 2, 4, 4, 4)


def test_wait_and_barrier_contract():
    def pending(c, src, dst):
        c.copy(src[:], dst[:], peer=c.rank)
        c.barrier()

    with pytest.raises(ValueError, match="wait_all"):
        _lower(pending, 2, 4, 4, 4)

    def divergent(c, src, dst):
        if c.rank == 0:
            c.barrier()
        c.barrier()

    with pytest.raises(ValueError, match="number of barriers"):
        _lower(divergent, 2, 4, 4, 4)


def test_out_of_bounds_and_event_limit():
    def invalid(c, src, dst):
        c.copy(src[:], dst[1:5], peer=c.rank)

    with pytest.raises(ValueError, match="exceeds"):
        _lower(invalid, 2, 4, 4, 4)

    def full(c, src, dst):
        for i in range(17):
            c.copy(src[i : i + 1], dst[i : i + 1], peer=c.rank)
        c.wait_all()
        c.barrier()

    with pytest.raises(ValueError, match="16 outstanding"):
        _lower(full, 2, 20, 20, 4)
