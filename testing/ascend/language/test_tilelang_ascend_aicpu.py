"""Cross-rank validation for the AICPU queue and rendezvous semantics."""

import pytest
from tilelang.ascend.language.aicpu import _lower


def test_ring_offsets_and_zero_count():
    def ring(a, src, dst):
        a.copy(src[1:], dst[2 : src.size + 1], peer=(a.rank + 1) % a.size)
        a.barrier()

    _, plans = _lower(ring, 3, 4, 5, 4)
    assert [p[0] for p in plans] == [(0, peer, 4, 8, 12) for peer in (1, 2, 0)]
    _, empty = _lower(ring, 2, 1, 2, 4)
    assert empty == [[(1, 0, 0, 0, 0)]] * 2


def test_rendezvous_and_overlaps():
    def divergent(a, src, dst):
        if a.rank:
            a.barrier()
        a.barrier()

    def overlap(a, src, dst):
        a.copy(src[:], dst[:], peer=0)
        a.barrier()

    with pytest.raises(ValueError, match="number of barriers"):
        _lower(divergent, 2, 4, 4, 1)
    with pytest.raises(ValueError, match="overlapping"):
        _lower(overlap, 2, 4, 4, 1)


def test_ordered_overwrite_in_another_phase():
    def phased(a, src, dst):
        for producer in range(a.size):
            if a.rank == producer:
                a.copy(src[:], dst[:], peer=0)
            a.barrier()

    _, plans = _lower(phased, 3, 4, 4, 1)
    assert all(len(p) == 4 for p in plans)
