"""Exercise mixed-core handoff and both output layouts on a single Ascend 950."""

import pytest
import torch
import tilelang.testing
from ain_all_gather_matmul import build_agmm


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("rank", [0, 1])
def test_agmm_staged_tail(rank):
    # Preload the two input slots. The invalid-peer guard suppresses all fabric
    # traffic while the actual mixed kernel executes its ready/free protocol.
    rows, padded, width = 77, 96, 64
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(rows, width, generator=generator).half()
    stage = torch.randn(64, width, generator=generator).half()
    b = torch.randn(64, width, generator=generator).half()
    ctx = torch.tensor([0, rank, 0, 0, 0, 0, 0], dtype=torch.int64).to("npu")
    padded_a = torch.nn.functional.pad(a, (0, 0, 0, padded - rows))
    inputs = [x.to("npu") for x in (padded_a, stage, b)]
    y = torch.empty(2 * padded, width, dtype=torch.float32, device="npu")
    build_agmm(rows)(ctx, *inputs, y)
    actual = y.cpu().view(2, padded, width)[:, :rows].reshape(2 * rows, width)
    assert ctx.cpu()[6].item() == -1
    peer = torch.cat((stage, stage[: rows - 64]))
    gathered = torch.cat((a, peer) if rank == 0 else (peer, a))
    torch.testing.assert_close(actual, gathered.float() @ b.float().T, rtol=1e-4, atol=1e-4)
