"""Compiler contracts and a device guard that needs only one NPU."""

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T


def transfer_case(kind):
    @T.prim_func
    def kernel(ctx: T.comm.Team, src: T.comm.Window((4, 16), "float32"), dst: T.comm.Window((4, 16), "float32")):
        with T.Kernel(1):
            if kind == "cube":
                with T.Cube():
                    T.comm.wait_peer(ctx, 0)
            else:
                with T.Vector(vector=1):
                    if kind == "strided":
                        T.comm.get(ctx, 1, src=src[:, 0:2], dst=dst[0, 0:8])
                    elif kind == "size":
                        T.comm.put(ctx, 1, src=src[1, 2:5], dst=dst[0, 0:4])
                    elif kind == "empty":
                        T.comm.get(ctx, 0, src=src[1, 4:4], dst=dst[0, 2:2])
                    elif kind == "bounds":
                        row = T.comm.rank(ctx)
                        T.comm.get(ctx, 1, src=src[0, row : row + 2], dst=dst[0, 0:2])
                    else:
                        T.comm.get(ctx, 1, src=src[1, 2:5], dst=dst[0, 7:10])
                        T.comm.wait_peer(ctx, 1)

    return kernel


@pytest.mark.parametrize(
    ("case", "message"), [("cube", "explicit T.Vector"), ("strided", "contiguous rectangular"), ("size", "equal dtype")]
)
def test_invalid_communication(case, message):
    with pytest.raises(ValueError, match=message):
        transfer_case(case)


def test_communication_lowering():
    kernel = transfer_case("get")
    assert int(kernel.attrs["tl.ascend_ain"]) == 1
    source = tilelang.lower(kernel, target="ascend").kernel_source
    assert source.index("tl_ain::transfer<true>") < source.index("tl_ain::wait_peer")
    with pytest.raises(ValueError, match="require plain AscendC"):
        tilelang.lower(kernel, target="pto")


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(("case", "rank", "expected"), [("empty", 0, 0), ("get", 0, -1), ("bounds", 15, -2)])
def test_device_guards(case, rank, expected):
    # No fabric access can occur. The descriptor intentionally contains no handles.
    ctx = torch.tensor([0, rank, 1, 0, 0, 0, 0], dtype=torch.int64).to("npu")
    src = torch.arange(64, dtype=torch.float32).reshape(4, 16).to("npu")
    original = torch.full((4, 16), -1.0)
    dst = original.to("npu")
    kernel = tilelang.compile(transfer_case(case), target="ascend", out_idx=None)
    kernel(ctx, src, dst)
    assert ctx.cpu().tolist()[-1] == expected
    torch.testing.assert_close(dst.cpu(), original, rtol=0, atol=0)


def signal_case(kind):
    @T.prim_func
    def run(ctx: T.comm.Team, src: T.comm.Window((4,), "float32"), flag: T.comm.Window((4,), "uint64")):
        with T.Kernel(1), T.Vector(vector=1):
            if kind == "type":
                T.comm.signal_add(ctx, 1, src[0])
            elif kind == "extent":
                T.comm.wait_signal(ctx, flag[:2], 1, max_polls=5)
            elif kind == "defer":
                T.comm.put(ctx, 1, src=src[:], dst=src[:], defer=T.comm.rank(ctx))
            elif kind == "limit":
                T.comm.team_sync(ctx, max_polls=0)
            elif kind == "threshold":
                T.comm.wait_signal(ctx, flag[0], -1, max_polls=3)
            elif kind == "float_threshold":
                T.comm.wait_signal(ctx, flag[0], src[0], max_polls=3)
            elif kind == "negative_dynamic":
                T.comm.wait_signal(ctx, flag[0], T.comm.rank(ctx) - 1, max_polls=3)
            elif kind == "commit":
                T.comm.get(ctx, 1, src=src[:], dst=src[:], defer=True)
                T.comm.commit_peer(ctx, 1)
            else:
                T.comm.team_sync(ctx, max_polls=2)

    return run


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("type", ValueError, "one uint64"),
        ("extent", ValueError, "one uint64"),
        ("limit", ValueError, "max_polls"),
        ("threshold", ValueError, "nonnegative"),
        ("float_threshold", TypeError, "integer scalar"),
        ("defer", TypeError, "compile-time bool"),
    ],
)
def test_invalid_publication(case, error, message):
    with pytest.raises(error, match=message):
        signal_case(case)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(("case", "expected"), [("commit", -1), ("barrier", -3), ("negative_dynamic", -2)])
def test_device_publication_guards(case, expected):
    ctx = torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.int64).to("npu")
    a = torch.arange(4, dtype=torch.float32).to("npu")
    flag = torch.zeros(4, dtype=torch.uint64).to("npu")
    run = tilelang.compile(signal_case(case), target="ascend", out_idx=None)
    run(ctx, a, flag)
    assert ctx.cpu()[6].item() == expected


def test_channel_rejects_ain_only_kernel_before_resource_setup():
    from types import SimpleNamespace
    from tilelang.ascend.distributed import Team

    kernel = SimpleNamespace(prim_func=signal_case("barrier"))
    assert int(kernel.prim_func.attrs["tl.ascend_ain_sync"]) == 1
    team = Team.__new__(Team)
    team.transport = "channel"
    with pytest.raises(ValueError, match="require transport='ain'"):
        team.launch(kernel)
