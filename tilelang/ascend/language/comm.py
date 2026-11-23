"""Kernel-side symmetric window operations for the AscendC AIV backend."""

from functools import lru_cache
from hashlib import sha256
from math import prod
import os
from pathlib import Path

from tvm import arith, tirx
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.language.common import Tensor, import_source
from tilelang.utils.language import to_buffer_region
from .allocate import alloc_shared


Team = Tensor((7,), "int64")
Window = Tensor


def rank(team):
    return tirx.Cast("int32", team[1])


def size(team):
    return tirx.Cast("int32", team[2])


def _region(view, analyzer):
    r = to_buffer_region(view)
    if r.buffer.scope() != "global":
        raise ValueError("AIN transfers require GM windows")
    for i, stride in enumerate(r.buffer.strides):
        if not analyzer.can_prove_equal(stride, prod(r.buffer.shape[i + 1 :])):
            raise ValueError("AIN transfers require compact GM windows")
    offset = tirx.IntImm("int64", 0)
    count = tirx.IntImm("int64", 1)
    span = tirx.IntImm("int64", 0)
    valid = tirx.const(True, "bool")
    for bound, dim in zip(r.region, r.buffer.shape):
        offset = offset * dim + bound.min
        span = span * dim + bound.extent - 1
        count = count * bound.extent
        if analyzer.can_prove(bound.min < 0) or analyzer.can_prove(bound.extent < 0) or analyzer.can_prove(bound.min + bound.extent > dim):
            raise ValueError("Window slice is out of bounds")
        valid = tirx.all(valid, bound.min >= 0, bound.extent >= 0, bound.min + bound.extent <= dim)
    if not analyzer.can_prove_equal(count, 0) and not analyzer.can_prove_equal(span + 1, count):
        raise ValueError("AIN transfers require a contiguous rectangular slice")
    return r.buffer, offset + r.buffer.elem_offset, count, analyzer.simplify(valid)


@lru_cache(maxsize=4)
def _device_source(root):
    # Public and inline implementation headers are delivered together in CANN.
    # Include their identity in the PrimFunc so both JIT and binary caches see it.
    sdk = Path(root) / "asc"
    if not (sdk / "include/comm_api/aicore/ain/ain.h").is_file():
        raise RuntimeError("T.comm requires the CANN AscendC AIN headers under ASCEND_HOME_PATH")
    digest = sha256()
    for kind in ("include", "impl"):
        for family in ("ain", "hcomm"):
            directory = sdk / kind / "comm_api/aicore" / family
            for header in sorted(directory.rglob("*.h")):
                digest.update(header.relative_to(sdk).as_posix().encode())
                digest.update(header.read_bytes())
    helper = (Path(TILELANG_TEMPLATE_PATH) / "tl_templates/ascend/ain.h").read_text()
    return f"// AIN SDK {digest.hexdigest()}\n{helper}"


def _import_ain(*, symmetric=False):
    from tilelang.language.eager.builder import Builder
    from .frame import CubeFrame, SimdVFFrame, SimtVFFrame, VectorFrame

    builder = Builder.current()
    scopes = (CubeFrame, VectorFrame, SimdVFFrame, SimtVFFrame)
    owner = next((frame for frame in reversed(builder.frames) if isinstance(frame, scopes)), None)
    if not isinstance(owner, VectorFrame):
        raise ValueError("T.comm transfer and wait require an explicit T.Vector() AIV scope")
    builder.inferred_func_attrs["tl.ascend_ain"] = 1
    if symmetric:
        builder.inferred_func_attrs["tl.ascend_ain_sync"] = 1
    import_source(_device_source(os.environ.get("ASCEND_HOME_PATH", "")))


def _transfer(team, peer, src, dst, pull, defer):
    if not isinstance(defer, bool):
        raise TypeError("defer must be a compile-time bool")
    analyzer = arith.Analyzer()
    s, so, n, src_valid = _region(src, analyzer)
    d, do, m, dst_valid = _region(dst, analyzer)
    if s.dtype != d.dtype or not analyzer.can_prove_equal(n, m):
        raise ValueError("Window slices must have equal dtype and element count")
    from tvm import DataType

    ty = DataType(s.dtype)
    if ty.lanes != 1 or ty.bits % 8 != 0:
        raise ValueError("AIN supports whole-byte scalar element types")
    item = ty.bits // 8
    _import_ain()
    scratch = alloc_shared((512,), "uint8")
    direction = "true" if pull else "false"
    template = f"{direction}, false" if defer else direction
    return tirx.call_extern(
        "void",
        f"tl_ain::transfer<{template}>",
        team.data,
        peer,
        s.data,
        so * item,
        d.data,
        do * item,
        n * item,
        analyzer.simplify(tirx.all(src_valid, dst_valid)),
        scratch.data,
    )


def get(team, peer, *, src, dst, defer=False):
    """Post a transfer (optionally defer its doorbell) from the peer's src slice into the local dst slice."""
    return _transfer(team, peer, src, dst, True, defer)


def put(team, peer, *, src, dst, defer=False):
    """Post a transfer (optionally defer its doorbell) from the local src slice into the peer's dst slice."""
    return _transfer(team, peer, src, dst, False, defer)


def wait_peer(team, peer):
    """Drain this AIV owner's pending operations on the peer channel."""
    _import_ain()
    scratch = alloc_shared((512,), "uint8")
    return tirx.call_extern("void", "tl_ain::wait_peer", team.data, peer, scratch.data)


def commit_peer(team, peer):
    """Publish deferred operations on this owner's UB_CTP channel; use wait_peer for completion."""
    _import_ain()
    scratch = alloc_shared((512,), "uint8")
    return tirx.call_extern("void", "tl_ain::commit_peer", team.data, peer, scratch.data)


def _signal_slot(view):
    analyzer = arith.Analyzer()
    buf, index, count, valid = _region(view, analyzer)
    if buf.dtype != "uint64" or not analyzer.can_prove_equal(count, 1):
        raise ValueError("Signal operations require exactly one uint64 GM window element")
    _import_ain(symmetric=True)
    return buf.data, index, valid


def _positive_constant(value, name):
    if type(value) is not int or not 0 < value < (1 << 63):
        raise ValueError(f"{name} must be a positive compile-time int below 2**63")
    return tirx.IntImm("int64", value)


def signal_add(team, peer, slot, *, delta=1):
    """Atomically add to a peer's uint64 signal. Drain prior data before publishing readiness."""
    args = _signal_slot(slot)
    delta = _positive_constant(delta, "delta")
    scratch = alloc_shared((512,), "uint8")
    return tirx.call_extern("void", "tl_ain::signal_add", team.data, peer, *args, delta, scratch.data)


def read_signal(team, slot):
    """Read the local monotonic signal with AIN relaxed ordering."""
    return tirx.call_extern("uint64", "tl_ain::read_signal", team.data, *_signal_slot(slot))


def wait_signal(team, slot, least, *, max_polls):
    """Wait until the local signal reaches least, or record a sticky device timeout."""
    if isinstance(least, int):
        if least < 0 or least >= 1 << 64:
            raise ValueError("Signal threshold must be nonnegative and fit uint64")
        least = tirx.IntImm("uint64", least)
    elif getattr(least, "dtype", "") not in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        raise TypeError("Signal threshold must be an integer scalar")
    limit = _positive_constant(max_polls, "max_polls")
    return tirx.call_extern("void", "tl_ain::wait_signal", team.data, *_signal_slot(slot), least, limit)


def team_sync(team, *, max_polls):
    """Join AIN barrier session 0. Exactly one AIV owner per rank must participate."""
    _import_ain(symmetric=True)
    limit = _positive_constant(max_polls, "max_polls")
    scratch = alloc_shared((512,), "uint8")
    return tirx.call_extern("void", "tl_ain::team_sync", team.data, limit, scratch.data)
