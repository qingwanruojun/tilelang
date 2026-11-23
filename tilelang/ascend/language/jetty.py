"""Cooperative writes to a peer's UB_CTP send queue from one 32-lane SIMT warp."""

from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path

from tvm import DataType, arith, tirx
from tvm.script.ir_builder.tirx import frame
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.language.common import import_source
from tilelang.language.eager.builder import Builder
from .allocate import alloc_shared
from .comm import _region
from .frame import SimtVFFrame, VectorFrame


def workspace():
    """Allocate private Jetty queue metadata in the owning AIV's UB."""
    return alloc_shared((512,), "uint8")


@lru_cache(maxsize=4)
def _device_source(root):
    sdk = Path(root) / "asc"
    digest = sha256()
    for prefix in ("include", "impl"):
        for family in ("jetty", "asccomm", "hcomm", "utils"):
            directory = sdk / prefix / "comm_api/aicore" / family
            for header in sorted(directory.rglob("*.h")):
                digest.update(header.relative_to(sdk).as_posix().encode())
                digest.update(header.read_bytes())
    source = (Path(TILELANG_TEMPLATE_PATH) / "tl_templates/ascend/jetty.h").read_text()
    return f"// SIMT Jetty SDK {digest.hexdigest()}\n{source}"


def _call(ctx, queue, name, *args):
    builder = Builder.current()
    frames = builder.frames
    simt = next((f for f in reversed(frames) if isinstance(f, SimtVFFrame)), None)
    if simt is None or list(map(int, simt.thread_extents)) != [32, 1, 1] or not any(isinstance(f, VectorFrame) for f in frames):
        raise ValueError("Jetty requires T.Vector and a converged T.SimtVF(threads=32)")
    if queue.dtype != "uint8" or queue.scope() not in ("shared", "shared.dyn") or tuple(map(int, queue.shape)) != (512,):
        raise ValueError("Use T.jetty.workspace() in the owning vector scope")
    inner_frames = frames[builder.find_frame_idx(SimtVFFrame) + 1 :]
    if any(isinstance(f, frame.IfFrame) for f in inner_frames):
        raise ValueError("Jetty operations must be reached by the entire warp outside a device conditional")
    if name == "init" and any(isinstance(f, frame.ForFrame) for f in inner_frames):
        raise ValueError("Initialize the Jetty queue before the SIMT loop")
    if not hasattr(builder, "_jetty_owners"):
        builder._jetty_owners = {}
    key = simt, queue.data
    if name == "init":
        if key in builder._jetty_owners:
            raise ValueError("Initialize a Jetty workspace only once per SIMT region")
        builder._jetty_owners[key] = ctx.data
    elif key not in builder._jetty_owners or not ctx.data.same_as(builder._jetty_owners[key]):
        raise ValueError("Initialize the workspace with the same team in this SIMT region first")
    builder.inferred_func_attrs["tl.ascend_ain"] = 1  # Ordered AscendC lowering for remote side effects.
    builder.inferred_func_attrs["tl.ascend_jetty"] = 1
    import_source(_device_source(os.environ.get("ASCEND_HOME_PATH", "")))
    return tirx.call_extern("void", "tl_jetty::" + name, ctx.data, queue.data, *args)


def init(ctx, peer, queue):
    """Bind one warp to one channel. All lanes pass the same group-local peer."""
    return _call(ctx, queue, "init", peer)


def post_write(ctx, queue, *, src, dst, segments):
    """First segments lanes contribute one equal-sized slice each to a contiguous remote dst."""
    if type(segments) is not int or not 1 <= segments <= 12:
        raise ValueError("segments must be a constant from 1 through 12")
    analyzer = arith.Analyzer()
    s, so, n, sv = _region(src, analyzer)
    d, do, m, dv = _region(dst, analyzer)
    ty = DataType(s.dtype)
    if s.dtype != d.dtype or ty.lanes != 1 or ty.bits % 8 or not analyzer.can_prove_equal(n * segments, m):
        raise ValueError("dst must fit exactly segments equal-sized source slices of the same whole-byte dtype")
    item = ty.bits // 8
    return _call(ctx, queue, f"post_write<{segments}>", s.data, so * item, d.data, do * item, n * item, sv, dv)


def publish(ctx, queue):
    """Advance the send queue and ring its doorbell once for all pending writes."""
    return _call(ctx, queue, "publish")


def drain(ctx, queue, *, timeout_cycles=1000000):
    """Wait for this queue's published operations. Drain after the final publish."""
    if type(timeout_cycles) is not int or not 0 < timeout_cycles < 2**63:
        raise ValueError("timeout_cycles must be a positive constant below 2**63")
    return _call(ctx, queue, f"drain<{timeout_cycles}>")
