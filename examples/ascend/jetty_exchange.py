"""Two-rank gathered write, optionally fused with local elementwise computation."""

import argparse
import os
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.distributed import Team


def exchange(fused):
    @T.prim_func
    def run(ctx: T.comm.Team, buf: T.comm.Window((128,), "int32"), out: T.Tensor((32,), "int32")):
        with T.Kernel(1), T.Vector(vector=1):
            queue = T.jetty.workspace()
            with T.SimtVF(threads=32):
                lane = T.get_thread_binding()
                T.jetty.init(ctx, 1 - T.comm.rank(ctx), queue)
                # Lanes 0 and 1 gather [0:4] and [8:12] into one remote slice.
                T.jetty.post_write(ctx, queue, src=buf[lane * 8 : lane * 8 + 4], dst=buf[64:72], segments=2)
                T.jetty.publish(ctx, queue)
                if fused:
                    out[lane] = buf[lane] * 2
                T.jetty.drain(ctx, queue)

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.lower_only:
        for fused in (False, True):
            print(tilelang.lower(exchange(fused), target="ascend").kernel_source)
        return
    torch.npu.set_device(int(os.getenv("LOCAL_RANK", "0")))
    kernels = [tilelang.compile(exchange(f), target="ascend", out_idx=None) for f in (False, True)]
    if args.compile_only:
        return
    dist.init_process_group("gloo")
    if dist.get_world_size() != 2:
        raise ValueError("Run with exactly two ranks")
    with torch.npu.stream(torch.npu.Stream()), Team.from_group(transport="channel") as team:
        window = team.alloc_window((128,), torch.int32)
        for i, kernel in enumerate(kernels * 2):
            local = torch.arange(128, dtype=torch.int32) + 1000 * team.rank + 2000 * i
            remote = local + 1000 * (1 - 2 * team.rank)
            window.local.copy_(local)
            out = torch.full((32,), -1, dtype=local.dtype, device=team.device)
            team.barrier()
            team.launch(kernel, window, out)
            team.barrier()
            expected = local.clone()
            expected[64:72] = torch.cat((remote[:4], remote[8:12]))
            torch.testing.assert_close(window.local.cpu(), expected)
            torch.testing.assert_close(out.cpu(), local[:32] * 2 if i % 2 else torch.full((32,), -1, dtype=local.dtype))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
