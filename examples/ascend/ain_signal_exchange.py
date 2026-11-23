"""Two-rank publication of two data tiles followed by a monotonic ready signal."""

import argparse
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.distributed import Team


def build_exchange():
    @T.prim_func
    def exchange(
        ctx: T.comm.Team,
        src: T.comm.Window((128,), "float32"),
        dst: T.comm.Window((128,), "float32"),
        flag: T.comm.Window((1,), "uint64"),
        generation: T.Tensor((1,), "int32"),
        seen: T.Tensor((1,), "uint64"),
    ):
        with T.Kernel(1), T.Vector(vector=1):
            peer = 1 - T.comm.rank(ctx)
            T.comm.put(ctx, peer, src=src[:64], dst=dst[:64], defer=True)
            T.comm.put(ctx, peer, src=src[64:], dst=dst[64:], defer=True)
            T.comm.commit_peer(ctx, peer)
            T.comm.wait_peer(ctx, peer)
            T.comm.signal_add(ctx, peer, flag[0])
            T.comm.wait_peer(ctx, peer)
            T.comm.wait_signal(ctx, flag[0], generation[0], max_polls=10000000)
            seen[0] = T.comm.read_signal(ctx, flag[0])
            T.comm.team_sync(ctx, max_polls=10000000)
            T.comm.wait_peer(ctx, peer)

    return exchange


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    kernel = tilelang.compile(build_exchange(), target="ascend", out_idx=None)
    if args.compile_only:
        print("compiled deferred transfers, signal and device barrier")
        return
    dist.init_process_group("gloo")
    if dist.get_world_size() != 2:
        raise ValueError("This example requires two ranks")
    with Team.from_group() as team:
        src = team.alloc_window((128,), "float32")
        dst = team.alloc_window((128,), "float32")
        flag = team.alloc_window((1,), "uint64")
        team.prepare()
        flag.local.copy_(torch.zeros(1, dtype=torch.uint64))
        seen = torch.empty(1, dtype=torch.uint64, device=team.device)
        for epoch in (1, 2):
            base = torch.arange(128, dtype=torch.float32) + 1000 * epoch
            src.local.copy_(base + team.rank)
            generation = torch.tensor([epoch], dtype=torch.int32, device=team.device)
            team.barrier()
            team.launch(kernel, src, dst, flag, generation, seen)
            team.barrier()
            torch.testing.assert_close(dst.local.cpu(), base + 1 - team.rank, rtol=0, atol=0)
            assert seen.cpu().item() == epoch
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
