"""Run on two 950 devices: torchrun --standalone --nproc-per-node=2 examples/ascend/ain_exchange.py."""

import os
import sys
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.distributed import Team


def exchange(pull):
    @T.prim_func
    def run(ctx: T.comm.Team, src: T.comm.Window((128,), "float32"), dst: T.comm.Window((128,), "float32")):
        with T.Kernel(1):  # noqa: SIM117
            with T.Vector(vector=1):
                peer = 1 - T.comm.rank(ctx)
                if pull:
                    T.comm.get(ctx, peer, src=src[8:71], dst=dst[16:79])
                else:
                    T.comm.put(ctx, peer, src=src[8:71], dst=dst[16:79])
                T.comm.wait_peer(ctx, peer)

    return run


def main():
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    kernels = [tilelang.compile(exchange(pull), target="ascend", out_idx=None) for pull in (True, False)]
    if "--compile-only" in sys.argv:
        print("compiled AIN get/put", flush=True)
        return
    dist.init_process_group("gloo")
    if dist.get_world_size() != 2:
        raise ValueError("This example requires two ranks")
    rank = dist.get_rank()
    with torch.npu.stream(torch.npu.Stream()), Team.from_group(transport="channel" if "--channel" in sys.argv else "ain") as team:
        src = team.alloc_window((128,), torch.float32)
        dst = team.alloc_window((128,), torch.float32)
        sequence = torch.arange(128, dtype=torch.float32)
        for generation, kernel in enumerate(kernels * 2):
            base = 1000 * generation
            src.local.copy_(sequence + base + 100 * rank)
            dst.local.copy_(torch.full((128,), -1.0))
            team.barrier()
            team.launch(kernel, src, dst)
            team.barrier()
            expected = torch.full((128,), -1.0)
            expected[16:79] = sequence[8:71] + base + 100 * (1 - rank)
            torch.testing.assert_close(dst.local.cpu(), expected, rtol=0, atol=0)
    dist.destroy_process_group()
    print(f"rank {rank}: AIN window exchange passed", flush=True)


if __name__ == "__main__":
    main()
