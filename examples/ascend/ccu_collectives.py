"""Run with torchrun --standalone --nproc-per-node=2 examples/ascend/ccu_collectives.py."""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import tilelang.ascend.language as T
from tilelang.ascend.distributed import Team


@T.ccu.kernel
def all_gather(C, src, dst):
    n = src.size
    for peer in range(C.size):
        C.copy(src[:], dst[C.rank * n : (C.rank + 1) * n], peer=peer)
    C.wait_all()
    C.barrier()


@T.ccu.kernel
def rotate(C, src, dst):
    C.copy(src[1:], dst[2 : src.size + 1], peer=(C.rank + 1) % C.size)
    C.wait_all()
    C.barrier()


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    with torch.npu.stream(torch.npu.Stream()), Team.from_group() as team:
        for step in (0, 1000):
            base = torch.arange(32, dtype=torch.int32) + step
            src = (base + 100 * team.rank).to(team.device)
            dst = torch.empty(src.numel() * team.size, dtype=src.dtype, device=team.device)
            all_gather(src, dst, team=team)
            expected = torch.cat([base + 100 * r for r in range(team.size)])
            torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
            ref = torch.full((src.numel() + 2,), -1, dtype=src.dtype)
            out = ref.to(team.device)
            rotate(src, out, team=team)
            ref[2:-1] = base[1:] + 100 * ((team.rank - 1) % team.size)
            torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        print(f"rank {team.rank}: CCU AllGather and rotate passed twice", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
