"""Check HCCL AllGather in a world group or a selected subgroup."""

import argparse
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from tilelang.ascend.distributed import NativeAllGather, Team, all_gather_into


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["aicpu", "ccu"], required=True)
    parser.add_argument("--ranks", nargs="+", type=int, help="Global ranks to include in the group")
    parser.add_argument("--library", help="Host library of a custom AICPU or CCU AllGather")
    parser.add_argument("--symbol", default="AllGatherCustom", help="C-linkage AllGather host entry")
    args = parser.parse_args()
    program = NativeAllGather(args.library, args.symbol) if args.library else None
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    group = dist.group.WORLD if args.ranks is None else dist.new_group(args.ranks, backend="gloo")
    if group != dist.GroupMember.NON_GROUP_MEMBER:
        members = dist.get_process_group_ranks(group)
        with torch.npu.stream(torch.npu.Stream()), Team.from_group(group) as team:
            base = torch.arange(32, dtype=torch.float32)
            for step in (0, 100):
                src = (base + 32 * rank + step).to(team.device)
                dst = torch.empty(32 * team.size, dtype=src.dtype, device=team.device)
                all_gather_into(dst, src, team=team, engine=args.engine, program=program)
                expected = torch.cat([base + 32 * member + step for member in members])
                torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
        print(f"{args.engine} AllGather passed on rank {rank} in {members}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
