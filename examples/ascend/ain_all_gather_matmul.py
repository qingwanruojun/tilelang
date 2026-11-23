"""Two-rank FP16 AllGather(A) @ B.T, pipelined across an AIV and its paired AIC."""

import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.distributed import Team


def build_agmm(rows=96):
    bm, k, n = 32, 64, 64
    if rows <= 0:
        raise ValueError("rows must be positive")
    rounds = (rows + bm - 1) // bm
    padded = rounds * bm

    @T.prim_func
    def agmm(
        ctx: T.comm.Team,
        a: T.comm.Window((padded, k), "float16"),
        stage: T.comm.Window((2 * bm, k), "float16"),
        b: T.Tensor((n, k), "float16"),
        y: T.Tensor((2 * padded, n), "float32"),
    ):
        with T.Kernel(1):
            a1 = T.alloc_l1((bm, k), "float16")
            b1 = T.alloc_l1((n, k), "float16")
            acc = T.alloc_l0c((bm, n), "float32")
            rank = T.comm.rank(ctx)
            with T.Vector() as sid:
                if sid == 0:
                    for r in T.unroll(rounds, explicit=True):
                        slot = r % 2
                        if r >= 2:
                            T.ascend_cross_core_wait_flag(4, "PIPE_S", 2 + slot)
                        valid = T.min(bm, rows - r * bm)
                        T.comm.get(ctx, 1 - rank, src=a[r * bm : r * bm + valid, :], dst=stage[slot * bm : slot * bm + valid, :])
                        T.comm.wait_peer(ctx, 1 - rank)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE3", slot)
                    for r in T.unroll(min(2, rounds), explicit=True):
                        T.ascend_cross_core_wait_flag(4, "PIPE_S", 2 + ((rounds - min(2, rounds) + r) % 2))
            with T.Cube():
                for r in T.unroll(rounds, explicit=True):
                    slot = r % 2
                    T.copy(b, b1)
                    T.copy(a[r * bm : (r + 1) * bm, :], a1)
                    T.ascend_pipe_barrier("PIPE_ALL")
                    T.gemm(a1, b1, acc, transpose_B=True, clear_accum=True)
                    T.ascend_pipe_barrier("PIPE_ALL")
                    T.copy(acc, y[rank * padded + r * bm, 0])
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE2", slot)
                    T.copy(stage[slot * bm : (slot + 1) * bm, :], a1, l2_cache_ctrl="NOTALLOC_DROP")
                    T.ascend_pipe_barrier("PIPE_ALL")
                    T.gemm(a1, b1, acc, transpose_B=True, clear_accum=True)
                    T.ascend_pipe_barrier("PIPE_ALL")
                    T.copy(acc, y[(1 - rank) * padded + r * bm, 0])
                    T.ascend_pipe_barrier("PIPE_ALL")
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 2 + slot)

    return tilelang.compile(agmm, target="ascend", out_idx=None, pass_configs={"tl.enable_auto_schedule": False})


def main():
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rows = 77
    kernel = build_agmm(rows)
    if "--compile-only" in sys.argv:
        print("compiled mixed AIV/AIC AGMM", flush=True)
        return
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 2
    with Team.from_group(transport="channel" if "--channel" in sys.argv else "ain") as team:
        padded = ((rows + 31) // 32) * 32
        a = team.alloc_window((padded, 64), torch.float16)
        stage = team.alloc_window((64, 64), torch.float16)
        torch.manual_seed(42)
        full = torch.randn(2 * rows, 64).half()
        b = torch.randn(64, 64).half().to(team.device)
        a.local.copy_(torch.nn.functional.pad(full.chunk(2)[team.rank], (0, 0, 0, padded - rows)))
        y = torch.empty(2 * padded, 64, device=team.device)
        team.barrier()
        team.launch(kernel, a, stage, b, y)
        team.barrier()
        actual = y.cpu().view(2, padded, 64)[:, :rows].reshape(2 * rows, 64)
        torch.testing.assert_close(actual, full.float() @ b.cpu().float().T, rtol=1e-4, atol=1e-4)
    dist.destroy_process_group()
    print("AGMM passed", flush=True)


if __name__ == "__main__":
    main()
