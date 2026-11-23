# Ascend 950 distributed communication

A TileLang kernel can submit a tile transfer on an AIV while its paired AIC runs matrix operations. Applications can also submit a whole AllGather on an NPU stream. The examples use CANN 9.2.0, an Ascend-enabled TileLang build, Bisheng, torch_npu and a PyTorch process group. Set the current device per process before creating a team. Source the CANN environment so the compiler can find the AIN and Hcomm headers and libraries. The host adapter builds on first use with the PyTorch C++ extension toolchain.

## Window exchange

```python
from tilelang.ascend.distributed import Team

with Team.from_group(group) as team:
    src = team.alloc_window((128,), "float32")
    dst = team.alloc_window((128,), "float32")
    src.local.copy_(x)
    team.barrier()
    team.launch(exchange_kernel, src, dst)
    team.barrier()
    y.copy_(dst.local)
```

Each member creates windows in the same order with the same shapes and dtypes. The team has its own HCCL domain with group-local rank numbering and captures the current stream. Use that stream for local accesses and launches. Keep window storage at its registered address and capacity for the lifetime of the team. Keep ordinary tensor arguments alive until the stream completes. Finish peer accesses before overwriting input or releasing windows. All members participate in setup, barriers and teardown.

The compiled kernel takes its team as its first argument. Window annotations describe the local layout. At launch the device looks up both GM base addresses in the team's registration table. Only contiguous slices with equal dtype and element count are supported. Dynamic coordinates are checked on device. Each nonempty transfer must target another group member and fit in 256 MiB. Empty slices are valid.

```python
with T.Kernel(1):
    with T.Vector(vector=1):
        peer = 1 - T.comm.rank(ctx)
        T.comm.get(ctx, peer, src=src[8:71], dst=dst[16:79])
        T.comm.wait_peer(ctx, peer)
```

`get` reads the peer source into the local destination. `put` writes the local source into the peer destination. Assign exactly one AIV to drive each peer channel and explicitly wait before consuming or reusing its buffers. For a remote consumer, arrange a rank rendezvous or a separate notification. `team.barrier()` synchronizes the submission stream and exchanges status with the group. Local AIC consumers in a mixed kernel need an explicit ready/free protocol. The AscendC backend keeps the author's communication schedule. PTO compilation of these operations raises a diagnostic.

The default transport registers Hcomm allocations as symmetric windows and uses AIN over UB_CTP. Explicit `Team.from_group(transport="channel")` registers local tensors and acquires Hcomm AIV channels over the same protocol. Both paths require a working UB fabric between participating NPUs. HCCL setup errors carry the failing function, group rank and native status. The wrapper also reports device argument errors and the return code from Hcomm channel drain. CANN's symmetric AIN get/put entry points return void. A runtime stream fault can still surface at synchronization.

## Post, publish, complete and notify

After the last collective window allocation, call `team.prepare()` on all ranks to move AIV resource setup outside a measured loop. The first `team.launch` also prepares the team if needed. The prepared window set is fixed until close.

For multiple requests to one peer, post `T.comm.put(..., defer=True)` or `get(..., defer=True)`, call `T.comm.commit_peer(ctx, peer)` once to publish the queued requests, then `T.comm.wait_peer(ctx, peer)` before consuming the result or recycling storage. This works with either transport. One AIV owns each channel. Batch size must fit the native send queue. Commit rings the doorbell. Drain waits for prior operations on that channel.

With the symmetric AIN transport, allocate a registered uint64 signal window and initialize the local counters before the group begins. The sender drains the data writes, calls `T.comm.signal_add(ctx, peer, flag[0])`, then drains the notification. The receiver calls `T.comm.wait_signal(ctx, flag[0], generation, max_polls=...)` or samples the local counter with `T.comm.read_signal(ctx, flag[0])`. Use increasing generations and avoid wraparound. Signal reads use relaxed device ordering. A local AIC consumer still needs a ready/free handoff from its AIV owner.

`T.comm.team_sync(ctx, max_polls=...)` runs AIN barrier session zero on the device. Each rank must enter the same sequence of barriers with one AIV owner. Serialize signal submissions and barrier use of the team's shared AIN shadow storage. After a device barrier, drain its outgoing peer notifications before reusing the session or releasing resources. The poll limit is a retry budget, not a wall-clock duration. A device timeout records status -4. Discard results and close the team after an error reported by `team.barrier()`. Explicit channel transport supports transfers, commit and drain. The native AIN team is required for counters and device rendezvous.

## Whole-operation AllGather

```python
from tilelang.ascend.distributed import all_gather_into

all_gather_into(dst, src, team=team, engine="ccu")
```

Pass contiguous tensors on the team device. The output stores all inputs flattened in group rank order. Supported dtypes are FP16, BF16, FP32 and INT32. An engine policy selects HCCL AICPU_TS or CCU_SCHED on a separate cached communicator. HCCL selects the concrete algorithm. Each call collectively validates the arguments before enqueueing the operation on the team stream.

### Author a CCU communication kernel in TileLang

Use the Ascend language module and submit the decorated function on a Team. The function body builds a rank-specialized CCU program; the CPU does not execute the tensor copies. Each rank runs the same Python definition.

```python
import tilelang.ascend.language as T

@T.ccu.kernel
def all_gather(C, src, dst):
    n = src.size
    for peer in range(C.size):
        C.copy(src[:], dst[C.rank * n : (C.rank + 1) * n], peer=peer)
    C.wait_all()
    C.barrier()

all_gather(src_tensor, dst_tensor, team=team)
```

Input and output are distinct contiguous tensors of the same dtype. Slices use flattened element offsets. `C.rank` and `C.size` describe the group; Python loops and conditionals specialize on these and on tensor sizes. A self copy writes local GM; another peer writes its output buffer. `wait_all()` drains this rank's copies. All ranks execute matching barriers, including the final rendezvous that makes incoming writes safe for subsequent work on the Team stream. Copies to overlapping destination ranges in one phase are rejected. The first implementation supports one input and one output, contiguous copies of up to 256 MiB each, up to 16 outstanding copies between waits, and direct mesh channels. Its primitives expose transfer and synchronization; arithmetic reductions and device-data-dependent branches require additional lowering support.

TileLang traces each rank's schedule and checks region bounds, matching barriers and write conflicts. On first use it binds channels, exchanges output addresses and memory tokens, emits local copies, remote writes, completion events and notify/wait, and registers a CCU microprogram. The Team reuses one CCU communicator, stream thread and peer channels. It caches the registered microprogram handle by schedule digest. Each call supplies current buffer addresses and tokens on its stream. CANN 9.2.0 provides the Hcomm CCU builder used by this backend. Users provide Python and tensor arguments.

Run `torchrun --standalone --nproc-per-node=2 examples/ascend/ccu_collectives.py` for both mesh AllGather and ring rotation with nonzero offsets and untouched guard elements.

### Interoperate with an existing native algorithm

```python
from tilelang.ascend.distributed import NativeAllGather, all_gather_into

program = NativeAllGather("/path/to/liballgather.so", symbol="HcclAllGatherCustom")
all_gather_into(dst, src, team=team, engine="ccu", program=program)
```

Build the native program for the installed CANN release and target. Its C-linkage host entry has ABI `HcclResult(const void *src, void *dst, uint64_t count, HcclDataType dtype, HcclComm comm, aclrtStream stream)`. Count is per rank, in elements. The entry must enqueue its work on the supplied stream and return the launch status. Device code and deployment dependencies are the algorithm author's responsibility. An AICPU program can schedule thread/channel notifications and data movement. A CCU program can register and launch microcode using events. The wrapper retains the loaded entry and a dedicated communicator per engine, library digest and symbol for the team lifetime. All members must supply the same binary and symbol. Local file paths may differ. With `program=None`, the built-in HCCL path remains available.

The CANN HCCL examples under `examples/05_custom_ops_allgather/{ccu,aicpu}` expose this ABI. To exercise a compatible native build, pass `--library /path/to/lib.so --symbol HcclAllGatherCustom` to the collective example. Native program calls and device-side communication are separate submission paths. The AIV and CCU frontends use different execution domains. The SIMT Jetty API below exposes cooperative queue posting. The application controls the communication schedule.

## SIMT Jetty writes in a communication or fused kernel

Create a team with `transport="channel"` and register a GM window. Within an
explicit `T.Vector(vector=1)` scope, allocate `q = T.jetty.workspace()`. One
converged `T.SimtVF(threads=32)` owns the channel and calls:

1. `T.jetty.init(ctx, peer, q)` to bind the workspace to a group-local peer.
2. `T.jetty.post_write(ctx, q, src=..., dst=..., segments=N)` to append a
   gathered write. The first N lanes each supply a contiguous, equal-length
   source slice. All 32 lanes call the operation with the same destination.
   N ranges from 1 to 12. The destination length is N times the source length.
3. `T.jetty.publish(ctx, q)` to advance the send queue and ring the doorbell
   for the pending writes. Multiple posts can share one publication.
4. `T.jetty.drain(ctx, q)` to wait for completion. Independent local work
   may run between publish and drain. Coordinate remote consumption and window
   reuse explicitly.

All lanes must follow the same queue operations, including loop trip counts.
Hold exclusive ownership through the final drain. Use a destination inside the
window covered by the peer descriptor. The example registers one window and
uses disjoint source and destination slices. The queue must have space for all
requests in each unpublished batch. Team synchronization reports device errors.

`examples/ascend/jetty_exchange.py` contains both a pure communication kernel
and a fused write and local scale. It tests noncontiguous source segments,
unchanged destination guards and repeated launches against CPU references.
It targets Ascend 950, plain AscendC and CANN 9.3.0+ with SIMT Jetty headers.
Run with `torchrun --standalone --nproc-per-node=2 examples/ascend/jetty_exchange.py`.
Use `--lower-only` to inspect generated source.

## Run

```bash
torchrun --standalone --nproc-per-node=2 examples/ascend/ain_exchange.py
torchrun --standalone --nproc-per-node=2 examples/ascend/ain_signal_exchange.py
torchrun --standalone --nproc-per-node=2 examples/ascend/ain_all_gather_matmul.py
torchrun --standalone --nproc-per-node=2 examples/ascend/ain_collective.py --engine aicpu
torchrun --standalone --nproc-per-node=2 examples/ascend/ain_collective.py --engine ccu
torchrun --standalone --nproc-per-node=3 examples/ascend/ain_collective.py --engine ccu --ranks 1 2
```

The two window examples accept `--channel` to select Hcomm explicitly, or `--compile-only` when run with Python. The exchange checks offset slices, tail bytes, untouched elements, repeated launches and a non-default stream. The two-rank fused example computes `AllGather(A) @ B.T` for FP16 inputs and FP32 output, using two staging slots for 77 rows per rank. Its final transfer carries 13 rows. The Cube pipeline uses 32-row padded GM tiles and the caller selects the 77 valid output rows. Its reference uses PyTorch on CPU.

## Ascend 950 validation

On physical devices 1 and 3 of `ascend-npu-950-shell` with CANN 9.2.0, the Python kernels in `ccu_collectives.py` each passed two rounds on a non-default stream. AllGather outputs matched the rank-ordered reference; rotation also preserved the guard elements.

In the earlier native-ABI test, both ranks matched the reference in two consecutive launches of the HCCL example's custom CCU AllGather on a non-default stream. The built-in AICPU, built-in CCU and custom CCU paths also ran on one team. A missing library on one rank propagated an error to both ranks before native submission.

Bisheng compiled the two-tile signal exchange. Its two-rank run stopped at `HcclCommSymWinRegister` with status 15 on both ranks. The explicit-channel exchange previously reached stream synchronization and timed out. End-to-end AIV data correctness and communication/computation overlap still require verification on a working fabric. The custom AICPU host library and device kernel built and launched, but stream synchronization reported 507018 in both TileLang and the standalone native example. The built-in AICPU numerical check passed separately.
