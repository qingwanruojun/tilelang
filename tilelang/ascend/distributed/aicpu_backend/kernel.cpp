#include "plan.h"
#include <hccl/hcomm_primitives.h>

#define CHECK(op)                                                              \
  do {                                                                         \
    auto rc = (op);                                                            \
    if (rc != 0)                                                               \
      return rc;                                                               \
  } while (false)
namespace {
// All ranks post before any rank waits. Alternate slots between barriers.
int barrier(const Arguments &a, uint32_t slot) {
  for (uint32_t r = 0; r < a.size; ++r)
    if (r != a.rank)
      CHECK(
          HcommChannelNotifyRecordOnThread(a.queue, a.peers[r].channel, slot));
  for (uint32_t r = 0; r < a.size; ++r)
    if (r != a.rank)
      CHECK(HcommChannelNotifyWaitOnThread(a.queue, a.peers[r].channel, slot,
                                           1800));
  return 0;
}
int schedule(const Arguments &a) {
  auto *local = reinterpret_cast<char *>(a.peers[a.rank].window);
  // Preserve output holes. One ordered queue owns all copies in this program.
  if (a.srcBytes)
    CHECK(HcommLocalCopyOnThread(a.queue, local, a.src, a.srcBytes));
  if (a.dstBytes)
    CHECK(HcommLocalCopyOnThread(a.queue, local + a.outputOffset, a.dst,
                                 a.dstBytes));
  CHECK(barrier(a, 0));
  uint32_t slot = 1;
  for (uint64_t i = 0; i < a.count; ++i) {
    const auto &op = a.ops[i];
    if (op.kind == 1) {
      CHECK(barrier(a, slot));
      slot ^= 1;
    } else {
      const Peer &peer = a.peers[op.peer];
      auto *target = reinterpret_cast<char *>(peer.window) + a.outputOffset +
                     op.destination;
      if (op.peer == a.rank)
        CHECK(HcommLocalCopyOnThread(a.queue, target, local + op.source,
                                     op.bytes));
      else
        CHECK(HcommWriteOnThread(a.queue, peer.channel, target,
                                 local + op.source, op.bytes));
    }
  }
  if (a.dstBytes)
    CHECK(HcommLocalCopyOnThread(a.queue, a.dst, local + a.outputOffset,
                                 a.dstBytes));
  return 0;
}
} // namespace
extern "C" __attribute__((visibility("default"))) uint32_t
TileLangAicpuTransfer(void *raw) {
  const auto &a = *static_cast<const Arguments *>(raw);
  constexpr char tag[] = "tilelang_aicpu";
  CHECK(HcommBatchModeStart(tag));
  auto rc = HcommThreadNotifyWaitOnThread(a.queue, 0, 1800);
  if (!rc)
    rc = schedule(a);
  if (!rc)
    rc = HcommThreadNotifyRecordOnThread(a.queue, a.hostQueueOnDevice, 0);
  auto end = HcommBatchModeEnd(tag);
  return rc ? rc : end;
}
