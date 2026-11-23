#include "plan.h"
#include <acl/acl_rt.h>
#include <cstdio>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hcomm_primitives.h>
#include <vector>

#define CHECK(op)                                                              \
  do {                                                                         \
    auto rc = (op);                                                            \
    if (rc != 0) {                                                             \
      std::fprintf(stderr, "AICPU prepare: %s returned %d\n", #op, int(rc));   \
      return rc;                                                               \
    }                                                                          \
  } while (false)

// The caller owns a dedicated AICPU communicator and binds it to one stream.
extern "C" int tl_aicpu_prepare(HcclComm comm, aclrtStream stream,
                                uint32_t rank, uint32_t size, uint64_t *queues,
                                Peer *peers) try {
  ThreadHandle host, device, hostOnDevice, deviceOnHost;
  CHECK(
      HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &host));
  CHECK(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, &device));
  CHECK(HcclThreadExportToCommEngine(comm, 1, &host, COMM_ENGINE_AICPU_TS,
                                     &hostOnDevice));
  CHECK(HcclThreadExportToCommEngine(comm, 1, &device, COMM_ENGINE_CPU_TS,
                                     &deviceOnHost));
  queues[0] = device;
  queues[1] = hostOnDevice;
  queues[2] = host;
  queues[3] = deviceOnHost;
  std::vector<HcclChannelDesc> descs(size - 1);
  std::vector<ChannelHandle> channels(size - 1);
  if (!descs.empty())
    CHECK(HcclChannelDescInit(descs.data(), descs.size()));
  for (uint32_t r = 0; r < size; ++r) {
    if (r == rank)
      continue;
    CommLink *links = nullptr;
    uint32_t count = 0;
    CHECK(HcclRankGraphGetLinks(comm, 0, rank, r, &links, &count));
    uint32_t i = 0;
    while (i < count && links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UB_CTP)
      ++i;
    if (i == count)
      return HCCL_E_NOT_SUPPORT;
    auto &d = descs[r - (r > rank)];
    d.remoteRank = r;
    d.notifyNum = 2;
    d.channelProtocol = links[i].linkAttr.linkProtocol;
    d.localEndpoint = links[i].srcEndpointDesc;
    d.remoteEndpoint = links[i].dstEndpointDesc;
  }
  if (!descs.empty())
    CHECK(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(),
                             descs.size(), channels.data()));
  for (uint32_t r = 0; r < size; ++r) {
    void *addr = nullptr;
    if (r == rank) {
      CHECK(HcclGetHcclBuffer(comm, &addr, &peers[r].capacity));
    } else {
      peers[r].channel = channels[r - (r > rank)];
      CHECK(HcclChannelGetHcclBuffer(comm, peers[r].channel, &addr,
                                     &peers[r].capacity));
    }
    peers[r].window = reinterpret_cast<uint64_t>(addr);
  }
  return 0;
} catch (...) {
  return HCCL_E_INTERNAL;
}
