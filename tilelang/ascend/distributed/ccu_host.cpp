// Lower TileLang CCU transfer plans with the host-side CCU builder in CANN 9.2.
#include <acl/acl_rt.h>
#include <ccu_launch.h>
#include <ccu_primitives.hpp>
#include <ccu_res.h>
#include <cstdint>
#include <exception>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <vector>

namespace {
namespace ccu = AscendC::ccu;
#define TL_CHECK(call)                                                         \
  do {                                                                         \
    auto rc = (call);                                                          \
    if (rc != 0)                                                               \
      return rc;                                                               \
  } while (0)

// ctypes supplies five uint64 words per instruction. Offset and count use
// bytes.
enum Opcode : uint64_t { Copy, WaitAll, Barrier };
struct Instruction {
  uint64_t opcode, peer, source, destination, length;
};
struct Plan {
  uint32_t rank, size;
  const Instruction *code;
  uint32_t count;
  const ChannelHandle *channels;
};

CcuResult compilePlan(CcuKernelArg arg) {
  auto &p = *static_cast<const Plan *>(arg);
  ccu::Variable input, inputToken;
  std::vector<ccu::Variable> output(p.size), token(p.size);
  for (uint32_t r = 0; r < p.size; ++r) {
    if (r == p.rank)
      continue;
    output[r] = ccu::GetResByChannel<ccu::Variable>(p.channels[r], 1);
    token[r] = ccu::GetResByChannel<ccu::Variable>(p.channels[r], 2);
  }
  TL_CHECK(ccu::LoadArg(input, 0));
  TL_CHECK(ccu::LoadArg(output[p.rank], 1));
  TL_CHECK(ccu::LoadArg(inputToken, 2));
  TL_CHECK(ccu::LoadArg(token[p.rank], 3));

  for (uint32_t r = 0; r < p.size; ++r) {
    if (r == p.rank)
      continue;
    TL_CHECK(
        ccu::WriteVariableWithNotify(p.channels[r], output[p.rank], 1, 0, 2));
    TL_CHECK(
        ccu::WriteVariableWithNotify(p.channels[r], token[p.rank], 2, 0, 4));
  }
  for (uint32_t r = 0; r < p.size; ++r)
    if (r != p.rank) {
      TL_CHECK(ccu::NotifyWait(p.channels[r], 0, 6));
    }

  ccu::Event done;
  ccu::LocalAddr src, local;
  ccu::RemoteAddr remote;
  ccu::Variable len, offset;
  uint16_t mask = 0, phase = 8;
  for (uint32_t i = 0; i < p.count; ++i) {
    const auto &op = p.code[i];
    if (op.opcode == WaitAll) {
      if (mask) {
        TL_CHECK(ccu::EventWait(done, mask));
      }
      mask = 0;
    } else if (op.opcode == Barrier) {
      for (uint32_t r = 0; r < p.size; ++r)
        if (r != p.rank) {
          TL_CHECK(ccu::NotifyRecord(p.channels[r], 0, phase));
        }
      for (uint32_t r = 0; r < p.size; ++r)
        if (r != p.rank) {
          TL_CHECK(ccu::NotifyWait(p.channels[r], 0, phase));
        }
      phase = phase == 8 ? 16 : 8;
    } else {
      if (op.peer >= p.size || mask == UINT16_MAX || op.opcode != Copy)
        return CCU_E_PARA;
      uint16_t bit = mask + 1;
      mask |= bit;
      offset = op.source;
      src.addr = input;
      src.addr += offset;
      src.token = inputToken;
      offset = op.destination;
      len = op.length;
      if (op.peer == p.rank) {
        local.addr = output[p.rank];
        local.addr += offset;
        local.token = token[p.rank];
        TL_CHECK(ccu::LocalCopy(local, src, len, done, bit));
      } else {
        remote.addr = output[op.peer];
        remote.addr += offset;
        remote.token = token[op.peer];
        TL_CHECK(ccu::Write(p.channels[op.peer], remote, src, len, done, bit));
      }
    }
  }
  return CCU_SUCCESS;
}
} // namespace

extern "C" int tl_ccu_channels(HcclComm comm, aclrtStream stream, uint32_t rank,
                               uint32_t size, ThreadHandle *thread,
                               ChannelHandle *channels) try {
  TL_CHECK(
      HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 0, thread));
  for (uint32_t peer = 0; peer < size; ++peer) {
    if (rank == peer)
      continue;
    CommLink *links = nullptr;
    uint32_t n = 0;
    TL_CHECK(HcclRankGraphGetLinks(comm, 0, rank, peer, &links, &n));
    HcclChannelDesc desc;
    TL_CHECK(HcclChannelDescInit(&desc, 1));
    uint32_t i = 0;
    while (i < n && links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UB_CTP)
      ++i;
    if (i == n)
      return HCCL_E_NOT_SUPPORT;
    desc.remoteRank = peer;
    desc.notifyNum = 3;
    desc.localEndpoint = links[i].srcEndpointDesc;
    desc.remoteEndpoint = links[i].dstEndpointDesc;
    desc.channelProtocol = links[i].linkAttr.linkProtocol;
    TL_CHECK(
        HcclChannelAcquire(comm, COMM_ENGINE_CCU, &desc, 1, &channels[peer]));
  }
  return HCCL_SUCCESS;
} catch (const std::exception &) {
  return HCCL_E_INTERNAL;
}

extern "C" int tl_ccu_register(HcclComm comm, uint32_t rank, uint32_t size,
                               const ChannelHandle *channels,
                               const Instruction *ops, uint32_t count,
                               const char *name, CcuKernelHandle *kernel) try {
  Plan p{rank, size, ops, count, channels};
  CcuInsHandle instance{};
  uint32_t instances = 0;
  TL_CHECK(HcclCommQueryCcuIns(comm, &instance, &instances));
  if (instances != 1)
    return HCCL_E_NOT_SUPPORT;
  const void *args[]{&p};
  TL_CHECK(HcommCcuKernelRegisterStart(instance));
  TL_CHECK(HcommCcuKernelRegister(instance, 0, name,
                                  reinterpret_cast<void *>(compilePlan), args,
                                  1, kernel));
  return HcommCcuKernelRegisterEnd(instance);
} catch (const std::exception &) {
  return HCCL_E_INTERNAL;
}

extern "C" int tl_ccu_args(uint64_t src, uint64_t dst, uint64_t srcBytes,
                           uint64_t dstBytes, uint64_t *args) {
  args[0] = src;
  args[1] = dst;
  args[2] = args[3] = 0;
  if (srcBytes) {
    TL_CHECK(HcommCcuGetMemToken(src, srcBytes, &args[2]));
  }
  if (dstBytes) {
    TL_CHECK(HcommCcuGetMemToken(dst, dstBytes, &args[3]));
  }
  return HCCL_SUCCESS;
}

extern "C" int tl_ccu_launch(ThreadHandle thread, CcuKernelHandle kernel,
                             const uint64_t *args) {
  return HcommCcuKernelLaunch(thread, kernel, args, 4);
}
