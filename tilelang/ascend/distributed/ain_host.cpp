#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <hccl/hccl.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_team.h>
#include <hcomm/hcomm_res.h>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <torch/library.h>
#include <torch_npu/csrc/aten/common/from_blob.h>

extern "C" size_t tl_ain_root_size() { return sizeof(HcclRootInfo); }

extern "C" int tl_ain_comm(const HcclRootInfo *root, uint32_t size,
                           uint32_t rank, uint32_t expansion, HcclComm *out) {
  HcclCommConfig config;
  HcclCommConfigInit(&config);
  config.hcclWorldRankID = rank;
  config.hcclOpExpansionMode = expansion;
  return HcclCommInitRootInfoConfig(size, root, rank, &config, out);
}

extern "C" int tl_ain_team(HcclComm comm, uint32_t size, uint32_t rank,
                           HcommTeamHandle *out) {
  std::vector<uint32_t> ranks(size);
  std::iota(ranks.begin(), ranks.end(), 0);
  HcclTeamCreateDesc desc;
  auto ret = HcclTeamCreateDescInit(&desc);
  if (ret != HCCL_SUCCESS)
    return ret;
  desc.rankIds = ranks.data();
  desc.rankNum = size;
  desc.selfRankId = rank;
  desc.engine = COMM_ENGINE_AIV;
  desc.protocol = COMM_PROTOCOL_UB_CTP;
  desc.channelCnt = 1;
  // Session zero also provides shadow storage for AIN signal fetch-add.
  desc.requirement.barrierCount = 1;
  return HcclTeamCreate(comm, &desc, out);
}

namespace {
at::Tensor alloc(c10::IntArrayRef shape, at::ScalarType dtype,
                 c10::Device device) {
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1 &&
                  device.has_index(),
              "Expected an indexed NPU device");
  c10::DeviceGuard guard(device);
  const auto bytes = at::detail::computeStorageNbytesContiguous(
      shape, c10::elementSize(dtype));
  TORCH_CHECK(bytes > 0, "AIN windows require nonempty storage");
  void *ptr = nullptr;
  auto status = HcommMemAlloc(&ptr, bytes);
  TORCH_CHECK(status == 0, "HcommMemAlloc returned ", status);
  // Aliases of Window.local retain the allocation after deregistration.
  auto owner = std::shared_ptr<void>(ptr, [device](void *p) {
    c10::DeviceGuard guard(device);
    auto result = HcommMemFree(p);
    if (result != 0)
      TORCH_WARN("HcommMemFree returned ", result);
  });
  return at_npu::native::from_blob(
      ptr, shape, [owner](void *) {},
      at::TensorOptions().dtype(dtype).device(device), device);
}
} // namespace
TORCH_LIBRARY(tilelang_ain, m) {
  m.def("alloc(int[] shape, ScalarType dtype, Device device) -> Tensor",
        &alloc);
}

extern "C" int tl_ain_register(HcclComm comm, void *addr, uint64_t bytes,
                               uint32_t index, HcclMemHandle *out) {
  CommMem memory{};
  memory.type = COMM_MEM_TYPE_DEVICE;
  memory.addr = addr;
  memory.size = bytes;
  return HcclCommMemReg(comm, ("tl_window_" + std::to_string(index)).c_str(),
                        &memory, out);
}

// Register every local window with every peer channel in one collective
// acquisition.
extern "C" int tl_ain_channels(HcclComm comm, uint32_t size, uint32_t rank,
                               HcclMemHandle *handles, uint32_t count,
                               uint64_t *channels, uint64_t *addresses) {
  uint32_t *layers = nullptr;
  uint32_t num_layers = 0;
  auto ret = HcclRankGraphGetLayers(comm, &layers, &num_layers);
  if (ret != HCCL_SUCCESS)
    return ret;
  std::vector<HcclChannelDesc> descs(size - 1);
  ret = HcclChannelDescInit(descs.data(), descs.size());
  if (ret != HCCL_SUCCESS)
    return ret;
  for (uint32_t peer = 0; peer < size; ++peer) {
    if (peer == rank)
      continue;
    auto &desc = descs[peer - (peer > rank)];
    bool found = false;
    for (uint32_t l = 0; l < num_layers && !found; ++l) {
      CommLink *links = nullptr;
      uint32_t n = 0;
      ret = HcclRankGraphGetLinks(comm, layers[l], rank, peer, &links, &n);
      if (ret != HCCL_SUCCESS)
        return ret;
      for (uint32_t i = 0; i < n; ++i) {
        if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UB_CTP)
          continue;
        desc.remoteRank = peer;
        desc.localEndpoint = links[i].srcEndpointDesc;
        desc.remoteEndpoint = links[i].dstEndpointDesc;
        desc.channelProtocol = COMM_PROTOCOL_UB_CTP;
        desc.memHandles = handles;
        desc.memHandleNum = count;
        found = true;
        break;
      }
    }
    if (!found)
      return HCCL_E_NOT_SUPPORT;
  }
  std::vector<ChannelHandle> acquired(size - 1);
  ret = HcclChannelAcquire(comm, COMM_ENGINE_AIV, descs.data(), descs.size(),
                           acquired.data());
  if (ret != HCCL_SUCCESS)
    return ret;
  for (uint32_t peer = 0; peer < size; ++peer) {
    if (peer == rank)
      continue;
    auto channel = acquired[peer - (peer > rank)];
    channels[peer] = channel;
    CommMem *memories = nullptr;
    char **tags = nullptr;
    uint32_t n = 0;
    ret = HcclChannelGetRemoteMems(comm, channel, &n, &memories, &tags);
    if (ret != HCCL_SUCCESS)
      return ret;
    for (uint32_t j = 0; j < count; ++j) {
      const auto tag = "tl_window_" + std::to_string(j);
      bool found = false;
      for (uint32_t k = 0; tags && k < n; ++k) {
        if (tags[k] && tag == tags[k]) {
          addresses[peer * count + j] =
              reinterpret_cast<uint64_t>(memories[k].addr);
          found = true;
          break;
        }
      }
      if (!found)
        return HCCL_E_NOT_FOUND;
    }
  }
  return HCCL_SUCCESS;
}
