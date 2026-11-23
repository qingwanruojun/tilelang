#include "plan.h"
#include <acl/acl_rt.h>
#include <hccl/hcomm_primitives.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    const auto rc = (call);                                                    \
    if (rc)                                                                    \
      return rc;                                                               \
  } while (false)

extern "C" int tl_aicpu_load(const char *config, aclrtBinHandle *binary,
                             aclrtFuncHandle *function) {
  aclrtBinaryLoadOption option{};
  option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
  option.value.cpuKernelMode = 0; // Registered standard AICPU kernel JSON.
  aclrtBinaryLoadOptions options{};
  options.numOpt = 1;
  options.options = &option;
  CHECK(aclrtBinaryLoadFromFile(config, &options, binary));
  auto rc = aclrtBinaryGetFunction(*binary, "TileLangAicpuTransfer", function);
  if (rc) {
    aclrtBinaryUnLoad(*binary);
    *binary = nullptr;
  }
  return rc;
}

extern "C" int tl_aicpu_unload(aclrtBinHandle binary) {
  return aclrtBinaryUnLoad(binary);
}

extern "C" int tl_aicpu_launch(aclrtFuncHandle function, const Arguments *args,
                               uint64_t hostQueue, uint64_t deviceOnHost,
                               aclrtStream stream) {
  aclrtArgsHandle handle;
  aclrtParamHandle param;
  CHECK(aclrtKernelArgsInit(function, &handle));
  CHECK(aclrtKernelArgsAppend(handle, const_cast<Arguments *>(args),
                              sizeof(*args), &param));
  CHECK(aclrtKernelArgsFinalize(handle));
  aclrtLaunchKernelAttr attr{};
  attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
  attr.value.timeout = 1836;
  aclrtLaunchKernelCfg cfg{};
  cfg.numAttrs = 1;
  cfg.attrs = &attr;
  CHECK(HcommThreadNotifyRecordOnThread(hostQueue, deviceOnHost, 0));
  CHECK(
      aclrtLaunchKernelWithConfig(function, 1, stream, &cfg, handle, nullptr));
  return HcommThreadNotifyWaitOnThread(hostQueue, 0, 1800);
}
