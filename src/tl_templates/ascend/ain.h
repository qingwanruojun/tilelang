#pragma once
#include <comm_api/aicore/ain/ain.h>

// Shared with Team.prepare: native team, rank, size, window count, window
// table, optional peer channel table, sticky device status. Each window entry
// holds the local GM base, symmetric handle (or remote GM base), and capacity.
namespace tl_ain {
using namespace AscendC;
__aicore__ inline void fail(__gm__ int64_t *ctx, int32_t status) {
  st_dev(static_cast<int64_t>(status), ctx + 6, 0);
}
__aicore__ inline __gm__ const int64_t *window(__gm__ const int64_t *ctx,
                                               int peer, uint64_t base) {
  auto table = reinterpret_cast<__gm__ const int64_t *>(ctx[4]);
  if (ctx[5])
    table += 3 * peer * ctx[3];
  for (int i = 0; i < ctx[3]; ++i) {
    if (static_cast<uint64_t>(table[3 * i]) == base)
      return table + 3 * i;
  }
  return nullptr;
}
template <bool pull, bool publish = true, class S, class D>
__aicore__ inline void transfer(__gm__ int64_t *ctx, int peer, __gm__ S *src,
                                int64_t src_offset, __gm__ D *dst,
                                int64_t dst_offset, int64_t bytes, bool valid,
                                __ubuf__ uint8_t *scratch) {
  if (!valid) {
    fail(ctx, -2);
    return;
  }
  if (bytes == 0)
    return;
  if (peer < 0 || peer >= ctx[2] || peer == ctx[1]) {
    fail(ctx, -1);
    return;
  }
  auto s = window(ctx, peer, reinterpret_cast<uint64_t>(src));
  auto d = window(ctx, peer, reinterpret_cast<uint64_t>(dst));
  if (!s || !d || bytes < 0 || bytes > (1LL << 28) || src_offset < 0 ||
      dst_offset < 0 || src_offset > s[2] - bytes ||
      dst_offset > d[2] - bytes) {
    fail(ctx, -2);
    return;
  }
  if (ctx[5]) {
    const auto channel =
        reinterpret_cast<__gm__ const uint64_t *>(ctx[5])[peer];
    auto source =
        reinterpret_cast<__gm__ uint8_t *>(s[pull ? 1 : 0]) + src_offset;
    auto target =
        reinterpret_cast<__gm__ uint8_t *>(d[pull ? 0 : 1]) + dst_offset;
    Hcomm<> comm;
    int status = comm.Init(scratch, 512);
    if (status == 0) {
      if constexpr (pull)
        status = comm.ReadNbi<publish>(channel, target, source, bytes);
      else
        status = comm.WriteNbi<publish>(channel, target, source, bytes);
    }
    if (status)
      fail(ctx, status);
    return;
  }
  Ain<> ain(0);
  auto team = reinterpret_cast<AscendC::HcommTeamHandle>(ctx[0]);
  auto sw = reinterpret_cast<HcclCommSymWindow>(s[1]);
  auto dw = reinterpret_cast<HcclCommSymWindow>(d[1]);
  AinDescriptorUbuf ub{scratch, 512, 0};
  constexpr auto flags = publish ? AIN_COMMIT_IMMED : AIN_COMMIT_DELAYED;
  if constexpr (pull)
    ain.Get<AinDescriptorUbuf, flags>(team, peer, dw, dst_offset, sw,
                                      src_offset, bytes, ub);
  else
    ain.Put<AinRemoteNone, AinDescriptorUbuf, flags>(
        team, peer, dw, dst_offset, sw, src_offset, bytes, AinRemoteNone{}, ub);
}
template <bool publish>
__aicore__ inline void finish_peer(__gm__ int64_t *ctx, int peer,
                                   __ubuf__ uint8_t *scratch) {
  if (peer < 0 || peer >= ctx[2] || peer == ctx[1]) {
    fail(ctx, -1);
    return;
  }
  ChannelHandle channel;
  if (ctx[5]) {
    channel = reinterpret_cast<__gm__ const uint64_t *>(ctx[5])[peer];
  } else {
    Ain<> ain(0);
    ain.FlushAsync(reinterpret_cast<AscendC::HcommTeamHandle>(ctx[0]), peer,
                   &channel);
  }
  // CANN's Ain::Wait discards the result of Drain. Preserve it for
  // Team.barrier.
  Hcomm<> comm;
  int status = comm.Init(scratch, 512);
  if (status == 0) {
    if constexpr (publish)
      status = comm.Commit(channel);
    else
      status = comm.Drain(channel);
  }
  if (status)
    fail(ctx, status);
}

__aicore__ inline void wait_peer(__gm__ int64_t *ctx, int peer,
                                 __ubuf__ uint8_t *scratch) {
  finish_peer<false>(ctx, peer, scratch);
}
__aicore__ inline void commit_peer(__gm__ int64_t *ctx, int peer,
                                   __ubuf__ uint8_t *scratch) {
  finish_peer<true>(ctx, peer, scratch);
}
__aicore__ inline HcclCommSymWindow
signal_window(__gm__ int64_t *ctx, uint64_t base, int64_t index, bool valid) {
  if (!ctx[0] || ctx[5]) {
    fail(ctx, -3);
    return nullptr;
  }
  auto entry = window(ctx, 0, base);
  if (!valid || !entry || index < 0 || index >= entry[2] / 8) {
    fail(ctx, -2);
    return nullptr;
  }
  return reinterpret_cast<HcclCommSymWindow>(entry[1]);
}
template <typename T>
__aicore__ inline void signal_add(__gm__ int64_t *ctx, int peer, __gm__ T *base,
                                  int64_t index, bool valid, uint64_t delta,
                                  __ubuf__ uint8_t *scratch) {
  if (peer < 0 || peer >= ctx[2] || peer == ctx[1]) {
    fail(ctx, -1);
    return;
  }
  auto win = signal_window(ctx, reinterpret_cast<uint64_t>(base), index, valid);
  if (!win)
    return;
  Ain<> ain(0);
  ain.Signal(reinterpret_cast<AscendC::HcommTeamHandle>(ctx[0]), peer,
             AinSignalAdd{win, static_cast<size_t>(index) * 8, delta},
             AinDescriptorUbuf{scratch, 512, 0});
}
template <typename T>
__aicore__ inline uint64_t read_signal(__gm__ int64_t *ctx, __gm__ T *base,
                                       int64_t index, bool valid) {
  auto win = signal_window(ctx, reinterpret_cast<uint64_t>(base), index, valid);
  if (!win)
    return 0;
  Ain<> ain(0);
  return ain.ReadSignal(reinterpret_cast<AscendC::HcommTeamHandle>(ctx[0]), win,
                        index * 8);
}
template <typename T, typename Threshold>
__aicore__ inline void wait_signal(__gm__ int64_t *ctx, __gm__ T *base,
                                   int64_t index, bool valid, Threshold least,
                                   int64_t polls) {
  if (least < 0) {
    fail(ctx, -2);
    return;
  }
  auto win = signal_window(ctx, reinterpret_cast<uint64_t>(base), index, valid);
  if (!win)
    return;
  Ain<> ain(0);
  for (int64_t i = 0; i < polls; ++i)
    if (ain.ReadSignal(reinterpret_cast<AscendC::HcommTeamHandle>(ctx[0]), win,
                       index * 8) >= static_cast<uint64_t>(least))
      return;
  fail(ctx, -4);
}
__aicore__ inline void team_sync(__gm__ int64_t *ctx, int64_t polls,
                                 __ubuf__ uint8_t *scratch) {
  if (!ctx[0] || ctx[5]) {
    fail(ctx, -3);
    return;
  }
  Ain<> ain(0);
  AinBarrierSession<> session(
      &ain, reinterpret_cast<AscendC::HcommTeamHandle>(ctx[0]), 0);
  if (session.Sync(AIN_MEMORY_ORDER_RELAX, polls,
                   AinDescriptorUbuf{scratch, 512, 0}) != 0)
    fail(ctx, -4);
}
} // namespace tl_ain
