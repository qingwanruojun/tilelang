#pragma once
#include <simt_api/asc_simt.h>
#if __has_include(<asccomm/asccomm_simt.h>)
#include <asccomm/asccomm_simt.h>
#include <jetty/asccomm_jetty_simt.h>

namespace tl_jetty {
using Warp = asc::simt::asccomm_coop_warp;
using SqInfo = asc::asccomm_jetty_info;
using PeerInfo = asc::simt::asccomm_peer_info;
using Peer = asc::simt::asccomm_peer;
using Jetty = asc::simt::asccomm_jetty;
using Segment = asc::simt::asccomm_sge;
constexpr unsigned stage_words = 2 * Jetty::num_wqebb_lanes;
#define TL_JETTY_WRITE asccomm_jetty_write_nbi
#define TL_JETTY_ADVANCE asccomm_jetty_advance_sq
#define TL_JETTY_PUBLISH asccomm_jetty_publish_sq
#define TL_JETTY_DRAIN asccomm_jetty_drain
#define TL_JETTY_BASE remote_base_addr
#define TL_JETTY_SIZE remote_buffer_size
} // namespace tl_jetty
#else
#include <hcomm/hcomm_simt.h>
#include <jetty/hcomm_jetty_simt.h>
namespace tl_jetty {
using Warp = AscendC::simt::HcommCoopWarp;
using SqInfo = AscendC::HcommJettyInfo;
using PeerInfo = AscendC::simt::jetty::HcommPeerInfo;
using Peer = AscendC::simt::jetty::HcommPeer;
using Jetty = AscendC::simt::jetty::HcommJetty;
using Segment = AscendC::simt::jetty::HcommSge;
constexpr unsigned stage_words = 2 * Jetty::kNumWQEBBLanes;
#define TL_JETTY_WRITE Write
#define TL_JETTY_ADVANCE AdvanceSq
#define TL_JETTY_PUBLISH PublishSq
#define TL_JETTY_DRAIN Drain
#define TL_JETTY_BASE remoteBaseAddr
#define TL_JETTY_SIZE remoteBufferSize
} // namespace tl_jetty
#endif

// A single owner drives this channel. All operations are converged warp calls.
namespace tl_jetty {

struct alignas(16) Queue {
  uint64_t reserved[2]; // UB offset zero is valid storage but native metadata
                        // must be non-null.
  SqInfo sq;
  PeerInfo remote;
  alignas(16) uint64_t stage[stage_words];
  uint64_t channel;
  int peer, error, pending;
};
static_assert(sizeof(Queue) <= 512);

__simt_callee__ inline void report(__gm__ int64_t *ctx, __ubuf__ Queue *q,
                                   int code) {
  if (threadIdx.x == 0) {
    q->error = code;
    ctx[6] = code;
  }
  Warp{}.sync();
}

__simt_callee__ inline bool uniform64(uint64_t v) {
  uint32_t lo = v, hi = v >> 32;
  return (asc_shfl(lo, 0, 32) == lo) & (asc_shfl(hi, 0, 32) == hi);
}

__simt_callee__ inline void init(__gm__ int64_t *ctx, __ubuf__ uint8_t *raw,
                                 int peer) {
  Warp warp;
  auto q = reinterpret_cast<__ubuf__ Queue *>(raw);
  if (threadIdx.x == 0) {
    q->error = 0;
    q->pending = 0;
    q->channel = 0;
    q->peer = peer;
  }
  warp.sync();
  if (peer != q->peer || peer < 0 || peer >= ctx[2] || peer == ctx[1] ||
      !ctx[5])
    asc_atomic_exch(&q->error, -5);
  warp.sync();
  if (q->error) {
    report(ctx, q, q->error);
    return;
  }
  auto table = reinterpret_cast<__gm__ uint64_t *>(ctx[5]);
  if (!table[peer]) {
    report(ctx, q, -5);
    return;
  }
  if (threadIdx.x == 0) {
    q->channel = table[peer];
    Peer p(&q->remote, table, peer);
    Jetty j(&q->sq, reinterpret_cast<__ubuf__ uint8_t *>(q->stage), table,
            peer);
  }
  warp.sync();
}

__simt_callee__ inline __gm__ const int64_t *window(__gm__ int64_t *ctx,
                                                    int peer, uint64_t base) {
  auto table =
      reinterpret_cast<__gm__ const int64_t *>(ctx[4]) + 3 * peer * ctx[3];
  for (int i = 0; i < ctx[3]; ++i)
    if (static_cast<uint64_t>(table[3 * i]) == base)
      return table + 3 * i;
  return nullptr;
}

template <int N, class S, class D>
__simt_callee__ inline void post_write(__gm__ int64_t *ctx,
                                       __ubuf__ uint8_t *raw, __gm__ S *src,
                                       int64_t so, __gm__ D *dst, int64_t off,
                                       int64_t bytes, bool sv, bool dv) {
  auto q = reinterpret_cast<__ubuf__ Queue *>(raw);
  if (q->error)
    return;
  Warp warp;
  const bool active = threadIdx.x < N;
  auto s =
      active ? window(ctx, q->peer, reinterpret_cast<uint64_t>(src)) : nullptr;
  auto d = window(ctx, q->peer, reinterpret_cast<uint64_t>(dst));
  bool ok = uniform64(bytes) & uniform64(off) &
            uniform64(reinterpret_cast<uint64_t>(dst));
  ok = ok && bytes >= 0 && bytes <= (1LL << 28) / N && d && dv && off >= 0 &&
       off <= d[2] - bytes * N;
  if (active)
    ok = ok && s && sv && so >= 0 && so <= s[2] - bytes;
  uint64_t remote = 0;
  if (ok && bytes) {
    remote = static_cast<uint64_t>(d[1]) + off;
    // The peer descriptor carries the access token of its registered remote
    // region.
    uint64_t base = q->remote.TL_JETTY_BASE, capacity = q->remote.TL_JETTY_SIZE;
    ok = remote >= base && remote - base <= capacity &&
         uint64_t(bytes * N) <= capacity - (remote - base);
  }
  if (!ok)
    asc_atomic_exch(&q->error, -6);
  warp.sync();
  if (q->error) {
    report(ctx, q, q->error);
    return;
  }
  if (!bytes)
    return;
  Segment segment{};
  if (active) {
    segment.addr = static_cast<uint64_t>(s[0]) + so;
    segment.len = static_cast<int>(bytes);
  }
  Peer peer(&q->remote, q->channel);
  Jetty jetty(&q->sq, reinterpret_cast<__ubuf__ uint8_t *>(q->stage),
              q->channel);
  auto rc = jetty.template TL_JETTY_WRITE<N, false>(
      warp, peer, reinterpret_cast<__gm__ void *>(remote), segment);
  if (rc) {
    report(ctx, q, rc);
    return;
  }
  if (threadIdx.x == 0)
    q->pending = 1;
  warp.sync();
}
__simt_callee__ inline void publish(__gm__ int64_t *, __ubuf__ uint8_t *raw) {
  auto q = reinterpret_cast<__ubuf__ Queue *>(raw);
  if (q->error || !q->pending)
    return;
  Warp warp;
  Jetty j(&q->sq, reinterpret_cast<__ubuf__ uint8_t *>(q->stage), q->channel);
  j.TL_JETTY_ADVANCE(warp);
  j.TL_JETTY_PUBLISH(warp);
  if (threadIdx.x == 0)
    q->pending = 0;
  warp.sync();
}
template <int64_t timeout>
__simt_callee__ inline void drain(__gm__ int64_t *ctx, __ubuf__ uint8_t *raw) {
  auto q = reinterpret_cast<__ubuf__ Queue *>(raw);
  if (q->error)
    return;
  if (q->pending) {
    report(ctx, q, -7);
    return;
  }
  Jetty j(&q->sq, reinterpret_cast<__ubuf__ uint8_t *>(q->stage), q->channel);
  int rc = j.template TL_JETTY_DRAIN<timeout>(Warp{});
  if (rc)
    report(ctx, q, rc);
}
} // namespace tl_jetty

#undef TL_JETTY_WRITE
#undef TL_JETTY_ADVANCE
#undef TL_JETTY_PUBLISH
#undef TL_JETTY_DRAIN
#undef TL_JETTY_BASE
#undef TL_JETTY_SIZE
