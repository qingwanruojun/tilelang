#pragma once
#include <cstdint>

// Every address here is rebound on the host for the submitting device.
struct Peer {
  uint64_t channel, window, capacity;
};
struct Operation {
  uint64_t kind, peer, source, destination, bytes;
};
struct Arguments {
  uint64_t queue, hostQueueOnDevice;
  uint32_t rank, size;
  const Peer *peers;
  const Operation *ops;
  uint64_t count;
  void *src, *dst;
  uint64_t srcBytes, dstBytes, outputOffset;
};
