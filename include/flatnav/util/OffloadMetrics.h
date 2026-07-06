#pragma once

#include <cstdint>

namespace flatnav::util {

/// Metrics collected by the Index Process after a search using CXL offload
/// distance. Captures end-to-end timing and IPC cost breakdown.
struct OffloadSearchMetrics {
  /// Total wall-clock search time in nanoseconds (monotonic clock).
  uint64_t total_search_time_ns;

  /// Number of distance requests issued to the Distance Server during
  /// this search.
  uint64_t num_distance_requests;

  /// Cumulative round-trip latency in nanoseconds for all distance
  /// requests made during this search.
  uint64_t cumulative_round_trip_ns;
};

/// Aggregated statistics returned by the Distance Server in response
/// to a StatsRequest message.
struct ServerStats {
  /// Arithmetic mean of per-request processing latency in nanoseconds.
  uint64_t mean_latency_ns;

  /// 50th percentile (median) per-request latency in nanoseconds.
  uint64_t p50_latency_ns;

  /// 99th percentile per-request latency in nanoseconds.
  uint64_t p99_latency_ns;

  /// Total number of requests processed since the last statistics reset.
  uint64_t total_requests;
};

}  // namespace flatnav::util
