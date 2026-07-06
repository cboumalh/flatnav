#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/IPDistanceDispatcher.h>
#include <flatnav/distances/L2DistanceDispatcher.h>
#include <flatnav/util/OffloadMetrics.h>
#include <flatnav/util/SharedMemoryIpc.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <signal.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace flatnav::util;
using namespace flatnav::distances;

// ----------------------------------------------------------------------------
// Global shutdown flag (async-signal-safe)
// ----------------------------------------------------------------------------
static std::atomic<bool> g_running{true};

static void sigterm_handler(int /*sig*/) { g_running.store(false); }

// ----------------------------------------------------------------------------
// Server Configuration
// ----------------------------------------------------------------------------
struct ServerConfig {
  int numa_node = 0;
  int thread_count = 1;
  MetricType metric = MetricType::L2;
  std::string data_file_path;
  std::string shm_name = "/flatnav_cxl";
};

// ----------------------------------------------------------------------------
// Distance Server
// ----------------------------------------------------------------------------
class DistanceServer {
public:
  explicit DistanceServer(const ServerConfig &config)
      : _config(config), _vectors(nullptr), _num_vectors(0), _dimension(0),
        _running(true) {}

  ~DistanceServer() {
    shutdown();
    if (_vectors != nullptr) {
      delete[] _vectors;
      _vectors = nullptr;
    }
  }

  /// Load vector data, create IPC region, spawn workers, and run the main loop.
  void run() {
    loadVectorData(_config.data_file_path);

    // Create the IPC shared memory region with one slot per thread.
    _ipc_server = std::make_unique<CxlServer>(
        _config.shm_name, static_cast<uint32_t>(_config.thread_count),
        static_cast<uint32_t>(_dimension));
    _ipc_server->create();
    _ipc_server->setReady();

    std::cout << "READY: Distance Server listening on " << _config.shm_name
              << " with " << _config.thread_count << " threads, "
              << _num_vectors << " vectors of dimension " << _dimension
              << std::endl;

    // Spawn worker threads — each polls its own slot.
    for (int i = 0; i < _config.thread_count; i++) {
      _workers.emplace_back(&DistanceServer::workerLoop, this,
                            static_cast<uint32_t>(i));
    }

    // Main loop: wait for shutdown signal.
    while (_running.load() && g_running.load() &&
           !_ipc_server->isShutdownRequested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    shutdown();
  }

  /// Initiate graceful shutdown: stop workers, drain, and clean up.
  void shutdown() {
    if (!_running.exchange(false)) {
      return; // Already shut down.
    }

    // Wait for workers to finish (up to 5 seconds).
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (auto &worker : _workers) {
      if (worker.joinable()) {
        auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining > std::chrono::milliseconds(0)) {
          // We can't join with a timeout in standard C++, so just join.
          worker.join();
        } else {
          worker.detach();
        }
      }
    }
    _workers.clear();

    // Destroy the IPC region (unmaps + unlinks shared memory).
    if (_ipc_server) {
      _ipc_server->destroy();
      _ipc_server.reset();
    }
  }

private:
  // --------------------------------------------------------------------------
  // Vector data loading
  // --------------------------------------------------------------------------
  void loadVectorData(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open data file '" << path << "'" << std::endl;
      std::exit(1);
    }

    uint32_t num_vectors = 0;
    uint32_t dimension = 0;
    file.read(reinterpret_cast<char *>(&num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&dimension), sizeof(uint32_t));

    if (!file.good()) {
      std::cerr << "Error: Failed to read header from '" << path << "'"
                << std::endl;
      std::exit(1);
    }

    _num_vectors = static_cast<size_t>(num_vectors);
    _dimension = static_cast<size_t>(dimension);

    size_t total_floats = _num_vectors * _dimension;
    _vectors = new float[total_floats];

    file.read(reinterpret_cast<char *>(_vectors),
              static_cast<std::streamsize>(total_floats * sizeof(float)));

    if (!file.good()) {
      std::cerr << "Error: Failed to read vector data from '" << path
                << "' (expected " << total_floats << " floats)" << std::endl;
      std::exit(1);
    }
  }

  // --------------------------------------------------------------------------
  // Worker loop — one per slot
  // --------------------------------------------------------------------------
  void workerLoop(uint32_t slot_id) {
    // Temporary buffer for incoming request payloads.
    std::vector<char> payload_buf(IpcSlot::REQUEST_BUFFER_SIZE);

    while (_running.load(std::memory_order_relaxed)) {
      IpcRequestHeader header;
      if (!_ipc_server->pollRequest(slot_id, header, payload_buf.data())) {
        // No request available — brief pause to avoid busy-spin.
        std::this_thread::yield();
        continue;
      }

      auto start_time = std::chrono::steady_clock::now();

      processRequest(slot_id, header, payload_buf.data());

      auto end_time = std::chrono::steady_clock::now();
      uint64_t latency_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                              start_time)
              .count());

      // Record latency for stats.
      {
        std::lock_guard<std::mutex> lock(_metrics.mutex);
        _metrics.request_latencies_ns.push_back(latency_ns);
      }
      _metrics.total_requests.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // --------------------------------------------------------------------------
  // Request dispatch
  // --------------------------------------------------------------------------
  void processRequest(uint32_t slot_id, const IpcRequestHeader &header,
                      const void *payload) {
    switch (header.type) {
    case IpcMessageType::CacheQuery:
      handleCacheQuery(slot_id, header, payload);
      break;
    case IpcMessageType::EvictQuery:
      handleEvictQuery(slot_id, header);
      break;
    case IpcMessageType::DistanceRequest:
      handleDistanceRequest(slot_id, header, payload);
      break;
    case IpcMessageType::StatsRequest:
      handleStatsRequest(slot_id);
      break;
    case IpcMessageType::StatsReset:
      handleStatsReset(slot_id);
      break;
    case IpcMessageType::Shutdown:
      handleShutdown(slot_id);
      break;
    default:
      sendError(slot_id, 6, "Unknown message type");
      break;
    }
  }

  // --------------------------------------------------------------------------
  // CacheQuery: store query vector in cache
  // --------------------------------------------------------------------------
  void handleCacheQuery(uint32_t slot_id, const IpcRequestHeader &header,
                        const void *payload) {
    // Validate dimension.
    size_t expected_size = _dimension * sizeof(float);
    if (header.payload_size != static_cast<uint32_t>(expected_size)) {
      sendError(slot_id, 2, "Dimension mismatch");
      return;
    }

    const float *vec = static_cast<const float *>(payload);
    std::vector<float> query_vec(vec, vec + _dimension);

    {
      std::unique_lock<std::shared_mutex> lock(_cache_mutex);
      _query_cache[header.query_id] = std::move(query_vec);
    }

    // Send Ack.
    IpcResponseHeader resp{};
    resp.type = IpcMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _ipc_server->sendResponse(slot_id, resp, nullptr);
  }

  // --------------------------------------------------------------------------
  // EvictQuery: remove from cache (silent no-op if not present)
  // --------------------------------------------------------------------------
  void handleEvictQuery(uint32_t slot_id, const IpcRequestHeader &header) {
    {
      std::unique_lock<std::shared_mutex> lock(_cache_mutex);
      _query_cache.erase(header.query_id);
    }

    // Send Ack.
    IpcResponseHeader resp{};
    resp.type = IpcMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _ipc_server->sendResponse(slot_id, resp, nullptr);
  }

  // --------------------------------------------------------------------------
  // DistanceRequest: compute distances for a batch of node IDs
  // --------------------------------------------------------------------------
  void handleDistanceRequest(uint32_t slot_id, const IpcRequestHeader &header,
                             const void *payload) {
    uint32_t count = header.num_node_ids;

    // Validate batch size.
    if (count > 1024) {
      sendError(slot_id, 3, "Batch size exceeds limit (>1024)");
      return;
    }

    // Look up the cached query vector.
    const float *query_vec = nullptr;
    {
      std::shared_lock<std::shared_mutex> lock(_cache_mutex);
      auto it = _query_cache.find(header.query_id);
      if (it == _query_cache.end()) {
        sendError(slot_id, 1, "Unknown query ID");
        return;
      }
      query_vec = it->second.data();
    }

    const uint32_t *node_ids = static_cast<const uint32_t *>(payload);

    // Validate all node IDs are in range.
    for (uint32_t i = 0; i < count; i++) {
      if (node_ids[i] >= _num_vectors) {
        sendError(slot_id, 4, "Node ID out of range");
        return;
      }
    }

    // Compute distances.
    std::vector<float> distances(count);
    for (uint32_t i = 0; i < count; i++) {
      const float *node_vec = _vectors + (static_cast<size_t>(node_ids[i]) * _dimension);
      distances[i] = computeDistance(query_vec, node_vec);
    }

    // Send DistanceResponse.
    IpcResponseHeader resp{};
    resp.type = IpcMessageType::DistanceResponse;
    resp.num_distances = count;
    resp.error_code = 0;
    _ipc_server->sendResponse(slot_id, resp, distances.data());
  }

  // --------------------------------------------------------------------------
  // Distance computation — uses the same dispatchers as the local distance
  // classes for bitwise-identical results.
  // --------------------------------------------------------------------------
  float computeDistance(const float *query, const float *node) const {
    if (_config.metric == MetricType::L2) {
      return L2DistanceDispatcher::dispatch(query, node, _dimension);
    } else {
      return IPDistanceDispatcher::dispatch(query, node, _dimension);
    }
  }

  // --------------------------------------------------------------------------
  // StatsRequest: compute and return statistics
  // --------------------------------------------------------------------------
  void handleStatsRequest(uint32_t slot_id) {
    ServerStats stats{};

    {
      std::lock_guard<std::mutex> lock(_metrics.mutex);
      size_t n = _metrics.request_latencies_ns.size();
      if (n > 0) {
        // Mean
        uint64_t sum = 0;
        for (auto lat : _metrics.request_latencies_ns) {
          sum += lat;
        }
        stats.mean_latency_ns = sum / n;

        // Sort for percentiles.
        std::vector<uint64_t> sorted = _metrics.request_latencies_ns;
        std::sort(sorted.begin(), sorted.end());

        // p50
        stats.p50_latency_ns = sorted[n / 2];

        // p99
        size_t p99_idx = static_cast<size_t>(
            static_cast<double>(n - 1) * 0.99);
        stats.p99_latency_ns = sorted[p99_idx];
      }
      stats.total_requests = _metrics.total_requests.load();
    }

    // Send StatsResponse.
    IpcResponseHeader resp{};
    resp.type = IpcMessageType::StatsResponse;
    resp.num_distances = 0;
    resp.error_code = 0;
    _ipc_server->sendResponse(slot_id, resp, &stats);
  }

  // --------------------------------------------------------------------------
  // StatsReset: clear accumulated metrics
  // --------------------------------------------------------------------------
  void handleStatsReset(uint32_t slot_id) {
    {
      std::lock_guard<std::mutex> lock(_metrics.mutex);
      _metrics.request_latencies_ns.clear();
      _metrics.total_requests.store(0);
    }

    IpcResponseHeader resp{};
    resp.type = IpcMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _ipc_server->sendResponse(slot_id, resp, nullptr);
  }

  // --------------------------------------------------------------------------
  // Shutdown: set flag and ack
  // --------------------------------------------------------------------------
  void handleShutdown(uint32_t slot_id) {
    // Ack the shutdown request.
    IpcResponseHeader resp{};
    resp.type = IpcMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _ipc_server->sendResponse(slot_id, resp, nullptr);

    // Signal main loop to stop.
    _running.store(false);
  }

  // --------------------------------------------------------------------------
  // Error response helper
  // --------------------------------------------------------------------------
  void sendError(uint32_t slot_id, uint32_t error_code,
                 const char *message) {
    IpcResponseHeader resp{};
    resp.type = IpcMessageType::ErrorResponse;
    resp.error_code = error_code;

    size_t msg_len = std::strlen(message);
    if (msg_len > 255) {
      msg_len = 255;
    }
    // Reuse num_distances as payload size for error messages.
    resp.num_distances = static_cast<uint32_t>(msg_len);

    _ipc_server->sendResponse(slot_id, resp, message);
  }

  // --------------------------------------------------------------------------
  // Member data
  // --------------------------------------------------------------------------
  ServerConfig _config;
  std::unique_ptr<CxlServer> _ipc_server;

  // Vector storage (read-only after load).
  float *_vectors;
  size_t _num_vectors;
  size_t _dimension;

  // Query cache (reader-writer lock protected).
  std::shared_mutex _cache_mutex;
  std::unordered_map<uint64_t, std::vector<float>> _query_cache;

  // Thread pool.
  std::vector<std::thread> _workers;
  std::atomic<bool> _running;

  // Metrics.
  struct Metrics {
    std::mutex mutex;
    std::vector<uint64_t> request_latencies_ns;
    std::atomic<uint64_t> total_requests{0};
  } _metrics;
};

// ----------------------------------------------------------------------------
// Argument parsing
// ----------------------------------------------------------------------------
static void printUsage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " --data-file <path> [--numa-node <id>] [--threads <N>] "
               "[--metric l2|ip] [--shm-name <name>]\n"
            << "\n"
            << "Options:\n"
            << "  --data-file <path>   Path to binary vector data file "
               "(required)\n"
            << "  --numa-node <id>     NUMA node to bind to (default: 0)\n"
            << "  --threads <N>        Number of worker threads, 1-256 "
               "(default: 1)\n"
            << "  --metric l2|ip       Distance metric (default: l2)\n"
            << "  --shm-name <name>    Shared memory segment name (default: "
               "/flatnav_cxl)\n";
}

static ServerConfig parseArgs(int argc, char *argv[]) {
  ServerConfig config;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--numa-node" && i + 1 < argc) {
      config.numa_node = std::atoi(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      config.thread_count = std::atoi(argv[++i]);
    } else if (arg == "--metric" && i + 1 < argc) {
      std::string m = argv[++i];
      if (m == "l2") {
        config.metric = MetricType::L2;
      } else if (m == "ip") {
        config.metric = MetricType::IP;
      } else {
        std::cerr << "Error: Unknown metric '" << m
                  << "'. Valid values: l2, ip\n";
        std::exit(1);
      }
    } else if (arg == "--data-file" && i + 1 < argc) {
      config.data_file_path = argv[++i];
    } else if (arg == "--shm-name" && i + 1 < argc) {
      config.shm_name = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      printUsage(argv[0]);
      std::exit(1);
    }
  }

  // Validate required arguments.
  if (config.data_file_path.empty()) {
    std::cerr << "Error: --data-file is required\n";
    printUsage(argv[0]);
    std::exit(1);
  }

  // Validate thread count.
  if (config.thread_count < 1 || config.thread_count > 256) {
    std::cerr << "Error: Thread count must be in range [1, 256], got "
              << config.thread_count << "\n";
    std::exit(1);
  }

  return config;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  ServerConfig config = parseArgs(argc, argv);

  // Install SIGTERM handler (async-signal-safe: only sets atomic flag).
  struct sigaction sa {};
  sa.sa_handler = sigterm_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGTERM, &sa, nullptr);

  DistanceServer server(config);
  server.run();

  return 0;
}
