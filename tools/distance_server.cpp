#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/IPDistanceDispatcher.h>
#include <flatnav/distances/L2DistanceDispatcher.h>
#include <flatnav/util/OffloadMetrics.h>
#include <flatnav/util/CxlSimulation.h>

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
    _cxl_server = std::make_unique<CxlServer>(
        _config.shm_name, static_cast<uint32_t>(_config.thread_count),
        static_cast<uint32_t>(_dimension));
    _cxl_server->create();
    _cxl_server->setReady();

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
           !_cxl_server->isShutdownRequested()) {
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
    if (_cxl_server) {
      _cxl_server->destroy();
      _cxl_server.reset();
    }
  }

private:
  // --------------------------------------------------------------------------
  // Vector data loading — reads a .fvecs file (standard ANN benchmark format).
  // Format: each vector is stored as [int32 dim][dim × float32 values].
  // --------------------------------------------------------------------------
  void loadVectorData(const std::string &path) {
    std::cout << "[loadVectorData] Loading vectors from: " << path << std::endl;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open data file '" << path << "'" << std::endl;
      std::exit(1);
    }

    // Read dimension from the first vector's header.
    int32_t dimension = 0;
    file.read(reinterpret_cast<char *>(&dimension), sizeof(int32_t));
    if (!file.good() || dimension <= 0) {
      std::cerr << "Error: Failed to read dimension from '" << path << "'"
                << std::endl;
      std::exit(1);
    }
    _dimension = static_cast<size_t>(dimension);

    // Each vector record: 4 bytes (dim) + dim * 4 bytes (floats).
    size_t vec_size_bytes = sizeof(int32_t) + _dimension * sizeof(float);

    // Determine total number of vectors from file size.
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (file_size % vec_size_bytes != 0) {
      std::cerr << "Error: File size (" << file_size
                << ") is not a multiple of vector record size ("
                << vec_size_bytes << ")" << std::endl;
      std::exit(1);
    }
    _num_vectors = file_size / vec_size_bytes;

    std::cout << "[loadVectorData] File metadata:" << std::endl;
    std::cout << "  dimension     = " << _dimension << std::endl;
    std::cout << "  num_vectors   = " << _num_vectors << std::endl;
    std::cout << "  file_size     = " << file_size << " bytes" << std::endl;

    // Allocate contiguous float array for all vectors.
    size_t total_floats = _num_vectors * _dimension;
    _vectors = new float[total_floats];

    // Read each vector: skip the 4-byte dimension prefix, read the float data.
    for (size_t i = 0; i < _num_vectors; i++) {
      int32_t vec_dim = 0;
      file.read(reinterpret_cast<char *>(&vec_dim), sizeof(int32_t));

      if (static_cast<size_t>(vec_dim) != _dimension) {
        std::cerr << "Error: Vector " << i << " has dimension " << vec_dim
                  << " (expected " << _dimension << ")" << std::endl;
        std::exit(1);
      }

      file.read(reinterpret_cast<char *>(_vectors + (i * _dimension)),
                static_cast<std::streamsize>(_dimension * sizeof(float)));
    }

    if (!file.good()) {
      std::cerr << "Error: Failed to read all vector data from '" << path
                << "'" << std::endl;
      std::exit(1);
    }

    // Sanity check: print first vector's first few values.
    if (_num_vectors > 0 && _dimension > 0) {
      std::cout << "[loadVectorData] First vector (first "
                << std::min(_dimension, size_t(5)) << " values): [";
      for (size_t d = 0; d < std::min(_dimension, size_t(5)); d++) {
        if (d > 0) std::cout << ", ";
        std::cout << _vectors[d];
      }
      if (_dimension > 5) std::cout << ", ...";
      std::cout << "]" << std::endl;
    }

    std::cout << "[loadVectorData] Done. Loaded " << _num_vectors
              << " vectors (dim=" << _dimension << ")" << std::endl;
  }

  // --------------------------------------------------------------------------
  // Worker loop — one per slot
  // --------------------------------------------------------------------------
  void workerLoop(uint32_t slot_id) {
    // Temporary buffer for incoming request payloads.
    std::vector<char> payload_buf(Slot::REQUEST_BUFFER_SIZE);

    while (_running.load(std::memory_order_relaxed)) {
      CxlRequest header;
      if (!_cxl_server->pollRequest(slot_id, header, payload_buf.data())) {
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
  void processRequest(uint32_t slot_id, const CxlRequest &header,
                      const void *payload) {
    switch (header.type) {
    case CxlMessageType::CacheQuery:
      handleCacheQuery(slot_id, header, payload);
      break;
    case CxlMessageType::EvictQuery:
      handleEvictQuery(slot_id, header);
      break;
    case CxlMessageType::DistanceRequest:
      handleDistanceRequest(slot_id, header, payload);
      break;
    case CxlMessageType::StatsRequest:
      handleStatsRequest(slot_id);
      break;
    case CxlMessageType::StatsReset:
      handleStatsReset(slot_id);
      break;
    case CxlMessageType::Shutdown:
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
  void handleCacheQuery(uint32_t slot_id, const CxlRequest &header,
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
    CxlResponse resp{};
    resp.type = CxlMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _cxl_server->sendResponse(slot_id, resp, nullptr);
  }

  // --------------------------------------------------------------------------
  // EvictQuery: remove from cache (silent no-op if not present)
  // --------------------------------------------------------------------------
  void handleEvictQuery(uint32_t slot_id, const CxlRequest &header) {
    {
      std::unique_lock<std::shared_mutex> lock(_cache_mutex);
      _query_cache.erase(header.query_id);
    }

    // Send Ack.
    CxlResponse resp{};
    resp.type = CxlMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _cxl_server->sendResponse(slot_id, resp, nullptr);
  }

  // --------------------------------------------------------------------------
  // DistanceRequest: compute distances for a batch of node IDs
  // --------------------------------------------------------------------------
  void handleDistanceRequest(uint32_t slot_id, const CxlRequest &header,
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
    CxlResponse resp{};
    resp.type = CxlMessageType::DistanceResponse;
    resp.num_distances = count;
    resp.error_code = 0;
    _cxl_server->sendResponse(slot_id, resp, distances.data());
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
    CxlResponse resp{};
    resp.type = CxlMessageType::StatsResponse;
    resp.num_distances = 0;
    resp.error_code = 0;
    _cxl_server->sendResponse(slot_id, resp, &stats);
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

    CxlResponse resp{};
    resp.type = CxlMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _cxl_server->sendResponse(slot_id, resp, nullptr);
  }

  // --------------------------------------------------------------------------
  // Shutdown: set flag and ack
  // --------------------------------------------------------------------------
  void handleShutdown(uint32_t slot_id) {
    // Ack the shutdown request.
    CxlResponse resp{};
    resp.type = CxlMessageType::Ack;
    resp.num_distances = 0;
    resp.error_code = 0;
    _cxl_server->sendResponse(slot_id, resp, nullptr);

    // Signal main loop to stop.
    _running.store(false);
  }

  // --------------------------------------------------------------------------
  // Error response helper
  // --------------------------------------------------------------------------
  void sendError(uint32_t slot_id, uint32_t error_code,
                 const char *message) {
    CxlResponse resp{};
    resp.type = CxlMessageType::ErrorResponse;
    resp.error_code = error_code;

    size_t msg_len = std::strlen(message);
    if (msg_len > 255) {
      msg_len = 255;
    }
    // Reuse num_distances as payload size for error messages.
    resp.num_distances = static_cast<uint32_t>(msg_len);

    _cxl_server->sendResponse(slot_id, resp, message);
  }

  // --------------------------------------------------------------------------
  // Member data
  // --------------------------------------------------------------------------
  ServerConfig _config;
  std::unique_ptr<CxlServer> _cxl_server;

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
            << "  --data-file <path>   Path to .fvecs vector data file "
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
