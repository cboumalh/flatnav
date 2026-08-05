#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/VisitedSetPool.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/NumaAllocation.h>
#include <flatnav/util/CxlSimulation.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <optional>

using flatnav::distances::DistanceInterface;
using flatnav::util::VisitedSet;
using flatnav::util::VisitedSetPool;
using flatnav::util::DataType;
using flatnav::util::kNoNumaNode;

#ifdef FLATNAV_PROFILE_PQ
#pragma message("Index.h: FLATNAV_PROFILE_PQ is defined")
#else
#pragma message("Index.h: FLATNAV_PROFILE_PQ is NOT defined")
#endif

#ifdef FLATNAV_CXL_OFFLOAD
#pragma message("Index.h: FLATNAV_CXL_OFFLOAD is defined")
#else
#pragma message("Index.h: FLATNAV_CXL_OFFLOAD is NOT defined")
#endif

#ifdef FLATNAV_PROFILE_VISITS
#pragma message("Index.h: FLATNAV_PROFILE_VISITS is defined")
#else
#pragma message("Index.h: FLATNAV_PROFILE_VISITS is NOT defined")
#endif

#ifdef FLATNAV_PROFILE_NOINLINE
#pragma message("Index.h: FLATNAV_PROFILE_NOINLINE is defined")
#else
#pragma message("Index.h: FLATNAV_PROFILE_NOINLINE is NOT defined")
#endif

#ifdef FLATNAV_DISABLE_PREFETCH
#pragma message("Index.h: FLATNAV_DISABLE_PREFETCH is defined")
#else
#pragma message("Index.h: FLATNAV_DISABLE_PREFETCH is NOT defined")
#endif

#ifdef USE_SSE
#pragma message("Index.h: USE_SSE is defined")
#else
#pragma message("Index.h: USE_SSE is NOT defined")
#endif

namespace flatnav {

#ifdef FLATNAV_PROFILE_PQ
// Candidate-PQ residency instrumentation (tier-prefetch study). A node's prefetch
// lead time = its PQ residency = (pop_step - discovery_step). Per-thread maps track
// discovery step + parent residency; global atomic histograms aggregate across queries.
inline constexpr int kPQHistCap = 4096;
inline std::atomic<uint64_t> g_pq_residency_hist[kPQHistCap];     // residency of every expanded node
inline std::atomic<uint64_t> g_leapfrog_parent_hist[kPQHistCap];  // parent residency of residency==1 nodes
// Per-search-step (pop/hop index) buckets, for the early-vs-late predictability split.
inline constexpr int kPQStepCap = 1024;
inline std::atomic<uint64_t> g_step_count[kPQStepCap];   // # expansions at this step
inline std::atomic<uint64_t> g_step_leap[kPQStepCap];    // of those, residency==1 (leapfroggers)
inline std::atomic<uint64_t> g_step_sumres[kPQStepCap];  // sum of residency (for mean)
inline thread_local std::unordered_map<uint32_t, uint32_t> tl_disc;        // node -> discovery step
inline thread_local std::unordered_map<uint32_t, uint32_t> tl_parent_res;  // node -> parent's residency
inline thread_local uint32_t tl_step;          // pops done so far this query
inline thread_local uint32_t tl_cur_residency; // residency of the node currently being expanded
// "top-K candidate prefetch" policy: each step, just after the current node is popped (and BEFORE
// its neighbors are discovered), we'd prefetch the K closest pending candidates. Because the snapshot
// is taken pre-expansion, this step's leapfroggers (not yet born) are excluded from every K. A node
// is "prefetched" once, at the first step it enters the top-K. success = eventually popped; waste =
// never popped. Buckets keyed by first-prefetch step; lead = pop_step - first_prefetch_step.
inline constexpr int kPFnumK = 4;
inline constexpr int kPFK[kPFnumK] = {1, 5, 10, 50};
inline std::atomic<uint64_t> g_pf_prefetched[kPFnumK][kPQStepCap];
inline std::atomic<uint64_t> g_pf_success[kPFnumK][kPQStepCap];
inline std::atomic<uint64_t> g_pf_leadsum[kPFnumK];
struct PFEntry {
  uint16_t fe[kPFnumK] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF}; // first step entered top-K (0xFFFF=never)
  uint16_t pop_step = 0;
  bool popped = false;
};
inline thread_local std::unordered_map<uint32_t, PFEntry> tl_pf;
// Pre-expansion K=1 NEXT-STEP precision: at each step we prefetch the 2nd-min (next candidate);
// did exactly that node get popped at the immediately next step? (= used with 1-step lead, the
// strictest "is the next vector known" test). Bucketed by the prefetch step.
inline std::atomic<uint64_t> g_pf1_next_total[kPQStepCap];
inline std::atomic<uint64_t> g_pf1_next_hit[kPQStepCap];
inline thread_local uint32_t tl_pf1_node;   // 2nd-min prefetched last step
inline thread_local uint32_t tl_pf1_step;   // the step it was prefetched at
inline thread_local bool tl_pf1_valid;
#endif

// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
template <typename dist_t, typename label_t>
class Index {
  typedef std::pair<float, label_t> dist_label_t;
  // internal node numbering scheme. We might need to change this to uint64_t
  typedef uint32_t node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;

  // NOTE: by default this is a max-heap. We could make this a min-heap
  // by using std::greater, but we want to use the queue as both a max-heap and
  // min-heap depending on the context.

  struct CompareByFirst {
    constexpr bool operator()(dist_node_t const& a, dist_node_t const& b) const noexcept{
      return a.first < b.first;
    }
  };

  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>, CompareByFirst> PriorityQueue;

  // NUMA-aware storage. The data and the graph are kept in two separate,
  // contiguous (structure-of-arrays) allocations so each can be bound to its
  // own NUMA node.
  //   Vectors block: one [data] entry per node,         stride _data_size_bytes.
  //   Graph block:   one [M links][label] entry per node, stride
  //                  _graph_node_size_bytes.
  char* _vectors_memory = nullptr;
  char* _graph_memory = nullptr;
  int _vectors_numa_node = kNoNumaNode;
  int _graph_numa_node = kNoNumaNode;

  size_t _M;
  // size of one data point (does not support variable-size data, strings)
  size_t _data_size_bytes;
  // Logical size of a node: [data] + [M links] + [label]. The data lives in
  // _vectors_memory and the links+label live in _graph_memory, so this is kept
  // only for reporting/serialization metadata. It satisfies
  // _node_size_bytes == _data_size_bytes + _graph_node_size_bytes.
  size_t _node_size_bytes;
  // Size of one entry in _graph_memory: [M links][label].
  size_t _graph_node_size_bytes;
  size_t _max_node_count;  // Determines size of internal pre-allocated memory
  size_t _cur_num_nodes;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
  std::mutex _index_data_guard;

  uint32_t _num_threads;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  VisitedSetPool* _visited_set_pool;
  std::vector<std::mutex> _node_links_mutexes;

  bool _collect_stats = false;
  DataType _data_type;

#ifdef FLATNAV_CXL_OFFLOAD
  // Name of the distance server's shared-memory segment. Each OS thread that
  // performs search gets its own CxlClient, connected lazily (see
  // cxlClient()) rather than eagerly sharing a single client across threads.
  std::string _cxl_shm_name;
  std::atomic<uint32_t> _cxl_next_slot{0};
  std::atomic<uint64_t> _cxl_query_id_counter{0};

  // Path + byte offset of the vectors region within the serialized index
  // file, captured by loadIndex() when vectors are skipped (not resident in
  // _vectors_memory). Lets cacheHubVectors() fetch individual hub vectors
  // on demand without keeping the whole vectors block resident on the host.
  std::string _index_file_path;
  std::streamoff _vectors_file_offset = 0;
#endif

  // NOTE: These metrics are meaningful the most with single-threaded search.
  // With multi-threaded search, for instance, the number of distance computations will 
  // accumulate across queries, which means at the end of the batched search, the number 
  // you get is the cumulative sum of all distance computations across all queries.
  // Maybe that's what you want, but it's worth noting
  mutable std::atomic<uint64_t> _distance_computations = 0;
  mutable std::atomic<uint64_t> _metric_hops = 0;

  // Keep track of the sequence of nodes visited during search.
  // Each internal list consists of a sequence of boolean flags indicating
  // whether a visited node is a hub node or not.
  std::vector<std::vector<bool>> _visited_nodes_sequence;

  bool* _hub_nodes = nullptr; // A boolean array to keep track of hub nodes.
  // If a node is a hub, then _hub_nodes[node] = true, else false.

  // CXL-offload hub caching: host-side cache of hub nodes' vector data, so
  // processCandidateNode can compute their distances locally instead of
  // round-tripping to the distance server. Parallel to _hub_nodes; entry is
  // null unless that node is both a hub and has been cached via
  // cacheHubVectors(). Populated/freed via freeHubVectorCache().
  char** _hub_vector_cache = nullptr;

  // Tracking metrics for node access patterns. This unordered map is used to
  // record how many times each node is visited during search. The key is the
  // node id and the value is the number of times the node is visited.
  std::unordered_map<uint32_t, uint32_t> _node_access_counts;

  // Optional flat per-node graph-link visit counter for the caching study.
  // Allocated only when enableVisitProfiling() is called (400 MB at 100M nodes);
  // incremented in processCandidateNode (hot path) only under FLATNAV_PROFILE_VISITS,
  // so non-profiling builds are byte-identical on the search path.
  std::atomic<uint32_t>* _node_visit_counts = nullptr;
  // Companion counter: per-node DATA (vector) accesses = full activated footprint
  // (superset of link-expanded nodes). Also allocated by enableVisitProfiling().
  std::atomic<uint32_t>* _node_data_counts = nullptr;

  // Design-1 (SSSP) common source: when >= 0, every search starts from this fixed
  // entry node (skipping the per-query initialization scan), so all traversals are
  // rooted at one vertex. Set via setFixedEntryNode(); -1 = normal per-query entry.
  int64_t _fixed_entry_node = -1;

  // Randomization parameters
  bool _use_random_initialization = false;
  std::mt19937 _generator;
  std::uniform_int_distribution<> _distribution;


  Index(const Index &) = delete;
  Index &operator=(const Index &) = delete;

  // A custom move constructor is needed because the class manages dynamic
  // resources (_vectors_memory, _graph_memory, _visited_set_pool),
  // which require explicit ownership transfer and cleanup to avoid resource
  // leaks or double frees. The default move constructor cannot ensure these
  // resources are safely transferred and the source object is left in a valid
  // state.
  Index(Index&& other) noexcept
      : _vectors_memory(other._vectors_memory),
        _graph_memory(other._graph_memory),
        _vectors_numa_node(other._vectors_numa_node),
        _graph_numa_node(other._graph_numa_node),
        _M(other._M),
        _data_size_bytes(other._data_size_bytes),
        _node_size_bytes(other._node_size_bytes),
        _graph_node_size_bytes(other._graph_node_size_bytes),
        _max_node_count(other._max_node_count),
        _cur_num_nodes(other._cur_num_nodes),
        _distance(std::move(other._distance)),
        _num_threads(other._num_threads),
        _visited_set_pool(std::move(other._visited_set_pool)),
        _node_links_mutexes(std::move(other._node_links_mutexes)),
        _hub_nodes(other._hub_nodes),
        _hub_vector_cache(other._hub_vector_cache) {
    other._vectors_memory = nullptr;
    other._graph_memory = nullptr;
    other._visited_set_pool = nullptr;
    other._hub_nodes = nullptr;
    other._hub_vector_cache = nullptr;
  }

  Index& operator=(Index&& other) noexcept {
    if (this != &other) {
      util::freeBytes(_vectors_memory, vectorsMemoryBytes(), _vectors_numa_node);
      util::freeBytes(_graph_memory, graphMemoryBytes(), _graph_numa_node);
      delete _visited_set_pool;
      delete[] _hub_nodes;
      freeHubVectorCache();

      _vectors_memory = other._vectors_memory;
      _graph_memory = other._graph_memory;
      _vectors_numa_node = other._vectors_numa_node;
      _graph_numa_node = other._graph_numa_node;
      _M = other._M;
      _data_size_bytes = other._data_size_bytes;
      _node_size_bytes = other._node_size_bytes;
      _graph_node_size_bytes = other._graph_node_size_bytes;
      _max_node_count = other._max_node_count;
      _cur_num_nodes = other._cur_num_nodes;
      _distance = std::move(other._distance);
      _num_threads = other._num_threads;
      _visited_set_pool = std::move(other._visited_set_pool);
      _node_links_mutexes = std::move(other._node_links_mutexes);
      _hub_nodes = other._hub_nodes;
      _hub_vector_cache = other._hub_vector_cache;

      other._vectors_memory = nullptr;
      other._graph_memory = nullptr;
      other._visited_set_pool = nullptr;
      other._hub_nodes = nullptr;
      other._hub_vector_cache = nullptr;
    }
    return *this;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_data_type, _M, _data_size_bytes, _node_size_bytes,
            _graph_node_size_bytes, _max_node_count, _cur_num_nodes, *_distance);

    // Serialize the two storage regions separately. NUMA placement is a runtime
    // concern and is intentionally not persisted.
    archive(cereal::binary_data(_vectors_memory, vectorsMemoryBytes()));
    archive(cereal::binary_data(_graph_memory, graphMemoryBytes()));
  }

 public:
  /**
   * @brief Construct a new Index object for approximate near neighbor search.
   *
   * This constructor initializes an Index object with the specified distance
   * metric, dataset size, and maximum number of links per node. It also allows
   * for collecting statistics during the search process.
   *
   * @param dist The distance metric for the index. Options include l2
   * (euclidean) and inner product.
   * @param dataset_size The maximum number of vectors that can be inserted in
   * the index.
   * @param max_edges_per_node The maximum number of links per node.
   * @param collect_stats Flag indicating whether to collect statistics during
   * the search process.
   */
  Index(std::unique_ptr<DistanceInterface<dist_t>> dist, int dataset_size,
        int max_edges_per_node, bool collect_stats = false,
        bool use_random_initialization = false,
        std::optional<size_t> random_seed = std::nullopt,
        DataType data_type = DataType::float32,
        int vectors_numa_node = kNoNumaNode, int graph_numa_node = kNoNumaNode)
      : _vectors_numa_node(vectors_numa_node), _graph_numa_node(graph_numa_node),
        _M(max_edges_per_node), _max_node_count(dataset_size),
        _cur_num_nodes(0), _distance(std::move(dist)), _num_threads(1),
        _visited_set_pool(new VisitedSetPool(
            /* initial_pool_size = */ 1,
            /* num_elements = */ dataset_size)),
        _node_links_mutexes(dataset_size), _collect_stats(collect_stats),
        _use_random_initialization(use_random_initialization),
        _data_type(data_type) {

    if (random_seed.has_value()) {
      _generator = std::mt19937(random_seed.value());
      _distribution = std::uniform_int_distribution<>(0, _max_node_count - 1);
    }

    initNodeAccessCounts();

    _data_size_bytes = _distance->dataSize();
    _graph_node_size_bytes = (sizeof(node_id_t) * _M) + sizeof(label_t);
    _node_size_bytes = _data_size_bytes + _graph_node_size_bytes;

    _vectors_memory =
        util::allocateBytes(vectorsMemoryBytes(), _vectors_numa_node);
    _graph_memory = util::allocateBytes(graphMemoryBytes(), _graph_numa_node);

    _hub_nodes = new bool[_max_node_count];
    std::fill_n(_hub_nodes, _max_node_count, false);
#ifdef FLATNAV_CXL_OFFLOAD
    connectToDistanceServer("/flatnav_cxl");
#endif
  }

  void initNodeAccessCounts() {
    // Initialize the node access counts to 0 for all nodes.
    for (uint32_t i = 0; i < _max_node_count; i++) {
      _node_access_counts[i] = 0;
    }
  }

  ~Index() {
    util::freeBytes(_vectors_memory, vectorsMemoryBytes(), _vectors_numa_node);
    util::freeBytes(_graph_memory, graphMemoryBytes(), _graph_numa_node);
    delete _visited_set_pool;
    delete[] _hub_nodes;
    delete[] _node_visit_counts;
    delete[] _node_data_counts;
    freeHubVectorCache();
  }

  /**
   * @brief re-prune the graph by removing edges to hub nodes.
   * @param hub_nodes The hub nodes to prune edges from.
   * @param alpha The pruning threshold. \alpha ranges from 0 to 1.
   * Ex. if alpha = 0.5, then we remove 50% of the edges from the hub nodes.
   * Edge removal is done by setting the edge to the node itself.
   * The edge selection process is done using random selection.
   */
  void rePruneGraph(const std::vector<uint32_t> &hub_nodes, float alpha) {

    if (alpha < 0 || alpha > 1) {
      throw std::invalid_argument("Alpha must be in the range [0, 1].");
    }

    std::vector<std::pair<uint32_t, uint32_t>> edges_between_hub_nodes;
    for (const auto &hub_node : hub_nodes) {
      node_id_t *links = getNodeLinks(hub_node);
      for (size_t i = 0; i < _M; i++) {
        if (links[i] != hub_node) {
          edges_between_hub_nodes.emplace_back(hub_node, links[i]);
        }
      }
    }

    // Now randomly pick alpha * |edges_between_hub_nodes| edges to remove.
    std::shuffle(edges_between_hub_nodes.begin(), edges_between_hub_nodes.end(),
                 _generator);

    size_t num_edges_to_remove =
        static_cast<size_t>(alpha * edges_between_hub_nodes.size());
    for (size_t i = 0; i < num_edges_to_remove; i++) {
      auto [a, b] = edges_between_hub_nodes[i];
      node_id_t *links = getNodeLinks(a);
      for (size_t j = 0; j < _M; j++) {
        if (links[j] == b) {
          links[j] = a;
          break;
        }
      }
    }
  }

  void resetNodeAccessDistribution() { _node_access_counts.clear(); }

  // Caching study: allocate the flat per-node link-expansion + data-access counters (zeroed).
  void enableVisitProfiling() {
    delete[] _node_visit_counts;
    delete[] _node_data_counts;
    _node_visit_counts = new std::atomic<uint32_t>[_max_node_count]();
    _node_data_counts = new std::atomic<uint32_t>[_max_node_count]();
  }
  const std::atomic<uint32_t>* nodeVisitCounts() const { return _node_visit_counts; }
  const std::atomic<uint32_t>* nodeDataCounts() const { return _node_data_counts; }

  // Design-1 SSSP: fix the common source vertex for all subsequent searches.
  void setFixedEntryNode(int64_t s) { _fixed_entry_node = s; }

  // Medoid = node nearest to the dataset mean vector (a principled central source).
  // Two streaming passes over the vectors.
  node_id_t computeMedoid() {
    const size_t dim = _data_size_bytes / sizeof(float);
    std::vector<double> mean(dim, 0.0);
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      const float* v = reinterpret_cast<const float*>(getNodeData(n));
      for (size_t d = 0; d < dim; d++) mean[d] += v[d];
    }
    std::vector<float> m(dim);
    for (size_t d = 0; d < dim; d++) m[d] = static_cast<float>(mean[d] / (double)_cur_num_nodes);
    node_id_t best = 0; double best_dist = std::numeric_limits<double>::max();
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      const float* v = reinterpret_cast<const float*>(getNodeData(n));
      double s = 0;
      for (size_t d = 0; d < dim; d++) { double diff = (double)v[d] - m[d]; s += diff * diff; }
      if (s < best_dist) { best_dist = s; best = n; }
    }
    return best;
  }

  // Public access to relabel for external placement policies (P[old]=new id).
  void reorderByPermutation(const std::vector<node_id_t>& P) { relabel(P); }

  // In-degree of every node = number of incoming out-edges from other nodes
  // (self-loops excluded). A "hub" is then defined as a top-percentile in-degree node.
  std::vector<uint32_t> computeInDegrees() {
    std::vector<uint32_t> indeg(_cur_num_nodes, 0);
    for (node_id_t u = 0; u < _cur_num_nodes; u++) {
      const node_id_t* links = getNodeLinks(u);
      for (uint32_t i = 0; i < _M; i++) {
        node_id_t w = links[i];
        if (w < _cur_num_nodes && w != u) indeg[w]++;
      }
    }
    return indeg;
  }

  // Hop (graph-BFS) distance from `source` to every node over out-edges; 255 = unreachable.
  std::vector<uint8_t> bfsHopDistances(node_id_t source) {
    std::vector<uint8_t> dist(_cur_num_nodes, 255);
    std::vector<node_id_t> frontier, next;
    dist[source] = 0; frontier.push_back(source);
    uint8_t h = 0;
    while (!frontier.empty() && h < 254) {
      for (node_id_t u : frontier) {
        const node_id_t* links = getNodeLinks(u);
        for (uint32_t i = 0; i < _M; i++) {
          node_id_t w = links[i];
          if (w < _cur_num_nodes && dist[w] == 255) { dist[w] = h + 1; next.push_back(w); }
        }
      }
      frontier.swap(next); next.clear(); h++;
    }
    return dist;
  }

  void setHubNodeFlags(const std::vector<uint32_t>& hub_nodes) {
      for (const auto& hub_node: hub_nodes) {
        _hub_nodes[hub_node] = true;  
      }
  }

  std::vector<std::vector<bool>> getVisitedNodesSequence() {
    return _visited_nodes_sequence;
  }

#ifdef FLATNAV_CXL_OFFLOAD
  // Records the distance server's shared-memory segment name. Actual
  // connections happen lazily, per calling thread, in cxlClient() -- this
  // just makes the name available to that lazy path.
  void connectToDistanceServer(const std::string &shm_name) {
    _cxl_shm_name = shm_name;
  }

  // Per-thread CxlClients live in thread_local storage and are torn down
  // automatically when their owning thread exits, so there is nothing to
  // release here beyond asking the server to stop and preventing further
  // lazy connects.
  void disconnectFromDistanceServer() {
    if (!_cxl_shm_name.empty()) {
      flatnav::util::CxlClient shutdown_client(_cxl_shm_name, /* slot_id = */ 0);
      if (shutdown_client.connect()) {
        shutdown_client.sendShutdown();
        shutdown_client.disconnect();
      }
    }
    _cxl_shm_name.clear();
  }

  // Reads and caches every currently-flagged hub node's vector data directly
  // from the serialized index file, so processCandidateNode can compute
  // their distances locally instead of paying a CXL round trip for them.
  // Must be called after loadIndex() (which records _index_file_path and
  // _vectors_file_offset) and after setHubNodeFlags() has marked the hubs.
  void cacheHubVectors() {
    if (_index_file_path.empty()) {
      throw std::runtime_error(
          "cacheHubVectors requires an index loaded via loadIndex() under FLATNAV_CXL_OFFLOAD.");
    }

    std::ifstream stream(_index_file_path, std::ios::binary);
    if (!stream.is_open()) {
      throw std::runtime_error("cacheHubVectors: unable to reopen index file: " + _index_file_path);
    }

    freeHubVectorCache();
    _hub_vector_cache = new char*[_max_node_count]();

    for (node_id_t node = 0; node < _max_node_count; node++) {
      if (!_hub_nodes[node]) {
        continue;
      }
      stream.seekg(_vectors_file_offset + static_cast<std::streamoff>(node) * _data_size_bytes);
      char* buf = new char[_data_size_bytes];
      stream.read(buf, static_cast<std::streamsize>(_data_size_bytes));
      _hub_vector_cache[node] = buf;
    }
  }

private:
  // One CxlClient per OS thread, bound to a distinct server slot. This is
  // what lets executeInParallel / executeInParallelPinned run several
  // concurrent search threads that each talk to the distance server over
  // their own IPC channel instead of contending on a single shared one.
  // NOTE: assumes a single live Index per process (matches the rest of the
  // CXL-offload design, e.g. distance_server.cpp / hbw_search.cpp).
  static inline thread_local std::unique_ptr<flatnav::util::CxlClient> _cxl_client_tl;
  static inline thread_local uint64_t _cxl_active_query_id = 0;

  flatnav::util::CxlClient &cxlClient() {
    if (!_cxl_client_tl) {
      // _num_threads is the upper bound on how many threads any single
      // parallel search call spawns (see setNumThreads); it must also be
      // <= the distance server's --threads count, or two threads could be
      // handed the same slot and corrupt each other's IPC requests.
      uint32_t slot_id = _cxl_next_slot.fetch_add(1, std::memory_order_relaxed) % _num_threads;
      _cxl_client_tl = std::make_unique<flatnav::util::CxlClient>(_cxl_shm_name, slot_id);
      if (!_cxl_client_tl->connect()) {
        throw std::runtime_error("Failed to connect to distance server at '" +
                                 _cxl_shm_name + "' (slot " + std::to_string(slot_id) + ")");
      }
    }
    return *_cxl_client_tl;
  }

  float computeDistanceCxl(uint32_t node_id) {
    float result = 0.0f;
    if (!cxlClient().computeDistances(_cxl_active_query_id, &node_id, 1, &result)) {
      throw std::runtime_error("CXL distance computation failed for node " + std::to_string(node_id));
    }
    return result;
  }

  // Batched version of computeDistanceCxl: one IPC round-trip for `count`
  // node ids instead of `count` round-trips. Callers should prefer this over
  // looping computeDistanceCxl() whenever multiple distances are needed for
  // the same active query, since the fixed per-round-trip overhead (shared
  // memory memcpy, atomic fence, spin-wait on both client and server) far
  // exceeds the cost of the distance computation itself.
  void computeDistancesCxl(const uint32_t *node_ids, uint32_t count, float *out_distances) {
    // The wire protocol caps a single request at 1024 node ids (see
    // Slot::REQUEST_BUFFER_SIZE in CxlSimulation.h), so chunk transparently
    // for callers that may exceed that (e.g. initializeSearch with a large
    // num_initializations).
    constexpr uint32_t kMaxBatch = 1024;
    for (uint32_t offset = 0; offset < count; offset += kMaxBatch) {
      uint32_t chunk = std::min(kMaxBatch, count - offset);
      if (!cxlClient().computeDistances(_cxl_active_query_id, node_ids + offset, chunk,
                                        out_distances + offset)) {
        throw std::runtime_error("CXL batch distance computation failed for " +
                                 std::to_string(chunk) + " nodes");
      }
    }
  }

  // Send half of computeDistancesCxl's round trip, without waiting for the
  // response -- pairs with recvDistancesCxl(). Lets processCandidateNode
  // compute hub-cached distances locally while the server works on this
  // batch instead of blocking on it immediately. Unlike computeDistancesCxl,
  // this does not chunk: callers must keep `count` <= 1024 (the wire
  // protocol's hard limit), which holds for a per-hop neighbor batch since
  // that's bounded by _M.
  void sendDistancesCxl(const uint32_t *node_ids, uint32_t count) {
    if (!cxlClient().sendDistanceRequest(_cxl_active_query_id, node_ids, count)) {
      throw std::runtime_error("CXL distance request failed to send for " +
                               std::to_string(count) + " nodes");
    }
  }

  // Blocking half of the send/recv split -- see sendDistancesCxl().
  void recvDistancesCxl(uint32_t count, float *out_distances) {
    if (!cxlClient().recvDistanceResponse(count, out_distances)) {
      throw std::runtime_error("CXL distance response failed for " +
                               std::to_string(count) + " nodes");
    }
  }

public:
#endif


  void buildGraphLinks(const std::string& mtx_filename) {
    std::ifstream input_file(mtx_filename);
    if (!input_file.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + mtx_filename);
    }

    std::string line;
    // Skip the header
    while (std::getline(input_file, line)) {
      if (line[0] != '%')
        break;
    }

    std::istringstream iss(line);
    int num_vertices, num_edges;
    iss >> num_vertices >> num_vertices >> num_edges;

    // check that the number of vertices in the mtx file matches the number of
    // nodes in the index and that the number of edges is equal to the number of
    // links per node.
    if (num_vertices != _max_node_count) {
      throw std::runtime_error(
          "Number of vertices in the mtx file does not "
          "match the size allocated for the index.");
    }

    if (num_edges != _M) {
      throw std::runtime_error(
          "Number of edges in the mtx file does not match "
          "the number of links per node.");
    }

    int u, v;
    while (input_file >> u >> v) {
      // Adjust for 1-based indexing in Matrix Market format
      u--;
      v--;
      node_id_t* links = getNodeLinks(u);
      // Now add a directed edge from u to v. We need to check for the first
      // available slot in the links array since there might be other edges
      // added before this one. By definition, a slot is available if and only
      // if it points to the node itself.
      for (size_t i = 0; i < _M; i++) {
        if (links[i] == u) {
          links[i] = v;
          break;
        }
      }
    }

    input_file.close();
  }

  std::vector<std::vector<uint32_t>> getGraphOutdegreeTable() {
    std::vector<std::vector<uint32_t>> outdegree_table(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      // allocate a vector of size 0 so that each node has an entry in the
      // outdegree table.
      outdegree_table[node] = std::vector<uint32_t>();
      node_id_t *links = getNodeLinks(node);
      for (int i = 0; i < _M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    return outdegree_table;
  }

  size_t cantorPairing(node_id_t a, node_id_t b) {
    // if (a > b) {
    //   std::swap(a, b);
    // }
    return (a + b) * (a + b + 1) / 2 + b;
  }

  /**
   * @brief Store the new node in the global data structure. In a
   * multi-threaded setting, the index data guard should be held by the caller
   * with an exclusive lock.
   *
   * @param data The vector to add.
   * @param label The label (meta-data) of the vector.
   * @param new_node_id The id of the new node.
   */
  void allocateNode(void* data, label_t& label, node_id_t& new_node_id) {
    new_node_id = _cur_num_nodes;
    _distance->transformData(
        /* destination = */ getNodeData(new_node_id),
        /* src = */ data);
    *(getNodeLabel(new_node_id)) = label;
    node_id_t* links = getNodeLinks(new_node_id);
    // Initialize all edges to self
    std::fill_n(links, _M, new_node_id);
    _cur_num_nodes++;
  }

  /**
   * @brief Adds vectors to the index in batches.
   *
   * This method is responsible for adding vectors in batches, represented by
   * `data`, to the underlying graph. Each vector is associated with a label
   * provided in the `labels` vector. The method efficiently handles concurrent
   * additions by dividing the workload among multiple threads, defined by
   * `_num_threads`.
   *
   * The method ensures thread safety by employing locking mechanisms at the
   * node level in the underlying `connectNeighbors` and `beamSearch` methods.
   * This allows multiple threads to safely add vectors to the index without
   * causing data races or inconsistencies in the graph structure.
   *
   * @param data Pointer to the array of vectors to be added.
   * @param labels A vector of labels corresponding to each vector in `data`.
   * @param ef_construction Parameter for controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm. Must be greater than 0.
   *
   * @exception std::invalid_argument Thrown if `num_initializations` is less
   * than or equal to 0.
   * @exception std::runtime_error Thrown if the maximum number of nodes in the
   * index is reached.
   */
  template <typename data_type>
  void addBatch(void* data, std::vector<label_t>& labels, int ef_construction,
                int num_initializations = 100) {
      if (num_initializations <= 0) {
          throw std::invalid_argument("num_initializations must be greater than 0.");
      }
      uint32_t total_num_nodes = labels.size();
      uint32_t data_dimension = _distance->dimension();

      // Don't spawn any threads if we are only using one.
      if (_num_threads == 1) {
          for (uint32_t row_index = 0; row_index < total_num_nodes; row_index++) {
              uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
              void* vector = (data_type*)data + offset;
              label_t label = labels[row_index];
              this->add(vector, label, ef_construction, num_initializations);
          }
          return;
      }

      flatnav::executeInParallel(
          /* start_index = */ 0, /* end_index = */ total_num_nodes,
          /* num_threads = */ _num_threads, /* function = */
          [&](uint32_t row_index) {
              uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
              void* vector = (data_type*)data + offset;
              label_t label = labels[row_index];
              this->add(vector, label, ef_construction, num_initializations);
          });
  }

  /**
   * @brief Adds a single vector to the index.
   *
   * This method is called internally by `addBatch` for each vector in the
   * batch. The method ensures thread safety by using locking primitives,
   * allowing it to be safely used in a multi-threaded environment.
   *
   * The method first checks if the current number of nodes has reached the
   * maximum capacity. If so, it throws a runtime error. It then locks the index
   * structure to prevent concurrent modifications while allocating a new node.
   * After unlocking, it connects the new node to its neighbors in the graph.
   *
   * @param data Pointer to the vector data being added.
   * @param label Label associated with the vector.
   * @param ef_construction Parameter controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm.
   *
   * @exception std::runtime_error Thrown if the maximum number of nodes is
   * reached.
   */
  void add(void* data, label_t& label, int ef_construction, int num_initializations) {

    if (_cur_num_nodes >= _max_node_count) {
      throw std::runtime_error(
          "Maximum number of nodes reached. Consider "
          "increasing the `max_node_count` parameter to "
          "create a larger index.");
    }
    std::unique_lock<std::mutex> global_lock(_index_data_guard);
    auto entry_node = initializeSearch(data, num_initializations);
    node_id_t new_node_id;
    allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = beamSearch<false>(
        /* query = */ data, /* entry_node = */ entry_node,
        /* buffer_size = */ ef_construction);

    int selection_M = std::max(static_cast<int>(_M / 2), 1);
    selectNeighbors(/* neighbors = */ neighbors, /* M = */ selection_M);
    connectNeighbors(neighbors, new_node_id);
  }

  /***
   * @brief Search the index for the k nearest neighbors of the query.
   * @param query The query vector.
   * @param K The number of nearest neighbors to return.
   * @param ef_search The search beam width.
   * @param num_initializations The number of random initializations to use.
   */
  std::vector<dist_label_t> search(const void* query, const int K, int ef_search,
                                   int num_initializations = 100) {
#ifdef FLATNAV_CXL_OFFLOAD
    uint64_t curr_query_id = _cxl_query_id_counter.fetch_add(1);
    _cxl_active_query_id = curr_query_id;
    cxlClient().cacheQuery(curr_query_id,
                           static_cast<const float *>(query), _distance->dimension());
#endif

    node_id_t entry_node;
    if (_fixed_entry_node >= 0) {
      entry_node = static_cast<node_id_t>(_fixed_entry_node);
    } else if (_use_random_initialization) {
      entry_node = randomlyInitializeSearch(query, num_initializations);
    } else {
      entry_node = initializeSearch(query, num_initializations);
    }
    PriorityQueue neighbors =
        beamSearch<true>(/* query = */ query,
                         /* entry_node = */ entry_node,
                         /* buffer_size = */ std::max(K, ef_search));
    auto size = neighbors.size();
    std::vector<dist_label_t> results;
    results.reserve(size);
    while (!neighbors.empty()) {
      auto [distance, node_id] = neighbors.top();
      auto label = *getNodeLabel(node_id);
      results.emplace_back(distance, label);
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t& left, const dist_label_t& right) { return left.first < right.first; });
    if (results.size() > static_cast<size_t>(K)) {
      results.resize(K);
    }

#ifdef FLATNAV_CXL_OFFLOAD
  cxlClient().evictQuery(curr_query_id);
#endif
    return results;
  }


  void doGraphReordering(const std::vector<std::string>& reordering_methods) {

    for (const auto& method : reordering_methods) {
      auto outdegree_table = getGraphOutdegreeTable();
      std::vector<node_id_t> P;
      if (method == "gorder") {
        P = std::move(util::gOrder<node_id_t>(outdegree_table, 5));
      } else if (method == "rcm") {
        P = std::move(util::rcmOrder<node_id_t>(outdegree_table));
      } else {
        throw std::invalid_argument("Invalid reordering method: " + method);
      }

      relabel(P);
    }
  }

  void reorderGOrder(const int window_size = 5) {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::gOrder<node_id_t>(outdegree_table, window_size);

    relabel(P);
  }

  void reorderRCM() {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = util::rcmOrder<node_id_t>(outdegree_table);
    relabel(P);
  }

  // NUMA placement (vectors_numa_node, graph_numa_node) lets the caller bind each
  // storage region to a specific NUMA node at load time (caching study: hot graph
  // links on the local node, vectors on the remote node). kNoNumaNode = default
  // allocator. Requires building with FLATNAV_USE_NUMA.
  // `cxl_shm_name` is only used in FLATNAV_CXL_OFFLOAD builds: it is the
  // distance server's shared-memory segment name that every search thread's
  // lazily-created CxlClient (see cxlClient()) will connect to.
  static std::unique_ptr<Index<dist_t, label_t>> loadIndex(
      const std::string& filename,
      int vectors_numa_node = util::kNoNumaNode,
      int graph_numa_node = util::kNoNumaNode,
      const std::string& cxl_shm_name = "/flatnav_cxl") {
    std::ifstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    cereal::BinaryInputArchive archive(stream);
    std::unique_ptr<Index<dist_t, label_t>> index(new Index<dist_t, label_t>());

    std::unique_ptr<DistanceInterface<dist_t>> dist = std::make_unique<dist_t>();

    // 1. Deserialize metadata
    archive(index->_data_type,
            index->_M,
            index->_data_size_bytes,
            index->_node_size_bytes,
            index->_graph_node_size_bytes,
            index->_max_node_count,
            index->_cur_num_nodes,
            *dist
    );
    index->_visited_set_pool = new VisitedSetPool(
        /* initial_pool_size = */ 1,
        /* num_elements = */ index->_max_node_count);
    index->_distance = std::move(dist);
    index->_num_threads = std::max((uint32_t)1, (uint32_t)std::thread::hardware_concurrency() / 2);
    index->_node_links_mutexes = std::vector<std::mutex>(index->_max_node_count);

    // Hub-node flags are not serialized; a loaded index starts with no hubs
    // marked (matching a freshly constructed index before setHubNodeFlags).
    index->_hub_nodes = new bool[index->_max_node_count];
    std::fill_n(index->_hub_nodes, index->_max_node_count, false);

    // 2. Allocate the two storage regions using deserialized metadata. NUMA
    // placement is not persisted; the caller may bind each region via the
    // loadIndex(filename, vectors_node, graph_node) overload.
    index->_vectors_numa_node = vectors_numa_node;
    index->_graph_numa_node = graph_numa_node;
    index->_graph_memory =
        util::allocateBytes(index->graphMemoryBytes(), index->_graph_numa_node);

#ifdef FLATNAV_CXL_OFFLOAD
    // Vectors are served by the distance server (search() offloads every
    // distance computation to it via computeDistanceCxl), so there is no
    // need to allocate or load them locally. We still have to skip past
    // their serialized bytes so the graph region below is read from the
    // right file offset -- cereal's binary_data is a plain byte range with
    // no length prefix, so seeking past it on the underlying stream is safe.
    // The stream position right now, before the skip, is exactly where the
    // vectors region begins -- record it so cacheHubVectors() can later seek
    // to any individual hub node's vector (offset + node_id * data_size_bytes)
    // without keeping the whole block resident.
    index->_index_file_path = filename;
    index->_vectors_file_offset = stream.tellg();
    index->_vectors_memory = nullptr;
    stream.seekg(static_cast<std::streamoff>(index->vectorsMemoryBytes()), std::ios::cur);
    index->connectToDistanceServer(cxl_shm_name);
#else
    index->_vectors_memory =
        util::allocateBytes(index->vectorsMemoryBytes(), index->_vectors_numa_node);
    archive(cereal::binary_data(index->_vectors_memory, index->vectorsMemoryBytes()));
#endif

    // 3. Deserialize the graph region.
    archive(cereal::binary_data(index->_graph_memory, index->graphMemoryBytes()));

    return index;
  }

  void saveIndex(const std::string& filename) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  inline void setNumThreads(uint32_t num_threads) {
    if (num_threads == 0 || num_threads > std::thread::hardware_concurrency()) {
      throw std::invalid_argument(
          "Number of threads must be greater than 0 and less than or equal to "
          "the number of hardware threads.");
    }
    _num_threads = num_threads;
    if (_num_threads == 1) {
      _visited_set_pool->setPoolSize(1);
    }
  }


  inline uint64_t getTotalIndexMemory() const {
    return static_cast<uint64_t>(_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
  }

  // Byte size of the vectors region ([data] per node).
  inline uint64_t vectorsMemoryBytes() const {
    return static_cast<uint64_t>(_data_size_bytes) * static_cast<uint64_t>(_max_node_count);
  }

  // Byte size of the graph region ([M links][label] per node).
  inline uint64_t graphMemoryBytes() const {
    return static_cast<uint64_t>(_graph_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
  }
  inline uint64_t mutexesAllocatedMemory() const {
    return static_cast<uint64_t>(_node_links_mutexes.size() * sizeof(std::mutex));
  }

  inline uint64_t visitedSetPoolAllocatedMemory() const {
    size_t pool_size = _visited_set_pool->poolSize();
    return static_cast<uint64_t>(pool_size * sizeof(VisitedSet));
  }

  inline uint32_t getNumThreads() const { return _num_threads; }

  inline size_t maxEdgesPerNode() const { return _M; }
  inline size_t dataSizeBytes() const { return _data_size_bytes; }

  inline size_t nodeSizeBytes() const { return _node_size_bytes; }

  inline size_t maxNodeCount() const { return _max_node_count; }

  inline size_t currentNumNodes() const { return _cur_num_nodes; }
  inline size_t dataDimension() const { return _distance->dimension(); }

  // Region pointers + graph stride — for external NUMA tiering (mbind) after relabel.
  inline char* vectorsMemory() const { return _vectors_memory; }
  inline char* graphMemory() const { return _graph_memory; }
  inline size_t graphNodeSizeBytes() const { return _graph_node_size_bytes; }

  inline uint64_t distanceComputations() const { return _distance_computations.load(); }

  inline DataType getDataType() const { return _data_type; }

  void resetStats() {
    _distance_computations = 0;
    _metric_hops = 0;
  }

  // Return a reference to the node access counts
  inline const std::unordered_map<uint32_t, uint32_t> &
  getNodeAccessCounts() const {
    return _node_access_counts;
  }

  void getIndexSummary() const {
    std::cout << "\nIndex Parameters\n" << std::flush;
    std::cout << "-----------------------------\n" << std::flush;
    std::cout << "max_edges_per_node (M): " << _M << "\n" << std::flush;
    std::cout << "data_size_bytes: " << _data_size_bytes << "\n" << std::flush;
    std::cout << "node_size_bytes: " << _node_size_bytes << "\n" << std::flush;
    std::cout << "max_node_count: " << _max_node_count << "\n" << std::flush;
    std::cout << "cur_num_nodes: " << _cur_num_nodes << "\n" << std::flush;

    _distance->getSummary();
  }

 private:
  friend class cereal::access;
  // Default constructor for cereal
  Index() = default;

  char* getNodeData(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_data_size_bytes);
    return _vectors_memory + byte_offset;
  }

  node_id_t* getNodeLinks(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_graph_node_size_bytes);
    char* location = _graph_memory + byte_offset;
    return reinterpret_cast<node_id_t*>(location);
  }

  label_t* getNodeLabel(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_graph_node_size_bytes);
    byte_offset += (_M * sizeof(node_id_t));
    char* location = _graph_memory + byte_offset;
    return reinterpret_cast<label_t*>(location);
  }

  // Releases every cached hub vector buffer plus the cache array itself.
  void freeHubVectorCache() {
    if (_hub_vector_cache) {
      for (size_t i = 0; i < _max_node_count; i++) {
        delete[] _hub_vector_cache[i];
      }
      delete[] _hub_vector_cache;
      _hub_vector_cache = nullptr;
    }
  }

  inline void swapNodes(node_id_t a, node_id_t b, void* temp_data, node_id_t* temp_links,
                        label_t* temp_label) {

    // stash b in temp
    std::memcpy(temp_data, getNodeData(b), _data_size_bytes);
    std::memcpy(temp_links, getNodeLinks(b), _M * sizeof(node_id_t));
    std::memcpy(temp_label, getNodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(getNodeData(b), getNodeData(a), _data_size_bytes);
    std::memcpy(getNodeLinks(b), getNodeLinks(a), _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(b), getNodeLabel(a), sizeof(label_t));

    // put node b in a
    std::memcpy(getNodeData(a), temp_data, _data_size_bytes);
    std::memcpy(getNodeLinks(a), temp_links, _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(a), temp_label, sizeof(label_t));
  }

  /**
   * @brief Performs beam search for the nearest neighbors of the query.
   * @TODO: Add `entry_node_dist` argument to this function since we expect to
   * have computed that a priori.
   *
   * @param query               The query vector.
   * @param entry_node          The node to start the search from.
   * @param buffer_size         This is equivalent to `ef_search` in the HNSW
   *
   * @return PriorityQueue
   */
  template <bool is_search_stage = false>
  PriorityQueue beamSearch(const void *query, const node_id_t entry_node,
                           const int buffer_size) {
    PriorityQueue neighbors;
    PriorityQueue candidates;

    // Keep track of the nodes visited during the search.
    // Add True if the node is a hub node, else False.
    std::vector<bool> query_visited_nodes_flags;

    auto *visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    // Prefetch the data for entry node before computing its distance. Skipped
    // under CXL offload: there is no local vector memory to prefetch there.
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH) && !defined(FLATNAV_CXL_OFFLOAD)
    _mm_prefetch(getNodeData(entry_node), _MM_HINT_T0);
#endif

#ifdef FLATNAV_CXL_OFFLOAD
    float dist = computeDistanceCxl(entry_node);
#else
    float dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(entry_node),
                                     /* asymmetric = */ true);
#endif

#ifdef FLATNAV_PROFILE_VISITS
    if (_node_data_counts) _node_data_counts[entry_node].fetch_add(1, std::memory_order_relaxed);
#endif

    float max_dist = dist;
    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    query_visited_nodes_flags.push_back(_hub_nodes[entry_node]);
    visited_set->insert(entry_node);
#ifdef FLATNAV_PROFILE_PQ
    tl_disc.clear(); tl_parent_res.clear(); tl_step = 0; tl_cur_residency = 0;
    tl_disc[entry_node] = 0; tl_parent_res[entry_node] = 0;
    tl_pf.clear();
    tl_pf1_valid = false;
#endif

    while (!candidates.empty()) {
      auto [distance, node] = candidates.top();

      if (-distance > max_dist && neighbors.size() >= buffer_size) {
        break;
      }
      candidates.pop();
#ifdef FLATNAV_PROFILE_PQ
      // Residency of the node being expanded = tl_step (its pop step) - its discovery step.
      // tl_step is NOT incremented until after the expansion, so neighbors discovered below
      // are tagged with this node's pop step.
      {
        auto it = tl_disc.find(node);
        uint32_t res = tl_step - (it != tl_disc.end() ? it->second : tl_step);
        g_pq_residency_hist[res < kPQHistCap ? res : kPQHistCap - 1].fetch_add(1, std::memory_order_relaxed);
        // per-step (pop index) buckets for the early/late split
        uint32_t sidx = tl_step < kPQStepCap ? tl_step : kPQStepCap - 1;
        g_step_count[sidx].fetch_add(1, std::memory_order_relaxed);
        g_step_sumres[sidx].fetch_add(res, std::memory_order_relaxed);
        if (res == 1) g_step_leap[sidx].fetch_add(1, std::memory_order_relaxed);
        tl_cur_residency = res;            // neighbors discovered now inherit this as parent residency
        if (res == 1) {                    // leapfrogger (res 0 = entry node): how much lead did its parent give?
          auto pit = tl_parent_res.find(node);
          uint32_t pr = (pit != tl_parent_res.end()) ? pit->second : 0;
          g_leapfrog_parent_hist[pr < kPQHistCap ? pr : kPQHistCap - 1].fetch_add(1, std::memory_order_relaxed);
        }
        // top-K prefetch policy: mark this node as popped (a prefetch "success").
        { auto& e = tl_pf[node]; e.popped = true; e.pop_step = (uint16_t)tl_step; }
        // top-K prefetch policy: snapshot the K closest pending candidates NOW -- just after the
        // pop, before processCandidateNode discovers this node's neighbors -- so this step's
        // leapfroggers are excluded from every K. Record each node's first-prefetch step. tl_step
        // is still this step.
        {
          PriorityQueue tmp = candidates;  // copy; pop to read closest-first
          int maxK = kPFK[kPFnumK - 1];
          for (int r = 1; r <= maxK && !tmp.empty(); ++r) {
            node_id_t x = tmp.top().second; tmp.pop();
            auto& e = tl_pf[x];
            for (int ki = 0; ki < kPFnumK; ++ki)
              if (r <= kPFK[ki] && e.fe[ki] == 0xFFFF) e.fe[ki] = (uint16_t)tl_step;
          }
        }
        // pre-expansion K=1 NEXT-STEP precision: did last step's prefetched 2nd-min == the node
        // popped now? (i.e. used at the immediately next step). Then record this step's 2nd-min.
        if (tl_pf1_valid) {
          uint32_t b = tl_pf1_step < kPQStepCap ? tl_pf1_step : kPQStepCap - 1;
          g_pf1_next_total[b].fetch_add(1, std::memory_order_relaxed);
          if (node == tl_pf1_node) g_pf1_next_hit[b].fetch_add(1, std::memory_order_relaxed);
        }
        if (!candidates.empty()) {
          tl_pf1_node = candidates.top().second; tl_pf1_step = tl_step; tl_pf1_valid = true;
        } else {
          tl_pf1_valid = false;  // nothing left to prefetch this step
        }
      }
#endif

      // Prefetching the next candidate node data and visited set marker
      // before processing it. Note that this might not be useful if the current
      // iteration finds a neighbor that is closer than the current max
      // distance. In that case we would have prefetched data that is not used
      // immediately, but I think the cost of prefetching is low enough that
      // it's probably worth it.
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH)
      if (!candidates.empty()) {
#ifndef FLATNAV_CXL_OFFLOAD
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
#endif
        visited_set->prefetch(candidates.top().second);
      }
#endif

      processCandidateNode<is_search_stage>(
          /* query = */ query, /* node = */ node,
          /* max_dist = */ max_dist, /* buffer_size = */ buffer_size,
          /* visited_set = */ visited_set,
          /* neighbors = */ neighbors, /* candidates = */ candidates,
          /* query_visited_nodes = */ query_visited_nodes_flags);
#ifdef FLATNAV_PROFILE_PQ
      tl_step++;  // advance the step counter after the expansion completes
#endif
    }
#ifdef FLATNAV_PROFILE_PQ
    // Terminal: the 2nd-min prefetched at the final step has no "next step" (search ended) -> miss.
    if (tl_pf1_valid) {
      uint32_t b = tl_pf1_step < kPQStepCap ? tl_pf1_step : kPQStepCap - 1;
      g_pf1_next_total[b].fetch_add(1, std::memory_order_relaxed);
    }
    // End of query: tally each prefetched node as success (popped) or waste (never popped),
    // bucketed by its first-prefetch step, for every K.
    for (auto& kv : tl_pf) {
      PFEntry& e = kv.second;
      for (int ki = 0; ki < kPFnumK; ++ki) {
        if (e.fe[ki] == 0xFFFF) continue;
        uint32_t s = e.fe[ki] < kPQStepCap ? e.fe[ki] : kPQStepCap - 1;
        g_pf_prefetched[ki][s].fetch_add(1, std::memory_order_relaxed);
        if (e.popped) {
          g_pf_success[ki][s].fetch_add(1, std::memory_order_relaxed);
          g_pf_leadsum[ki].fetch_add((uint32_t)(e.pop_step - e.fe[ki]), std::memory_order_relaxed);
        }
      }
    }
#endif

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    return neighbors;
  }

  template <bool is_search_stage>
#ifdef FLATNAV_PROFILE_NOINLINE
  __attribute__((noinline))
#endif
  void processCandidateNode(const void *query, node_id_t &node, float &max_dist,
                            const int buffer_size, VisitedSet *visited_set,
                            PriorityQueue &neighbors,
                            PriorityQueue &candidates, std::vector<bool>& query_visited_nodes_flags) {
    // Lock all operations on this specific node. During search the graph is
    // frozen (read-only), so the per-node lock is pure overhead and is skipped;
    // it is only needed during construction (is_search_stage == false) to guard
    // concurrent writers.
    std::unique_lock<std::mutex> lock(_node_links_mutexes[node], std::defer_lock);
    if constexpr (!is_search_stage) {
      lock.lock();
    }

    node_id_t *neighbor_node_links = getNodeLinks(node);
#ifdef FLATNAV_PROFILE_VISITS
    // Count this node's graph-link access (the tiered/cached quantity).
    if (_node_visit_counts) _node_visit_counts[node].fetch_add(1, std::memory_order_relaxed);
#endif
    query_visited_nodes_flags.push_back(_hub_nodes[node]);

#ifdef FLATNAV_CXL_OFFLOAD
    // Collect all unvisited neighbors first so their distances can be
    // fetched from the distance server in a single batched round-trip
    // instead of one round-trip per neighbor (see computeDistancesCxl).
    static thread_local std::vector<node_id_t> tl_cxl_batch_ids;
    static thread_local std::vector<float> tl_cxl_batch_dists;
    static thread_local std::vector<node_id_t> tl_hub_ids;
    tl_cxl_batch_ids.clear();
    tl_hub_ids.clear();

    // Shared acceptance logic for a (neighbor, distance) pair, used both for
    // hub-cache hits (computed inline below, no round trip) and for the
    // batched CXL response (computed after the loop).
    auto accept = [&](node_id_t neighbor_node_id, float dist) {
      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace(-dist, neighbor_node_id);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef FLATNAV_PROFILE_PQ
        // Node enters the PQ now (discovered during the current expansion).
        tl_disc[neighbor_node_id] = tl_step;
        tl_parent_res[neighbor_node_id] = tl_cur_residency;
#endif
        if (neighbors.size() > buffer_size) {
          neighbors.pop();
        }
        if (!neighbors.empty()) {
          max_dist = neighbors.top().first;
        }
      }
    };
#endif

    for (uint32_t i = 0; i < _M; i++) {
      node_id_t neighbor_node_id = neighbor_node_links[i];

      // If using SSE, prefetch the next neighbor node data and the visited
      // marker
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH)
      if (i != _M - 1) {
#ifndef FLATNAV_CXL_OFFLOAD
        _mm_prefetch(getNodeData(neighbor_node_links[i + 1]), _MM_HINT_T0);
#endif
        visited_set->prefetch(neighbor_node_links[i + 1]);
      }
#endif

      bool neighbor_is_visited = visited_set->isVisited(/* num = */ neighbor_node_id);

      if (neighbor_is_visited) {
        continue;
      }
      visited_set->insert(/* num = */ neighbor_node_id);
#ifdef FLATNAV_PROFILE_VISITS
      // Count this neighbor's DATA (vector) access — the full activated footprint,
      // a superset of link-expanded nodes (these neighbors may never be expanded).
      if (_node_data_counts) _node_data_counts[neighbor_node_id].fetch_add(1, std::memory_order_relaxed);
#endif

#ifdef FLATNAV_CXL_OFFLOAD
      // Just bucket here -- the hub distances are computed further down,
      // after the non-hub batch's CXL request has been sent (not yet
      // waited on), so that local compute overlaps the round trip instead
      // of happening serially before it.
      //
      // Gate on _hub_nodes (1 byte/entry) rather than _hub_vector_cache
      // (8 bytes/entry, sparse) so the common non-hub case only touches the
      // smaller, more cache-friendly array. cacheHubVectors() only ever
      // populates _hub_vector_cache[node] where _hub_nodes[node] is true,
      // so once that's confirmed the pointer is known non-null.
      if (_hub_vector_cache && _hub_nodes[neighbor_node_id]) {
        tl_hub_ids.push_back(neighbor_node_id);
      } else {
        tl_cxl_batch_ids.push_back(neighbor_node_id);
      }
#else
      float dist = _distance->distance(/* x = */ query,
                                 /* y = */ getNodeData(neighbor_node_id),
                                 /* asymmetric = */ true);

      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }

      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace(-dist, neighbor_node_id);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef FLATNAV_PROFILE_PQ
        // Node enters the PQ now (discovered during the current expansion).
        tl_disc[neighbor_node_id] = tl_step;
        tl_parent_res[neighbor_node_id] = tl_cur_residency;
#endif
        // query_visited_nodes_flags.push_back(_hub_nodes[neighbor_node_id]);
#if defined(USE_SSE) && !defined(FLATNAV_DISABLE_PREFETCH)
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
#endif
        if (neighbors.size() > buffer_size) {
          neighbors.pop();
        }
        if (!neighbors.empty()) {
          max_dist = neighbors.top().first;
        }
      }
#endif
    }

#ifdef FLATNAV_CXL_OFFLOAD
    // Fire the non-hub batch request without waiting for the response, so
    // the hub-vector distances below (pure local compute) overlap with the
    // server's round trip instead of paying for it serially.
    bool cxl_pending = !tl_cxl_batch_ids.empty();
    // sendDistancesCxl/recvDistancesCxl don't chunk (unlike computeDistancesCxl),
    // so the overlap path only applies within the wire protocol's single-request
    // limit; an oversized batch (would require max_edges_per_node > 1024, not a
    // realistic graph config) falls back to the blocking, chunked call instead.
    bool cxl_async = cxl_pending && tl_cxl_batch_ids.size() <= 1024;
    if (cxl_pending) {
      tl_cxl_batch_dists.resize(tl_cxl_batch_ids.size());
      if (cxl_async) {
        sendDistancesCxl(tl_cxl_batch_ids.data(),
                         static_cast<uint32_t>(tl_cxl_batch_ids.size()));
      } else {
        computeDistancesCxl(tl_cxl_batch_ids.data(),
                            static_cast<uint32_t>(tl_cxl_batch_ids.size()),
                            tl_cxl_batch_dists.data());
      }
    }

    for (node_id_t neighbor_node_id : tl_hub_ids) {
      float dist = _distance->distance(/* x = */ query,
                                 /* y = */ _hub_vector_cache[neighbor_node_id],
                                 /* asymmetric = */ true);
      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }
      accept(neighbor_node_id, dist);
    }

    if (cxl_pending) {
      if (cxl_async) {
        recvDistancesCxl(static_cast<uint32_t>(tl_cxl_batch_ids.size()),
                         tl_cxl_batch_dists.data());
      }
      if (_collect_stats) {
        _distance_computations.fetch_add(tl_cxl_batch_ids.size());
      }
      for (size_t k = 0; k < tl_cxl_batch_ids.size(); k++) {
        accept(tl_cxl_batch_ids[k], tl_cxl_batch_dists[k]);
      }
    }
#endif
  }

  /**
   * @brief Selects neighbors from the PriorityQueue, according to the HNSW
   * heuristic. The neighbors priority queue contains elements sorted by
   * distance where the top element is the furthest neighbor from the query.
   */
  void selectNeighbors(PriorityQueue& neighbors, int M) {
    if (neighbors.size() < M) {
      return;
    }

    std::priority_queue<std::pair<float, node_id_t>> candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(M);

    while (neighbors.size() > 0) {
      auto [distance, id] = neighbors.top();

      candidates.emplace(-distance, id);
      neighbors.pop();
    }

    while (candidates.size() > 0) {
      if (saved_candidates.size() >= M) {
        break;
      }
      // Extract the closest element from candidates.
      auto [distance_to_query, current_node_id] = candidates.top();
      distance_to_query = -distance_to_query;
      candidates.pop();

      bool should_keep_candidate = true;
      for (const auto& [_, second_pair_node_id] : saved_candidates) {
        float cur_dist = _distance->distance(/* x = */ getNodeData(second_pair_node_id),
                                       /* y = */ getNodeData(current_node_id));

        if (cur_dist < distance_to_query) {
          should_keep_candidate = false;
          break;
        }
      }
      if (should_keep_candidate) {
        // We could do neighbors.emplace except we have to iterate
        // through saved_candidates, and std::priority_queue doesn't
        // support iteration (there is no technical reason why not).
        auto current_pair = std::make_pair(-distance_to_query, current_node_id);
        saved_candidates.push_back(current_pair);
      }
    }
    // TODO: implement my own priority queue, get rid of vector
    // saved_candidates, add directly to neighborqueue earlier.
    for (const dist_node_t& current_pair : saved_candidates) {
      neighbors.emplace(-current_pair.first, current_pair.second);
    }

  }

  void connectNeighbors(PriorityQueue& neighbors, node_id_t new_node_id) {
    // connects neighbors according to the HSNW heuristic

    // Lock all operations on this node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[new_node_id]);

    node_id_t* new_node_links = getNodeLinks(new_node_id);
    int i = 0;  // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)

      std::unique_lock<std::mutex> neighbor_lock(_node_links_mutexes[neighbor_node_id]);
      node_id_t* neighbor_node_links = getNodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _M; j++) {
        if (neighbor_node_links[j] == neighbor_node_id) {
          // If there is a self-loop, replace the self-loop with
          // the desired link.
          neighbor_node_links[j] = new_node_id;
          is_inserted = true;
          break;
        }
      }
      if (!is_inserted) {
        // now, we may to replace one of the links. This will disconnect
        // the old neighbor and create a directed edge, so we have to be
        // very careful. To ensure we respect the pruning heuristic, we
        // construct a candidate set including the old links AND our new
        // one, then prune this candidate set to get the new neighbors.

        float max_dist = _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                             /* y = */ getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance = _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                                /* y = */ getNodeData(label));
            candidates.emplace(distance, label);
          }
        }
        // 2X larger than the previous call to selectNeighbors.
        selectNeighbors(candidates, _M);
        // connect the pruned set of candidates, including self-loops:
        size_t j = 0;
        while (candidates.size() > 0) {  // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < _M) {  // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      // Unlock the current node we are iterating over
      neighbor_lock.unlock();

      // loop increments:
      i++;
      neighbors.pop();
    }
  }

  /**
   * @brief Selects a node to use as the entry point for a new node.
   * This proceeds in a greedy fashion, by selecting the node with
   * the smallest distance to the query.
   *
   * @param query
   * @param num_initializations
   * @return node_id_t
   */
  inline node_id_t initializeSearch(const void* query, int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }

    int step_size = _cur_num_nodes / num_initializations;
    step_size = step_size ? step_size : 1;

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

#ifdef FLATNAV_CXL_OFFLOAD
    std::vector<node_id_t> candidate_nodes;
    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
      candidate_nodes.push_back(node);
    }
    std::vector<float> candidate_dists(candidate_nodes.size());
    computeDistancesCxl(candidate_nodes.data(),
                        static_cast<uint32_t>(candidate_nodes.size()),
                        candidate_dists.data());
    for (size_t i = 0; i < candidate_nodes.size(); i++) {
      if (candidate_dists[i] < min_dist) {
        min_dist = candidate_dists[i];
        entry_node = candidate_nodes[i];
      }
    }
#else
    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
      float dist = _distance->distance(/* x = */ query,
                                 /* y = */ getNodeData(node),
                                 /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
#endif
    return entry_node;
  }

  // Use this during search to select a random entry point
  node_id_t randomlyInitializeSearch(const void *query,
                                     int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument(
          "num_initializations must be greater than 0.");
    }

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

#ifdef FLATNAV_CXL_OFFLOAD
    std::vector<node_id_t> candidate_nodes(num_initializations);
    for (int i = 0; i < num_initializations; i++) {
      candidate_nodes[i] = _distribution(_generator);
    }
    std::vector<float> candidate_dists(candidate_nodes.size());
    computeDistancesCxl(candidate_nodes.data(),
                        static_cast<uint32_t>(candidate_nodes.size()),
                        candidate_dists.data());
    for (size_t i = 0; i < candidate_nodes.size(); i++) {
      if (candidate_dists[i] < min_dist) {
        min_dist = candidate_dists[i];
        entry_node = candidate_nodes[i];
      }
    }
#else
    for (int i = 0; i < num_initializations; i++) {
      node_id_t node = _distribution(_generator);
      float dist = _distance->distance(/* x = */ query,
                                 /* y = */ getNodeData(node),
                                 /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
#endif
    return entry_node;
  }

  void relabel(const std::vector<node_id_t> &P) {
    // 1. Rewire all of the node connections
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      node_id_t* links = getNodeLinks(n);
      for (int m = 0; m < _M; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char* temp_data = new char[_data_size_bytes];
    node_id_t* temp_links = new node_id_t[_M];
    label_t* temp_label = new label_t;

    auto* visited_set = _visited_set_pool->pollAvailableSet();

    // In this context, is_visited stores which nodes have been relocated
    // (it would be equivalent to name this variable "is_relocated").
    visited_set->clear();

    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      if (visited_set->isVisited(/* num = */ n)) {
        continue;
      }

      node_id_t src = n;
      node_id_t dest = P[src];

      // swap node at src with node at dest
      swapNodes(src, dest, temp_data, temp_links, temp_label);

      // mark src as having been relocated
      visited_set->insert(src);

      // recursively relocate the node from "dest"
      while (!visited_set->isVisited(/* num = */ dest)) {
        // mark node as having been relocated
        visited_set->insert(dest);
        // the value of src remains the same. However, dest needs
        // to change because the node located at src was previously
        // located at dest, and must be relocated to P[dest].
        dest = P[dest];

        // swap node at src with node at dest
        swapNodes(src, dest, temp_data, temp_links, temp_label);
      }
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    delete[] temp_data;
    delete[] temp_links;
    delete temp_label;
  }
}; // namespace flatnav

}  // namespace flatnav