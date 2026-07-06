#pragma once

#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/index/QueryState.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/VisitedSetPool.h>
#include <flatnav/util/Datatype.h>
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
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>
#include <optional>
#include <flatnav/util/PhaseProfiler.h>

using flatnav::distances::DistanceInterface;
using flatnav::util::VisitedSet;
using flatnav::util::VisitedSetPool;
using flatnav::util::DataType;

namespace flatnav {

// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
template <typename dist_t, typename label_t>
class Index {
  typedef std::pair<float, label_t> dist_label_t;
  // internal node numbering scheme. We might need to change this to uint64_t
  typedef uint32_t node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;
  using Qstate = QueryState<dist_node_t, node_id_t>;

  // NOTE: by default this is a max-heap. We could make this a min-heap
  // by using std::greater, but we want to use the queue as both a max-heap and
  // min-heap depending on the context.

  struct CompareByFirst {
    constexpr bool operator()(dist_node_t const& a, dist_node_t const& b) const noexcept{
      return a.first < b.first;
    }
  };
  static constexpr CompareByFirst cmp{};

  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>, CompareByFirst> PriorityQueue;

  // Large (several GB), pre-allocated block of memory.
  char* _index_memory;

  size_t _M;
  // size of one data point (does not support variable-size data, strings)
  size_t _data_size_bytes;
  // Node consists of: ([data] [M links] [data label]). This layout was chosen
  // after benchmarking - it's slightly more cache-efficient than others.
  size_t _node_size_bytes;
  size_t _max_node_count;  // Determines size of internal pre-allocated memory
  size_t _cur_num_nodes;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
  std::mutex _index_data_guard;

  uint32_t _num_threads;
  uint32_t _search_concurrency = 4;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  VisitedSetPool* _visited_set_pool;
  std::vector<std::mutex> _node_links_mutexes;

  bool _collect_stats = false;
  DataType _data_type;

#ifdef FLATNAV_CXL_OFFLOAD
  std::unique_ptr<flatnav::util::CxlClient> _cxl_client;
  std::atomic<uint64_t> _cxl_query_id_counter{0};
#endif

  // NOTE: These metrics are meaningful the most with single-threaded search.
  // With multi-threaded search, for instance, the number of distance computations will 
  // accumulate across queries, which means at the end of the batched search, the number 
  // you get is the cumulative sum of all distance computations across all queries.
  // Maybe that's what you want, but it's worth noting
  mutable std::atomic<uint64_t> _distance_computations = 0;
  mutable std::atomic<uint64_t> _metric_hops = 0;

  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;

  // A custom move constructor is needed because the class manages dynamic
  // resources (_index_memory, _visited_set_pool),
  // which require explicit ownership transfer and cleanup to avoid resource
  // leaks or double frees. The default move constructor cannot ensure these
  // resources are safely transferred and the source object is left in a valid
  // state.
  Index(Index&& other) noexcept
      : _index_memory(other._index_memory),
        _M(other._M),
        _data_size_bytes(other._data_size_bytes),
        _node_size_bytes(other._node_size_bytes),
        _max_node_count(other._max_node_count),
        _cur_num_nodes(other._cur_num_nodes),
        _distance(std::move(other._distance)),
        _num_threads(other._num_threads),
        _visited_set_pool(std::move(other._visited_set_pool)),
        _node_links_mutexes(std::move(other._node_links_mutexes)) {
    other._index_memory = nullptr;
    other._visited_set_pool = nullptr;
  }

  Index& operator=(Index&& other) noexcept {
    if (this != &other) {
      delete[] _index_memory;
      delete _visited_set_pool;

      _index_memory = other._index_memory;
      _M = other._M;
      _data_size_bytes = other._data_size_bytes;
      _node_size_bytes = other._node_size_bytes;
      _max_node_count = other._max_node_count;
      _cur_num_nodes = other._cur_num_nodes;
      _distance = std::move(other._distance);
      _num_threads = other._num_threads;
      _visited_set_pool = std::move(other._visited_set_pool);
      _node_links_mutexes = std::move(other._node_links_mutexes);

      other._index_memory = nullptr;
      other._visited_set_pool = nullptr;
    }
    return *this;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_data_type, _M, _data_size_bytes, _node_size_bytes, _max_node_count, _cur_num_nodes, *_distance);

    // Serialize the allocated memory for the index & query.
    uint64_t total_mem = static_cast<uint64_t>(_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
    archive(cereal::binary_data(_index_memory, total_mem));
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
  Index(std::unique_ptr<DistanceInterface<dist_t>> dist, int dataset_size, int max_edges_per_node,
        bool collect_stats = false, DataType data_type = DataType::float32)
      : _M(max_edges_per_node),
        _max_node_count(dataset_size),
        _cur_num_nodes(0),
        _distance(std::move(dist)),
        _num_threads(1),
        _visited_set_pool(new VisitedSetPool(
            /* initial_pool_size = */ 1,
            /* num_elements = */ dataset_size)),
        _node_links_mutexes(dataset_size),
        _collect_stats(collect_stats), _data_type(data_type) {

    // Get the size in bytes of the _node_links_mutexes vector.
    size_t mutexes_size_bytes = _node_links_mutexes.size() * sizeof(std::mutex);

    _data_size_bytes = _distance->dataSize();
    _node_size_bytes = _data_size_bytes + (sizeof(node_id_t) * _M) + sizeof(label_t);
    uint64_t index_size = static_cast<uint64_t>(_node_size_bytes) * static_cast<uint64_t>(_max_node_count);
    _index_memory = new char[index_size];
  }

  ~Index() {
    delete[] _index_memory;
    delete _visited_set_pool;
  }

#ifdef FLATNAV_CXL_OFFLOAD
  void connectToDistanceServer(const std::string &shm_name, uint32_t slot_id = 0) {
    _cxl_client = std::make_unique<flatnav::util::CxlClient>(shm_name, slot_id);
    if (!_cxl_client->connect()) {
      throw std::runtime_error("Failed to connect to distance server at '" + shm_name + "'");
    }
  }

  void disconnectFromDistanceServer() {
    if (_cxl_client) {
      _cxl_client->sendShutdown();
      _cxl_client->disconnect();
      _cxl_client.reset();
    }
  }

private:
  static inline thread_local uint64_t _cxl_active_query_id = 0;

  float computeDistanceCxl(uint32_t node_id) {
    float result = 0.0f;
    if (!_cxl_client->computeDistances(_cxl_active_query_id, &node_id, 1, &result)) {
      throw std::runtime_error("CXL distance computation failed for node " + std::to_string(node_id));
    }
    return result;
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
      node_id_t* links = getNodeLinks(node);
      for (int i = 0; i < _M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    return outdegree_table;
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
    auto [entry_node, _init_dist] = initializeSearch(data, num_initializations);
    node_id_t new_node_id;
    allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = beamSearch(
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
    _cxl_client->cacheQuery(curr_query_id,
                            static_cast<const float *>(query), _distance->dimension());
#endif

    auto [entry_node, _init_dist] = initializeSearch(query, num_initializations);
    PriorityQueue neighbors = beamSearch(/* query = */ query,
                                         /* entry_node = */ entry_node,
                                         /* buffer_size = */ std::max(ef_search, K));
                                         
    flatnav::profiling::dump("my_query");
    
    while (neighbors.size() > static_cast<size_t>(K)) {
      neighbors.pop();
    }

    auto result_size = neighbors.size();
    std::vector<dist_label_t> results(result_size);

    for (size_t i = result_size; i-- > 0;) {
      auto [dist, node_id] = neighbors.top();
      results[i] = {dist, *(getNodeLabel(node_id))};
      neighbors.pop();
    }

    flatnav::profiling::reset();

#ifdef FLATNAV_CXL_OFFLOAD
    _cxl_client->evictQuery(curr_query_id);
#endif

    return results;
  }

  std::vector<std::vector<dist_label_t>> concurrentBatchSearch(const void * queries,
      size_t num_queries, int K, int ef_search,
      int num_initializations = 100,
      uint32_t concurrency = 4) {

    if (concurrency == 0) {
      throw std::invalid_argument("concurrency must be greater than 0.");
    }

    std::vector<std::vector<dist_label_t>> all_results(num_queries);
    const size_t buffer_size = static_cast<size_t>(std::max(ef_search, K));
    const size_t actual_concurrency = std::min(concurrency, static_cast<uint32_t>(num_queries));
    std::vector<Qstate> states(actual_concurrency);

    // We need to track the idx of the query each slot is working on
    std::vector<size_t> slot_query_idx(actual_concurrency);
    size_t next_query = actual_concurrency;

    for(size_t q = 0; q < actual_concurrency; q++) {
      slot_query_idx[q] = q;
      states[q].query = static_cast<const char*>(queries) + q * _data_size_bytes;
    }

    auto init_results = interleavedInitializeSearch(states, num_initializations);

    for (size_t q = 0; q < actual_concurrency; q++) {
      seedBeamSearchState(states[q], init_results[q], buffer_size);
    }

    size_t active_count = actual_concurrency;

    while(active_count > 0){
      for (size_t q = 0; q < actual_concurrency; q++){
        auto &s = states[q];
        if(s.execution_state == QueryExecutionState::Done) continue;

        switch (s.execution_state){
          case QueryExecutionState::Unscheduled: {
            popCandidateOrFinish(
              s, K, buffer_size, num_initializations, queries,
              num_queries, next_query, slot_query_idx[q],
              all_results, active_count);
            break;
          }
          case QueryExecutionState::ProcessingLinks: {
            FN_PHASE_BEGIN(Node);
            processOneLink(s, buffer_size);
            FN_PHASE_END(Node);
            break;
          }
          case QueryExecutionState::Done:
            break;
        }
      }
    }

    flatnav::profiling::dump("my_query");
    flatnav::profiling::reset();

    return all_results;
  }

  std::vector<std::pair<node_id_t, float>> interleavedInitializeSearch(
      const std::vector<Qstate> &states, int num_initializations) {

    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }
    
    int step_size = _cur_num_nodes / num_initializations;
    step_size = std::max(step_size, 1);

    const size_t num_queries = states.size();
    struct InitState {
      node_id_t current_node = 0;
      node_id_t best_node = 0;
      float min_dist = std::numeric_limits<float>::max();
      bool done = false;
    };

    std::vector<InitState> init_states(num_queries);

    if(_collect_stats) {
      _distance_computations.fetch_add(
        static_cast<uint64_t>(num_initializations) * num_queries
      );
    }

    FN_PHASE_BEGIN(Initialize);
    size_t queries_remaning = num_queries;
    while(queries_remaning > 0) {
      for(size_t q = 0; q < num_queries; q++) {
        auto &is = init_states[q];
        if(is.done) continue;

#ifdef FLATNAV_CXL_OFFLOAD
        float dist = computeDistanceCxl(is.current_node);
#else
        float dist = _distance->distance(states[q].query,
          getNodeData(is.current_node), true);
#endif

        if (dist < is.min_dist) {
          is.min_dist = dist;
          is.best_node = is.current_node;
        }

        is.current_node += step_size;

        if (is.current_node >= _cur_num_nodes) {
          is.done = true;
          queries_remaning--;
        } else {
#ifdef USE_SSE
          _mm_prefetch(getNodeData(is.current_node), _MM_HINT_T0);
#endif
        }
      }
    }
    FN_PHASE_END(Initialize);

    std::vector<std::pair<node_id_t, float>> results(num_queries);
    for(size_t q = 0; q < num_queries; q++) {
      results[q] = {init_states[q].best_node, init_states[q].min_dist};
    }

    return results;
  }

  void seedBeamSearchState(Qstate& s, const std::pair<node_id_t, float>& entry, size_t buffer_size) {
    auto [entry_node, entry_dist] = entry;

    s.visited = _visited_set_pool->pollAvailableSet();
    s.visited->clear();  // Clear stale entries from previous query
    s.visited->insert(entry_node);

    s.max_dist = entry_dist;

    s.candidates.clear();
    s.candidates.emplace_back(-entry_dist, entry_node);
    std::push_heap(s.candidates.begin(), s.candidates.end(), cmp);

    s.neighbors.clear();
    s.neighbors.reserve(buffer_size);
    s.neighbors.emplace_back(entry_dist, entry_node);
    std::push_heap(s.neighbors.begin(), s.neighbors.end(), cmp);


    s.current_links = nullptr;
    s.link_idx = 0;
    s.execution_state = QueryExecutionState::Unscheduled;
  }

  void finishQuery(Qstate& s, int K, std::vector<dist_label_t>& results) {
    s.execution_state = QueryExecutionState::Done;
    while (s.neighbors.size() > static_cast<size_t>(K)) {
      std::pop_heap(s.neighbors.begin(), s.neighbors.end(), cmp);
      s.neighbors.pop_back();
    }

    size_t result_size = s.neighbors.size();
    results.resize(result_size);

    for (size_t i = result_size; i-- > 0;) {
      auto [dist, node_id] = s.neighbors.front();
      std::pop_heap(s.neighbors.begin(), s.neighbors.end(), cmp);
      s.neighbors.pop_back();
      results[i] = {dist, *(getNodeLabel(node_id))};
    }
  }

  inline void setSearchConcurrency(uint32_t concurrency) {
    if (concurrency == 0) {
      throw std::invalid_argument(
        "Search concurrency must be greater than 0");
    }
    _search_concurrency = concurrency;
  }

  bool popCandidateOrFinish(Qstate& s, int K, size_t buffer_size,
                            int num_initializations, const void* queries,
                            size_t num_queries, size_t& next_query,
                            size_t& slot_query_idx,
                            std::vector<std::vector<dist_label_t>>& all_results,
                            size_t& active_count) {
    
    // No candidates left to explore, search is done here
    if(s.candidates.empty()) {
      return replaceSlot(s, K, buffer_size, num_initializations, queries,
        num_queries, next_query, slot_query_idx, all_results, active_count);
    }

    FN_PHASE_BEGIN(Select);
    auto [neg_dist, node_id] = s.candidates.front();
    std::pop_heap(s.candidates.begin(), s.candidates.end(), cmp);
    s.candidates.pop_back();
    FN_PHASE_END(Select);


    // We ain't finding anything better, finish the search for this query
    if (-neg_dist > s.max_dist && s.neighbors.size() >= buffer_size) {
      return replaceSlot(s, K, buffer_size, num_initializations, queries,
        num_queries, next_query, slot_query_idx, all_results, active_count);
    }

    s.current_links = getNodeLinks(node_id);
    s.execution_state = QueryExecutionState::ProcessingLinks;
    s.link_idx = 0;

#ifdef USE_SSE
    _mm_prefetch(getNodeData(s.current_links[0]), _MM_HINT_T0);
#endif

    return false;
  }

  bool replaceSlot(Qstate& s, int K, size_t buffer_size,
                   int num_initializations, const void* queries,
                   size_t num_queries, size_t& next_query,
                   size_t& slot_query_idx,
                   std::vector<std::vector<dist_label_t>>& all_results,
                   size_t& active_count) {
    finishQuery(s, K, all_results[slot_query_idx]);

    flatnav::profiling::dump("my_query");
    flatnav::profiling::reset();

    if (next_query >= num_queries) {
      active_count--;
      return false;
    }

    slot_query_idx = next_query;
    next_query++;

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ s.visited);
    s.query = static_cast<const char*>(queries) + slot_query_idx * _data_size_bytes;
    FN_PHASE_BEGIN(Initialize);
    auto init_result = initializeSearch(s.query, num_initializations);
    seedBeamSearchState(s, init_result, buffer_size);
    FN_PHASE_END(Initialize);
    return true;
  }

  void processOneLink(Qstate& s, size_t buffer_size) {

    while (s.link_idx < _M) {
      node_id_t neighbor_id = s.current_links[s.link_idx];
      s.link_idx++;

#ifdef USE_SSE
    if (s.link_idx < _M) {
      _mm_prefetch(getNodeData(s.current_links[s.link_idx]), _MM_HINT_T0);
      s.visited->prefetch(s.current_links[s.link_idx]);
    }
#endif

      if(s.visited->isVisited(neighbor_id)) continue;
      s.visited->insert(neighbor_id);

      FN_PHASE_BEGIN(Dist);
#ifdef FLATNAV_CXL_OFFLOAD
      float dist = computeDistanceCxl(neighbor_id);
#else
      float dist = _distance->distance(s.query, getNodeData(neighbor_id), true);
#endif
      FN_PHASE_END(Dist);
      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }

      FN_PHASE_BEGIN(CI);
      if (s.neighbors.size() < buffer_size || dist < s.max_dist) {
        s.candidates.emplace_back(-dist, neighbor_id);
        std::push_heap(s.candidates.begin(), s.candidates.end(), cmp);

        s.neighbors.emplace_back(dist, neighbor_id);
        std::push_heap(s.neighbors.begin(), s.neighbors.end(), cmp);

        if (s.neighbors.size() > buffer_size) {
          std::pop_heap(s.neighbors.begin(), s.neighbors.end(), cmp);
          s.neighbors.pop_back();
        }
        if (!s.neighbors.empty()) {
          s.max_dist = s.neighbors.front().first;
        }
      }
      FN_PHASE_END(CI);

      break; // yield
    }

    if (s.link_idx >= _M) {
      s.execution_state = QueryExecutionState::Unscheduled;
    }
  }

  inline uint32_t getSearchConcurrency() const {
    return _search_concurrency;
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

  static std::unique_ptr<Index<dist_t, label_t>> loadIndex(const std::string& filename) {
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

    // 2. Allocate memory using deserialized metadata
    uint64_t mem_size = static_cast<uint64_t>(index->_node_size_bytes) * static_cast<uint64_t>(index->_max_node_count);

    index->_index_memory = new char[mem_size];

    // 3. Deserialize content into allocated memory
    archive(cereal::binary_data(index->_index_memory, mem_size));

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

  inline uint64_t distanceComputations() const { return _distance_computations.load(); }

  inline DataType getDataType() const { return _data_type; }

  void resetStats() {
    _distance_computations = 0;
    _metric_hops = 0;
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
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_node_size_bytes);
    return _index_memory + byte_offset;
  }

  node_id_t* getNodeLinks(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_node_size_bytes);
    byte_offset += _data_size_bytes;
    char* location = _index_memory + byte_offset;
    return reinterpret_cast<node_id_t*>(location);
  }

  label_t* getNodeLabel(const node_id_t& n) const {
    uint64_t byte_offset = static_cast<uint64_t>(n) * static_cast<uint64_t>(_node_size_bytes);
    byte_offset += _data_size_bytes;
    byte_offset += (_M * sizeof(node_id_t));
    char* location = _index_memory + byte_offset;
    return reinterpret_cast<label_t*>(location);
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

  PriorityQueue beamSearch(const void* query, const node_id_t entry_node, 
          const int buffer_size) {
    PriorityQueue neighbors;

    thread_local std::vector<dist_node_t> candidates;
    candidates.clear();

    auto* visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    // Prefetch the data for entry node before computing its distance.
#ifdef USE_SSE
    _mm_prefetch(getNodeData(entry_node), _MM_HINT_T0);
#endif

    // // Dereference data to ensure it's loaded into cache before timing Dist phase
    // volatile float* entry_data_ptr = (float*)getNodeData(entry_node);
    // volatile float ignored_val = entry_data_ptr[0];
    // (void)ignored_val;  // Suppress unused warning
    
    FN_PHASE_BEGIN(Initialize);
#ifdef FLATNAV_CXL_OFFLOAD
    float dist = computeDistanceCxl(entry_node);
#else
    float dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(entry_node),
                                     /* asymmetric = */ true);
#endif
    FN_PHASE_END(Initialize);

    float max_dist = dist;
    candidates.emplace_back(-dist, entry_node);
    std::push_heap(candidates.begin(), candidates.end(), cmp);
    neighbors.emplace(dist, entry_node);
    visited_set->insert(entry_node);

    while (!candidates.empty()) {
      FN_PHASE_BEGIN(Select);
      auto [distance, node] = candidates.front();

      if (-distance > max_dist && neighbors.size() >= buffer_size) {
        FN_PHASE_END(Select);
        break;
      }
      std::pop_heap(candidates.begin(), candidates.end(), cmp);
      candidates.pop_back();
      FN_PHASE_END(Select);

      // Prefetching the next candidate node data and visited set marker
      // before processing it. Note that this might not be useful if the current
      // iteration finds a neighbor that is closer than the current max
      // distance. In that case we would have prefetched data that is not used
      // immediately, but I think the cost of prefetching is low enough that
      // it's probably worth it.
#ifdef USE_SSE
      if (!candidates.empty()) {
        _mm_prefetch(getNodeData(candidates.front().second), _MM_HINT_T0);
        visited_set->prefetch(candidates.front().second);
      }
#endif

      FN_PHASE_BEGIN(Node);
      processCandidateNode(
          /* query = */ query, /* node = */ node,
          /* max_dist = */ max_dist, /* buffer_size = */ buffer_size,
          /* visited_set = */ visited_set,
          /* neighbors = */ neighbors, /* candidates = */ candidates);
      FN_PHASE_END(Node);
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    return neighbors;
  }

  void processCandidateNode(const void* query, node_id_t& node, float& max_dist, const int buffer_size,
                            VisitedSet* visited_set, PriorityQueue& neighbors, std::vector<dist_node_t>& candidates) {
    // Lock all operations on this specific node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[node]);

    node_id_t* neighbor_node_links = getNodeLinks(node);
    for (uint32_t i = 0; i < _M; i++) {
      node_id_t neighbor_node_id = neighbor_node_links[i];

      // If using SSE, prefetch the next neighbor node data and the visited
      // marker
#ifdef USE_SSE
      if (i != _M - 1) {
        _mm_prefetch(getNodeData(neighbor_node_links[i + 1]), _MM_HINT_T0);
        visited_set->prefetch(neighbor_node_links[i + 1]);
      }
#endif

      bool neighbor_is_visited = visited_set->isVisited(/* num = */ neighbor_node_id);

      if (neighbor_is_visited) {
        continue;
      }
      visited_set->insert(/* num = */ neighbor_node_id);

      // Dereference data to ensure it's loaded into cache before timing Dist phase
      // volatile float* neighbor_data_ptr = (float*)getNodeData(neighbor_node_id);
      // volatile float ignored_val = neighbor_data_ptr[0];
      // (void)ignored_val;  // Suppress unused warning

      FN_PHASE_BEGIN(Dist);
#ifdef FLATNAV_CXL_OFFLOAD
      float dist = computeDistanceCxl(neighbor_node_id);
#else
      float dist = _distance->distance(/* x = */ query,
                                 /* y = */ getNodeData(neighbor_node_id),
                                 /* asymmetric = */ true);
#endif
      FN_PHASE_END(Dist);

      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }

      FN_PHASE_BEGIN(CI);
      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace_back(-dist, neighbor_node_id);
        std::push_heap(candidates.begin(), candidates.end(), cmp);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef USE_SSE
        _mm_prefetch(getNodeData(candidates.front().second), _MM_HINT_T0);
#endif
        if (neighbors.size() > buffer_size) {
          neighbors.pop();
        }
        if (!neighbors.empty()) {
          max_dist = neighbors.top().first;
        }
      }
      FN_PHASE_END(CI);
    }
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
  inline std::pair<node_id_t, float> initializeSearch(const void* query, int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }

    FN_PHASE_BEGIN(Initialize);

    int step_size = _cur_num_nodes / num_initializations;
    step_size = step_size ? step_size : 1;

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
#ifdef USE_SSE
      node_id_t next_node = node + step_size;
      if (next_node < _cur_num_nodes) {
        _mm_prefetch(getNodeData(next_node), _MM_HINT_T0);
      }
#endif
#ifdef FLATNAV_CXL_OFFLOAD
      float dist = computeDistanceCxl(node);
#else
      float dist = _distance->distance(/* x = */ query, /* y = */ getNodeData(node),
                                       /* asymmetric = */ true);
#endif
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }

    FN_PHASE_END(Initialize);
    return {entry_node, min_dist};
  }

  void relabel(const std::vector<node_id_t>& P) {
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
};

}  // namespace flatnav
