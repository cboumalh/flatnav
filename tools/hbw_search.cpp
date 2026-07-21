// High-bandwidth batch-search benchmark for a serialized FlatNav index.
//
//   hbw_search <index.bin> <query.fvecs> <gtruth.ivecs> [K=10] [threads=32] [ef=10,20,40,80,120]
//
// Run pinned to one NUMA node for the single-node high-bandwidth config, e.g.:
//   numactl --cpunodebind=0 --membind=0 ./hbw_search index.bin q.fvecs gt.ivecs
//
// Reports, per ef_search: wall time, QPS, and recall@K vs ground truth.
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/NumaThreadPool.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

using flatnav::Index;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

// Reads an .fvecs file fully into a float buffer; returns row count, sets dim.
static std::vector<float> readFvecs(const char* path, int& dim, size_t& n) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror("open fvecs"); exit(1); }
  struct stat st; fstat(fd, &st);
  size_t fsize = st.st_size;
  void* map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap fvecs"); exit(1); }
  const char* base = static_cast<const char*>(map);
  dim = *reinterpret_cast<const int32_t*>(base);
  size_t rec = 4 + static_cast<size_t>(dim) * 4;
  n = fsize / rec;
  std::vector<float> out(n * dim);
  for (size_t i = 0; i < n; i++)
    memcpy(&out[i * dim], base + i * rec + 4, dim * 4);
  munmap(map, fsize); close(fd);
  return out;
}

// Reads an .ivecs file (ground truth) into rows of int32 ids; sets width.
static std::vector<int> readIvecs(const char* path, int& width, size_t& n) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror("open ivecs"); exit(1); }
  struct stat st; fstat(fd, &st);
  size_t fsize = st.st_size;
  void* map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) { perror("mmap ivecs"); exit(1); }
  const char* base = static_cast<const char*>(map);
  width = *reinterpret_cast<const int32_t*>(base);
  size_t rec = 4 + static_cast<size_t>(width) * 4;
  n = fsize / rec;
  std::vector<int> out(n * width);
  for (size_t i = 0; i < n; i++)
    memcpy(&out[i * width], base + i * rec + 4, width * 4);
  munmap(map, fsize); close(fd);
  return out;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <index.bin> <query.fvecs> <gtruth.ivecs> "
                    "[K=10] [threads=32] [ef_list=10,20,40,80,120]\n", argv[0]);
    return 1;
  }
  const char* index_path = argv[1];
  const char* query_path = argv[2];
  const char* gt_path = argv[3];
  int K = argc > 4 ? atoi(argv[4]) : 10;
  int threads = argc > 5 ? atoi(argv[5]) : 32;
  std::vector<int> efs;
  if (argc > 6) {
    std::string s = argv[6], cur;
    for (char c : s) { if (c == ',') { efs.push_back(atoi(cur.c_str())); cur.clear(); } else cur += c; }
    if (!cur.empty()) efs.push_back(atoi(cur.c_str()));
  } else {
    efs = {10, 20, 40, 80, 120};
  }

  // Monotonic start reference. Phase markers below print elapsed seconds since
  // here, which aligns with `perf stat -I`'s interval timestamps (both count
  // from ~process start), so perf intervals can be accrued strictly after the
  // load phase without any fixed-delay guess.
  auto prog_start = clk::now();
  auto elapsed = [&] { return std::chrono::duration<double>(clk::now() - prog_start).count(); };

  int qdim; size_t nq;
  std::vector<float> queries = readFvecs(query_path, qdim, nq);
  int gw; size_t ngt;
  std::vector<int> gt = readIvecs(gt_path, gw, ngt);
  printf("[load] queries n=%zu dim=%d | gtruth n=%zu width=%d\n", nq, qdim, ngt, gw);
  fflush(stdout);
  if (ngt < nq) { fprintf(stderr, "gtruth rows < queries\n"); return 1; }

  // Optional per-region NUMA placement for the caching study (requires building
  // with -DFLATNAV_USE_NUMA -lnuma). Unset => default allocator (single-node).
  //   FLATNAV_GRAPH_NUMA=<node>   place graph links on this node (the local cache)
  //   FLATNAV_VECTORS_NUMA=<node> place vectors on this node (e.g. remote)
  const char* gnuma = getenv("FLATNAV_GRAPH_NUMA");
  const char* vnuma = getenv("FLATNAV_VECTORS_NUMA");
  int graph_node = gnuma ? atoi(gnuma) : -1;
  int vec_node = vnuma ? atoi(vnuma) : -1;

  auto l0 = clk::now();
  auto index = Index<dist_t, int>::loadIndex(index_path, vec_node, graph_node);
  printf("[numa] vectors_node=%d graph_node=%d\n", vec_node, graph_node);
  printf("[load] index in %.1fs: cur_nodes=%zu dim=%zu threads=%d K=%d\n",
         std::chrono::duration<double>(clk::now() - l0).count(),
         index->currentNumNodes(), index->dataDimension(), threads, K);
  printf("[phase] load_done elapsed_s=%.3f\n", elapsed());

  // Design-1 SSSP: optional fixed common source for all queries.
  //   FLATNAV_FIXED_ENTRY=<node_id>  or  FLATNAV_FIXED_ENTRY=medoid
  const char* fe = getenv("FLATNAV_FIXED_ENTRY");
  if (fe) {
    long S = (strcmp(fe, "medoid") == 0) ? (long)index->computeMedoid() : atol(fe);
    index->setFixedEntryNode(S);
    printf("[sssp] fixed common source S=%ld\n", S);
  }
  fflush(stdout);

  // Precompute ground-truth top-K sets per query.
  std::vector<std::unordered_set<int>> gtTopK(nq);
  for (size_t i = 0; i < nq; i++)
    for (int j = 0; j < K && j < gw; j++) gtTopK[i].insert(gt[i * gw + j]);

  std::vector<std::vector<std::pair<float, int>>> results(nq);

  // Optional per-core thread pinning via FLATNAV_PIN_CPUS (comma-separated cpu
  // list). When set, the worker count equals the number of pinned cpus and each
  // worker is bound to a fixed core (stable for perf attribution, NUMA-local).
  const char* pin_env = getenv("FLATNAV_PIN_CPUS");
  std::vector<int> pin_cpus = flatnav::parseCpuList(pin_env);
  bool pinned = !pin_cpus.empty();
  if (pinned) {
    threads = static_cast<int>(pin_cpus.size());
    printf("[pin] %zu cores: %s\n", pin_cpus.size(), pin_env);
    fflush(stdout);
  }

#ifdef FLATNAV_CXL_OFFLOAD
  // Each search thread gets its own CxlClient, bound to a slot in
  // [0, num_threads) on the distance server (see Index::cxlClient). This
  // must match the exact thread count used below by executeInParallel /
  // executeInParallelPinned, and the distance server must have been started
  // with at least this many --threads, or two search threads could collide
  // on the same slot.
  index->setNumThreads(static_cast<uint32_t>(threads));
  printf("[cxl] search threads=%d (must be <= distance server --threads)\n", threads);
  fflush(stdout);
#endif

  auto runBatch = [&](int ef) {
    auto body = [&](uint32_t i) {
      const float* q = &queries[(size_t)i * qdim];
      results[i] = index->search(reinterpret_cast<const void*>(q), K, ef);
    };
    if (pinned) flatnav::executeInParallelPinned(0, nq, pin_cpus, body);
    else flatnav::executeInParallel(0, nq, threads, body);
  };

  // Warmup: fault the index working set into caches/TLB so the first measured
  // ef isn't penalized by cold-start cost.
  runBatch(efs.back());
  printf("[phase] warmup_done elapsed_s=%.3f\n", elapsed());
  fflush(stdout);

  printf("%-8s %-12s %-12s %-10s\n", "ef", "time_s", "QPS", "recall@K");
  for (int ef : efs) {
    auto t0 = clk::now();
    runBatch(ef);
    double dt = std::chrono::duration<double>(clk::now() - t0).count();

    size_t hits = 0, total = 0;
    for (size_t i = 0; i < nq; i++) {
      for (auto& pr : results[i]) if (gtTopK[i].count(pr.second)) hits++;
      total += std::min((size_t)K, gtTopK[i].size());
    }
    printf("%-8d %-12.4f %-12.0f %-10.4f\n", ef, dt, nq / dt, (double)hits / total);
    fflush(stdout);
  }
  return 0;
}
