// Graph-link visit-frequency profiler for the caching-techniques study.
//
//   profile_visits <index.bin> <query.fvecs> [ef=200] [threads=32] [ranking_out.bin]
//
// Runs FlatNav search over all queries (entries unchanged = Design-2 / frequency
// ranking) while counting, per node, how often its graph links are accessed
// (processCandidateNode). Prints the coverage curve: what fraction of all
// link-accesses fall in the hottest f% of nodes -- i.e. the achievable LOCAL-HIT
// rate if the hottest f% of the graph lives on local DRAM. Optionally writes the
// hot->cold node ranking (uint32 ids) for the later relabel/placement step.
//
// Pin to one node, e.g.:  numactl --cpunodebind=0 --membind=0 ./profile_visits ...
#define FLATNAV_PROFILE_VISITS
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>
#include <flatnav/util/NumaThreadPool.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using flatnav::Index;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using dist_t = SquaredL2Distance<DataType::float32>;
using clk = std::chrono::steady_clock;

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

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <index.bin> <query.fvecs> [ef=200] [threads=32] [ranking_out.bin]\n", argv[0]);
    return 1;
  }
  const char* index_path = argv[1];
  const char* query_path = argv[2];
  int ef = argc > 3 ? atoi(argv[3]) : 200;
  int threads = argc > 4 ? atoi(argv[4]) : 32;
  const char* ranking_out = argc > 5 ? argv[5] : nullptr;
  const int K = 100;

  int qdim; size_t nq;
  std::vector<float> queries = readFvecs(query_path, qdim, nq);
  printf("[load] queries n=%zu dim=%d ef=%d threads=%d\n", nq, qdim, ef, threads);
  fflush(stdout);

  auto l0 = clk::now();
  auto index = Index<dist_t, int>::loadIndex(index_path);
  size_t N = index->currentNumNodes();
  printf("[load] index in %.1fs: nodes=%zu dim=%zu\n",
         std::chrono::duration<double>(clk::now() - l0).count(), N, index->dataDimension());
  fflush(stdout);

  index->enableVisitProfiling();

  // Design-1 SSSP: optional fixed common source (env FLATNAV_FIXED_ENTRY=<id>|medoid).
  long S = -1;
  const char* fe = getenv("FLATNAV_FIXED_ENTRY");
  if (fe) {
    S = (strcmp(fe, "medoid") == 0) ? (long)index->computeMedoid() : atol(fe);
    index->setFixedEntryNode(S);
    printf("[sssp] fixed common source S=%ld\n", S);
    fflush(stdout);
  }

  const char* pin_env = getenv("FLATNAV_PIN_CPUS");
  std::vector<int> pin_cpus = flatnav::parseCpuList(pin_env);
  bool pinned = !pin_cpus.empty();
  if (pinned) threads = static_cast<int>(pin_cpus.size());

  auto body = [&](uint32_t i) {
    const float* q = &queries[(size_t)i * qdim];
    volatile auto r = index->search(reinterpret_cast<const void*>(q), K, ef);
    (void)r;
  };
  auto t0 = clk::now();
  if (pinned) flatnav::executeInParallelPinned(0, nq, pin_cpus, body);
  else flatnav::executeInParallel(0, nq, threads, body);
  printf("[profile] searched %zu queries in %.1fs\n",
         nq, std::chrono::duration<double>(clk::now() - t0).count());
  fflush(stdout);

  // Snapshot the two counters once (reused for coverage + hop analysis).
  auto snapshot = [&](const std::atomic<uint32_t>* counts) {
    std::vector<uint32_t> c(N);
    for (size_t i = 0; i < N; i++) c[i] = counts[i].load(std::memory_order_relaxed);
    return c;
  };
  std::vector<uint32_t> c_link = snapshot(index->nodeVisitCounts());
  std::vector<uint32_t> c_data = snapshot(index->nodeDataCounts());

  // Report a footprint: print touched-fraction + coverage curve, optionally dump ranking.
  auto report = [&](const char* name, const std::vector<uint32_t>& c, const char* rank_out) {
    unsigned long long total = 0, touched = 0;
    for (size_t i = 0; i < N; i++) { total += c[i]; if (c[i]) touched++; }
    printf("\n===== %s footprint =====\n", name);
    printf("[stats] total accesses=%llu  touched-nodes=%llu (%.2f%% of %zu)  accesses/query=%.1f\n",
           total, touched, 100.0 * touched / N, N, (double)total / nq);
    std::vector<uint32_t> sorted = c;
    std::sort(sorted.begin(), sorted.end(), std::greater<uint32_t>());
    const double fgrid[] = {1,5,10,20,25,30,40,50,60,70,75,80,90,100};
    printf("  f%%(hottest nodes)   cumulative %% of accesses (= max local-hit)\n");
    for (double f : fgrid) {
      size_t k = (size_t)(f / 100.0 * N);
      unsigned long long cum = 0;
      for (size_t i = 0; i < k; i++) cum += sorted[i];
      printf("    %6.1f%%            %7.3f%%\n", f, total ? 100.0 * cum / total : 0.0);
    }
    if (rank_out) {
      std::vector<uint32_t> order(N);
      std::iota(order.begin(), order.end(), 0u);
      std::sort(order.begin(), order.end(),
                [&](uint32_t a, uint32_t b) { return c[a] > c[b]; });
      FILE* f = fopen(rank_out, "wb");
      if (!f) { perror("open rank_out"); return; }
      fwrite(order.data(), sizeof(uint32_t), N, f);
      fclose(f);
      printf("[out] hot->cold ranking (%zu ids) -> %s\n", N, rank_out);
    }
  };

  // Graph-LINK expansions (the links-only cache target) and full DATA activation.
  std::string data_rank = ranking_out ? std::string(ranking_out) + ".data" : std::string();
  report("LINK-expansion (graph-links cache)", c_link, ranking_out);
  report("DATA-access (full activation)", c_data,
         ranking_out ? data_rank.c_str() : nullptr);

  // SSSP hop-neighborhood analysis: where do the visited nodes sit relative to the
  // common source S? Cache the <=h-hop neighborhood of S => what % of accesses.
  if (S >= 0) {
    printf("\n===== N-hop neighborhood of common source S=%ld =====\n", S);
    auto t = clk::now();
    std::vector<uint8_t> hop = index->bfsHopDistances((uint32_t)S);
    int maxh = 0;
    for (size_t i = 0; i < N; i++) if (hop[i] != 255 && hop[i] > maxh) maxh = hop[i];
    printf("[bfs] done in %.1fs, max hop=%d\n",
           std::chrono::duration<double>(clk::now() - t).count(), maxh);

    std::vector<unsigned long long> nodes(maxh + 2, 0), lvis(maxh + 2, 0), dvis(maxh + 2, 0);
    unsigned long long ltot = 0, dtot = 0, unreached = 0;
    for (size_t i = 0; i < N; i++) {
      ltot += c_link[i]; dtot += c_data[i];
      int b = (hop[i] == 255) ? maxh + 1 : hop[i];
      nodes[b]++; lvis[b] += c_link[i]; dvis[b] += c_data[i];
      if (hop[i] == 255) unreached++;
    }
    printf("  hop   #nodes        %%link-visits  %%data-visits   cum%%link  cum%%data  cumNodes\n");
    unsigned long long cn = 0, cl = 0, cd = 0;
    for (int h = 0; h <= maxh; h++) {
      cn += nodes[h]; cl += lvis[h]; cd += dvis[h];
      printf("  %3d   %-12llu  %8.3f%%     %8.3f%%      %7.3f%%  %7.3f%%  %llu\n",
             h, nodes[h], ltot ? 100.0 * lvis[h] / ltot : 0, dtot ? 100.0 * dvis[h] / dtot : 0,
             ltot ? 100.0 * cl / ltot : 0, dtot ? 100.0 * cd / dtot : 0, cn);
    }
    printf("  unreached nodes=%llu (%.2f%%)  (their visits: link=%.3f%% data=%.3f%%)\n",
           unreached, 100.0 * unreached / N,
           ltot ? 100.0 * lvis[maxh + 1] / ltot : 0, dtot ? 100.0 * dvis[maxh + 1] / dtot : 0);

    // Top-visited (by link) nodes and their hop distance from S.
    std::vector<uint32_t> order(N);
    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(order.begin(), order.begin() + 20, order.end(),
                      [&](uint32_t a, uint32_t b) { return c_link[a] > c_link[b]; });
    printf("  top-20 link-hot nodes (id : link-visits : hop-from-S):\n   ");
    for (int i = 0; i < 20; i++)
      printf(" %u:%u:h%d", order[i], c_link[order[i]],
             hop[order[i]] == 255 ? -1 : hop[order[i]]);
    printf("\n");
  }
  return 0;
}
