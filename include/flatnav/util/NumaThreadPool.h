#pragma once

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace flatnav {

// Parses a comma-separated CPU list (e.g. "0,2,4,6") into a vector of ints.
// Returns an empty vector for a null/empty string.
inline std::vector<int> parseCpuList(const char* s) {
  std::vector<int> cpus;
  if (s == nullptr) return cpus;
  std::string cur;
  for (const char* p = s;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!cur.empty()) cpus.push_back(std::stoi(cur));
      cur.clear();
      if (*p == '\0') break;
    } else {
      cur += *p;
    }
  }
  return cpus;
}

// Like flatnav::executeInParallel, but spawns cpus.size() workers and pins
// worker k to cpus[k] via pthread affinity. Each worker stays on a fixed core,
// which keeps execution NUMA-local and makes perf cycle attribution stable
// (no scheduler migration). Work items in [start_index, end_index) are pulled
// off a shared atomic counter.
template <typename Function>
void executeInParallelPinned(uint32_t start_index, uint32_t end_index,
                             const std::vector<int>& cpus, Function function) {
  if (cpus.empty()) {
    throw std::invalid_argument("executeInParallelPinned: empty cpu list");
  }
  std::atomic<uint32_t> current(start_index);
  std::vector<std::thread> threads;
  threads.reserve(cpus.size());
  for (size_t t = 0; t < cpus.size(); t++) {
    int cpu = cpus[t];
    threads.emplace_back([&current, end_index, cpu, &function]() {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
      while (true) {
        uint32_t i = current.fetch_add(1);
        if (i >= end_index) break;
        function(i);
      }
    });
  }
  for (auto& th : threads) th.join();
}

}  // namespace flatnav
