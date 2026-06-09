#pragma once

// Per-phase timing instrumentation for the FlatNav search path.
//
// Maps to the CXL-ANNS query-scheduling phases:
//   Select (S)   -- pop nearest unvisited candidate            (beamSearch)
//   Traverse (T) -- read adjacency + per-node bookkeeping       (derived: Node - Dist - CI)
//   Dist         -- distance() call, INCLUDES the neighbor      (processCandidateNode)
//                   vector fetch (GraphRead) which is fused in
//   CI           -- candidate/neighbor heap insert + trim        (processCandidateNode)
//
// Zero cost unless built with -DFLATNAV_PROFILE_PHASES. Counters are
// thread_local with no locking, so profile SINGLE-THREADED for clean numbers
// (run one query stream on the calling thread).

#include <cstdint>
#include <cstdio>

#if defined(FLATNAV_PROFILE_PHASES)
#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>  // __rdtsc
#else
#include <chrono>
#endif
#endif

namespace flatnav {
namespace profiling {

#if defined(FLATNAV_PROFILE_PHASES)

enum class Phase : int { Select = 0, Initialize, Traverse, Dist, CI, Node, COUNT };

struct PhaseAccum {
  uint64_t cycles[static_cast<int>(Phase::COUNT)] = {0};
  uint64_t count[static_cast<int>(Phase::COUNT)] = {0};
};

// One set of counters per thread. Profile single-threaded; dump() reports the
// calling thread's counters.
inline thread_local PhaseAccum g_phase;

inline uint64_t now_ticks() {
#if defined(__x86_64__) || defined(_M_X64)
  return __rdtsc();
#else
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
#endif
}

inline void add(Phase p, uint64_t dt) {
  g_phase.cycles[static_cast<int>(p)] += dt;
  g_phase.count[static_cast<int>(p)] += 1;
}

inline void reset() { g_phase = PhaseAccum{}; }

inline void dump(const char* tag = "") {
  using P = Phase;
  // Traverse + per-node overhead = processCandidateNode total - Dist - CI.
  uint64_t node = g_phase.cycles[static_cast<int>(P::Node)];
  uint64_t dist = g_phase.cycles[static_cast<int>(P::Dist)];
  uint64_t ci = g_phase.cycles[static_cast<int>(P::CI)];
  g_phase.cycles[static_cast<int>(P::Traverse)] =
      (node > dist + ci) ? (node - dist - ci) : 0;
  g_phase.count[static_cast<int>(P::Traverse)] =
      g_phase.count[static_cast<int>(P::Node)];

  const char* names[] = {"Select (S)",
                        "Initialize",
                        "Traverse (T) + node overhead",
                        "Dist + GraphRead (fused)",
                        "Candidate Insert (CI)",
                        "processCandidateNode (total)"};
#if defined(__x86_64__) || defined(_M_X64)
  const char* unit = "cyc";
#else
  const char* unit = "ns";
#endif
  std::printf("=== FlatNav phase profile %s (this thread) ===\n", tag);
  for (int i = 0; i < static_cast<int>(P::COUNT); ++i) {
    uint64_t c = g_phase.cycles[i];
    uint64_t n = g_phase.count[i];
    double avg = n ? static_cast<double>(c) / static_cast<double>(n) : 0.0;
    std::printf("  %-30s total=%14llu %s  calls=%12llu  avg=%10.1f %s\n",
                names[i], static_cast<unsigned long long>(c), unit,
                static_cast<unsigned long long>(n), avg, unit);
  }
}

// Tight begin/end around a region in the SAME lexical scope. Phase tokens get
// distinct local names so multiple regions can coexist in one scope.
#define FN_PHASE_BEGIN(p) \
  uint64_t _fn_phase_t0_##p = ::flatnav::profiling::now_ticks()
#define FN_PHASE_END(p)                          \
  ::flatnav::profiling::add(::flatnav::profiling::Phase::p, \
                            ::flatnav::profiling::now_ticks() - _fn_phase_t0_##p)

#else  // profiling disabled -> no-ops

inline void reset() {}
inline void dump(const char* = "") {}
#define FN_PHASE_BEGIN(p) ((void)0)
#define FN_PHASE_END(p) ((void)0)

#endif

}  // namespace profiling
}  // namespace flatnav