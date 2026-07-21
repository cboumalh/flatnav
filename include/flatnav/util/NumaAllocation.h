#pragma once

#include <cstddef>
#include <new>

#if defined(FLATNAV_USE_NUMA)
#include <numa.h>
#endif

namespace flatnav::util {

// Sentinel meaning "no NUMA binding": fall back to the standard allocator.
// This preserves the original (pre-NUMA) allocation behavior.
static constexpr int kNoNumaNode = -1;

// Returns true if NUMA-aware allocation is both compiled in and available at
// runtime (i.e. the kernel exposes a NUMA API on this machine).
inline bool numaIsAvailable() {
#if defined(FLATNAV_USE_NUMA)
  return numa_available() != -1;
#else
  return false;
#endif
}

// Allocates `size` bytes. When NUMA support is compiled in, NUMA is available
// at runtime, and `node >= 0`, the memory is bound to the given NUMA node via
// numa_alloc_onnode (page-aligned). Otherwise this falls back to a standard
// `new char[]` allocation, which matches the original behavior.
//
// The (`size`, `node`) pair must be passed back to freeBytes() so the matching
// deallocator is used.
inline char* allocateBytes(size_t size, int node) {
#if defined(FLATNAV_USE_NUMA)
  if (node >= 0 && numaIsAvailable()) {
    void* ptr = numa_alloc_onnode(size, node);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<char*>(ptr);
  }
#else
  (void)node;
#endif
  return new char[size];
}

// Frees memory returned by allocateBytes(). Must be called with the same
// `size` and `node` that were used to allocate it (numa_free requires the
// original size). A null pointer is ignored.
inline void freeBytes(char* ptr, size_t size, int node) {
  if (ptr == nullptr) {
    return;
  }
#if defined(FLATNAV_USE_NUMA)
  if (node >= 0 && numaIsAvailable()) {
    numa_free(ptr, size);
    return;
  }
#else
  (void)node;
  (void)size;
#endif
  delete[] ptr;
}

}  // namespace flatnav::util
