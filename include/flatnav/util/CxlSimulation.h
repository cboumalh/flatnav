#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace flatnav::util {


/**** ENUMS ****/
enum class CxlMessageType : uint8_t {
  CacheQuery = 1,
  EvictQuery = 2,
  DistanceRequest = 3,
  DistanceResponse = 4,
  ErrorResponse = 5,
  StatsRequest = 6,
  StatsResponse = 7,
  StatsReset = 8,
  Shutdown = 9,
  Ack = 10
};

/**** MEMORY STRUCTS ****/

struct CxlRequest {
  std::atomic<uint32_t> ready_flag;
  CxlMessageType type;
  uint64_t query_id;
  uint32_t num_node_ids;
  uint32_t payload_size;
};

struct CxlResponse {
  std::atomic<uint32_t> ready_flag;
  CxlMessageType type;
  uint32_t num_distances;
  uint32_t error_code;
};

struct Slot {
  static constexpr size_t REQUEST_BUFFER_SIZE =
      sizeof(CxlRequest) + (1024 * sizeof(uint32_t)) +
      (2048 * sizeof(float)); // max 1024 node ids, max 2048-dim vectors

  static constexpr size_t RESPONSE_BUFFER_SIZE =
      sizeof(CxlResponse) + (1024 * sizeof(float)) + 256;

  alignas(64) char request_buffer[REQUEST_BUFFER_SIZE];
  alignas(64) char response_buffer[RESPONSE_BUFFER_SIZE];
};

struct IpcSharedRegion {
  uint32_t num_slots;
  uint32_t dimension;
  std::atomic<uint32_t> server_ready;
  std::atomic<uint32_t> shutdown_flag;
  Slot slots[];
};


/**** IPC SERVER ****/
class CxlServer {
public:
  CxlServer(const std::string &shm_name, uint32_t num_slots, uint32_t dimension)
      : _shm_name(shm_name), _num_slots(num_slots), _dimension(dimension),
        _region(nullptr), _shm_fd(-1), _total_size(0) {}

  ~CxlServer() {
    if (_region != nullptr) {
      destroy();
    }
  }

  void create() {
    _total_size = sizeof(IpcSharedRegion) + _num_slots * sizeof(Slot);

    _shm_fd = shm_open(_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (_shm_fd == -1) {
      throw std::runtime_error("CxlServer::create: shm_open failed for '" +
                               _shm_name + "': " + std::strerror(errno));
    }

    if (ftruncate(_shm_fd, static_cast<off_t>(_total_size)) == -1) {
      close(_shm_fd);
      shm_unlink(_shm_name.c_str());
      throw std::runtime_error("CxlServer::create: ftruncate failed: " +
                               std::string(std::strerror(errno)));
    }

    void *mapped = mmap(nullptr, _total_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, _shm_fd, 0);
    if (mapped == MAP_FAILED) {
      close(_shm_fd);
      shm_unlink(_shm_name.c_str());
      throw std::runtime_error("CxlServer::create: mmap failed: " +
                               std::string(std::strerror(errno)));
    }

    std::memset(mapped, 0, _total_size);

    _region = static_cast<IpcSharedRegion *>(mapped);
    _region->num_slots = _num_slots;
    _region->dimension = _dimension;
  }

  void destroy() {
    if (_region != nullptr) {
      munmap(_region, _total_size);
      _region = nullptr;
    }
    if (_shm_fd != -1) {
      close(_shm_fd);
      _shm_fd = -1;
    }
    shm_unlink(_shm_name.c_str());
  }

  bool pollRequest(uint32_t slot_id, CxlRequest &header, void *payload) {
    if (slot_id >= _num_slots || _region == nullptr) {
      return false;
    }

    Slot &slot = _region->slots[slot_id];
    CxlRequest *req_header =
        reinterpret_cast<CxlRequest *>(slot.request_buffer);

    if (req_header->ready_flag.load(std::memory_order_acquire) != 1) {
      return false;
    }

    header.type = req_header->type;
    header.query_id = req_header->query_id;
    header.num_node_ids = req_header->num_node_ids;
    header.payload_size = req_header->payload_size;

    if (payload != nullptr && header.payload_size > 0) {
      const char *payload_src =
          slot.request_buffer + sizeof(CxlRequest);
      std::memcpy(payload, payload_src, header.payload_size);
    }

    req_header->ready_flag.store(0, std::memory_order_release);

    return true;
  }

  void sendResponse(uint32_t slot_id, const CxlResponse &header,
                    const void *payload) {
    if (slot_id >= _num_slots || _region == nullptr) {
      return;
    }

    Slot &slot = _region->slots[slot_id];
    CxlResponse *resp_header =
        reinterpret_cast<CxlResponse *>(slot.response_buffer);

    resp_header->type = header.type;
    resp_header->num_distances = header.num_distances;
    resp_header->error_code = header.error_code;

    if (payload != nullptr) {
      size_t payload_size = 0;
      if (header.type == CxlMessageType::DistanceResponse) {
        payload_size = header.num_distances * sizeof(float);
      } else if (header.type == CxlMessageType::ErrorResponse) {
        payload_size = header.num_distances;
      } else if (header.type == CxlMessageType::StatsResponse) {
        payload_size = 4 * sizeof(uint64_t);
      }

      if (payload_size > 0) {
        char *payload_dst = slot.response_buffer + sizeof(CxlResponse);
        std::memcpy(payload_dst, payload, payload_size);
      }
    }

    resp_header->ready_flag.store(1, std::memory_order_release);
  }

  void setReady() {
    if (_region != nullptr) {
      _region->server_ready.store(1, std::memory_order_release);
    }
  }

  bool isShutdownRequested() const {
    if (_region == nullptr) {
      return false;
    }
    return _region->shutdown_flag.load(std::memory_order_acquire) != 0;
  }

  IpcSharedRegion *getRegion() { return _region; }

private:
  std::string _shm_name;
  uint32_t _num_slots;
  uint32_t _dimension;
  IpcSharedRegion *_region;
  int _shm_fd;
  size_t _total_size;
};




/**** IPC Client ****/
class CxlClient {
public:
  CxlClient(const std::string &shm_name, uint32_t slot_id)
      : _shm_name(shm_name), _slot_id(slot_id), _region(nullptr),
        _shm_fd(-1), _mapped_size(0), _connected(false) {}

  ~CxlClient() {
    if (_connected) {
      disconnect();
    }
  }

  bool connect(std::chrono::milliseconds timeout =
                   std::chrono::milliseconds(5000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;

    while (std::chrono::steady_clock::now() < deadline) {
      _shm_fd = shm_open(_shm_name.c_str(), O_RDWR, 0666);
      if (_shm_fd != -1) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (_shm_fd == -1) {
      return false;
    }

    struct stat shm_stat;
    if (fstat(_shm_fd, &shm_stat) == -1) {
      ::close(_shm_fd);
      _shm_fd = -1;
      return false;
    }
    _mapped_size = static_cast<size_t>(shm_stat.st_size);

    void *mapped =
        mmap(nullptr, _mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED,
             _shm_fd, 0);
    if (mapped == MAP_FAILED) {
      ::close(_shm_fd);
      _shm_fd = -1;
      return false;
    }

    _region = static_cast<IpcSharedRegion *>(mapped);

    if (_slot_id >= _region->num_slots) {
      munmap(_region, _mapped_size);
      ::close(_shm_fd);
      _region = nullptr;
      _shm_fd = -1;
      return false;
    }

    while (std::chrono::steady_clock::now() < deadline) {
      if (_region->server_ready.load(std::memory_order_acquire) == 1) {
        _connected = true;
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    munmap(_region, _mapped_size);
    ::close(_shm_fd);
    _region = nullptr;
    _shm_fd = -1;
    return false;
  }

  void disconnect() {
    if (_region != nullptr) {
      munmap(_region, _mapped_size);
      _region = nullptr;
    }
    if (_shm_fd != -1) {
      ::close(_shm_fd);
      _shm_fd = -1;
    }
    _connected = false;
  }

  bool cacheQuery(uint64_t query_id, const float *vector, size_t dimension) {
    if (!_connected || _region == nullptr) {
      return false;
    }

    Slot &slot = _region->slots[_slot_id];
    CxlRequest *req_header =
        reinterpret_cast<CxlRequest *>(slot.request_buffer);

    uint32_t payload_size = static_cast<uint32_t>(dimension * sizeof(float));
    char *payload_dst = slot.request_buffer + sizeof(CxlRequest);
    std::memcpy(payload_dst, vector, payload_size);

    req_header->type = CxlMessageType::CacheQuery;
    req_header->query_id = query_id;
    req_header->num_node_ids = 0;
    req_header->payload_size = payload_size;

    req_header->ready_flag.store(1, std::memory_order_release);

    return waitForResponse(CxlMessageType::Ack);
  }

  bool evictQuery(uint64_t query_id) {
    if (!_connected || _region == nullptr) {
      return false;
    }

    Slot &slot = _region->slots[_slot_id];
    CxlRequest *req_header =
        reinterpret_cast<CxlRequest *>(slot.request_buffer);

    req_header->type = CxlMessageType::EvictQuery;
    req_header->query_id = query_id;
    req_header->num_node_ids = 0;
    req_header->payload_size = 0;

    req_header->ready_flag.store(1, std::memory_order_release);

    return waitForResponse(CxlMessageType::Ack);
  }

  bool computeDistances(uint64_t query_id, const uint32_t *node_ids,
                        uint32_t count, float *out_distances) {
    if (!_connected || _region == nullptr) {
      return false;
    }
    if (count == 0 || count > 1024) {
      return false;
    }

    Slot &slot = _region->slots[_slot_id];
    CxlRequest *req_header =
        reinterpret_cast<CxlRequest *>(slot.request_buffer);

    uint32_t payload_size = count * sizeof(uint32_t);
    char *payload_dst = slot.request_buffer + sizeof(CxlRequest);
    std::memcpy(payload_dst, node_ids, payload_size);

    req_header->type = CxlMessageType::DistanceRequest;
    req_header->query_id = query_id;
    req_header->num_node_ids = count;
    req_header->payload_size = payload_size;

    req_header->ready_flag.store(1, std::memory_order_release);

    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    CxlResponse *resp_header =
        reinterpret_cast<CxlResponse *>(slot.response_buffer);

    while (std::chrono::steady_clock::now() < deadline) {
      if (resp_header->ready_flag.load(std::memory_order_acquire) == 1) {
        if (resp_header->type == CxlMessageType::ErrorResponse) {
          resp_header->ready_flag.store(0, std::memory_order_release);
          return false;
        }

        if (resp_header->type == CxlMessageType::DistanceResponse &&
            resp_header->num_distances == count) {
          const char *payload_src =
              slot.response_buffer + sizeof(CxlResponse);
          std::memcpy(out_distances, payload_src, count * sizeof(float));
          resp_header->ready_flag.store(0, std::memory_order_release);
          return true;
        }

        resp_header->ready_flag.store(0, std::memory_order_release);
        return false;
      }
      std::this_thread::yield();
    }

    return false;
  }

  bool sendShutdown() {
    if (!_connected || _region == nullptr) {
      return false;
    }

    Slot &slot = _region->slots[_slot_id];
    CxlRequest *req_header =
        reinterpret_cast<CxlRequest *>(slot.request_buffer);

    req_header->type = CxlMessageType::Shutdown;
    req_header->query_id = 0;
    req_header->num_node_ids = 0;
    req_header->payload_size = 0;

    req_header->ready_flag.store(1, std::memory_order_release);

    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    CxlResponse *resp_header =
        reinterpret_cast<CxlResponse *>(slot.response_buffer);

    while (std::chrono::steady_clock::now() < deadline) {
      if (resp_header->ready_flag.load(std::memory_order_acquire) == 1) {
        resp_header->ready_flag.store(0, std::memory_order_release);
        return (resp_header->type == CxlMessageType::Ack);
      }
      std::this_thread::yield();
    }

    return false;
  }

  bool requestStats(uint64_t &mean_ns, uint64_t &p50_ns, uint64_t &p99_ns,
                    uint64_t &total_requests) {
    if (!_connected || _region == nullptr) {
      return false;
    }

    Slot &slot = _region->slots[_slot_id];
    CxlRequest *req_header =
        reinterpret_cast<CxlRequest *>(slot.request_buffer);

    req_header->type = CxlMessageType::StatsRequest;
    req_header->query_id = 0;
    req_header->num_node_ids = 0;
    req_header->payload_size = 0;

    req_header->ready_flag.store(1, std::memory_order_release);

    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    CxlResponse *resp_header =
        reinterpret_cast<CxlResponse *>(slot.response_buffer);

    while (std::chrono::steady_clock::now() < deadline) {
      if (resp_header->ready_flag.load(std::memory_order_acquire) == 1) {
        if (resp_header->type == CxlMessageType::StatsResponse) {
          const uint64_t *stats_payload = reinterpret_cast<const uint64_t *>(
              slot.response_buffer + sizeof(CxlResponse));
          mean_ns = stats_payload[0];
          p50_ns = stats_payload[1];
          p99_ns = stats_payload[2];
          total_requests = stats_payload[3];
          resp_header->ready_flag.store(0, std::memory_order_release);
          return true;
        }
        resp_header->ready_flag.store(0, std::memory_order_release);
        return false;
      }
      std::this_thread::yield();
    }

    return false;
  }

  bool resetStats() {
    if (!_connected || _region == nullptr) {
      return false;
    }

    Slot &slot = _region->slots[_slot_id];
    CxlRequest *req_header =
        reinterpret_cast<CxlRequest *>(slot.request_buffer);

    req_header->type = CxlMessageType::StatsReset;
    req_header->query_id = 0;
    req_header->num_node_ids = 0;
    req_header->payload_size = 0;

    req_header->ready_flag.store(1, std::memory_order_release);

    return waitForResponse(CxlMessageType::Ack);
  }

  bool isConnected() const { return _connected; }

private:
  bool waitForResponse(CxlMessageType expected_type) {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);

    Slot &slot = _region->slots[_slot_id];
    CxlResponse *resp_header =
        reinterpret_cast<CxlResponse *>(slot.response_buffer);

    while (std::chrono::steady_clock::now() < deadline) {
      if (resp_header->ready_flag.load(std::memory_order_acquire) == 1) {
        bool success = (resp_header->type == expected_type &&
                        resp_header->error_code == 0);
        resp_header->ready_flag.store(0, std::memory_order_release);
        return success;
      }
      std::this_thread::yield();
    }

    return false;
  }

  std::string _shm_name;
  uint32_t _slot_id;
  IpcSharedRegion *_region;
  int _shm_fd;
  size_t _mapped_size;
  bool _connected;
};

}
