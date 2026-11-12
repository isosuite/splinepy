#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/memory_arena.hpp"

namespace splinepy::core {

/// Multi-threaded workspace holding independent MemoryArena instances per worker thread.
/// Provides zero-lock thread-local scratch allocation in parallel evaluation loops.
class Workspace {
 public:
  explicit Workspace(std::size_t per_thread_bytes = 64 * 1024,
                     std::size_t num_threads = 0)
      : per_thread_bytes_(std::max(per_thread_bytes, std::size_t{1024})) {
    const std::size_t hw_threads =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
    const std::size_t count = (num_threads > 0) ? num_threads : hw_threads;

    arenas_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      arenas_.emplace_back(std::make_unique<MemoryArena>(per_thread_bytes_));
    }
  }

  ~Workspace() = default;

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  Workspace(Workspace&&) = delete;
  Workspace& operator=(Workspace&&) = delete;

  /// Retrieves the MemoryArena corresponding to the given thread ID.
  /// If thread_id exceeds the current pool, safely expands under lock.
  MemoryArena& GetThreadArena(int thread_id = 0) {
    const std::size_t idx = static_cast<std::size_t>(thread_id >= 0 ? thread_id : 0);

    if (idx < arenas_.size()) {
      return *arenas_[idx];
    }

    std::lock_guard<std::mutex> lock(mutex_);
    while (arenas_.size() <= idx) {
      arenas_.emplace_back(std::make_unique<MemoryArena>(per_thread_bytes_));
    }
    return *arenas_[idx];
  }

  /// Resets all thread arenas without releasing their allocated block buffers.
  void ResetAll() noexcept {
    for (auto& a : arenas_) {
      if (a) {
        a->Reset();
      }
    }
  }

  /// Releases excess block memory back to the OS across all thread arenas.
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& a : arenas_) {
      if (a) {
        a->Clear();
      }
    }
  }

  /// Returns the current number of initialized thread arenas.
  [[nodiscard]] std::size_t NumThreadArenas() const noexcept {
    return arenas_.size();
  }

  /// Returns the default configured per-thread block size.
  [[nodiscard]] std::size_t PerThreadBytes() const noexcept {
    return per_thread_bytes_;
  }

 private:
  std::size_t per_thread_bytes_{64 * 1024};
  std::vector<std::unique_ptr<MemoryArena>> arenas_;
  std::mutex mutex_;
};

}  // namespace splinepy::core
