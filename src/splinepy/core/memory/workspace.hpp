#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>

#include "splinepy/core/common.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Multi-threaded workspace holding independent MemoryArena instances per
/// worker thread. Provides zero-lock thread-local scratch allocation in
/// parallel evaluation loops.
///
/// Arenas live in segments that never move once created, and the arena count
/// is published atomically after an arena is fully built. GetThreadArena() is
/// therefore safe to call from many threads, also while another thread grows
/// the pool. One arena must still be used by one thread at a time, so a single
/// Workspace must not be shared by concurrent kernel calls.
class Workspace {
public:
  explicit Workspace(std::size_t per_thread_bytes = 64 * 1024,
                     std::size_t num_threads = 0)
      : per_thread_bytes_(std::max(per_thread_bytes, std::size_t{1024})) {
    const std::size_t hw_threads = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(std::thread::hardware_concurrency()));
    Reserve((num_threads > 0) ? num_threads : hw_threads);
  }

  ~Workspace() = default;

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  Workspace(Workspace&&) = delete;
  Workspace& operator=(Workspace&&) = delete;

  /// Retrieves the MemoryArena corresponding to the given thread ID.
  /// If thread_id exceeds the current pool, the pool grows under lock.
  MemoryArena& GetThreadArena(int thread_id = 0) {
    return *SlotAt(ToIndex(thread_id)).arena;
  }

  /// A workspace owned by worker thread_id, for kernels that worker calls
  /// itself. It is separate from every arena of this workspace, created on
  /// first use and kept for later calls. Like an arena, it belongs to one
  /// thread at a time.
  Workspace& GetWorkerWorkspace(int thread_id) {
    Slot& slot = SlotAt(ToIndex(thread_id));
    if (!slot.worker) {
      slot.worker = std::make_unique<Workspace>(per_thread_bytes_, 1);
    }
    return *slot.worker;
  }

  /// Ensures at least count thread arenas exist. Call before workers start.
  void Reserve(std::size_t count) {
    if (count <= count_.load(std::memory_order_acquire)) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t n = count_.load(std::memory_order_relaxed);
    for (; n < count; ++n) {
      std::size_t segment, offset;
      Locate(n, segment, offset);
      if (!segments_[segment]) {
        segments_[segment] =
            std::make_unique<Slot[]>(std::size_t{1} << segment);
      }
      segments_[segment][offset].arena =
          std::make_unique<MemoryArena>(per_thread_bytes_);
      count_.store(n + 1, std::memory_order_release);
    }
  }

  /// Releases excess block memory back to the OS across all thread arenas.
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    ForEachSlot([](Slot& slot) {
      slot.arena->Clear();
      if (slot.worker) {
        slot.worker->Clear();
      }
    });
  }

  /// Returns the current number of initialized thread arenas.
  [[nodiscard]] std::size_t NumThreadArenas() const noexcept {
    return count_.load(std::memory_order_acquire);
  }

private:
  struct Slot {
    std::unique_ptr<MemoryArena> arena;
    std::unique_ptr<Workspace> worker;
  };

  // Segment k holds 2^k slots, enough for any std::size_t index.
  static constexpr std::size_t kNumSegments = sizeof(std::size_t) * 8;

  static std::size_t ToIndex(int thread_id) noexcept {
    return static_cast<std::size_t>(thread_id >= 0 ? thread_id : 0);
  }

  static void Locate(std::size_t index,
                     std::size_t& segment,
                     std::size_t& offset) noexcept {
    const std::size_t v = index + 1;
    segment = 0;
    while ((v >> (segment + 1)) != 0) {
      ++segment;
    }
    offset = v - (std::size_t{1} << segment);
  }

  Slot& SlotAt(std::size_t index) {
    if (index >= count_.load(std::memory_order_acquire)) {
      Reserve(index + 1);
    }
    std::size_t segment, offset;
    Locate(index, segment, offset);
    return segments_[segment][offset];
  }

  template<typename Fn>
  void ForEachSlot(Fn&& fn) {
    const std::size_t n = count_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t segment, offset;
      Locate(i, segment, offset);
      fn(segments_[segment][offset]);
    }
  }

  std::size_t per_thread_bytes_{64 * 1024};
  std::unique_ptr<Slot[]> segments_[kNumSegments];
  std::atomic<std::size_t> count_{0};
  std::mutex mutex_;
};

/// Workspace used by kernels of the calling thread when Options::workspace is
/// null. Created on the first call of the thread, whose per_thread_bytes sets
/// its block size; later values are ignored. It keeps its blocks for reuse
/// until the thread ends or ReleaseDefaultWorkspaceMemory() is called.
inline Workspace& DefaultWorkspace(std::size_t per_thread_bytes = 64 * 1024) {
  thread_local Workspace workspace(per_thread_bytes, 1);
  return workspace;
}

/// Frees the extra blocks the calling thread's default workspace grew to (and
/// those of its worker workspaces). Only call it while no kernel runs on this
/// thread's default workspace.
inline void ReleaseDefaultWorkspaceMemory() { DefaultWorkspace().Clear(); }

/// Borrows options.workspace when one is given. Otherwise the calling thread's
/// default workspace is used, which is created once per thread. Either way the
/// workspace is grown to num_threads arenas here, before any worker runs.
/// Kernels restore their arena scope on exit, so nested and repeated calls
/// reuse the same memory instead of allocating a workspace per call.
class WorkspaceRef {
public:
  /// nthreads follows Options::nthreads: negative means one per hardware
  /// thread (see ResolveThreadCount).
  WorkspaceRef(const Options& options,
               std::size_t per_thread_bytes,
               int nthreads) {
    if (options.workspace != nullptr) {
      ws_ = options.workspace;
    } else {
      ws_ = &DefaultWorkspace(per_thread_bytes);
    }
    ws_->Reserve(static_cast<std::size_t>(ResolveThreadCount(nthreads)));
  }

  WorkspaceRef(const WorkspaceRef&) = delete;
  WorkspaceRef& operator=(const WorkspaceRef&) = delete;

  Workspace* get() noexcept { return ws_; }
  Workspace* operator->() noexcept { return ws_; }
  Workspace& operator*() noexcept { return *ws_; }

private:
  Workspace* ws_{nullptr};
};

/// Options for a kernel called by worker thread_id of a parallel loop over ws:
/// single threaded, and with that worker's own workspace, so workers never
/// share an arena with each other or with the loop's own scratch.
inline Options
WorkerOptions(const Options& options, Workspace& ws, int thread_id) {
  Options worker = options;
  worker.nthreads = 1;
  worker.workspace = &ws.GetWorkerWorkspace(thread_id);
  return worker;
}

/// What one call of a loop body may use for scratch, see ParallelForScratch.
struct Scratch {
  /// The arena of the thread that runs the call. Whatever the call allocates
  /// from it is released when the call returns.
  MemoryArena& arena;
  Workspace& workspace;
  int thread;

  /// Options for kernels the body calls itself: single threaded, with a
  /// workspace of this thread that is separate from arena, and no validation
  /// (the caller of the loop validated).
  Options Nested(const Options& options) const {
    Options nested = WorkerOptions(options, workspace, thread);
    nested.validate = false;
    return nested;
  }
};

/// The loop of a kernel that needs scratch memory: fn(i, scratch) for every
/// i in [0, count), on up to nthreads threads (see ParallelFor). This is the
/// one place that pairs threads with arenas: thread t allocates from arena t
/// of options' workspace, or of the calling thread's default workspace.
template<typename Index, typename Function>
void ParallelForScratch(Index count,
                        int nthreads,
                        const Options& options,
                        Function&& fn) {
  if (!(count > 0)) {
    return;
  }
  const int threads = static_cast<int>(
      std::min(static_cast<std::size_t>(count),
               static_cast<std::size_t>(ResolveThreadCount(nthreads))));
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, threads);
  ParallelForWithThreadId(Index{0}, count, threads, [&](Index i, int thread) {
    MemoryArena& arena = workspace->GetThreadArena(thread);
    const auto scope = arena.CreateScope();
    fn(i, Scratch{arena, *workspace, thread});
  });
}

} // namespace splinepy::core
