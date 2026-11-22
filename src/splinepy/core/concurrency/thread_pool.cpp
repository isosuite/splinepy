#include "splinepy/core/concurrency/thread_pool.hpp"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace splinepy::core::detail {
namespace {

/// Workers wait for a new generation, run their id of it if it has one, and
/// report back. One caller at a time: run_mutex_ is only tried, never waited
/// for. Never destroyed, so workers outlive static destruction.
class Pool {
public:
  bool Run(int n, PoolTask task, void* context) {
    if (n <= 1) {
      return false;
    }
    std::unique_lock<std::mutex> busy(run_mutex_, std::try_to_lock);
    if (!busy.owns_lock()) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      while (static_cast<int>(workers_.size()) < n - 1) {
        const int id = static_cast<int>(workers_.size()) + 1;
        workers_.emplace_back(
            [this, id, seen = generation_] { Work(id, seen); });
      }
      task_ = task;
      context_ = context;
      active_ = n - 1;
      remaining_ = n - 1;
      error_ = nullptr;
      ++generation_;
    }
    wake_.notify_all();
    Call(task, context, 0);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return remaining_ == 0; });
    if (error_) {
      std::rethrow_exception(std::exchange(error_, nullptr));
    }
    return true;
  }

private:
  void Call(PoolTask task, void* context, int id) {
    try {
      task(context, id);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!error_) {
        error_ = std::current_exception();
      }
    }
  }

  void Work(int id, uint64_t seen) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      wake_.wait(lock, [&] { return generation_ != seen; });
      seen = generation_;
      if (id > active_) {
        continue;
      }
      const PoolTask task = task_;
      void* const context = context_;
      lock.unlock();
      Call(task, context, id);
      lock.lock();
      if (--remaining_ == 0) {
        done_.notify_one();
      }
    }
  }

  std::mutex run_mutex_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::vector<std::thread> workers_;
  uint64_t generation_{0};
  PoolTask task_{nullptr};
  void* context_{nullptr};
  int active_{0};
  int remaining_{0};
  std::exception_ptr error_;
};

Pool*& Instance() {
  static Pool* pool = [] {
#if !defined(_WIN32)
    // a forked child has none of the parent's workers: start a fresh pool
    pthread_atfork(nullptr, nullptr, [] { Instance() = new Pool(); });
#endif
    return new Pool();
  }();
  return pool;
}

} // namespace

bool RunOnPool(int n, PoolTask task, void* context) {
  return Instance()->Run(n, task, context);
}

} // namespace splinepy::core::detail
