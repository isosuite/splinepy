#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>

#include "splinepy/core/concurrency/parallel_for.hpp"

namespace splinepy::core {

/// A fixed group of threads that runs one body together in stages (SPMD), for
/// work whose stages depend on each other. Unlike calling ParallelFor once per
/// stage, the threads are woken once per RunTeam call and meet at a barrier
/// between stages.
///
///   RunTeam(nthreads, [&](Team& team) {
///     team.For(n_a, [&](std::size_t k) { a[k] = ...; });
///     team.For(n_b, [&](std::size_t k) { b[k] = f(a); }); // sees all of a
///   });

/// State a Team shares: a reusable barrier and the first error.
class TeamState {
public:
  explicit TeamState(int size) : size_(size) {}

  /// Blocks until all size members arrived. Reusable: each round waits for
  /// its own generation.
  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const std::size_t generation = generation_;
    if (++arrived_ == size_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return generation_ != generation; });
  }

  void Fail(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_) {
      error_ = std::move(error);
    }
    failed_.store(true, std::memory_order_relaxed);
  }

  bool Failed() const { return failed_.load(std::memory_order_relaxed); }

  void RethrowIfFailed() const {
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  const int size_;
  int arrived_{0};
  std::size_t generation_{0};
  std::atomic<bool> failed_{false};
  std::exception_ptr error_;
};

/// One member's handle of a fixed group of threads that runs a body together,
/// see RunTeam. Every member makes the same sequence of For calls.
class Team {
public:
  /// A team of one: For runs everything right away and exceptions propagate.
  static Team Solo() { return Team(0, 1, nullptr); }

  Team(int id, int size, TeamState* state)
      : id_(id),
        size_(size),
        state_(state) {}

  int Id() const { return id_; }

  /// Runs fn(k) for every k in [0, count), split cyclically over the members,
  /// and returns once every member finished its share. After a failure the
  /// remaining calls skip their work and only synchronize.
  template<typename Function>
  void For(std::size_t count, Function&& fn) {
    if (state_ == nullptr) {
      for (std::size_t k = 0; k < count; ++k) {
        fn(k);
      }
      return;
    }
    if (!state_->Failed()) {
      try {
        for (auto k = static_cast<std::size_t>(id_); k < count;
             k += static_cast<std::size_t>(size_)) {
          fn(k);
        }
      } catch (...) {
        state_->Fail(std::current_exception());
      }
    }
    state_->ArriveAndWait();
  }

private:
  int id_;
  int size_;
  TeamState* state_;
};

/// Runs body(team) on ResolveThreadCount(nthreads) threads, the caller
/// included, which are woken once and synchronize at every team.For. body
/// must do all its work inside team.For, so that every member reaches each
/// barrier. The first exception is rethrown after all members finished.
template<typename Body>
void RunTeam(int nthreads, Body&& body) {
  const int n = ResolveThreadCount(nthreads);
  if (n == 1) {
    Team solo = Team::Solo();
    body(solo);
    return;
  }
  TeamState state(n);
  // the barrier needs all n members running at once: without them, one
  // member does everything
  const bool together = RunOnThreads(n, [&](int id) {
    Team team(id, n, &state);
    body(team);
  });
  if (!together) {
    Team solo = Team::Solo();
    body(solo);
    return;
  }
  state.RethrowIfFailed();
}

} // namespace splinepy::core
