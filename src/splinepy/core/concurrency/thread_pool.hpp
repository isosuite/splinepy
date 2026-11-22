#pragma once

#include <type_traits>
#include <utility>

namespace splinepy::core {

namespace detail {

using PoolTask = void (*)(void* context, int id);

/// See RunOnThreads.
bool RunOnPool(int n, PoolTask task, void* context);

} // namespace detail

/// Runs fn(id) for every id in [0, n) at the same time, on n threads: the
/// caller runs id 0, workers of a process-wide pool the rest. The workers are
/// started once and then reused, so a call costs a wake-up, not a thread
/// start. The first exception thrown by any fn is rethrown here once all ids
/// finished.
///
/// Returns false without running anything if the pool is in use by another
/// caller; the caller then runs the work on its own thread. Calls must not
/// nest: work run here never starts parallel work itself (kernels inside a
/// parallel loop get single threaded WorkerOptions).
template<typename Function>
bool RunOnThreads(int n, Function&& fn) {
  using Fn = std::remove_reference_t<Function>;
  return detail::RunOnPool(
      n,
      [](void* context, int id) { (*static_cast<Fn*>(context))(id); },
      const_cast<void*>(static_cast<const void*>(&fn)));
}

} // namespace splinepy::core
