#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>

#include "splinepy/core/concurrency/thread_pool.hpp"

namespace splinepy::core {

/// Number of threads an Options::nthreads value asks for: a negative value
/// means one per hardware thread, 0 and 1 mean sequential on the caller.
inline int ResolveThreadCount(int nthreads) noexcept {
  if (nthreads < 0) {
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  }
  return std::max(nthreads, 1);
}

/// Threads worth using for about work multiply-adds: one per kGrain, at
/// least one, at most ResolveThreadCount(nthreads). Waking a pool thread costs
/// about as much as kGrain multiply-adds, so small work stays on the caller.
inline int ThreadsFor(double work, int nthreads) {
  constexpr double kGrain = 32768.0;
  const double useful = std::max(1.0, work / kGrain);
  return static_cast<int>(
      std::min<double>(ResolveThreadCount(nthreads), useful));
}

/// Splits [start, end) into contiguous chunks of near equal size and runs
/// fn(chunk_begin, chunk_end, thread_id) once per chunk, at most one chunk per
/// thread. thread_id is in [0, ResolveThreadCount(nthreads)); the calling
/// thread runs as thread 0. The threads come from the shared pool (see
/// RunOnThreads); when it is busy, everything runs as one chunk on the
/// caller. The first exception thrown by any chunk is
/// rethrown on the caller after all chunks finished.
template<typename IndexType, typename Function>
void ParallelForChunks(IndexType start,
                       IndexType end,
                       int nthreads,
                       Function&& fn) {
  if (!(start < end)) {
    return;
  }
  const auto total = static_cast<std::size_t>(end - start);
  const auto threads =
      std::min(static_cast<std::size_t>(ResolveThreadCount(nthreads)), total);
  // the first `remainder` chunks take one extra item
  const std::size_t size = total / threads;
  const std::size_t remainder = total % threads;
  const auto chunk = [&](int t) {
    const auto i = static_cast<std::size_t>(t);
    const std::size_t first = i * size + std::min(i, remainder);
    const std::size_t count = size + (i < remainder ? 1 : 0);
    fn(start + static_cast<IndexType>(first),
       start + static_cast<IndexType>(first + count),
       t);
  };
  if (threads <= 1 || !RunOnThreads(static_cast<int>(threads), chunk)) {
    fn(start, end, 0);
  }
}

/// Parallelizes a loop [start, end), calling fn(i, thread_id) for every index.
/// thread_id identifies the worker, e.g. for per-thread workspaces.
template<typename IndexType, typename Function>
void ParallelForWithThreadId(IndexType start,
                             IndexType end,
                             int nthreads,
                             Function&& fn) {
  ParallelForChunks(start,
                    end,
                    nthreads,
                    [&](IndexType chunk_start, IndexType chunk_end, int tid) {
                      for (IndexType i = chunk_start; i < chunk_end; ++i) {
                        fn(i, tid);
                      }
                    });
}

/// Parallelizes a loop [start, end), calling fn(i) for every index.
template<typename IndexType, typename Function>
void ParallelFor(IndexType start, IndexType end, int nthreads, Function&& fn) {
  ParallelForChunks(start,
                    end,
                    nthreads,
                    [&](IndexType chunk_start, IndexType chunk_end, int) {
                      for (IndexType i = chunk_start; i < chunk_end; ++i) {
                        fn(i);
                      }
                    });
}

} // namespace splinepy::core
