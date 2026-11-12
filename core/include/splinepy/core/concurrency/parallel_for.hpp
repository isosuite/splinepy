#pragma once

#include <algorithm>
#include <exception>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace splinepy::core {

/// Parallelizes a loop [start, end) across worker threads using standard std::thread.
/// Zero external dependency (no OpenMP).
template <typename IndexType, typename Function>
void ParallelFor(IndexType start, IndexType end, int nthreads, Function&& fn) {
  if (start >= end) {
    return;
  }

  const auto total_work = static_cast<std::size_t>(end - start);
  const int max_threads = static_cast<int>(std::clamp(
      static_cast<std::size_t>(nthreads > 0 ? nthreads : 1),
      std::size_t{1},
      total_work));

  if (max_threads <= 1 || total_work <= 1) {
    for (IndexType i = start; i < end; ++i) {
      fn(i);
    }
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(max_threads - 1);
  std::exception_ptr error_ptr{nullptr};
  std::mutex error_mutex;

  const std::size_t chunk_size = total_work / max_threads;
  const std::size_t remainder = total_work % max_threads;

  auto run_chunk = [&](IndexType chunk_start, IndexType chunk_end) {
    try {
      for (IndexType i = chunk_start; i < chunk_end; ++i) {
        fn(i);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error_ptr) {
        error_ptr = std::current_exception();
      }
    }
  };

  std::size_t current_offset = 0;
  for (int t = 0; t < max_threads - 1; ++t) {
    const std::size_t this_chunk = chunk_size + (static_cast<std::size_t>(t) < remainder ? 1 : 0);
    const IndexType s = start + static_cast<IndexType>(current_offset);
    const IndexType e = s + static_cast<IndexType>(this_chunk);
    current_offset += this_chunk;
    workers.emplace_back(run_chunk, s, e);
  }

  // Master thread runs the final chunk
  const IndexType master_start = start + static_cast<IndexType>(current_offset);
  run_chunk(master_start, end);

  for (auto& w : workers) {
    if (w.joinable()) {
      w.join();
    }
  }

  if (error_ptr) {
    std::rethrow_exception(error_ptr);
  }
}

/// Parallelizes a loop [start, end) across worker threads, passing unique thread_id to fn(i, thread_id).
/// Useful for index-based access to per-thread workspaces.
template <typename IndexType, typename Function>
void ParallelForWithThreadId(IndexType start, IndexType end, int nthreads, Function&& fn) {
  if (start >= end) {
    return;
  }

  const auto total_work = static_cast<std::size_t>(end - start);
  const int max_threads = static_cast<int>(std::clamp(
      static_cast<std::size_t>(nthreads > 0 ? nthreads : 1),
      std::size_t{1},
      total_work));

  if (max_threads <= 1 || total_work <= 1) {
    for (IndexType i = start; i < end; ++i) {
      fn(i, 0);
    }
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(max_threads - 1);
  std::exception_ptr error_ptr{nullptr};
  std::mutex error_mutex;

  const std::size_t chunk_size = total_work / max_threads;
  const std::size_t remainder = total_work % max_threads;

  auto run_chunk = [&](IndexType chunk_start, IndexType chunk_end, int thread_id) {
    try {
      for (IndexType i = chunk_start; i < chunk_end; ++i) {
        fn(i, thread_id);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error_ptr) {
        error_ptr = std::current_exception();
      }
    }
  };

  std::size_t current_offset = 0;
  for (int t = 0; t < max_threads - 1; ++t) {
    const std::size_t this_chunk = chunk_size + (static_cast<std::size_t>(t) < remainder ? 1 : 0);
    const IndexType s = start + static_cast<IndexType>(current_offset);
    const IndexType e = s + static_cast<IndexType>(this_chunk);
    current_offset += this_chunk;
    workers.emplace_back(run_chunk, s, e, t + 1);
  }

  // Master thread runs the final chunk as thread 0
  const IndexType master_start = start + static_cast<IndexType>(current_offset);
  run_chunk(master_start, end, 0);

  for (auto& w : workers) {
    if (w.joinable()) {
      w.join();
    }
  }

  if (error_ptr) {
    std::rethrow_exception(error_ptr);
  }
}

}  // namespace splinepy::core
