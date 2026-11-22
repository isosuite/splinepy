#pragma once

#include "splinepy/core/common.hpp"

namespace splinepy::core {

class Workspace;

struct Options {
  /// Whether to validate inputs prior to kernel execution.
  /// Defaults to true per project design decisions.
  bool validate{true};

  /// Number of worker threads for parallelized evaluation (std::thread).
  /// 0 and 1 run sequentially on the calling thread, a negative value uses one
  /// thread per hardware thread (see ResolveThreadCount).
  int nthreads{1};

  /// Numerical and geometric tolerance for knots, roots, and inversions.
  double tolerance{kDefaultTolerance};

  /// Optional workspace providing per-worker memory arenas. If null, kernels
  /// use a default workspace of the calling thread that persists across calls.
  /// A workspace must not be shared by kernel calls running concurrently.
  Workspace* workspace{nullptr};
};

} // namespace splinepy::core
