#pragma once

#include <algorithm>
#include <new>

#include "splinepy/core/hierarchy/detail/knots.hpp"
#include "splinepy/core/memory/memory_arena.hpp"

namespace splinepy::core::detail {

struct Axis {
  Span<int32_t> boundaries; // Finest span ordinal of each local boundary.
  Span<int32_t> knot_cells; // Local boundary ordinal of each knot, clipped.
  Span<int32_t> first_knot; // First knot index mapped to each local boundary.
  int32_t functions;
};

template<class Traits>
struct Grid {
  Span<Axis> axes; // Level-major.
  typename Traits::CellLayout cells;
  typename Traits::FunctionLayout functions;
  int32_t dim;
  int32_t levels;

  Span<Axis> At(int32_t level) const {
    return axes.subspan(static_cast<std::size_t>(level) * dim, dim);
  }
};

template<class Traits>
Grid<Traits> BuildGrid(MemoryArena& arena,
                       Span<const int32_t> degrees,
                       Span<const LevelKnotsView> levels) {
  if (degrees.empty() || degrees.size() > INT32_MAX || levels.empty()
      || levels.size() > INT32_MAX
      || levels.size() > SIZE_MAX / sizeof(Axis) / degrees.size()) {
    throw SplinepyException("Invalid hierarchy dimensions");
  }
  const auto dim = static_cast<int32_t>(degrees.size());
  const auto count = static_cast<int32_t>(levels.size());
  auto axes = arena.AllocateSpan<Axis>(levels.size() * degrees.size());
  auto counts = arena.AllocateSpan<int32_t>(dim);
  int64_t finest_functions = 1;
  for (int32_t level = count - 1; level >= 0; --level) {
    if (levels[level].size() != degrees.size()) {
      throw SplinepyException(
          "Knot vector count differs from hierarchy dimension");
    }
    for (int32_t axis = 0; axis < dim; ++axis) {
      const auto knots = levels[level][axis];
      const auto functions =
          detail::ValidateHierarchyKnots(degrees[axis], knots);
      if (level + 1 < count) {
        detail::ValidateNested(degrees[axis], knots, levels[level + 1][axis]);
      }
      auto knot_cells = arena.AllocateSpan<int32_t>(knots.size());
      int32_t unique = 0;
      for (std::size_t i = 0; i < knots.size(); ++i) {
        if (i > static_cast<std::size_t>(degrees[axis])
            && i <= static_cast<std::size_t>(functions)
            && knots[i] != knots[i - 1]) {
          ++unique;
        }
        knot_cells[i] = unique;
      }
      auto boundaries = arena.AllocateSpan<int32_t>(unique + 1);
      auto first_knot = arena.AllocateSpan<int32_t>(unique + 1);
      for (std::size_t i = 0; i < knots.size(); ++i) {
        if (i == 0 || knot_cells[i] != knot_cells[i - 1]) {
          first_knot[knot_cells[i]] = static_cast<int32_t>(i);
        }
      }
      if (level == count - 1) {
        counts[axis] = unique;
        if (finest_functions > INT64_MAX / functions) {
          throw SplinepyException("Hierarchy function count exceeds int64",
                                  Status::kOutOfBounds);
        }
        finest_functions *= functions;
        for (int32_t i = 0; i <= unique; ++i) {
          boundaries[i] = i;
        }
      } else {
        const auto fine = levels.back()[axis];
        const auto& finest = axes[(levels.size() - 1) * dim + axis];
        auto match = fine.begin();
        for (int32_t i = 0; i <= unique; ++i) {
          const auto knot_index = i == 0 ? degrees[axis] : first_knot[i];
          match = std::lower_bound(match, fine.end(), knots[knot_index]);
          boundaries[i] = finest.knot_cells[match - fine.begin()];
        }
      }
      new (axes.data() + static_cast<std::size_t>(level) * dim + axis)
          Axis{boundaries, knot_cells, first_knot, functions};
    }
  }
  return {axes,
          MakeCellKeyLayout64(counts, count, arena.AllocateSpan<int32_t>(dim)),
          MakeFunctionKeyLayout64(finest_functions, count),
          dim,
          count};
}

inline bool AdvanceIndex(Span<int32_t> index,
                         Span<const int32_t> first,
                         Span<const int32_t> stop) {
  for (std::size_t axis = 0; axis < index.size(); ++axis) {
    if (++index[axis] < stop[axis]) {
      return true;
    }
    index[axis] = first[axis];
  }
  return false;
}

inline std::size_t CheckedProduct(std::size_t a, std::size_t b) {
  if (b && a > (SIZE_MAX / sizeof(double)) / b) {
    throw SplinepyException("Hierarchy extraction exceeds addressable size",
                            Status::kOutOfBounds);
  }
  return a * b;
}

} // namespace splinepy::core::detail
