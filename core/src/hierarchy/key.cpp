#include "splinepy/core/hierarchy/key.hpp"

namespace splinepy::core {

CellKeyLayout64 MakeCellKeyLayout64(Span<const int32_t> counts,
                                    int32_t num_levels,
                                    Span<int32_t> axis_bits) {
  if (counts.empty() || counts.size() > INT32_MAX ||
      counts.size() != axis_bits.size() || num_levels < 1) {
    throw SplinepyException("Invalid cell key dimensions");
  }
  const auto level_bits = BitWidth(num_levels - 1);
  CellKeyLayout64 layout{static_cast<int32_t>(counts.size()), level_bits,
                         0, level_bits, axis_bits};
  for (std::size_t axis = 0; axis < counts.size(); ++axis) {
    if (counts[axis] < 1) {
      throw SplinepyException("Cell key axes must have positive extent");
    }
    const auto bits = BitWidth(counts[axis] - 1);
    if (bits > 64 - layout.total_bits) {
      throw SplinepyException("Cell key exceeds 64 bits", Status::kOutOfBounds);
    }
    axis_bits[axis] = bits;
    if (bits > layout.max_axis_bits) {
      layout.max_axis_bits = bits;
    }
    layout.total_bits += bits;
  }
  return layout;
}

FunctionKeyLayout64 MakeFunctionKeyLayout64(int64_t count, int32_t num_levels) {
  if (count < 1 || num_levels < 1) {
    throw SplinepyException("Invalid function key dimensions");
  }
  const auto level_bits = BitWidth(num_levels - 1);
  const auto index_bits = BitWidth(count - 1);
  if (level_bits + index_bits > 64) {
    throw SplinepyException("Function key exceeds 64 bits", Status::kOutOfBounds);
  }
  return {level_bits, index_bits, level_bits + index_bits};
}

}  // namespace splinepy::core
