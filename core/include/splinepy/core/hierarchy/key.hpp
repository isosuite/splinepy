#pragma once

#include <cstdint>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"

namespace splinepy::core {

/// Storage format of hierarchy keys (cells and active functions).
/// Keys are stored as a flat sequence of uint64_t words; a format may use more
/// than one word per key. Only kKey64 exists today. Further formats (e.g. a
/// two-word Key128 or an int32 multi-index) only need a new KeyTraits
/// specialization and an entry in DispatchKeyFormat.
enum class KeyFormat : int32_t {
  kKey64 = 0,
};

/// Number of uint64_t words per key for a format.
constexpr int32_t KeyWords(KeyFormat format) noexcept {
  switch (format) {
    case KeyFormat::kKey64:
      return 1;
  }
  return 1;
}

/// Number of bits needed to represent v (0 for v == 0).
constexpr int32_t BitWidth(uint64_t v) noexcept {
  int32_t bits = 0;
  while (v != 0) {
    ++bits;
    v >>= 1;
  }
  return bits;
}

/// Bit layout of a 64-bit cell key.
///
///   key = (morton << level_bits) | level
///
/// morton interleaves, from the least significant bit up, the per-axis
/// "padded" index of the cell: the index of the first finest-level element the
/// cell contains. Axes with fewer bits drop out of the interleaving once their
/// bits are used up. For dyadic hierarchies this is the classic Z-order: a
/// cell's descendants form a contiguous key range.
struct CellKeyLayout64 {
  int32_t para_dim{0};
  int32_t level_bits{0};
  int32_t max_axis_bits{0};
  int32_t total_bits{0};
  /// Bits per axis, length para_dim. Points into caller-owned memory.
  Span<const int32_t> axis_bits{};
};

/// Bit layout of a 64-bit function key.
///
///   key = (level << index_bits) | linear_index
///
/// linear_index is the tensor index of the function within its level (axis 0
/// fastest, the same ordering as tensor control points). Sorting function keys
/// therefore orders active functions by level, then by tensor index.
struct FunctionKeyLayout64 {
  int32_t level_bits{0};
  int32_t index_bits{0};
  int32_t total_bits{0};
};

/// Builds a cell layout. axis_bits_out (length para_dim) receives the bits per
/// axis and is referenced by the returned layout.
/// Throws SplinepyException(kOutOfBounds) if the key would need more than 64
/// bits.
CellKeyLayout64 MakeCellKeyLayout64(Span<const int32_t> num_finest_elements,
                                    int32_t num_levels,
                                    Span<int32_t> axis_bits_out);

/// Builds a function layout from the finest-level number of tensor functions.
/// Throws SplinepyException(kOutOfBounds) if more than 64 bits are needed.
FunctionKeyLayout64 MakeFunctionKeyLayout64(int64_t num_finest_functions,
                                            int32_t num_levels);

template <KeyFormat F>
struct KeyTraits;

template <>
struct KeyTraits<KeyFormat::kKey64> {
  using Key = uint64_t;
  using CellLayout = CellKeyLayout64;
  using FunctionLayout = FunctionKeyLayout64;
  static constexpr KeyFormat kFormat = KeyFormat::kKey64;
  static constexpr int32_t kWords = 1;

  static Key EncodeCell(const CellLayout& layout,
                        int32_t level,
                        const int32_t* padded) noexcept {
    uint64_t morton = 0;
    int32_t pos = 0;
    for (int32_t bit = 0; bit < layout.max_axis_bits; ++bit) {
      for (int32_t a = 0; a < layout.para_dim; ++a) {
        if (bit < layout.axis_bits[a]) {
          morton |= ((static_cast<uint64_t>(padded[a]) >> bit) & 1u) << pos;
          ++pos;
        }
      }
    }
    return layout.level_bits == 0
               ? morton
               : (morton << layout.level_bits) | static_cast<uint64_t>(level);
  }

  /// Returns the level and writes the padded indices.
  static int32_t DecodeCell(const CellLayout& layout,
                            Key key,
                            int32_t* padded) noexcept {
    const uint64_t level_mask =
        layout.level_bits == 0 ? 0u : ((uint64_t{1} << layout.level_bits) - 1u);
    const int32_t level = static_cast<int32_t>(key & level_mask);
    const uint64_t morton = layout.level_bits == 0 ? key : key >> layout.level_bits;
    for (int32_t a = 0; a < layout.para_dim; ++a) {
      padded[a] = 0;
    }
    int32_t pos = 0;
    for (int32_t bit = 0; bit < layout.max_axis_bits; ++bit) {
      for (int32_t a = 0; a < layout.para_dim; ++a) {
        if (bit < layout.axis_bits[a]) {
          padded[a] |= static_cast<int32_t>(((morton >> pos) & 1u) << bit);
          ++pos;
        }
      }
    }
    return level;
  }

  static Key EncodeFunction(const FunctionLayout& layout,
                            int32_t level,
                            int64_t linear) noexcept {
    return (static_cast<uint64_t>(level) << layout.index_bits) |
           static_cast<uint64_t>(linear);
  }

  static int32_t DecodeFunction(const FunctionLayout& layout,
                                Key key,
                                int64_t* linear) noexcept {
    const uint64_t mask =
        layout.index_bits == 0 ? 0u : ((uint64_t{1} << layout.index_bits) - 1u);
    *linear = static_cast<int64_t>(key & mask);
    return static_cast<int32_t>(key >> layout.index_bits);
  }
};

/// Calls fn(KeyTraits<F>{}) for the runtime format. One branch per kernel call.
template <typename Fn>
decltype(auto) DispatchKeyFormat(KeyFormat format, Fn&& fn) {
  switch (format) {
    case KeyFormat::kKey64:
      return fn(KeyTraits<KeyFormat::kKey64>{});
  }
  throw SplinepyException("Unknown hierarchy key format",
                          Status::kInvalidArgument);
}

}  // namespace splinepy::core
