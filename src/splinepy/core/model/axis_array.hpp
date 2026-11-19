#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"

/// Most parametric axes a spline can have. Per-axis data of views lives
/// inline, so views never allocate. Set it at build time, e.g.
/// -DSPLINEPY_MAX_PARA_DIM=6. A value <= 0 is reserved for a dynamic build
/// without a cap, see docs/design/spline_data.md.
#ifndef SPLINEPY_MAX_PARA_DIM
#define SPLINEPY_MAX_PARA_DIM 4
#endif

#if SPLINEPY_MAX_PARA_DIM <= 0
#error "SPLINEPY_MAX_PARA_DIM <= 0 (dynamic para_dim) is not implemented yet."
#endif

namespace splinepy::core {

inline constexpr int32_t kMaxParaDim = SPLINEPY_MAX_PARA_DIM;

/// Up to kMaxParaDim values, one per parametric axis, stored inline. All
/// per-axis data of a view goes through this type, so that a dynamic build
/// only has to change it. Reads like a Span and converts to one.
template<typename T>
class AxisArray {
  static_assert(std::is_trivially_copyable_v<T>,
                "AxisArray holds trivially copyable values");

public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  constexpr AxisArray() noexcept = default;

  /// Copies values, which must fit kMaxParaDim.
  constexpr AxisArray(Span<const T> values) { assign(values); }

  /// Copies n values from data, which must fit kMaxParaDim.
  constexpr AxisArray(const T* data, size_type n) {
    assign(Span<const T>(data, n));
  }

  /// Copies the values of a contiguous container, e.g. a Buffer.
  template<typename Container,
           typename = std::enable_if_t<
               !std::is_same_v<std::decay_t<Container>, AxisArray>
               && !std::is_same_v<std::decay_t<Container>, Span<const T>>
               && std::is_convertible_v<
                   decltype(std::declval<const Container&>().data()),
                   const T*>>>
  constexpr AxisArray(const Container& values) {
    assign(Span<const T>(values.data(), values.size()));
  }

  /// Copies values, which must fit kMaxParaDim.
  constexpr void assign(Span<const T> values) {
    resize(values.size());
    for (size_type i = 0; i < values.size(); ++i) {
      values_[i] = values[i];
    }
  }

  /// Sets the size. New values are left as they are; write them before
  /// reading.
  constexpr void resize(size_type n) {
    if (n > static_cast<size_type>(kMaxParaDim)) {
      ThrowTooManyAxes(n);
    }
    size_ = static_cast<int32_t>(n);
  }

  constexpr void clear() noexcept { size_ = 0; }

  constexpr const_pointer data() const noexcept { return values_; }
  constexpr pointer data() noexcept { return values_; }
  constexpr size_type size() const noexcept {
    return static_cast<size_type>(size_);
  }
  constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr const_reference operator[](size_type i) const noexcept {
    assert(i < size() && "AxisArray index out of range");
    return values_[i];
  }
  constexpr reference operator[](size_type i) noexcept {
    assert(i < size() && "AxisArray index out of range");
    return values_[i];
  }

  constexpr const_reference front() const noexcept { return (*this)[0]; }
  constexpr const_reference back() const noexcept {
    return (*this)[size() - 1];
  }

  constexpr const_pointer begin() const noexcept { return values_; }
  constexpr const_pointer end() const noexcept { return values_ + size_; }
  constexpr pointer begin() noexcept { return values_; }
  constexpr pointer end() noexcept { return values_ + size_; }

  constexpr Span<const T> cspan() const noexcept {
    return Span<const T>(values_, size());
  }
  constexpr operator Span<const T>() const noexcept { return cspan(); }

  constexpr Span<const T> subspan(size_type offset,
                                  size_type count = dynamic_extent) const {
    return cspan().subspan(offset, count);
  }

private:
  [[noreturn]] static void ThrowTooManyAxes(size_type n) {
    throw SplinepyException(
        "Spline has " + std::to_string(n)
            + " parametric axes, more than SPLINEPY_MAX_PARA_DIM ("
            + std::to_string(kMaxParaDim) + ").",
        Status::kInvalidArgument);
  }

  // tiny and fixed: value-initialized, so copies never read indeterminate
  // values
  T values_[SPLINEPY_MAX_PARA_DIM]{};
  int32_t size_{0};
};

} // namespace splinepy::core
