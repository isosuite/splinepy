#pragma once

#include <cstddef>
#include <type_traits>

#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"

namespace splinepy::core {

/// Uninitialized scratch of n trivially constructible elements: on the stack
/// up to N elements, otherwise an aligned heap Buffer. For small per-call
/// arrays whose size is bounded in practice (degrees, parametric dimension)
/// in code that has no arena at hand.
template<typename T, std::size_t N>
class ScratchArray {
  static_assert(std::is_trivially_default_constructible_v<T>
                    && std::is_trivially_destructible_v<T>,
                "ScratchArray holds trivial elements only");

public:
  explicit ScratchArray(std::size_t n) : size_(n) {
    if (n > N) {
      heap_ = Buffer<T>::Uninitialized(n);
    }
  }

  ScratchArray(const ScratchArray&) = delete;
  ScratchArray& operator=(const ScratchArray&) = delete;

  T* data() noexcept { return size_ > N ? heap_.data() : stack_; }
  const T* data() const noexcept { return size_ > N ? heap_.data() : stack_; }
  std::size_t size() const noexcept { return size_; }
  Span<T> span() noexcept { return Span<T>(data(), size_); }
  Span<const T> span() const noexcept { return Span<const T>(data(), size_); }
  T& operator[](std::size_t i) noexcept { return data()[i]; }
  const T& operator[](std::size_t i) const noexcept { return data()[i]; }

private:
  T stack_[N];
  Buffer<T> heap_;
  std::size_t size_;
};

} // namespace splinepy::core
