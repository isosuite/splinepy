#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace splinepy::core {

inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

/// Non-owning contiguous view over a sequence of elements.
template<typename T>
class Span {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = pointer;
  using const_iterator = const_pointer;

  constexpr Span() noexcept : data_(nullptr), size_(0) {}
  constexpr Span(pointer data, size_type size) noexcept
      : data_(data),
        size_(size) {}

  template<
      typename Container,
      typename = std::enable_if_t<
          !std::is_same_v<std::decay_t<Container>, Span>
          && std::is_convertible_v<decltype(std::declval<Container>().data()),
                                   pointer>>>
  constexpr Span(Container&& c) noexcept
      : data_(c.data()),
        size_(static_cast<size_type>(c.size())) {}

  template<typename U,
           typename = std::enable_if_t<std::is_convertible_v<U*, pointer>>>
  constexpr Span(const Span<U>& other) noexcept
      : data_(other.data()),
        size_(other.size()) {}

  constexpr pointer data() const noexcept { return data_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr reference operator[](size_type idx) const noexcept {
    assert(idx < size_ && "Span index out of range");
    return data_[idx];
  }

  constexpr reference front() const noexcept {
    assert(size_ > 0 && "Span::front() called on empty span");
    return data_[0];
  }

  constexpr reference back() const noexcept {
    assert(size_ > 0 && "Span::back() called on empty span");
    return data_[size_ - 1];
  }

  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + size_; }

  constexpr Span subspan(size_type offset,
                         size_type count = dynamic_extent) const noexcept {
    assert(offset <= size_ && "Span::subspan offset exceeds size");
    const size_type actual_count =
        (count == dynamic_extent) ? (size_ - offset) : count;
    assert(offset + actual_count <= size_
           && "Span::subspan range out of bounds");
    return Span(data_ + offset, actual_count);
  }

private:
  pointer data_{nullptr};
  size_type size_{0};
};

/// Deduction guides
template<typename T>
Span(T*, std::size_t) -> Span<T>;

template<typename Container>
Span(Container&) -> Span<typename Container::value_type>;

template<typename Container>
Span(const Container&) -> Span<const typename Container::value_type>;

/// Non-owning strided view over an array where consecutive elements
/// are separated by a constant stride (in number of elements T).
template<typename T>
class StridedSpan {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  constexpr StridedSpan() noexcept : data_(nullptr), size_(0), stride_(1) {}
  constexpr StridedSpan(pointer data,
                        size_type size,
                        difference_type stride = 1) noexcept
      : data_(data),
        size_(size),
        stride_(stride) {}

  constexpr pointer data() const noexcept { return data_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr difference_type stride() const noexcept { return stride_; }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr bool IsContiguous() const noexcept { return stride_ == 1; }

  constexpr reference operator[](size_type idx) const noexcept {
    assert(idx < size_ && "StridedSpan index out of range");
    return *(data_ + static_cast<difference_type>(idx) * stride_);
  }

  constexpr pointer ptr_at(size_type idx) const noexcept {
    assert(idx <= size_ && "StridedSpan ptr_at index out of range");
    return data_ + static_cast<difference_type>(idx) * stride_;
  }

private:
  pointer data_{nullptr};
  size_type size_{0};
  difference_type stride_{1};
};

} // namespace splinepy::core
