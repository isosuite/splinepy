#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"

namespace splinepy::core {

/// Growable aligned storage for building a result whose size is known only
/// after the work. Capacity beyond size() is never constructed, so growth
/// does not zero memory. Growth reallocates: never keep a pointer or span into
/// a DynamicBuffer while it may still grow. When done, Finish() hands the
/// elements over to a fixed Buffer, which is what gets stored or shared.
template<typename T>
class DynamicBuffer {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  DynamicBuffer() noexcept = default;

  explicit DynamicBuffer(size_type capacity) { reserve(capacity); }

  ~DynamicBuffer() { clear(); }

  DynamicBuffer(DynamicBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {}

  DynamicBuffer& operator=(DynamicBuffer&& other) noexcept {
    if (this != &other) {
      clear();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }

  // a builder is not shared; copy the finished Buffer instead
  DynamicBuffer(const DynamicBuffer&) = delete;
  DynamicBuffer& operator=(const DynamicBuffer&) = delete;

  /// Destroys the elements and frees the memory.
  void clear() noexcept {
    if (data_) {
      Buffer<T>::Destroy(data_, size_);
      detail::AlignedFree(data_);
    }
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  /// Makes room for at least capacity elements without constructing any.
  void reserve(size_type capacity) {
    if (capacity > capacity_) {
      Reallocate(capacity);
    }
  }

  void push_back(const T& value) { emplace_back(value); }
  void push_back(T&& value) { emplace_back(std::move(value)); }

  template<typename... Args>
  reference emplace_back(Args&&... args) {
    if (size_ == capacity_) {
      // a reference into this buffer stays valid while the element is built
      T value(std::forward<Args>(args)...);
      Reallocate(capacity_ == 0 ? kInitialCapacity : 2 * capacity_);
      new (data_ + size_) T(std::move(value));
    } else {
      new (data_ + size_) T(std::forward<Args>(args)...);
    }
    return data_[size_++];
  }

  /// Appends n uninitialized trivial elements and returns a pointer to the
  /// first, for bulk writes.
  pointer append_uninitialized(size_type n) {
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "Uninitialized elements need a trivial type");
    if (size_ + n > capacity_) {
      Reallocate(std::max(size_ + n, 2 * capacity_));
    }
    pointer first = data_ + size_;
    size_ += n;
    return first;
  }

  /// Destroys the last element.
  void pop_back() noexcept {
    assert(size_ > 0 && "DynamicBuffer is empty");
    --size_;
    if constexpr (!std::is_trivially_destructible_v<T>) {
      data_[size_].~T();
    }
  }

  /// Hands the elements over to a fixed Buffer and leaves this empty. Slack of
  /// more than a quarter of the capacity is given back with one reallocation.
  Buffer<T> Finish() && {
    if (size_ != 0 && capacity_ - size_ > capacity_ / 4) {
      Reallocate(size_);
    }
    Buffer<T> result;
    if (size_ == 0) {
      clear();
      return result;
    }
    result.Adopt(std::exchange(data_, nullptr), std::exchange(size_, 0));
    capacity_ = 0;
    return result;
  }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }
  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size_ == 0; }

  reference operator[](size_type idx) noexcept {
    assert(idx < size_ && "DynamicBuffer index out of range");
    return data_[idx];
  }

  const_reference operator[](size_type idx) const noexcept {
    assert(idx < size_ && "DynamicBuffer index out of range");
    return data_[idx];
  }

  reference back() noexcept {
    assert(size_ > 0 && "DynamicBuffer is empty");
    return data_[size_ - 1];
  }

  const_reference back() const noexcept {
    assert(size_ > 0 && "DynamicBuffer is empty");
    return data_[size_ - 1];
  }

  Span<T> span() noexcept { return Span<T>(data_, size_); }
  Span<const T> span() const noexcept { return Span<const T>(data_, size_); }
  Span<const T> cspan() const noexcept { return Span<const T>(data_, size_); }

  pointer begin() noexcept { return data_; }
  pointer end() noexcept { return data_ + size_; }
  const_pointer begin() const noexcept { return data_; }
  const_pointer end() const noexcept { return data_ + size_; }

private:
  static constexpr size_type kInitialCapacity = 8;

  /// Moves the elements into a new allocation of capacity elements.
  void Reallocate(size_type capacity) {
    auto* fresh = static_cast<pointer>(
        detail::AlignedAlloc(detail::ByteCount<T>(capacity)));
    if constexpr (std::is_trivially_copyable_v<T>) {
      if (size_ != 0) {
        std::memcpy(fresh, data_, size_ * sizeof(T));
      }
    } else {
      static_assert(std::is_nothrow_move_constructible_v<T>,
                    "DynamicBuffer elements need a noexcept move");
      for (size_type i = 0; i < size_; ++i) {
        new (fresh + i) T(std::move(data_[i]));
      }
      Buffer<T>::Destroy(data_, size_);
    }
    detail::AlignedFree(data_);
    data_ = fresh;
    capacity_ = capacity;
  }

  pointer data_{nullptr};
  size_type size_{0};
  size_type capacity_{0};
};

} // namespace splinepy::core
