#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace splinepy::core {

namespace detail {

inline void* AlignedAlloc(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  // Ensure alignment is at least sizeof(void*) and a power of 2
  alignment = std::max(alignment, sizeof(void*));
#if defined(_MSC_VER)
  void* ptr = _aligned_malloc(bytes, alignment);
#else
  void* ptr = nullptr;
  int res = posix_memalign(&ptr, alignment, bytes);
  if (res != 0) {
    ptr = nullptr;
  }
#endif
  if (!ptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

inline void AlignedFree(void* ptr) noexcept {
  if (!ptr) {
    return;
  }
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

}  // namespace detail

/// Owning contiguous memory buffer with guaranteed alignment.
template <typename T>
class Buffer {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  Buffer() noexcept : data_(nullptr), size_(0), alignment_(kDefaultAlignment) {}

  explicit Buffer(size_type size, size_type alignment = kDefaultAlignment)
      : data_(nullptr), size_(size), alignment_(alignment) {
    if (size_ > 0) {
      data_ = static_cast<pointer>(
          detail::AlignedAlloc(size_ * sizeof(T), alignment_));
      for (size_type i = 0; i < size_; ++i) {
        new (data_ + i) T();
      }
    }
  }

  Buffer(size_type size, const T& val, size_type alignment = kDefaultAlignment)
      : data_(nullptr), size_(size), alignment_(alignment) {
    if (size_ > 0) {
      data_ = static_cast<pointer>(
          detail::AlignedAlloc(size_ * sizeof(T), alignment_));
      for (size_type i = 0; i < size_; ++i) {
        new (data_ + i) T(val);
      }
    }
  }

  Buffer(std::initializer_list<T> init,
         size_type alignment = kDefaultAlignment)
      : Buffer(init.size(), alignment) {
    std::copy(init.begin(), init.end(), data_);
  }

  ~Buffer() {
    clear();
  }

  // Move semantics (O(1) pointer swap)
  Buffer(Buffer&& other) noexcept
      : data_(other.data_),
        size_(other.size_),
        alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      clear();
      data_ = other.data_;
      size_ = other.size_;
      alignment_ = other.alignment_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // Copy semantics
  Buffer(const Buffer& other)
      : Buffer(other.size_, other.alignment_) {
    if (size_ > 0) {
      std::copy_n(other.data_, size_, data_);
    }
  }

  Buffer& operator=(const Buffer& other) {
    if (this != &other) {
      Buffer tmp(other);
      *this = std::move(tmp);
    }
    return *this;
  }

  void clear() noexcept {
    if (data_) {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        for (size_type i = 0; i < size_; ++i) {
          data_[i].~T();
        }
      }
      detail::AlignedFree(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  void resize(size_type new_size, const T& val = T()) {
    if (new_size == size_) {
      return;
    }
    if (new_size == 0) {
      clear();
      return;
    }
    pointer new_data = static_cast<pointer>(
        detail::AlignedAlloc(new_size * sizeof(T), alignment_));
    const size_type copy_count = std::min(size_, new_size);
    for (size_type i = 0; i < copy_count; ++i) {
      new (new_data + i) T(std::move(data_[i]));
    }
    for (size_type i = copy_count; i < new_size; ++i) {
      new (new_data + i) T(val);
    }
    clear();
    data_ = new_data;
    size_ = new_size;
  }

  template <typename InputIt>
  void assign(InputIt first, InputIt last) {
    const size_type n = std::distance(first, last);
    resize(n);
    std::copy(first, last, data_);
  }

  /// Relinquishes ownership of the raw data pointer.
  /// Used for zero-copy ownership transfer to host (e.g. pybind11 capsule).
  pointer release() noexcept {
    pointer ptr = data_;
    data_ = nullptr;
    size_ = 0;
    return ptr;
  }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }
  size_type size() const noexcept { return size_; }
  size_type alignment() const noexcept { return alignment_; }
  bool empty() const noexcept { return size_ == 0; }

  reference operator[](size_type idx) noexcept {
    assert(idx < size_ && "Buffer index out of range");
    return data_[idx];
  }

  const_reference operator[](size_type idx) const noexcept {
    assert(idx < size_ && "Buffer index out of range");
    return data_[idx];
  }

  reference front() noexcept {
    assert(size_ > 0 && "Buffer is empty");
    return data_[0];
  }

  const_reference front() const noexcept {
    assert(size_ > 0 && "Buffer is empty");
    return data_[0];
  }

  reference back() noexcept {
    assert(size_ > 0 && "Buffer is empty");
    return data_[size_ - 1];
  }

  const_reference back() const noexcept {
    assert(size_ > 0 && "Buffer is empty");
    return data_[size_ - 1];
  }

  Span<T> span() noexcept { return Span<T>(data_, size_); }
  Span<const T> span() const noexcept { return Span<const T>(data_, size_); }
  Span<const T> cspan() const noexcept { return Span<const T>(data_, size_); }

  pointer begin() noexcept { return data_; }
  pointer end() noexcept { return data_ + size_; }
  const_pointer begin() const noexcept { return data_; }
  const_pointer end() const noexcept { return data_ + size_; }
  const_pointer cbegin() const noexcept { return data_; }
  const_pointer cend() const noexcept { return data_ + size_; }

 private:
  pointer data_{nullptr};
  size_type size_{0};
  size_type alignment_{kDefaultAlignment};
};

}  // namespace splinepy::core
