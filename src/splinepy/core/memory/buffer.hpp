#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
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

/// Allocates bytes: aligned to a cache line (kDefaultAlignment) from one
/// cache line on, so that arrays vectorize and never share a line with a
/// neighbour, and like malloc below that, where it would only waste memory.
/// Returns nullptr for zero bytes and throws std::bad_alloc on failure.
inline void* AlignedAlloc(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  const std::size_t alignment = bytes >= kDefaultAlignment
                                    ? kDefaultAlignment
                                    : alignof(std::max_align_t);
#if defined(_MSC_VER)
  void* ptr = _aligned_malloc(bytes, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, bytes) != 0) {
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

/// Bytes for count elements of T, throwing std::bad_alloc on overflow.
template<typename T>
std::size_t ByteCount(std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::bad_alloc();
  }
  return count * sizeof(T);
}

} // namespace detail

/// Owned contiguous memory of a fixed size. It never reallocates, so its
/// address is stable: spans, views and NumPy arrays of it stay valid until it
/// is destroyed or assigned. Everything that is stored or handed out is a
/// Buffer; what still grows is a DynamicBuffer until it is finished.
template<typename T>
class Buffer {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  Buffer() noexcept = default;

  explicit Buffer(size_type size) {
    Adopt(Build(size, [](pointer p) { new (p) T(); }), size);
  }

  /// Buffer of trivially constructible T whose elements are left uninitialized,
  /// for scratch memory that is written before it is read.
  static Buffer Uninitialized(size_type size) {
    static_assert(
        std::is_trivially_default_constructible_v<T>,
        "Uninitialized buffers need trivially constructible elements");
    Buffer buffer;
    buffer.Adopt(
        static_cast<pointer>(detail::AlignedAlloc(detail::ByteCount<T>(size))),
        size);
    return buffer;
  }

  Buffer(size_type size, const T& val) {
    Adopt(Build(size, [&](pointer p) { new (p) T(val); }), size);
  }

  Buffer(std::initializer_list<T> init) {
    auto it = init.begin();
    Adopt(Build(init.size(), [&](pointer p) { new (p) T(*it++); }),
          init.size());
  }

  ~Buffer() { clear(); }

  // Move semantics (O(1) pointer swap)
  Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      clear();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // Copy semantics: elements are copy constructed once.
  Buffer(const Buffer& other) {
    const_pointer source = other.data_;
    Adopt(Build(other.size_, [&](pointer p) { new (p) T(*source++); }),
          other.size_);
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
      Destroy(data_, size_);
      detail::AlignedFree(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  /// Resizes to new_size, keeping the first min(size(), new_size) elements and
  /// filling new ones with val. Leaves the buffer unchanged if an element
  /// constructor throws.
  void resize(size_type new_size, const T& val = T()) {
    if (new_size == size_) {
      return;
    }
    if (new_size == 0) {
      clear();
      return;
    }
    const size_type keep = std::min(size_, new_size);
    size_type i = 0;
    pointer new_data = Build(new_size, [&](pointer p) {
      if (i < keep) {
        new (p) T(std::move_if_noexcept(data_[i]));
      } else {
        new (p) T(val);
      }
      ++i;
    });
    clear();
    Adopt(new_data, new_size);
  }

  template<typename InputIt>
  void assign(InputIt first, InputIt last) {
    const auto n = static_cast<size_type>(std::distance(first, last));
    Buffer tmp;
    tmp.Adopt(Build(n, [&](pointer p) { new (p) T(*first++); }), n);
    *this = std::move(tmp);
  }

  /// Relinquishes ownership of the raw data pointer, for zero-copy ownership
  /// transfer to a host (e.g. a nanobind capsule). The caller destroys the
  /// elements if needed and frees the memory with detail::AlignedFree.
  pointer release() noexcept {
    pointer ptr = data_;
    data_ = nullptr;
    size_ = 0;
    return ptr;
  }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }
  size_type size() const noexcept { return size_; }
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
  template<typename>
  friend class DynamicBuffer;

  static void Destroy(pointer p, size_type n) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (size_type i = 0; i < n; ++i) {
        p[i].~T();
      }
    }
  }

  /// Allocates n elements and constructs each with construct(pointer). If a
  /// constructor throws, the elements built so far are destroyed and the
  /// memory is freed before the exception propagates.
  template<typename Construct>
  static pointer Build(size_type n, Construct&& construct) {
    auto* p =
        static_cast<pointer>(detail::AlignedAlloc(detail::ByteCount<T>(n)));
    size_type built = 0;
    try {
      for (; built < n; ++built) {
        construct(p + built);
      }
    } catch (...) {
      Destroy(p, built);
      detail::AlignedFree(p);
      throw;
    }
    return p;
  }

  void Adopt(pointer p, size_type n) noexcept {
    data_ = p;
    size_ = p ? n : 0;
  }

  pointer data_{nullptr};
  size_type size_{0};
};

} // namespace splinepy::core
