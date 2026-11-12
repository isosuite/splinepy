#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"

namespace splinepy::core {

/// High-performance linear bump allocator (memory arena).
/// Allocates 64-byte aligned contiguous blocks with O(1) allocation and O(1) scoped reset.
class MemoryArena {
 public:
  static constexpr std::size_t kDefaultBlockSize = 64 * 1024;  // 64 KB
  static constexpr std::size_t kDefaultAlignment = 64;         // 64-byte cache line / AVX-512

  struct Checkpoint {
    std::size_t block_index{0};
    std::size_t block_offset{0};
  };

  /// RAII scope for restoring arena allocation state upon exit.
  class Scope {
   public:
    explicit Scope(MemoryArena& arena) noexcept
        : arena_(&arena), checkpoint_(arena.GetCheckpoint()) {}

    ~Scope() noexcept {
      if (arena_) {
        arena_->Restore(checkpoint_);
      }
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    Scope(Scope&& other) noexcept
        : arena_(other.arena_), checkpoint_(other.checkpoint_) {
      other.arena_ = nullptr;
    }

    Scope& operator=(Scope&& other) noexcept {
      if (this != &other) {
        if (arena_) {
          arena_->Restore(checkpoint_);
        }
        arena_ = other.arena_;
        checkpoint_ = other.checkpoint_;
        other.arena_ = nullptr;
      }
      return *this;
    }

   private:
    MemoryArena* arena_{nullptr};
    Checkpoint checkpoint_{};
  };

  explicit MemoryArena(std::size_t block_size = kDefaultBlockSize,
                       std::size_t alignment = kDefaultAlignment)
      : block_size_(std::max(block_size, std::size_t{1024})),
        alignment_(alignment > 0 ? alignment : kDefaultAlignment) {
    blocks_.emplace_back(Block{Buffer<uint8_t>::Uninitialized(block_size_), 0});
  }

  ~MemoryArena() = default;

  MemoryArena(const MemoryArena&) = delete;
  MemoryArena& operator=(const MemoryArena&) = delete;

  MemoryArena(MemoryArena&&) noexcept = default;
  MemoryArena& operator=(MemoryArena&&) noexcept = default;

  /// Creates an RAII scope checkpointing the arena's current allocation position.
  [[nodiscard]] Scope CreateScope() noexcept {
    return Scope(*this);
  }

  /// Allocates raw aligned memory.
  void* AllocateRaw(std::size_t bytes, std::size_t align = kDefaultAlignment) {
    if (bytes == 0) {
      return nullptr;
    }

    const std::size_t eff_align = std::max(alignment_, align);
    std::size_t aligned_offset =
        (blocks_[current_block_index_].offset + eff_align - 1) & ~(eff_align - 1);

    if (aligned_offset + bytes > blocks_[current_block_index_].buffer.size()) {
      // Current block cannot fit request. Advance to next block or allocate new.
      const std::size_t new_block_size = std::max(block_size_, bytes + eff_align);

      if (current_block_index_ + 1 < blocks_.size()) {
        current_block_index_++;
        if (blocks_[current_block_index_].buffer.size() < bytes + eff_align) {
          blocks_[current_block_index_].buffer = Buffer<uint8_t>::Uninitialized(new_block_size);
        }
        blocks_[current_block_index_].offset = 0;
      } else {
        blocks_.emplace_back(Block{Buffer<uint8_t>::Uninitialized(new_block_size), 0});
        current_block_index_ = blocks_.size() - 1;
      }

      aligned_offset = 0;
    }

    uint8_t* ptr = blocks_[current_block_index_].buffer.data() + aligned_offset;
    blocks_[current_block_index_].offset = aligned_offset + bytes;
    return ptr;
  }

  /// Allocates typed array of elements.
  template <typename T>
  T* Allocate(std::size_t count = 1) {
    if (count == 0) {
      return nullptr;
    }
    return static_cast<T*>(AllocateRaw(sizeof(T) * count, alignof(T)));
  }

  /// Allocates typed Span of elements.
  template <typename T>
  Span<T> AllocateSpan(std::size_t count) {
    if (count == 0) {
      return Span<T>();
    }
    return Span<T>(Allocate<T>(count), count);
  }

  /// Allocates typed Span filled with a constant value.
  template <typename T>
  Span<T> AllocateSpan(std::size_t count, const T& fill_value) {
    if (count == 0) {
      return Span<T>();
    }
    T* ptr = Allocate<T>(count);
    std::fill_n(ptr, count, fill_value);
    return Span<T>(ptr, count);
  }

  /// Returns current allocation checkpoint.
  [[nodiscard]] Checkpoint GetCheckpoint() const noexcept {
    return Checkpoint{current_block_index_,
                      blocks_.empty() ? 0 : blocks_[current_block_index_].offset};
  }

  /// Restores arena allocation state to a previous checkpoint.
  void Restore(const Checkpoint& cp) noexcept {
    current_block_index_ = std::min(cp.block_index, blocks_.size() - 1);
    if (!blocks_.empty()) {
      blocks_[current_block_index_].offset = cp.block_offset;
      for (std::size_t i = current_block_index_ + 1; i < blocks_.size(); ++i) {
        blocks_[i].offset = 0;
      }
    }
  }

  /// Resets arena to beginning without freeing underlying block memory.
  void Reset() noexcept {
    current_block_index_ = 0;
    for (auto& b : blocks_) {
      b.offset = 0;
    }
  }

  /// Releases all blocks except the initial one.
  void Clear() {
    blocks_.resize(1);
    blocks_[0].offset = 0;
    current_block_index_ = 0;
  }

  /// Returns total capacity in bytes across all blocks.
  [[nodiscard]] std::size_t TotalCapacityBytes() const noexcept {
    std::size_t total = 0;
    for (const auto& b : blocks_) {
      total += b.buffer.size();
    }
    return total;
  }

 private:
  struct Block {
    Buffer<uint8_t> buffer;
    std::size_t offset{0};
  };

  std::size_t block_size_{kDefaultBlockSize};
  std::size_t alignment_{kDefaultAlignment};
  std::vector<Block> blocks_;
  std::size_t current_block_index_{0};
};

}  // namespace splinepy::core
