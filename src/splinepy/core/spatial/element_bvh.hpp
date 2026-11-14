#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core {

/// Bounding volume hierarchy over the elements (knot spans) of a spline.
///
/// Each element is bounded by the axis aligned box of the control points that
/// support it. By the convex hull property this box contains the element, as
/// long as the weights of rational splines are positive. Use IsValid() to
/// check this.
///
/// The hierarchy is only meant to prune: it answers "which elements can hold a
/// point closer than d to this query" without sampling the spline.
/// Dimensions are runtime values with no upper limit.
class ElementBvh {
public:
  ElementBvh() = default;

  /// Builds boxes for all non-empty knot spans. The spline is not referenced
  /// afterwards.
  explicit ElementBvh(const SplineView& spline);

  /// Builds from contiguous Bezier controls in element order. Parameter
  /// bounds have shape (elements, 2, para_dim). The caller guarantees
  /// positive weights for rational elements.
  ElementBvh(int32_t para_dim,
             int32_t dim,
             std::size_t controls_per_element,
             Span<const double> bounds,
             Span<const double> control_points);

  /// False if the box bound is not guaranteed (non-positive weights) or the
  /// spline has no elements.
  bool IsValid() const noexcept { return valid_; }

  int32_t ParaDim() const noexcept { return para_dim_; }
  int32_t Dim() const noexcept { return dim_; }
  int64_t NumElements() const noexcept { return num_elements_; }

  /// Number of elements along each parametric dimension.
  const std::vector<int32_t>& ElementsPerAxis() const noexcept {
    return elements_per_axis_;
  }

  /// Parametric bounds of an element. Both pointers need para_dim entries.
  void ElementBounds(const int64_t element,
                     double* lower,
                     double* upper) const noexcept {
    std::copy_n(&param_lower_[element * para_dim_], para_dim_, lower);
    std::copy_n(&param_upper_[element * para_dim_], para_dim_, upper);
  }

  /// Squared distance between a physical point and the box of an element.
  double ElementDistanceSquared(const int64_t element,
                                const double* point) const noexcept {
    return BoxDistanceSquared(&box_lower_[element * dim_],
                              &box_upper_[element * dim_],
                              point);
  }

  /// Visits elements by increasing lower distance bound until none can be
  /// closer than the bound the visitor reports.
  ///
  /// visitor(element, lower_bound_squared) is called for each element that may
  /// improve, and returns the smallest squared distance known so far (an upper
  /// bound for the answer). Traversal stops as soon as the next lower bound
  /// exceeds that, or after `max_visits` calls. Returns the number of calls.
  template<typename Visitor>
  int64_t Traverse(
      const double* point,
      Visitor&& visitor,
      const int64_t max_visits = std::numeric_limits<int64_t>::max()) const {
    if (!valid_ || max_visits <= 0) {
      return 0;
    }
    MemoryArena arena(2048);
    return Traverse(point, std::forward<Visitor>(visitor), arena, max_visits);
  }

  /// Allocates scratch in the caller’s arena. The caller controls its scope.
  template<typename Visitor>
  int64_t Traverse(
      const double* point,
      Visitor&& visitor,
      MemoryArena& arena,
      const int64_t max_visits = std::numeric_limits<int64_t>::max()) const {
    if (!valid_ || max_visits <= 0) {
      return 0;
    }

    // A negative ID encodes an element; nonnegative IDs encode tree nodes.
    // Comparing the pair preserves the distance and ID tie-breaking order.
    using Entry = std::pair<double, int64_t>;
    auto heap = arena.AllocateSpan<Entry>(64);
    std::size_t pending = 0;
    const std::greater<Entry> farther;
    const auto push = [&](double bound, int64_t id) {
      if (pending == heap.size()) {
        auto grown = arena.AllocateSpan<Entry>(2 * heap.size());
        std::uninitialized_copy_n(heap.data(), pending, grown.data());
        heap = grown;
      }
      // Entries popped from the heap are trivially destructible.
      ::new (static_cast<void*>(heap.data() + pending)) Entry(bound, id);
      ++pending;
      std::push_heap(heap.data(), heap.data() + pending, farther);
    };
    push(NodeDistanceSquared(0, point), 0);

    double upper_bound = std::numeric_limits<double>::infinity();
    int64_t visits{};
    while (pending != 0 && visits < max_visits) {
      std::pop_heap(heap.data(), heap.data() + pending, farther);
      const auto [lower_bound, entry] = heap[--pending];
      if (lower_bound > upper_bound) {
        break;
      }

      if (entry < 0) {
        upper_bound = std::min(upper_bound, visitor(-entry - 1, lower_bound));
        ++visits;
        continue;
      }

      const Node& node = nodes_[entry];
      if (node.count > 0) {
        for (int64_t i{}; i < node.count; ++i) {
          const int64_t element = element_order_[node.first + i];
          push(ElementDistanceSquared(element, point), -element - 1);
        }
      } else {
        push(NodeDistanceSquared(node.left, point), node.left);
        push(NodeDistanceSquared(node.right, point), node.right);
      }
    }
    return visits;
  }

private:
  struct Node {
    int64_t left{-1};
    int64_t right{-1};
    int64_t first{0};
    int64_t count{0}; // > 0 for leaves
  };

  static double BoxDistanceSquared(const double* lower,
                                   const double* upper,
                                   const double* point,
                                   const int32_t dim) noexcept {
    double d2{};
    for (int32_t i{}; i < dim; ++i) {
      const double excess =
          std::max({lower[i] - point[i], point[i] - upper[i], 0.0});
      d2 += excess * excess;
    }
    return d2;
  }

  double BoxDistanceSquared(const double* lower,
                            const double* upper,
                            const double* point) const noexcept {
    return BoxDistanceSquared(lower, upper, point, dim_);
  }

  double NodeDistanceSquared(const int64_t node,
                             const double* point) const noexcept {
    return BoxDistanceSquared(&node_lower_[node * dim_],
                              &node_upper_[node * dim_],
                              point);
  }

  void BuildTree();
  int64_t BuildNode(const int64_t begin, const int64_t end);

  int32_t para_dim_{0};
  int32_t dim_{0};
  int64_t num_elements_{0};
  bool valid_{false};

  std::vector<int32_t> elements_per_axis_;
  std::vector<double> param_lower_, param_upper_; // elements x para_dim
  std::vector<double> box_lower_, box_upper_;     // elements x dim

  std::vector<Node> nodes_;
  std::vector<double> node_lower_, node_upper_; // nodes x dim
  std::vector<int64_t> element_order_;
};

} // namespace splinepy::core
