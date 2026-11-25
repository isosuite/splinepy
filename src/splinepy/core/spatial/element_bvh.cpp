#include "splinepy/core/spatial/element_bvh.hpp"

#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace splinepy::core {

namespace {

constexpr int64_t kLeafSize = 2;
// same tolerance as the bezier extraction, to keep element orders identical
constexpr double kSpanTolerance = 1e-12;

} // namespace

ElementBvh::ElementBvh(const SplineView& spline)
    : para_dim_(spline.para_dim),
      dim_(spline.dim) {
  if (para_dim_ <= 0 || dim_ <= 0 || !spline.control_points_data) {
    return;
  }

  // finite control points and positive weights are required for convex hull
  const int64_t total_coords = spline.num_control_points * dim_;
  for (int64_t i{}; i < total_coords; ++i) {
    if (!std::isfinite(spline.control_points_data[i])) {
      return;
    }
  }
  if (spline.IsRational()) {
    for (int64_t i{}; i < spline.num_control_points; ++i) {
      if (!(spline.Weight(i) > 0.0) || !std::isfinite(spline.Weight(i))) {
        return;
      }
    }
  }

  // parametric bounds of the non-empty spans (elements) per axis
  Buffer<DynamicBuffer<double>> span_lo(para_dim_), span_hi(para_dim_);
  elements_per_axis_ = Buffer<int32_t>(para_dim_);
  num_elements_ = 1;
  for (int32_t d{}; d < para_dim_; ++d) {
    if (spline.HasKnotVectors()) {
      const auto kv = spline.knot_vectors[d];
      for (double knot : kv) {
        if (!std::isfinite(knot)) {
          return;
        }
      }
      for (int64_t k = spline.degrees[d]; k < spline.NumControlPointsPerAxis(d);
           ++k) {
        if (kv[k + 1] - kv[k] > kSpanTolerance) {
          span_lo[d].push_back(kv[k]);
          span_hi[d].push_back(kv[k + 1]);
        }
      }
    } else {
      span_lo[d].push_back(0.0);
      span_hi[d].push_back(1.0);
    }
    elements_per_axis_[d] = static_cast<int32_t>(span_lo[d].size());
    num_elements_ *= elements_per_axis_[d];
  }
  if (num_elements_ <= 0) {
    return;
  }

  param_lower_ = Buffer<double>(num_elements_ * para_dim_);
  param_upper_ = Buffer<double>(num_elements_ * para_dim_);
  for (int64_t e{}; e < num_elements_; ++e) {
    // element id -> span ids (first dim fastest)
    int64_t rest = e;
    for (int32_t d{}; d < para_dim_; ++d) {
      const int32_t span_id =
          static_cast<int32_t>(rest % elements_per_axis_[d]);
      rest /= elements_per_axis_[d];
      param_lower_[e * para_dim_ + d] = span_lo[d][span_id];
      param_upper_[e * para_dim_ + d] = span_hi[d][span_id];
    }
  }

  // The control points of the bezier patch of each element are the tightest
  // boxes we get from control points. Patches come in the same order
  // (first dim fastest).
  Options options;
  options.validate = false;
  SplineBatchData patches = ExtractBezierPatches(spline, options);
  if (static_cast<int64_t>(patches.size()) != num_elements_) {
    return;
  }

  box_lower_ =
      Buffer<double>(num_elements_ * dim_, std::numeric_limits<double>::max());
  box_upper_ = Buffer<double>(num_elements_ * dim_,
                              std::numeric_limits<double>::lowest());
  for (int64_t e{}; e < num_elements_; ++e) {
    const auto& cps = patches[e].control_points;
    const int64_t n = static_cast<int64_t>(cps.size()) / dim_;
    for (int64_t i{}; i < n; ++i) {
      for (int32_t k{}; k < dim_; ++k) {
        const double x = cps[i * dim_ + k];
        box_lower_[e * dim_ + k] = std::min(box_lower_[e * dim_ + k], x);
        box_upper_[e * dim_ + k] = std::max(box_upper_[e * dim_ + k], x);
      }
    }
  }

  BuildTree();
}

ElementBvh::ElementBvh(int32_t para_dim,
                       int32_t dim,
                       std::size_t controls_per_element,
                       Span<const double> bounds,
                       Span<const double> control_points)
    : para_dim_(para_dim),
      dim_(dim) {
  if (para_dim <= 0 || dim <= 0 || !controls_per_element || bounds.empty()) {
    return;
  }
  const auto pd = static_cast<std::size_t>(para_dim);
  const auto xd = static_cast<std::size_t>(dim);
  const auto elements = bounds.size() / 2 / pd;
  if (elements > INT64_MAX / 2 || bounds.size() % (2 * pd)
      || control_points.size() % xd
      || control_points.size() / xd % controls_per_element
      || control_points.size() / xd / controls_per_element != elements) {
    return;
  }

  num_elements_ = elements;
  param_lower_ = Buffer<double>(elements * pd);
  param_upper_ = Buffer<double>(elements * pd);
  box_lower_ =
      Buffer<double>(elements * xd, std::numeric_limits<double>::max());
  box_upper_ =
      Buffer<double>(elements * xd, std::numeric_limits<double>::lowest());
  for (std::size_t e = 0; e < elements; ++e) {
    for (std::size_t axis = 0; axis < pd; ++axis) {
      const double lo = bounds[2 * e * pd + axis];
      const double hi = bounds[2 * e * pd + pd + axis];
      if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) {
        return;
      }
      param_lower_[e * pd + axis] = lo;
      param_upper_[e * pd + axis] = hi;
    }
    for (std::size_t i = 0; i < controls_per_element; ++i) {
      for (std::size_t axis = 0; axis < xd; ++axis) {
        const auto value =
            control_points[(e * controls_per_element + i) * xd + axis];
        if (!std::isfinite(value)) {
          return;
        }
        auto& lo = box_lower_[e * xd + axis];
        auto& hi = box_upper_[e * xd + axis];
        lo = std::min(lo, value);
        hi = std::max(hi, value);
      }
    }
  }
  BuildTree();
}

void ElementBvh::BuildTree() {
  element_order_ = Buffer<int64_t>(num_elements_);
  std::iota(element_order_.begin(), element_order_.end(), int64_t{0});
  // a binary tree over n leaf ranges of at least one element has fewer
  // than 2 n nodes; trimmed after the build
  const int64_t max_nodes = 2 * num_elements_;
  nodes_ = Buffer<Node>(max_nodes);
  node_lower_ = Buffer<double>::Uninitialized(max_nodes * dim_);
  node_upper_ = Buffer<double>::Uninitialized(max_nodes * dim_);
  num_nodes_ = 0;
  BuildNode(0, num_elements_);
  nodes_.resize(num_nodes_);
  node_lower_.resize(num_nodes_ * dim_);
  node_upper_.resize(num_nodes_ * dim_);
  valid_ = true;
}

int64_t ElementBvh::BuildNode(const int64_t begin, const int64_t end) {
  const int64_t id = num_nodes_++;

  // node box = union of element boxes
  double* lower = &node_lower_[id * dim_];
  double* upper = &node_upper_[id * dim_];
  std::fill_n(lower, dim_, std::numeric_limits<double>::max());
  std::fill_n(upper, dim_, std::numeric_limits<double>::lowest());
  for (int64_t i{begin}; i < end; ++i) {
    const int64_t e = element_order_[i];
    for (int32_t k{}; k < dim_; ++k) {
      lower[k] = std::min(lower[k], box_lower_[e * dim_ + k]);
      upper[k] = std::max(upper[k], box_upper_[e * dim_ + k]);
    }
  }

  if (end - begin <= kLeafSize) {
    nodes_[id].first = begin;
    nodes_[id].count = end - begin;
    return id;
  }

  // split at the median of box centers along the widest axis of the centers
  int32_t axis{};
  double widest{-1.0};
  for (int32_t k{}; k < dim_; ++k) {
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    for (int64_t i{begin}; i < end; ++i) {
      const int64_t e = element_order_[i];
      const double c =
          .5 * box_lower_[e * dim_ + k] + .5 * box_upper_[e * dim_ + k];
      lo = std::min(lo, c);
      hi = std::max(hi, c);
    }
    if (hi - lo > widest) {
      widest = hi - lo;
      axis = k;
    }
  }

  const int64_t mid = begin + (end - begin) / 2;
  std::nth_element(element_order_.begin() + begin,
                   element_order_.begin() + mid,
                   element_order_.begin() + end,
                   [&](const int64_t a, const int64_t b) {
                     return .5 * box_lower_[a * dim_ + axis]
                                + .5 * box_upper_[a * dim_ + axis]
                            < .5 * box_lower_[b * dim_ + axis]
                                  + .5 * box_upper_[b * dim_ + axis];
                   });

  const int64_t left = BuildNode(begin, mid);
  const int64_t right = BuildNode(mid, end);
  nodes_[id].left = left;
  nodes_[id].right = right;
  return id;
}

} // namespace splinepy::core
