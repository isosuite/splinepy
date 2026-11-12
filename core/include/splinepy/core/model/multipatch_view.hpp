#pragma once

#include <vector>

#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core {

/// Non-owning view of a multipatch geometry (batch of splines + topology tables).
struct MultipatchView {
  /// Batch of patches (each patch is a SplineView).
  SplineBatchView patches{};

  /// Interfaces table: each entry connects (patch_a, face_a, patch_b, face_b).
  /// Stride is 4 integers per interface.
  Span<const int32_t> interfaces{};
  int64_t num_interfaces{0};

  /// Orientations table: orientation transformation data per interface.
  Span<const int32_t> interface_orientations{};

  /// Boundary specifications: patch index, boundary face index, boundary ID tag.
  Span<const int32_t> boundaries{};
  int64_t num_boundaries{0};

  /// Global control point offsets per patch for seamless DOF numbering.
  Span<const int64_t> control_point_offsets{};

  /// Number of patches in this multipatch.
  size_t size() const noexcept { return patches.size(); }
  bool empty() const noexcept { return patches.empty(); }
};

/// Owning container for multipatch geometry and topology.
struct MultipatchData {
  SplineBatchData patches;
  Buffer<int32_t> interfaces;
  int64_t num_interfaces{0};
  Buffer<int32_t> interface_orientations;
  Buffer<int32_t> boundaries;
  int64_t num_boundaries{0};
  Buffer<int64_t> control_point_offsets;

  // Cached views of patches for view() construction
  mutable std::vector<SplineView> patch_views_cache_;

  MultipatchView View() const {
    MultipatchView v;
    patch_views_cache_.resize(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) {
      patch_views_cache_[i] = patches[i].View();
    }
    v.patches = Span<const SplineView>(patch_views_cache_.data(),
                                       patch_views_cache_.size());
    v.interfaces = interfaces.cspan();
    v.num_interfaces = num_interfaces;
    v.interface_orientations = interface_orientations.cspan();
    v.boundaries = boundaries.cspan();
    v.num_boundaries = num_boundaries;
    v.control_point_offsets = control_point_offsets.cspan();
    return v;
  }
};

}  // namespace splinepy::core
