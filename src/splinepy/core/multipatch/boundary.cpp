#include "splinepy/core/multipatch/boundary.hpp"

#include "splinepy/core/memory/dynamic_buffer.hpp"
#include <stdexcept>

#include "splinepy/core/manipulation/extract.hpp"

namespace splinepy::core {

Buffer<int32_t> BoundaryPatchIds(Span<const int32_t> interfaces,
                                 int32_t boundary_id) {
  // faces on any boundary (boundary_id < 0) or on boundary -boundary_id
  DynamicBuffer<int32_t> pids(interfaces.size());
  for (std::size_t i = 0; i < interfaces.size(); ++i) {
    if (boundary_id < 0 ? interfaces[i] < 0 : interfaces[i] == -boundary_id) {
      pids.push_back(static_cast<int32_t>(i));
    }
  }
  return std::move(pids).Finish();
}

SplineBatchData BoundaryMultipatch(const SplineBatchView& patches,
                                   Span<const int32_t> interfaces,
                                   int32_t boundary_id,
                                   const Options& options) {
  if (patches.empty()) {
    return {};
  }

  const int32_t para_dim = patches[0].para_dim;
  const int32_t num_faces = 2 * para_dim;

  Buffer<int32_t> pids = BoundaryPatchIds(interfaces, boundary_id);
  SplineBatchData result(pids.size());
  for (std::size_t i = 0; i < pids.size(); ++i) {
    const int32_t pid = pids[i];
    const int32_t patch_id = pid / num_faces;
    const int32_t face_id = pid % num_faces;
    if (patch_id < 0 || patch_id >= static_cast<int32_t>(patches.size())) {
      throw SplinepyException("BoundaryMultipatch: patch_id out of range.",
                              Status::kOutOfBounds);
    }
    result[i] = ExtractBoundary(patches[patch_id], face_id, options);
  }

  return result;
}

SplineBatchData ExtractAllBoundarySplines(const SplineBatchView& patches,
                                          Span<const int32_t> interfaces,
                                          const Options& options) {
  return BoundaryMultipatch(patches, interfaces, -1, options);
}

} // namespace splinepy::core
