#include "splinepy/core/multipatch/boundary.hpp"

#include <stdexcept>
#include <vector>

#include "splinepy/core/manipulation/extract.hpp"

namespace splinepy::core {

Buffer<int32_t> BoundaryPatchIds(Span<const int32_t> interfaces,
                                 int32_t boundary_id) {
  std::vector<int32_t> pids;
  pids.reserve(interfaces.size());

  if (boundary_id < 0) {
    for (std::size_t i = 0; i < interfaces.size(); ++i) {
      if (interfaces[i] < 0) {
        pids.push_back(static_cast<int32_t>(i));
      }
    }
  } else {
    const int32_t target = -boundary_id;
    for (std::size_t i = 0; i < interfaces.size(); ++i) {
      if (interfaces[i] == target) {
        pids.push_back(static_cast<int32_t>(i));
      }
    }
  }

  Buffer<int32_t> result(pids.size());
  for (std::size_t i = 0; i < pids.size(); ++i) {
    result[i] = pids[i];
  }
  return result;
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
  SplineBatchData result;
  result.reserve(pids.size());

  for (int32_t pid : pids) {
    const int32_t patch_id = pid / num_faces;
    const int32_t face_id = pid % num_faces;
    if (patch_id < 0 || patch_id >= static_cast<int32_t>(patches.size())) {
      throw std::runtime_error("BoundaryMultipatch: patch_id out of range.");
    }
    result.push_back(ExtractBoundary(patches[patch_id], face_id, options));
  }

  return result;
}

SplineBatchData ExtractAllBoundarySplines(const SplineBatchView& patches,
                                         Span<const int32_t> interfaces,
                                         const Options& options) {
  return BoundaryMultipatch(patches, interfaces, -1, options);
}

}  // namespace splinepy::core
