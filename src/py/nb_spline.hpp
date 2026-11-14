#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "nb_common.hpp"
#include "splinepy/core/options.hpp"
#include "splinepy/proximity/proximity.hpp"

namespace splinepy::py {

namespace nb = nanobind;

// Writes spline data to a python Spline's `_data`. The spline's arrays are
// replaced, other saved data is left untouched.
void SetSplineData(nb::handle spline, const splinepy::core::SplineData& data);

// Creates a python spline (derived type, matching the data) from spline data.
nb::object SplineFromData(const splinepy::core::SplineData& data);

// Keeps the kd-tree and the proximity search state of a spline.
// The python spline owns one and drops it, if its data changes.
class ProximityHandle {
public:
  std::unique_ptr<splinepy::proximity::Proximity> proximity_{nullptr};
  std::vector<int> last_proximity_resolutions_{};

  // solver order ("newton", "dogleg", "slsqp", "lm") and acceptance guard.
  // Mostly for experiments and benchmarks
  std::vector<std::string> chain_{"dogleg", "slsqp", "lm"};
  bool guard_{true};

  // initial guess strategy: "grid" (nearest sample) or "bvh" (element boxes,
  // several starts)
  std::string guess_{"bvh"};
  std::string planted_guess_{};
  int max_starts_{4};

  nb::tuple Query(nb::handle spline,
                  nb::object queries,
                  nb::object initial_guess_sample_resolutions,
                  double tolerance,
                  int max_iterations,
                  bool aggressive_search_bounds,
                  int nthreads);
};

void init_nb_spline(nb::module_& m);

} // namespace splinepy::py
