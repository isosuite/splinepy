#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <string>
#include <vector>

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
void SetSplineData(nb::handle spline, splinepy::core::SplineData&& data);

// Creates a python spline (derived type, matching the data) from spline data.
nb::object SplineFromData(splinepy::core::SplineData&& data);

// Owns the geometry and the initial guess structure for all query batches.
class ProximityHandle {
public:
  ProximityHandle(nb::handle spline,
                  nb::object resolutions,
                  std::string guess,
                  int nthreads,
                  std::vector<std::string> chain,
                  bool guard,
                  int max_starts);

  int ParaDim() const { return proximity_->spline_.para_dim; }
  int Dim() const { return proximity_->spline_.dim; }

  void Query(nb::object queries,
             double tolerance,
             int max_iterations,
             bool aggressive_search_bounds,
             int nthreads,
             nb::tuple out) const;

private:
  splinepy::core::SplineData geometry_;
  std::unique_ptr<splinepy::proximity::Proximity> proximity_;
};

void init_nb_spline(nb::module_& m);

} // namespace splinepy::py
