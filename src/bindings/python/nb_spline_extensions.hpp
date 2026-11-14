#pragma once

#include <nanobind/nanobind.h>

namespace splinepy::py {

namespace nb = nanobind;

void init_nb_spline_extensions(nb::module_& m);

} // namespace splinepy::py
